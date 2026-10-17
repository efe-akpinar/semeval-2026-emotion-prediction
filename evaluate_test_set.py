"""
Evaluate the trained models on the official SemEval-2026 Task 2 (Subtask 1) test set.

The test set contains 91 users: 46 who also appear in the official training data
("seen") and 45 who do not ("unseen"). Gold labels were released separately.

Models (weights produced by the notebook):
  - Baseline: fine-tuned DistilBERT regressor        -> best_baseline_model.bin
  - Final:    LSTM over [CLS] embeddings + history    -> best_sequence_model.bin

The sequence model is evaluated in two modes:
  - autoregressive:  the model's own previous prediction is fed back as history
                     (the realistic test-time setting, as in the notebook's inference pipeline)
  - teacher_forcing: the gold previous labels are fed as history (the setting used for
                     validation in the notebook; reported only for comparison)

Metrics:
  - Official Subtask 1 metric (same computation as the task's eval.py):
    r_within (mean per-user Pearson r), r_between (Pearson r of per-user means) and
    r_composite = tanh((atanh(r_within) + atanh(r_between)) / 2)
  - Flat Pearson r over all texts (the metric used in the notebook)

Usage:
  python evaluate_test_set.py \
      --test-csv test_subtask1.csv --labels-csv test_labels_subtask1.csv \
      --baseline-weights best_baseline_model.bin --sequence-weights best_sequence_model.bin
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
HIDDEN_DIM = 64


# =============================================================================
# MODEL DEFINITIONS (identical to the notebook)
# =============================================================================
class SentimentRegressor(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(SentimentRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)


class AdvancedSequenceModel(nn.Module):
    """Sequence model WITHOUT user embeddings - matches the trained sequence model"""
    def __init__(self, input_dim=768, hidden_dim=64):
        super(AdvancedSequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim + 2,
                            hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(hidden_dim, 2)
        )

    def forward_step(self, text_embed, prev_value, hidden_state=None):
        t_embed = text_embed.unsqueeze(0).unsqueeze(0)   # (1, 1, 768)
        p_val = prev_value.unsqueeze(0).unsqueeze(0)     # (1, 1, 2)
        combined = torch.cat((t_embed, p_val), dim=2)
        out, new_hidden = self.lstm(combined, hidden_state)
        pred = self.regressor(out)
        return pred.squeeze(0).squeeze(0), new_hidden


# =============================================================================
# METRICS
# =============================================================================
def official_subtask1_metric(user_ids, predictions, labels):
    """Within-, between- and composite Pearson r, following the task's eval.py."""
    user_arr = np.asarray(user_ids)
    preds = np.asarray(predictions, dtype=float)
    labs = np.asarray(labels, dtype=float)
    unique_users = np.unique(user_arr)

    # Within-person: per-user Pearson r, averaged (users with <2 texts or constant labels skipped)
    r_vals = []
    for u in unique_users:
        mask = user_arr == u
        if mask.sum() < 2 or np.var(labs[mask]) == 0.0:
            continue
        if np.var(preds[mask]) == 0.0:
            r_vals.append(0.0)
            continue
        r_vals.append(float(pearsonr(preds[mask], labs[mask])[0]))
    r_within = float(np.mean(r_vals))

    # Between-person: Pearson r of per-user means
    pred_means = [preds[user_arr == u].mean() for u in unique_users]
    lab_means = [labs[user_arr == u].mean() for u in unique_users]
    r_between = float(pearsonr(pred_means, lab_means)[0])

    r_composite = float(np.tanh(0.5 * (np.arctanh(r_within) + np.arctanh(r_between))))
    return {'r_within': r_within, 'r_between': r_between, 'r_composite': r_composite,
            'n_users_within': len(r_vals)}


def evaluate(df, pred_cols):
    """All metrics for valence and arousal on the rows of df."""
    results = {'n_texts': int(len(df)), 'n_users': int(df['user_id'].nunique())}
    for dim, col in zip(['valence', 'arousal'], pred_cols):
        metrics = official_subtask1_metric(df['user_id'], df[col], df[dim])
        metrics['r_flat'] = float(pearsonr(df[col], df[dim])[0])
        metrics['mae'] = float(np.mean(np.abs(df[col] - df[dim])))
        results[dim] = metrics
    return results


# =============================================================================
# INFERENCE
# =============================================================================
def extract_features(model, tokenizer, texts, device, batch_size=32):
    """[CLS] embeddings and baseline predictions from the fine-tuned DistilBERT."""
    embeddings, baseline_preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True,
                               truncation=True, max_length=MAX_LEN).to(device)
            cls = model.bert(**inputs).last_hidden_state[:, 0, :]
            embeddings.append(cls.cpu().numpy())
            baseline_preds.append(model.out(model.drop(cls)).cpu().numpy())
    return np.vstack(embeddings), np.vstack(baseline_preds)


def predict_sequences(seq_model, df, embeddings, device, teacher_forcing=False):
    """Step through each user's texts in time order; history = own or gold previous values."""
    preds = np.zeros((len(df), 2), dtype=np.float32)
    with torch.no_grad():
        for _, group in df.groupby('user_id'):
            group = group.sort_values('timestamp')
            prev = torch.zeros(2, device=device)
            hidden = None
            for idx in group.index:
                embed = torch.tensor(embeddings[idx], device=device)
                pred, hidden = seq_model.forward_step(embed, prev, hidden)
                preds[idx] = pred.cpu().numpy()
                if teacher_forcing:
                    prev = torch.tensor(df.loc[idx, ['valence', 'arousal']].values.astype(np.float32),
                                        device=device)
                else:
                    prev = pred
    return preds


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--test-csv', default='test_subtask1.csv')
    parser.add_argument('--labels-csv', default='test_labels_subtask1.csv')
    parser.add_argument('--baseline-weights', default='best_baseline_model.bin')
    parser.add_argument('--sequence-weights', default='best_sequence_model.bin')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    device = torch.device('cpu')
    torch.manual_seed(42)

    # --- Data: test texts joined with the released gold labels ---
    test_df = pd.read_csv(args.test_csv)
    labels_df = pd.read_csv(args.labels_csv)[['user_id', 'text_id', 'valence', 'arousal']]
    df = test_df.merge(labels_df, on=['user_id', 'text_id'], how='inner', validate='one_to_one')
    assert len(df) == len(test_df), 'every test text must have a gold label'
    df = df.reset_index(drop=True)
    print(f"Test set: {len(df)} texts, {df['user_id'].nunique()} users "
          f"({df.loc[df['is_seen_user'], 'user_id'].nunique()} seen, "
          f"{df.loc[~df['is_seen_user'], 'user_id'].nunique()} unseen)")

    # --- Models ---
    base_model = SentimentRegressor(MODEL_NAME)
    base_model.load_state_dict(torch.load(args.baseline_weights, map_location=device))
    base_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seq_model = AdvancedSequenceModel(hidden_dim=HIDDEN_DIM)
    seq_model.load_state_dict(torch.load(args.sequence_weights, map_location=device))
    seq_model.to(device).eval()

    # --- Predictions ---
    print('Extracting DistilBERT features...')
    embeddings, baseline_preds = extract_features(base_model, tokenizer,
                                                  df['text'].astype(str).tolist(), device)
    df[['base_valence', 'base_arousal']] = baseline_preds
    print('Running sequence model (autoregressive and teacher-forced)...')
    df[['ar_valence', 'ar_arousal']] = predict_sequences(seq_model, df, embeddings, device)
    df[['tf_valence', 'tf_arousal']] = predict_sequences(seq_model, df, embeddings, device,
                                                         teacher_forcing=True)

    # --- Evaluation ---
    models = {
        'baseline_distilbert': ['base_valence', 'base_arousal'],
        'lstm_history_autoregressive': ['ar_valence', 'ar_arousal'],
        'lstm_history_teacher_forcing': ['tf_valence', 'tf_arousal'],
    }
    subsets = {'all': df, 'seen_users': df[df['is_seen_user']], 'unseen_users': df[~df['is_seen_user']]}
    results = {model: {name: evaluate(sub, cols) for name, sub in subsets.items()}
               for model, cols in models.items()}

    # --- Report ---
    print(f"\n{'Model':<30} {'Subset':<13} {'Val comp':>8} {'Aro comp':>8} {'Val flat':>8} {'Aro flat':>8}")
    for model, by_subset in results.items():
        for name, r in by_subset.items():
            print(f"{model:<30} {name:<13} {r['valence']['r_composite']:>8.4f} {r['arousal']['r_composite']:>8.4f} "
                  f"{r['valence']['r_flat']:>8.4f} {r['arousal']['r_flat']:>8.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'test_set_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    df[['user_id', 'text_id', 'is_seen_user'] + [c for cols in models.values() for c in cols]] \
        .to_csv(os.path.join(args.out_dir, 'test_predictions.csv'), index=False)
    print(f"\nSaved {args.out_dir}/test_set_results.json and {args.out_dir}/test_predictions.csv")


if __name__ == '__main__':
    main()
