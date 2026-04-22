"""
Feature Discriminability Evaluation
-------------------------------------
Evaluates which of the 5 lip features are most discriminative using
methods that do NOT require retraining:

  1. Fisher Discriminant Ratio (FDR) — class-separability on raw features
  2. Mutual Information           — non-linear dependence on digit identity
  3. Permutation Importance       — AUC drop when a feature is shuffled (uses
                                     trained DigitVerifier, no retraining)

Run:
    python feature_discriminability.py [--mode digit|sequence]
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import os

from model import DigitVerifier, SequenceVerifier, CHAR_TO_IDX, N_CLASSES
from train import (LipVerificationDataset, SequenceVerificationDataset,
                   sequence_collate_fn, MAX_SEQ_LEN, N_FEATURES,
                   EMBED_DIM, HIDDEN_DIM, PROCESSED_DIR, MODEL_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

FEATURE_NAMES = ['vert_aperture', 'horiz_spread', 'inner_area', 'compactness', 'lip_speed']

# ---------------------------------------------------------------------------
# 1. Fisher Discriminant Ratio
# ---------------------------------------------------------------------------

def fisher_discriminant_ratio(segments_by_digit):
    """
    For each feature, compute FDR = between-class variance / within-class variance.
    segments_by_digit: dict {digit_idx: list of (T, 5) arrays}
    We summarise each segment by its mean over time → shape (N, 5) per digit.
    """
    all_classes = [d for d in sorted(segments_by_digit.keys()) if len(segments_by_digit[d]) > 0]

    # Per-class mean vectors: (n_classes, 5)
    class_means = {}
    class_vars  = {}
    class_counts = {}
    for d in all_classes:
        segs = segments_by_digit[d]
        # Summarise each segment: concatenate mean and std → (N, 10) → use mean only here
        X = np.stack([s.mean(axis=0) for s in segs])   # (N, 5)
        class_means[d]  = X.mean(axis=0)
        class_vars[d]   = X.var(axis=0)
        class_counts[d] = len(X)

    n_total = sum(class_counts.values())
    global_mean = sum(class_counts[d] * class_means[d] for d in all_classes) / n_total

    between = sum(class_counts[d] * (class_means[d] - global_mean) ** 2
                  for d in all_classes) / len(all_classes)
    within  = sum(class_counts[d] * class_vars[d]
                  for d in all_classes) / n_total

    fdr = between / (within + 1e-12)
    return fdr   # (5,)


# ---------------------------------------------------------------------------
# 2. Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(segments_by_digit):
    """
    Build a feature matrix X of shape (N_segments, 10) — per-segment mean + std
    of each feature — and a label vector y.
    Returns MI scores of shape (5,) averaged over mean and std representations.
    """
    X_rows, y_rows = [], []
    for digit, segs in sorted(segments_by_digit.items()):
        for s in segs:
            row = np.concatenate([s.mean(axis=0), s.std(axis=0)])  # (10,)
            X_rows.append(row)
            y_rows.append(digit)

    X = np.array(X_rows)   # (N, 10)
    y = np.array(y_rows)

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    # Average MI over the mean and std halves
    mi_per_feature = (mi[:5] + mi[5:]) / 2
    return mi_per_feature


# ---------------------------------------------------------------------------
# 3. Permutation Importance
# ---------------------------------------------------------------------------

def get_baseline_auc(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                feats, mask, digit, label = [b.to(DEVICE) for b in batch]
                logits = model(feats, mask, digit)
            else:
                segs, masks, digits, label, seq_mask = [b.to(DEVICE) for b in batch]
                logits = model(segs, masks, digits, seq_mask)
            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    return roc_auc_score(np.array(all_labels), np.array(all_probs))


def permuted_auc(model, loader, feat_idx, mode, rng, n_repeats=5):
    """
    Shuffle feature feat_idx in-batch and measure average AUC drop.
    """
    model.eval()
    aucs = []
    for _ in range(n_repeats):
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                if mode == 'digit':
                    feats, mask, digit, label = [b.to(DEVICE) for b in batch]
                    # Permute feature feat_idx across the batch
                    B, T, F = feats.shape
                    perm = torch.randperm(B * T, generator=None)
                    col = feats[:, :, feat_idx].reshape(-1)
                    feats = feats.clone()
                    feats[:, :, feat_idx] = col[perm].reshape(B, T)
                    logits = model(feats, mask, digit)
                else:
                    segs, masks, digits, label, seq_mask = [b.to(DEVICE) for b in batch]
                    B, S, T, F = segs.shape
                    perm = torch.randperm(B * S * T)
                    col = segs[:, :, :, feat_idx].reshape(-1)
                    segs = segs.clone()
                    segs[:, :, :, feat_idx] = col[perm].reshape(B, S, T)
                    logits = model(segs, masks, digits, seq_mask)
                probs = torch.sigmoid(logits)
                all_labels.extend(label.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
        try:
            aucs.append(roc_auc_score(np.array(all_labels), np.array(all_probs)))
        except ValueError:
            aucs.append(0.5)
    return np.mean(aucs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_segments_by_digit(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    digit_segments = data['digit_segments']
    digit_sequences = data['digit_sequences']

    by_digit = {d: [] for d in range(N_CLASSES)}
    for vid_segs, vid_digits in zip(digit_segments, digit_sequences):
        for seg, digit in zip(vid_segs, vid_digits):
            idx = CHAR_TO_IDX[str(digit)]
            by_digit[idx].append(seg)
    return by_digit


def print_ranking(scores, label):
    order = np.argsort(scores)[::-1]
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  {'Feature':<18}  {'Score':>10}  {'Rank':>4}")
    print(f"  {'-'*36}")
    for rank, i in enumerate(order, 1):
        print(f"  {FEATURE_NAMES[i]:<18}  {scores[i]:>10.4f}  {rank:>4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='digit')
    parser.add_argument('--perm_repeats', type=int, default=5,
                        help='Number of permutation repeats per feature (more = stabler estimate)')
    args = parser.parse_args()

    test_path = os.path.join(PROCESSED_DIR, 'test.npz')
    model_path = os.path.join(MODEL_DIR, f'best_{args.mode}_verifier.pt')

    # ---- 1 & 2: Model-free methods ----------------------------------------
    print("Loading segments...")
    by_digit = load_segments_by_digit(test_path)
    n_segs = sum(len(v) for v in by_digit.values())
    print(f"  {n_segs} digit segments across {N_CLASSES} classes")

    fdr = fisher_discriminant_ratio(by_digit)
    print_ranking(fdr, "Fisher Discriminant Ratio (higher = more separable)")

    mi = mutual_information(by_digit)
    print_ranking(mi, "Mutual Information (higher = more informative)")

    # ---- 3: Permutation Importance ----------------------------------------
    if not os.path.exists(model_path):
        print(f"\nSkipping permutation importance: model not found at {model_path}")
    else:
        print(f"\nLoading model from {model_path}...")
        if args.mode == 'digit':
            model = DigitVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                  n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)
            dataset = LipVerificationDataset(test_path, seed=99)
            loader = DataLoader(dataset, batch_size=256, shuffle=False,
                                num_workers=0, collate_fn=None)
        else:
            model = SequenceVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                     n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)
            dataset = SequenceVerificationDataset(test_path, seed=99)
            loader = DataLoader(dataset, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=sequence_collate_fn)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        print(f"Computing baseline AUC...")
        baseline_auc = get_baseline_auc(model, loader)
        print(f"  Baseline AUC: {baseline_auc:.4f}")

        rng = np.random.RandomState(0)
        perm_drops = np.zeros(N_FEATURES)
        for i, name in enumerate(FEATURE_NAMES):
            auc_i = permuted_auc(model, loader, i, args.mode, rng, args.perm_repeats)
            perm_drops[i] = baseline_auc - auc_i
            print(f"  {name:<18} permuted AUC={auc_i:.4f}  drop={perm_drops[i]:+.4f}")

        print_ranking(perm_drops, f"Permutation Importance — AUC drop (baseline={baseline_auc:.4f})")

    # ---- Summary table ----------------------------------------------------
    print(f"\n{'='*55}")
    print("  Summary: feature ranks across methods")
    print(f"{'='*55}")
    fdr_rank = np.argsort(np.argsort(fdr)[::-1]) + 1
    mi_rank  = np.argsort(np.argsort(mi)[::-1]) + 1
    print(f"  {'Feature':<18}  {'FDR rank':>8}  {'MI rank':>7}")
    print(f"  {'-'*37}")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<18}  {fdr_rank[i]:>8}  {mi_rank[i]:>7}")
