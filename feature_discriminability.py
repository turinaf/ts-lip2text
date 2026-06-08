"""
Feature Discriminability Evaluation
-------------------------------------
Evaluates which of the lip features are most discriminative using
methods that do NOT require retraining:

  1. Fisher Discriminant Ratio (FDR) — class-separability on raw features
  2. Mutual Information           — non-linear dependence on digit identity
  3. Permutation Importance       — AUC drop when a feature is shuffled (uses
                                     trained DigitVerifier, no retraining)

Run:
    python feature_discriminability.py [--mode digit|sequence]
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import os
import matplotlib.pyplot as plt

from model import DigitVerifier, SequenceVerifier
from train import (LipVerificationDataset, SequenceVerificationDataset,
                   sequence_collate_fn,
                   EMBED_DIM, HIDDEN_DIM, PROCESSED_ROOT, MODEL_ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# 1. Fisher Discriminant Ratio
# ---------------------------------------------------------------------------

def fisher_discriminant_ratio(segments_by_digit):
    """
    For each feature, compute FDR = between-class variance / within-class variance.
    segments_by_digit: dict {digit_idx: list of (T, N_FEATURES) arrays}
    We summarise each segment by its mean over time → shape (N, N_FEATURES) per digit.
    """
    all_classes = [d for d in sorted(segments_by_digit.keys()) if len(segments_by_digit[d]) > 0]

    # Per-class mean vectors: (n_classes, N_FEATURES)
    class_means = {}
    class_vars  = {}
    class_counts = {}
    for d in all_classes:
        segs = segments_by_digit[d]
        # Summarise each segment by temporal mean only: (N, N_FEATURES)
        X = np.stack([s.mean(axis=0) for s in segs])   # (N, N_FEATURES)
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
    return fdr   # (N_FEATURES,)


# ---------------------------------------------------------------------------
# 2. Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(segments_by_digit, n_features):
    """
    Build a feature matrix X of shape (N_segments, 2*N_FEATURES) — per-segment mean + std
    of each feature — and a label vector y.
    Returns MI scores of shape (N_FEATURES,) averaged over mean and std representations.
    """
    X_rows, y_rows = [], []
    for digit, segs in sorted(segments_by_digit.items()):
        for s in segs:
            row = np.concatenate([s.mean(axis=0), s.std(axis=0)])  # (2*N_FEATURES,)
            X_rows.append(row)
            y_rows.append(digit)

    X = np.array(X_rows)   # (N, 2*N_FEATURES)
    y = np.array(y_rows)

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    # Average MI over the mean and std halves
    mi_per_feature = (mi[:n_features] + mi[n_features:]) / 2
    return mi_per_feature


# ---------------------------------------------------------------------------
# 3. Permutation Importance
# ---------------------------------------------------------------------------

def _adapt_tensor_feature_dim(x, target_dim):
    """Slice or zero-pad feature dimension on tensors with features in the last axis."""
    current_dim = x.shape[-1]
    if current_dim == target_dim:
        return x
    if current_dim > target_dim:
        return x[..., :target_dim]
    pad_shape = list(x.shape[:-1]) + [target_dim - current_dim]
    pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def get_baseline_auc(model, loader, expected_n_features=None):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                feats, mask, digit, label = [b.to(DEVICE) for b in batch]
                if expected_n_features is not None:
                    feats = _adapt_tensor_feature_dim(feats, expected_n_features)
                logits = model(feats, mask, digit)
            else:
                segs, masks, digits, label, seq_mask = [b.to(DEVICE) for b in batch]
                if expected_n_features is not None:
                    segs = _adapt_tensor_feature_dim(segs, expected_n_features)
                logits = model(segs, masks, digits, seq_mask)
            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    return roc_auc_score(np.array(all_labels), np.array(all_probs))


def permuted_auc(model, loader, feat_idx, mode, rng, n_repeats=5, expected_n_features=None):
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
                    if expected_n_features is not None:
                        feats = _adapt_tensor_feature_dim(feats, expected_n_features)
                    # Permute feature feat_idx across the batch
                    B, T, F = feats.shape
                    perm = torch.randperm(B * T, generator=None)
                    col = feats[:, :, feat_idx].reshape(-1)
                    feats = feats.clone()
                    feats[:, :, feat_idx] = col[perm].reshape(B, T)
                    logits = model(feats, mask, digit)
                else:
                    segs, masks, digits, label, seq_mask = [b.to(DEVICE) for b in batch]
                    if expected_n_features is not None:
                        segs = _adapt_tensor_feature_dim(segs, expected_n_features)
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

def _load_token_to_idx(npz_path, model_dir):
    vocab_path = os.path.join(model_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            token_to_idx = json.load(f)
        # Ensure indices are integers even if loaded from JSON.
        return {str(k): int(v) for k, v in token_to_idx.items()}

    data = np.load(npz_path, allow_pickle=True)
    digit_sequences = data['digit_sequences']
    vocab = sorted({str(token) for seq in digit_sequences for token in np.asarray(seq).tolist()})
    return {token: idx for idx, token in enumerate(vocab)}


def load_segments_by_digit(npz_path, token_to_idx):
    data = np.load(npz_path, allow_pickle=True)
    digit_segments = data['digit_segments']
    digit_sequences = data['digit_sequences']

    by_digit = {d: [] for d in range(len(token_to_idx))}
    for vid_segs, vid_digits in zip(digit_segments, digit_sequences):
        for seg, digit in zip(vid_segs, vid_digits):
            idx = token_to_idx[str(digit)]
            by_digit[idx].append(seg)
    return by_digit


def print_ranking(scores, feature_names, label):
    order = np.argsort(scores)[::-1]
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  {'Feature':<18}  {'Score':>10}  {'Rank':>4}")
    print(f"  {'-'*36}")
    for rank, i in enumerate(order, 1):
        print(f"  {feature_names[i]:<18}  {scores[i]:>10.4f}  {rank:>4}")


def _normalize_for_plot(values):
    v = np.asarray(values, dtype=np.float64)
    vmax = np.max(v)
    vmin = np.min(v)
    if np.isclose(vmax, vmin):
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def _metric_filename(metric_name):
    if metric_name == 'Fisher Ratio':
        return 'fisher.png'
    if metric_name == 'Mutual Info':
        return 'mi.png'
    if metric_name == 'Permutation AUC Drop':
        return 'permutation.png'
    return metric_name.lower().replace(' ', '_') + '.png'


def _metric_color(metric_name):
    color_map = {
        'Fisher Ratio': '#2C7FB8',
        'Mutual Info': '#D95F5F',
        'Permutation AUC Drop': '#41AB5D',
    }
    return color_map.get(metric_name, '#4C78A8')


def plot_single_metric_bar(feature_names, scores, metric_name, output_dir,
                           dataset, top_k=None):
    """Create a publication-ready horizontal bar chart for one method."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        top_k = max(1, min(int(top_k), len(feature_names)))
        order = order[:top_k]

    selected_names = [feature_names[i] for i in order]
    selected_scores = _normalize_for_plot(scores[order])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'axes.titlesize': 15,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    n_rows = len(selected_names)
    fig_h = max(4.5, 0.45 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))

    y = np.arange(n_rows)
    ax.barh(
        y,
        selected_scores,
        height=0.72,
        color=_metric_color(metric_name),
        edgecolor='black',
        linewidth=0.6,
        alpha=0.95,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(selected_names)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel('Normalized Discriminability Score')
    ax.set_title(f'Feature Discriminability ({dataset})')

    ax.grid(axis='x', linestyle='--', alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, _metric_filename(metric_name))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path


def load_feature_names(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if 'feature_names' not in data:
        return [f'feature_{i}' for i in range(data['digit_segments'][0][0].shape[1])]
    return [str(x) for x in data['feature_names'].tolist()]


def _default_model_candidates(model_dir, mode):
    return [
        os.path.join(model_dir, f'best_{mode}_verifier.pt'),
        os.path.join(model_dir, 'transformer_encoder', f'best_{mode}_verifier.pt'),
    ]


def _resolve_model_path(model_dir, mode):
    for p in _default_model_candidates(model_dir, mode):
        if os.path.exists(p):
            return p
    # Keep old behavior in the error message path.
    return os.path.join(model_dir, f'best_{mode}_verifier.pt')


def _infer_encoder_type(state_dict):
    if 'lip_encoder.gru.weight_ih_l0' in state_dict:
        return 'bigru'
    if 'lip_encoder.input_proj.weight' in state_dict:
        return 'transformer'
    raise RuntimeError('Could not infer encoder type from checkpoint keys.')


def _infer_n_features(state_dict):
    key = 'lip_encoder.conv.0.weight'
    if key not in state_dict:
        raise RuntimeError('Could not infer n_features from checkpoint.')
    return int(state_dict[key].shape[1])


def _infer_hidden_dim(state_dict, encoder_type):
    if encoder_type == 'bigru':
        key = 'lip_encoder.gru.weight_ih_l0'
        if key not in state_dict:
            raise RuntimeError('Could not infer bigru hidden_dim from checkpoint.')
        return int(state_dict[key].shape[0] // 3)

    key = 'lip_encoder.input_proj.weight'
    if key not in state_dict:
        raise RuntimeError('Could not infer transformer hidden_dim from checkpoint.')
    return int(state_dict[key].shape[0])


def _infer_embed_dim(state_dict):
    key = 'digit_embedding.weight'
    if key not in state_dict:
        raise RuntimeError('Could not infer embed_dim from checkpoint.')
    return int(state_dict[key].shape[1])


def _infer_n_classes(state_dict):
    key = 'digit_embedding.weight'
    if key not in state_dict:
        raise RuntimeError('Could not infer n_classes from checkpoint.')
    return int(state_dict[key].shape[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit')
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='digit')
    parser.add_argument('--perm_repeats', type=int, default=5,
                        help='Number of permutation repeats per feature (more = stabler estimate)')
    parser.add_argument('--plot_root', type=str, default='ablation',
                        help='Root folder for saving plots (plots go to <plot_root>/<dataset>/)')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Deprecated: ignored. Use --plot_root.')
    parser.add_argument('--plot_top_k', type=int, default=None,
                        help='If set, each method plot shows top-k features')
    args = parser.parse_args()

    processed_dir = os.path.join(PROCESSED_ROOT, args.dataset)
    model_dir = os.path.join(MODEL_ROOT, args.dataset)
    test_path = os.path.join(processed_dir, 'test.npz')
    model_path = _resolve_model_path(model_dir, args.mode)
    token_to_idx = _load_token_to_idx(test_path, model_dir)
    n_classes = len(token_to_idx)
    feature_names = load_feature_names(test_path)
    n_features = len(feature_names)

    # ---- 1 & 2: Model-free methods ----------------------------------------
    print("Loading segments...")
    by_digit = load_segments_by_digit(test_path, token_to_idx)
    n_segs = sum(len(v) for v in by_digit.values())
    print(f"  {n_segs} digit segments across {n_classes} classes")

    fdr = fisher_discriminant_ratio(by_digit)
    print_ranking(fdr, feature_names, "Fisher Discriminant Ratio (higher = more separable)")

    mi = mutual_information(by_digit, n_features)
    print_ranking(mi, feature_names, "Mutual Information (higher = more informative)")
    scores_for_plot = {
        'Fisher Ratio': fdr,
        'Mutual Info': mi,
    }

    # ---- 3: Permutation Importance ----------------------------------------
    if not os.path.exists(model_path):
        print(f"\nSkipping permutation importance: model not found at {model_path}")
    else:
        print(f"\nLoading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        encoder_type = _infer_encoder_type(state_dict)
        ckpt_n_features = _infer_n_features(state_dict)
        ckpt_hidden_dim = _infer_hidden_dim(state_dict, encoder_type)
        ckpt_embed_dim = _infer_embed_dim(state_dict)
        ckpt_n_classes = _infer_n_classes(state_dict)

        if args.mode == 'digit':
            dataset = LipVerificationDataset(
                test_path,
                dataset=args.dataset,
                token_to_idx=token_to_idx,
                seed=99,
            )
            model = DigitVerifier(
                n_classes=ckpt_n_classes,
                embed_dim=ckpt_embed_dim,
                n_features=ckpt_n_features,
                hidden_dim=ckpt_hidden_dim,
                encoder_type=encoder_type,
            ).to(DEVICE)
            loader = DataLoader(dataset, batch_size=256, shuffle=False,
                                num_workers=0, collate_fn=None)
        else:
            dataset = SequenceVerificationDataset(
                test_path,
                dataset=args.dataset,
                token_to_idx=token_to_idx,
                seed=99,
            )
            model = SequenceVerifier(
                n_classes=ckpt_n_classes,
                embed_dim=ckpt_embed_dim,
                n_features=ckpt_n_features,
                hidden_dim=ckpt_hidden_dim,
                encoder_type=encoder_type,
            ).to(DEVICE)
            loader = DataLoader(dataset, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=sequence_collate_fn)

        model.load_state_dict(state_dict)
        model.eval()
        print(f"  Loaded checkpoint encoder={encoder_type}, n_features={ckpt_n_features}, hidden_dim={ckpt_hidden_dim}, embed_dim={ckpt_embed_dim}")
        if dataset.n_features != ckpt_n_features:
            print(f"  Dataset has {dataset.n_features} features; adapting batches to checkpoint width {ckpt_n_features}.")

        print(f"Computing baseline AUC...")
        baseline_auc = get_baseline_auc(model, loader, expected_n_features=ckpt_n_features)
        print(f"  Baseline AUC: {baseline_auc:.4f}")

        rng = np.random.RandomState(0)
        perm_drops = np.zeros(n_features)
        for i, name in enumerate(feature_names):
            if i >= ckpt_n_features:
                perm_drops[i] = 0.0
                print(f"  {name:<18} skipped (feature not used by checkpoint)")
                continue
            auc_i = permuted_auc(
                model,
                loader,
                i,
                args.mode,
                rng,
                args.perm_repeats,
                expected_n_features=ckpt_n_features,
            )
            perm_drops[i] = baseline_auc - auc_i
            print(f"  {name:<18} permuted AUC={auc_i:.4f}  drop={perm_drops[i]:+.4f}")

        print_ranking(perm_drops, feature_names, f"Permutation Importance — AUC drop (baseline={baseline_auc:.4f})")
        scores_for_plot['Permutation AUC Drop'] = perm_drops

    # ---- Summary table ----------------------------------------------------
    print(f"\n{'='*55}")
    print("  Summary: feature ranks across methods")
    print(f"{'='*55}")
    fdr_rank = np.argsort(np.argsort(fdr)[::-1]) + 1
    mi_rank  = np.argsort(np.argsort(mi)[::-1]) + 1
    print(f"  {'Feature':<18}  {'FDR rank':>8}  {'MI rank':>7}")
    print(f"  {'-'*37}")
    for i, name in enumerate(feature_names):
        print(f"  {name:<18}  {fdr_rank[i]:>8}  {mi_rank[i]:>7}")

    plot_dir = os.path.join(args.plot_root, args.dataset)
    saved_paths = []
    for metric_name, metric_scores in scores_for_plot.items():
        p = plot_single_metric_bar(
            feature_names=feature_names,
            scores=metric_scores,
            metric_name=metric_name,
            output_dir=plot_dir,
            dataset=args.dataset,
            top_k=args.plot_top_k,
        )
        saved_paths.append(p)

    print("\nSaved publication-style charts:")
    for p in saved_paths:
        print(f"  - {p}")
