"""
Lip-Text Verification Testing
------------------------------
Evaluate a trained verification model on the test split.
Reports AUC, EER, accuracy, per-token breakdown, confusion matrix.

Models are reconstructed from the config_<mode>.json written next to the
checkpoint by train.py. Legacy checkpoints (no config) fall back to the
historical hardcoded dimensions.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import checkpoint as ckpt
from dataset import (
    LipVerificationDataset,
    SequenceVerificationDataset,
    sequence_collate_fn,
)
from model import DigitVerifier, SequenceVerifier, CHAR_TO_IDX
from transforms import load_feature_stats

PROCESSED_ROOT = 'processed_data'
MODEL_ROOT = 'models'
EMBED_DIM = 64      # legacy fallback (checkpoints without config)
HIDDEN_DIM = 128    # legacy fallback
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def _candidate_model_paths(dataset, mode, encoder=None):
    """New layout first, then legacy layouts."""
    encoders = ([encoder] if encoder else []) + ['transformer', 'bigru']
    paths = [os.path.join(MODEL_ROOT, dataset, enc, f'best_{mode}_verifier.pt')
             for enc in encoders]
    paths.append(os.path.join(MODEL_ROOT, 'transformer_encoder', f'best_{mode}_verifier.pt'))
    paths.append(os.path.join(MODEL_ROOT, f'best_{mode}_verifier.pt'))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def evaluate(model, loader, device):
    """Run model on all batches, return labels, probabilities, and metrics."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing', leave=False):
            if len(batch) == 4:
                feats, mask, digit, label = [b.to(device) for b in batch]
                logits = model(feats, mask, digit)
            else:
                segs, masks, digits, label, seq_mask = [b.to(device) for b in batch]
                logits = model(segs, masks, digits, seq_mask)

            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
    preds_eer = (all_probs >= eer_threshold).astype(int)
    preds_05 = (all_probs >= 0.5).astype(int)

    metrics = {
        'auc': float(auc),
        'eer': eer,
        'eer_threshold': eer_threshold,
        'acc_at_eer': float(np.mean(preds_eer == all_labels)),
        'acc_at_05': float(np.mean(preds_05 == all_labels)),
        'n_samples': int(len(all_labels)),
        'n_positive': int(all_labels.sum()),
        'n_negative': int((1 - all_labels).sum()),
    }
    return all_labels, all_probs, metrics


def per_token_analysis(dataset, all_probs, threshold, idx_to_token):
    """Per-token accuracy breakdown over claimed tokens (digit mode only)."""
    token_stats = {token: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                   for token in idx_to_token.values()}
    for i in range(len(dataset)):
        _, _, claimed_idx, label = dataset.get_pair_info(i)
        pred = 1 if all_probs[i] >= threshold else 0
        claimed_char = idx_to_token[claimed_idx]
        if label == 1:
            token_stats[claimed_char]['tp' if pred == 1 else 'fn'] += 1
        else:
            token_stats[claimed_char]['tn' if pred == 0 else 'fp'] += 1
    return token_stats


def print_results(metrics, token_stats=None):
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  EER:            {metrics['eer']:.4f}")
    print(f"  EER threshold:  {metrics['eer_threshold']:.4f}")
    print(f"  Acc @ EER:      {metrics['acc_at_eer']:.4f}")
    print(f"  Acc @ 0.5:      {metrics['acc_at_05']:.4f}")
    print(f"  Samples:        {metrics['n_samples']} "
          f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})")
    print(f"{'='*60}")

    if token_stats:
        print(f"\n  Per-token breakdown (threshold={metrics['eer_threshold']:.4f}):")
        print(f"  {'Token':>7} | {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} | {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-'*7}-+-{'-'*23}-+-{'-'*20}")
        for token, s in sorted(token_stats.items()):
            total = s['tp'] + s['fp'] + s['tn'] + s['fn']
            if total == 0:
                continue
            prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
            rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {token:>7} | {s['tp']:>5} {s['fp']:>5} {s['tn']:>5} {s['fn']:>5} | "
                  f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        print()


def _load_vocab(model_path, dataset):
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            return {str(k): int(v) for k, v in json.load(f).items()}
    if dataset == 'digit':
        return dict(CHAR_TO_IDX)
    raise RuntimeError(f'vocab.json not found next to checkpoint ({model_path}); '
                       'required for non-digit datasets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lip-text verification model')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit')
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='digit')
    parser.add_argument('--encoder', choices=['bigru', 'transformer'], default=None,
                        help='Encoder variant (default: taken from checkpoint config)')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data-dir', default=PROCESSED_ROOT)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.model_path is not None:
        model_path = args.model_path
    else:
        candidates = _candidate_model_paths(args.dataset, args.mode, args.encoder)
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    test_path = os.path.join(args.data_dir, args.dataset, 'test.npz')
    if not os.path.exists(test_path):
        print(f'ERROR: Test data not found at {test_path}. Run preprocess.py first.')
        raise SystemExit(1)
    if not os.path.exists(model_path):
        print(f'ERROR: Model checkpoint not found at {model_path}. Run train.py first.')
        raise SystemExit(1)

    config = ckpt.load_model_config(model_path, mode=args.mode)
    encoder = config['encoder_type'] if config else (args.encoder or 'transformer')

    print(f'Device: {DEVICE}')
    print(f'Dataset: {args.dataset}')
    print(f'Mode: {args.mode}-level verification')
    print(f'Encoder: {encoder}')
    print(f'Model: {model_path}')

    token_to_idx = _load_vocab(model_path, args.dataset)
    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    resample = bool(config.get('resample', False)) if config else False
    seg_len = int(config.get('seg_len', 16)) if config else 16
    feature_stats = None
    if config and config.get('standardized'):
        stats_path = os.path.join(args.data_dir, args.dataset, 'feature_stats.json')
        feature_stats = load_feature_stats(stats_path)
        if list(feature_stats['feature_names']) != list(config['feature_names']):
            raise RuntimeError('feature_stats.json feature order does not match checkpoint config')

    if args.mode == 'digit':
        test_ds = LipVerificationDataset(test_path, dataset=args.dataset,
                                         token_to_idx=token_to_idx, seed=99,
                                         resample=resample, seg_len=seg_len,
                                         feature_stats=feature_stats)
    else:
        test_ds = SequenceVerificationDataset(test_path, dataset=args.dataset,
                                              token_to_idx=token_to_idx, seed=99,
                                              resample=resample, seg_len=seg_len,
                                              feature_stats=feature_stats)

    collate_fn = sequence_collate_fn if args.mode == 'sequence' else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, collate_fn=collate_fn)
    print(f'Test samples: {len(test_ds)}')

    if config:
        model = ckpt.build_model_from_config(config, DEVICE)
    elif args.mode == 'digit':
        model = DigitVerifier(n_classes=test_ds.vocab_size, embed_dim=EMBED_DIM,
                              n_features=test_ds.n_features, hidden_dim=HIDDEN_DIM,
                              encoder_type=encoder).to(DEVICE)
    else:
        model = SequenceVerifier(n_classes=test_ds.vocab_size, embed_dim=EMBED_DIM,
                                 n_features=test_ds.n_features, hidden_dim=HIDDEN_DIM,
                                 encoder_type=encoder).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    all_labels, all_probs, metrics = evaluate(model, test_loader, DEVICE)

    threshold = args.threshold if args.threshold is not None else metrics['eer_threshold']
    if args.threshold is not None:
        custom_preds = (all_probs >= threshold).astype(int)
        metrics['custom_threshold'] = threshold
        metrics['acc_at_custom'] = float(np.mean(custom_preds == all_labels))

    token_stats = None
    if args.mode == 'digit':
        token_stats = per_token_analysis(test_ds, all_probs, threshold, idx_to_token)

    print_results(metrics, token_stats)

    preds = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    print(f'  Confusion matrix (threshold={threshold:.4f}):')
    print(f'               Predicted')
    print(f'              No-match  Match')
    print(f'  Actual No   {cm[0][0]:>7}  {cm[0][1]:>5}')
    print(f'  Actual Yes  {cm[1][0]:>7}  {cm[1][1]:>5}')
    print()

    if args.save:
        output = {
            'dataset': args.dataset,
            'mode': args.mode,
            'encoder': encoder,
            'model_path': model_path,
            'resample': resample,
            'standardized': feature_stats is not None,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
        }
        if token_stats:
            output['per_token'] = token_stats

        save_dir = os.path.dirname(model_path) or MODEL_ROOT
        out_path = os.path.join(save_dir, f'test_results_{args.mode}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'Results saved to {out_path}')
