"""
Lip-Text Verification Testing
------------------------------
Evaluate a trained model on the test set.
Reports: AUC, EER, accuracy, per-digit breakdown, confusion-style analysis.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from tqdm import tqdm
import os
import json
import argparse

from model import (DigitVerifier, SequenceVerifier,
                   CHAR_TO_IDX, VOCAB, N_CLASSES)

# --- Configuration ---
PROCESSED_DIR = 'processed_data'
MODEL_DIR = 'models'
MAX_SEQ_LEN = 30
N_FEATURES = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')


def char_to_idx(c):
    return CHAR_TO_IDX[c]


# --- Datasets (same as train.py) ---
class LipVerificationDataset(Dataset):
    def __init__(self, npz_path, max_seq_len=MAX_SEQ_LEN, neg_ratio=1, seed=99):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, char_to_idx(str(digit))))

        all_indices = list(range(N_CLASSES))
        self.pairs = []
        for i, (seg, cidx) in enumerate(self.segments):
            self.pairs.append((i, cidx, 1))
            wrong = [d for d in all_indices if d != cidx]
            for _ in range(neg_ratio):
                self.pairs.append((i, self.rng.choice(wrong), 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seg_idx, claimed, label = self.pairs[idx]
        seg_features, _ = self.segments[seg_idx]
        T = seg_features.shape[0]
        if T >= self.max_seq_len:
            feat = seg_features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            feat = np.zeros((self.max_seq_len, N_FEATURES), dtype=np.float32)
            feat[:T] = seg_features
            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:T] = 1.0
        return (
            torch.FloatTensor(feat),
            torch.FloatTensor(mask),
            torch.LongTensor([claimed]),
            torch.FloatTensor([label]),
        )

    def get_pair_info(self, idx):
        """Return (seg_idx, true_digit, claimed_digit, label) for analysis."""
        seg_idx, claimed, label = self.pairs[idx]
        _, true_digit = self.segments[seg_idx]
        return seg_idx, true_digit, claimed, label


class SequenceVerificationDataset(Dataset):
    def __init__(self, npz_path, max_seg_len=MAX_SEQ_LEN, neg_ratio=1, seed=99):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seg_len = max_seg_len
        self.rng = np.random.RandomState(seed)
        self.n_videos = len(self.digit_segments)

        self.pairs = []
        for i in range(self.n_videos):
            digits = [char_to_idx(str(d)) for d in self.digit_sequences[i]]
            self.pairs.append((i, digits, 1))
            for _ in range(neg_ratio):
                wrong_digits = digits.copy()
                if self.rng.random() < 0.5:
                    self.rng.shuffle(wrong_digits)
                    if wrong_digits == digits:
                        wrong_digits[0] = (wrong_digits[0] + 1) % N_CLASSES
                else:
                    n_replace = self.rng.randint(2, 5)
                    positions = self.rng.choice(len(wrong_digits), n_replace, replace=False)
                    for pos in positions:
                        wrong_digits[pos] = self.rng.choice(
                            [d for d in range(N_CLASSES) if d != wrong_digits[pos]]
                        )
                self.pairs.append((i, wrong_digits, 0))

    def __len__(self):
        return len(self.pairs)

    def _pad_segment(self, seg):
        T = seg.shape[0]
        if T >= self.max_seg_len:
            feat = seg[:self.max_seg_len].astype(np.float32)
            mask = np.ones(self.max_seg_len, dtype=np.float32)
        else:
            feat = np.zeros((self.max_seg_len, N_FEATURES), dtype=np.float32)
            feat[:T] = seg
            mask = np.zeros(self.max_seg_len, dtype=np.float32)
            mask[:T] = 1.0
        return feat, mask

    def __getitem__(self, idx):
        vid_idx, claimed_digits, label = self.pairs[idx]
        segments = self.digit_segments[vid_idx]
        all_feats, all_masks = [], []
        for seg in segments:
            f, m = self._pad_segment(seg)
            all_feats.append(f)
            all_masks.append(m)
        n_digits = len(segments)
        return (
            torch.FloatTensor(np.array(all_feats)),
            torch.FloatTensor(np.array(all_masks)),
            torch.LongTensor(claimed_digits),
            torch.FloatTensor([label]),
            n_digits,
        )


def sequence_collate_fn(batch):
    """Collate sequences with variable number of digits by padding to max in batch."""
    max_digits = max(item[4] for item in batch)
    T = batch[0][0].shape[1]
    F = batch[0][0].shape[2]

    batch_feats, batch_masks, batch_digits, batch_labels, batch_seq_masks = [], [], [], [], []
    for feats, masks, digits, label, n_dig in batch:
        pad_n = max_digits - n_dig
        if pad_n > 0:
            feats = torch.cat([feats, torch.zeros(pad_n, T, F)], dim=0)
            masks = torch.cat([masks, torch.zeros(pad_n, T)], dim=0)
            digits = torch.cat([digits, torch.zeros(pad_n, dtype=torch.long)], dim=0)
        seq_mask = torch.zeros(max_digits)
        seq_mask[:n_dig] = 1.0
        batch_feats.append(feats)
        batch_masks.append(masks)
        batch_digits.append(digits)
        batch_labels.append(label)
        batch_seq_masks.append(seq_mask)

    return (
        torch.stack(batch_feats),
        torch.stack(batch_masks),
        torch.stack(batch_digits),
        torch.stack(batch_labels),
        torch.stack(batch_seq_masks),
    )


# --- Evaluation ---
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

    return all_labels, all_probs, preds_eer, preds_05, metrics


def per_digit_analysis(dataset, all_labels, all_probs, threshold):
    """Per-digit accuracy breakdown (digit-mode only)."""
    digit_stats = {v: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for v in VOCAB}

    for i in range(len(dataset)):
        _, true_digit, claimed, label = dataset.get_pair_info(i)
        pred = 1 if all_probs[i] >= threshold else 0
        claimed_char = VOCAB[claimed]

        if label == 1:  # positive pair (true_digit == claimed)
            if pred == 1:
                digit_stats[claimed_char]['tp'] += 1
            else:
                digit_stats[claimed_char]['fn'] += 1
        else:  # negative pair
            if pred == 0:
                digit_stats[claimed_char]['tn'] += 1
            else:
                digit_stats[claimed_char]['fp'] += 1

    return digit_stats


def print_results(metrics, digit_stats=None):
    """Print formatted test results."""
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

    if digit_stats:
        print(f"\n  Per-digit breakdown (threshold={metrics['eer_threshold']:.4f}):")
        print(f"  {'Digit':>5} | {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} | {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-'*5}-+-{'-'*23}-+-{'-'*20}")
        for digit in VOCAB:
            s = digit_stats[digit]
            total = s['tp'] + s['fp'] + s['tn'] + s['fn']
            if total == 0:
                continue
            prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
            rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {digit:>5} | {s['tp']:>5} {s['fp']:>5} {s['tn']:>5} {s['fn']:>5} | "
                  f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        print()


# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lip-text verification model')
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='sequence',
                        help='digit: per-digit, sequence: full 8-digit')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (default: models/best_{mode}_verifier.pt)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom decision threshold (default: use EER threshold)')
    parser.add_argument('--save', action='store_true',
                        help='Save detailed results to JSON')
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(MODEL_DIR, f'best_{args.mode}_verifier.pt')
    test_path = os.path.join(PROCESSED_DIR, 'test.npz')

    if not os.path.exists(test_path):
        print("ERROR: Test data not found. Run preprocess.py first.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint not found at {model_path}. Run train.py first.")
        exit(1)

    print(f"Device: {DEVICE}")
    print(f"Mode: {args.mode}-level verification")
    print(f"Model: {model_path}")

    # Load model
    if args.mode == 'digit':
        model = DigitVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                              n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)
    else:
        model = SequenceVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                 n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    if args.mode == 'digit':
        test_ds = LipVerificationDataset(test_path)
    else:
        test_ds = SequenceVerificationDataset(test_path)

    collate_fn = sequence_collate_fn if args.mode == 'sequence' else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, collate_fn=collate_fn)
    print(f"Test samples: {len(test_ds)}")

    # Evaluate
    all_labels, all_probs, preds_eer, preds_05, metrics = evaluate(model, test_loader, DEVICE)

    # Use custom threshold if provided
    threshold = args.threshold if args.threshold is not None else metrics['eer_threshold']
    if args.threshold is not None:
        custom_preds = (all_probs >= threshold).astype(int)
        metrics['custom_threshold'] = threshold
        metrics['acc_at_custom'] = float(np.mean(custom_preds == all_labels))

    # Per-digit breakdown (digit mode only)
    digit_stats = None
    if args.mode == 'digit':
        digit_stats = per_digit_analysis(test_ds, all_labels, all_probs, threshold)

    print_results(metrics, digit_stats)

    # Confusion matrix (binary: match vs no-match)
    preds = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    print(f"  Confusion matrix (threshold={threshold:.4f}):")
    print(f"               Predicted")
    print(f"              No-match  Match")
    print(f"  Actual No   {cm[0][0]:>7}  {cm[0][1]:>5}")
    print(f"  Actual Yes  {cm[1][0]:>7}  {cm[1][1]:>5}")
    print()

    # Save results
    if args.save:
        output = {
            'mode': args.mode,
            'model_path': model_path,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
        }
        if digit_stats:
            output['per_digit'] = digit_stats

        out_path = os.path.join(MODEL_DIR, f'test_results_{args.mode}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")
