"""
Lip-Text Verification Training
-------------------------------
Task: Given a lip movement time series and a digit sequence, verify whether they match.

Approach: Siamese-style contrastive learning.
- Encode lip segments per digit using a 1D-CNN + GRU encoder
- Compare encoded lip representation against a digit embedding
- Train with positive (matching) and negative (mismatched) pairs
- Evaluate with EER, accuracy, ROC-AUC
- Logs metrics to TensorBoard
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import os
import json
import time

from model import DigitVerifier, SequenceVerifier, CHAR_TO_IDX, N_CLASSES

# --- Configuration ---
PROCESSED_DIR = 'processed_data'
MODEL_DIR = 'models'
LOG_DIR = 'runs'
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
MAX_SEQ_LEN = 30        # max frames per digit segment (pad/truncate)
N_FEATURES = 5          # lip features per frame
EMBED_DIM = 64          # embedding dimension
HIDDEN_DIM = 128        # GRU hidden dimension
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 50
NEG_RATIO = 1           # number of negative pairs per positive pair
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')


def char_to_idx(c):
    """Convert a digit character (0-9 or !) to an integer index."""
    return CHAR_TO_IDX[c]


# --- Dataset ---
class LipVerificationDataset(Dataset):
    """
    Per-digit verification dataset.
    Each sample is a (lip_segment, claimed_digit_idx, match_label) triplet.
    """
    def __init__(self, npz_path, max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        # Build flat list of (segment_features, char_idx)
        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, char_to_idx(str(digit))))

        # Build pairs: positive + negative
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


class SequenceVerificationDataset(Dataset):
    """
    Full-sequence verification: verify entire 8-digit sequence at once.
    """
    def __init__(self, npz_path, max_seg_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        self.digit_segments = data['digit_segments']
        self.digit_sequences = data['digit_sequences']
        self.max_seg_len = max_seg_len
        self.rng = np.random.RandomState(seed)
        self.n_videos = len(self.digit_segments)

        self.pairs = []
        for i in range(self.n_videos):
            digits = [char_to_idx(str(d)) for d in self.digit_sequences[i]]
            # Positive
            self.pairs.append((i, digits, 1))
            # Negatives
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

        return (
            torch.FloatTensor(np.array(all_feats)),
            torch.FloatTensor(np.array(all_masks)),
            torch.LongTensor(claimed_digits),
            torch.FloatTensor([label]),
        )


# --- Training ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        if len(batch[0].shape) == 3:
            feats, mask, digit, label = [b.to(device) for b in batch]
            logits = model(feats, mask, digit)
        else:
            segs, masks, digits, label = [b.to(device) for b in batch]
            logits = model(segs, masks, digits)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f'{total_loss / n_batches:.4f}')

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            if len(batch[0].shape) == 3:
                feats, mask, digit, label = [b.to(device) for b in batch]
                logits = model(feats, mask, digit)
            else:
                segs, masks, digits, label = [b.to(device) for b in batch]
                logits = model(segs, masks, digits)

            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    acc = np.mean((all_probs >= eer_threshold).astype(int) == all_labels)
    acc_05 = np.mean((all_probs >= 0.5).astype(int) == all_labels)

    return {
        'auc': auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'acc_at_eer': acc,
        'acc_at_05': acc_05,
        'n_samples': len(all_labels),
        'n_positive': int(all_labels.sum()),
        'n_negative': int((1 - all_labels).sum()),
    }


# --- Main ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='sequence',
                        help='digit: per-digit verification, sequence: full 8-digit verification')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    train_path = os.path.join(PROCESSED_DIR, 'train.npz')
    test_path = os.path.join(PROCESSED_DIR, 'test.npz')

    if not os.path.exists(train_path):
        print("ERROR: Preprocessed data not found. Run preprocess.py first.")
        exit(1)

    print(f"\nMode: {args.mode}-level verification")
    print(f"Vocabulary: {N_CLASSES} classes (0-9 + !)")
    print(f"Loading data...")

    if args.mode == 'digit':
        train_ds = LipVerificationDataset(train_path)
        test_ds = LipVerificationDataset(test_path, seed=99)
        model = DigitVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                              n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)
    else:
        train_ds = SequenceVerificationDataset(train_path)
        test_ds = SequenceVerificationDataset(test_path, seed=99)
        model = SequenceVerifier(n_classes=N_CLASSES, embed_dim=EMBED_DIM,
                                 n_features=N_FEATURES, hidden_dim=HIDDEN_DIM).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} pairs, Test: {len(test_ds)} pairs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f'{args.mode}_lr{args.lr}_bs{args.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    print(f"TensorBoard logs: {LOG_DIR}/{run_name}")

    best_auc = 0
    results_log = []

    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Evaluate every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, DEVICE)
            results_log.append({'epoch': epoch, 'loss': loss, **metrics})

            writer.add_scalar('eval/auc', metrics['auc'], epoch)
            writer.add_scalar('eval/eer', metrics['eer'], epoch)
            writer.add_scalar('eval/acc_at_eer', metrics['acc_at_eer'], epoch)
            writer.add_scalar('eval/acc_at_05', metrics['acc_at_05'], epoch)

            print(f"Epoch {epoch:3d} | loss={loss:.4f} | AUC={metrics['auc']:.4f} | "
                  f"EER={metrics['eer']:.4f} | Acc@EER={metrics['acc_at_eer']:.4f} | "
                  f"Acc@0.5={metrics['acc_at_05']:.4f}")

            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                torch.save(model.state_dict(),
                           os.path.join(MODEL_DIR, f'best_{args.mode}_verifier.pt'))
        else:
            print(f"Epoch {epoch:3d} | loss={loss:.4f}")

    # Final evaluation
    print(f"\n{'='*55}")
    print("Final evaluation on test set:")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'best_{args.mode}_verifier.pt'),
                                     weights_only=True))
    final_metrics = evaluate(model, test_loader, DEVICE)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    writer.add_hparams(
        {'mode': args.mode, 'lr': args.lr, 'batch_size': args.batch_size,
         'epochs': args.epochs, 'embed_dim': EMBED_DIM, 'hidden_dim': HIDDEN_DIM},
        {'hparam/auc': final_metrics['auc'], 'hparam/eer': final_metrics['eer'],
         'hparam/acc': final_metrics['acc_at_eer']},
    )
    writer.close()

    # Save results
    results = {
        'mode': args.mode,
        'hyperparams': {
            'max_seq_len': MAX_SEQ_LEN,
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'neg_ratio': NEG_RATIO,
            'n_classes': N_CLASSES,
        },
        'final_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v
                          for k, v in final_metrics.items()},
        'training_log': results_log,
    }
    results_path = os.path.join(MODEL_DIR, f'results_{args.mode}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"\nResults saved to {results_path}")
