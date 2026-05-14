"""Training entrypoint for verification and seq2seq transcription models."""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
)
from model import DigitVerifier, N_CLASSES, SequenceVerifier, TinyLipSeq2Seq

# --- Configuration ---
PROCESSED_DIR = 'processed_data'
MODEL_DIR = 'models'
LOG_DIR = 'runs'
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
MAX_SEQ_LEN = 30        # max frames per digit segment (pad/truncate)
N_FEATURES = 8          # 7 lip features + rms_energy per frame
EMBED_DIM = 64          # embedding dimension
HIDDEN_DIM = 128        # GRU hidden dimension
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
N_EPOCHS = 50
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

PAD_IDX = N_CLASSES
BOS_IDX = N_CLASSES + 1
EOS_IDX = N_CLASSES + 2
SEQ2SEQ_VOCAB_SIZE = N_CLASSES + 3


def sequence_collate_fn(batch):
    """Collate sequences with variable number of digits by padding to max in batch."""
    max_digits = max(item[4] for item in batch)
    t = batch[0][0].shape[1]
    f = batch[0][0].shape[2]

    batch_feats = []
    batch_masks = []
    batch_digits = []
    batch_labels = []
    batch_seq_masks = []

    for feats, masks, digits, label, n_dig in batch:
        pad_n = max_digits - n_dig
        if pad_n > 0:
            feats = torch.cat([feats, torch.zeros(pad_n, t, f)], dim=0)
            masks = torch.cat([masks, torch.zeros(pad_n, t)], dim=0)
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


def transcription_collate_fn(batch):
    """Pad variable-length segment and token sequences for seq2seq training."""
    max_digits = max(item[3] for item in batch)
    t = batch[0][0].shape[1]
    f = batch[0][0].shape[2]
    max_tgt_len = max_digits + 1  # digits + EOS

    batch_feats = []
    batch_masks = []
    batch_src_pad = []
    batch_targets = []

    for feats, masks, digits, n_dig in batch:
        pad_n = max_digits - n_dig
        if pad_n > 0:
            feats = torch.cat([feats, torch.zeros(pad_n, t, f)], dim=0)
            masks = torch.cat([masks, torch.zeros(pad_n, t)], dim=0)

        src_pad = torch.ones(max_digits, dtype=torch.bool)
        src_pad[:n_dig] = False

        target = torch.full((max_tgt_len,), PAD_IDX, dtype=torch.long)
        target[:n_dig] = digits
        target[n_dig] = EOS_IDX

        batch_feats.append(feats)
        batch_masks.append(masks)
        batch_src_pad.append(src_pad)
        batch_targets.append(target)

    return (
        torch.stack(batch_feats),
        torch.stack(batch_masks),
        torch.stack(batch_src_pad),
        torch.stack(batch_targets),
    )


# --- Training ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        if len(batch) == 4:
            feats, mask, digit, label = [b.to(device) for b in batch]
            logits = model(feats, mask, digit)
        else:
            segs, masks, digits, label, seq_mask = [b.to(device) for b in batch]
            logits = model(segs, masks, digits, seq_mask)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f'{total_loss / n_batches:.4f}')

    return total_loss / max(n_batches, 1)


def train_epoch_seq2seq(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for segments, masks, src_pad, targets in pbar:
        segments = segments.to(device)
        masks = masks.to(device)
        src_pad = src_pad.to(device)
        targets = targets.to(device)

        tgt_in = torch.full_like(targets, PAD_IDX)
        tgt_in[:, 0] = BOS_IDX
        tgt_in[:, 1:] = targets[:, :-1]

        logits = model(segments, masks, src_pad, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

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


def _strip_special(tokens):
    out = []
    for tok in tokens:
        if tok == EOS_IDX or tok == PAD_IDX:
            break
        out.append(tok)
    return out


def evaluate_seq2seq(model, loader, device):
    model.eval()
    total_tokens = 0
    total_correct_tokens = 0
    total_seq = 0
    exact_match = 0

    with torch.no_grad():
        for segments, masks, src_pad, targets in tqdm(loader, desc='Eval', leave=False):
            segments = segments.to(device)
            masks = masks.to(device)
            src_pad = src_pad.to(device)
            targets = targets.to(device)

            preds = model.greedy_decode(
                segments,
                masks,
                src_pad,
                bos_idx=BOS_IDX,
                max_len=targets.shape[1],
            )

            valid = targets.ne(PAD_IDX)
            token_correct = (preds == targets) & valid
            total_correct_tokens += token_correct.sum().item()
            total_tokens += valid.sum().item()

            for pred_row, tgt_row in zip(preds.cpu().tolist(), targets.cpu().tolist()):
                total_seq += 1
                if _strip_special(pred_row) == _strip_special(tgt_row):
                    exact_match += 1

    return {
        'token_acc': total_correct_tokens / max(total_tokens, 1),
        'exact_match_acc': exact_match / max(total_seq, 1),
        'n_sequences': total_seq,
        'n_tokens': total_tokens,
    }


# --- Main ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['digit', 'sequence', 'seq2seq'],
        default='sequence',
        help='digit: per-digit verification, sequence: sequence verification, seq2seq: transcription',
    )
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    train_path = os.path.join(PROCESSED_DIR, 'train.npz')
    test_path = os.path.join(PROCESSED_DIR, 'test.npz')

    if not os.path.exists(train_path):
        print('ERROR: Preprocessed data not found. Run preprocess.py first.')
        raise SystemExit(1)

    print(f'\nMode: {args.mode}')
    print(f'Vocabulary: {N_CLASSES} classes (0-9 + !)')
    print('Loading data...')

    if args.mode == 'digit':
        train_ds = LipVerificationDataset(train_path)
        test_ds = LipVerificationDataset(test_path, seed=99)
        model = DigitVerifier(
            n_classes=N_CLASSES,
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            hidden_dim=HIDDEN_DIM,
        ).to(DEVICE)
    elif args.mode == 'sequence':
        train_ds = SequenceVerificationDataset(train_path)
        test_ds = SequenceVerificationDataset(test_path, seed=99)
        model = SequenceVerifier(
            n_classes=N_CLASSES,
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            hidden_dim=HIDDEN_DIM,
        ).to(DEVICE)
    else:
        train_ds = LipTranscriptionDataset(train_path)
        test_ds = LipTranscriptionDataset(test_path)
        model = TinyLipSeq2Seq(
            vocab_size=SEQ2SEQ_VOCAB_SIZE,
            pad_idx=PAD_IDX,
            n_features=N_FEATURES,
            seg_embed_dim=48,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            ff_dim=128,
            dropout=0.1,
            max_src_len=12,
            max_tgt_len=12,
            hidden_dim=64,
        ).to(DEVICE)

    collate_fn = None
    if args.mode == 'sequence':
        collate_fn = sequence_collate_fn
    elif args.mode == 'seq2seq':
        collate_fn = transcription_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    print(f'Train: {len(train_ds)} samples, Test: {len(test_ds)} samples')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = (
        nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        if args.mode == 'seq2seq'
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f'{args.mode}_lr{args.lr}_bs{args.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    print(f'TensorBoard logs: {LOG_DIR}/{run_name}')

    best_metric = float('-inf')
    results_log = []
    model_suffix = 'seq2seq' if args.mode == 'seq2seq' else f'{args.mode}_verifier'
    model_save_path = os.path.join(MODEL_DIR, f'best_{model_suffix}.pt')

    print(f'\nTraining for {args.epochs} epochs...')

    for epoch in range(1, args.epochs + 1):
        if args.mode == 'seq2seq':
            loss = train_epoch_seq2seq(model, train_loader, optimizer, criterion, DEVICE)
        else:
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        eval_now = (epoch % 5 == 0 or epoch == args.epochs)
        if args.mode == 'seq2seq':
            eval_now = True

        if eval_now:
            metrics = (
                evaluate_seq2seq(model, test_loader, DEVICE)
                if args.mode == 'seq2seq'
                else evaluate(model, test_loader, DEVICE)
            )
            results_log.append({'epoch': epoch, 'loss': loss, **metrics})

            if args.mode == 'seq2seq':
                writer.add_scalar('eval/token_acc', metrics['token_acc'], epoch)
                writer.add_scalar('eval/exact_match_acc', metrics['exact_match_acc'], epoch)
                print(
                    f"Epoch {epoch:3d} | loss={loss:.4f} | "
                    f"TokenAcc={metrics['token_acc']:.4f} | "
                    f"ExactMatch={metrics['exact_match_acc']:.4f}"
                )
                track_metric = metrics['exact_match_acc']
            else:
                writer.add_scalar('eval/auc', metrics['auc'], epoch)
                writer.add_scalar('eval/eer', metrics['eer'], epoch)
                writer.add_scalar('eval/acc_at_eer', metrics['acc_at_eer'], epoch)
                writer.add_scalar('eval/acc_at_05', metrics['acc_at_05'], epoch)
                print(
                    f"Epoch {epoch:3d} | loss={loss:.4f} | AUC={metrics['auc']:.4f} | "
                    f"EER={metrics['eer']:.4f} | Acc@EER={metrics['acc_at_eer']:.4f} | "
                    f"Acc@0.5={metrics['acc_at_05']:.4f}"
                )
                track_metric = metrics['auc']

            if track_metric > best_metric:
                best_metric = track_metric
                torch.save(model.state_dict(), model_save_path)
        else:
            print(f'Epoch {epoch:3d} | loss={loss:.4f}')

    print(f"\n{'=' * 55}")
    print('Final evaluation on test set:')
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    final_metrics = (
        evaluate_seq2seq(model, test_loader, DEVICE)
        if args.mode == 'seq2seq'
        else evaluate(model, test_loader, DEVICE)
    )
    for k, v in final_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    if args.mode == 'seq2seq':
        writer.add_hparams(
            {
                'mode': args.mode,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'seg_embed_dim': 48,
                'hidden_dim': 64,
            },
            {
                'hparam/token_acc': final_metrics['token_acc'],
                'hparam/exact_match_acc': final_metrics['exact_match_acc'],
            },
        )
    else:
        writer.add_hparams(
            {
                'mode': args.mode,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
            },
            {
                'hparam/auc': final_metrics['auc'],
                'hparam/eer': final_metrics['eer'],
                'hparam/acc': final_metrics['acc_at_eer'],
            },
        )
    writer.close()

    results = {
        'mode': args.mode,
        'hyperparams': {
            'hidden_dim': HIDDEN_DIM,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'n_classes': N_CLASSES,
        },
        'final_metrics': {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in final_metrics.items()
        },
        'training_log': results_log,
    }
    results_suffix = 'seq2seq' if args.mode == 'seq2seq' else args.mode
    results_path = os.path.join(MODEL_DIR, f'results_{results_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    print(f'\nResults saved to {results_path}')
