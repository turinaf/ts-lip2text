"""Training entrypoint for verification and seq2seq transcription models."""
import json
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    SEG_LEN,
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
    compute_split_stats,
    sequence_collate_fn,
    transcription_collate_fn,
)
from transforms import load_feature_stats, save_feature_stats
import checkpoint as ckpt

# --- Configuration ---
PROCESSED_ROOT = 'processed_data'
MODEL_ROOT = 'models'
LOG_ROOT = 'runs'
os.makedirs(MODEL_ROOT, exist_ok=True)

# Hyperparameters
MAX_SEQ_LEN = 30        # max frames per digit segment (pad/truncate)
N_FEATURES = 7          # 7 lip features  per frame
EMBED_DIM = 64          # embedding dimension
HIDDEN_DIM = 128        # GRU hidden dimension
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
N_EPOCHS = 100
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

PAD_IDX = None
BOS_IDX = None
EOS_IDX = None
SEQ2SEQ_VOCAB_SIZE = None


def _resolve_dirs(args):
    processed_dir = os.path.join(args.data_dir, args.dataset)
    model_dir = os.path.join(args.model_root, args.dataset, args.encoder)
    log_dir = os.path.join(args.log_root, args.dataset, args.encoder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return processed_dir, model_dir, log_dir


def _speaker_split(train_path):
    """Deterministic speaker-level split: every 10th sorted speaker -> val."""
    data = np.load(train_path, allow_pickle=True)
    if 'speakers' not in data.files:
        return None, set()
    speakers = sorted({str(s) for s in data['speakers']})
    val = set(speakers[::10])
    return set(speakers) - val, val


def _max_sequence_length(dataset):
    return max(len(seq) for seq in dataset.digit_sequences)


def _build_digit_balanced_sampler(dataset):
    """Build inverse-frequency sampler over claimed tokens for digit mode."""
    claimed_tokens = np.array([int(claimed) for _, claimed, _ in dataset.pairs], dtype=np.int64)
    counts = np.bincount(claimed_tokens, minlength=dataset.vocab_size).astype(np.float64)
    counts[counts == 0.0] = 1.0
    weights = 1.0 / counts[claimed_tokens]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
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
        '--dataset',
        choices=['digit', 'grid'],
        default='digit',
        help='Select processed_data/<dataset> and dataset-specific vocabulary.',
    )
    parser.add_argument(
        '--mode',
        choices=['digit', 'sequence', 'seq2seq'],
        default='sequence',
        help='digit: per-digit verification, sequence: sequence verification, seq2seq: transcription',
    )
    parser.add_argument(
        '--encoder',
        choices=['bigru', 'transformer'],
        default='transformer',
        help='Sequence encoder for the lip feature extractor.',
    )
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--seg-len', type=int, default=SEG_LEN,
                        help='Fixed frame length for resampled segments.')
    parser.add_argument('--no-resample', action='store_true',
                        help='Disable fps-corrected fixed-length resampling (legacy pad-to-30 path).')
    parser.add_argument('--no-standardize', action='store_true',
                        help='Disable per-feature standardization with train-split stats.')
    parser.add_argument('--data-dir', default=PROCESSED_ROOT,
                        help='Root containing <dataset>/train.npz and test.npz.')
    parser.add_argument('--model-root', default=MODEL_ROOT, help='Root for checkpoints/configs.')
    parser.add_argument('--log-root', default=LOG_ROOT, help='Root for TensorBoard logs.')
    parser.add_argument(
        '--no_balanced_sampler',
        action='store_true',
        help='Disable claimed-token balanced sampling for digit mode training.',
    )
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    use_resample = not args.no_resample
    use_standardize = not args.no_standardize

    processed_dir, model_dir, log_dir = _resolve_dirs(args)
    train_path = os.path.join(processed_dir, 'train.npz')
    test_path = os.path.join(processed_dir, 'test.npz')

    if not os.path.exists(train_path):
        print('ERROR: Preprocessed data not found. Run preprocess.py first.')
        raise SystemExit(1)

    train_speakers, val_speakers = _speaker_split(train_path)
    if not val_speakers:
        print('WARNING: no speakers array in npz; best checkpoint selected on TEST (legacy behavior)')

    feature_stats = None
    if use_standardize:
        stats_path = os.path.join(processed_dir, 'feature_stats.json')
        if os.path.exists(stats_path):
            feature_stats = load_feature_stats(stats_path)
            print(f'Loaded feature stats from {stats_path}')
        else:
            print('Computing train-split feature stats...')
            raw_stats = compute_split_stats(
                train_path, dataset=args.dataset,
                resample=use_resample, seg_len=args.seg_len,
                speaker_filter=train_speakers,
            )
            with np.load(train_path, allow_pickle=True) as data:
                feature_names = (
                    [str(n) for n in data['feature_names'].tolist()]
                    if 'feature_names' in data.files
                    else [f'f{i}' for i in range(len(raw_stats['mean']))]
                )
            feature_stats = raw_stats
            save_feature_stats(stats_path, raw_stats, feature_names)
            print(f'Saved feature stats to {stats_path}')

    print(f'\nDataset: {args.dataset}')
    print(f'Mode: {args.mode}')
    print(f'Encoder: {args.encoder}')
    print(f'Resample: {use_resample} (seg_len={args.seg_len}) | Standardize: {use_standardize}')
    print('Loading data...')

    common = dict(dataset=args.dataset, resample=use_resample,
                  seg_len=args.seg_len, feature_stats=feature_stats)

    if args.mode == 'digit':
        train_ds = LipVerificationDataset(train_path, speaker_filter=train_speakers, **common)
        val_ds = (
            LipVerificationDataset(train_path, token_to_idx=train_ds.token_to_idx,
                                   seed=99, speaker_filter=val_speakers, **common)
            if val_speakers else None
        )
        test_ds = LipVerificationDataset(test_path, token_to_idx=train_ds.token_to_idx,
                                         seed=99, **common)
    elif args.mode == 'sequence':
        train_ds = SequenceVerificationDataset(train_path, speaker_filter=train_speakers, **common)
        val_ds = (
            SequenceVerificationDataset(train_path, token_to_idx=train_ds.token_to_idx,
                                        seed=99, speaker_filter=val_speakers, **common)
            if val_speakers else None
        )
        test_ds = SequenceVerificationDataset(test_path, token_to_idx=train_ds.token_to_idx,
                                              seed=99, **common)
    else:
        train_ds = LipTranscriptionDataset(train_path, speaker_filter=train_speakers, **common)
        val_ds = (
            LipTranscriptionDataset(train_path, token_to_idx=train_ds.token_to_idx,
                                    speaker_filter=val_speakers, **common)
            if val_speakers else None
        )
        test_ds = LipTranscriptionDataset(test_path, token_to_idx=train_ds.token_to_idx, **common)

    n_features = train_ds.n_features
    n_classes = train_ds.vocab_size

    if args.mode == 'seq2seq':
        seq_len = max(_max_sequence_length(train_ds), _max_sequence_length(test_ds))
        PAD_IDX = n_classes
        BOS_IDX = n_classes + 1
        EOS_IDX = n_classes + 2
        SEQ2SEQ_VOCAB_SIZE = n_classes + 3
        model_kwargs = {
            'vocab_size': SEQ2SEQ_VOCAB_SIZE, 'pad_idx': PAD_IDX, 'n_features': n_features,
            'seg_embed_dim': 48, 'n_heads': 4, 'n_encoder_layers': 1, 'n_decoder_layers': 1,
            'ff_dim': 128, 'dropout': 0.1, 'max_src_len': seq_len, 'max_tgt_len': seq_len + 1,
            'hidden_dim': 64, 'encoder_type': args.encoder,
        }
    else:
        model_kwargs = {
            'n_classes': n_classes, 'embed_dim': EMBED_DIM, 'n_features': n_features,
            'hidden_dim': HIDDEN_DIM, 'encoder_type': args.encoder,
        }

    config = {
        'config_version': 1,
        'dataset': args.dataset,
        'mode': args.mode,
        'encoder_type': args.encoder,
        'n_features': n_features,
        'feature_names': train_ds.feature_names,
        'seg_len': args.seg_len,
        'resample': use_resample,
        'standardized': feature_stats is not None,
        'vocab_size': n_classes,
        'model': model_kwargs,
    }
    model = ckpt.build_model_from_config(config, DEVICE)

    if args.mode != 'seq2seq':
        n_classes = train_ds.vocab_size
        print(f'Vocabulary size: {n_classes}')
    else:
        print(f'Vocabulary size: {n_classes} tokens (+3 special tokens)')

    vocab_path = os.path.join(model_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(train_ds.token_to_idx, f, indent=2)
    print(f'Vocabulary saved to {vocab_path}')

    config_path = ckpt.save_model_config(model_dir, config, filename=f'config_{args.mode}.json')
    print(f'Model config saved to {config_path}')

    collate_fn = None
    if args.mode == 'sequence':
        collate_fn = sequence_collate_fn
    elif args.mode == 'seq2seq':
        collate_fn = partial(transcription_collate_fn, pad_idx=PAD_IDX, eos_idx=EOS_IDX)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False,
                         collate_fn=collate_fn)
    train_sampler = None
    train_shuffle = True
    if args.mode == 'digit' and not args.no_balanced_sampler:
        train_sampler = _build_digit_balanced_sampler(train_ds)
        train_shuffle = False
        print('Using claimed-token balanced sampler for digit mode training')

    train_loader = DataLoader(train_ds, shuffle=train_shuffle, sampler=train_sampler, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None
    )
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else 0} samples, '
          f'Test: {len(test_ds)} samples')
    print(f'Feature dim: {n_features}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = (
        nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        if args.mode == 'seq2seq'
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f'{args.dataset}_{args.mode}_lr{args.lr}_bs{args.batch_size}_{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f'TensorBoard logs: {log_dir}/{run_name}')

    best_metric = float('-inf')
    results_log = []
    model_suffix = 'seq2seq' if args.mode == 'seq2seq' else f'{args.mode}_verifier'
    model_save_path = os.path.join(model_dir, f'best_{model_suffix}.pt')

    selection_loader = val_loader if val_loader is not None else test_loader
    selection_name = 'val' if val_loader is not None else 'test'
    print(f'Best-checkpoint selection on: {selection_name}')

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
                evaluate_seq2seq(model, selection_loader, DEVICE)
                if args.mode == 'seq2seq'
                else evaluate(model, selection_loader, DEVICE)
            )
            results_log.append({'epoch': epoch, 'loss': loss, **metrics})

            if args.mode == 'seq2seq':
                writer.add_scalar('eval/token_acc', metrics['token_acc'], epoch)
                writer.add_scalar('eval/exact_match_acc', metrics['exact_match_acc'], epoch)
                print(
                    f"Epoch {epoch:3d} | loss={loss:.4f} | "
                    f"{selection_name} TokenAcc={metrics['token_acc']:.4f} | "
                    f"ExactMatch={metrics['exact_match_acc']:.4f}"
                )
                track_metric = metrics['exact_match_acc']
            else:
                writer.add_scalar('eval/auc', metrics['auc'], epoch)
                writer.add_scalar('eval/eer', metrics['eer'], epoch)
                writer.add_scalar('eval/acc_at_eer', metrics['acc_at_eer'], epoch)
                writer.add_scalar('eval/acc_at_05', metrics['acc_at_05'], epoch)
                print(
                    f"Epoch {epoch:3d} | loss={loss:.4f} | {selection_name} AUC={metrics['auc']:.4f} | "
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
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))
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
        'dataset': args.dataset,
        'mode': args.mode,
        'selection': selection_name,
        'best_selection_metric': float(best_metric),
        'hyperparams': {
            'encoder_type': args.encoder,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'n_classes': n_classes,
            'seg_len': args.seg_len,
            'resample': use_resample,
            'standardized': feature_stats is not None,
        },
        'final_metrics': {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in final_metrics.items()
        },
        'training_log': results_log,
    }
    results_suffix = 'seq2seq' if args.mode == 'seq2seq' else args.mode
    results_path = os.path.join(model_dir, f'results_{results_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    print(f'\nResults saved to {results_path}')
