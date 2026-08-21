"""Preflight checks before training.

Use this script to validate that a processed dataset is loadable, contains
finite features, and can run a single forward pass through the selected model.

Examples:
    conda activate torch && python preflight.py --dataset digit --mode sequence
    conda activate torch && python preflight.py --dataset grid --mode seq2seq
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
)
from model import DigitVerifier, SequenceVerifier, TinyLipSeq2Seq


PROCESSED_ROOT = 'processed_data'
MODEL_ROOT = 'models'
LOG_ROOT = 'runs'
MAX_SEQ_LEN = 30
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ2SEQ_SEG_EMBED_DIM = 48
SEQ2SEQ_HIDDEN_DIM = 64
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


@dataclass
class CheckResult:
    ok: bool
    message: str


def dataset_paths(dataset_name: str) -> tuple[str, str, str]:
    processed_dir = os.path.join(PROCESSED_ROOT, dataset_name)
    model_dir = os.path.join(MODEL_ROOT, dataset_name)
    log_dir = os.path.join(LOG_ROOT, dataset_name)
    return processed_dir, model_dir, log_dir


def max_sequence_length(dataset) -> int:
    return max(len(seq) for seq in dataset.digit_sequences)


def _assert_finite_array(name: str, arr: np.ndarray) -> None:
    array = np.asarray(arr)
    if array.dtype == object:
        for item in array:
            _assert_finite_array(name, item)
        return

    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains NaN or infinity')


def check_npz_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = {'digit_segments', 'digit_sequences', 'full_features', 'feature_names'}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f'{path} is missing keys: {sorted(missing)}')

    _assert_finite_array(f'{path}:full_features', data['full_features'])
    _assert_finite_array(f'{path}:digit_segments', data['digit_segments'])


def build_dataset(dataset_name: str, mode: str, npz_path: str, token_to_idx=None):
    if mode == 'digit':
        return LipVerificationDataset(
            npz_path,
            dataset=dataset_name,
            token_to_idx=token_to_idx,
            max_seq_len=MAX_SEQ_LEN,
        )
    if mode == 'sequence':
        return SequenceVerificationDataset(
            npz_path,
            dataset=dataset_name,
            token_to_idx=token_to_idx,
            max_seg_len=MAX_SEQ_LEN,
        )
    if mode == 'seq2seq':
        return LipTranscriptionDataset(
            npz_path,
            dataset=dataset_name,
            token_to_idx=token_to_idx,
            max_seg_len=MAX_SEQ_LEN,
        )
    raise ValueError(f'Unknown mode: {mode}')


def build_model(mode: str, vocab_size: int, n_features: int, seq_len: int):
    if mode == 'digit':
        return DigitVerifier(
            n_classes=vocab_size,
            embed_dim=EMBED_DIM,
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
        ).to(DEVICE)

    if mode == 'sequence':
        return SequenceVerifier(
            n_classes=vocab_size,
            embed_dim=EMBED_DIM,
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
        ).to(DEVICE)

    pad_idx = vocab_size
    return TinyLipSeq2Seq(
        vocab_size=vocab_size + 3,
        pad_idx=pad_idx,
        n_features=n_features,
        seg_embed_dim=SEQ2SEQ_SEG_EMBED_DIM,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ff_dim=128,
        dropout=0.1,
        max_src_len=seq_len,
        max_tgt_len=seq_len + 1,
        hidden_dim=SEQ2SEQ_HIDDEN_DIM,
    ).to(DEVICE)


def check_forward_pass(dataset_name: str, mode: str, train_path: str) -> None:
    train_ds = build_dataset(dataset_name, mode, train_path)
    test_ds = build_dataset(dataset_name, mode, train_path, token_to_idx=train_ds.token_to_idx)
    seq_len = max(max_sequence_length(train_ds), max_sequence_length(test_ds))
    model = build_model(mode, train_ds.vocab_size, train_ds.n_features, seq_len)
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        if mode == 'digit':
            feats, mask, claimed, _ = [b.to(DEVICE) for b in batch]
            logits = model(feats, mask, claimed)
        elif mode == 'sequence':
            segs, masks, digits, _, seq_mask = [b.to(DEVICE) for b in batch]
            logits = model(segs, masks, digits, seq_mask)
        else:
            segments, masks, src_pad, targets = batch
            segments = segments.to(DEVICE)
            masks = masks.to(DEVICE)
            src_pad = src_pad.to(DEVICE)
            targets = targets.to(DEVICE)
            tgt_in = torch.full_like(targets, train_ds.vocab_size)
            tgt_in[:, 0] = train_ds.vocab_size + 1
            tgt_in[:, 1:] = targets[:, :-1]
            logits = model(segments, masks, src_pad, tgt_in)

    if not torch.isfinite(logits).all():
        raise ValueError(f'{dataset_name}/{mode} model forward pass produced NaN or infinity')


def run_checks(dataset_name: str, mode: str) -> list[CheckResult]:
    processed_dir, model_dir, log_dir = dataset_paths(dataset_name)
    train_path = os.path.join(processed_dir, 'train.npz')
    test_path = os.path.join(processed_dir, 'test.npz')

    checks = [
        ('paths', processed_dir, model_dir, log_dir),
        ('train.npz', train_path),
        ('test.npz', test_path),
    ]

    results: list[CheckResult] = []
    for check in checks:
        try:
            if check[0] == 'paths':
                _, processed_dir, model_dir, log_dir = check
                for p in (processed_dir, model_dir, log_dir):
                    os.makedirs(p, exist_ok=True)
                results.append(CheckResult(True, f'Created/verified {processed_dir}, {model_dir}, {log_dir}'))
            else:
                check_npz_file(check[1])
                results.append(CheckResult(True, f'Validated {check[1]}'))
        except Exception as exc:
            results.append(CheckResult(False, f'{check[0]} failed: {exc}'))
            return results

    try:
        check_forward_pass(dataset_name, mode, train_path)
        results.append(CheckResult(True, f'Forward pass ok for {dataset_name}/{mode}'))
    except Exception as exc:
        results.append(CheckResult(False, f'Forward pass failed: {exc}'))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description='Preflight checks before training.')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit')
    parser.add_argument('--mode', choices=['digit', 'sequence', 'seq2seq'], default='sequence')
    args = parser.parse_args()

    results = run_checks(args.dataset, args.mode)
    failed = False
    for result in results:
        prefix = 'OK' if result.ok else 'FAIL'
        print(f'{prefix}: {result.message}')
        failed = failed or not result.ok
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())