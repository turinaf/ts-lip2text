# Design: Pipeline Correctness + Methodology Fixes

**Date:** 2026-08-20
**Branch:** `methodology-fixes` (off `main`)
**Scope approved by:** turinaf — "Bugs + core methodology", code + unit/smoke tests only (no local retraining; owner retrains), new behavior **on by default** with escape-hatch flags.
**Out of scope:** end2end/ directory (different, non-time-series method); metric-learning objective, hard negatives, augmentation, GRID alignment-heuristic rework. The `!` token sparsity is a **label-generation issue the owner is fixing separately** — do not merge `!` into `1` and do not special-case it.

## Problem

An inspection of the time-series verification pipeline (`preprocess.py` → `dataset.py` → `model.py` → `train.py` → `test.py` → `inference.py`) found outright bugs and methodology problems that affect both correctness and validity of reported results:

- **Bugs:** seq2seq inference crashes on checkpoints trained with `max_src_len != 12` (hardcoded `inference.py:619-620` vs `train.py:413-414`); transformer checkpoints for digit and grid overwrite each other in `models/transformer_encoder/` (`train.py:_output_dirs`); `test.py` re-implements drifted dataset copies (stale `processed_data/test.npz` path, hardcoded `N_FEATURES=8`/`N_CLASSES=11`, no fixed-length filter, no GRID support); stale `FEATURE_NAME_TO_INDEX` in `dataset.py:13-19`; dead unreachable code in `inference.py:322-329`; unused `SequenceVerifier.seq_len` param; stale `n_features=5` default in `model.py`.
- **Methodology:** model selection happens on the test set (no val split); no feature normalization (verified ~50× scale disparity across features); `lip_speed` computed per-frame while video fps varies 23.8–35.0; ~22% of digit train videos silently dropped by the fixed-length filter without any metadata trace.

## Approach

Transforms live in the **dataset layer**, not re-baked into npz or embedded in the model:

- npz files remain the canonical raw-feature store; no MediaPipe re-extraction.
- A new `transforms.py` module holds pure functions shared by train/test/inference.
- Per-feature stats are computed from the train split only and persisted as JSON.

Rejected alternatives: re-baking at preprocess time (forces multi-hour GRID re-extraction, loses raw data for ablations); model-internal normalization layers (inference scripts must wrap models, messier checkpoint compat).

## Component Design

### 1. Fixed-length resampling (`transforms.py`)

- `resample_segment(seg, target_len)` — per-feature linear interpolation (`np.interp`) onto a uniform time grid; segments shorter than 2 frames are tiled to length 2 first (observed minimum is 2).
- `SEG_LEN = 16` default (CLI `--seg-len`). Justification from data: segment lengths 2–25 frames, mean 6, p99 13.
- Applied **before** standardization. Masks become all-ones; mask tensors stay in the API for compatibility.
- `--no-resample` falls back to the existing pad-to-`MAX_SEQ_LEN=30` path unchanged.

### 2. fps-corrected lip speed

- `lip_speed` (the only derivative feature, index 6) is multiplied by the video's fps (already stored per-sample in the npz) before resampling, making it per-second. No other feature depends on fps.

### 3. Feature standardization

- Per-feature mean/std over **resampled train-split segments only**, saved to `processed_data/<dataset>/feature_stats.json` (includes feature order).
- `train.py` computes and saves the file when missing (and refuses silently-wrong setups: test/inference require the file when standardization is on).
- `--no-standardize` escape hatch.
- Dataset-level (not per-segment instance) normalization: 6-frame segments make per-segment z-scoring amplify noise on near-static features.

### 4. Validation split

- `train.py` splits train-npz speakers deterministically: sort unique speakers, every 10th → val. Speaker-disjoint, seed-free, reproducible.
- Periodic eval + best-checkpoint selection track **val AUC** (val `exact_match_acc` for seq2seq); the test set is evaluated once at the end.

### 5. Checkpoint configs + layout

- Every checkpoint gets a sibling `config.json`: `dataset, mode, encoder_type, n_features, feature_names, embed_dim, hidden_dim, seg_len, resample, standardized, vocab_size, max_src_len, max_tgt_len`.
- `test.py`/`inference.py` construct models from `config.json` (CLI args may override; legacy checkpoints without config fall back to today's hardcoded constants).
- New output layout `models/<dataset>/<encoder_type>/` eliminates the digit/grid collision. Existing checkpoints remain loadable via explicit `--model_path`.
- `test.py` rewritten to import the `dataset.py` classes (dedupe drifted copies), use dataset-scoped npz paths, and support GRID.

### 6. Straight bug fixes

- `FEATURE_NAME_TO_INDEX` derived from the npz `feature_names` dynamically.
- Remove dead block in `inference.py:322-329`; remove unused `SequenceVerifier.seq_len` (update callers); fix stale `n_features` default; `train.py` final eval loads checkpoint with `map_location`.
- `preprocess.py` writes sequence-length distribution and drop counts into `metadata.json` (digit fixed-length filtering loss becomes visible; the filter itself is unchanged — the underlying malformed-`.lab` loss is a preprocessing-data issue handled by the owner's label regeneration).

## Testing

- **Unit tests (new `tests/test_transforms.py` + updates to `tests/test_preflight.py`):**
  - resample: shapes and values for T=2, T=16, T=25; constant-series invariance; monotone time mapping.
  - lip_speed fps correction: doubling fps doubles the column.
  - standardization: stats from train split only; standardized mean≈0/std≈1 on train; stats JSON roundtrip.
  - config JSON save/load roundtrip; seq2seq model reconstructed from config matches checkpoint shapes (the pos-emb crash regression test).
  - dataset classes with resample+standardize on and off; digit fixed-length filter preserved; GRID dynamic vocab intact.
- **Smoke:** `preflight.py` for digit+grid × {digit, sequence, seq2seq}; 1-epoch `train.py` run on a synthetic tiny npz proving train → config → test wiring end-to-end.
- No local retraining on real data (owner retrains; GRID training happens on the server).

## Acceptance Criteria

1. `python test.py --dataset grid --mode digit` works using a config-bearing checkpoint; no hardcoded `N_FEATURES`/`N_CLASSES` in the eval path.
2. A seq2seq checkpoint trained at any `max_src_len` loads in `inference.py` without shape errors.
3. Digit and grid transformer runs no longer share output paths.
4. With defaults, segments fed to models are resampled to `SEG_LEN` and standardized with train-split stats; both behaviors revert cleanly via `--no-resample` / `--no-standardize`.
5. Best checkpoint is selected on val, not test; final test metrics computed once.
6. `metadata.json` records sequence-length distribution and drop counts.
7. All unit tests and preflight checks pass.
