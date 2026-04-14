# Lip-Text Verification Pipeline

## Overview

This project verifies whether a lip movement video matches a claimed text (digit sequence). It extracts time series features from lip landmarks and trains a neural verification model.

## Pipeline

```
Video (.mp4) + Annotation (.lab)
        │
        ▼
  preprocess.py          Extract 5D lip features, segment per digit, split train/test
        │
        ▼
  processed_data/        train.npz, test.npz, metadata.json
        │
        ▼
  train.py               Train & evaluate verification model
        │
        ▼
  models/                best_model.pt, results.json
```

## Data Format

Each video contains a person reading **8 Chinese digits**. The `.lab` annotation file provides:
- Line 1: digit sequence (e.g., `3 5 3 9 6 7 8 7`)
- Line 2: time ranges in seconds (e.g., `0.54-0.83 0.84-1.03 ...`)

## Features (5D per frame)

| # | Feature | Formula | What it captures |
|---|---------|---------|-----------------|
| 1 | Vertical Aperture | $\|\mathbf{p}\_{13} - \mathbf{p}\_{14}\| / d\_{\text{eye}}$ | Mouth opening height (inner lip) |
| 2 | Horizontal Spread | $\|\mathbf{p}\_{61} - \mathbf{p}\_{291}\| / d\_{\text{eye}}$ | Lip width |
| 3 | Inner Lip Area | $A\_{\text{inner}} / d\_{\text{eye}}^2$ | Oral opening area |
| 4 | Compactness | $4\pi A\_{\text{outer}} / P\_{\text{outer}}^2$ | Round vs elongated shape |
| 5 | Lip Speed | $\sqrt{\dot{v}^2 + \dot{h}^2}$ | Overall movement speed |

All spatial features normalized by inter-ocular distance for head-size invariance.

## Preprocessing (`preprocess.py`)

1. **Landmark extraction**: MediaPipe Face Mesh → 478 face landmarks per frame
2. **Feature computation**: 5 lip features per frame (see above)
3. **Segmentation**: Use `.lab` timing to cut per-digit segments from feature sequence
4. **Speaker-level split**: ~80% train / ~20% test by speaker (no speaker overlap ensures the model generalizes across people, not memorizes faces)

Output:
- `processed_data/train.npz` — training samples
- `processed_data/test.npz` — test samples
- `processed_data/metadata.json` — dataset statistics

Each `.npz` contains per-video:
- `full_features`: (T_video, 5) — full video feature sequence
- `digit_segments`: list of 8 arrays, each (T_digit, 5) — per-digit segments
- `digit_sequences`: the 8-digit label
- `speakers`, `video_ids`, `fps`

## Model Architecture (`train.py`)

### Verification Task

Given: lip feature time series + claimed digit/sequence → output: match (1) or mismatch (0)

Two modes:

### 1. Per-Digit Verification (`--mode digit`)

```
Lip Segment (T, 5)  →  [Conv1D → BN → ReLU] × 2  →  [Bi-GRU]  →  Masked Mean Pool  →  lip_emb (64D)
Claimed Digit       →  [Embedding(10, 64)]                                            →  digit_emb (64D)
                                                                                            │
                                                                    [lip_emb ∥ digit_emb]  →  [FC → ReLU → Dropout → FC]  →  match logit
```

### 2. Full-Sequence Verification (`--mode sequence`) ← recommended

```
8 Lip Segments  →  [Shared LipEncoder] × 8  →  8 lip embeddings (64D each)
8 Claimed Digits →  [Embedding] × 8         →  8 digit embeddings (64D each)
                                                       │
                    Per-digit comparison  →  8 match scores  →  [FC aggregator]  →  sequence match logit
```

**Why this architecture:**
- **1D-CNN** captures local temporal patterns (consonant bursts, rapid transitions)
- **Bidirectional GRU** models sequential dependencies (how the mouth opens *then* closes)
- **Masked mean pooling** handles variable-length segments without padding artifacts
- **Per-digit comparison + aggregation** lets the model detect *which* digit doesn't match

### Training Details

- **Loss**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam with weight decay 1e-4
- **Scheduler**: Cosine annealing
- **Negative sampling**: For each positive (matching) pair, generate 1 negative pair where:
  - 50% chance: shuffle the digit sequence 
  - 50% chance: replace 2-4 random positions with wrong digits
- **Evaluation metrics**: ROC-AUC, Equal Error Rate (EER), accuracy at EER threshold

### Why Speaker-Level Split?

Splitting by speaker (not by video) ensures:
- Test speakers are **never seen** during training
- The model must generalize to new faces/lip shapes
- Prevents the model from memorizing speaker-specific patterns
- More realistic evaluation of deployment performance

## Usage

```bash
# 1. Preprocess dataset
python preprocess.py

# 2. Train sequence-level verifier (recommended)
python train.py --mode sequence --epochs 50

# 3. Train digit-level verifier
python train.py --mode digit --epochs 50
```

## File Structure

```
├── preprocess.py              # Dataset preprocessing
├── train.py                   # Model training & evaluation
├── digit_visual_signal.py     # Single-sample feature visualization
├── compare_samples.py         # Multi-sample comparison
├── FEATURES.md                # Feature documentation
├── PIPELINE.md                # This file
├── processed_data/
│   ├── train.npz
│   ├── test.npz
│   └── metadata.json
├── models/
│   ├── best_sequence_verifier.pt
│   └── results_sequence.json
└── data/
    ├── face_landmarker.task
    └── lipdata0405-filter/
        ├── 1003/
        │   ├── *.mp4, *.lab, *.wav, *.txt
        ...
```
