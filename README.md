# ts-lip2text

Lip-text verification using time series analysis. Given a lip movement video and a claimed digit sequence, the system extracts temporal lip features and verifies whether the lip movements match the claimed text.

## Pipeline

```
Video (.mp4) + Annotation (.lab)
        │
        ▼
  preprocess.py          Extract 5D lip features, segment per digit, speaker-level split
        │
        ▼
  processed_data/        train.npz, test.npz, metadata.json
        │
        ▼
  train.py               Train verification model (CNN + BiGRU)
        │
        ▼
  models/                best_{mode}_verifier.pt, results_{mode}.json
        │
        ▼
  test.py / inference.py Evaluate on test set / run on new videos
```

## Features

5D multivariate time series extracted per frame using MediaPipe Face Mesh (478 landmarks):

| # | Feature | What it captures |
|---|---------|-----------------|
| 1 | Vertical Aperture | Mouth opening height (inner lip) |
| 2 | Horizontal Spread | Lip width |
| 3 | Inner Lip Area | Oral opening area |
| 4 | Compactness | Round vs elongated lip shape |
| 5 | Lip Speed | Overall movement velocity |

All spatial features are normalized by inter-ocular distance for head-size invariance. See [FEATURES.md](FEATURES.md) for detailed formulas.

## Data Format

Each video contains a person reading **8 Chinese digits** (vocabulary: 0-9 and `!` for alternate pronunciation of 1). The `.lab` annotation provides:
- Line 1: space-separated digit sequence (e.g., `3 5 3 9 6 7 8 7`)
- Line 2: space-separated time ranges in seconds (e.g., `0.54-0.83 0.84-1.03 ...`)

## Model Architecture

Two verification modes, plus one transcription mode:

- **Sequence-level** (`--mode sequence`, recommended): Encodes all 8 digit segments with a shared encoder, compares each against its claimed digit embedding, aggregates into a single match/mismatch decision.
- **Digit-level** (`--mode digit`): Verifies each lip segment against a single claimed digit independently.
- **Seq2seq transcription** (`--mode seq2seq`): Tiny Transformer encoder-decoder that predicts the digit string directly from the segment sequence.

Architecture: `Conv1D x2 → BatchNorm → ReLU → Bidirectional GRU → Masked Mean Pooling → FC`. See [PIPELINE.md](PIPELINE.md) for full details.

## Usage

### 1. Preprocess

```bash
python preprocess.py
```

Extracts features from all videos, segments per digit, and splits into train/test by speaker (~80/20, no speaker overlap).

### 2. Train

```bash
# Sequence-level verifier (recommended)
python train.py --mode sequence --epochs 50

# Digit-level verifier
python train.py --mode digit --epochs 50

# Seq2seq transcription model
python train.py --mode seq2seq --epochs 50

# Custom hyperparameters
python train.py --mode sequence --epochs 100 --lr 5e-4 --batch_size 32
```

Training logs are saved to `runs/` for TensorBoard:

```bash
tensorboard --logdir runs/
```

### 3. Test

Evaluate the trained model on the held-out test set:

```bash
# Sequence-level
python test.py --mode sequence

# Digit-level (includes per-digit precision/recall/F1 breakdown)
python test.py --mode digit

# Save detailed results to JSON
python test.py --mode sequence --save
```

### 4. Inference

Run verification on a new video:

```bash
# With .lab annotation (precise timing-based segmentation)
python inference.py --video path/to/video.mp4 --lab path/to/annotation.lab

# With raw digit string (evenly splits video across digits)
python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6"

# Per-digit mode
python inference.py --video path/to/video.mp4 --lab path/to/file.lab --mode digit

# Seq2seq transcription
python inference.py --video path/to/video.mp4 --mode seq2seq --lab path/to/file.lab

# Seq2seq transcription with auto-segmentation length
python inference.py --video path/to/video.mp4 --mode seq2seq --n_digits 8
```

## File Structure

```
├── preprocess.py              # Full dataset preprocessing
├── train.py                   # Model training with TensorBoard logging
├── test.py                    # Model evaluation on test set
├── inference.py               # Single-video inference
├── model.py                   # Model definitions (LipEncoder, DigitVerifier, SequenceVerifier)
├── digit_visual_signal.py     # Single-sample feature visualization
├── compare_samples.py         # Multi-speaker feature comparison
├── FEATURES.md                # Feature documentation
├── PIPELINE.md                # Pipeline documentation
├── data/
│   ├── face_landmarker.task   # MediaPipe face landmark model
│   └── lipdata0405-filter/    # Raw dataset (videos + annotations)
├── processed_data/            # Preprocessed .npz files (generated)
├── models/                    # Trained model checkpoints (generated)
└── runs/                      # TensorBoard logs (generated)
```

## Requirements

- Python 3.10+
- PyTorch
- MediaPipe
- OpenCV
- scikit-learn
- tqdm
- tensorboard
