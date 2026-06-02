# ts-lip2text

Lip-based digit verification and transcription from time-series facial landmarks.
The project supports:

- sequence-level verification (`--mode sequence`)
- per-digit verification (`--mode digit`)
- seq2seq transcription (`--mode seq2seq`)

## Current Pipeline

```text
Video (.mp4) + Lab Annotation (.lab) + Audio (.wav, optional)
        |
        v
preprocess.py
  - extract MediaPipe landmarks
  - compute 8D frame-level features
  - segment by .lab alignments
  - split by speaker (no speaker overlap)
        |
        v
processed_data/
  - train.npz
  - test.npz
  - metadata.json
        |
        v
train.py
  - digit / sequence / seq2seq training
        |
        v
models/
  - best_digit_verifier.pt
  - best_sequence_verifier.pt
  - best_seq2seq.pt
  - results_digit.json
  - results_sequence.json
  - results_seq2seq.json
        |
        v
test.py / inference.py
```

## Features (Current)

The model now uses **8 features per frame**:

1. inner vertical aperture
2. outer vertical aperture
3. horizontal spread
4. inner lip area
5. outer lip area
6. compactness
7. lip speed
8. RMS audio energy

Spatial terms are normalized by inter-ocular distance. Detailed formulas are in [FEATURES.md](FEATURES.md).

## Dataset Assumptions

- Vocabulary: `0-9` and `!` (`!` is alternate pronunciation for digit 1)
- Each utterance is expected to contain **8 digits**
- `.lab` format:
  - line 1: token sequence (`3 5 3 9 6 7 8 7`)
  - line 2: per-digit time spans (`0.54-0.83 0.84-1.03 ...`)

Expected raw data layout:

```text
data/
  face_landmarker.task
  lipdata-digit/
    subset_01/
      <speaker_id>/
        video/*.mp4
        lab/*.lab
        audio/*.wav
    subset_02/
    ...
```

## Installation

Python 3.10+ is recommended.

```bash
pip install numpy torch torchvision torchaudio opencv-python mediapipe librosa scipy scikit-learn tqdm tensorboard
```

## Usage

### 1) Preprocess

```bash
python preprocess.py
```

This generates:

- `processed_data/train.npz`
- `processed_data/test.npz`
- `processed_data/metadata.json`

Notes:

- speaker-disjoint split (`TEST_SPEAKER_RATIO = 0.1`)
- failed samples are reported in preprocessing logs

### 2) Train

```bash
# Sequence-level verification (default)
python train.py --mode sequence --epochs 50

# Digit-level verification
python train.py --mode digit --epochs 50

# Seq2seq transcription
python train.py --mode seq2seq --epochs 50

# Custom hyperparameters
python train.py --mode sequence --epochs 100 --lr 3e-4 --batch_size 32
```

Training outputs:

- checkpoints in `models/`
- metrics json in `models/results_<mode>.json`
- TensorBoard logs in `runs/`

```bash
tensorboard --logdir runs/
```

### 3) Evaluate

`test.py` currently supports verification modes (`digit`, `sequence`):

```bash
python test.py --mode sequence
python test.py --mode digit
python test.py --mode sequence --save
```

`--save` writes `models/test_results_<mode>.json`.

For `seq2seq`, validation is run inside `train.py` (`token_acc`, `exact_match_acc`), and you can also inspect predictions using `inference.py`.

### 4) Inference

```bash
# Sequence verification using exact .lab boundaries
python inference.py --video path/to/video.mp4 --lab path/to/file.lab --mode sequence

# Sequence verification with claimed digits and auto-segmentation
python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6" --mode sequence

# Per-digit verification
python inference.py --video path/to/video.mp4 --lab path/to/file.lab --mode digit

# Seq2seq transcription from .lab segmentation
python inference.py --video path/to/video.mp4 --lab path/to/file.lab --mode seq2seq

# Seq2seq transcription with auto segmentation length
python inference.py --video path/to/video.mp4 --mode seq2seq --n_digits 8
```

Important CLI constraints:

- verification modes (`digit`, `sequence`): provide exactly one of `--lab` or `--digits`
- seq2seq mode: provide `--lab`, or provide `--n_digits` for auto-segmentation

## Core Files

```text
preprocess.py   # data preprocessing and split
dataset.py      # datasets for digit/sequence/seq2seq modes
model.py        # LipEncoder, DigitVerifier, SequenceVerifier, TinyLipSeq2Seq
train.py        # training entry point for all modes
test.py         # evaluation for verification modes
inference.py    # single-video inference / transcription
FEATURES.md     # feature definitions
PIPELINE.md     # architecture and pipeline notes
```

## Device Support

The code auto-selects device in this order:

1. CUDA
2. Apple MPS
3. CPU

## Troubleshooting

- If preprocessing or inference fails immediately, verify `data/face_landmarker.task` exists.
- If `processed_data/train.npz` or `processed_data/test.npz` is missing, run `python preprocess.py` first.
- If a checkpoint is missing, run the corresponding `train.py --mode ...` command first.
