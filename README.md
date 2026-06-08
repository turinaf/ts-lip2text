# ts-lip2text

Lip-based digit verification and transcription from time-series facial landmarks.
The project supports:

- sequence-level verification (`--mode sequence`)
- per-digit (per word) verification (`--mode digit`)


## Features (Current)

The core visual pipeline uses **7 features per frame**:

1. inner vertical aperture
2. outer vertical aperture
3. horizontal spread
4. inner lip area
5. outer lip area
6. compactness
7. lip speed

Optional:

8. RMS audio energy (enabled with `--use-audio-rms`)

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

GRID layout (supported via `--dataset grid`):

```text
../liptev/data/grid/
  s10_processed/
    align/*.align
    audio/*.wav
    video/*.mp4          # ignored for feature extraction

../data/
  s10_processed/
    *.mpg                # used as uncropped input video
```

For GRID, alignment/audio are read from `../liptev/data/grid/sXX_processed`, and
the video is resolved from `../data/sXX_processed/<same_basename>.mpg`.


## Usage

### 1) Preprocess

```bash
python preprocess.py

# Include audio RMS as the 8th feature
python preprocess.py --use-audio-rms


# GRID (all speakers found under ../liptev/data/grid)
python preprocess.py --dataset grid

# GRID with resume cache reset (start fresh)
python preprocess.py --dataset grid --reset-resume

# GRID without resume cache
python preprocess.py --dataset grid --no-resume

# GRID + audio RMS
python preprocess.py --dataset grid --use-audio-rms
```

This generates:

- `processed_data/grid/train.npz`
- `processed_data/grid/test.npz`
- `processed_data/grid/metadata.json`

Notes:

- speaker-disjoint split (`TEST_SPEAKER_RATIO = 0.1`)
- failed samples are reported in preprocessing logs
- GRID preprocessing now resumes automatically using per-video cache under `processed_data/grid/_resume_grid`
- if preprocessing arguments or speaker split change, resume cache is rejected to avoid mixing incompatible runs

### 2) Train

```bash
# Sequence-level verification (default)
python train.py --mode sequence --epochs 50

# Use the lightweight transformer encoder explicitly
python train.py --mode sequence --encoder transformer --epochs 50

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

Transformer-encoder runs are written under `models/transformer_encoder/` and `runs/transformer_encoder/` so they do not overwrite the older BiGRU results.

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

# Add RMS from the video audio track at inference time
python inference.py --video path/to/video.mp4 --digits "1 3 5 7 9 2 4 6" --mode sequence --use-audio-rms
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
