# ts-lip2text

Lip-based digit verification and transcription from time-series facial landmarks.
The project supports:

- sequence-level verification (`--mode sequence`), this being influenced by highest probability in per digit/word probability.
- per-digit (per word) verification (`--mode digit`)
- segment-level transcription (`--mode seq2seq`)
- **direct lip reading** (`--mode lipread`): decodes the whole-utterance lip-motion
  time series straight into text — no negative sampling, no per-digit segmentation.


## Features

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

For GRID, alignment/audio are read from cropped and processed data for end2end `../liptev/data/grid/sXX_processed`, and
The original un video is resolved from `../data/sXX_processed/<same_basename>.mpg`.


## Usage

### 1) Preprocess
For end to end model check this [README](end2end/README.md)
```bash

python preprocess.py --dataset [digit or grid]

# GRID with resume cache reset (start fresh)
python preprocess.py --dataset grid --reset-resume

# GRID + audio RMS
python preprocess.py --dataset grid --use-audio-rms
```

This generates:

- `processed_data/grid/train.npz`
- `processed_data/grid/test.npz`
- `processed_data/grid/metadata.json`


### 2) Train

```bash

python train.py -dataset [digit or grid] --mode digit --encoder [transformer or bigru] --epochs 100
python train.py -dataset grid --mode lipread --epochs 100

```

`--mode` selects the task:

- `digit` / `sequence`: verification (requires negative-sample generation).
- `seq2seq`: segment-level transcription (lip reading over pre-segmented tokens).
- `lipread`: **frame-level direct lip reading**. The full-utterance lip-motion
  time series `(T, F)` is fed through a 1D-conv + transformer encoder and decoded
  autoregressively into tokens. No negative samples, no segmentation.

Training outputs:

- checkpoints in `models/`
- metrics json in `models/results_<mode>.json`
- TensorBoard logs in `runs/`
- for `lipread`, a `lipread_config.json` is saved next to the checkpoint so
  `test.py` / `inference.py` can reconstruct the model.

Transformer-encoder runs are written under `models/transformer_encoder/` and `runs/transformer_encoder/` 

```bash
tensorboard --logdir runs/
```

### 3) Evaluate

`test.py` supports verification modes (`digit`, `sequence`) and the `lipread` mode:

```bash
python test.py --mode sequence
python test.py --mode digit
python test.py --mode sequence --save
python test.py --mode lipread --dataset grid
```

`--save` writes `models/test_results_<mode>.json`.

For `seq2seq` and `lipread`, validation (`token_acc`, `exact_match_acc`) is run
inside `train.py`, and you can also inspect predictions using `inference.py`.

### 4) Inference

```bash
python inference.py --dataset grid --video path/to/video.mp4 --lab path/to/file.lab [file.align for grid] --mode digit
python inference.py --dataset grid --video path/to/video.mp4 --mode lipread
```

`--mode lipread` decodes the entire video into text directly (no `--digits`,
`--lab`, or `--n_digits` required).


## Device Support

The code auto-selects device in this order:

1. CUDA
2. Apple MPS
3. CPU
