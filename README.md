# ts-lip2text

Lip-based digit verification and transcription from time-series facial landmarks.
The project supports:

- sequence-level verification (`--mode sequence`), this being influenced by highest probability in per digit/word probability. 
- per-digit (per word) verification (`--mode digit`)


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

```

#### Data pipeline flags (default on)

- Segments are resampled to a fixed length (`--seg-len`, default 16) after
  converting `lip_speed` to per-second units — makes trajectories comparable
  across videos with different fps.
- Features are standardized with train-split per-feature stats stored in
  `processed_data/<dataset>/feature_stats.json` (created on first run).
- Escape hatches reproduce the legacy pipeline: `--no-resample --no-standardize`.
- Best checkpoint is selected on a speaker-disjoint validation split (every
  10th sorted train speaker); test is evaluated once at the end.
- Every checkpoint dir also stores `vocab.json` and `config_<mode>.json`;
  `test.py`/`inference.py` rebuild the model from the config.
- Checkpoints live in `models/<dataset>/<encoder_type>/`.

Training outputs:

- checkpoints in `models/`
- metrics json in `models/results_<mode>.json`
- TensorBoard logs in `runs/`

Transformer-encoder runs are written under `models/transformer_encoder/` and `runs/transformer_encoder/` 

```bash
tensorboard --logdir runs/
```

### 3) Evaluate

`test.py` currently supports verification modes (`digit`, `sequence`):

```bash
python test.py --dataset digit --mode sequence
python test.py --dataset grid --mode digit
python test.py --dataset digit --mode digit --save
```

`--save` writes `models/test_results_<mode>.json`.

For `seq2seq`, validation is run inside `train.py` (`token_acc`, `exact_match_acc`), and you can also inspect predictions using `inference.py`.

### 4) Inference

```bash
python inference.py --dataset grid --video path/to/video.mp4 --lab path/to/file.lab [file.align for grid] --mode digit

```


## Device Support

The code auto-selects device in this order:

1. CUDA
2. Apple MPS
3. CPU
