# Pipeline Correctness + Methodology Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the drifted/broken train→test→inference wiring and land three methodology corrections (train-split feature standardization, fps-corrected fixed-length segment resampling, speaker-disjoint validation for model selection) with checkpoint-borne configs.

**Architecture:** A new pure-function `transforms.py` (resample / fps-correct / standardize / stats IO) is applied inside the `dataset.py` loading layer — npz files stay canonical raw features, no MediaPipe re-extraction. A new `checkpoint.py` writes and reads a `config.json` beside every checkpoint; `test.py`/`inference.py` reconstruct models from it instead of hardcoded constants. `train.py` selects the best checkpoint on a speaker-disjoint val split (every 10th sorted train speaker) and evaluates test once at the end.

**Tech Stack:** Python 3.14, PyTorch 2.9, NumPy 2.4, scikit-learn, pytest 9.

**Spec:** `docs/superpowers/specs/2026-08-20-pipeline-methodology-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- Work on branch `methodology-fixes` (created in Task 1), one commit per task.
- Every shell command runs inside the conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch` (from repo root `/Users/turi/localgit/ts-lip2text`).
- Tests: `python -m pytest tests/<file>.py -v`. Full suite: `python -m pytest tests/ -v`.
- New pipeline behavior is ON by default; `--no-resample` / `--no-standardize` must reproduce the old pad-to-30 raw-feature path (same shapes, same pre-standardization values).
- npz schema is read-only: datasets may *optionally* read `fps` and `speakers` arrays; nothing new is written into npz files. `feature_stats.json` is a new sibling file under `processed_data/<dataset>/`.
- Checkpoint layout changes to `models/<dataset>/<encoder_type>/` (legacy paths remain loadable via explicit `--model_path` and via fallback candidates).
- The `!` vocabulary token is NOT merged, dropped, or special-cased — its sparsity is a label-generation issue the owner is fixing separately (spec, out-of-scope).
- No retraining on real data. Validation = unit tests + preflight + 1-epoch smoke runs on synthetic npz.

## File Structure

| File | Responsibility |
|---|---|
| `transforms.py` (new) | Pure segment transforms: fps-correct `lip_speed`, fixed-length resample, standardize, feature-stats compute/save/load |
| `checkpoint.py` (new) | `config.json` save/load + `build_model_from_config` shared by train/test/inference |
| `dataset.py` (modify) | Transform integration in the 3 dataset classes, `speaker_filter`, fps loading, `get_pair_info`, dynamic feature-name index, `compute_split_stats`, collates move here |
| `model.py` (modify) | Remove unused `SequenceVerifier.seq_len`; fix stale `n_features=5` defaults |
| `train.py` (modify) | New flags, stats computation, val split + val-based selection, new output dirs, config/vocab saving, `--data-dir/--model-root/--log-root`, `map_location` fix |
| `test.py` (rewrite) | Imports dataset.py classes, config-driven model construction, `--dataset` + GRID support, stats/vocab loading |
| `inference.py` (modify) | Config-driven construction, transform application, legacy pos-emb shape inference, dead-code removal, new layout with legacy fallbacks |
| `preprocess.py` (modify) | Metadata: sequence-length distribution + failure counts (pure helper `_summarize_samples`) |
| `preflight.py` (modify) | Transform params in `build_dataset`, forward-pass check for both paths |
| `tests/test_transforms.py`, `tests/test_checkpoint.py`, `tests/test_datasets.py` (new), `tests/test_preflight.py`, `tests/test_preprocess_metadata.py` (new) | Tests |
| `README.md` (modify) | Document new flags, layout, feature_stats.json |

---

### Task 1: Branch + `transforms.py` core (fps-correct, resample)

**Files:**
- Create: `transforms.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: nothing (stdlib + numpy only).
- Produces: `LIP_SPEED_INDEX = 6`; `correct_lip_speed_fps(seg: np.ndarray, fps: float) -> np.ndarray`; `resample_segment(seg: np.ndarray, target_len: int) -> np.ndarray`. Later tasks rely on these exact names and float32 2-D `(T, F)` in/out contracts.

- [ ] **Step 1: Create branch**

```bash
cd /Users/turi/localgit/ts-lip2text
git checkout -b methodology-fixes
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_transforms.py`:

```python
import unittest

import numpy as np

from transforms import correct_lip_speed_fps, resample_segment


class CorrectLipSpeedFpsTests(unittest.TestCase):
    def test_multiplies_speed_column_by_fps(self):
        seg = np.array([[1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.02, 0.4]], dtype=np.float32)
        out = correct_lip_speed_fps(seg, fps=25.0)
        self.assertAlmostEqual(float(out[0, 6]), 0.5, places=6)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=6)

    def test_narrow_segment_is_unchanged(self):
        seg = np.array([[1.0, 2.0]], dtype=np.float32)
        out = correct_lip_speed_fps(seg, fps=50.0)
        np.testing.assert_array_equal(out, seg)

    def test_input_not_mutated(self):
        seg = np.ones((3, 8), dtype=np.float32)
        correct_lip_speed_fps(seg, fps=2.0)
        np.testing.assert_array_equal(seg, np.ones((3, 8), dtype=np.float32))


class ResampleSegmentTests(unittest.TestCase):
    def test_output_shape(self):
        seg = np.random.rand(6, 8).astype(np.float32)
        for target in (2, 6, 16, 25):
            with self.subTest(target=target):
                out = resample_segment(seg, target)
                self.assertEqual(out.shape, (target, 8))
                self.assertEqual(out.dtype, np.float32)

    def test_endpoints_preserved(self):
        seg = np.random.rand(9, 4).astype(np.float32)
        out = resample_segment(seg, 16)
        np.testing.assert_allclose(out[0], seg[0], atol=1e-6)
        np.testing.assert_allclose(out[-1], seg[-1], atol=1e-6)

    def test_constant_series_is_invariant(self):
        seg = np.full((5, 3), 2.5, dtype=np.float32)
        out = resample_segment(seg, 16)
        np.testing.assert_allclose(out, 2.5, atol=1e-6)

    def test_single_frame_is_tiled(self):
        seg = np.array([[1.0, 2.0]], dtype=np.float32)
        out = resample_segment(seg, 4)
        self.assertEqual(out.shape, (4, 2))
        np.testing.assert_array_equal(out[0], out[3])

    def test_empty_segment_raises(self):
        with self.assertRaises(ValueError):
            resample_segment(np.zeros((0, 8), dtype=np.float32), 16)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch && python -m pytest tests/test_transforms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transforms'`

- [ ] **Step 4: Write minimal implementation** — create `transforms.py`:

```python
"""Deterministic segment transforms shared by training, evaluation, and inference.

The npz files keep raw features; these pure functions are applied at load time:
fps-corrected lip speed -> fixed-length resampling -> standardization.
"""
import json
import os

import numpy as np

# Index of the lip_speed column in the canonical feature order
# (vert_aperture, outer_vert_aperture, horiz_spread, inner_area,
#  outer_area, compactness, lip_speed, [rms_energy]).
LIP_SPEED_INDEX = 6


def correct_lip_speed_fps(seg, fps):
    """Convert the per-frame lip_speed column to per-second units (copy)."""
    out = np.array(seg, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] <= LIP_SPEED_INDEX:
        return out
    out[:, LIP_SPEED_INDEX] *= float(fps)
    return out


def resample_segment(seg, target_len):
    """Linearly resample a (T, F) segment onto target_len uniform time steps."""
    seg = np.asarray(seg, dtype=np.float32)
    if seg.ndim != 2:
        raise ValueError(f'expected 2D segment, got shape {seg.shape}')
    if target_len < 1:
        raise ValueError('target_len must be >= 1')
    t = seg.shape[0]
    if t == 0:
        raise ValueError('cannot resample an empty segment')
    if t == 1:
        return np.repeat(seg, target_len, axis=0)
    if t == target_len:
        return seg.astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_len)
    resampled = np.stack(
        [np.interp(dst, src, seg[:, j]) for j in range(seg.shape[1])], axis=1
    )
    return resampled.astype(np.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_transforms.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add transforms.py tests/test_transforms.py
git commit -m "feat: add fps-corrected fixed-length segment resampling transforms"
```

---

### Task 2: `transforms.py` stats (standardize, compute, save/load)

**Files:**
- Modify: `transforms.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: Task 1 module.
- Produces: `compute_feature_stats(segments, n_features) -> dict` with keys `mean` (list[float]), `std` (list[float]), `n_frames` (int); `standardize_segment(seg, mean, std) -> np.ndarray` (mean/std are sequences; zero std clamped to 1e-6); `save_feature_stats(path, stats, feature_names) -> str` (file also stores `feature_names` and `n_frames`); `load_feature_stats(path) -> dict` with keys `feature_names`, `mean`, `std`, `n_frames` — raises `FileNotFoundError` if missing.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_transforms.py` (above the `if __name__` block, and add `import os, tempfile` plus new imports at top):

```python
import os
import tempfile

from transforms import (
    compute_feature_stats,
    load_feature_stats,
    resample_segment,  # existing
    save_feature_stats,
    standardize_segment,
)
```

(Keep the existing imports; merge into one `from transforms import (...)` list.)

```python
class FeatureStatsTests(unittest.TestCase):
    def test_stats_match_manual_computation(self):
        segs = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])]
        stats = compute_feature_stats(segs, n_features=2)
        np.testing.assert_allclose(stats['mean'], [3.0, 4.0], atol=1e-6)
        np.testing.assert_allclose(
            stats['std'], np.sqrt([8.0 / 3.0, 8.0 / 3.0]), atol=1e-5
        )
        self.assertEqual(stats['n_frames'], 3)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_feature_stats([], n_features=2)


class StandardizeTests(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        seg = np.random.RandomState(0).rand(10, 3) * np.array([10.0, 1.0, 0.01])
        stats = compute_feature_stats([seg], n_features=3)
        out = standardize_segment(seg, stats['mean'], stats['std'])
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-3)
        self.assertEqual(out.dtype, np.float32)

    def test_zero_std_is_clamped(self):
        seg = np.full((4, 2), 3.0, dtype=np.float32)
        out = standardize_segment(seg, mean=[3.0, 3.0], std=[0.0, 2.0])
        np.testing.assert_allclose(out, 0.0, atol=1e-5)


class FeatureStatsIOTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'feature_stats.json')
            stats = {'mean': [0.1, 0.2], 'std': [1.0, 2.0], 'n_frames': 12}
            saved = save_feature_stats(path, stats, feature_names=['a', 'b'])
            self.assertEqual(saved, path)
            loaded = load_feature_stats(path)
            self.assertEqual(loaded['feature_names'], ['a', 'b'])
            self.assertEqual(loaded['mean'], [0.1, 0.2])
            self.assertEqual(loaded['std'], [1.0, 2.0])
            self.assertEqual(loaded['n_frames'], 12)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_feature_stats('/nonexistent/feature_stats.json')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transforms.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_feature_stats'`

- [ ] **Step 3: Write minimal implementation** — append to `transforms.py`:

```python
def compute_feature_stats(segments, n_features):
    """Per-feature mean/std over all frames of an iterable of (T, F) segments."""
    acc = np.zeros(n_features, dtype=np.float64)
    acc_sq = np.zeros(n_features, dtype=np.float64)
    count = 0
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        acc += seg.sum(axis=0)
        acc_sq += (seg ** 2).sum(axis=0)
        count += seg.shape[0]
    if count == 0:
        raise ValueError('no frames available to compute feature statistics')
    mean = acc / count
    var = acc_sq / count - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return {
        'mean': mean.astype(np.float32).tolist(),
        'std': std.astype(np.float32).tolist(),
        'n_frames': int(count),
    }


def standardize_segment(seg, mean, std):
    """z-score a segment with precomputed per-feature stats (zero std clamped)."""
    seg = np.asarray(seg, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (seg - mean) / np.maximum(std, 1e-6)


def save_feature_stats(path, stats, feature_names):
    """Persist stats + feature order so eval/inference apply the same transform."""
    payload = {
        'feature_names': list(feature_names),
        'mean': stats['mean'],
        'std': stats['std'],
        'n_frames': stats['n_frames'],
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def load_feature_stats(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Feature stats not found: {path}. Run train.py (with standardization '
            'enabled) once so the file is created next to the npz data.'
        )
    with open(path) as f:
        stats = json.load(f)
    return {
        'feature_names': list(stats['feature_names']),
        'mean': list(stats['mean']),
        'std': list(stats['std']),
        'n_frames': int(stats['n_frames']),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transforms.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add transforms.py tests/test_transforms.py
git commit -m "feat: add feature standardization with train-split stats persistence"
```

---

### Task 3: `model.py` cleanups + callers

**Files:**
- Modify: `model.py:20-33` (LipEncoder default), `model.py:112-128` (DigitVerifier default), `model.py:152-166` (SequenceVerifier signature)
- Modify: `train.py:381-388` (SequenceVerifier call), `preflight.py:119-127` (build_model call)
- Test: existing suite must stay green.

**Interfaces:**
- Consumes: nothing new.
- Produces: `SequenceVerifier.__init__(self, n_classes=N_CLASSES, embed_dim=64, n_features=8, hidden_dim=128, encoder_type='transformer')` — **no `seq_len` parameter**. `n_features` default is `8` on `LipEncoder`, `DigitVerifier`, `SequenceVerifier` (was 5). Later tasks construct these without `seq_len`.

- [ ] **Step 1: Edit `model.py`**

1. `LipEncoder.__init__` (model.py:23): change `n_features=5,` → `n_features=8,`
2. `DigitVerifier.__init__` (model.py:117): change `n_features=5,` → `n_features=8,`
3. `SequenceVerifier.__init__` (model.py:157): remove the `seq_len=8,` parameter and change `n_features=5,` → `n_features=8,`. New signature:

```python
    def __init__(self, n_classes=N_CLASSES, embed_dim=64,
                 n_features=8, hidden_dim=128, encoder_type='transformer'):
```

4. Update the class docstring line "verify all 8 digits at once" — leave as is (it describes the digit dataset, still true).

- [ ] **Step 2: Update callers**

`train.py:381-388` — remove `seq_len=seq_len,` from the `SequenceVerifier(...)` call:

```python
        model = SequenceVerifier(
            n_classes=n_classes,
            embed_dim=EMBED_DIM,
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
            encoder_type=args.encoder,
        ).to(DEVICE)
```

`preflight.py:119-127` — remove `seq_len=seq_len,` from the `SequenceVerifier(...)` call inside `build_model` (keep the `seq_len` function parameter; `TinyLipSeq2Seq` still uses it for `max_src_len`/`max_tgt_len`).

- [ ] **Step 3: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all existing tests — shapes untouched, dataset layer not yet modified)

- [ ] **Step 4: Commit**

```bash
git add model.py train.py preflight.py
git commit -m "fix: remove unused SequenceVerifier.seq_len and stale n_features default"
```

---

### Task 4: `checkpoint.py` — config save/load + model factory

**Files:**
- Create: `checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `DigitVerifier`, `SequenceVerifier`, `TinyLipSeq2Seq` from `model.py` (post-Task-3 signatures).
- Produces:
  - `CONFIG_FILENAME = 'config.json'`
  - `save_model_config(model_dir: str, config: dict, filename: str = CONFIG_FILENAME) -> str` — train.py passes `filename=f'config_<mode>.json'` because modes share `models/<dataset>/<encoder>/`.
  - `load_model_config(checkpoint_path: str, mode: str | None = None) -> dict | None` — accepts a checkpoint file path or a directory; with `mode` given it looks for `config_<mode>.json`, else `config.json`; `None` when absent (legacy checkpoints).
  - `build_model_from_config(config: dict, device: torch.device) -> nn.Module` — `config['mode']` ∈ {`digit`, `sequence`, `seq2seq`}; `config['model']` holds constructor kwargs passed verbatim.

- [ ] **Step 1: Write the failing tests** — create `tests/test_checkpoint.py`:

```python
import os
import tempfile
import unittest

import torch

import checkpoint as ckpt
from model import DigitVerifier, SequenceVerifier, TinyLipSeq2Seq


class SaveLoadConfigTests(unittest.TestCase):
    def test_save_load_roundtrip_from_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ckpt.save_model_config(tmp, {'mode': 'digit', 'model': {}})
            self.assertEqual(os.path.basename(path), 'config.json')
            loaded = ckpt.load_model_config(os.path.join(tmp, 'best_digit_verifier.pt'))
            self.assertEqual(loaded, {'mode': 'digit', 'model': {}})

    def test_mode_keyed_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt.save_model_config(tmp, {'mode': 'digit', 'model': {}},
                                   filename='config_digit.json')
            loaded = ckpt.load_model_config(
                os.path.join(tmp, 'best_digit_verifier.pt'), mode='digit'
            )
            self.assertEqual(loaded, {'mode': 'digit', 'model': {}})

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt.save_model_config(tmp, {'mode': 'digit', 'model': {}})
            self.assertEqual(ckpt.load_model_config(tmp), {'mode': 'digit', 'model': {}})

    def test_missing_config_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(ckpt.load_model_config(os.path.join(tmp, 'best.pt')))
            self.assertIsNone(ckpt.load_model_config(os.path.join(tmp, 'best.pt'), mode='digit'))


class BuildModelFromConfigTests(unittest.TestCase):
    def test_digit_verifier(self):
        config = {
            'mode': 'digit',
            'model': {'n_classes': 11, 'embed_dim': 64, 'n_features': 8,
                      'hidden_dim': 128, 'encoder_type': 'bigru'},
        }
        model = ckpt.build_model_from_config(config, torch.device('cpu'))
        self.assertIsInstance(model, DigitVerifier)

    def test_sequence_verifier_has_no_seq_len_kwarg(self):
        config = {
            'mode': 'sequence',
            'model': {'n_classes': 11, 'embed_dim': 64, 'n_features': 8,
                      'hidden_dim': 128, 'encoder_type': 'transformer'},
        }
        model = ckpt.build_model_from_config(config, torch.device('cpu'))
        self.assertIsInstance(model, SequenceVerifier)

    def test_seq2seq_respects_config_max_lens(self):
        config = {
            'mode': 'seq2seq',
            'model': {'vocab_size': 14, 'pad_idx': 11, 'n_features': 8,
                      'seg_embed_dim': 48, 'n_heads': 4, 'n_encoder_layers': 1,
                      'n_decoder_layers': 1, 'ff_dim': 128, 'dropout': 0.1,
                      'max_src_len': 8, 'max_tgt_len': 9, 'hidden_dim': 64,
                      'encoder_type': 'transformer'},
        }
        model = ckpt.build_model_from_config(config, torch.device('cpu'))
        self.assertIsInstance(model, TinyLipSeq2Seq)
        self.assertEqual(model.src_pos_emb.num_embeddings, 8)
        # Regression for the inference crash: a second build must load the
        # first model's state_dict without shape mismatches.
        twin = ckpt.build_model_from_config(config, torch.device('cpu'))
        twin.load_state_dict(model.state_dict())

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ckpt.build_model_from_config({'mode': 'nope', 'model': {}}, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'checkpoint'`

- [ ] **Step 3: Write minimal implementation** — create `checkpoint.py`:

```python
"""Checkpoint config save/load shared by train, test, and inference.

Every checkpoint trained by train.py ships a sibling config file holding
the full model constructor config plus pipeline flags, so downstream
scripts rebuild the exact model instead of relying on hardcoded constants.
Modes share a model dir, so train.py writes mode-keyed config_<mode>.json.
"""
import json
import os

import torch

from model import DigitVerifier, SequenceVerifier, TinyLipSeq2Seq

CONFIG_FILENAME = 'config.json'


def save_model_config(model_dir, config, filename=CONFIG_FILENAME):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return path


def load_model_config(checkpoint_path, mode=None):
    """Return the config dict for a checkpoint path/dir, or None if absent."""
    base = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    if mode is not None:
        path = os.path.join(base, f'config_{mode}.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    path = os.path.join(base, CONFIG_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_model_from_config(config, device):
    mode = config['mode']
    model_kwargs = dict(config['model'])
    if mode == 'digit':
        model = DigitVerifier(**model_kwargs)
    elif mode == 'sequence':
        model = SequenceVerifier(**model_kwargs)
    elif mode == 'seq2seq':
        model = TinyLipSeq2Seq(**model_kwargs)
    else:
        raise ValueError(f'Unknown mode in checkpoint config: {mode}')
    return model.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_checkpoint.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint config.json save/load + model factory"
```

---

### Task 5: `dataset.py` transform integration

**Files:**
- Modify: `dataset.py` (constants `:9-19`, `_prepare_samples :87-103`, `_filter_fixed_length_samples :72-84`, all three dataset classes `:107-313`, collates added, `_print_sample_summary :316-339`, CLI `:342-355`)
- Modify: `preflight.py:86-108` (`build_dataset`)
- Test: `tests/test_datasets.py` (new), `tests/test_preflight.py` (update shapes)

**Interfaces:**
- Consumes: `transforms.correct_lip_speed_fps`, `transforms.resample_segment`, `transforms.standardize_segment` (Tasks 1–2).
- Produces (exact names later tasks rely on):
  - `SEG_LEN = 16` module constant; `MAX_SEQ_LEN = 30` unchanged.
  - `_prepare_samples(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS, speaker_filter=None) -> dict` with keys `segments`, `sequences`, `token_to_idx`, `feature_names`, `fps` (list[float], default 25.0 when npz lacks `fps`).
  - `_transform_segment(seg, fps, n_features, resample, seg_len, feature_stats) -> np.ndarray` — `_adapt_feature_dim` → fps-correct+resample (when `resample`) → standardize (when `feature_stats is not None`).
  - Dataset ctors (all three) gain keyword params: `resample=True, seg_len=SEG_LEN, feature_stats=None, speaker_filter=None`. When `resample=True` the padded length is `seg_len` and masks are all-ones; otherwise the legacy `max_seq_len`/`max_seg_len` pad-to-30 path.
  - `LipVerificationDataset.get_pair_info(idx) -> (seg_idx, true_digit_idx, claimed_idx, label)`.
  - `sequence_collate_fn(batch)` and `transcription_collate_fn(batch, pad_idx, eos_idx)` now live in `dataset.py` (train.py still has its own copies until Task 7 — temporary duplication keeps this task green).
  - `compute_split_stats` is Task 6 (not in this task).
  - Datasets get attribute `feature_name_to_index` (dict); module function `_feature_name_to_index(feature_names)`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_datasets.py`:

```python
import unittest
from functools import partial

import numpy as np
import torch

from dataset import (
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
    SEG_LEN,
    sequence_collate_fn,
    transcription_collate_fn,
)

FEATURE_DIM = 8


def _seg(rng, length, feature_dim=FEATURE_DIM):
    return rng.rand(length, feature_dim).astype(np.float32)


def _write_npz(tmpdir, name, sequences, segments, speakers=None, fps=None):
    import os

    n = len(sequences)
    path = os.path.join(tmpdir, name)
    np.savez_compressed(
        path,
        digit_sequences=np.array(sequences, dtype=object),
        digit_segments=np.array(segments, dtype=object),
        full_features=np.array([np.concatenate(s, axis=0) for s in segments], dtype=object),
        feature_names=np.array([f'f{i}' for i in range(FEATURE_DIM)]),
        speakers=np.array(speakers if speakers is not None else [f'spk{i}' for i in range(n)]),
        fps=np.array(fps if fps is not None else [25.0] * n),
    )
    return path


def _digit_fixtures(rng, n_videos=4):
    sequences = [[str(rng.randint(0, 10)) for _ in range(8)] for _ in range(n_videos)]
    segments = [[_seg(rng, rng.randint(4, 12)) for _ in range(8)] for _ in range(n_videos)]
    return sequences, segments


class TransformIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rng = np.random.RandomState(0)

    def test_resample_default_shapes_and_mask(self):
        sequences, segments = _digit_fixtures(self.rng)
        path = _write_npz(self._tmp.name, 'a.npz', sequences, segments)
        for ds_cls in (LipVerificationDataset, SequenceVerificationDataset, LipTranscriptionDataset):
            with self.subTest(ds_cls=ds_cls.__name__):
                ds = ds_cls(path, dataset='digit')
                feats, mask = ds[0][0], ds[0][1]
                self.assertEqual(tuple(feats.shape[-2:]), (SEG_LEN, FEATURE_DIM))
                self.assertTrue(torch.all(mask == 1.0))

    def test_no_resample_keeps_legacy_padded_shape(self):
        sequences, segments = _digit_fixtures(self.rng)
        path = _write_npz(self._tmp.name, 'a.npz', sequences, segments)
        ds = LipVerificationDataset(path, dataset='digit', resample=False)
        self.assertEqual(ds[0][0].shape, (30, FEATURE_DIM))

    def test_fps_correction_scales_lip_speed(self):
        sequences, segments = _digit_fixtures(self.rng, n_videos=2)
        for seg_list in segments:
            for seg in seg_list:
                seg[:, 6] = 0.01  # constant lip_speed for exact comparison
        fast_path = _write_npz(self._tmp.name, 'fast.npz', sequences, segments, fps=[50.0] * 2)
        slow_path = _write_npz(self._tmp.name, 'slow.npz', sequences, segments, fps=[25.0] * 2)
        fast = LipVerificationDataset(fast_path, dataset='digit')[0][0]
        slow = LipVerificationDataset(slow_path, dataset='digit')[0][0]
        np.testing.assert_allclose(
            fast[:, 6].numpy(), slow[:, 6].numpy() * 2.0, atol=1e-6
        )

    def test_standardization_zeroes_train_mean(self):
        sequences, segments = _digit_fixtures(self.rng)
        for seg_list in segments:
            for seg in seg_list:
                seg += np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        path = _write_npz(self._tmp.name, 'a.npz', sequences, segments)
        feature_stats = {
            'feature_names': [f'f{i}' for i in range(FEATURE_DIM)],
            'mean': [10.0] + [0.0] * 7,
            'std': [1.0] * FEATURE_DIM,
            'n_frames': 100,
        }
        ds = LipVerificationDataset(path, dataset='digit', feature_stats=feature_stats)
        feats = ds[0][0]
        np.testing.assert_allclose(feats[:, 0].numpy(), 0.0, atol=1e-5)


class SpeakerFilterTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rng = np.random.RandomState(1)

    def test_filter_keeps_only_requested_speakers(self):
        sequences, segments = _digit_fixtures(self.rng, n_videos=4)
        path = _write_npz(
            self._tmp.name, 'a.npz', sequences, segments,
            speakers=['alice', 'alice', 'bob', 'bob'],
        )
        ds = SequenceVerificationDataset(path, dataset='digit', speaker_filter={'bob'})
        self.assertEqual(len(ds.digit_sequences), 2)

    def test_filter_requires_speakers_array(self):
        import os

        sequences, segments = _digit_fixtures(self.rng, n_videos=2)
        path = os.path.join(self._tmp.name, 'nope.npz')
        np.savez_compressed(
            path,
            digit_sequences=np.array(sequences, dtype=object),
            digit_segments=np.array(segments, dtype=object),
            full_features=np.array([np.concatenate(s, axis=0) for s in segments], dtype=object),
            feature_names=np.array([f'f{i}' for i in range(FEATURE_DIM)]),
        )
        with self.assertRaises(ValueError):
            LipVerificationDataset(path, dataset='digit', speaker_filter={'x'})


class GetPairInfoTests(unittest.TestCase):
    def test_returns_indices_and_label(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.RandomState(2)
            sequences, segments = _digit_fixtures(rng, n_videos=2)
            path = _write_npz(tmp, 'a.npz', sequences, segments)
            ds = LipVerificationDataset(path, dataset='digit')
            seg_idx, true_idx, claimed_idx, label = ds.get_pair_info(0)
            self.assertEqual(true_idx, claimed_idx)
            self.assertEqual(label, 1)
            seg_idx2, true_idx2, claimed_idx2, label2 = ds.get_pair_info(1)
            self.assertEqual(label2, 0)
            self.assertNotEqual(true_idx2, claimed_idx2)


class CollateTests(unittest.TestCase):
    def test_transcription_collate_pads_with_specials(self):
        batch = [
            (torch.randn(8, SEG_LEN, FEATURE_DIM), torch.ones(8, SEG_LEN),
             torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), 8),
            (torch.randn(6, SEG_LEN, FEATURE_DIM), torch.ones(6, SEG_LEN),
             torch.tensor([9, 8, 7, 6, 5, 4]), 6),
        ]
        collate = partial(transcription_collate_fn, pad_idx=10, eos_idx=12)
        feats, masks, src_pad, targets = collate(batch)
        self.assertEqual(feats.shape, (2, 8, SEG_LEN, FEATURE_DIM))
        self.assertTrue(src_pad[1, 6:].all() and not src_pad[1, :6].any())
        self.assertEqual(targets.shape, (2, 9))
        self.assertEqual(targets[1, 6].item(), 12)
        self.assertEqual(targets[1, 7].item(), 10)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: FAIL — `ImportError: cannot import name 'SEG_LEN'`

- [ ] **Step 3: Implement `dataset.py` changes**

1. **Imports + constants** (top of file): add

```python
from transforms import correct_lip_speed_fps, resample_segment, standardize_segment
```

Replace the stale `FEATURE_NAME_TO_INDEX` dict (dataset.py:13-19) with:

```python
SEG_LEN = 16
```

Add:

```python
def _feature_name_to_index(feature_names):
    return {name: idx for idx, name in enumerate(feature_names or [])}
```

2. **`_filter_fixed_length_samples`** — extend to filter fps in lockstep:

```python
def _filter_fixed_length_samples(digit_segments, digit_sequences, fps, expected_len=EXPECTED_N_DIGITS):
    """Keep only samples where both sequence and segment count match expected length."""
    keep_idx = []
    for i, (segs, seq) in enumerate(zip(digit_segments, digit_sequences)):
        if len(segs) == expected_len and len(seq) == expected_len:
            keep_idx.append(i)

    if not keep_idx:
        raise ValueError(
            f'No samples with exactly {expected_len} digits found after filtering'
        )

    return digit_segments[keep_idx], digit_sequences[keep_idx], [fps[i] for i in keep_idx]
```

3. **`_prepare_samples`** — new signature/return:

```python
def _prepare_samples(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS, speaker_filter=None):
    data = np.load(npz_path, allow_pickle=True)
    digit_segments = data['digit_segments']
    digit_sequences = data['digit_sequences']
    feature_names = [str(name) for name in data['feature_names'].tolist()] if 'feature_names' in data else None
    fps = data['fps'].tolist() if 'fps' in data.files else [25.0] * len(digit_segments)
    speakers = [str(s) for s in data['speakers']] if 'speakers' in data.files else None

    if speaker_filter is not None:
        if speakers is None:
            raise ValueError('speaker_filter requested but npz has no speakers array')
        keep = [i for i, spk in enumerate(speakers) if spk in speaker_filter]
        digit_segments = digit_segments[keep]
        digit_sequences = digit_sequences[keep]
        fps = [fps[i] for i in keep]

    # Digit dataset is expected to contain fixed-length sequences (default: 8).
    # GRID uses variable-length word sequences, so we keep all samples.
    if dataset == 'digit':
        digit_segments, digit_sequences, fps = _filter_fixed_length_samples(
            digit_segments,
            digit_sequences,
            fps,
            expected_len=expected_len,
        )

    token_to_idx = _build_token_to_idx(digit_sequences, dataset=dataset)
    return {
        'segments': digit_segments,
        'sequences': digit_sequences,
        'token_to_idx': token_to_idx,
        'feature_names': feature_names,
        'fps': fps,
    }
```

4. **Transform helper** (module level):

```python
def _transform_segment(seg, fps, n_features, resample, seg_len, feature_stats):
    seg = _adapt_feature_dim(seg, n_features)
    if resample:
        seg = correct_lip_speed_fps(seg, fps)
        seg = resample_segment(seg, seg_len)
    if feature_stats is not None:
        seg = standardize_segment(seg, feature_stats['mean'], feature_stats['std'])
    return seg
```

5. **`LipVerificationDataset`** — constructor unpacks the dict, stores fps per segment as 3-tuples, gains new params:

```python
    def __init__(self, npz_path, dataset='digit', token_to_idx=None,
                 max_seq_len=MAX_SEQ_LEN, neg_ratio=NEG_RATIO, seed=42,
                 expected_len=EXPECTED_N_DIGITS, resample=True, seg_len=SEG_LEN,
                 feature_stats=None, speaker_filter=None):
        self.dataset = dataset
        prepared = _prepare_samples(
            npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
        )
        self.digit_segments = prepared['segments']
        self.digit_sequences = prepared['sequences']
        self.token_to_idx = token_to_idx or prepared['token_to_idx']
        self.feature_names = prepared['feature_names']
        self.feature_name_to_index = _feature_name_to_index(self.feature_names)
        self.fps = prepared['fps']
        self.max_seq_len = max_seq_len
        self.pad_len = seg_len if resample else max_seq_len
        self.resample = resample
        self.seg_len = seg_len
        self.feature_stats = feature_stats
        self.rng = np.random.RandomState(seed)
        self.n_features = len(self.feature_names) if self.feature_names else _infer_n_features(self.digit_segments)
        self.vocab_size = len(self.token_to_idx)

        # Build flat list of (segment_features, char_idx, fps)
        self.segments = []
        for vid_idx in range(len(self.digit_segments)):
            segs = self.digit_segments[vid_idx]
            digits = self.digit_sequences[vid_idx]
            video_fps = self.fps[vid_idx]
            for seg, digit in zip(segs, digits):
                self.segments.append((seg, self.token_to_idx[str(digit)], video_fps))

        all_indices = list(range(self.vocab_size))
        self.pairs = []
        for i, (_, cidx, _) in enumerate(self.segments):
            self.pairs.append((i, cidx, 1))
            wrong = [d for d in all_indices if d != cidx]
            for _ in range(neg_ratio):
                self.pairs.append((i, self.rng.choice(wrong), 0))
```

`__getitem__` — transform before padding (pad logic unchanged except `self.pad_len` replaces `self.max_seq_len`):

```python
    def __getitem__(self, idx):
        seg_idx, claimed, label = self.pairs[idx]
        seg_features, _, fps = self.segments[seg_idx]
        seg_features = _transform_segment(
            seg_features, fps, self.n_features, self.resample, self.seg_len, self.feature_stats
        )

        t = seg_features.shape[0]
        if t >= self.pad_len:
            feat = seg_features[:self.pad_len].astype(np.float32)
            mask = np.ones(self.pad_len, dtype=np.float32)
        else:
            feat = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
            feat[:t] = seg_features
            mask = np.zeros(self.pad_len, dtype=np.float32)
            mask[:t] = 1.0

        return (
            torch.FloatTensor(feat),
            torch.FloatTensor(mask),
            torch.LongTensor([claimed]),
            torch.FloatTensor([label]),
        )

    def get_pair_info(self, idx):
        """Return (seg_idx, true_digit, claimed_digit, label) for analysis."""
        seg_idx, claimed, label = self.pairs[idx]
        _, true_digit, _ = self.segments[seg_idx]
        return seg_idx, true_digit, claimed, label
```

6. **`SequenceVerificationDataset`** — same new params; store `self.fps`; `_pad_segment(seg, fps)` becomes:

```python
    def _pad_segment(self, seg, fps):
        seg = _transform_segment(
            seg, fps, self.n_features, self.resample, self.seg_len, self.feature_stats
        )
        t = seg.shape[0]
        if t >= self.pad_len:
            feat = seg[:self.pad_len].astype(np.float32)
            mask = np.ones(self.pad_len, dtype=np.float32)
        else:
            feat = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
            feat[:t] = seg
            mask = np.zeros(self.pad_len, dtype=np.float32)
            mask[:t] = 1.0
        return feat, mask
```

and `__getitem__` passes `self.fps[vid_idx]`:

```python
        for seg, fps in zip(segments, [self.fps[vid_idx]] * len(segments)):
            f, m = self._pad_segment(seg, fps)
```

(Keep the rest of `__getitem__` identical.)

7. **`LipTranscriptionDataset`** — same treatment as `SequenceVerificationDataset` (params, `self.fps`, `self.pad_len`, `_pad_segment(seg, fps)`, fps loop in `__getitem__`).

8. **Add collates to `dataset.py`** — copy `sequence_collate_fn` verbatim from `train.py:47-79` and `transcription_collate_fn` from `train.py:82-117` with the signature changed to `def transcription_collate_fn(batch, pad_idx, eos_idx):` (body identical, using the parameters instead of globals).

9. **`_print_sample_summary`** — replace `FEATURE_NAME_TO_INDEX[feature_name]` with `ds.feature_name_to_index[feature_name]`; in `_parse_args` remove `choices=list(FEATURE_NAME_TO_INDEX.keys())` from `--feature-name` and validate after load:

```python
    if args.feature_name not in ds.feature_name_to_index:
        raise SystemExit(
            f"Unknown feature '{args.feature_name}'. Available: {sorted(ds.feature_name_to_index)}"
        )
```

- [ ] **Step 4: Update `preflight.py` `build_dataset`** (preflight.py:86-108) — add transform params and pass them through:

```python
def build_dataset(dataset_name, mode, npz_path, token_to_idx=None,
                  resample=True, seg_len=16):
    common = dict(dataset=dataset_name, token_to_idx=token_to_idx,
                  resample=resample, seg_len=seg_len)
    if mode == 'digit':
        return LipVerificationDataset(npz_path, max_seq_len=MAX_SEQ_LEN, **common)
    if mode == 'sequence':
        return SequenceVerificationDataset(npz_path, max_seg_len=MAX_SEQ_LEN, **common)
    if mode == 'seq2seq':
        return LipTranscriptionDataset(npz_path, max_seg_len=MAX_SEQ_LEN, **common)
    raise ValueError(f'Unknown mode: {mode}')
```

- [ ] **Step 5: Update `tests/test_preflight.py`** — in `test_digit_dataset_keeps_variable_lengths` change

```python
            self.assertEqual(trn_ds[0][0].shape[1:], (30, 8))
```

to

```python
            self.assertEqual(trn_ds[0][0].shape[1:], (16, 8))
```

and add a new test:

```python
    def test_datasets_support_legacy_no_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['3', '5', '7', '9', '2', '4', '6', '8']]
            segments = [[_make_segment(5) for _ in range(8)]]
            _write_npz(npz_path, sequences, segments)

            ds = build_dataset('digit', 'digit', npz_path, resample=False)
            self.assertEqual(ds[0][0].shape, (30, 8))
```

(`build_dataset` import already exists in that file.)

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/ -v`
Expected: PASS (transforms 13, checkpoint 7, datasets 8, preflight 6)

- [ ] **Step 7: Commit**

```bash
git add dataset.py preflight.py tests/test_datasets.py tests/test_preflight.py
git commit -m "feat: integrate resample/standardize/speaker-filter into datasets"
```

---

### Task 6: `compute_split_stats`

**Files:**
- Modify: `dataset.py` (add function after `_transform_segment`)
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `_prepare_samples`, `_transform_segment` (Task 5), `transforms.compute_feature_stats` (Task 2).
- Produces: `compute_split_stats(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS, resample=True, seg_len=SEG_LEN, speaker_filter=None) -> dict` with keys `mean`, `std`, `n_frames` (no `feature_names` — callers add those when saving via `save_feature_stats`).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_datasets.py` (update the `from dataset import (...)` list to include `compute_split_stats`):

```python
class ComputeSplitStatsTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rng = np.random.RandomState(4)

    def test_stats_use_only_filtered_speakers(self):
        sequences = [['1'] * 8, ['2'] * 8]
        segs_a = [np.full((5, FEATURE_DIM), 0.0, dtype=np.float32) for _ in range(8)]
        segs_b = [np.full((5, FEATURE_DIM), 10.0, dtype=np.float32) for _ in range(8)]
        path = _write_npz(
            self._tmp.name, 'a.npz', sequences, [segs_a, segs_b],
            speakers=['alice', 'bob'],
        )
        stats = compute_split_stats(path, dataset='digit', resample=False,
                                    speaker_filter={'alice'})
        np.testing.assert_allclose(stats['mean'], [0.0] * FEATURE_DIM, atol=1e-6)
        self.assertEqual(stats['n_frames'], 40)

    def test_stats_reflect_resampled_lengths(self):
        sequences = [['1'] * 8]
        segments = [[_seg(self.rng, 6) for _ in range(8)]]
        path = _write_npz(self._tmp.name, 'a.npz', sequences, segments)
        stats = compute_split_stats(path, dataset='digit', resample=True, seg_len=16)
        self.assertEqual(stats['n_frames'], 8 * 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_split_stats'`

- [ ] **Step 3: Implement** — add to `dataset.py` after `_transform_segment`:

```python
def compute_split_stats(npz_path, dataset='digit', expected_len=EXPECTED_N_DIGITS,
                        resample=True, seg_len=SEG_LEN, speaker_filter=None):
    """Per-feature stats over transformed (unstandardized) segments of a split."""
    prepared = _prepare_samples(
        npz_path, dataset=dataset, expected_len=expected_len, speaker_filter=speaker_filter,
    )
    segments = prepared['segments']
    fps = prepared['fps']
    feature_names = prepared['feature_names']
    n_features = len(feature_names) if feature_names else _infer_n_features(segments)

    def iter_segments():
        for vid_idx in range(len(segments)):
            video_fps = fps[vid_idx]
            for seg in segments[vid_idx]:
                yield _transform_segment(seg, video_fps, n_features, resample, seg_len, None)

    return compute_feature_stats(iter_segments(), n_features)
```

and add `compute_feature_stats` to the `from transforms import ...` line at the top of `dataset.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (datasets 10)

- [ ] **Step 5: Commit**

```bash
git add dataset.py tests/test_datasets.py
git commit -m "feat: train-split feature stats computation for standardization"
```

---

### Task 7: `train.py` — flags, stats, val split, config, new layout

**Files:**
- Modify: `train.py` (imports `:14-19`, delete collates `:47-117`, delete `_dataset_paths`/`_output_dirs` `:120-138`, main `:306-603`)
- Test: smoke run (Steps below); no new unit test file — verified end-to-end.

**Interfaces:**
- Consumes: `dataset.SEG_LEN`, `dataset.compute_split_stats`, `dataset.sequence_collate_fn`, `dataset.transcription_collate_fn` (Task 5/6); `transforms.load_feature_stats`, `transforms.save_feature_stats` (Task 2); `checkpoint.save_model_config`, `checkpoint.build_model_from_config` (Task 4).
- Produces: CLI flags `--seg-len` (default `SEG_LEN`=16), `--no-resample`, `--no-standardize`, `--data-dir` (default `processed_data`), `--model-root` (default `models`), `--log-root` (default `runs`). Output layout `models/<dataset>/<encoder>/` + `runs/<dataset>/<encoder>/`. Writes `models/<dataset>/<encoder>/config.json`, `vocab.json`, `feature_stats.json` under `<data-dir>/<dataset>/`. Best checkpoint selected on val (every 10th sorted train speaker) — `results_*.json` gains `selection` and `best_selection_metric`.

- [ ] **Step 1: Update imports and delete duplicated code**

1. Replace the dataset import block (train.py:14-19) with:

```python
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
```

Add `from functools import partial` to the stdlib imports at the top.

2. Delete the local `sequence_collate_fn` (train.py:47-79) and `transcription_collate_fn` (train.py:82-117) definitions — they now come from `dataset.py`.

- [ ] **Step 2: Replace dir helpers** — delete `_dataset_paths` (train.py:120-126) and `_output_dirs` (train.py:129-138); add:

```python
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
```

- [ ] **Step 3: Add CLI flags** — in the argparse block after `--lr`:

```python
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
```

- [ ] **Step 4: Rewrite the main setup section**

Replace the body between `print(f'Device: {DEVICE}')` and the `if args.mode == 'digit':` dataset construction with:

```python
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
```

Then the three dataset-construction branches become:

```python
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
```

Then replace the model construction for all three modes with config-driven construction (delete the three `model = DigitVerifier(...)` / `SequenceVerifier(...)` / `TinyLipSeq2Seq(...)` blocks):

```python
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
```

(Note the module-level `PAD_IDX = None` etc. placeholders at train.py:41-44 stay; `__main__` assigns real values as before, now in one place.)

- [ ] **Step 5: Save vocab + config** — after the existing `vocab.json` save block, add:

```python
    config_path = ckpt.save_model_config(model_dir, config, filename=f'config_{args.mode}.json')
    print(f'Model config saved to {config_path}')
```

- [ ] **Step 6: Loaders** — build the three loaders; collate for seq2seq uses partial:

```python
    collate_fn = None
    if args.mode == 'sequence':
        collate_fn = sequence_collate_fn
    elif args.mode == 'seq2seq':
        collate_fn = partial(transcription_collate_fn, pad_idx=PAD_IDX, eos_idx=EOS_IDX)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False,
                         collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=train_shuffle, sampler=train_sampler, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None
    )
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
```

Print line update — extend the data print with val info:

```python
    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else 0} samples, '
          f'Test: {len(test_ds)} samples')
```

- [ ] **Step 7: Selection loop** — introduce a selection loader and use it for periodic eval:

```python
    selection_loader = val_loader if val_loader is not None else test_loader
    selection_name = 'val' if val_loader is not None else 'test'
```

Inside the epoch loop, replace `evaluate(model, test_loader, DEVICE)` / `evaluate_seq2seq(model, test_loader, DEVICE)` with the same calls on `selection_loader`, and prefix the metric print lines with the split name, e.g. `f"Epoch {epoch:3d} | loss={loss:.4f} | {selection_name} AUC={metrics['auc']:.4f} | ..."` (keep the existing metric formatting; only the evaluated loader and the label change).

- [ ] **Step 8: Final evaluation + results** — replace `model.load_state_dict(torch.load(model_save_path, weights_only=True))` with:

```python
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))
```

and evaluate on `test_loader` (unchanged calls). Extend the results dict:

```python
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
```

- [ ] **Step 9: Smoke run (digit mode, new pipeline)**

Create synthetic data and run one epoch:

```bash
mkdir -p /tmp/lip2text_smoke/processed_data/digit
python - <<'EOF'
import numpy as np

rng = np.random.RandomState(0)
speakers = [f'spk{i:02d}' for i in range(6)]

def make_split(n, path):
    segments, sequences, fps, spk = [], [], [], []
    for i in range(n):
        seq = [str(rng.randint(0, 10)) for _ in range(8)]
        segs = [rng.rand(rng.randint(4, 12), 8).astype(np.float32) for _ in range(8)]
        sequences.append(seq)
        segments.append(segs)
        fps.append(25.0)
        spk.append(speakers[i % len(speakers)])
    full = [np.concatenate(s, axis=0) for s in segments]
    np.savez_compressed(
        path,
        video_ids=np.array([f'v{i}' for i in range(n)]),
        speakers=np.array(spk),
        digit_sequences=np.array(sequences, dtype=object),
        full_features=np.array(full, dtype=object),
        digit_segments=np.array(segments, dtype=object),
        fps=np.array(fps),
        feature_names=np.array([f'f{i}' for i in range(8)]),
    )

make_split(24, '/tmp/lip2text_smoke/processed_data/digit/train.npz')
make_split(8, '/tmp/lip2text_smoke/processed_data/digit/test.npz')
EOF
python train.py --dataset digit --mode digit --epochs 1 --batch_size 8 \
  --data-dir /tmp/lip2text_smoke/processed_data \
  --model-root /tmp/lip2text_smoke/models --log-root /tmp/lip2text_smoke/runs
```

Expected: run completes; verify artifacts:

```bash
python - <<'EOF'
import json, os

root = '/tmp/lip2text_smoke'
assert os.path.exists(f'{root}/processed_data/digit/feature_stats.json')
assert os.path.exists(f'{root}/models/digit/transformer/best_digit_verifier.pt')
assert os.path.exists(f'{root}/models/digit/transformer/vocab.json')
config = json.load(open(f'{root}/models/digit/transformer/config_digit.json'))
assert config['resample'] is True and config['standardized'] is True
assert config['seg_len'] == 16 and config['n_features'] == 8
results = json.load(open(f'{root}/models/digit/transformer/results_digit.json'))
assert results['selection'] == 'val'
print('smoke artifacts OK')
EOF
```

- [ ] **Step 10: Smoke run (seq2seq + escape hatches)**

```bash
python train.py --dataset digit --mode seq2seq --epochs 1 --batch_size 8 \
  --data-dir /tmp/lip2text_smoke/processed_data \
  --model-root /tmp/lip2text_smoke/models --log-root /tmp/lip2text_smoke/runs
python - <<'EOF'
import json
c1 = json.load(open('/tmp/lip2text_smoke/models/digit/transformer/config_digit.json'))
c2 = json.load(open('/tmp/lip2text_smoke/models/digit/transformer/config_seq2seq.json'))
assert c1['mode'] == 'digit' and c2['mode'] == 'seq2seq'
assert c2['model']['max_src_len'] == 8, c2['model']
print('mode-keyed configs OK')
EOF
rm -rf /tmp/lip2text_smoke_legacy
mkdir -p /tmp/lip2text_smoke_legacy/processed_data/digit
cp /tmp/lip2text_smoke/processed_data/digit/{train,test}.npz /tmp/lip2text_smoke_legacy/processed_data/digit/
python train.py --dataset digit --mode digit --epochs 1 --batch_size 8 --no-resample --no-standardize \
  --data-dir /tmp/lip2text_smoke_legacy/processed_data \
  --model-root /tmp/lip2text_smoke_legacy/models --log-root /tmp/lip2text_smoke_legacy/runs
python - <<'EOF'
import json, os
config = json.load(open('/tmp/lip2text_smoke_legacy/models/digit/transformer/config_digit.json'))
assert config['resample'] is False and config['standardized'] is False
assert not os.path.exists('/tmp/lip2text_smoke_legacy/processed_data/digit/feature_stats.json')
print('legacy flags smoke OK')
EOF
```

- [ ] **Step 11: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add train.py checkpoint.py tests/test_checkpoint.py
git commit -m "feat: train with val-split selection, standardization, resampling, checkpoint configs"
```

---

### Task 8: `test.py` rewrite

**Files:**
- Rewrite: `test.py` (full file replacement)
- Test: smoke run reusing Task 7 artifacts; no new unit test file.

**Interfaces:**
- Consumes: `dataset.LipVerificationDataset`, `dataset.SequenceVerificationDataset`, `dataset.sequence_collate_fn`; `checkpoint.load_model_config`, `checkpoint.build_model_from_config`; `transforms.load_feature_stats`.
- Produces: CLI `--dataset {digit,grid}` (required in practice), `--mode`, `--encoder` (default: from config), `--model_path`, `--data-dir` (default `processed_data`), `--batch_size`, `--threshold`, `--save`. Reads `config_<mode>.json` / `vocab.json` next to the checkpoint. GRID supported.

- [ ] **Step 1: Replace `test.py` content** with:

```python
"""
Lip-Text Verification Testing
------------------------------
Evaluate a trained verification model on the test split.
Reports AUC, EER, accuracy, per-token breakdown, confusion matrix.

Models are reconstructed from the config_<mode>.json written next to the
checkpoint by train.py. Legacy checkpoints (no config) fall back to the
historical hardcoded dimensions.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import checkpoint as ckpt
from dataset import (
    LipVerificationDataset,
    SequenceVerificationDataset,
    sequence_collate_fn,
)
from model import DigitVerifier, SequenceVerifier, CHAR_TO_IDX
from transforms import load_feature_stats

PROCESSED_ROOT = 'processed_data'
MODEL_ROOT = 'models'
EMBED_DIM = 64      # legacy fallback (checkpoints without config)
HIDDEN_DIM = 128    # legacy fallback
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def _candidate_model_paths(dataset, mode, encoder=None):
    """New layout first, then legacy layouts."""
    encoders = ([encoder] if encoder else []) + ['transformer', 'bigru']
    paths = [os.path.join(MODEL_ROOT, dataset, enc, f'best_{mode}_verifier.pt')
             for enc in encoders]
    paths.append(os.path.join(MODEL_ROOT, 'transformer_encoder', f'best_{mode}_verifier.pt'))
    paths.append(os.path.join(MODEL_ROOT, f'best_{mode}_verifier.pt'))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def evaluate(model, loader, device):
    """Run model on all batches, return labels, probabilities, and metrics."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing', leave=False):
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
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
    preds_eer = (all_probs >= eer_threshold).astype(int)
    preds_05 = (all_probs >= 0.5).astype(int)

    metrics = {
        'auc': float(auc),
        'eer': eer,
        'eer_threshold': eer_threshold,
        'acc_at_eer': float(np.mean(preds_eer == all_labels)),
        'acc_at_05': float(np.mean(preds_05 == all_labels)),
        'n_samples': int(len(all_labels)),
        'n_positive': int(all_labels.sum()),
        'n_negative': int((1 - all_labels).sum()),
    }
    return all_labels, all_probs, metrics


def per_token_analysis(dataset, all_probs, threshold, idx_to_token):
    """Per-token accuracy breakdown over claimed tokens (digit mode only)."""
    token_stats = {token: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                   for token in idx_to_token.values()}
    for i in range(len(dataset)):
        _, _, claimed_idx, label = dataset.get_pair_info(i)
        pred = 1 if all_probs[i] >= threshold else 0
        claimed_char = idx_to_token[claimed_idx]
        if label == 1:
            token_stats[claimed_char]['tp' if pred == 1 else 'fn'] += 1
        else:
            token_stats[claimed_char]['tn' if pred == 0 else 'fp'] += 1
    return token_stats


def print_results(metrics, token_stats=None):
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  EER:            {metrics['eer']:.4f}")
    print(f"  EER threshold:  {metrics['eer_threshold']:.4f}")
    print(f"  Acc @ EER:      {metrics['acc_at_eer']:.4f}")
    print(f"  Acc @ 0.5:      {metrics['acc_at_05']:.4f}")
    print(f"  Samples:        {metrics['n_samples']} "
          f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})")
    print(f"{'='*60}")

    if token_stats:
        print(f"\n  Per-token breakdown (threshold={metrics['eer_threshold']:.4f}):")
        print(f"  {'Token':>7} | {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} | {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-'*7}-+-{'-'*23}-+-{'-'*20}")
        for token, s in sorted(token_stats.items()):
            total = s['tp'] + s['fp'] + s['tn'] + s['fn']
            if total == 0:
                continue
            prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
            rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {token:>7} | {s['tp']:>5} {s['fp']:>5} {s['tn']:>5} {s['fn']:>5} | "
                  f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        print()


def _load_vocab(model_path, dataset):
    vocab_path = os.path.join(os.path.dirname(model_path), 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            return {str(k): int(v) for k, v in json.load(f).items()}
    if dataset == 'digit':
        return dict(CHAR_TO_IDX)
    raise RuntimeError(f'vocab.json not found next to checkpoint ({model_path}); '
                       'required for non-digit datasets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lip-text verification model')
    parser.add_argument('--dataset', choices=['digit', 'grid'], default='digit')
    parser.add_argument('--mode', choices=['digit', 'sequence'], default='digit')
    parser.add_argument('--encoder', choices=['bigru', 'transformer'], default=None,
                        help='Encoder variant (default: taken from checkpoint config)')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data-dir', default=PROCESSED_ROOT)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.model_path is not None:
        model_path = args.model_path
    else:
        candidates = _candidate_model_paths(args.dataset, args.mode, args.encoder)
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    test_path = os.path.join(args.data_dir, args.dataset, 'test.npz')
    if not os.path.exists(test_path):
        print(f'ERROR: Test data not found at {test_path}. Run preprocess.py first.')
        raise SystemExit(1)
    if not os.path.exists(model_path):
        print(f'ERROR: Model checkpoint not found at {model_path}. Run train.py first.')
        raise SystemExit(1)

    config = ckpt.load_model_config(model_path, mode=args.mode)
    encoder = config['encoder_type'] if config else (args.encoder or 'transformer')

    print(f'Device: {DEVICE}')
    print(f'Dataset: {args.dataset}')
    print(f'Mode: {args.mode}-level verification')
    print(f'Encoder: {encoder}')
    print(f'Model: {model_path}')

    token_to_idx = _load_vocab(model_path, args.dataset)
    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    resample = bool(config.get('resample', False)) if config else False
    seg_len = int(config.get('seg_len', 16)) if config else 16
    feature_stats = None
    if config and config.get('standardized'):
        stats_path = os.path.join(args.data_dir, args.dataset, 'feature_stats.json')
        feature_stats = load_feature_stats(stats_path)
        if list(feature_stats['feature_names']) != list(config['feature_names']):
            raise RuntimeError('feature_stats.json feature order does not match checkpoint config')

    if args.mode == 'digit':
        test_ds = LipVerificationDataset(test_path, dataset=args.dataset,
                                         token_to_idx=token_to_idx, seed=99,
                                         resample=resample, seg_len=seg_len,
                                         feature_stats=feature_stats)
    else:
        test_ds = SequenceVerificationDataset(test_path, dataset=args.dataset,
                                              token_to_idx=token_to_idx, seed=99,
                                              resample=resample, seg_len=seg_len,
                                              feature_stats=feature_stats)

    collate_fn = sequence_collate_fn if args.mode == 'sequence' else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, collate_fn=collate_fn)
    print(f'Test samples: {len(test_ds)}')

    if config:
        model = ckpt.build_model_from_config(config, DEVICE)
    elif args.mode == 'digit':
        model = DigitVerifier(n_classes=test_ds.vocab_size, embed_dim=EMBED_DIM,
                              n_features=test_ds.n_features, hidden_dim=HIDDEN_DIM,
                              encoder_type=encoder).to(DEVICE)
    else:
        model = SequenceVerifier(n_classes=test_ds.vocab_size, embed_dim=EMBED_DIM,
                                 n_features=test_ds.n_features, hidden_dim=HIDDEN_DIM,
                                 encoder_type=encoder).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    all_labels, all_probs, metrics = evaluate(model, test_loader, DEVICE)

    threshold = args.threshold if args.threshold is not None else metrics['eer_threshold']
    if args.threshold is not None:
        custom_preds = (all_probs >= threshold).astype(int)
        metrics['custom_threshold'] = threshold
        metrics['acc_at_custom'] = float(np.mean(custom_preds == all_labels))

    token_stats = None
    if args.mode == 'digit':
        token_stats = per_token_analysis(test_ds, all_probs, threshold, idx_to_token)

    print_results(metrics, token_stats)

    preds = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    print(f'  Confusion matrix (threshold={threshold:.4f}):')
    print(f'               Predicted')
    print(f'              No-match  Match')
    print(f'  Actual No   {cm[0][0]:>7}  {cm[0][1]:>5}')
    print(f'  Actual Yes  {cm[1][0]:>7}  {cm[1][1]:>5}')
    print()

    if args.save:
        output = {
            'dataset': args.dataset,
            'mode': args.mode,
            'encoder': encoder,
            'model_path': model_path,
            'resample': resample,
            'standardized': feature_stats is not None,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
        }
        if token_stats:
            output['per_token'] = token_stats

        save_dir = os.path.dirname(model_path) or MODEL_ROOT
        out_path = os.path.join(save_dir, f'test_results_{args.mode}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'Results saved to {out_path}')
```

- [ ] **Step 2: Smoke run against Task 7 artifacts**

```bash
python test.py --dataset digit --mode digit --save \
  --model_path /tmp/lip2text_smoke/models/digit/transformer/best_digit_verifier.pt \
  --data-dir /tmp/lip2text_smoke/processed_data
```

Expected: prints TEST RESULTS table; `test_results_digit.json` written next to the checkpoint. Also run `--mode sequence` after a 1-epoch sequence training if a sequence checkpoint is not present:

```bash
python train.py --dataset digit --mode sequence --epochs 1 --batch_size 8 \
  --data-dir /tmp/lip2text_smoke/processed_data \
  --model-root /tmp/lip2text_smoke/models --log-root /tmp/lip2text_smoke/runs
python test.py --dataset digit --mode sequence \
  --data-dir /tmp/lip2text_smoke/processed_data \
  --model_path /tmp/lip2text_smoke/models/digit/transformer/best_sequence_verifier.pt
```

Expected: sequence metrics printed without shape/dim errors.

- [ ] **Step 3: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add test.py
git commit -m "feat: config-driven test.py with dataset scoping, GRID support, transforms"
```

---

### Task 9: `inference.py` — config-driven construction + transforms

**Files:**
- Modify: `inference.py` (imports `:24-25`, dirs `:41-57`, dead code `:322-329`, pad helpers `:332-343`, infer helpers `:374-438`, main `:442-664`)
- Test: `tests/test_inference_helpers.py` (new)

**Interfaces:**
- Consumes: `checkpoint.load_model_config`, `checkpoint.build_model_from_config`; `transforms.correct_lip_speed_fps`, `resample_segment`, `standardize_segment`, `load_feature_stats`; `SEG_LEN` not needed (config supplies `seg_len`).
- Produces:
  - `_candidate_model_paths(dataset, mode, encoder=None)` — new layout then legacy.
  - `_infer_legacy_seq2seq_lens(state_dict) -> tuple[int, int, int]` returning `(max_src_len, max_tgt_len, seg_embed_dim)` from `src_pos_emb.weight` / `tgt_pos_emb.weight` shapes (legacy fallback for the pos-emb crash).
  - `_apply_transforms(segments, fps, config, stats_dir) -> list[np.ndarray]` — applies fps-correct + resample + standardize per config; returns segments unchanged when config is None.
  - `infer_sequence(model, segments, tokens, token_to_idx, device, pad_len)`, `infer_per_digit(model, segments, tokens, token_to_idx, device, pad_len)`, `infer_seq2seq(model, segments, device, max_len, bos_idx, pad_idx, eos_idx, pad_len)` — pad_len threaded through.

- [ ] **Step 1: Write the failing tests** — create `tests/test_inference_helpers.py`:

```python
import os
import unittest

import numpy as np

from inference import _apply_transforms, _infer_legacy_seq2seq_lens
from model import TinyLipSeq2Seq


class InferLegacySeq2SeqLensTests(unittest.TestCase):
    def test_shapes_read_from_state_dict(self):
        model = TinyLipSeq2Seq(vocab_size=14, pad_idx=11, n_features=8,
                               seg_embed_dim=48, max_src_len=8, max_tgt_len=9,
                               encoder_type='transformer')
        src, tgt, dim = _infer_legacy_seq2seq_lens(model.state_dict())
        self.assertEqual((src, tgt, dim), (8, 9, 48))


class ApplyTransformsTests(unittest.TestCase):
    def test_none_config_returns_segments_unchanged(self):
        seg = np.random.rand(5, 8).astype(np.float32)
        out = _apply_transforms([seg.copy()], fps=25.0, config=None, stats_dir='/tmp')
        self.assertEqual(out[0].shape, seg.shape)
        np.testing.assert_array_equal(out[0], seg)

    def test_resample_only(self):
        seg = np.random.rand(5, 8).astype(np.float32)
        config = {'resample': True, 'standardized': False, 'seg_len': 16}
        out = _apply_transforms([seg], fps=25.0, config=config, stats_dir='/tmp')
        self.assertEqual(out[0].shape, (16, 8))

    def test_standardize_uses_stats_file(self):
        import json
        import tempfile

        seg = np.full((4, 2), 5.0, dtype=np.float32)
        config = {'resample': False, 'standardized': True, 'seg_len': 16,
                  'feature_names': ['a', 'b']}
        with tempfile.TemporaryDirectory() as tmp:
            stats = {'feature_names': ['a', 'b'], 'mean': [5.0, 0.0],
                     'std': [2.0, 1.0], 'n_frames': 10}
            with open(os.path.join(tmp, 'feature_stats.json'), 'w') as f:
                json.dump(stats, f)
            out = _apply_transforms([seg], fps=25.0, config=config, stats_dir=tmp)
        np.testing.assert_allclose(out[0][:, 0], 0.0, atol=1e-6)

    def test_standardize_missing_stats_raises(self):
        config = {'resample': False, 'standardized': True, 'seg_len': 16,
                  'feature_names': ['a', 'b']}
        with self.assertRaises(FileNotFoundError):
            _apply_transforms([np.ones((4, 2))], fps=25.0, config=config,
                              stats_dir='/nonexistent-dir')

    def test_feature_order_mismatch_raises(self):
        import json
        import tempfile

        config = {'resample': False, 'standardized': True, 'seg_len': 16,
                  'feature_names': ['a', 'b']}
        with tempfile.TemporaryDirectory() as tmp:
            stats = {'feature_names': ['b', 'a'], 'mean': [0.0, 0.0],
                     'std': [1.0, 1.0], 'n_frames': 10}
            with open(os.path.join(tmp, 'feature_stats.json'), 'w') as f:
                json.dump(stats, f)
            with self.assertRaises(RuntimeError):
                _apply_transforms([np.ones((4, 2))], fps=25.0, config=config, stats_dir=tmp)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name '_apply_transforms'`

- [ ] **Step 3: Implement `inference.py` changes**

1. Imports: add

```python
import checkpoint as ckpt
from transforms import correct_lip_speed_fps, load_feature_stats, resample_segment, standardize_segment
```

2. Replace `_dataset_model_dir`/`_default_model_path` (inference.py:41-53) with:

```python
def _candidate_model_paths(dataset, mode, encoder=None):
    """New layout first, then legacy layouts."""
    encoders = ([encoder] if encoder else []) + ['transformer', 'bigru']
    paths = [os.path.join(MODEL_DIR, dataset, enc, f'best_{mode}.pt' if mode == 'seq2seq'
                          else f'best_{mode}_verifier.pt') for enc in encoders]
    paths.append(os.path.join(MODEL_DIR, 'transformer_encoder',
                              f'best_{mode}.pt' if mode == 'seq2seq' else f'best_{mode}_verifier.pt'))
    if dataset == 'digit':
        paths.append(os.path.join(MODEL_DIR, f'best_{mode}.pt' if mode == 'seq2seq'
                                  else f'best_{mode}_verifier.pt'))
    else:
        paths.append(os.path.join(MODEL_DIR, dataset, f'best_{mode}.pt' if mode == 'seq2seq'
                                  else f'best_{mode}_verifier.pt'))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
```

`_default_vocab_path(dataset, encoder_type)` similarly returns candidates:

```python
def _default_vocab_path_candidates(dataset, encoder_type=None):
    encoders = ([encoder_type] if encoder_type else []) + ['transformer', 'bigru']
    paths = [os.path.join(MODEL_DIR, dataset, enc, 'vocab.json') for enc in encoders]
    paths.append(os.path.join(MODEL_DIR, 'transformer_encoder', 'vocab.json'))
    paths.append(os.path.join(MODEL_DIR, dataset, 'vocab.json'))
    if dataset == 'digit':
        paths.append(os.path.join(MODEL_DIR, 'vocab.json'))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
```

and the main-block vocab resolution (inference.py:519-531) picks the first existing candidate from this list instead of calling the old single-path `_default_vocab_path`.

3. Delete the unreachable duplicate block in `segment_by_lip_speed` (inference.py:322-329 — everything after the first `return segments`).

4. Add helpers after `infer_input_feature_dim`:

```python
def _infer_legacy_seq2seq_lens(state_dict):
    """(max_src_len, max_tgt_len, seg_embed_dim) from a legacy checkpoint."""
    src = state_dict.get('src_pos_emb.weight')
    tgt = state_dict.get('tgt_pos_emb.weight')
    max_src_len = int(src.shape[0]) if src is not None else 12
    max_tgt_len = int(tgt.shape[0]) if tgt is not None else 12
    seg_embed_dim = int(src.shape[1]) if src is not None else 48
    return max_src_len, max_tgt_len, seg_embed_dim


def _apply_transforms(segments, fps, config, stats_dir):
    """Apply config-declared fps-correct + resample + standardize to segments."""
    if config is None:
        return segments
    out = list(segments)
    if config.get('resample', False):
        out = [resample_segment(correct_lip_speed_fps(s, fps), config['seg_len'])
               for s in out]
    if config.get('standardized', False):
        stats = load_feature_stats(os.path.join(stats_dir, 'feature_stats.json'))
        if list(stats['feature_names']) != list(config['feature_names']):
            raise RuntimeError(
                'feature_stats.json feature order does not match checkpoint config'
            )
        out = [standardize_segment(s, stats['mean'], stats['std']) for s in out]
    return out
```

5. Thread `pad_len` through the three infer helpers — change their signatures and their `pad_segment(seg, MAX_SEQ_LEN)` calls to `pad_segment(seg, pad_len)`:

```python
def infer_sequence(model, segments, tokens, token_to_idx, device, pad_len):
def infer_per_digit(model, segments, tokens, token_to_idx, device, pad_len):
def infer_seq2seq(model, segments, device, max_len, bos_idx, pad_idx, eos_idx, pad_len):
```

6. Main-block changes:
   - model path resolution uses `_candidate_model_paths(args.dataset, args.mode, args.encoder if args.encoder != 'auto' else None)`.
   - after `state_dict = torch.load(...)` (inference.py:527) add:

```python
    config = ckpt.load_model_config(model_path, mode=args.mode)
    if resolved_encoder == 'auto':
        resolved_encoder = config['encoder_type'] if config else _detect_encoder_type(state_dict)
```

   - replace the model construction block (inference.py:599-625) with:

```python
    if config is not None:
        model = ckpt.build_model_from_config(config, DEVICE)
    elif args.mode == 'digit':
        model = DigitVerifier(n_classes=n_classes, embed_dim=EMBED_DIM,
                              n_features=n_features, hidden_dim=HIDDEN_DIM,
                              encoder_type=resolved_encoder).to(DEVICE)
    elif args.mode == 'sequence':
        model = SequenceVerifier(n_classes=n_classes, embed_dim=EMBED_DIM,
                                 n_features=n_features, hidden_dim=HIDDEN_DIM,
                                 encoder_type=resolved_encoder).to(DEVICE)
    else:
        legacy_src, legacy_tgt, legacy_dim = _infer_legacy_seq2seq_lens(state_dict)
        model = TinyLipSeq2Seq(
            vocab_size=seq2seq_vocab_size, pad_idx=pad_idx, n_features=n_features,
            seg_embed_dim=legacy_dim, n_heads=4, n_encoder_layers=1, n_decoder_layers=1,
            ff_dim=128, dropout=0.1, max_src_len=legacy_src, max_tgt_len=legacy_tgt,
            hidden_dim=64, encoder_type=resolved_encoder,
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()
```

   - after `segments = [adapt_feature_dim(seg, n_features) for seg in segments]` (inference.py:589) add:

```python
    stats_dir = os.path.join('processed_data', args.dataset)
    segments = _apply_transforms(segments, fps, config, stats_dir)
    pad_len = config['seg_len'] if config and config.get('resample') else MAX_SEQ_LEN
```

   and pass `pad_len` to the three `infer_*` calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference_helpers.py -v` then `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inference.py tests/test_inference_helpers.py
git commit -m "fix: config-driven inference with transforms and legacy pos-emb inference"
```

---

### Task 10: `preprocess.py` metadata (seq-length distribution, failure counts)

**Files:**
- Modify: `preprocess.py` (add `_summarize_samples`, extend metadata `:636-652`)
- Test: `tests/test_preprocess_metadata.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_summarize_samples(all_samples, failed_videos) -> dict` with keys `seq_len_dist` (`{str(len): count}`) and `failure_counts` (`{reason: count}`); metadata.json gains both keys.

- [ ] **Step 1: Write the failing tests** — create `tests/test_preprocess_metadata.py`:

```python
import unittest

from preprocess import _summarize_samples


class SummarizeSamplesTests(unittest.TestCase):
    def test_seq_len_dist_and_failure_counts(self):
        samples = [
            {'digit_sequence': ['1'] * 8},
            {'digit_sequence': ['2'] * 7},
            {'digit_sequence': ['3'] * 7},
        ]
        failed = [('a', 'video not found'), ('b', 'video not found'), ('c', 'empty alignment')]
        summary = _summarize_samples(samples, failed)
        self.assertEqual(summary['seq_len_dist'], {'7': 2, '8': 1})
        self.assertEqual(summary['failure_counts'],
                         {'video not found': 2, 'empty alignment': 1})

    def test_empty_inputs(self):
        self.assertEqual(_summarize_samples([], []),
                         {'seq_len_dist': {}, 'failure_counts': {}})


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess_metadata.py -v`
Expected: FAIL — `ImportError: cannot import name '_summarize_samples'`

- [ ] **Step 3: Implement** — add to `preprocess.py` above `main()`:

```python
def _summarize_samples(all_samples, failed_videos):
    """Sequence-length distribution and failure-reason counts for metadata."""
    seq_len_counts = defaultdict(int)
    for s in all_samples:
        seq_len_counts[len(s['digit_sequence'])] += 1
    failure_counts = defaultdict(int)
    for _, reason in failed_videos:
        failure_counts[reason] += 1
    return {
        'seq_len_dist': {str(k): int(v) for k, v in sorted(seq_len_counts.items())},
        'failure_counts': dict(sorted(failure_counts.items())),
    }
```

and extend the `metadata` dict (preprocess.py:636-650) with:

```python
        **_summarize_samples(all_samples, failed_videos),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add preprocess.py tests/test_preprocess_metadata.py
git commit -m "feat: record sequence-length distribution and failure counts in metadata"
```

---

### Task 11: README + final verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything landed in Tasks 1–10.
- Produces: documentation only.

- [ ] **Step 1: Update `README.md`**

After the "### 2) Train" code block, add:

```markdown
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
```

Update the "### 3) Evaluate" section commands to:

```bash
python test.py --dataset digit --mode sequence
python test.py --dataset grid --mode digit
python test.py --dataset digit --mode digit --save
```

- [ ] **Step 2: Full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (transforms 13, checkpoint 7+, datasets 10, preflight 7, inference helpers 5, preprocess 2)

- [ ] **Step 3: Preflight on real data**

```bash
python preflight.py --dataset digit --mode digit
python preflight.py --dataset digit --mode sequence
python preflight.py --dataset digit --mode seq2seq
```

Expected: all `OK` lines, exit 0. (GRID preflight only if `processed_data/grid/train.npz` is present on this machine; skip otherwise.)

- [ ] **Step 4: Verify legacy path byte-compatibility**

Run the Task 7 Step 9 synthetic-data generation again into `/tmp/lip2text_check`, then compare a dataset sample between old code (git stash is NOT used — instead checkout main in a temp worktree) and new code:

```bash
git worktree add -b verify-legacy-tmp /tmp/lip2text_main main
cd /tmp/lip2text_main && source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch && \
python - <<'EOF'
import numpy as np
from dataset import LipVerificationDataset
ds = LipVerificationDataset('/tmp/lip2text_check/processed_data/digit/train.npz', dataset='digit')
feat, mask, _, _ = ds[0]
np.save('/tmp/legacy_sample.npy', feat.numpy())
np.save('/tmp/legacy_mask.npy', mask.numpy())
EOF
cd /Users/turi/localgit/ts-lip2text && git worktree remove --force /tmp/lip2text_main && git branch -D verify-legacy-tmp
python - <<'EOF'
import numpy as np
from dataset import LipVerificationDataset
ds = LipVerificationDataset('/tmp/lip2text_check/processed_data/digit/train.npz',
                            dataset='digit', resample=False, feature_stats=None)
feat, mask, _, _ = ds[0]
np.testing.assert_array_equal(feat.numpy(), np.load('/tmp/legacy_sample.npy'))
np.testing.assert_array_equal(mask.numpy(), np.load('/tmp/legacy_mask.npy'))
print('legacy path byte-identical')
EOF
```

Expected: `legacy path byte-identical`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document pipeline flags, checkpoint layout, and eval usage"
```

- [ ] **Step 6: Final sanity**

```bash
git log --oneline main..methodology-fixes
python -m pytest tests/ -v
```

Expected: ~11 commits, all tests green. The branch is ready for the owner to retrain (digit locally, GRID on the server).

