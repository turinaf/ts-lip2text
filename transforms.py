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
