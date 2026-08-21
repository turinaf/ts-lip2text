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
