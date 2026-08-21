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
