import os
import tempfile
import unittest

import numpy as np
import torch

from dataset import (
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
)
from model import DigitVerifier, SequenceVerifier, TinyLipSeq2Seq
from preflight import build_dataset, build_model, check_forward_pass, check_npz_file, dataset_paths


def _make_segment(length: int, feature_dim: int = 8) -> np.ndarray:
    return np.random.rand(length, feature_dim).astype(np.float32)


def _write_npz(path: str, sequences, segments) -> None:
    np.savez_compressed(
        path,
        digit_segments=np.array(segments, dtype=object),
        digit_sequences=np.array(sequences, dtype=object),
        full_features=np.array([np.concatenate(sample, axis=0) for sample in segments], dtype=object),
        feature_names=np.array([f'f{i}' for i in range(8)]),
    )


class PreflightTests(unittest.TestCase):
    def test_dataset_paths_are_scoped(self):
        processed_dir, model_dir, log_dir = dataset_paths('digit')
        self.assertTrue(processed_dir.endswith(os.path.join('processed_data', 'digit')))
        self.assertTrue(model_dir.endswith(os.path.join('models', 'digit')))
        self.assertTrue(log_dir.endswith(os.path.join('runs', 'digit')))

    def test_digit_dataset_keeps_variable_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [
                ['1', '2', '3', '4', '5', '6', '7', '8'],
                ['3', '5', '7', '9', '2', '4', '6', '8'],
            ]
            segments = [
                [_make_segment(4), _make_segment(6), _make_segment(5), _make_segment(7),
                 _make_segment(8), _make_segment(5), _make_segment(4), _make_segment(6)],
                [_make_segment(5), _make_segment(6), _make_segment(7), _make_segment(8),
                 _make_segment(5), _make_segment(4), _make_segment(6), _make_segment(5)],
            ]
            _write_npz(npz_path, sequences, segments)

            ds = LipVerificationDataset(npz_path, dataset='digit')
            seq_ds = SequenceVerificationDataset(npz_path, dataset='digit')
            trn_ds = LipTranscriptionDataset(npz_path, dataset='digit')

            self.assertEqual(len(ds.digit_sequences), 2)
            self.assertEqual(len(seq_ds.digit_sequences), 2)
            self.assertEqual(len(trn_ds.digit_sequences), 2)
            self.assertEqual(ds.vocab_size, 11)
            self.assertEqual(trn_ds[0][0].shape[1:], (16, 8))

    def test_grid_dataset_uses_dynamic_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['HELLO', 'WORLD'], ['OPEN', 'THE', 'DOOR']]
            segments = [
                [_make_segment(4), _make_segment(5)],
                [_make_segment(3), _make_segment(4), _make_segment(5)],
            ]
            _write_npz(npz_path, sequences, segments)

            ds = SequenceVerificationDataset(npz_path, dataset='grid')
            trn_ds = LipTranscriptionDataset(npz_path, dataset='grid')

            self.assertEqual(ds.vocab_size, 5)
            self.assertEqual(trn_ds.vocab_size, 5)
            self.assertEqual(trn_ds.token_to_idx['DOOR'], 0)
            self.assertEqual(trn_ds.token_to_idx['WORLD'], 4)

    def test_check_npz_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'bad.npz')
            segments = [[_make_segment(4)]]
            full_features = np.array([np.array([[1.0, np.nan] + [0.0] * 6], dtype=np.float32)], dtype=object)
            np.savez_compressed(
                npz_path,
                digit_segments=np.array(segments, dtype=object),
                digit_sequences=np.array([['1']], dtype=object),
                full_features=full_features,
                feature_names=np.array([f'f{i}' for i in range(8)]),
            )

            with self.assertRaises(ValueError):
                check_npz_file(npz_path)

    def test_datasets_support_legacy_no_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['3', '5', '7', '9', '2', '4', '6', '8']]
            segments = [[_make_segment(5) for _ in range(8)]]
            _write_npz(npz_path, sequences, segments)

            ds = build_dataset('digit', 'digit', npz_path, resample=False)
            self.assertEqual(ds[0][0].shape, (30, 8))

    def test_forward_pass_digit_mode_is_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [
                ['1', '2', '3', '4', '5', '6', '7', '8'],
                ['3', '5', '7', '9', '2', '4', '6', '8'],
            ]
            segments = [
                [_make_segment(4), _make_segment(6), _make_segment(5), _make_segment(7),
                 _make_segment(8), _make_segment(5), _make_segment(4), _make_segment(6)],
                [_make_segment(5), _make_segment(6), _make_segment(7), _make_segment(8),
                 _make_segment(5), _make_segment(4), _make_segment(6), _make_segment(5)],
            ]
            _write_npz(npz_path, sequences, segments)

            check_forward_pass('digit', 'digit', npz_path)

    def test_build_model_smoke(self):
        digit_model = build_model('digit', 11, 8, 8)
        sequence_model = build_model('sequence', 11, 8, 8)
        seq2seq_model = build_model('seq2seq', 5, 8, 3)

        self.assertIsInstance(digit_model, DigitVerifier)
        self.assertIsInstance(sequence_model, SequenceVerifier)
        self.assertIsInstance(seq2seq_model, TinyLipSeq2Seq)


if __name__ == '__main__':
    unittest.main()