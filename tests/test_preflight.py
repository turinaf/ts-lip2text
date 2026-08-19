import os
import tempfile
import unittest

import numpy as np
import torch

from dataset import (
    FrameLevelTranscriptionDataset,
    LipTranscriptionDataset,
    LipVerificationDataset,
    SequenceVerificationDataset,
)
from model import DigitVerifier, FrameLevelLipSeq2Seq, SequenceVerifier, TinyLipSeq2Seq
from preflight import build_model, check_forward_pass, check_npz_file, dataset_paths
from test import _compute_wer_cer, _edit_distance, _tokens_to_text


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
            sequences = [['1'], ['3', '5', '7', '9', '2', '4', '6', '8']]
            segments = [
                [_make_segment(4)],
                [_make_segment(5), _make_segment(6), _make_segment(7), _make_segment(8), _make_segment(5), _make_segment(4), _make_segment(6), _make_segment(5)],
            ]
            _write_npz(npz_path, sequences, segments)

            ds = LipVerificationDataset(npz_path, dataset='digit')
            seq_ds = SequenceVerificationDataset(npz_path, dataset='digit')
            trn_ds = LipTranscriptionDataset(npz_path, dataset='digit')

            self.assertEqual(len(ds.digit_sequences), 2)
            self.assertEqual(len(seq_ds.digit_sequences), 2)
            self.assertEqual(len(trn_ds.digit_sequences), 2)
            self.assertEqual(ds.vocab_size, 11)
            self.assertEqual(trn_ds[0][0].shape[1:], (30, 8))

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

    def test_forward_pass_digit_mode_is_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['1'], ['3']]
            segments = [[_make_segment(4)], [_make_segment(5)]]
            _write_npz(npz_path, sequences, segments)

            check_forward_pass('digit', 'digit', npz_path)

    def test_build_model_smoke(self):
        digit_model = build_model('digit', 11, 8, 8)
        sequence_model = build_model('sequence', 11, 8, 8)
        seq2seq_model = build_model('seq2seq', 5, 8, 3)
        lipread_model = build_model('lipread', 11, 8, 8)

        self.assertIsInstance(digit_model, DigitVerifier)
        self.assertIsInstance(sequence_model, SequenceVerifier)
        self.assertIsInstance(seq2seq_model, TinyLipSeq2Seq)
        self.assertIsInstance(lipread_model, FrameLevelLipSeq2Seq)

    def test_frame_level_dataset_uses_full_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['1', '2', '3', '4', '5', '6', '7', '8'],
                         ['8', '7', '6', '5', '4', '3', '2', '1']]
            segments = [
                [_make_segment(5) for _ in range(8)],
                [_make_segment(6) for _ in range(8)],
            ]
            features = [
                np.concatenate([_make_segment(5) for _ in range(8)], axis=0),
                np.concatenate([_make_segment(6) for _ in range(8)], axis=0),
            ]
            np.savez_compressed(
                npz_path,
                digit_segments=np.array(segments, dtype=object),
                digit_sequences=np.array(sequences, dtype=object),
                full_features=np.array(features, dtype=object),
                feature_names=np.array([f'f{i}' for i in range(8)]),
            )

            ds = FrameLevelTranscriptionDataset(npz_path, dataset='digit')
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.vocab_size, 11)
            feat, tokens, n_tokens = ds[0]
            self.assertEqual(feat.shape[1], 8)
            self.assertEqual(n_tokens, 8)
            self.assertEqual(list(tokens.shape), [8])

    def test_lipread_forward_pass_is_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['1', '2', '3', '4', '5', '6', '7', '8']]
            segments = [[_make_segment(5) for _ in range(8)]]
            features = [np.concatenate([_make_segment(5) for _ in range(8)], axis=0)]
            np.savez_compressed(
                npz_path,
                digit_segments=np.array(segments, dtype=object),
                digit_sequences=np.array(sequences, dtype=object),
                full_features=np.array(features, dtype=object),
                feature_names=np.array([f'f{i}' for i in range(8)]),
            )

            check_forward_pass('digit', 'lipread', npz_path)

    def test_lipread_forward_pass_grid_is_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'sample.npz')
            sequences = [['BIN', 'BLUE', 'AT', 'F', 'TWO', 'NOW']]
            segments = [[_make_segment(5) for _ in range(6)]]
            features = [np.concatenate([_make_segment(5) for _ in range(6)], axis=0)]
            np.savez_compressed(
                npz_path,
                digit_segments=np.array(segments, dtype=object),
                digit_sequences=np.array(sequences, dtype=object),
                full_features=np.array(features, dtype=object),
                feature_names=np.array([f'f{i}' for i in range(8)]),
            )

            check_forward_pass('grid', 'lipread', npz_path)


class WerCerTests(unittest.TestCase):
    def test_edit_distance_basics(self):
        self.assertEqual(_edit_distance('', ''), 0)
        self.assertEqual(_edit_distance('abc', 'abc'), 0)
        self.assertEqual(_edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(_edit_distance('abc', 'ab'), 1)
        self.assertEqual(_edit_distance('ab', 'abc'), 1)
        self.assertEqual(_edit_distance('abc', 'axc'), 1)

    def test_edit_distance_works_on_token_lists(self):
        self.assertEqual(_edit_distance(['a', 'b', 'c'], ['a', 'b', 'c']), 0)
        self.assertEqual(_edit_distance(['a', 'b'], ['a']), 1)
        self.assertEqual(_edit_distance(['a'], ['a', 'b']), 1)

    def test_tokens_to_text(self):
        self.assertEqual(_tokens_to_text(['1', '2', '3'], 'digit'), '123')
        self.assertEqual(_tokens_to_text(['BIN', 'BLUE'], 'grid'), 'BIN BLUE')

    def test_wer_cer_digit_uses_cer_only(self):
        token_to_idx = {'0': 0, '1': 1, '2': 2}
        pad, eos = 3, 4
        pred_rows = [[0, 2, 2, pad, eos]]
        tgt_rows = [[0, 1, 2, eos]]
        metrics = _compute_wer_cer(pred_rows, tgt_rows, token_to_idx, 'digit')
        # "022" vs "012": one substitution, 3 reference chars.
        self.assertAlmostEqual(metrics['cer'], 1 / 3)
        self.assertNotIn('wer', metrics)

    def test_wer_cer_grid(self):
        token_to_idx = {'BIN': 0, 'BLUE': 1, 'AT': 2}
        # Word order swap: two edits over three reference words -> WER 2/3.
        metrics = _compute_wer_cer([[0, 2, 1]], [[0, 1, 2]], token_to_idx, 'grid')
        self.assertAlmostEqual(metrics['wer'], 2 / 3)
        self.assertIn('cer', metrics)

        # Single-word substitution: WER 1.0, CER 1/3.
        token_to_idx = {'CAT': 0, 'CAR': 1}
        metrics = _compute_wer_cer([[1]], [[0]], token_to_idx, 'grid')
        self.assertAlmostEqual(metrics['wer'], 1.0)
        self.assertAlmostEqual(metrics['cer'], 1 / 3)

    def test_wer_cer_aggregates_across_rows(self):
        token_to_idx = {'A': 0, 'B': 1}
        pred_rows = [[0], [0, 0]]
        tgt_rows = [[0], [0]]
        metrics = _compute_wer_cer(pred_rows, tgt_rows, token_to_idx, 'grid')
        # Insertion of one extra word: total edits 1, ref words 2.
        self.assertAlmostEqual(metrics['wer'], 0.5)
        # CER on space-joined text: "A A" vs "A" -> 2 char edits over 2 ref chars.
        self.assertAlmostEqual(metrics['cer'], 1.0)


if __name__ == '__main__':
    unittest.main()