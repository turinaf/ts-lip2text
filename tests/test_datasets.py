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
                seg[:, 0] = 10.0  # constant feature-0 offset for exact comparison
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
