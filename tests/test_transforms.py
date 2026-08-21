import os
import tempfile
import unittest

import numpy as np

from transforms import (
    compute_feature_stats,
    correct_lip_speed_fps,
    load_feature_stats,
    resample_segment,
    save_feature_stats,
    standardize_segment,
)


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


if __name__ == '__main__':
    unittest.main()
