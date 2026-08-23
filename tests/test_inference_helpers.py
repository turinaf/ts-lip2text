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
