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
