"""
Lip-Text Verification Models
-----------------------------
LipEncoder: 1D-CNN + lightweight encoder for lip segment time series
DigitVerifier: per-digit verification (lip segment vs single digit)
SequenceVerifier: full 8-digit sequence verification
"""
import math

import torch
import torch.nn as nn


# Vocabulary: digits 0-9 plus "!" (alternate pronunciation of 1)
VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!']
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
N_CLASSES = len(VOCAB)  # 11


class LipEncoder(nn.Module):
    """Encode a lip segment time series into a fixed-size embedding."""

    def __init__(
        self,
        n_features=8,
        hidden_dim=128,
        embed_dim=64,
        encoder_type='transformer',
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=128,
        max_frames=256,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),          # dilation=1
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),      # dilation=2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),      # dilation=4
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        encoder_output_dim = hidden_dim if encoder_type == 'transformer' else hidden_dim * 2
        self.fc = nn.Linear(encoder_output_dim, embed_dim)
        self.attn = nn.Linear(encoder_output_dim, 1)

        if encoder_type == 'transformer':
            self.input_proj = nn.Linear(64, hidden_dim)
            self.pos_emb = nn.Embedding(max_frames, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
                activation='gelu',
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        elif encoder_type == 'bigru':
            self.gru = nn.GRU(64, hidden_dim, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type}')

    def _masked_attention_pool(self, h, mask):
        valid = mask.bool()
        attn_w = self.attn(h).squeeze(-1)
        attn_w = attn_w.masked_fill(~valid, -1e9)
        attn_w = torch.softmax(attn_w, dim=1)
        attn_w = attn_w * valid.float()
        attn_w = attn_w / attn_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (h * attn_w.unsqueeze(-1)).sum(dim=1)

    def _encode_transformer(self, h, mask):
        valid = mask.bool()
        h = self.input_proj(h)
        pos = torch.arange(h.size(1), device=h.device).unsqueeze(0)
        h = h + self.pos_emb(pos)

        encoded = torch.zeros_like(h)
        valid_rows = valid.any(dim=1)
        if valid_rows.any():
            encoded_valid = self.encoder(
                h[valid_rows],
                src_key_padding_mask=~valid[valid_rows],
            )
            encoded[valid_rows] = encoded_valid
        return encoded

    def forward(self, x, mask):
        """
        x: (B, T, F) lip features
        mask: (B, T) valid frame mask
        """
        h = self.conv(x.permute(0, 2, 1))   # (B, 64, T)
        h = h.permute(0, 2, 1)               # (B, T, 64)
        if self.encoder_type == 'transformer':
            h = self._encode_transformer(h, mask)
        else:
            h, _ = self.gru(h)                   # (B, T, hidden*2)

        # Safe masked attention pooling.
        # For all-padded segments, this yields a zero vector instead of NaN.
        h = self._masked_attention_pool(h, mask)
        return self.fc(h)                     # (B, embed_dim)


class DigitVerifier(nn.Module):
    """Per-digit verification: compare lip embedding with digit/char embedding."""
    def __init__(
        self,
        n_classes=N_CLASSES,
        embed_dim=64,
        n_features=8,
        hidden_dim=128,
        encoder_type='transformer',
    ):
        super().__init__()
        self.lip_encoder = LipEncoder(
            n_features,
            hidden_dim,
            embed_dim,
            encoder_type=encoder_type,
        )
        self.digit_embedding = nn.Embedding(n_classes, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, lip_features, mask, claimed_digit):
        """
        lip_features: (B, T, 5)
        mask: (B, T)
        claimed_digit: (B, 1)
        Returns: (B, 1) match logit
        """
        lip_emb = self.lip_encoder(lip_features, mask)
        digit_emb = self.digit_embedding(claimed_digit.squeeze(1))
        diff = lip_emb - digit_emb
        prod = lip_emb * digit_emb
        combined = torch.cat([lip_emb, digit_emb, diff, prod], dim=1)
        return self.classifier(combined)


class SequenceVerifier(nn.Module):
    """
    Full-sequence verification: verify all 8 digits at once.
    Encodes each segment, compares with claimed digit, aggregates scores.
    """
    def __init__(self, n_classes=N_CLASSES, embed_dim=64,
                 n_features=8, hidden_dim=128, encoder_type='transformer'):
        super().__init__()
        self.lip_encoder = LipEncoder(
            n_features,
            hidden_dim,
            embed_dim,
            encoder_type=encoder_type,
        )
        self.digit_embedding = nn.Embedding(n_classes, embed_dim)
        self.digit_compare = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.seq_agg = nn.GRU(1, 32, batch_first=True)
        self.seq_out = nn.Linear(32, 1)

    def forward(self, segments, masks, claimed_digits, seq_mask=None):
        """
        segments: (B, S, T, 5)
        masks: (B, S, T)
        claimed_digits: (B, S)
        seq_mask: (B, S) — 1 for real digits, 0 for padding. Optional.
        Returns: (B, 1) sequence match logit
        """
        B, S, T, F = segments.shape
        segments_flat = segments.view(B * S, T, F)
        masks_flat = masks.view(B * S, T)
        lip_embs = self.lip_encoder(segments_flat, masks_flat).view(B, S, -1)
        digit_embs = self.digit_embedding(claimed_digits)
        combined = torch.cat([lip_embs, digit_embs], dim=2)
        per_digit_scores = self.digit_compare(combined).squeeze(-1)  # (B, S)

        # Masked mean pooling over digits
        if seq_mask is not None:
            masked_scores = per_digit_scores * seq_mask
            pooled = masked_scores.sum(dim=1, keepdim=True) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            masked_scores = per_digit_scores
            pooled = per_digit_scores.mean(dim=1, keepdim=True)

        scores_input = masked_scores.unsqueeze(-1)  # (B, S, 1)
        _, h = self.seq_agg(scores_input)                 # h: (1, B, 32)
        # Combine sequence dynamics with masked pooled evidence.
        return self.seq_out(h.squeeze(0)) + pooled


class TinyLipSeq2Seq(nn.Module):
    """Small Transformer encoder-decoder for lip sequence transcription."""

    def __init__(
        self,
        vocab_size,
        pad_idx,
        n_features=8,
        seg_embed_dim=48,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ff_dim=128,
        dropout=0.1,
        max_src_len=12,
        max_tgt_len=12,
        hidden_dim=64,
        encoder_type='transformer',
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.seg_encoder = LipEncoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            embed_dim=seg_embed_dim,
            encoder_type=encoder_type,
        )
        self.src_pos_emb = nn.Embedding(max_src_len, seg_embed_dim)
        self.tgt_tok_emb = nn.Embedding(vocab_size, seg_embed_dim)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, seg_embed_dim)

        self.transformer = nn.Transformer(
            d_model=seg_embed_dim,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        # MPS currently lacks the nested-tensor op used by the encoder fast path.
        # Keep masking behavior but force the regular code path on Apple Silicon.
        if torch.backends.mps.is_available():
            self.transformer.encoder.enable_nested_tensor = False
            self.transformer.encoder.use_nested_tensor = False
        self.out = nn.Linear(seg_embed_dim, vocab_size)

    @staticmethod
    def _causal_mask(length, device):
        mask = torch.full((length, length), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1)

    def encode(self, segments, masks, src_key_padding_mask):
        """
        segments: (B, S, T, F)
        masks: (B, S, T)
        src_key_padding_mask: (B, S), True means padded position
        """
        bsz, src_len, t_len, n_feat = segments.shape
        seg_flat = segments.view(bsz * src_len, t_len, n_feat)
        mask_flat = masks.view(bsz * src_len, t_len)
        src = self.seg_encoder(seg_flat, mask_flat).view(bsz, src_len, -1)

        src_pos = torch.arange(src_len, device=segments.device).unsqueeze(0)
        src = src + self.src_pos_emb(src_pos)
        return self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, memory, src_key_padding_mask, tgt_in):
        """
        memory: (B, S, D)
        tgt_in: (B, L)
        """
        tgt_len = tgt_in.shape[1]
        tgt_pos = torch.arange(tgt_len, device=tgt_in.device).unsqueeze(0)
        tgt = self.tgt_tok_emb(tgt_in) * math.sqrt(self.tgt_tok_emb.embedding_dim)
        tgt = tgt + self.tgt_pos_emb(tgt_pos)

        tgt_mask = self._causal_mask(tgt_len, tgt_in.device)
        tgt_key_padding_mask = tgt_in.eq(self.pad_idx)

        dec = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(dec)

    def forward(self, segments, masks, src_key_padding_mask, tgt_in):
        memory = self.encode(segments, masks, src_key_padding_mask)
        return self.decode(memory, src_key_padding_mask, tgt_in)

    def greedy_decode(self, segments, masks, src_key_padding_mask, bos_idx, max_len):
        memory = self.encode(segments, masks, src_key_padding_mask)
        bsz = segments.shape[0]
        ys = torch.full((bsz, 1), bos_idx, dtype=torch.long, device=segments.device)

        for _ in range(max_len):
            logits = self.decode(memory, src_key_padding_mask, ys)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

        return ys[:, 1:]
