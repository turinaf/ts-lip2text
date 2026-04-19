"""
Lip-Text Verification Models
-----------------------------
LipEncoder: 1D-CNN + BiGRU encoder for lip segment time series
DigitVerifier: per-digit verification (lip segment vs single digit)
SequenceVerifier: full 8-digit sequence verification
"""
import torch
import torch.nn as nn


# Vocabulary: digits 0-9 plus "!" (alternate pronunciation of 1)
VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!']
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
N_CLASSES = len(VOCAB)  # 11


class LipEncoder(nn.Module):
    """Encode a lip segment time series into a fixed-size embedding."""
    def __init__(self, n_features=5, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x, mask):
        """
        x: (B, T, F) lip features
        mask: (B, T) valid frame mask
        """
        h = self.conv(x.permute(0, 2, 1))   # (B, 64, T)
        h = h.permute(0, 2, 1)               # (B, T, 64)
        h, _ = self.gru(h)                   # (B, T, hidden*2)

        # Masked mean pooling
        mask_exp = mask.unsqueeze(-1)         # (B, T, 1)
        h = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
        return self.fc(h)                     # (B, embed_dim)


class DigitVerifier(nn.Module):
    """Per-digit verification: compare lip embedding with digit/char embedding."""
    def __init__(self, n_classes=N_CLASSES, embed_dim=64, n_features=5, hidden_dim=128):
        super().__init__()
        self.lip_encoder = LipEncoder(n_features, hidden_dim, embed_dim)
        self.digit_embedding = nn.Embedding(n_classes, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
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
        combined = torch.cat([lip_emb, digit_emb], dim=1)
        return self.classifier(combined)


class SequenceVerifier(nn.Module):
    """
    Full-sequence verification: verify all 8 digits at once.
    Encodes each segment, compares with claimed digit, aggregates scores.
    """
    def __init__(self, n_classes=N_CLASSES, embed_dim=64, seq_len=8,
                 n_features=5, hidden_dim=128):
        super().__init__()
        self.lip_encoder = LipEncoder(n_features, hidden_dim, embed_dim)
        self.digit_embedding = nn.Embedding(n_classes, embed_dim)
        self.digit_compare = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.seq_agg = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

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
            per_digit_scores = per_digit_scores * seq_mask
            pooled = per_digit_scores.sum(dim=1, keepdim=True) / seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = per_digit_scores.mean(dim=1, keepdim=True)

        return self.seq_agg(pooled)
