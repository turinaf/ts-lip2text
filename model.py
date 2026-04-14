"""
Lip-Text Verification Models
-----------------------------
LipEncoder: 1D-CNN + BiGRU encoder for lip segment time series
DigitVerifier: per-digit verification (lip segment vs single digit)
SequenceVerifier: full 8-digit sequence verification
CTCLipReader: CTC-based model that aligns full video to digit sequence
              without requiring pre-segmentation
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
            nn.Linear(seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, segments, masks, claimed_digits):
        """
        segments: (B, 8, T, 5)
        masks: (B, 8, T)
        claimed_digits: (B, 8)
        Returns: (B, 1) sequence match logit
        """
        B, S, T, F = segments.shape
        segments_flat = segments.view(B * S, T, F)
        masks_flat = masks.view(B * S, T)
        lip_embs = self.lip_encoder(segments_flat, masks_flat).view(B, S, -1)
        digit_embs = self.digit_embedding(claimed_digits)
        combined = torch.cat([lip_embs, digit_embs], dim=2)
        per_digit_scores = self.digit_compare(combined).squeeze(-1)
        return self.seq_agg(per_digit_scores)


# CTC blank index is N_CLASSES (after all digit classes)
CTC_BLANK = N_CLASSES  # 11


class CTCLipReader(nn.Module):
    """
    CTC-based lip reader: takes a full unsegmented video feature sequence
    and outputs per-frame logits over the vocabulary + CTC blank.
    Learns to align frames to digits internally — no pre-segmentation needed.

    Architecture:
        Conv1D x2 → BiGRU (2 layers) → per-frame Linear → (N_CLASSES + 1) logits

    For verification: decode predicted digits, compare with claimed sequence.
    """
    def __init__(self, n_features=5, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(128, hidden_dim, num_layers=n_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, N_CLASSES + 1)  # +1 for CTC blank

    def forward(self, x, lengths=None):
        """
        x: (B, T, n_features) — full video features, padded
        lengths: (B,) — actual frame counts before padding
        Returns: (B, T, N_CLASSES+1) log-probabilities per frame
        """
        h = self.conv(x.permute(0, 2, 1))   # (B, 128, T)
        h = h.permute(0, 2, 1)               # (B, T, 128)

        if lengths is not None:
            h = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False)

        h, _ = self.gru(h)

        if lengths is not None:
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        return self.fc(h)                     # (B, T, N_CLASSES+1)

    def decode_greedy(self, log_probs):
        """
        Greedy CTC decode: collapse repeated chars and remove blanks.
        log_probs: (T, N_CLASSES+1) for a single sequence
        Returns: list of decoded digit indices
        """
        preds = log_probs.argmax(dim=-1)      # (T,)
        decoded = []
        prev = CTC_BLANK
        for p in preds:
            p = p.item()
            if p != prev and p != CTC_BLANK:
                decoded.append(p)
            prev = p
        return decoded

    def decode_batch(self, log_probs, lengths):
        """
        Greedy decode for a batch.
        log_probs: (B, T, C)
        lengths: (B,)
        Returns: list of lists of digit indices
        """
        results = []
        for i in range(log_probs.size(0)):
            seq_len = lengths[i].item() if lengths is not None else log_probs.size(1)
            results.append(self.decode_greedy(log_probs[i, :seq_len]))
        return results
