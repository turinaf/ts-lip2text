import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    def __init__(self, feature_dim=256):
        super(CrossModalAlignment, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        # Learned projection to make alignment score more discriminative
        self.align_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, lip_feats, text_feats, lip_padding_mask=None, text_padding_mask=None):
        # lip_feats: B x T x D
        # text_feats: B x L x D

        # Cosine similarity matrix between all lip frames and all phonemes
        lip_norm = F.normalize(lip_feats, dim=-1)
        text_norm = F.normalize(text_feats, dim=-1)
        sim_matrix = torch.bmm(lip_norm, text_norm.transpose(1, 2))  # B x T x L

        if lip_padding_mask is not None:
            sim_matrix = sim_matrix.masked_fill(lip_padding_mask.unsqueeze(-1), -1e9)
        if text_padding_mask is not None:
            sim_matrix = sim_matrix.masked_fill(text_padding_mask.unsqueeze(1), -1e9)

        # Cross-attention: each text/phoneme position attends over lip frames
        attn_out, attn_weights = self.cross_attention(
            query=text_feats,
            key=lip_feats,
            value=lip_feats,
            key_padding_mask=lip_padding_mask
        )

        # Soft-max pooling over lip dimension — smoother gradients than hard argmax
        word_scores = torch.logsumexp(sim_matrix, dim=1)  # B x L

        # Alignment score via learned projection on concatenated features
        # This is more discriminative than pure cosine similarity
        align_input = torch.cat([attn_out, text_feats], dim=-1)  # B x L x 2D
        attn_scores = self.align_proj(align_input).squeeze(-1)   # B x L

        if text_padding_mask is not None:
            valid_word_mask = (~text_padding_mask).float()
            denom = valid_word_mask.sum(dim=-1).clamp(min=1.0)
            alignment_raw = (attn_scores * valid_word_mask).sum(dim=-1) / denom
        else:
            alignment_raw = attn_scores.mean(dim=-1)

        alignment_score = torch.sigmoid(alignment_raw)

        return {
            "alignment_score": alignment_score,
            "word_scores": word_scores,
            "sim_matrix": sim_matrix,
            "attn_weights": attn_weights,
            "aligned_features": attn_out
        }