import torch
import torch.nn as nn
import torch.nn.functional as F
from .lip_encoder import LipEncoder
from .text_encoder import TextEncoder
from .cross_modal_alignment import CrossModalAlignment
from .audio_encoder import AudioEncoder

class LipTextVerificationModel(nn.Module):
    def __init__(self, vocab_size=50, feature_dim=256, lip_backbone_pretrained=True):
        super(LipTextVerificationModel, self).__init__()
        self.lip_encoder = LipEncoder(feature_dim, backbone_pretrained=lip_backbone_pretrained)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, feature_dim=feature_dim)
        self.audio_encoder = AudioEncoder(feature_dim)
        
        self.alignment_lip_text = CrossModalAlignment(feature_dim)
        self.alignment_audio_text = CrossModalAlignment(feature_dim)
        self.alignment_lip_audio = CrossModalAlignment(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Learnable weights for combining the three alignment scores
        self.align_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))

    @staticmethod
    def _masked_mean(feats, mask):
        """Mean-pool over non-padding positions. mask: True = padding (B x T)."""
        if mask is None:
            return feats.mean(dim=1)
        valid = (~mask).float().unsqueeze(-1)  # B x T x 1
        return (feats * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, video_frames, phoneme_seqs, audio_features=None, video_padding_mask=None, phoneme_padding_mask=None):
        lip_feats = self.lip_encoder(video_frames, padding_mask=video_padding_mask)
        text_feats = self.text_encoder(phoneme_seqs, padding_mask=phoneme_padding_mask)
        
        if audio_features is not None:
            audio_feats = self.audio_encoder(audio_features)
            
            # Alignments
            align_lt = self.alignment_lip_text(
                lip_feats,
                text_feats,
                lip_padding_mask=video_padding_mask,
                text_padding_mask=phoneme_padding_mask
            )
            align_at = self.alignment_audio_text(audio_feats, text_feats)
            align_la = self.alignment_lip_audio(lip_feats, audio_feats, lip_padding_mask=video_padding_mask)
            
            lip_pool = self._masked_mean(lip_feats, video_padding_mask)
            text_pool = self._masked_mean(text_feats, phoneme_padding_mask)
            audio_pool = audio_feats.mean(dim=1)
            
            score_lt = align_lt["alignment_score"].unsqueeze(-1)
            score_at = align_at["alignment_score"].unsqueeze(-1)
            score_la = align_la["alignment_score"].unsqueeze(-1)
            
            # Combine features
            combined = torch.cat([lip_pool, text_pool, audio_pool, score_lt, score_at, score_la], dim=-1)
            
            consistency_score = self.classifier(combined)
            
            # Weighted alignment score with learned weights
            w = F.softmax(self.align_weights, dim=0)
            final_align_score = w[0] * align_lt["alignment_score"] + w[1] * align_at["alignment_score"] + w[2] * align_la["alignment_score"]
            align_out = {
                "alignment_score": final_align_score,
                "lip_text": align_lt,
                "audio_text": align_at,
                "lip_audio": align_la
            }
        else:
            audio_feats = torch.zeros(lip_feats.shape, device=lip_feats.device) # Dummy audio feats
            align_lt = self.alignment_lip_text(
                lip_feats,
                text_feats,
                lip_padding_mask=video_padding_mask,
                text_padding_mask=phoneme_padding_mask
            )
            
            lip_pool = self._masked_mean(lip_feats, video_padding_mask)
            text_pool = self._masked_mean(text_feats, phoneme_padding_mask)
            audio_pool = torch.zeros_like(lip_pool)
            
            score_lt = align_lt["alignment_score"].unsqueeze(-1)
            score_at = torch.zeros_like(score_lt)
            score_la = torch.zeros_like(score_lt)
            
            combined = torch.cat([lip_pool, text_pool, audio_pool, score_lt, score_at, score_la], dim=-1)
            consistency_score = self.classifier(combined)
            
            align_out = align_lt

        return consistency_score, align_out
