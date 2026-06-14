import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AudioEncoder(nn.Module):
    def __init__(self, feature_dim=256, n_mels=80):
        super(AudioEncoder, self).__init__()
        # Expects precomputed log-mel spectrogram: B x T_mel x n_mels
        # (mel transform is applied in the dataset loader, not here)
        self.proj = nn.Linear(n_mels, feature_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(feature_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, log_mel):
        # log_mel: B x T_mel x n_mels
        proj_feats = self.proj(log_mel)
        proj_feats = self.positional_encoding(proj_feats)
        audio_feats = self.transformer(proj_feats)
        return audio_feats
