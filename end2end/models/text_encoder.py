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

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=50, feature_dim=256):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, feature_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(feature_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, phoneme_seqs, padding_mask=None):
        # phoneme_seqs: B x L
        emb = self.embedding(phoneme_seqs) # B x L x D
        emb = self.positional_encoding(emb)
        text_feats = self.transformer(emb, src_key_padding_mask=padding_mask) # B x L x D
        return text_feats