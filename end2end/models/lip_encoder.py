import torch
import torch.nn as nn
import torchvision.models as models


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LipEncoder(nn.Module):
    def __init__(self, feature_dim=256, backbone_pretrained=True):
        super(LipEncoder, self).__init__()
        
        # 1. 3D Spatio-Temporal Front-End
        # Captures short-term dynamics. Stride=1 for time to keep sequence length T intact.
        self.frontend3D = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), 
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 2. 2D Spatial Backbone (ResNet18)
        # We strip the default conv1 and maxpool from ResNet since our 3D frontend does that now
        try:
            weights = models.ResNet18_Weights.DEFAULT if backbone_pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            # Fallback when pretrained weights are unavailable in offline setups.
            resnet = models.resnet18(weights=None)
        self.resnet2D = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(512, feature_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(feature_dim)
        
        # 3. Temporal Back-End
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, x, padding_mask=None):
        # x: B x T x C x H x W
        B, T, C, H, W = x.shape
        
        # PyTorch Conv3d expects inputs of shape (B, C, Time, H, W)
        x = x.transpose(1, 2) 
        
        # Spatio-Temporal feature extraction
        x = self.frontend3D(x) # Output: B x 64 x T x H' x W'
        
        # Prepare for 2D ResNet: Merge Batch and Time
        B, C_out, T, H_out, W_out = x.shape
        x = x.transpose(1, 2).contiguous().view(B * T, C_out, H_out, W_out)
        
        # Deep Spatial extraction
        spatial_feats = self.resnet2D(x) # (B*T) x 512 x 1 x 1
        spatial_feats = spatial_feats.view(B * T, -1) # (B*T) x 512
        
        # Projection
        spatial_feats = self.fc(spatial_feats) # (B*T) x feature_dim
        spatial_feats = spatial_feats.view(B, T, -1) # B x T x feature_dim
        spatial_feats = self.positional_encoding(spatial_feats)
        
        # Long-term Context over sequence
        temporal_feats = self.temporal_encoder(spatial_feats, src_key_padding_mask=padding_mask) # B x T x feature_dim
        
        return temporal_feats
