import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class SpatialExtractor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(SpatialExtractor, self).__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b4(weights=weights)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.feature_dim = 1792

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, channels, height, width)
        batch, frames, c, h, w = x.shape
        x = x.view(batch * frames, c, h, w)
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(batch * frames, -1)
        features = self.dropout(features)
        features = features.view(batch, frames, self.feature_dim)
        return features
