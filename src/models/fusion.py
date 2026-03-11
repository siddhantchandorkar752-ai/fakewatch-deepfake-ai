import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 1792,
        temporal_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super(FusionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.feature_dim = hidden_dim

    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        # spatial_features: (batch, frames, spatial_dim) -> mean pool
        spatial_pooled = spatial_features.mean(dim=1)
        combined = torch.cat([spatial_pooled, temporal_features], dim=1)
        fused = self.fusion(combined)
        return fused
