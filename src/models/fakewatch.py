import torch
import torch.nn as nn
from .spatial import SpatialExtractor
from .temporal import TemporalTransformer
from .fusion import FusionModule
from .classifier import DeepfakeClassifier


class FakeWatch(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        spatial_dropout: float = 0.3,
        temporal_hidden_dim: int = 512,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        temporal_dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        num_classes: int = 2,
        classifier_dropout: float = 0.3,
    ):
        super(FakeWatch, self).__init__()

        self.spatial = SpatialExtractor(
            pretrained=pretrained,
            dropout=spatial_dropout
        )
        self.temporal = TemporalTransformer(
            input_dim=self.spatial.feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
        )
        self.fusion = FusionModule(
            spatial_dim=self.spatial.feature_dim,
            temporal_dim=self.temporal.feature_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )
        self.classifier = DeepfakeClassifier(
            input_dim=self.fusion.feature_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, channels, height, width)
        spatial_features  = self.spatial(x)
        temporal_features = self.temporal(spatial_features)
        fused_features    = self.fusion(spatial_features, temporal_features)
        logits            = self.classifier(fused_features)
        return logits

    def get_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x)
