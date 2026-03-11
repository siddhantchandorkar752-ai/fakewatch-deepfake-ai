import torch
import torch.nn as nn


class DeepfakeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super(DeepfakeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
