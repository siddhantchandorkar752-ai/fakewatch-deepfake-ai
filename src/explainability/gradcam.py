import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.spatial.backbone[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        x: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        self.model.eval()
        x.requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        score = logits[:, target_class].sum()
        score.backward()

        gradients  = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations.mean(dim=0)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(gradients):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (1 - alpha) * image + alpha * heatmap
        return np.uint8(overlay)
