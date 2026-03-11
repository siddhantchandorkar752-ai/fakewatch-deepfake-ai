import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.models import FakeWatch
from src.data.preprocess import extract_frames
from src.explainability.gradcam import GradCAM
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("inference")


class FakeWatchInference:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load_model(checkpoint_path)
        self.gradcam = GradCAM(self.model)
        logger.info(f"FakeWatch loaded on {self.device}")

    def _load_model(self, checkpoint_path: str) -> FakeWatch:
        model = FakeWatch(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info(f"Model loaded from {checkpoint_path}")
        return model

    def predict(self, video_path: str) -> Dict:
        try:
            frames = extract_frames(
                video_path,
                num_frames=self.config["data"]["num_frames"],
                image_size=self.config["data"]["image_size"],
            )
            frames = frames.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(frames)
                probs  = torch.softmax(logits, dim=1)[0]
                fake_prob = probs[1].item()
                real_prob = probs[0].item()
                prediction = "FAKE" if fake_prob >= 0.5 else "REAL"

            cam = self.gradcam.generate(frames, target_class=1)

            return {
                "prediction":  prediction,
                "fake_prob":   round(fake_prob, 4),
                "real_prob":   round(real_prob, 4),
                "confidence":  round(max(fake_prob, real_prob), 4),
                "gradcam":     cam,
                "status":      "success",
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "prediction": "ERROR",
                "status":     "error",
                "message":    str(e),
            }

    def export_onnx(self, output_path: str = "fakewatch.onnx") -> None:
        dummy_input = torch.zeros(
            1,
            self.config["data"]["num_frames"],
            3,
            self.config["data"]["image_size"],
            self.config["data"]["image_size"],
        ).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=14,
            input_names=["video_frames"],
            output_names=["logits"],
        )
        logger.info(f"ONNX model exported to {output_path}")
