import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
from typing import Dict
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class FakeWatchEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for frames, labels in dataloader:
                frames = frames.to(self.device)
                logits = self.model(frames)
                preds  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

        auc       = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        f1        = f1_score(all_labels, binary_preds, zero_division=0)
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall    = recall_score(all_labels, binary_preds, zero_division=0)
        cm        = confusion_matrix(all_labels, binary_preds)
        report    = classification_report(all_labels, binary_preds, target_names=["Real", "Fake"])

        logger.info(f"\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}")

        return {
            "auc":       round(auc, 4),
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        }

    def cross_dataset_eval(
        self,
        dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for dataset_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating on: {dataset_name}")
            metrics = self.evaluate(dataloader)
            results[dataset_name] = metrics
            logger.info(f"{dataset_name} Results: {metrics}")
        return results
