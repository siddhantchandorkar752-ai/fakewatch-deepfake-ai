import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("trainer")


class FakeWatchTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_auc = 0.0
        self.patience_counter = 0

        self.optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
        )
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (frames, labels) in enumerate(dataloader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        return {
            "loss": total_loss / len(dataloader),
            "auc": auc,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for frames, labels in dataloader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        auc       = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        f1        = f1_score(all_labels, binary_preds, zero_division=0)
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall    = recall_score(all_labels, binary_preds, zero_division=0)

        return {
            "loss":      total_loss / len(dataloader),
            "auc":       auc,
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        path = self.checkpoint_dir / f"fakewatch_epoch{epoch}_auc{metrics['auc']:.4f}.pt"
        torch.save({
            "epoch":      epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":    metrics,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        epochs   = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.evaluate(val_loader)
            self.scheduler.step()

            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f}"
            )

            if val_metrics["auc"] > self.best_auc:
                self.best_auc = val_metrics["auc"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"New best AUC: {self.best_auc:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{patience}")
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break
