import torch
import random
import numpy as np
from src.models import FakeWatch
from src.data import get_dataloader
from src.training import FakeWatchTrainer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = load_config("configs/config.yaml")
    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader = get_dataloader(
        root_dir="datasets/celebdf",
        split="train",
        batch_size=8,
        num_workers=0,
        num_frames=config["data"]["num_frames"],
        image_size=config["data"]["image_size"],
    )
    val_loader = get_dataloader(
        root_dir="datasets/celebdf",
        split="val",
        batch_size=8,
        num_workers=0,
        num_frames=config["data"]["num_frames"],
        image_size=config["data"]["image_size"],
    )

    model = FakeWatch(
        pretrained=config["model"]["spatial"]["pretrained"],
        spatial_dropout=config["model"]["spatial"]["dropout"],
        temporal_hidden_dim=config["model"]["temporal"]["hidden_dim"],
        temporal_num_heads=config["model"]["temporal"]["num_heads"],
        temporal_num_layers=config["model"]["temporal"]["num_layers"],
        temporal_dropout=config["model"]["temporal"]["dropout"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        fusion_dropout=config["model"]["fusion"]["dropout"],
        num_classes=config["model"]["classifier"]["num_classes"],
    )
    logger.info("FakeWatch model initialized")

    trainer = FakeWatchTrainer(
        model=model,
        config=config,
        device=device,
    )
    trainer.fit(train_loader, val_loader)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
