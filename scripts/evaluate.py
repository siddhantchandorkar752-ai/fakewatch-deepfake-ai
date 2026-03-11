import argparse
import torch
from src.models import FakeWatch
from src.data import get_dataloader
from src.evaluation import FakeWatchEvaluator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="FAKEWATCH Cross-Dataset Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = FakeWatch(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from {args.checkpoint}")

    evaluator = FakeWatchEvaluator(model=model, device=device)

    datasets = config["data"]["datasets"]
    dataloaders = {}

    for dataset_name in datasets:
        try:
            loader = get_dataloader(
                root_dir=f"datasets/{dataset_name}",
                split="test",
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"]["num_workers"],
                num_frames=config["data"]["num_frames"],
                image_size=config["data"]["image_size"],
            )
            dataloaders[dataset_name] = loader
            logger.info(f"Loaded test set: {dataset_name}")
        except Exception as e:
            logger.warning(f"Skipping {dataset_name}: {e}")

    results = evaluator.cross_dataset_eval(dataloaders)

    print("\n" + "="*60)
    print("       FAKEWATCH CROSS-DATASET EVALUATION RESULTS")
    print("="*60)
    for dataset, metrics in results.items():
        print(f"\n  Dataset : {dataset.upper()}")
        print(f"  AUC     : {metrics['auc']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall  : {metrics['recall']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
