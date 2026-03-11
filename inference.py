import argparse
from src.deployment.inference import FakeWatchInference
from src.utils.logger import get_logger

logger = get_logger("inference_script")


def main():
    parser = argparse.ArgumentParser(description="FAKEWATCH Inference")
    parser.add_argument("--video",      type=str, required=True,  help="Path to video file")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Path to model checkpoint")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--export-onnx", action="store_true",     help="Export model to ONNX")
    args = parser.parse_args()

    engine = FakeWatchInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    if args.export_onnx:
        engine.export_onnx("fakewatch.onnx")
        logger.info("ONNX export complete!")
        return

    result = engine.predict(args.video)

    print("\n" + "="*50)
    print("       FAKEWATCH ANALYSIS RESULT")
    print("="*50)
    print(f"  Prediction : {result['prediction']}")
    print(f"  Fake Prob  : {result['fake_prob']:.4f}")
    print(f"  Real Prob  : {result['real_prob']:.4f}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
