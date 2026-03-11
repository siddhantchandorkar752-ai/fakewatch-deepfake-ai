import gradio as gr
import numpy as np
import cv2
from src.deployment.inference import FakeWatchInference
from src.utils.logger import get_logger
import os

logger = get_logger("app")

engine = None
checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
if os.path.exists(checkpoint_path):
    engine = FakeWatchInference(checkpoint_path=checkpoint_path)
    logger.info("Inference engine loaded")


def analyze(video_path: str):
    if engine is None:
        return "Model not loaded — train first or provide checkpoint.", None

    result = engine.predict(video_path)

    if result["status"] == "error":
        return f"Error: {result['message']}", None

    label      = result["prediction"]
    fake_prob  = result["fake_prob"]
    real_prob  = result["real_prob"]
    confidence = result["confidence"]
    cam        = result["gradcam"]

    output_text = f"""
╔══════════════════════════════════════╗
        FAKEWATCH ANALYSIS
╠══════════════════════════════════════╣
  Verdict    : {label}
  Fake Prob  : {fake_prob:.4f}
  Real Prob  : {real_prob:.4f}
  Confidence : {confidence:.4f}
╚══════════════════════════════════════╝
"""

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return output_text, heatmap


with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
    # 🕵️ FAKEWATCH — Deepfake Forensic Intelligence System
    ### Upload a video to detect if it is REAL or FAKE using AI
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
        with gr.Column():
            result_text = gr.Textbox(label="Analysis Result", lines=10)
            gradcam_img = gr.Image(label="GradCAM Heatmap — Forgery Region")

    analyze_btn.click(
        fn=analyze,
        inputs=video_input,
        outputs=[result_text, gradcam_img],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
