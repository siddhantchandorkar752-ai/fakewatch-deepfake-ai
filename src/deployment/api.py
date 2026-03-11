import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from src.deployment.inference import FakeWatchInference
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="FAKEWATCH API",
    description="Deepfake Forensic Intelligence System — Production API",
    version="2.0"
)

inference_engine: Optional[FakeWatchInference] = None


@app.on_event("startup")
async def startup_event():
    global inference_engine
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt")
    if os.path.exists(checkpoint_path):
        inference_engine = FakeWatchInference(checkpoint_path=checkpoint_path)
        logger.info("Inference engine loaded successfully")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")


@app.get("/")
def root():
    return {"message": "FAKEWATCH API is running", "version": "2.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
    }


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    allowed = {".mp4", ".avi", ".mov"}
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        result = inference_engine.predict(tmp_path)
        result.pop("gradcam", None)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
