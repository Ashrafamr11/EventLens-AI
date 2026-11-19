
import os
import sys
from pathlib import Path

from fastapi import FastAPI

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eventlens.model_registry import get_model_catalog

DATA_ROOT = Path(
    os.environ.get(
        "EVENTLENS_DATA_ROOT",
        "/content/drive/MyDrive/EventLens-AI",
    )
)

app = FastAPI(
    title="EventLens AI API",
    description="Audio-Visual Event Detection and Localization",
    version="0.1.0",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "EventLens AI",
    }


@app.get("/api/models")
def list_models():
    return {
        "models": get_model_catalog(
            DATA_ROOT / "checkpoints"
        )
    }
