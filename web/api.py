"""EventLens AI: FastAPI application + native web frontend.

All predictions are computed locally with PyTorch; no external AI APIs.
"""
from __future__ import annotations
import io
import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eventlens.model_comparison import compare_models
from eventlens.model_registry import get_model_catalog
from eventlens.prediction import validate_features

logger = logging.getLogger("eventlens")
ARTIFACT_ROOT = Path(
    os.environ.get("EVENTLENS_ARTIFACT_ROOT") or os.environ.get("EVENTLENS_DATA_ROOT") or str(PROJECT_ROOT)
).expanduser().resolve()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # guard against uncontrolled memory consumption
MAX_NPZ_BYTES = 15 * 1024 * 1024

app = FastAPI(title="EventLens AI", description="Audio-Visual Event Localization", version="1.0.0")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "web/static"), name="static")


class ComparisonRequest(BaseModel):
    audio: list[list[float]]
    visual: list[list[float]]
    motion: list[list[float]]
    model_ids: list[str] | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def parse_model_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values or len(values) > 4:
        raise ValueError("Choose one to four unique model IDs")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate model IDs")
    return values


def parse_npz(contents: bytes) -> dict:
    try:
        with np.load(io.BytesIO(contents), allow_pickle=False) as archive:
            return validate_features({key: archive[key] for key in ["audio", "visual", "motion"]})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid feature NPZ: {exc}") from exc


def compare_or_http(features: dict, model_ids: list[str] | None, threshold: float):
    try:
        return compare_models(features, ARTIFACT_ROOT, model_ids=model_ids, threshold=threshold)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "web/static/index.html")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "EventLens AI", "version": "1.0.0"}


@app.get("/api/models")
def list_models():
    return {"models": get_model_catalog(ARTIFACT_ROOT)}


@app.get("/api/status")
def status():
    root = ARTIFACT_ROOT
    assets = {
        "rgb_weights": (root / "weights/resnet152-b121ed2d.pth").is_file(),
        "motion_weights": (root / "weights/r2plus1d18-kinetics400-v1.pth").is_file(),
        "ffmpeg": __import__("shutil").which("ffmpeg") is not None,
        "ffprobe": __import__("shutil").which("ffprobe") is not None,
    }
    try:
        from eventlens.extractor_runtime import resolve_vggish_archive
        resolve_vggish_archive(root, os.environ.get("EVENTLENS_VGGISH_ARCHIVE"))
        assets["vggish_archive"] = True
    except FileNotFoundError:
        assets["vggish_archive"] = False
    try:
        import tensorflow  # noqa F401
        import tf_slim  # noqa F401
        assets["tensorflow_vggish"] = True
    except ImportError:
        assets["tensorflow_vggish"] = False
    catalog = get_model_catalog(root)
    return {
        "models": catalog,
        "assets": assets,
        "ready_for_features": any(m["available"] for m in catalog),
        "ready_for_video": any(m["available"] for m in catalog) and all(assets.values()),
        "video_duration_seconds": 10,
        "evaluation_threshold": 0.5,
    }


@app.post("/api/compare-features")
def compare_features(request: ComparisonRequest):
    return compare_or_http(
        {"audio": request.audio, "visual": request.visual, "motion": request.motion},
        request.model_ids, request.threshold,
    )


@app.post("/api/analyze-features")
async def analyze_features(
    file: Annotated[UploadFile, File(description="NPZ with audio, visual and motion arrays")],
    model_ids: Annotated[str | None, Form()] = None,
    threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
):
    try:
        ids = parse_model_ids(model_ids)
        contents = await file.read(MAX_NPZ_BYTES + 1)
        if len(contents) > MAX_NPZ_BYTES:
            raise HTTPException(status_code=413, detail="Features file too large")
        features = parse_npz(contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
    result = compare_or_http(features, ids, threshold)
    result["input_type"] = "precomputed_features"
    return result


@app.post("/api/analyze-video")
async def analyze_video(
    file: Annotated[UploadFile, File(description="Video with original audio; duration 9.5–10.5 seconds")],
    model_ids: Annotated[str | None, Form()] = None,
    threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
):
    try:
        ids = parse_model_ids(model_ids)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise HTTPException(status_code=422, detail="Use MP4, MOV, MKV, AVI or WEBM video")
        with TemporaryDirectory(prefix="eventlens_") as tmp:
            path = Path(tmp) / ("clip" + ext)
            total_bytes = 0
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Maximum video size is 100 MB")
                    output.write(chunk)
            if not total_bytes:
                raise HTTPException(status_code=422, detail="Empty video upload")
            from eventlens.video_info import get_video_metadata
            info = get_video_metadata(path)
            if not 9.5 <= info["duration_sec"] <= 10.5:
                raise HTTPException(status_code=422, detail="Video must be 9.5–10.5 seconds long")
            if not info["has_audio"]:
                raise HTTPException(status_code=422, detail="Video must contain an audio track")
            from eventlens.extractor_runtime import extract_new_video
            features = extract_new_video(path, ARTIFACT_ROOT, os.environ.get("EVENTLENS_VGGISH_ARCHIVE", ""))
        result = compare_or_http(features, ids, threshold)
        result["input_type"] = "real_video"
        result["video"] = info
        return result
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, RuntimeError, __import__("subprocess").CalledProcessError) as exc:
        logger.warning("Video inference unavailable: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
