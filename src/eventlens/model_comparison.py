"""Run the same validated feature tensor through one or more models."""
from functools import lru_cache
from pathlib import Path
from eventlens.model_loader import load_event_model
from eventlens.model_registry import get_model_catalog
from eventlens.prediction import predict_events


@lru_cache(maxsize=8)
def get_loaded_model(model_id: str, artifact_root: str):
    return load_event_model(model_id, Path(artifact_root))


def compare_models(features: dict, artifact_root: Path, model_ids: list[str] | None = None,
                   threshold: float = 0.50) -> dict:
    root = Path(artifact_root).resolve()
    catalog = {m["id"]: m for m in get_model_catalog(root)}
    if model_ids is None:
        model_ids = [key for key, info in catalog.items() if info["available"]]
    if not model_ids:
        raise ValueError("No models selected or no checkpoints available")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate model IDs")
    results = {}
    reference_classes = None
    for model_id in model_ids:
        if model_id not in catalog:
            raise ValueError(f"Unknown model: {model_id}")
        if not catalog[model_id]["available"]:
            raise ValueError(f"Checkpoint unavailable: {model_id}")
        model, class_names = get_loaded_model(model_id, str(root))
        if reference_classes is None:
            reference_classes = class_names
        elif reference_classes != class_names:
            raise ValueError("Model checkpoints have inconsistent class definitions")
        events = predict_events(model, features, class_names, threshold=threshold)
        results[model_id] = {
            "name": catalog[model_id]["name"],
            "family": catalog[model_id]["family"],
            "checkpoint": catalog[model_id]["checkpoint"],
            "event_count": len(events),
            "events": events,
        }
    return {"models": results, "threshold": float(threshold), "duration_sec": 10.0}
