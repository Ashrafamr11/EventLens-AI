"""Model families and checkpoint priorities.

Checkpoints are external artifacts and are never bundled with source code.
"""
from pathlib import Path

MODEL_REGISTRY = {
    "baseline": {
        "name": "Temporal MLP", "family": "Baseline",
        "checkpoints": ["experiments/multi_seed/baseline_seed42.pt", "checkpoints/baseline_seed42_5epochs.pt"],
        "hidden_dim": 256,
    },
    "bigru": {
        "name": "Bidirectional GRU", "family": "Classic DL",
        "checkpoints": ["experiments/multi_seed/bigru_seed44.pt", "checkpoints/bigru_seed42_5epochs.pt"],
        "hidden_dim": 128,
    },
    "transformer": {
        "name": "Temporal Transformer", "family": "Modern DL",
        "checkpoints": ["experiments/multi_seed/transformer_seed44.pt", "checkpoints/transformer_seed42_5epochs.pt"],
        "hidden_dim": 128,
    },
    "cross_modal": {
        "name": "Cross-Modal Attention", "family": "Advanced DL",
        "checkpoints": ["experiments/multi_seed/cross_modal_seed42.pt", "checkpoints/cross_modal_seed42_5epochs.pt"],
        "hidden_dim": 128,
    },
}


def resolve_checkpoint(model_id: str, artifact_root: Path) -> Path | None:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    base = Path(artifact_root).resolve()
    for candidate in MODEL_REGISTRY[model_id]["checkpoints"]:
        path = base / candidate
        if path.is_file():
            return path
    return None


def get_model_catalog(artifact_root: Path) -> list[dict]:
    return [
        {
            "id": model_id,
            "name": metadata["name"],
            "family": metadata["family"],
            "available": (checkpoint := resolve_checkpoint(model_id, artifact_root)) is not None,
            "checkpoint": str(checkpoint.relative_to(Path(artifact_root).resolve())) if checkpoint else None,
        }
        for model_id, metadata in MODEL_REGISTRY.items()
    ]
