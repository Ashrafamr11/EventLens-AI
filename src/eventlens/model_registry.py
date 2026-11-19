
from pathlib import Path


MODEL_REGISTRY = {
    "baseline": {
        "name": "Temporal MLP",
        "family": "Baseline",
        "checkpoint": "baseline_seed42_5epochs.pt",
        "implemented": True,
    },
    "bigru": {
        "name": "Bidirectional GRU",
        "family": "Classic DL",
        "checkpoint": "bigru_seed42_5epochs.pt",
        "implemented": True,
    },
    "transformer": {
        "name": "Temporal Transformer",
        "family": "Modern DL",
        "checkpoint": None,
        "implemented": False,
    },
    "cross_modal": {
        "name": "Cross-Modal Attention Fusion",
        "family": "Advanced DL",
        "checkpoint": None,
        "implemented": False,
    },
}


def get_model_catalog(checkpoints_dir: Path) -> list[dict]:
    catalog = []

    for model_id, info in MODEL_REGISTRY.items():
        checkpoint = info["checkpoint"]

        available = (
            info["implemented"]
            and checkpoint is not None
            and (Path(checkpoints_dir) / checkpoint).is_file()
        )

        catalog.append({
            "id": model_id,
            "name": info["name"],
            "family": info["family"],
            "available": available,
        })

    return catalog
