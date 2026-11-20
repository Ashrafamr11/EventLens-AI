"""Secure, CPU-compatible loading of four trained model families."""
from pathlib import Path
import torch
from eventlens.baseline_model import TemporalMLPBaseline
from eventlens.temporal_gru_model import TemporalGRUModel
from eventlens.temporal_transformer_model import TemporalTransformerModel
from eventlens.cross_modal_model import CrossModalAttentionModel
from eventlens.model_registry import MODEL_REGISTRY, resolve_checkpoint

ARCHITECTURES = {
    "baseline": TemporalMLPBaseline,
    "bigru": TemporalGRUModel,
    "transformer": TemporalTransformerModel,
    "cross_modal": CrossModalAttentionModel,
}


def load_event_model(model_id: str, artifact_root: Path):
    """Return (model, class names), enforcing exact state-dict compatibility."""
    path = resolve_checkpoint(model_id, artifact_root)
    if path is None:
        raise FileNotFoundError(f"Checkpoint unavailable for {model_id} under {artifact_root}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint.get("class_names")
    if not isinstance(classes, (tuple, list)) or not classes:
        raise ValueError(f"Missing class_names in {path}")
    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate class names in checkpoint")
    model = ARCHITECTURES[model_id](
        num_classes=len(classes), hidden_dim=MODEL_REGISTRY[model_id]["hidden_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, list(classes)
