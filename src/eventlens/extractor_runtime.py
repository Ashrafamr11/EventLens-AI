"""Initialize pretrained AVVP-compatible feature extractors lazily.

The trained classifier consumes original ResNet-152, R(2+1)D-18 and VGGish
embeddings. Substituting a different encoder invalidates the evaluation.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

import torch
from torch import nn

# From the verified official torchvision ResNet152 IMAGENET1K_V1 file.
RESNET152_SHA256 = "b121ed2db97ec7e9f55a91300ceaf85a326de955e8a4ae09e3a0c8170d27f14f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb_extractor(path: Path, device):
    from torchvision.models import resnet152
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ResNet152 weights missing: {path}")
    # Newer torch may reject the official old .tar-format checkpoint in safe mode.
    # Unsafe pickle fallback is allowed *only* after full SHA-256 verification.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        if "legacy .tar format" not in str(exc):
            raise
        if sha256_file(path) != RESNET152_SHA256:
            raise RuntimeError("Refusing to deserialize unverified legacy ResNet weights") from exc
        state = torch.load(path, map_location="cpu", weights_only=False)
    model = resnet152(weights=None)
    if not isinstance(state, dict):
        raise ValueError("ResNet checkpoint must contain a state dict")
    if "fc.weight" not in state:
        model.fc = nn.Identity()
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    return model.to(device).eval()


def load_motion_extractor(path: Path, device):
    from torchvision.models.video import r2plus1d_18
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"R(2+1)D motion weights missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Motion checkpoint must contain a state dict")
    model = r2plus1d_18(weights=None)
    # The project's local motion state dict has no fc.weight/fc.bias: it is
    # already a 512-dimensional feature extractor, not a broken checkpoint.
    if "fc.weight" not in state and "fc.bias" not in state:
        model.fc = nn.Identity()
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    return model.to(device).eval()


def resolve_vggish_archive(artifact_root: Path, override: str | None = None) -> Path:
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"EVENTLENS_VGGISH_ARCHIVE not found: {path}")
    root = Path(artifact_root)
    candidates = [
        root / "tools/feature_extractor_original.zip",
        root / "data/vggish.zip",
        root / "data/feature_extractor.zip",
        root / "weights/vggish.zip",
        root / "weights/feature_extractor.zip",
        root / "vggish.zip",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "VGGish feature_extractor archive is missing. Set EVENTLENS_VGGISH_ARCHIVE "
        "to the original AVVP/LLP extractor archive containing vggish_model.ckpt "
        "and the four VGGish Python source files. Do not substitute another audio model."
    )


@lru_cache(maxsize=2)
def load_extractors(artifact_root: str, vggish_override: str = ""):
    """Returns models, device, and original VGGish archive paths."""
    root = Path(artifact_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = load_rgb_extractor(root / "weights/resnet152-b121ed2d.pth", device)
    motion = load_motion_extractor(root / "weights/r2plus1d18-kinetics400-v1.pth", device)
    archive = resolve_vggish_archive(root, vggish_override or None)
    vggish_dir = root / "runtime/vggish"
    return rgb, motion, device, archive, vggish_dir


def extract_new_video(video_path: Path, artifact_root: Path, vggish_override: str = ""):
    """End-to-end real-video feature extraction, no external prediction service."""
    # Lazy import keeps health, status and precomputed-features routes usable
    # on machines without the optional heavy inference dependencies.
    from eventlens.inference_features import extract_video_features
    rgb, motion, device, archive, vggish_dir = load_extractors(
        str(Path(artifact_root).resolve()), vggish_override
    )
    return extract_video_features(
        video_path, rgb, motion, device, archive, vggish_dir
    )
