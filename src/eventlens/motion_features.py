
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


MOTION_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989],
    ),
])


def extract_motion_features(
    frames: list[Path],
    model,
    device,
    batch_size: int = 2,
) -> np.ndarray:

    if len(frames) != 80:
        raise ValueError("Expected exactly 80 frames.")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    clips = []

    # Divide 80 frames into 10 groups of 8
    for start in range(0, 80, 8):
        images = []

        for frame_path in frames[start:start + 8]:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                images.append(MOTION_TRANSFORM(image))

        # (3, 8, 112, 112)
        clip = torch.stack(images, dim=1)
        clips.append(clip)

    model.eval()
    all_features = []

    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            batch = torch.stack(
                clips[start:start + batch_size]
            ).to(device)

            features = model(batch)
            features = features.flatten(start_dim=1)

            all_features.append(
                features.cpu().numpy()
            )

    result = np.concatenate(all_features, axis=0)
    result = result.astype(np.float32)

    if result.shape != (10, 512):
        raise ValueError(
            f"Unexpected motion feature shape: {result.shape}"
        )

    if not np.isfinite(result).all():
        raise ValueError("Invalid motion feature values.")

    return result
