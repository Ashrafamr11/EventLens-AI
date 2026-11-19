
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


RGB_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_rgb_features(
    frames: list[Path],
    model,
    device,
    batch_size: int = 16,
) -> np.ndarray:

    if len(frames) != 80:
        raise ValueError("Expected exactly 80 frames.")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    model.eval()
    all_features = []

    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            images = []

            for frame_path in batch_frames:
                with Image.open(frame_path) as image:
                    image = image.convert("RGB")
                    images.append(RGB_TRANSFORM(image))

            inputs = torch.stack(images).to(device)

            features = model(inputs)

            all_features.append(
                features.cpu().numpy()
            )

    result = np.concatenate(all_features, axis=0)
    result = result.astype(np.float32)

    if result.shape != (80, 2048):
        raise ValueError(
            f"Unexpected RGB feature shape: {result.shape}"
        )

    return result
