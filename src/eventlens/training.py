
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        motion = batch["motion"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(audio, visual, motion)
        loss = criterion(outputs["clip_logits"], labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataset is empty.")

    return total_loss / total_samples

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        motion = batch["motion"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(audio, visual, motion)

        loss = criterion(
            outputs["clip_logits"],
            labels,
        )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataset is empty.")

    return total_loss / total_samples
