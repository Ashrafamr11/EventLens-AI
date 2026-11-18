
from pathlib import Path

import torch

from eventlens.training import (
    train_one_epoch,
    validate_one_epoch,
)


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    class_names,
):
    if epochs < 1:
        raise ValueError("Epochs must be positive.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        val_loss = validate_one_epoch(
            model, val_loader, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "class_names": list(class_names),
            }, checkpoint_path)

            print("  Best checkpoint saved!")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    return history
