
import torch
from torch import nn


class TemporalMLPBaseline(nn.Module):
    def __init__(self, num_classes=25, hidden_dim=256):
        super().__init__()

        input_dim = 128 + 2048 + 512

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, audio, visual, motion):
        features = torch.cat(
            [audio, visual, motion],
            dim=-1,
        )

        segment_logits = self.network(features)

        clip_logits = segment_logits.max(dim=1).values

        return {
            "segment_logits": segment_logits,
            "clip_logits": clip_logits,
        }
