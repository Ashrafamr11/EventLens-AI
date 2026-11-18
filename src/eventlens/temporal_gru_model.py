
import torch
from torch import nn


class TemporalGRUModel(nn.Module):
    def __init__(self, num_classes=25, hidden_dim=128):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(128 + 2048 + 512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, audio, visual, motion):
        features = torch.cat(
            [audio, visual, motion],
            dim=-1,
        )

        features = self.projection(features)

        temporal_features, _ = self.gru(features)

        segment_logits = self.classifier(
            temporal_features
        )

        clip_logits = segment_logits.max(dim=1).values

        return {
            "segment_logits": segment_logits,
            "clip_logits": clip_logits,
        }
