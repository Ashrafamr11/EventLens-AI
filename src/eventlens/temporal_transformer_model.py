
import torch
from torch import nn


class TemporalTransformerModel(nn.Module):
    def __init__(self, num_classes=25, hidden_dim=128):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(2688, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, 10, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.segment_head = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, audio, visual, motion):
        features = torch.cat(
            [audio, visual, motion],
            dim=-1,
        )

        x = self.projection(features)
        x = x + self.position_embedding
        x = self.transformer(x)

        segment_logits = self.segment_head(x)

        clip_logits = segment_logits.max(dim=1).values

        return {
            "segment_logits": segment_logits,
            "clip_logits": clip_logits,
        }
