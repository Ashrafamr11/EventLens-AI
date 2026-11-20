
import torch
from torch import nn


class CrossModalAttentionModel(nn.Module):
    def __init__(self, num_classes=25, hidden_dim=128):
        super().__init__()

        # Separate encoders for each modality
        self.audio_encoder = nn.Linear(128, hidden_dim)
        self.visual_encoder = nn.Linear(2048, hidden_dim)
        self.motion_encoder = nn.Linear(512, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        # Attention between audio, visual and motion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Learn the importance of each modality
        self.modality_gate = nn.Linear(
            hidden_dim * 3, 3
        )

        # Learn temporal relationships
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.segment_head = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, audio, visual, motion):
        a = self.audio_encoder(audio)
        v = self.visual_encoder(visual)
        m = self.motion_encoder(motion)

        # (Batch, Time, 3 modalities, Hidden)
        modalities = torch.stack(
            [a, v, m], dim=2
        )

        batch, time, _, hidden = modalities.shape

        tokens = modalities.reshape(
            batch * time, 3, hidden
        )

        tokens = self.norm(tokens)

        attended, _ = self.cross_attention(
            tokens, tokens, tokens,
            need_weights=False,
        )

        # Custom gated fusion
        gate_logits = self.modality_gate(
            attended.reshape(batch * time, hidden * 3)
        )

        weights = torch.softmax(
            gate_logits, dim=-1
        ).unsqueeze(-1)

        fused = (attended * weights).sum(dim=1)
        fused = fused.reshape(batch, time, hidden)

        temporal_features, _ = self.temporal_gru(fused)

        segment_logits = self.segment_head(
            temporal_features
        )

        clip_logits = segment_logits.max(dim=1).values

        return {
            "segment_logits": segment_logits,
            "clip_logits": clip_logits,
        }
