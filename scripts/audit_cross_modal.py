"""Non-destructive structural audit of the Cross-Modal Attention model.

Checks shape, finite outputs, attention parameter gradient, gate gradients,
and batch/time axis behavior on synthetic data. This is NOT an empirical proof
that the low event-F1 is not caused by a data or training bug.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from eventlens.cross_modal_model import CrossModalAttentionModel


def run():
    torch.manual_seed(42)
    model=CrossModalAttentionModel(num_classes=25).train()
    audio=torch.randn(2,10,128)
    rgb=torch.randn(2,10,2048)
    motion=torch.randn(2,10,512)
    pred=model(audio,rgb,motion)
    assert pred['segment_logits'].shape==(2,10,25)
    assert pred['clip_logits'].shape==(2,25)
    assert torch.isfinite(pred['segment_logits']).all()
    assert torch.allclose(pred['clip_logits'],pred['segment_logits'].max(dim=1).values)
    loss=pred['clip_logits'].pow(2).mean()
    loss.backward()
    for part in ('modality_gate','cross_attention','temporal_gru','audio_encoder','visual_encoder','motion_encoder'):
        module=getattr(model,part)
        has_nonzero_gradient=any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                 and bool((p.grad.abs()>0).any()) for p in module.parameters())
        if not has_nonzero_gradient:
            raise AssertionError(f'No finite nonzero gradient in {part}')
    print('Cross-modal structural smoke test PASSED; inspect data/optimization independently before claiming no bug.')


if __name__=='__main__':run()
