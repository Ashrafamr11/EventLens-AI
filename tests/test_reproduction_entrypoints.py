from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from train import MODEL_TYPES,make_model
from prepare_demo_cache import validate_and_copy


def test_all_four_models_return_same_shapes():
    import torch
    for model_id in MODEL_TYPES:
        model=make_model(model_id,25).eval()
        with torch.inference_mode():
            r=model(torch.zeros(1,10,128),torch.zeros(1,10,2048),torch.zeros(1,10,512))
        assert r['segment_logits'].shape==(1,10,25)
        assert r['clip_logits'].shape==(1,25)


def test_cached_feature_validator(tmp_path):
    src=tmp_path/'source.npz';dst=tmp_path/'cached.npz'
    np.savez_compressed(src,audio=np.ones((10,128)),visual=np.ones((10,2048)),motion=np.ones((10,512)))
    validate_and_copy(src,dst)
    with np.load(dst) as x:assert x['visual'].shape==(10,2048)
    with pytest.raises(FileExistsError):validate_and_copy(src,dst)
