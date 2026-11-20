"""End-to-end API tests without downloading third-party weights or datasets."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))
from web import api
from eventlens.temporal_gru_model import TemporalGRUModel


@pytest.fixture
def configured_api(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / 'experiments/multi_seed'
    checkpoint_dir.mkdir(parents=True)
    classes = ['Dog', 'Speech']
    model = TemporalGRUModel(num_classes=2, hidden_dim=128)
    torch.save({'model_state_dict': model.state_dict(), 'class_names': classes},
               checkpoint_dir / 'bigru_seed44.pt')
    monkeypatch.setattr(api, 'ARTIFACT_ROOT', tmp_path)
    from eventlens.model_comparison import get_loaded_model
    get_loaded_model.cache_clear()
    with TestClient(api.app) as client:
        yield client
    get_loaded_model.cache_clear()


def example_npz():
    bio = io.BytesIO()
    np.savez_compressed(bio, audio=np.zeros((10,128), dtype=np.float32),
                        visual=np.zeros((10,2048), dtype=np.float32),
                        motion=np.zeros((10,512), dtype=np.float32))
    return bio.getvalue()


def test_health(configured_api):
    result = configured_api.get('/api/health')
    assert result.status_code == 200 and result.json()['status'] == 'ok'


def test_model_registry_and_status(configured_api):
    models = configured_api.get('/api/models').json()['models']
    assert len(models) == 4
    assert {m['id'] for m in models if m['available']} == {'bigru'}
    status = configured_api.get('/api/status').json()
    assert status['ready_for_features'] is True
    assert status['ready_for_video'] is False


def test_features_end_to_end(configured_api):
    response = configured_api.post('/api/analyze-features',
       files={'file': ('demo_real_features.npz', example_npz(), 'application/octet-stream')},
       data={'model_ids':'bigru','threshold':'0.50'})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['input_type'] == 'precomputed_features'
    assert body['duration_sec'] == 10.0
    assert 'bigru' in body['models']
    for event in body['models']['bigru']['events']:
        assert 0 <= event['start_sec'] < event['end_sec'] <= 10


def test_invalid_npz_rejected(configured_api):
    response = configured_api.post('/api/analyze-features',
        files={'file': ('bad.npz', b'not npz', 'application/octet-stream')},
        data={'model_ids':'bigru'})
    assert response.status_code == 422


def test_unknown_model_rejected(configured_api):
    response = configured_api.post('/api/analyze-features',
        files={'file': ('ok.npz', example_npz(), 'application/octet-stream')},
        data={'model_ids':'inexistent'})
    assert response.status_code == 422


def test_video_non_video_rejected(configured_api):
    response = configured_api.post('/api/analyze-video',
        files={'file': ('bad.txt', b'a', 'text/plain')}, data={'model_ids':'bigru'})
    assert response.status_code == 422


def test_main_page(configured_api):
    response = configured_api.get('/')
    assert response.status_code == 200
    assert 'EventLens' in response.text
    assert 'Gradio' not in response.text


def test_video_upload_pipeline_wiring(configured_api, tmp_path, monkeypatch):
    """Validate media route, ffprobe integration, temp lifecycle and JSON output.

    Feature extraction is monkeypatched: actual VGGish third-party assets are not
    included in this source ZIP. A real-video integration run remains required.
    """
    import subprocess
    from eventlens import extractor_runtime
    from eventlens.video_info import get_video_metadata
    clip = tmp_path / 'clip.mp4'
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','lavfi','-i','color=c=black:s=160x120:r=8:d=10',
        '-f','lavfi','-i','sine=frequency=440:duration=10',
        '-c:v','mpeg4','-q:v','12','-c:a','aac','-shortest',str(clip)]
    subprocess.run(command, check=True)
    assert get_video_metadata(clip)['has_audio']
    def mock_extract(path, artifact_root, override=''):
        assert path.exists()
        assert get_video_metadata(path)['duration_sec'] > 9.5
        return {'audio':np.zeros((10,128),dtype=np.float32),
                'visual':np.zeros((10,2048),dtype=np.float32),
                'motion':np.zeros((10,512),dtype=np.float32)}
    monkeypatch.setattr(extractor_runtime,'extract_new_video',mock_extract)
    response=configured_api.post('/api/analyze-video',
        files={'file':('sample.mp4',clip.read_bytes(),'video/mp4')},
        data={'model_ids':'bigru','threshold':'0.50'})
    assert response.status_code == 200, response.text
    body=response.json()
    assert body['input_type']=='real_video'
    assert body['video']['has_audio'] is True
    assert 'bigru' in body['models']
