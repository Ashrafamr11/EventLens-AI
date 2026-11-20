import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]

def module(script):
    spec=importlib.util.spec_from_file_location('eventlens_'+script, P/'scripts'/f'{script}.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class Fake:
    def __len__(self):return 3
    def __getitem__(self,idx):
        return {'audio': np.arange(1280,dtype=np.float32).reshape(10,128),
                'visual': np.ones((10,2048),dtype=np.float32),
                'motion': np.linspace(-1,1,5120,dtype=np.float32).reshape(10,512),
                'labels':np.zeros(25,dtype=np.float32)}


def test_augmentation_training_only_shapes_and_no_aliasing():
    m=module('run_data_strategy')
    np.random.seed(42)
    original=Fake()[0]
    augmented=m.TrainOnlyFeatureNoise(Fake(),0.05)[0]
    assert len(m.TrainOnlyFeatureNoise(Fake()))==3
    assert augmented['visual'].shape==(10,2048)
    assert augmented['audio'].shape==(10,128)
    assert augmented['motion'].shape==(10,512)
    assert all(np.isfinite(augmented[k]).all() for k in ('audio','visual','motion'))
    assert not np.array_equal(augmented['audio'],original['audio'])
    assert np.array_equal(original['audio'],Fake()[0]['audio'])
    assert np.array_equal(augmented['labels'],original['labels'])


def test_noise_scale_0_preserves_inputs():
    m=module('run_data_strategy')
    x=m.TrainOnlyFeatureNoise(Fake(),0)[1]
    for k in ('audio','visual','motion'):
        assert np.array_equal(x[k],Fake()[1][k])


def test_metric_count_calculation():
    m=module('run_data_strategy')
    pred=np.array([[[1,0],[0,1]]],dtype=bool)
    gt=np.array([[[1,1],[0,0]]],dtype=bool)
    counts=np.zeros((3,2),dtype=np.int64)
    m.update_counts(counts,pred,gt,(0,1))
    assert (counts.sum(axis=1)==np.array([1,1,1])).all()
    result=m.metric_summary(counts)
    assert result['micro_f1']==0.5


def test_train_ids_duplicate_detected(tmp_path):
    m=module('prepare_training_assets')
    manifest=tmp_path/'ids.csv'
    manifest.write_text('filename\tevent_labels\nABCDEFGHIJK_10_10\tDog\nABCDEFGHIJK_11_12\tDog\n')
    with pytest.raises(ValueError,match='Duplicate'):
        m.read_ids(manifest)
