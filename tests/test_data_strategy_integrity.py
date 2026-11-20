"""Regression tests for the archived, completed EventLens study."""
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('eventlens_verify_data_strategy',ROOT/'scripts/verify_data_strategy_results.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_all_three_seeds_and_metrics_reconcile():
    _,(summaries,by_seed)=mod.validate(ROOT/'experiments/data_strategy')
    assert len(summaries)==3
    assert len(by_seed)==9
    assert next(x for x in summaries if x['metric']=='event')['delta_pp'] == pytest.approx(-.3202612746532385)


def test_one_seed_cannot_be_silently_omitted(tmp_path):
    import shutil
    source=ROOT/'experiments/data_strategy'
    for file in source.glob('*.json'):
        if 'seed44' not in file.name:
            shutil.copyfile(file,tmp_path/file.name)
    with pytest.raises(FileNotFoundError):
        mod.validate(tmp_path)


def test_cpu_label_does_not_invoke_cuda(monkeypatch):
    import torch
    source=(ROOT/'scripts/run_data_strategy.py').read_text(encoding='utf8')
    assert '"CPU" if device.type == "cpu" else torch.cuda.get_device_name(device)' in source
    def forbidden(*a,**kw):
        raise RuntimeError('GPU must not be queried on CPU')
    monkeypatch.setattr(torch.cuda,'get_device_name',forbidden)
    device=torch.device('cpu')
    label="CPU" if device.type=="cpu" else torch.cuda.get_device_name(device)
    assert label=="CPU"
