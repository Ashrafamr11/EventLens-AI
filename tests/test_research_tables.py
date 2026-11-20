import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_research_tables.py'
spec = importlib.util.spec_from_file_location('build_research_tables', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_writes_csv_with_header(tmp_path):
    path = tmp_path / 'result.csv'
    mod.write_csv(path, ['name','value'], [['test', 1]])
    with path.open(encoding='utf-8-sig', newline='') as f:
        assert list(csv.reader(f)) == [['name','value'],['test','1']]


def test_mean():
    assert mod.ms_mean([1,2,3]) == 2
