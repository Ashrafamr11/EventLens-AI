"""Small test archive verifies selective extraction, not a full dataset copy."""
import csv
import importlib.util
from pathlib import Path
from zipfile import ZipFile

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'prepare_error_assets.py'
spec=importlib.util.spec_from_file_location('eventlens_prepare_assets',SCRIPT)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_split_ids_tsv(tmp_path):
    f=tmp_path/'AVVP_val_pd.csv'
    f.write_text('filename\tevent_labels\nabc12345678_30_40\tDog\n',encoding='utf8')
    assert m.split_ids(f) == ['abc12345678']
