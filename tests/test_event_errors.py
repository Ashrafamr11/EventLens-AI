"""Tests of the exact notebook event matching and annotation parser."""
import importlib.util
from pathlib import Path
import csv
import numpy as np

P = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_event_errors.py'
spec = importlib.util.spec_from_file_location('eventlens_event_errors', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_normalize_names():
    assert m.normalize_video_id('4O9rI-FpqLg_10_20') == '4O9rI-FpqLg'
    assert m.normalize_video_id('4O9rI-FpqLg') == '4O9rI-FpqLg'


def test_get_events():
    assert m.get_events(np.array([0,1,1,0,1,1,1,0], bool)) == [(1,3),(4,7)]
    assert m.get_events(np.zeros(10, bool)) == []


def test_greedy_matching():
    p = [(0,3),(6,9)]
    gt = [(0,4),(4,6)]
    matches, fps, fns = m.match_events(p,gt,0.5)
    assert len(matches) == 1
    assert fps == [1] and fns == [1]
    assert matches[0][2] == 0.75


def test_annotations_audio_visual_union(tmp_path):
    d=tmp_path
    for fn, rows in [
        ('AVVP_eval_audio.csv', [['abc_10_20', '0','2','Dog'], ['abc_10_20','10','20','Dog']]),
        ('AVVP_eval_visual.csv', [['abc_10_20','1','4','Dog'], ['other_10_20','0','1','Dog']]),
    ]:
        with (d/fn).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['filename','onset','offset','event_labels']);w.writerows(rows)
    gt,invalid=m.build_temporal_gt(d,['abc'],['Dog'])
    assert invalid == 1
    assert gt.shape==(1,10,1)
    assert np.array_equal(gt[0,:,0],np.array([1,1,1,1,0,0,0,0,0,0],dtype=bool))
