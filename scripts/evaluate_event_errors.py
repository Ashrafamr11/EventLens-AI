"""Reproducible per-event error audit for the selected EventLens BiGRU.

Reads the *existing* precomputed LLP features and official audio+visual
annotations. Nothing is trained, downloaded, or changed in model checkpoints.
Metrics exactly match the team notebook's custom greedy IoU>=0.50 protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from eventlens.training_dataset import LLPDataset
from eventlens.temporal_gru_model import TemporalGRUModel


def normalize_video_id(filename: str) -> str:
    parts = filename.strip().rsplit('_', 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return filename.strip()


def read_delimited(path: Path):
    """Handle official TSV or CSV manifests; require named columns."""
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        header = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter='\t' if '\t' in header else ',')
        for row in reader:
            yield row


def build_temporal_gt(data_dir: Path, video_ids: list[str], class_names: list[str]):
    """Build the same pooled GT used by the original Colab evaluation."""
    video_to_ix = {video: i for i, video in enumerate(video_ids)}
    assert len(video_to_ix) == len(video_ids), 'Duplicate video IDs'
    class_to_ix = {name: i for i, name in enumerate(class_names)}
    gt = np.zeros((len(video_ids), 10, len(class_names)), dtype=np.bool_)
    invalid_rows = 0
    unknown = set()
    for fn in ('AVVP_eval_audio.csv', 'AVVP_eval_visual.csv'):
        path = data_dir / fn
        if not path.is_file():
            raise FileNotFoundError(f'Missing strong annotations: {path}')
        for row in read_delimited(path):
            video_id = normalize_video_id(row.get('filename', ''))
            if video_id not in video_to_ix:
                continue
            label = row.get('event_labels', '').strip()
            if label not in class_to_ix:
                unknown.add(label)
                continue
            try:
                start = int(float(row['onset']))
                end = int(float(row['offset']))
            except (ValueError, TypeError, KeyError):
                invalid_rows += 1
                continue
            if not 0 <= start < end <= 10:
                invalid_rows += 1
                continue
            gt[video_to_ix[video_id], start:end, class_to_ix[label]] = True
    if unknown:
        raise ValueError(f'Unknown labels in annotations: {sorted(unknown)}')
    if not gt.any():
        raise ValueError('No temporal annotations found for selected split')
    return gt, invalid_rows


def get_events(mask: np.ndarray):
    x = np.asarray(mask, dtype=bool)
    if x.ndim != 1:
        raise ValueError('Expected one temporal mask per class')
    changes = np.flatnonzero(np.diff(np.r_[False, x, False]))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    ps, pe = a
    gs, ge = b
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union else 0.0


def match_events(predicted, actual, iou_threshold=0.5):
    """Exact greedy one-to-one candidate sort used in saved notebook."""
    candidates = [(temporal_iou(p, g), i, j)
                  for i, p in enumerate(predicted)
                  for j, g in enumerate(actual)
                  if temporal_iou(p, g) >= iou_threshold]
    matched_pred, matched_gt, pairs = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            pairs.append((i, j, iou))
    return pairs, [i for i in range(len(predicted)) if i not in matched_pred], [j for j in range(len(actual)) if j not in matched_gt]


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args) -> dict:
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    data_dir = Path(args.annotations_dir).resolve() if args.annotations_dir else project_root / 'data/raw/llp'
    feature_dir = Path(args.features_dir).resolve() if args.features_dir else project_root / 'data/processed/feats'
    out = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = artifact_root / 'experiments/multi_seed/bigru_seed44.pt'

    # Fail quickly, before loading 1200 items, when the Colab runtime is fresh.
    if not feature_dir.is_dir():
        raise FileNotFoundError(
            f'Missing extracted LLP features at {feature_dir}. '
            'Use the original extracted feats directory; do not regenerate from videos.')
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Missing LLP annotation directory: {data_dir}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing selected BiGRU checkpoint: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    class_names = list(ckpt['class_names'])
    if len(class_names) != 25 or len(set(class_names)) != 25:
        raise ValueError('Selected checkpoint must have exactly 25 distinct LLP labels')

    dataset = LLPDataset(
        split=args.split,
        class_names=class_names,
        data_dir=str(data_dir),
        features_dir=str(feature_dir),
    )
    video_ids = [normalize_video_id(x['video_id']) for x in dataset.samples]
    expected = 1200 if args.split == 'test' else 649
    if len(dataset) != expected:
        raise ValueError(f'Expected {expected} {args.split} samples, got {len(dataset)}')
    gt, invalid_rows = build_temporal_gt(data_dir, video_ids, class_names)
    print(f'DATASET READY: {len(dataset)} clips, {len(class_names)} labels, invalid GT rows={invalid_rows}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGRUModel(num_classes=len(class_names), hidden_dim=128).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    counts = {label: Counter(tp=0, fp=0, fn=0) for label in class_names}
    issue_rows = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch['audio'].to(device, dtype=torch.float32),
                batch['visual'].to(device, dtype=torch.float32),
                batch['motion'].to(device, dtype=torch.float32),
            )
            probs = torch.sigmoid(output['segment_logits']).cpu().numpy()
            pred = probs >= args.threshold
            if pred.shape[1:] != (10, len(class_names)):
                raise ValueError(f'Wrong prediction shape: {pred.shape}')
            true = gt[offset:offset+len(pred)]
            for b in range(len(pred)):
                video_id = video_ids[offset+b]
                for c, label in enumerate(class_names):
                    p_events = get_events(pred[b, :, c])
                    t_events = get_events(true[b, :, c])
                    pairs, unmatched_p, unmatched_t = match_events(p_events, t_events, args.iou)
                    counts[label]['tp'] += len(pairs)
                    counts[label]['fp'] += len(unmatched_p)
                    counts[label]['fn'] += len(unmatched_t)

                    for i in unmatched_p:
                        ps, pe = p_events[i]
                        issue_rows.append(dict(video_id=video_id, label=label, issue='false_positive',
                            pred_start=ps, pred_end=pe, gt_start='', gt_end='',
                            max_score=round(float(probs[b, ps:pe, c].max()), 5), matched_iou=''))
                    for j in unmatched_t:
                        gs, ge = t_events[j]
                        issue_rows.append(dict(video_id=video_id, label=label, issue='false_negative',
                            pred_start='', pred_end='', gt_start=gs, gt_end=ge,
                            max_score=round(float(probs[b, gs:ge, c].max()), 5), matched_iou=''))
                    for i, j, score in pairs:
                        if score < 1:
                            ps, pe = p_events[i]
                            gs, ge = t_events[j]
                            issue_rows.append(dict(video_id=video_id, label=label, issue='boundary_mismatch',
                                pred_start=ps, pred_end=pe, gt_start=gs, gt_end=ge,
                                max_score=round(float(probs[b, ps:pe, c].max()), 5), matched_iou=round(score, 5)))
            offset += len(pred)
            print(f'Evaluated {offset}/{len(dataset)} clips', end='\r', flush=True)
    print()
    if offset != len(dataset):
        raise ValueError(f'Incomplete evaluation {offset}/{len(dataset)}')

    tp = sum(x['tp'] for x in counts.values())
    fp = sum(x['fp'] for x in counts.values())
    fn = sum(x['fn'] for x in counts.values())
    micro = 2 * tp / max(1, (2 * tp + fp + fn))

    # This is an audit, not a new experiment. On the unmodified test split the
    # output *must* reproduce the original saved result exactly.
    reference_path = artifact_root / 'experiments/final_test_bigru_seed44.json'
    if args.split == 'test' and abs(args.threshold - 0.5) < 1e-9 and abs(args.iou - 0.5) < 1e-9:
        if reference_path.is_file():
            ref = json.loads(reference_path.read_text(encoding='utf-8'))
            expected_f1 = ref['results']['event']['micro_f1']
            if abs(micro - expected_f1) > 1e-8:
                raise AssertionError(
                    f'Original F1={expected_f1:.6f}, audited F1={micro:.6f}. '
                    'Verify identical features, checkpoint, and annotation parsing before reporting these counts.')
        else:
            print('WARNING: original final test JSON missing; cannot cross-check F1')

    out.mkdir(parents=True, exist_ok=True)
    class_rows = []
    for label in class_names:
        c = counts[label]
        den = 2 * c['tp'] + c['fp'] + c['fn']
        class_rows.append(dict(label=label, tp=c['tp'], fp=c['fp'], fn=c['fn'],
                               event_f1=round(2*c['tp'] / den if den else 0.0, 6)))
    write_csv(out / 'event_errors_per_class.csv', class_rows, ['label','tp','fp','fn','event_f1'])
    fields = ['video_id','label','issue','pred_start','pred_end','gt_start','gt_end','max_score','matched_iou']
    write_csv(out / 'event_error_examples.csv', issue_rows, fields)
    summary = dict(split=args.split, checkpoint='bigru_seed44.pt', threshold=args.threshold,
                   event_iou_threshold=args.iou, annotation_union='audio_visual_union',
                   clips=len(dataset), invalid_annotation_rows=invalid_rows,
                   tp=tp, fp=fp, fn=fn, event_micro_f1=micro,
                   issues=dict(Counter(row['issue'] for row in issue_rows)),
                   important_note='These are errors relative to pooled temporal annotations, not human-reviewed video examples.')
    (out / 'error_analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"EVENT ERROR AUDIT SUCCESS | F1={micro:.4f} | TP={tp} FP={fp} FN={fn}")
    print('Saved:', out)
    return summary


def main():
    parser = argparse.ArgumentParser(description='BiGRU event-error audit; no training')
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--project-root', default=str(PROJECT))
    parser.add_argument('--annotations-dir')
    parser.add_argument('--features-dir')
    parser.add_argument('--split', choices=['validation','test'], default='test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.threshold <= 1 or not 0 < args.iou <= 1:
        parser.error('Invalid batch size, threshold or IoU')
    run(args)


if __name__ == '__main__':
    main()
