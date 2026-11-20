"""Turn the team's saved, real Colab audit into report-ready research tables.

No training, feature extraction, third-party packages, or fabricated counts.
Example:
 python scripts/build_research_tables.py --audit path/to/EventLens-AI-research-audit.zip
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path
from zipfile import ZipFile

FAMILIES = ('baseline', 'bigru', 'transformer', 'cross_modal')
SEEDS = (42, 43, 44)


def load_json(archive: ZipFile, path: str) -> dict:
    return json.loads(archive.read(path).decode('utf-8'))


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def ms_mean(vals: list[float]) -> float:
    return statistics.mean(vals)


def collect(archive: ZipFile, out: Path) -> dict:
    audit = load_json(archive, 'runtime_audit.json')
    multi = load_json(archive, 'experiments/multi_seed_validation_metrics.json')
    final = load_json(archive, 'experiments/final_test_bigru_seed44.json')
    ablation = load_json(archive, 'experiments/ablation/no_audio_validation_metrics.json')

    if multi.get('split') != 'validation' or final.get('split') != 'test':
        raise ValueError('Unexpected train/validation/test split: refusing to mix results')
    if set(multi['seeds']) != set(SEEDS):
        raise ValueError('Three-seed validation records are incomplete')
    if final['model'] != 'BiGRU' or int(final['seed']) != 44:
        raise ValueError('Unexpected held-out model, check test selection')

    validation_rows = []
    cost_rows = []
    for family in FAMILIES:
        train_records = [load_json(archive, f'experiments/multi_seed/{family}_seed{s}.json') for s in SEEDS]
        for s, record in zip(SEEDS, train_records):
            if record['seed'] != s or record['model'] != family:
                raise ValueError(f'Model/seed mismatch: {family} seed {s}')
        settings = ('epochs', 'optimizer', 'batch_size', 'learning_rate', 'weight_decay', 'hardware')
        for key in settings:
            if len({record.get(key) for record in train_records}) != 1:
                raise ValueError(f'Inconsistent {key} across {family} seeds')
        meta = audit['models'][family]
        params = {record['parameters'] for record in train_records}
        if params != {meta['parameters']}:
            raise ValueError(f'Parameter count mismatch for {family}')
        for scale in ('clip', 'segment', 'event'):
            metric = multi['summary'][family][scale]
            validation_rows.append([family, scale, round(metric['micro_f1']['mean'], 5),
                                    round(metric['micro_f1']['std'], 5),
                                    round(metric['macro_f1']['mean'], 5),
                                    round(metric['macro_f1']['std'], 5)])
        time_seconds = [float(r['training_seconds']) for r in train_records]
        cost_rows.append([
            family, meta['parameters'], meta['checkpoint_mib'],
            round(ms_mean(time_seconds), 2), round(statistics.stdev(time_seconds), 2),
            train_records[0]['hardware'],
            meta['latency_mean_ms'], meta['latency_sd_ms'],
            'CPU, classifier only (features pre-extracted)',
            'Not measured (peak RAM/VRAM)',
            train_records[0]['optimizer'], train_records[0]['batch_size'],
            train_records[0]['learning_rate'], train_records[0]['epochs'],
        ])
    write_csv(out/'validation_metrics.csv',
              ['model', 'level', 'micro_f1_mean', 'micro_f1_sd', 'macro_f1_mean', 'macro_f1_sd'], validation_rows)
    write_csv(out/'computational_cost.csv',
              ['model','head_parameters','checkpoint_size_mib','training_seconds_mean_3seeds',
               'training_seconds_sd_3seeds','training_hardware','cpu_head_inference_ms_mean_10runs',
               'cpu_head_inference_ms_sd_10runs','inference_scope','peak_memory',
               'optimizer','batch_size','learning_rate','epochs'], cost_rows)

    # Exact 25 class scores from the *held-out* selected checkpoint, not validation averages.
    per_class = final['results']['event']['per_class_f1']
    if len(per_class) != 25:
        raise ValueError(f'Expected 25 event classes, found {len(per_class)}')
    write_csv(out/'final_test_per_class.csv', ['class','clip_f1','segment_f1','event_f1'], [
        [name,round(final['results']['clip']['per_class_f1'][name],5),
         round(final['results']['segment']['per_class_f1'][name],5),
         round(val,5)] for name,val in sorted(per_class.items())
    ])
    zero_event_f1 = sorted(name for name, val in per_class.items() if val == 0)
    write_csv(out/'ablation.csv', ['metric','full_bigru','no_audio_bigru', 'delta_pp'], [
        [level, round(ablation['comparison'][level]['original_f1'],5),
         round(ablation['comparison'][level]['no_audio_mean_f1'],5),
         round(ablation['comparison'][level]['difference_percentage_points'],3)]
        for level in ('clip','segment','event')
    ])
    summary = {
        'source': 'Saved experimental audit ZIP',
        'selected_checkpoint': final['checkpoint'],
        'test_clips': final['num_clips'],
        'test_event_micro_f1': final['results']['event']['micro_f1'],
        'test_zero_event_f1_classes': zero_event_f1,
        'invalid_temporal_rows_test': final['invalid_annotation_rows'],
        'unverified': [
            'Per-class event TP/FP/FN counts (cannot be recovered from F1 alone)',
            'Per-model peak memory',
            'Class-specific annotated failure examples',
            'Controlled data-strategy experiment',
            'Pretrained-vs-scratch comparison where applicable',
        ],
        'note': 'A zero class F1 is not proof that the class was never predicted; it can reflect no true positives.',
    }
    (out/'research_audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path, help='EventLens-AI-research-audit.zip')
    parser.add_argument('--output', type=Path, default=Path('experiments/analysis'))
    args = parser.parse_args()
    if not args.audit.is_file():
        parser.error(f'Archive not found: {args.audit}')
    with ZipFile(args.audit) as z:
        summary = collect(z, args.output)
    print('RESEARCH TABLES SUCCESS')
    print('Output folder:', args.output.resolve())
    print('Test event micro-F1:', f"{summary['test_event_micro_f1']:.4f}")
    print('Zero-F1 event classes:', ', '.join(summary['test_zero_event_f1_classes']))
    print('CSV tables: validation_metrics, computational_cost, final_test_per_class, ablation')

if __name__ == '__main__':
    main()
