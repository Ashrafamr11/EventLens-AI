"""Research training CLI for the EventLens localization heads.

These are *new reproduction runs*, not retroactive original experimental records.
Large artifacts must remain outside Git. Original logs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from eventlens.baseline_model import TemporalMLPBaseline
from eventlens.temporal_gru_model import TemporalGRUModel
from eventlens.temporal_transformer_model import TemporalTransformerModel
from eventlens.cross_modal_model import CrossModalAttentionModel
from eventlens.experiment import fit_model
from eventlens.training_dataset import LLPDataset

MODEL_TYPES = {
    'baseline': TemporalMLPBaseline,
    'bigru': TemporalGRUModel,
    'transformer': TemporalTransformerModel,
    'cross_modal': CrossModalAttentionModel,
}


def make_model(model_id: str, classes: int):
    if model_id not in MODEL_TYPES:
        raise ValueError(f'Unknown model ID: {model_id}')
    return MODEL_TYPES[model_id](num_classes=classes)


def read_class_names(reference: Path) -> list[str]:
    if not reference.is_file():
        raise FileNotFoundError(f'Missing known LLP reference checkpoint: {reference}')
    obj = torch.load(reference, map_location='cpu', weights_only=True)
    names = list(obj['class_names'])
    if len(names) != 25 or len(set(names)) != 25:
        raise ValueError('LLP checkpoint must declare 25 unique classes')
    return names


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(name)


def train_once(args):
    if args.split != 'validation':
        raise ValueError('The training CLI may validate only on validation split, never test.')
    if not 1 <= args.epochs <= 100 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError('Invalid training configuration')
    project = args.project_root.resolve()
    root = args.artifact_root.resolve()
    if args.reference_checkpoint is None:
        reference = root / 'experiments/multi_seed/bigru_seed44.pt'
    else:
        reference = args.reference_checkpoint.resolve()
    names = read_class_names(reference)
    dataset_dir = project / 'data/raw/llp'
    feature_dir = project / 'data/processed/feats'
    training = LLPDataset('train', names, data_dir=dataset_dir, features_dir=feature_dir)
    validation = LLPDataset('validation', names, data_dir=dataset_dir, features_dir=feature_dir)
    if len(training) != 10000 or len(validation) != 649:
        raise ValueError(f'Unexpected LLP splits: {len(training)} train, {len(validation)} validation')
    # Ensure feature storage and label format are usable before training starts.
    for split_name, ds in [('train', training), ('validation', validation)]:
        item = ds[0]
        for key, expected in [('audio',(10,128)),('visual',(10,2048)),('motion',(10,512))]:
            if tuple(item[key].shape) != expected:
                raise ValueError(f'{split_name}/{key}: {item[key].shape} != {expected}')
    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = select_device(args.device)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              generator=gen, num_workers=args.workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type=='cuda')
    model = make_model(args.model, len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    out = (root / 'experiments/reproduction_runs' / f'{args.model}_seed{seed}_lr{args.learning_rate:g}')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/'best.pt'; record_path = out/'run.json'
    if checkpoint.exists() or record_path.exists():
        raise FileExistsError(f'Existing reproduction run: {out}. Choose another output or delete it intentionally.')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    history = fit_model(model, train_loader, val_loader, optimizer, device, args.epochs,
                        checkpoint, names)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-t0
    rec = dict(model=args.model, seed=seed, original_study=False,
               split='LLP train/validation', batch_size=args.batch_size, epochs=args.epochs,
               learning_rate=args.learning_rate, weight_decay=args.weight_decay,
               optimizer='AdamW', loss='clip BCEWithLogits',
               hardware=(torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'),
               gpu_peak_allocated_mib=(torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None),
               gpu_peak_reserved_mib=(torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None),
               train_time_seconds=elapsed,
               head_parameters=sum(x.numel() for x in model.parameters()),
               checkpoint_bytes=checkpoint.stat().st_size, history=history)
    record_path.write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(f'SAVED NEW REPRODUCTION RUN: {record_path}')
    print(f'GPU_PEAK_ALLOCATED_MIB: {rec["gpu_peak_allocated_mib"]} (None means not measured / CPU)')
    return rec


def get_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--model',required=True,choices=MODEL_TYPES)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split',choices=('validation',),default='validation')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    return p


if __name__=='__main__':
    train_once(get_parser().parse_args())
