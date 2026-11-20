"""Shared validation/test evaluator for all four EventLens localization heads.

Uses the project's custom audio+visual-union one-to-one temporal IoU metric.
Never substitutes for the official modality-specific AVVP evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from eventlens.training_dataset import LLPDataset
from evaluate_event_errors import build_temporal_gt, normalize_video_id
from run_data_strategy import evaluate
from train import make_model,read_class_names,select_device


def evaluate_checkpoint(args):
    if args.split=='test' and not args.allow_test:
        raise RuntimeError('Test split is held out; explicitly pass --allow-test only for final evaluation.')
    device = select_device(args.device)
    names=read_class_names(args.checkpoint)
    project=args.project_root.resolve()
    ds=LLPDataset(args.split, names, data_dir=project/'data/raw/llp',
                  features_dir=project/'data/processed/feats')
    expected={'validation':649,'test':1200}[args.split]
    if len(ds)!=expected:
        raise ValueError(f'{args.split}: expected {expected}, got {len(ds)}')
    model=make_model(args.model,len(names)).to(device)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    ids=[normalize_video_id(row['video_id']) for row in ds.samples]
    gt, invalid_rows=build_temporal_gt(project/'data/raw/llp',ids,names)
    metrics=evaluate(model,ds,gt,device,batch_size=args.batch_size)
    out={'model':args.model,'checkpoint':str(args.checkpoint.resolve()),'split':args.split,
         'custom_metric':'pooled audio-visual union', 'threshold':0.5,
         'temporal_iou_threshold':0.5,'invalid_annotation_rows':invalid_rows,
         'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
         'metrics':metrics}
    print(json.dumps(out,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--model',choices=('baseline','bigru','transformer','cross_modal'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--allow-test',action='store_true')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    evaluate_checkpoint(args)
