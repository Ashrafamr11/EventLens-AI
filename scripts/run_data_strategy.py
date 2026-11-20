"""Controlled train-only feature-noise augmentation study for EventLens BiGRU.

Comparison: original 5-epoch BiGRU with unchanged features vs. BiGRU trained
with a 5%-of-per-sample-feature-STD Gaussian noise augmentation.  All 3 seeds
keep the same LLP train/val split, classifier, optimizer and temporal scorer.
Never touches the test split and never overwrites selected production weights.
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
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from eventlens.experiment import fit_model
from eventlens.temporal_gru_model import TemporalGRUModel
from eventlens.training_dataset import LLPDataset
from evaluate_event_errors import build_temporal_gt, get_events, match_events, normalize_video_id

SEEDS = (42, 43, 44)


class TrainOnlyFeatureNoise(Dataset):
    """Apply scale-adaptive Gaussian noise to training features only.

    The random draw is independent across calls and deterministic per worker
    given seeds supplied to the PyTorch DataLoader.  Validation is untouched.
    """
    def __init__(self, base: Dataset, noise_scale: float = 0.05):
        if not 0 <= noise_scale <= 1:
            raise ValueError("noise_scale must be in [0,1]")
        self.base = base
        self.noise_scale = float(noise_scale)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        source = self.base[index]
        result = dict(source)
        for name in ("audio", "visual", "motion"):
            x = np.asarray(source[name], dtype=np.float32)
            if x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError(f"Invalid {name} feature tensor")
            scale = self.noise_scale * max(float(np.std(x)), 1e-6)
            noise = np.random.normal(0.0, scale, size=x.shape).astype(np.float32)
            result[name] = x + noise
        return result


def update_counts(counts: np.ndarray, pred: np.ndarray, actual: np.ndarray, axes):
    counts[0] += (pred & actual).sum(axis=axes)
    counts[1] += (pred & ~actual).sum(axis=axes)
    counts[2] += (~pred & actual).sum(axis=axes)


def metric_summary(counts: np.ndarray) -> dict:
    tp, fp, fn = counts
    t, f, n = int(tp.sum()), int(fp.sum()), int(fn.sum())
    precision = t / max(1, t + f)
    recall = t / max(1, t + n)
    micro = 2 * t / max(1, 2 * t + f + n)
    macro = float(np.mean(2 * tp / np.maximum(1, 2 * tp + fp + fn)))
    return {"precision": float(precision), "recall": float(recall), "micro_f1": float(micro), "macro_f1": macro, "tp": t, "fp": f, "fn": n}


def evaluate(model, dataset, gt: np.ndarray, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    num_classes = gt.shape[-1]
    counts = {level: np.zeros((3, num_classes), dtype=np.int64) for level in ("clip", "segment", "event")}
    model.eval()
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["audio"].to(device), batch["visual"].to(device), batch["motion"].to(device))
            clip_pred = (torch.sigmoid(outputs["clip_logits"]).cpu().numpy() >= 0.5)
            clip_gt = batch["labels"].cpu().numpy().astype(bool)
            segment_pred = (torch.sigmoid(outputs["segment_logits"]).cpu().numpy() >= 0.5)
            segment_gt = gt[offset:offset+len(segment_pred)]
            if segment_pred.shape != segment_gt.shape:
                raise ValueError("Validation feature/ground truth shape mismatch")
            update_counts(counts["clip"], clip_pred, clip_gt, (0,))
            update_counts(counts["segment"], segment_pred, segment_gt, (0,1))
            for batch_idx in range(len(segment_pred)):
                for class_idx in range(num_classes):
                    predicted_events = get_events(segment_pred[batch_idx,:,class_idx])
                    actual_events = get_events(segment_gt[batch_idx,:,class_idx])
                    matched, fps, fns = match_events(predicted_events, actual_events, iou_threshold=0.50)
                    counts["event"][:,class_idx] += (len(matched), len(fps), len(fns))
            offset += len(segment_pred)
    if offset != len(dataset):
        raise ValueError(f"Validation incomplete: {offset}/{len(dataset)}")
    return {level: metric_summary(c) for level,c in counts.items()}


def run(artifact_root: Path, project_root: Path, noise_scale: float=0.05, device_name: str="auto", only_seed: int | None=None) -> dict:
    annotation_dir = project_root / "data/raw/llp"
    features_dir = project_root / "data/processed/feats"
    base_summary = artifact_root / "experiments/multi_seed_validation_metrics.json"
    if not base_summary.is_file():
        raise FileNotFoundError(f"Need original saved validation benchmark: {base_summary}")
    original = json.loads(base_summary.read_text(encoding="utf-8"))
    original_runs = original["individual_runs"]["bigru"]
    original_params = (original.get("threshold"), original.get("event_iou_threshold"), original.get("ground_truth"))
    if original_params != (0.5, 0.5, "audio_visual_union"):
        raise ValueError(f"Original metric protocol mismatch: {original_params}")

    ref_ckpt = artifact_root / "experiments/multi_seed/bigru_seed44.pt"
    if not ref_ckpt.is_file():
        raise FileNotFoundError(f"Missing reference BiGRU checkpoint: {ref_ckpt}")
    ref = torch.load(ref_ckpt, map_location="cpu", weights_only=True)
    class_names = list(ref["class_names"])
    if len(class_names) != 25:
        raise ValueError("Expected 25 class names")
    train = LLPDataset("train", class_names, data_dir=annotation_dir, features_dir=features_dir)
    val = LLPDataset("validation", class_names, data_dir=annotation_dir, features_dir=features_dir)
    if len(train) != 10000 or len(val) != 649:
        raise ValueError(f"Expected 10000/649 examples, got {len(train)}/{len(val)}")
    first = train[0]
    if first["audio"].shape != (10,128) or first["visual"].shape != (10,2048) or first["motion"].shape != (10,512):
        raise ValueError("Saved feature shape is incorrect")

    ids = [normalize_video_id(row["video_id"]) for row in val.samples]
    gt, invalid_rows = build_temporal_gt(annotation_dir, ids, class_names)
    if invalid_rows != 1:
        raise ValueError(f"Expected 1 invalid validation annotation row as in original notebook, got {invalid_rows}")
    print(f"STUDY DATA READY: 10000 training / 649 validation; {invalid_rows} invalid GT row", flush=True)

    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else
                          "cpu" if device_name == "auto" else device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"STUDY DEVICE: {device}; will not require a GPU", flush=True)
    output_dir = artifact_root / "experiments/data_strategy"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    augmented_runs = {}
    for seed in ((only_seed,) if only_seed is not None else SEEDS):
        chk = output_dir / f"bigru_noise05_seed{seed}.pt"
        log_path = output_dir / f"bigru_noise05_seed{seed}.json"
        if log_path.is_file():
            rec = json.loads(log_path.read_text(encoding="utf-8"))
            if chk.is_file() and rec.get("noise_scale") == noise_scale and rec.get("seed") == seed and "metrics" in rec:
                augmented_runs[str(seed)] = rec
                print(f"SKIPPED seed {seed} (existing valid result)", flush=True)
                continue

        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        loader = DataLoader(TrainOnlyFeatureNoise(train, noise_scale), batch_size=64,
                            shuffle=True, num_workers=2 if device.type == "cuda" else 0, pin_memory=(device.type == "cuda"),
                            generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=2 if device.type == "cuda" else 0, pin_memory=(device.type == "cuda"))
        model = TemporalGRUModel(num_classes=len(class_names), hidden_dim=128).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        print(f"TRAINING NOISE AUGMENTATION SEED {seed}",flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        start = time.perf_counter()
        history = fit_model(model,loader,val_loader,optim,device,epochs=5,checkpoint_path=chk,class_names=class_names)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        peak_mib = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type == "cuda" else None
        chosen=torch.load(chk,map_location="cpu",weights_only=True)
        model.load_state_dict(chosen["model_state_dict"],strict=True)
        metrics = evaluate(model,val,gt,device)
        rec={"study":"train_only_scale_adaptive_gaussian_noise", "seed":seed, "model":"bigru", "noise_scale":noise_scale,
             "epochs":5,"batch_size":64,"optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0001,
             "train_count":len(train),"validation_count":len(val),"validation_invalid_annotation_rows":invalid_rows,
             "threshold":0.5,"event_iou_threshold":0.5,"annotation_union":"audio_visual_union",
             "hardware":("CPU" if device.type == "cpu" else torch.cuda.get_device_name(device)),"training_seconds":elapsed,"gpu_training_peak_mib":peak_mib,
             "best_val_loss":float(chosen["val_loss"]),"history":history,"metrics":metrics}
        log_path.write_text(json.dumps(rec,indent=2),encoding="utf-8")
        augmented_runs[str(seed)] = rec
        print(f"SAVED SEED {seed} | Val event F1 {metrics['event']['micro_f1']:.4f}", flush=True)
        del model, optim
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for saved_seed in SEEDS:
        if str(saved_seed) not in augmented_runs:
            file = output_dir / f"bigru_noise05_seed{saved_seed}.json"
            checkpoint = output_dir / f"bigru_noise05_seed{saved_seed}.pt"
            if file.is_file() and checkpoint.is_file():
                previous = json.loads(file.read_text(encoding="utf-8"))
                if previous.get("noise_scale") == noise_scale and "metrics" in previous:
                    augmented_runs[str(saved_seed)] = previous
    if len(augmented_runs) < len(SEEDS):
        print(f"SEED {only_seed} SAVED TO DRIVE; run the other seeds to complete 3-seed comparison", flush=True)
        return {"completed_seed": only_seed, "result_file": str(log_path)}

    comparison={}
    for level in ("clip","segment","event"):
        orig = [original_runs[str(seed)][level]["micro_f1"] for seed in SEEDS]
        aug = [augmented_runs[str(seed)]["metrics"][level]["micro_f1"] for seed in SEEDS]
        comparison[level]={"clean_original_mean":float(np.mean(orig)),"clean_original_sd":float(np.std(orig,ddof=1)),
                           "train_noise_mean":float(np.mean(aug)),"train_noise_sd":float(np.std(aug,ddof=1)),
                           "delta_percentage_points":float((np.mean(aug)-np.mean(orig))*100),
                           "per_seed": {str(seed):{"clean":orig[i],"augmented":aug[i]} for i,seed in enumerate(SEEDS)}}
        print(f"{level.upper():8s} ORIGINAL {np.mean(orig):.4f} ± {np.std(orig,ddof=1):.4f} | AUGMENTED {np.mean(aug):.4f} ± {np.std(aug,ddof=1):.4f} | Δ {comparison[level]['delta_percentage_points']:+.2f} pp",flush=True)

    result={"study":"Data strategy: train-only feature Gaussian noise", "models":"BiGRU vs original BiGRU baseline", "seeds":list(SEEDS),
            "noise_scale_relative_to_sample_std":noise_scale,"same_validation_split":True,"test_set_used":False,
            "original_results_source":str(base_summary),"comparison":comparison,
            "conclusion":"Improvement or degradation must be interpreted from the measured delta, not assumed."}
    (output_dir/"data_strategy_validation_comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DATA STRATEGY SUCCESS", flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact-root",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--noise-scale",type=float,default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    args=parser.parse_args()
    run(args.artifact_root.expanduser().resolve(),args.project_root.expanduser().resolve(),args.noise_scale,args.device,args.only_seed)
