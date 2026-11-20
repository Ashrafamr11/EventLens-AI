"""Validate the 3-seed EventLens train-only noise study and export report tables.

This tool reads existing JSON only. No training, model loading, or test-set use.
Usage: python scripts/verify_data_strategy_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

LEVELS = ("clip", "segment", "event")
SEEDS = (42, 43, 44)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing study result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, name: str, tol: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise ValueError(f"{name}: recorded {actual} != recomputed {expected}")


def validate(results_dir: Path) -> tuple[dict, list[dict]]:
    result = read_json(results_dir / "data_strategy_validation_comparison.json")
    if result.get("seeds") != list(SEEDS):
        raise ValueError("Expected exactly seeds [42, 43, 44]")
    if result.get("same_validation_split") is not True or result.get("test_set_used") is not False:
        raise ValueError("Study is not identified as validation-only on the same split")
    close(result["noise_scale_relative_to_sample_std"], .05, "Noise scale")
    runs = []
    for seed in SEEDS:
        record = read_json(results_dir / f"bigru_noise05_seed{seed}.json")
        if record.get("seed") != seed or record.get("model") != "bigru":
            raise ValueError(f"Wrong model or seed for {seed}")
        if record.get("train_count") != 10000 or record.get("validation_count") != 649:
            raise ValueError(f"Unexpected split sizes for seed {seed}")
        if (record.get("threshold"), record.get("event_iou_threshold"), record.get("annotation_union")) != (.5, .5, "audio_visual_union"):
            raise ValueError(f"Metric protocol mismatch for seed {seed}")
        if (record.get("epochs"), record.get("batch_size"), record.get("optimizer")) != (5,64,"AdamW"):
            raise ValueError(f"Training setup mismatch for seed {seed}")
        close(record["noise_scale"],.05,f"Seed {seed} noise scale")
        for level in LEVELS:
            metric=record["metrics"][level]
            tp,fp,fn=(int(metric[k]) for k in ("tp","fp","fn"))
            if min(tp,fp,fn)<0:
                raise ValueError(f"Negative counts in {seed}/{level}")
            close(metric["micro_f1"],2*tp/max(1,2*tp+fp+fn),f"Seed {seed}/{level} F1")
        runs.append(record)

    summary_rows=[]
    per_seed=[]
    for level in LEVELS:
        data=result["comparison"][level]
        clean=[float(data["per_seed"][str(s)]["clean"]) for s in SEEDS]
        augmented=[float(r["metrics"][level]["micro_f1"]) for r in runs]
        for seed,clean_f1,aug_f1 in zip(SEEDS,clean,augmented):
            close(data["per_seed"][str(seed)]["augmented"],aug_f1,f"Seed {seed}/{level}")
            per_seed.append({"metric":level,"seed":seed,"original_f1":clean_f1,"noise_f1":aug_f1,
                             "delta_pp":(aug_f1-clean_f1)*100})
        clean_mean,aug_mean=statistics.mean(clean),statistics.mean(augmented)
        clean_sd,aug_sd=statistics.stdev(clean),statistics.stdev(augmented)
        close(data["clean_original_mean"],clean_mean,f"{level} original mean")
        close(data["clean_original_sd"],clean_sd,f"{level} original std")
        close(data["train_noise_mean"],aug_mean,f"{level} noise mean")
        close(data["train_noise_sd"],aug_sd,f"{level} noise std")
        close(data["delta_percentage_points"],(aug_mean-clean_mean)*100,f"{level} delta")
        summary_rows.append({"metric":level,"original_mean_f1":clean_mean,"original_sd":clean_sd,
                             "noise_mean_f1":aug_mean,"noise_sd":aug_sd,"delta_pp":(aug_mean-clean_mean)*100})
    return result, [summary_rows,per_seed]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=Path("experiments/data_strategy"))
    parser.add_argument("--output-dir",type=Path,default=Path("experiments/analysis"))
    args=parser.parse_args()
    _, (summaries,per_seed)=validate(args.results_dir)
    write_csv(args.output_dir/"data_strategy_validation.csv",summaries)
    write_csv(args.output_dir/"data_strategy_by_seed.csv",per_seed)
    print("DATA STRATEGY RESULTS VERIFIED")
    for row in summaries:
        print(f"{row['metric']:<8} original {row['original_mean_f1']:.4f} ± {row['original_sd']:.4f} | "
              f"noise {row['noise_mean_f1']:.4f} ± {row['noise_sd']:.4f} | "
              f"change {row['delta_pp']:+.2f} pp")
    print("CSV tables:",args.output_dir/"data_strategy_validation.csv",",",args.output_dir/"data_strategy_by_seed.csv")


if __name__=="__main__":
    main()
