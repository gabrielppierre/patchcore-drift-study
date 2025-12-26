"""Baseline PatchCore runner with multi-seed support and CI aggregation.

This script trains PatchCore on MVTec AD categories and saves checkpoints
that will be used as the baseline for dynamic memory experiments.

Usage:
    python run_baseline.py --categories capsule grid leather screw wood
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from utils.metrics_ci import aggregate_samples, save_summary
from utils.memory import run_with_profiling

DEFAULT_CATEGORIES: List[str] = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_seed(
    *,
    category: str,
    seed: int,
    data_root: Path,
    results_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    coreset_sampling_ratio: float,
) -> Dict[str, float]:
    """Train and evaluate PatchCore for a single category and seed."""
    
    seed_everything(seed)
    
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=coreset_sampling_ratio,
    )

    seed_dir = results_dir / category / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        default_root_dir=str(seed_dir),
    )

    print(f"[Seed {seed}] Building memory bank...")
    _, train_seconds, train_peak_gpu_mem = run_with_profiling(
        lambda: engine.fit(model=model, datamodule=datamodule)
    )

    print(f"[Seed {seed}] Computing metrics...")
    results, test_seconds, test_peak_gpu_mem = run_with_profiling(
        lambda: engine.test(model=model, datamodule=datamodule)
    )

    metrics = results[0] if results else {}
    metrics.update({
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "train_peak_gpu_mem_mib": train_peak_gpu_mem,
        "test_peak_gpu_mem_mib": test_peak_gpu_mem,
        "seed": seed,
        "category": category,
    })
    
    with (seed_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    
    return metrics


def aggregate_category_metrics(
    category: str, 
    metrics_per_seed: List[Dict[str, float]], 
    results_dir: Path
) -> None:
    """Aggregate metrics across seeds with confidence intervals."""
    filtered = []
    for entry in metrics_per_seed:
        filtered.append({k: v for k, v in entry.items() if k not in {"seed"}})
    summary = aggregate_samples(filtered)
    save_summary(summary, results_dir / category / "summary_with_ci.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PatchCore baseline with multiple seeds"
    )
    parser.add_argument(
        "--categories", 
        nargs="*", 
        default=DEFAULT_CATEGORIES,
        help="MVTec AD categories to process"
    )
    parser.add_argument(
        "--seeds", 
        nargs="*", 
        type=int, 
        default=DEFAULT_SEEDS,
        help="Random seeds for reproducibility"
    )
    parser.add_argument(
        "--data-root", 
        type=Path, 
        default=Path("data/mvtec_anomaly_detection"),
        help="Path to MVTec AD dataset"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baseline/Patchcore/MVTecAD"),
        help="Output directory for checkpoints and metrics",
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing category directories before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"MVTec AD dataset not found at {args.data_root}. "
            "Please download from https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    for category in args.categories:
        category_dir = args.results_dir / category
        if args.clean and category_dir.exists():
            shutil.rmtree(category_dir)

        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        
        per_seed_metrics: List[Dict[str, float]] = []
        for seed in args.seeds:
            try:
                metrics = run_single_seed(
                    category=category,
                    seed=seed,
                    data_root=args.data_root,
                    results_dir=args.results_dir,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    coreset_sampling_ratio=args.coreset_sampling_ratio,
                )
                per_seed_metrics.append(metrics)
                print(f"[Seed {seed}] Image AUROC: {metrics.get('image_AUROC', 'N/A'):.4f}")
            except Exception as exc:
                print(f"[ERROR][{category}][seed {seed}] {exc}")

        if per_seed_metrics:
            aggregate_category_metrics(category, per_seed_metrics, args.results_dir)
            print(f"\nSaved aggregated metrics to {args.results_dir / category / 'summary_with_ci.json'}")


if __name__ == "__main__":
    main()
