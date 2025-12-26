"""Dynamic memory experiment runner with multi-seed support.

This script evaluates PatchCore under simulated covariate shift (drift),
comparing baseline performance against a model trained with dynamic
buffer augmentation.

Usage:
    python run_dynamic_memory.py --categories capsule grid leather screw wood
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
from torchvision.transforms import v2 as T

from utils.metrics_ci import aggregate_samples, save_summary
from utils.memory import run_with_profiling

DEFAULT_CATEGORIES = ["screw", "grid", "leather", "wood", "capsule"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# -----------------------------------------------------------------------------
# Drift Simulation Transforms
# -----------------------------------------------------------------------------

# Applied to validation/test data to simulate environmental drift
DRIFT_TRANSFORM = T.Compose([
    T.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.02),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.4, 1.2)),
    T.RandomAdjustSharpness(sharpness_factor=0.6, p=0.8),
])

# Default buffer augmentation for dynamic memory training
DEFAULT_BUFFER_TRANSFORM = T.Compose([
    T.RandomApply(
        transforms=[
            T.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.02),
            T.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 1.0)),
        ],
        p=0.6,
    )
])

# Strong buffer augmentation (for ablation study)
STRONG_BUFFER_TRANSFORM = T.Compose([
    T.RandomApply(
        transforms=[
            T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.35, hue=0.05),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
            T.RandomAdjustSharpness(sharpness_factor=0.4, p=1.0),
        ],
        p=0.85,
    )
])


def get_buffer_transform(mode: str) -> T.Transform | None:
    """Get buffer transform based on ablation mode."""
    if mode == "none":
        return None
    if mode == "strong":
        return STRONG_BUFFER_TRANSFORM
    return DEFAULT_BUFFER_TRANSFORM


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datamodule(
    category: str,
    *,
    data_root: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    train_aug: T.Transform | None = None,
    val_aug: T.Transform | None = None,
    test_aug: T.Transform | None = None,
) -> MVTecAD:
    """Create MVTecAD datamodule with optional augmentations."""
    return MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_augmentations=train_aug,
        val_augmentations=val_aug,
        test_augmentations=test_aug,
    )


def find_baseline_checkpoint(baseline_root: Path, category: str, seed: int) -> Path:
    """Locate the trained baseline checkpoint for a given category and seed."""
    base = baseline_root / category / f"seed_{seed}" / "Patchcore" / "MVTecAD" / category
    candidates = [
        base / "latest" / "weights" / "lightning" / "model.ckpt",
        base / "v0" / "weights" / "lightning" / "model.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Checkpoint not found for category '{category}' and seed {seed} at {base}. "
        "Please run the baseline experiment first."
    )


def evaluate_model(
    engine: Engine, 
    model: Patchcore, 
    datamodule: MVTecAD
) -> tuple[Dict[str, float], float]:
    """Evaluate model and return metrics with timing."""
    results, duration, peak_gpu_mem = run_with_profiling(
        lambda: engine.test(model=model, datamodule=datamodule)
    )
    metrics = results[0] if results else {}
    metrics = metrics.copy()
    metrics["test_seconds"] = duration
    metrics["test_peak_gpu_mem_mib"] = peak_gpu_mem
    return metrics, duration


def run_single_seed(
    *,
    category: str,
    seed: int,
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    coreset_sampling_ratio: float,
    buffer_transform: T.Transform | None,
) -> Dict[str, Dict[str, float]]:
    """Run complete experiment for a single seed.
    
    Evaluates:
    - baseline_clean: Baseline model on clean test data
    - baseline_drift: Baseline model on drift-augmented test data
    - dynamic_drift: Dynamic model trained with buffer augmentation, tested on drift
    """
    
    seed_everything(seed)
    
    # Create datamodules for different conditions
    clean_dm = build_datamodule(
        category,
        data_root=data_root,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_aug=None,
        val_aug=None,
        test_aug=None,
    )
    
    drift_dm = build_datamodule(
        category,
        data_root=data_root,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        val_aug=DRIFT_TRANSFORM,
        test_aug=DRIFT_TRANSFORM,
    )
    
    dynamic_dm = build_datamodule(
        category,
        data_root=data_root,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        train_aug=buffer_transform,
    )

    # Load baseline checkpoint
    ckpt_path = find_baseline_checkpoint(baseline_root, category, seed)
    print(f"[Seed {seed}] Loading baseline checkpoint: {ckpt_path}")
    baseline_model = Patchcore.load_from_checkpoint(str(ckpt_path))

    # Evaluate baseline on clean and drift conditions
    eval_engine = Engine(accelerator="auto", devices=1, max_epochs=1)
    
    print(f"[Seed {seed}] Evaluating baseline (clean)...")
    baseline_clean, _ = evaluate_model(eval_engine, baseline_model, clean_dm)
    
    print(f"[Seed {seed}] Evaluating baseline (drift)...")
    baseline_drift, _ = evaluate_model(eval_engine, baseline_model, drift_dm)

    # Train and evaluate dynamic model
    dynamic_model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=coreset_sampling_ratio,
    )
    
    seed_dir = output_root / category / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    dynamic_engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        default_root_dir=str(seed_dir),
    )
    
    print(f"[Seed {seed}] Training PatchCore with dynamic buffer...")
    _, dynamic_train_seconds, dynamic_train_peak = run_with_profiling(
        lambda: dynamic_engine.fit(model=dynamic_model, datamodule=dynamic_dm)
    )

    print(f"[Seed {seed}] Evaluating dynamic model (drift)...")
    dynamic_drift, _ = evaluate_model(dynamic_engine, dynamic_model, drift_dm)
    dynamic_drift = dynamic_drift.copy()
    dynamic_drift["train_seconds"] = dynamic_train_seconds
    dynamic_drift["train_peak_gpu_mem_mib"] = dynamic_train_peak

    # Compile results
    results = {
        "baseline_clean": baseline_clean,
        "baseline_drift": baseline_drift,
        "dynamic_drift": dynamic_drift,
        "seed": seed,
        "category": category,
    }
    
    with (seed_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    
    return results


def aggregate_category(
    category: str, 
    metrics_per_seed: List[Dict[str, Dict[str, float]]], 
    output_root: Path
) -> None:
    """Aggregate metrics across seeds with confidence intervals."""
    filtered = []
    for entry in metrics_per_seed:
        filtered.append({k: v for k, v in entry.items() if k not in {"seed"}})
    summary = aggregate_samples(filtered)
    save_summary(summary, output_root / category / "summary_with_ci.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic memory experiments with multiple seeds"
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
        "--baseline-root",
        type=Path,
        default=Path("results/baseline/Patchcore/MVTecAD"),
        help="Directory containing baseline checkpoints",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/dynamic_memory/Patchcore/MVTecAD"),
        help="Output directory for results",
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1)
    parser.add_argument(
        "--buffer-mode",
        choices=["default", "none", "strong"],
        default="default",
        help="Buffer augmentation mode (for ablation studies)",
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Remove existing category directories"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"MVTec AD dataset not found at {args.data_root}. "
            "Please download from https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )
    
    if not args.baseline_root.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {args.baseline_root}. "
            "Please run run_baseline.py first."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    buffer_transform = get_buffer_transform(args.buffer_mode)
    
    print(f"Buffer mode: {args.buffer_mode}")
    print(f"Output directory: {args.output_root}")

    for category in args.categories:
        category_dir = args.output_root / category
        if args.clean and category_dir.exists():
            shutil.rmtree(category_dir)

        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        
        per_seed: List[Dict[str, Dict[str, float]]] = []
        for seed in args.seeds:
            try:
                results = run_single_seed(
                    category=category,
                    seed=seed,
                    data_root=args.data_root,
                    baseline_root=args.baseline_root,
                    output_root=args.output_root,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    coreset_sampling_ratio=args.coreset_sampling_ratio,
                    buffer_transform=buffer_transform,
                )
                per_seed.append(results)
                
                # Print summary for this seed
                bc = results["baseline_clean"].get("image_AUROC", 0)
                bd = results["baseline_drift"].get("image_AUROC", 0)
                dd = results["dynamic_drift"].get("image_AUROC", 0)
                print(f"[Seed {seed}] AUROC: Clean={bc:.3f} | Drift={bd:.3f} | Dynamic={dd:.3f}")
                
            except Exception as exc:
                print(f"[ERROR][{category}][seed {seed}] {exc}")

        if per_seed:
            aggregate_category(category, per_seed, args.output_root)
            print(f"\nSaved aggregated metrics to {args.output_root / category / 'summary_with_ci.json'}")


if __name__ == "__main__":
    main()
