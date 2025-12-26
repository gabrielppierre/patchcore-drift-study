# Dynamic Memory for PatchCore under Covariate Shift

This repository contains the code to reproduce the main experiments from the paper:

> **"Enhancing PatchCore Robustness to Covariate Shift with Dynamic Memory"**  
> Gabriel Coelho, Cleber Zanchettin  
> Centro de Informática, Universidade Federal de Pernambuco

## 📋 Overview

We evaluate the effectiveness of a lightweight FIFO-based dynamic memory mechanism for mitigating covariate shift in PatchCore-based anomaly detection. The experiments focus on 5 challenging MVTec AD categories: `capsule`, `grid`, `leather`, `screw`, and `wood`.

## 🛠️ Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX 4060 8GB)
- ~10GB disk space for MVTec AD dataset

### Installation

```bash
# Create conda environment
conda create -n patchcore-drift python=3.10 -y
conda activate patchcore-drift

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
code/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_baseline.py              # Step 1: Train baseline PatchCore
├── run_dynamic_memory.py        # Step 2: Evaluate dynamic memory
└── utils/
    ├── __init__.py
    ├── memory.py                # GPU profiling utilities
    └── metrics_ci.py            # Confidence interval computation
```

## 📊 Dataset

Download MVTec AD from [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract to `data/mvtec_anomaly_detection/`.

Expected structure:
```
data/mvtec_anomaly_detection/
├── bottle/
├── cable/
├── capsule/
├── ...
└── zipper/
```

## 🚀 Running Experiments

### Step 1: Train Baseline PatchCore (all 15 categories)

```bash
python run_baseline.py \
    --categories capsule grid leather screw wood \
    --seeds 42 43 44 45 46 \
    --data-root data/mvtec_anomaly_detection \
    --results-dir results/baseline
```

This creates checkpoints and metrics for each category/seed combination.

### Step 2: Evaluate Dynamic Memory under Drift

```bash
python run_dynamic_memory.py \
    --categories capsule grid leather screw wood \
    --seeds 42 43 44 45 46 \
    --data-root data/mvtec_anomaly_detection \
    --baseline-root results/baseline/Patchcore/MVTecAD \
    --output-root results/dynamic_memory
```

This evaluates:
- **Baseline (Clean)**: Original model on clean test data
- **Baseline (Drift)**: Original model on drift-augmented test data  
- **Dynamic (Drift)**: Model trained with buffer augmentation, tested on drift data

### Ablation: Buffer Strategies

```bash
# No buffer augmentation
python run_dynamic_memory.py --buffer-mode none --output-root results/ablations/no_buffer

# Strong buffer augmentation
python run_dynamic_memory.py --buffer-mode strong --output-root results/ablations/strong_buffer
```

## 📈 Output Format

Results are saved as JSON files with confidence intervals:

```
results/dynamic_memory/capsule/
├── seed_42/metrics.json
├── seed_43/metrics.json
├── ...
└── summary_with_ci.json    # Aggregated metrics with 95% CI
```

Example `summary_with_ci.json`:
```json
{
  "baseline_clean": {
    "image_AUROC": {"mean": 0.992, "std": 0.002, "ci95_low": 0.990, "ci95_high": 0.994}
  },
  "baseline_drift": {
    "image_AUROC": {"mean": 0.711, "std": 0.063, "ci95_low": 0.648, "ci95_high": 0.774}
  },
  "dynamic_drift": {
    "image_AUROC": {"mean": 0.944, "std": 0.018, "ci95_low": 0.926, "ci95_high": 0.962}
  }
}
```

## 🔧 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--coreset-sampling-ratio` | 0.1 | Fraction of patches for memory bank |
| `--train-batch-size` | 16 | Batch size for training |
| `--eval-batch-size` | 16 | Batch size for evaluation |
| `--num-workers` | 4 | DataLoader workers |
| `--buffer-mode` | default | Buffer augmentation: `default`, `none`, `strong` |

## 📝 Drift Protocol

We simulate covariate shift using intensity-based augmentations applied to validation/test data:

```python
DRIFT_TRANSFORM = T.Compose([
    T.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.02),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.4, 1.2)),
    T.RandomAdjustSharpness(sharpness_factor=0.6, p=0.8),
])
```

This simulates lighting changes, sensor noise, and focus variations common in industrial settings.

## 📖 Citation

If you use this code, please cite:

```bibtex
@inproceedings{coelho2026patchcore,
  title={Enhancing PatchCore Robustness to Covariate Shift with Dynamic Memory},
  author={Coelho, Gabriel and Zanchettin, Cleber},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2026}
}
```

## 📄 License

This code is released for academic research purposes. The MVTec AD dataset has its own license terms.
