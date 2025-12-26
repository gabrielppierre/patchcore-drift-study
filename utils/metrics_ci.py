"""Utilities for aggregating experiment metrics with confidence intervals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class MetricStats:
    """Summary statistics for a single scalar metric."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "n": self.n,
        }


def _to_float_list(values: Iterable[float | int]) -> list[float]:
    data = [float(v) for v in values]
    if not data:
        raise ValueError("No values to aggregate.")
    return data


def bootstrap_ci(
    values: Sequence[float | int],
    *,
    num_samples: int = 1_000,
    ci: float = 0.95,
    rng_seed: int | None = 0,
) -> tuple[float, float]:
    """Return (low, high) percentile bounds for the mean via bootstrap."""

    data = _to_float_list(values)
    if len(data) == 1:
        return data[0], data[0]

    rng = np.random.default_rng(rng_seed)
    sample_means = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        sample = rng.choice(data, size=len(data), replace=True)
        sample_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    low = float(np.quantile(sample_means, alpha))
    high = float(np.quantile(sample_means, 1 - alpha))
    return low, high


def summarize_scalar(
    values: Sequence[float | int],
    *,
    num_bootstrap: int = 1_000,
    ci: float = 0.95,
    rng_seed: int | None = 0,
) -> MetricStats:
    """Compute mean, std, and 95% CI for a list of scalar values."""
    data = _to_float_list(values)
    if len(data) == 1:
        std = 0.0
    else:
        std = float(np.std(data, ddof=1))
    ci_low, ci_high = bootstrap_ci(data, num_samples=num_bootstrap, ci=ci, rng_seed=rng_seed)
    return MetricStats(mean=float(np.mean(data)), std=std, ci95_low=ci_low, ci95_high=ci_high, n=len(data))


def aggregate_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    num_bootstrap: int = 1_000,
    ci: float = 0.95,
    rng_seed: int | None = 0,
) -> dict[str, Any]:
    """Aggregate a list of nested metric dicts.

    Recursively traverses dictionaries. Scalars are summarized with mean/std/CI; nested
    mappings are aggregated field-by-field. Non-numeric leaves are copied from the first
    sample (useful for metadata such as category names).
    """

    if not samples:
        raise ValueError("Empty metrics list; nothing to aggregate.")

    aggregated: dict[str, Any] = {}
    keys = set().union(*(sample.keys() for sample in samples))
    for key in sorted(keys):
        values = [sample[key] for sample in samples if key in sample]
        first_value = values[0]
        if isinstance(first_value, Mapping):
            nested_samples = [value for value in values if isinstance(value, Mapping)]
            aggregated[key] = aggregate_samples(
                nested_samples,
                num_bootstrap=num_bootstrap,
                ci=ci,
                rng_seed=rng_seed,
            )
        elif isinstance(first_value, (int, float)):
            aggregated[key] = summarize_scalar(
                values,
                num_bootstrap=num_bootstrap,
                ci=ci,
                rng_seed=rng_seed,
            ).as_dict()
        else:
            aggregated[key] = first_value
    return aggregated


def save_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    """Save aggregated metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
