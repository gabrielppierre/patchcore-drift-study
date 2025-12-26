"""Helpers to profile execution time and GPU peak memory."""

from __future__ import annotations

import time
from typing import Callable, Tuple, TypeVar

import torch

T = TypeVar("T")


def _can_track_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def run_with_profiling(func: Callable[[], T]) -> Tuple[T, float, float | None]:
    """Execute ``func`` while measuring wall time and peak GPU memory.

    Returns a tuple ``(result, seconds, peak_mib)`` where ``peak_mib`` is ``None``
    if CUDA is not available.
    """

    use_cuda = _can_track_cuda()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = func()
    seconds = time.perf_counter() - start

    peak_mib: float | None = None
    if use_cuda:
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_reserved()
        peak_mib = peak_bytes / (1024 ** 2)

    return result, seconds, peak_mib
