"""Wall-clock timing with device synchronisation."""

from __future__ import annotations

import dataclasses
import time

import jax
import numpy as np


@dataclasses.dataclass(frozen=True)
class Timing:
    median_s: float
    min_s: float
    mean_s: float
    std_s: float
    iters: int

    @property
    def median_ms(self) -> float:
        return self.median_s * 1e3


def timeit(fn, *args, warmup: int = 2, iters: int = 10, **kw) -> Timing:
    """Time ``fn(*args, **kw)``; the first ``warmup`` calls (compile included) are discarded."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args, **kw))
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args, **kw))
        samples.append(time.perf_counter() - t)
    s = np.asarray(samples)
    return Timing(float(np.median(s)), float(s.min()), float(s.mean()), float(s.std()), iters)
