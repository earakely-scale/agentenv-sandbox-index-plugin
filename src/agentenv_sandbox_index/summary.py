"""Per-provider statistics from raw rows, with bootstrap 95% intervals. Resampling is by hour (the hour a sample
started), so samples taken close together count as one draw, as the methodology specifies. A run inside a single hour
has one block, which would give a zero-width interval, so it is resampled by sample and says so."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .methodology import BOOTSTRAP_RESAMPLES
from .rows import Row

# The statistics each measured metric contributes, as (statistic name, percentile).
STATISTICS: dict[str, tuple[tuple[str, float], ...]] = {"S1": (("s1", 50), ("s1p", 95))}


@dataclass
class Statistic:
    name: str
    value: float
    lo: float
    hi: float
    resampled_by: Literal["hour", "sample"]
    blocks: int


@dataclass
class Summary:
    provider: str
    metric: str
    samples: int
    failed: int
    statistics: list[Statistic]


def summarize(rows: list[Row], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> list[Summary]:
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.provider, r.metric)].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for (provider, metric), group in sorted(groups.items()):
        ok = [r for r in group if r.outcome == "ok" and r.value is not None]
        by_hour: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            by_hour[r.started_at[:13]].append(r.value)
        by: Literal["hour", "sample"] = "hour" if len(by_hour) > 1 else "sample"
        blocks = [np.array(v) for v in by_hour.values()] if by == "hour" else [np.array([r.value]) for r in ok]
        stats = []
        for name, q in STATISTICS.get(metric, ()):
            if not ok:
                continue
            value = float(np.percentile([r.value for r in ok], q))
            draws = np.empty(resamples)
            for i in range(resamples):
                pick = rng.integers(len(blocks), size=len(blocks))
                draws[i] = np.percentile(np.concatenate([blocks[j] for j in pick]), q)
            lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
            stats.append(Statistic(name, value, lo, hi, by, len(blocks)))
        out.append(Summary(provider, metric, len(group), len(group) - len(ok), stats))
    return out
