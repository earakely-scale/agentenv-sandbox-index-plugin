"""Per-provider statistics from raw rows, with bootstrap 95% intervals. Resampling is by hour (the hour a sample
started), so samples taken close together count as one draw, as the methodology specifies. A run inside a single hour
has one block, which would give a zero-width interval, so it is resampled by sample and says so. A success rate whose
bootstrap still has zero width (no failures at all) takes a Wilson interval instead, which stays honest at 100%."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .methodology import BOOTSTRAP_RESAMPLES
from .rows import Row

# The statistics each measured metric contributes: (statistic name, a percentile of the passing samples' values, or
# None for the pooled success rate).
STATISTICS: dict[str, tuple[tuple[str, float | None], ...]] = {"S1": (("s1", 50), ("s1p", 95)), "R1": (("r1", None),)}


@dataclass
class Statistic:
    name: str
    value: float
    lo: float
    hi: float
    resampled_by: Literal["hour", "sample", "wilson"]
    blocks: int


@dataclass
class Summary:
    provider: str
    metric: str
    samples: int
    failed: int
    discarded: int
    statistics: list[Statistic]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def summarize(rows: list[Row], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> list[Summary]:
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r.provider, r.metric)].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for (provider, metric), group in sorted(groups.items()):
        counted = [r for r in group if r.outcome != "discarded"]
        ok = [r for r in counted if r.outcome == "ok"]
        by_hour: dict[str, list[Row]] = defaultdict(list)
        for r in counted:
            by_hour[r.started_at[:13]].append(r)
        by: Literal["hour", "sample"] = "hour" if len(by_hour) > 1 else "sample"
        blocks = list(by_hour.values()) if by == "hour" else [[r] for r in counted]
        picks = rng.integers(len(blocks), size=(resamples, len(blocks))) if blocks else None
        stats = []
        for name, q in STATISTICS.get(metric, ()):
            if q is None and counted:
                passed = np.array([sum(r.outcome == "ok" for r in b) for b in blocks])
                sizes = np.array([len(b) for b in blocks])
                draws = 100 * passed[picks].sum(1) / sizes[picks].sum(1)
                value = 100 * len(ok) / len(counted)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                if math.isclose(lo, hi):
                    stats.append(Statistic(name, value, *wilson(len(ok), len(counted)), "wilson", len(counted)))
                    continue
            elif q is not None and ok:
                values = [np.array([r.value for r in b if r.outcome == "ok"]) for b in blocks]
                draws = [np.percentile(d, q) for d in (np.concatenate([values[j] for j in p]) for p in picks) if d.size]
                value = float(np.percentile([r.value for r in ok], q))
                lo, hi = np.percentile(draws, [2.5, 97.5])
            else:
                continue
            stats.append(Statistic(name, value, float(lo), float(hi), by, len(blocks)))
        out.append(Summary(provider, metric, len(group), len(counted) - len(ok), len(group) - len(counted), stats))
    return out
