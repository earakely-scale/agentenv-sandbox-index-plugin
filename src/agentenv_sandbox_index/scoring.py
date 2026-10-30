"""From each provider's metric statistics to the index: 0–100 metric scores against fixed reference points, weighted
geometric means into sub-indices and the index, the reliability cap, rank bands and the weight stress test."""

import math
import random
from dataclasses import dataclass, field

from .methodology import (
    METRICS,
    PARTS,
    RELIABILITY_CAP,
    RELIABILITY_FLOOR,
    STRESS_ALPHA,
    STRESS_DRAWS,
    SUB_INDICES,
    WEIGHTS,
    SubIndex,
)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def nines(pct: float) -> float:
    return -math.log10(1 - pct / 100)


def metric_score(metric_id: str, value: float) -> float:
    """A statistic's 0–100 score, floored at 1 so one metric can't zero a geometric mean."""
    m = METRICS[metric_id]
    if m.kind == "time":
        s = (math.log(m.worst) - math.log(value)) / (math.log(m.worst) - math.log(m.best))
    elif m.kind == "rate":
        s = (math.log(value) - math.log(m.worst)) / (math.log(m.best) - math.log(m.worst))
    else:
        s = (nines(value) - 1) / 3
    return max(1.0, 100 * _clip(s))


def geo_mean(parts: list[tuple[float, float]]) -> float:
    """Weighted geometric mean; the weights need not sum to one."""
    total = sum(w for _, w in parts)
    return math.exp(sum(w / total * math.log(v) for v, w in parts))


def sub_scores(stats: dict[str, float | None]) -> dict[SubIndex, float]:
    """The sub-indices that have at least one statistic."""
    out: dict[SubIndex, float] = {}
    for key, parts in PARTS.items():
        scored = [(metric_score(m, stats[s]), w) for s, m, w in parts if stats.get(s) is not None]
        if scored:
            out[key] = geo_mean(scored)
    return out


def index_under(sub: dict[SubIndex, float], r1: float | None, weights: dict[SubIndex, float]) -> float:
    """The index under any weights, zero weights dropped, capped when probe success is below the floor."""
    parts = [(sub[k], w) for k, w in weights.items() if w > 0 and k in sub]
    v = geo_mean(parts) if parts else 0.0
    return min(v, RELIABILITY_CAP) if r1 is not None and r1 < RELIABILITY_FLOOR else v


@dataclass
class Ranked:
    id: str
    stats: dict[str, float | None]
    ci: float
    sub: dict[SubIndex, float] = field(default_factory=dict)
    index: float = 0.0
    rank: int = 0
    band: int = 0
    band_label: str = ""
    rank_lo: int = 0
    rank_hi: int = 0

    @property
    def lo(self) -> float:
        return self.index - self.ci

    @property
    def hi(self) -> float:
        return self.index + self.ci


def rank(providers: dict[str, tuple[dict[str, float | None], float]], seed: int = 20261201) -> list[Ranked]:
    """Rank providers given each one's statistics and the half-width of its index's 95% interval.

    Providers whose intervals overlap the band leader's share its band. The rank range is the 5th to 95th percentile
    rank over Dirichlet re-weightings centred on the published weights."""
    rows = [Ranked(pid, stats, ci) for pid, (stats, ci) in providers.items()]
    for r in rows:
        r.sub = sub_scores(r.stats)
        if set(r.sub) != set(WEIGHTS):
            missing = sorted(SUB_INDICES[k] for k in set(WEIGHTS) - set(r.sub))
            raise ValueError(f"{r.id} has no statistics for {', '.join(missing)}, so it can't be ranked")
        r.index = index_under(r.sub, r.stats.get("r1"), WEIGHTS)
    rows.sort(key=lambda r: -r.index)

    rng = random.Random(seed)
    ranks: dict[str, list[int]] = {r.id: [] for r in rows}
    keys = list(WEIGHTS)
    for _ in range(STRESS_DRAWS):
        g = [rng.gammavariate(STRESS_ALPHA * WEIGHTS[k], 1.0) for k in keys]
        w = dict(zip(keys, (x / sum(g) for x in g), strict=True))
        order = sorted(rows, key=lambda r: -index_under(r.sub, r.stats.get("r1"), w))
        for i, r in enumerate(order):
            ranks[r.id].append(i + 1)

    band, leader = 0, None
    for i, r in enumerate(rows):
        if leader is None or r.hi < leader.lo:
            band, leader = band + 1, r
        r.rank, r.band = i + 1, band
        drawn = sorted(ranks[r.id])
        r.rank_lo, r.rank_hi = drawn[int(0.05 * (STRESS_DRAWS - 1))], drawn[int(0.95 * (STRESS_DRAWS - 1))]
    for r in rows:
        shared = sum(1 for o in rows if o.band == r.band) > 1
        r.band_label = f"{r.band}{'=' if shared else ''}"
    return rows
