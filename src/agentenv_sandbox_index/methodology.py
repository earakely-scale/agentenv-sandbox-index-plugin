"""The Sandbox Index methodology as data: the standard setup, the providers it ranks, each metric's reference points
and the weights. Changing anything here changes the methodology, so it bumps VERSION."""

from dataclasses import dataclass
from typing import Literal

VERSION = "1.0-draft"

CPU = 2.0
MEMORY_MB = 4096
REGION = "us-east"
CREATE_TIMEOUT_S = 300.0
STEP_TIMEOUT_S = 120.0

# The hosted providers on agentenvframework.com/docs/sandboxes, by their agent-env backend names. `local` and smol
# machines run on your own hardware, so they can be measured but are never ranked.
PROVIDERS = (
    "modal",
    "modal_vm",
    "coreweave",
    "e2b",
    "sail_vm",
    "tensorlake",
    "tenki",
    "vercel",
    "boxd",
    "daytona",
    "boat",
)

Kind = Literal["time", "rate", "nines"]
SubIndex = Literal["agent", "startup", "scale", "rel"]


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    sub_index: SubIndex
    kind: Kind
    # Score 100 and score 0. Times and rates use a log scale between them; nines are fixed at 99.99% and 90%.
    best: float
    worst: float
    status: Literal["in", "pending pilot", "if quotas allow"]


METRICS: dict[str, Metric] = {
    m.id: m
    for m in (
        Metric("A1", "Env deploy time", "agent", "time", 1, 300, "in"),
        Metric("A2", "Time to first tool call", "agent", "time", 2, 300, "in"),
        Metric("A3", "GridSim in-episode time", "agent", "time", 0.5, 60, "in"),
        Metric("S1", "Create to first command, cached image", "startup", "time", 0.1, 30, "in"),
        Metric("C1", "Time to 100 running", "scale", "time", 1, 600, "in"),
        Metric("C2", "Burst success", "scale", "nines", 99.99, 90, "in"),
        Metric("C3", "Time to 1,000 running", "scale", "time", 10, 1800, "if quotas allow"),
        # Provisional until the pilot fixes them.
        Metric("C4", "Sustained episode throughput", "scale", "rate", 600, 10, "pending pilot"),
        Metric("R1", "Probe success", "rel", "nines", 99.99, 90, "in"),
        Metric("R2", "Mid-episode loss", "rel", "nines", 99.99, 90, "in"),
        Metric("R3", "Teardown leaks", "rel", "nines", 99.99, 90, "pending pilot"),
    )
}

SUB_INDICES: dict[SubIndex, str] = {
    "agent": "Agent workload",
    "startup": "Startup",
    "scale": "Scale",
    "rel": "Reliability",
}
WEIGHTS: dict[SubIndex, float] = {"agent": 0.35, "startup": 0.20, "scale": 0.20, "rel": 0.25}

# Each sub-index's statistics: (statistic, metric, weight inside the sub-index). A statistic missing for a provider
# (C3 without a quota) drops out and the rest of its sub-index scales up.
PARTS: dict[SubIndex, tuple[tuple[str, str, float], ...]] = {
    "agent": (("a1", "A1", 0.35), ("a2", "A2", 0.175), ("a2p", "A2", 0.175), ("a3", "A3", 0.30)),
    "startup": (("s1", "S1", 0.6), ("s1p", "S1", 0.4)),
    "scale": (("c1", "C1", 0.30), ("c2", "C2", 0.20), ("c3", "C3", 0.25), ("c4", "C4", 0.25)),
    "rel": (("r1", "R1", 0.60), ("r2", "R2", 0.25), ("r3", "R3", 0.15)),
}

# Below this probe success (R1, %) the index is capped at RELIABILITY_CAP.
RELIABILITY_FLOOR = 99.0
RELIABILITY_CAP = 40.0

# How often each metric samples a provider in the scored run: (period in seconds, samples each period). Each
# provider's loop for a metric starts at a random offset inside the first period.
CADENCE: dict[str, tuple[float, int]] = {"S1": (3600, 10), "R1": (300, 1)}

# R1's network step fetches this from inside the sandbox; the client fetches it too, at the same moment, so a fault on
# the client's side discards the probe instead of failing the provider. A tiny response on a CDN built for checks.
PROBE_URL = "https://www.gstatic.com/generate_204"
PROBE_PATH = "/tmp/agentenv-sandbox-index-probe"

STRESS_DRAWS = 10_000
STRESS_ALPHA = 20.0
BOOTSTRAP_RESAMPLES = 10_000
