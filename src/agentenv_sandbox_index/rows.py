"""Raw rows: one JSON line per sample, the data the index is computed from and published as. Each run directory holds
`rows.jsonl` and `manifest.json` (the methodology version, the spec and every installed distribution's version)."""

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Literal

from . import methodology

SCHEMA = "agentenv-sandbox-index/raw/v1"
ROWS = "rows.jsonl"
MANIFEST = "manifest.json"


@dataclass
class Step:
    """A step's start and end, in seconds from the sample's start on the client's monotonic clock."""

    start: float
    end: float | None = None
    error: str | None = None


@dataclass
class Row:
    run_id: str
    sample_id: str
    provider: str
    metric: str
    started_at: str
    steps: dict[str, Step] = field(default_factory=dict)
    value: float | None = None
    # "discarded": the fault was on the client's side (R1's control fetch failed too), so the sample counts for no one.
    outcome: Literal["ok", "failed", "discarded"] = "ok"
    # The failed step and how it failed: "timeout" or "error".
    failed_step: str | None = None
    error_class: str | None = None
    error: str | None = None
    schema: str = SCHEMA

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Row":
        return cls(**{**d, "steps": {k: Step(**v) for k, v in d["steps"].items()}})


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def append(run_dir: Path, row: Row) -> None:
    with (run_dir / ROWS).open("a") as f:
        f.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")


def read(run_dirs: list[Path]) -> list[Row]:
    return [Row.from_dict(json.loads(line)) for d in run_dirs for line in (d / ROWS).read_text().splitlines() if line]


def write_manifest(run_dir: Path, run_id: str, providers: list[str], metrics: list[str]) -> None:
    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "started_at": now(),
        "methodology": methodology.VERSION,
        "spec": {"cpu": methodology.CPU, "memory_mb": methodology.MEMORY_MB, "region": methodology.REGION},
        "timeouts_s": {"create": methodology.CREATE_TIMEOUT_S, "step": methodology.STEP_TIMEOUT_S},
        "cadence": {m: {"period_s": p, "samples": n} for m, (p, n) in methodology.CADENCE.items() if m in metrics},
        "probe_url": methodology.PROBE_URL,
        "providers": providers,
        "metrics": metrics,
        "client": {"python": sys.version.split()[0], "platform": platform.platform()},
        "distributions": dict(sorted({d.metadata["Name"]: d.version for d in distributions()}.items())),
    }
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
