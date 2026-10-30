"""The runner: drives sandbox providers through agent-env's provider registry, times every step on the client with a
monotonic clock, and writes one raw row per sample. First try only: a failed or timed-out step fails the sample, and
nothing is retried."""

import asyncio
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_env.providers.sandbox_providers import SANDBOX_MODE_VM, Sandbox, SandboxProvider, build_sandbox_provider

from . import methodology, rows
from .rows import Row, Step

# How many S1 samples each provider takes an hour.
S1_PER_HOUR = 10


class Sample:
    """One sample's clock and its row; `step` times a step and records how it failed."""

    def __init__(self, run_id: str, provider: str, metric: str):
        self.row = Row(
            run_id=run_id, sample_id=uuid.uuid4().hex[:12], provider=provider, metric=metric, started_at=rows.now()
        )
        self._t0 = time.monotonic()

    def _now(self) -> float:
        return round(time.monotonic() - self._t0, 4)

    async def step(self, name: str, work: Awaitable[Any], timeout: float) -> Any:
        s = Step(start=self._now())
        self.row.steps[name] = s
        try:
            return await asyncio.wait_for(work, timeout)
        except TimeoutError:
            s.error = f"timed out after {timeout:g} s"
            raise
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"[:500]
            raise
        finally:
            s.end = self._now()

    def fail(self, error: BaseException) -> None:
        name = next((k for k, s in self.row.steps.items() if s.error), None)
        self.row.outcome, self.row.failed_step = "failed", name
        self.row.error_class = "timeout" if isinstance(error, TimeoutError) else "error"
        self.row.error = self.row.steps[name].error if name else f"{type(error).__name__}: {error}"[:500]


def attribution(run_id: str) -> dict[str, str]:
    return {"benchmark": "agentenv-sandbox-index", "run_id": run_id}


async def create(provider: SandboxProvider, run_id: str, image: str | None) -> Sandbox:
    """A sandbox at the standard spec: a bare VM where the provider has one (Docker readiness belongs to A1, so it isn't
    awaited), else a container from the pre-registered benchmark image."""
    if SANDBOX_MODE_VM in provider.SUPPORTED_SANDBOX_MODES:
        return await provider.create_vm(
            cpu=methodology.CPU, memory=methodology.MEMORY_MB, setup_for_gateway=False, attribution=attribution(run_id)
        )
    if image is None:
        raise ValueError(f"{type(provider).__name__} runs containers only, so S1 needs --image, the benchmark image")
    return await provider.create_sandbox(
        image_name=image,
        port=8080,
        env={},
        cpu=methodology.CPU,
        memory=methodology.MEMORY_MB,
        attribution=attribution(run_id),
    )


async def s1(provider: SandboxProvider, name: str, run_id: str, image: str | None) -> Row:
    """S1: create a sandbox and run `true`, timed from the create call to `true` returning. Teardown is timed too but
    is not part of S1."""
    sample = Sample(run_id, name, "S1")
    sandbox = None
    try:
        sandbox = await sample.step("create", create(provider, run_id, image), methodology.CREATE_TIMEOUT_S)
        code, _, err = await sample.step("first_command", sandbox.exec_with_output("true"), methodology.STEP_TIMEOUT_S)
        if code != 0:
            sample.row.steps["first_command"].error = f"exit {code}: {err.strip()[:400]}"
            raise RuntimeError(sample.row.steps["first_command"].error)
        sample.row.value = round(sample.row.steps["first_command"].end - sample.row.steps["create"].start, 4)
    except Exception as e:
        sample.fail(e)
    finally:
        if sandbox is not None:
            try:
                await sample.step(
                    "teardown", provider.terminate_sandbox(sandbox.sandbox_id), methodology.STEP_TIMEOUT_S
                )
            except Exception:
                pass  # recorded on the teardown step; a failed teardown is R1's and R3's business, not S1's
    return sample.row


MEASURES: dict[str, Callable[[SandboxProvider, str, str, str | None], Awaitable[Row]]] = {"S1": s1}


async def run(
    run_dir: Path,
    providers: list[str],
    metrics: list[str],
    *,
    samples: int | None,
    days: float,
    image: str | None,
    log: Callable[[Row], None] = lambda _: None,
) -> str:
    """Measure `metrics` on `providers` into `run_dir`: `samples` back to back per provider, or else the methodology's
    hourly cadence for `days`, providers in a fresh random order each hour. Returns the run id."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    rows.write_manifest(run_dir, run_id, providers, metrics)
    built = {name: build_sandbox_provider(name) for name in providers}

    async def measure(name: str, metric: str) -> None:
        row = await MEASURES[metric](built[name], name, run_id, image)
        rows.append(run_dir, row)
        log(row)

    try:
        if samples is not None:
            for name in providers:
                for metric in metrics:
                    for _ in range(samples):
                        await measure(name, metric)
            return run_id
        end = time.monotonic() + days * 86400
        while time.monotonic() < end:
            hour = time.monotonic()
            for name in random.sample(providers, len(providers)):
                for metric in metrics:
                    for _ in range(S1_PER_HOUR):
                        await measure(name, metric)
            await asyncio.sleep(max(0.0, min(3600 - (time.monotonic() - hour), end - time.monotonic())))
        return run_id
    finally:
        for provider in built.values():
            await provider.close()
