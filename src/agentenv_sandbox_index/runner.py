"""The runner: drives sandbox providers through agent-env's provider registry, times every step on the client with a
monotonic clock, and writes one raw row per sample. First try only: a failed or timed-out step fails the sample, and
nothing is retried."""

import asyncio
import random
import shlex
import time
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_env.providers.sandbox_providers import (
    SANDBOX_MODE_VM,
    Sandbox,
    SandboxProvider,
    VmSandbox,
    build_sandbox_provider,
)

from . import methodology, rows
from .rows import Row, Step


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

    async def command(self, name: str, sandbox: Sandbox, *argv: str) -> str:
        """Run a command as a step; a non-zero exit fails the step."""
        code, out, err = await self.step(name, sandbox.exec_with_output(*argv), methodology.STEP_TIMEOUT_S)
        if code != 0:
            self.row.steps[name].error = f"exit {code}: {err.strip()[:400]}"
            raise RuntimeError(self.row.steps[name].error)
        return out

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
        raise ValueError(f"{type(provider).__name__} runs containers only, so it needs --image, the benchmark image")
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
        await sample.command("first_command", sandbox, "true")
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


def client_fetch(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status < 400
    except Exception:
        return False


FETCH = "curl -fsS --max-time 60 -o /dev/null {url} || wget -q -T 60 -O /dev/null {url}"


async def r1(provider: SandboxProvider, name: str, run_id: str, image: str | None) -> Row:
    """R1: one probe, create, a command, a file written and read back, a network fetch and teardown, any failure or
    any step over its timeout failing it. A failed fetch the client's own fetch of the same URL also failed is
    discarded: the fault was on our side."""
    sample = Sample(run_id, name, "R1")
    sandbox, control = None, None
    try:
        sandbox = await sample.step("create", create(provider, run_id, image), methodology.CREATE_TIMEOUT_S)
        await sample.command("exec", sandbox, "true")
        token = sample.row.sample_id
        # A bare VM has no agent container to write into, so its file goes on the VM host, through the same transfer.
        write = (
            sandbox.write_host_file(token.encode(), methodology.PROBE_PATH)
            if isinstance(sandbox, VmSandbox)
            else sandbox.write_file_from_text(token, methodology.PROBE_PATH)
        )
        await sample.step("file_write", write, methodology.STEP_TIMEOUT_S)
        if (await sample.command("file_read", sandbox, "cat", methodology.PROBE_PATH)).strip() != token:
            sample.row.steps["file_read"].error = "read back something other than what was written"
            raise RuntimeError(sample.row.steps["file_read"].error)
        control = asyncio.create_task(asyncio.to_thread(client_fetch, methodology.PROBE_URL))
        await sample.command("network", sandbox, "sh", "-c", FETCH.format(url=shlex.quote(methodology.PROBE_URL)))
        doomed, sandbox = sandbox, None
        await sample.step("teardown", provider.terminate_sandbox(doomed.sandbox_id), methodology.STEP_TIMEOUT_S)
        sample.row.value = sample.row.steps["teardown"].end
    except Exception as e:
        sample.fail(e)
        if sample.row.failed_step == "network" and control is not None and not await control:
            sample.row.outcome = "discarded"
    finally:
        if control is not None and not control.done():
            control.cancel()
        if sandbox is not None:
            try:
                await sample.step(
                    "teardown", provider.terminate_sandbox(sandbox.sandbox_id), methodology.STEP_TIMEOUT_S
                )
            except Exception:
                pass  # the probe has already failed; the teardown step records this too
    return sample.row


MEASURES: dict[str, Callable[[SandboxProvider, str, str, str | None], Awaitable[Row]]] = {"S1": s1, "R1": r1}


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
    """Measure `metrics` on `providers` into `run_dir`: `samples` back to back per provider and metric, or else each
    metric's cadence for `days`, every provider and metric on its own loop. Returns the run id."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    rows.write_manifest(run_dir, run_id, providers, metrics)
    built = {name: build_sandbox_provider(name) for name in providers}

    async def measure(name: str, metric: str) -> None:
        row = await MEASURES[metric](built[name], name, run_id, image)
        rows.append(run_dir, row)
        log(row)

    async def cadence(name: str, metric: str, end: float) -> None:
        period, count = methodology.CADENCE[metric]
        due = time.monotonic() + random.uniform(0, period)
        while due < end:
            jitter = random.uniform(-1, 1) * methodology.CADENCE_JITTER * period
            await asyncio.sleep(max(0.0, due + jitter - time.monotonic()))
            for _ in range(count):
                await measure(name, metric)
            due += period

    try:
        if samples is not None:
            for name in providers:
                for metric in metrics:
                    for _ in range(samples):
                        await measure(name, metric)
        else:
            end = time.monotonic() + days * 86400
            await asyncio.gather(*(cadence(name, metric, end) for name in providers for metric in metrics))
        return run_id
    finally:
        for provider in built.values():
            await provider.close()
