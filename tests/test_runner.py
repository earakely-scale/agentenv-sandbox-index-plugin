import asyncio
import json

import pytest
from agent_env.providers.sandbox_providers import SandboxProvider, VmSandbox
from click.testing import CliRunner

from agentenv_sandbox_index import cli, runner
from agentenv_sandbox_index.rows import read
from agentenv_sandbox_index.summary import summarize


class FakeVm(VmSandbox):
    def __init__(self, sandbox_id: str, exit_code: int):
        self.sandbox_id, self.tunnel_urls, self.vnc_url, self.mode, self._exit = sandbox_id, {}, None, "vm", exit_code

    async def terminate(self) -> None:
        pass

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        return self._exit, "", "boom" if self._exit else ""


class FakeVmProvider(SandboxProvider):
    def __init__(self, *, fail_create: bool = False, exit_code: int = 0, create_delay: float = 0.0):
        self.fail_create, self.exit_code, self.create_delay = fail_create, exit_code, create_delay
        self.created: list[dict] = []
        self.terminated: list[str] = []

    async def create_vm(self, **kwargs) -> FakeVm:
        await asyncio.sleep(self.create_delay)
        if self.fail_create:
            raise RuntimeError("no capacity")
        self.created.append(kwargs)
        return FakeVm(f"sb-{len(self.created)}", self.exit_code)

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        self.terminated.append(sandbox_id)


class FakeContainerProvider(SandboxProvider):
    async def create_sandbox(self, **kwargs):
        raise AssertionError("not reached")


def s1(provider) -> runner.Row:
    return asyncio.run(runner.s1(provider, "fake", "run-1", None))


def test_s1_creates_at_the_standard_spec_and_tears_down():
    provider = FakeVmProvider()
    row = s1(provider)
    assert row.outcome == "ok" and row.value is not None and row.value >= 0
    assert set(row.steps) == {"create", "first_command", "teardown"}
    assert provider.created == [
        {
            "cpu": 2.0,
            "memory": 4096,
            "setup_for_gateway": False,
            "attribution": {"benchmark": "agentenv-sandbox-index", "run_id": "run-1"},
        }
    ]
    assert provider.terminated == ["sb-1"]


def test_s1_failed_create_is_a_failed_sample():
    row = s1(FakeVmProvider(fail_create=True))
    assert (row.outcome, row.failed_step, row.error_class, row.value) == ("failed", "create", "error", None)
    assert "no capacity" in row.error


def test_s1_nonzero_exit_fails_and_still_tears_down():
    provider = FakeVmProvider(exit_code=1)
    row = s1(provider)
    assert (row.outcome, row.failed_step) == ("failed", "first_command")
    assert provider.terminated == ["sb-1"]


def test_s1_create_timeout(monkeypatch):
    monkeypatch.setattr(runner.methodology, "CREATE_TIMEOUT_S", 0.01)
    row = s1(FakeVmProvider(create_delay=1))
    assert (row.outcome, row.failed_step, row.error_class) == ("failed", "create", "timeout")


def test_container_only_provider_needs_the_benchmark_image():
    row = s1(FakeContainerProvider())
    assert row.outcome == "failed" and "--image" in row.error


def test_cli_run_then_score(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_sandbox_provider", lambda name: FakeVmProvider())
    out = tmp_path / "run"
    result = CliRunner().invoke(cli.bench, ["sandboxes", "--providers", "a,b", "--samples", "3", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(read([out])) == 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["spec"] == {"cpu": 2.0, "memory_mb": 4096, "region": "us-east"}
    assert manifest["distributions"]["agentenv-framework"]

    result = CliRunner().invoke(cli.bench, ["score", str(out), "--resamples", "50", "--json"])
    assert result.exit_code == 0, result.output
    summaries = json.loads(result.output)["summaries"]
    assert [(s["provider"], s["samples"], [x["name"] for x in s["statistics"]]) for s in summaries] == [
        ("a", 3, ["s1", "s1p"]),
        ("b", 3, ["s1", "s1p"]),
    ]


def test_unmeasured_metric_is_refused(tmp_path):
    result = CliRunner().invoke(
        cli.bench, ["sandboxes", "--providers", "a", "--metrics", "R1", "--samples", "1", "--out", str(tmp_path)]
    )
    assert result.exit_code != 0 and "not measured yet: R1" in result.output


def test_summary_interval_contains_the_value():
    provider = FakeVmProvider()
    rows = [s1(provider) for _ in range(20)]
    (s,) = summarize(rows, resamples=100)
    for stat in s.statistics:
        assert stat.lo <= stat.value <= stat.hi or stat.lo == pytest.approx(stat.hi)
