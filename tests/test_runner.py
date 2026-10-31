import asyncio
import json

import pytest
from agent_env.providers.sandbox_providers import SandboxProvider, VmSandbox
from click.testing import CliRunner

from agentenv_sandbox_index import cli, runner
from agentenv_sandbox_index.rows import read
from agentenv_sandbox_index.summary import summarize


class FakeVm(VmSandbox):
    def __init__(self, sandbox_id: str, exit_code: int, *, fetch_ok: bool = True, corrupt: bool = False):
        self.sandbox_id, self.tunnel_urls, self.vnc_url, self.mode, self._exit = sandbox_id, {}, None, "vm", exit_code
        self.fetch_ok, self.corrupt, self.files = fetch_ok, corrupt, {}

    async def terminate(self) -> None:
        pass

    async def write_host_file(self, data: bytes, vm_path: str) -> None:
        self.files[vm_path] = "garbled" if self.corrupt else data.decode()

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        if self._exit:
            return self._exit, "", "boom"
        if args[0] == "cat":
            return 0, self.files[args[1]], ""
        if args[:2] == ("sh", "-c"):
            return (0, "", "") if self.fetch_ok else (6, "", "curl: (6) Could not resolve host")
        return 0, "", ""


class FakeVmProvider(SandboxProvider):
    def __init__(self, *, fail_create: bool = False, exit_code: int = 0, create_delay: float = 0.0, **vm):
        self.fail_create, self.exit_code, self.create_delay, self.vm = fail_create, exit_code, create_delay, vm
        self.created: list[dict] = []
        self.terminated: list[str] = []

    async def create_vm(self, **kwargs) -> FakeVm:
        await asyncio.sleep(self.create_delay)
        if self.fail_create:
            raise RuntimeError("no capacity")
        self.created.append(kwargs)
        return FakeVm(f"sb-{len(self.created)}", self.exit_code, **self.vm)

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
    monkeypatch.setattr(runner, "client_fetch", lambda url: True)
    result = CliRunner().invoke(cli.bench, ["sandboxes", "--providers", "a,b", "--samples", "3", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(read([out])) == 12
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["spec"] == {"cpu": 2.0, "memory_mb": 4096, "region": "us-east"}
    assert manifest["distributions"]["agentenv-framework"]
    assert manifest["cadence"] == {"S1": {"period_s": 360, "samples": 1}, "R1": {"period_s": 300, "samples": 1}}

    result = CliRunner().invoke(cli.bench, ["score", str(out), "--resamples", "50", "--json"])
    assert result.exit_code == 0, result.output
    summaries = json.loads(result.output)["summaries"]
    assert [(s["provider"], s["samples"], [x["name"] for x in s["statistics"]]) for s in summaries] == [
        ("a", 3, ["r1"]),
        ("a", 3, ["s1", "s1p"]),
        ("b", 3, ["r1"]),
        ("b", 3, ["s1", "s1p"]),
    ]


def test_unmeasured_metric_is_refused(tmp_path):
    result = CliRunner().invoke(
        cli.bench, ["sandboxes", "--providers", "a", "--metrics", "A1", "--samples", "1", "--out", str(tmp_path)]
    )
    assert result.exit_code != 0 and "not measured yet: A1" in result.output


def _row(hour: int, value: float) -> runner.Row:
    return runner.Row(
        run_id="r",
        sample_id=f"{hour}-{value}",
        provider="p",
        metric="S1",
        started_at=f"2026-10-31T{hour:02d}:00:00",
        value=value,
    )


def test_one_hour_resamples_by_sample_not_a_zero_width_block():
    (s,) = summarize([_row(1, v) for v in (0.6, 0.7, 0.8, 0.9, 1.5)], resamples=500)
    p50 = s.statistics[0]
    assert (p50.resampled_by, p50.blocks) == ("sample", 5)
    assert p50.lo < p50.value < p50.hi


def test_several_hours_resample_by_hour():
    (s,) = summarize([_row(h, v) for h in (1, 2, 3) for v in (0.6, 0.7 + h / 10)], resamples=500)
    assert (s.statistics[0].resampled_by, s.statistics[0].blocks) == ("hour", 3)


def test_summary_interval_contains_the_value():
    provider = FakeVmProvider()
    rows = [s1(provider) for _ in range(20)]
    (s,) = summarize(rows, resamples=100)
    for stat in s.statistics:
        assert stat.lo <= stat.value <= stat.hi or stat.lo == pytest.approx(stat.hi)


def r1(provider, client_ok: bool = True) -> runner.Row:
    runner.client_fetch = lambda url: client_ok
    return asyncio.run(runner.r1(provider, "fake", "run-1", None))


@pytest.fixture(autouse=True)
def _restore_client_fetch():
    original = runner.client_fetch
    yield
    runner.client_fetch = original


def test_r1_probe_passes_every_step_and_tears_down():
    provider = FakeVmProvider()
    row = r1(provider)
    assert row.outcome == "ok"
    assert list(row.steps) == ["create", "exec", "file_write", "file_read", "network", "teardown"]
    assert provider.terminated == ["sb-1"]


@pytest.mark.parametrize(
    ("provider", "step"),
    [
        (FakeVmProvider(fail_create=True), "create"),
        (FakeVmProvider(exit_code=1), "exec"),
        (FakeVmProvider(corrupt=True), "file_read"),
        (FakeVmProvider(fetch_ok=False), "network"),
    ],
)
def test_r1_failure_names_the_step(provider, step):
    row = r1(provider)
    assert (row.outcome, row.failed_step) == ("failed", step)


def test_r1_failed_fetch_is_discarded_when_the_client_cannot_fetch_either():
    provider = FakeVmProvider(fetch_ok=False)
    row = r1(provider, client_ok=False)
    assert (row.outcome, row.failed_step) == ("discarded", "network")
    assert provider.terminated == ["sb-1"]


def _probe(hour: int, outcome: str) -> runner.Row:
    return runner.Row(
        run_id="r",
        sample_id=f"{hour}-{outcome}",
        provider="p",
        metric="R1",
        started_at=f"2026-10-31T{hour:02d}:00:00",
        outcome=outcome,
    )


def test_r1_success_rate_leaves_out_discarded_probes():
    probes = [_probe(h, "ok") for h in range(24) for _ in range(11)] + [_probe(3, "failed"), _probe(5, "discarded")]
    (s,) = summarize(probes, resamples=500)
    (rate,) = s.statistics
    assert (s.failed, s.discarded) == (1, 1)
    assert rate.value == pytest.approx(100 * 264 / 265)
    assert rate.resampled_by == "hour" and rate.lo < rate.value <= rate.hi


def test_r1_with_no_failures_takes_a_wilson_interval_below_100():
    (s,) = summarize([_probe(h, "ok") for h in range(24) for _ in range(12)], resamples=200)
    (rate,) = s.statistics
    assert rate.value == 100 and rate.resampled_by == "wilson" and 98 < rate.lo < 100
