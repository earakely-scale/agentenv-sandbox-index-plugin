"""`agent-env bench`: `sandboxes` measures providers into a run directory, `score` summarizes one or more runs."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import click

from . import methodology
from .rows import Row, read
from .runner import MEASURES, run
from .summary import summarize


@click.group()
def bench():
    """Benchmarks of agent-env's providers, starting with the AgentEnv Sandbox Index."""


def _providers(value: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()]
    return list(methodology.PROVIDERS) if names == ["all"] else names


def _show(row: Row) -> None:
    if row.outcome == "ok":
        click.echo(f"{row.provider:<12} {row.metric}  {row.value:8.3f} s")
    else:
        click.echo(f"{row.provider:<12} {row.metric}  failed at {row.failed_step}: {row.error}", err=True)


@bench.command()
@click.option(
    "--providers", required=True, help="Comma-separated agent-env backend names, or `all` for the ranked providers."
)
@click.option("--metrics", default=",".join(MEASURES), show_default=True, help="Comma-separated metric ids.")
@click.option("--samples", type=int, help="Take this many samples per provider and metric back to back, then stop.")
@click.option(
    "--days", type=float, default=7.0, show_default=True, help="Without --samples, run the hourly cadence this long."
)
@click.option("--image", help="The pre-registered benchmark image, for providers that run containers only.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="The run directory to write.")
def sandboxes(providers: str, metrics: str, samples: int | None, days: float, image: str | None, out: Path):
    """Measure sandbox providers for the AgentEnv Sandbox Index and write raw rows to --out."""
    wanted = [m.strip() for m in metrics.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MEASURES]
    if unknown:
        raise click.BadParameter(
            f"not measured yet: {', '.join(unknown)} (measured: {', '.join(MEASURES)})", param_hint="--metrics"
        )
    run_id = asyncio.run(run(out, _providers(providers), wanted, samples=samples, days=days, image=image, log=_show))
    click.echo(f"run {run_id} written to {out}")


@bench.command()
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--resamples", type=int, default=methodology.BOOTSTRAP_RESAMPLES, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print the summary as JSON.")
def score(runs: tuple[Path, ...], resamples: int, as_json: bool):
    """Summarize run directories: each provider's statistics with bootstrap 95% intervals."""
    summaries = summarize(read(list(runs)), resamples=resamples)
    if as_json:
        click.echo(
            json.dumps({"methodology": methodology.VERSION, "summaries": [asdict(s) for s in summaries]}, indent=2)
        )
        return
    for s in summaries:
        stats = "  ".join(
            f"{x.name} {x.value:.3f} s [{x.lo:.3f}, {x.hi:.3f}] by {x.resampled_by}" for x in s.statistics
        )
        click.echo(f"{s.provider:<12} {s.metric}  n={s.samples} failed={s.failed}  {stats}")
