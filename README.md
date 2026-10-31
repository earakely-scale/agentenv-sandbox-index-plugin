# agentenv-sandbox-index

The harness behind the **AgentEnv Sandbox Index**: one score for how well each sandbox cloud runs AI agents, measured
on real agent workloads, with every raw number published. It is an [AgentEnv Framework](https://github.com/scaleapi/agentenv-framework)
plugin that adds `agent-env bench`. It drives every provider through agent-env's sandbox-provider interface, so each one
runs the same code.

**Status: pre-release.** Methodology `1.0-draft`. S1 (create to first command) and R1 (probe success) are measured;
the other metrics land one by one, each prototyped on `modal_vm` first. Reference points are calibrated by a 2-day pilot before v1.0 freezes.

## Install

```bash
pip install agentenv-framework
agent-env plugin add "agentenv-sandbox-index @ git+https://github.com/earakely-scale/agentenv-sandbox-index-plugin"
```

Each provider needs its own credentials and extra, as the [Sandboxes](https://www.agentenvframework.com/docs/sandboxes)
docs describe.

## Run

```bash
# Prototype: 20 S1 samples on modal_vm, back to back
agent-env bench sandboxes --providers modal_vm --metrics S1 --samples 20 --out runs/modal_vm-s1

# The scored cadence on every ranked provider for 7 days: S1 10 times an hour, an R1 probe every 5 minutes
agent-env bench sandboxes --providers all --days 7 --out runs/week

# Statistics with bootstrap 95% intervals
agent-env bench score runs/week
```

`--providers all` is the hosted providers in agent-env's sandbox docs: `modal`, `modal_vm`, `coreweave`, `e2b`,
`sail_vm`, `tensorlake`, `tenki`, `vercel`, `boxd`, `daytona` and `boat`. `local` and smol machines run on your own
hardware: you can measure them, but they are never ranked.

## What is measured

Eleven metrics in four sub-indices. Each scores 0–100 against fixed reference points: times on a log scale, success
rates in nines (90% = 0, 99% = 33, 99.99% = 100). Metrics combine into sub-indices, and sub-indices into the index,
by weighted geometric mean. A provider whose probe success (R1) is below 99% is capped at 40.

| Sub-index (weight) | Metric | Score 100 / 0 |
| --- | --- | --- |
| Agent workload (35%) | A1 env deploy time, A2 time to first tool call, A3 GridSim in-episode time | 1 / 300 s, 2 / 300 s, 0.5 / 60 s |
| Startup (20%) | S1 create to first command, cached image, p50 and p95 | 100 ms / 30 s |
| Scale (20%) | C1 time to 100 running, C2 burst success, C3 time to 1,000 running, C4 episodes a minute at 64 concurrent | 1 / 600 s, nines, 10 / 1,800 s, set by the pilot |
| Reliability (25%) | R1 probe success, R2 mid-episode loss, R3 teardown leaks | nines |

Everything runs at 2 vCPU and 4 GB on default settings, with a neutral client in the same region. First try only:
create times out at 300 s, every other step at 120 s, and nothing is retried. `methodology.py` holds the reference
points and weights; any change to them bumps the methodology version.

## Raw data

A run directory holds `manifest.json` (the run id, methodology version, spec, timeouts and every installed
distribution's version) and `rows.jsonl`, one sample per line:

```json
{"run_id": "20261030T231500Z-1a2b3c", "sample_id": "9f2c41d07e15", "provider": "modal_vm", "metric": "S1",
 "started_at": "2026-10-30T23:15:02.118+00:00",
 "steps": {"create": {"start": 0.0, "end": 2.91, "error": null}, "first_command": {"start": 2.91, "end": 3.04, "error": null},
           "teardown": {"start": 3.04, "end": 3.31, "error": null}},
 "value": 3.04, "outcome": "ok", "failed_step": null, "error_class": null, "error": null,
 "schema": "agentenv-sandbox-index/raw/v1"}
```

Step times are seconds from the sample's start on the client's monotonic clock. A failed sample names the step and
whether it timed out or errored, with the provider's error text.

An R1 probe creates a sandbox, runs a command, writes a file and reads it back, fetches a URL from inside the sandbox
and tears it down. If the fetch fails while the client's own fetch of the same URL, made at the same moment, fails too,
the fault was on our side: the probe is `discarded`, counted and published, but scored for no one.

`agent-env bench score` gives each statistic a 95% interval from a bootstrap over hour blocks; a run inside one hour
resamples by sample, and a success rate with no failures takes a Wilson interval, so it never claims zero width.

## Develop

```bash
uv venv && uv pip install -e '.[dev]'
ruff check . && pytest
```

Licensed under Apache 2.0. Provider names are their owners' trademarks.
