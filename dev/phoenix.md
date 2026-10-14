# Local Phoenix observability

Phoenix **20.21.4** is installed in `.venv-phoenix`, using the same Python 3.13
interpreter as the harness but separate dependencies. The official harness
environment and installed source files are unchanged. The server uses SQLite in
`.cache/phoenix/` and serves the UI and OTLP/HTTP receiver on **127.0.0.1:6006**.
Raw traces, credentials and the database remain outside Git.

## Start and reproduce

```sh
# Only needed on a fresh checkout; uses the existing Python interpreter.
UV_CACHE_DIR=.cache/uv uv venv .venv-phoenix --python .venv/bin/python
UV_CACHE_DIR=.cache/uv uv pip sync --python .venv-phoenix/bin/python dev/requirements.phoenix.txt

# Foreground server; Ctrl-C stops this terminal's server without deleting data.
sh scripts/start_phoenix.sh
```

Open [Phoenix](http://127.0.0.1:6006). The preflight examples are in the
`gemma-observability-preflight` project. Actual task runs use
`gemma-agent-comparison`. A run is grouped by its result-directory name in the
`session.id` attribute; `metadata.backend` distinguishes local and remote runs.
Use the trace tree to expand `llm.agent`, `model.http`, `tool.*`, and
`history.compaction` → `llm.compaction`.

The launcher disables Phoenix telemetry, its AI assistant, MCP server, external
resources and code sandbox providers. It does not configure model credentials
inside Phoenix. The pinned server normally binds gRPC to all interfaces,
independently of its HTTP host. Our small `phoenix_http.py` launcher disables
that unused listener in-process; it does not edit installed Phoenix files.
Only HTTP on loopback is used. The upstream startup banner may still advertise
gRPC; no gRPC socket is opened. Trace retention is explicitly unlimited (0 days)
until you choose to remove data.

The server requirements are pinned with hashes in `requirements.phoenix.txt`.
The harness needs no new tracing dependencies: the official environment already
contains OpenTelemetry SDK and OTLP/HTTP exporter 1.41.1. We use OpenInference
attributes with that existing SDK. We do not upgrade ADK to match newer vendor
integration examples or install general-purpose auto-instrumentation.

## Capture boundaries

`scripts/observability.py` installs scoped wrappers and restores them on exit:

| Boundary | What is recorded |
|---|---|
| ADK `LiteLlm.generate_content_async` | Full request, returned responses/available thoughts, tools requested, usage, finish reason, latency and streaming first-output time. Includes direct calls outside the agent loop. |
| ADK's LiteLLM client | Converted message history and tool schemas; actual SDK parameters after the vLLM remaining-context adapter. |
| HTTPX to the configured model endpoint | Request body, route, status and each visible HTTP attempt. Covers both Chat Completions and Responses paths. Auth headers are excluded. |
| `FunctionTool.run_async` | Actual function arguments, results, exceptions, duration and call ID. The three code-graph tools are labeled separately. |
| Official event summarizer | Input events, summary output, a distinct compaction span and its child model call. |
| External verifier | Patch, test output and result in a separate trace linked to the task trace. Never added to model-visible history. |
| Existing monitor | Harness events, pauses, steering notifications, workspace diff/status snapshots and time to first observed edit. |
| macOS monitor | Timestamped thermal pressure, VM/swap, GPU counters and process statistics in `telemetry.jsonl`, using the existing sampler. |

These wrappers pass through the original arguments, results and exceptions. They
do not change retries, history, tool definitions, generation budgets or task
limits. There is one LLM span per ADK model call; transport attempts are children,
so a retry does not become a second model completion. The normal tool registry,
compiler, agent loop, sandbox and verifier are retained.

Both full inputs/outputs and span summaries are saved locally. Credentials known
to the runner and credential-named fields are redacted. Redaction is not a
guarantee that arbitrary sensitive source content is removed: these are private
experiment artifacts, not files to publish. No prompt or output truncation is
added by this instrumentation; span attribute count/length truncation is disabled.

Each result directory contains:

- `telemetry.jsonl`: immediate request/response, start/end, monitor and machine
  events with timestamps, run IDs and trace/span IDs. Inspect this and
  `progress.jsonl` while a model request is still running.
- `spans.jsonl`: completed OpenTelemetry spans, independent of Phoenix availability.
- `telemetry.json` and `trace-link.json`: collection settings and task trace ID.
- Existing run/config manifests, plus a redacted model mapping, script hashes,
  harness versions, dataset/agent hashes and per-change workspace snapshots.

Phoenix receives spans asynchronously. Export failure does not stop model/tool
execution; local records remain available. The receiver's five-second export
timeout and final flush are observability settings, not model/task limits.
Completed spans may arrive after a delay; the JSONL start events and heartbeat
remain useful during an unfinished call.

## Local and remote task commands

These commands **run a competition task**; they were not executed during setup.
Keep the appropriate model server and Phoenix running first.

```sh
# First dataset entry, local Gemma, current audited configuration.
.venv/bin/python scripts/run_supervised.py

# Same harness with an explicitly chosen remote model and effort.
# Replace the placeholders after deciding the teacher policy.
.venv/bin/python scripts/run_supervised.py --backend cliproxy \
  --remote-model MODEL_ID --reasoning-effort EFFORT

# Inspect remote configuration without inference or a sandbox.
.venv/bin/python scripts/run_supervised.py --backend cliproxy \
  --remote-model MODEL_ID --reasoning-effort EFFORT --print-config
```

The proxy key is read from the existing ignored CLIProxyAPI configuration. No
default teacher model or effort is selected. `--no-phoenix` skips export while
retaining local tracing. `verify_patch.py` also records verification; it infers
the task ID from the patch run's `summary.json`, otherwise uses the first dataset
entry, with `--task-id` available for an explicit override.

## Validation and practical limits

```sh
# Offline checks: scripted transport, no real model or container.
.venv/bin/python -m unittest discover -s tests -v

# Export the scripted trace to local Phoenix; still no inference.
.venv/bin/python scripts/check_observability.py

# Five small REAL remote model calls, using the existing proxy preflight.
.venv/bin/python scripts/check_cliproxy.py --model gpt-6.1-sol --reasoning low
```

Fourteen offline checks pass. The scripted test uses real ADK/LiteLLM conversion, agent/tool execution,
streaming, a 503 retry and summarization. HTTP responses and the verifier are
fixtures. It also checks exception propagation, cancellation, wrapper restoration,
full-content capture and secret redaction. Its finite scripted request sequence
is a test assertion, not a limit added to the task runner.

The synthetic local-compatible check emitted 18 spans; the live remote preflight
emitted 12, including five model calls, five HTTP requests, compaction and the
root. Phoenix ingestion is checked against the locally persisted span IDs.
No Gemma model, task container or Kaggle job was started. The local model's
actual inference under the audited settings and the full teacher task remain
untested; synthetic transport success is not task-solving success.

The observed HTTP boundary stops at CLIProxyAPI. It cannot show the proxy's final
upstream request, internal retries or provider-hidden reasoning. We preserve the
known proxy parameter limitations in run metadata. Provider token counts are
recorded as reported; missing values stay unknown. For non-streaming calls,
first-token time cannot be measured from the client, so it is not invented.
Phoenix's automatic dollar estimates are API-price estimates, not ChatGPT
subscription billing or the cost of local compute.

Machine samples are host-wide counters, not per-token GPU attribution. Full
tracing, workspace inspection and the viewer have some overhead; use the
dedicated benchmark for engine throughput. The current phase is trajectory
inspection and external test results; no automatic LLM judges, training export
pipeline or task sweep was enabled.

References: [Phoenix self-hosting](https://arize.com/docs/phoenix/self-hosting),
[OpenTelemetry setup](https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel),
[configuration audit](challenge-configuration-audit.md),
[CLIProxyAPI limitations](cliproxyapi.md).
