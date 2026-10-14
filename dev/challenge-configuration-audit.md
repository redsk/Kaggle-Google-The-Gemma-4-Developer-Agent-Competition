# Competition configuration audit — 2026-10-14

The user selected **maximum permitted generation settings**, not the smaller
starter defaults. The agent now declares `max_output_tokens: 32768` and
`thinking_budget: 32768`. Input, reasoning and visible output share a single
32,768-token context. The local client allows vLLM to use the remaining context
for generation; it does not reserve 32K output tokens in addition to the input.

The former 4,096-token total cap and 1,024-token thinking request were local
choices, not competition requirements. The prior task runs do not represent the
configuration established by this audit.

## Evidence and scope

Sources checked:

- A fresh download of the organizer's [Getting Started notebook](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent).
  Its cells match the earlier local copy. Fresh notebook SHA256:
  `8e7d84a326acae509fb596163ba085831961b588b86d51af31e037c5319b917c`.
- The [official wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse),
  confirmed through Kaggle's API to still be version **31**. All four downloaded
  wheel hashes match [the manifest](harness-wheelhouse.json). All **640 Python
  source files** in those installed packages match the wheel contents.
- `swegemma/config.py::build_submission_limits`, `swegemma/budget.py`, the real
  compiler/client and agent loop, `adk_submission/server.py::VllmServer.build_cmd`.
- Downloaded `HARNESS_README.md`, sample agent, `configs/sampling.yaml`, and
  `eval_config.yaml`; local rules snapshot contains no 4,096-total-output rule.
- Every local launcher, task runner, verifier, diagnostic/benchmark utility,
  model mapping, runtime manifest and the obsolete generated Kaggle notebook.
- Installed vLLM 0.31.0 / vLLM-Metal 0.31.0 request validation and token-budget
  calculation; the pinned MLX checkpoint's generation configuration.

The organizer's starter is a configurable example, not a list of hard maxima.
In particular, the sample `eval_config.yaml` has a one-minute/10-tool/50-turn
demonstration budget and a 60-second command timeout. Those are not mandatory
competition ceilings and have not been copied into our task runner.

## Changes made

| Setting | Before | Now | Basis |
|---|---|---|---|
| Agent total output request | 4,096 | 32,768 declared; remaining context on the local wire | Official allowed range is 1–32,768; user chose maximum. |
| Agent thinking request | 1,024 | 32,768 requested | Official allowed range is 0–32,768; Metal still does not enforce this separate budget. |
| Server-wide output ceiling | `max_new_tokens=4096`, including summaries | Removed | No such override in the official starter. |
| Generation defaults source | `--generation-config vllm` | Upstream default `auto` | Match official serving behavior; load checkpoint defaults for omitted parameters. |
| GPU memory fraction | 0.35 | 0.90 | Organizer notebook / official `VllmConfig` default; a serving default, not a scoring rule. |
| Maximum concurrent sequences | Forced to 1 | No local override | Let upstream select its defaults. |
| Maximum batched tokens | Forced to 2,048 | No local override | Let upstream select its defaults. |
| Full-input scheduler reservation | Implicit runtime default | Explicitly disabled | Match official server builder, which avoids hybrid-model admission stalls. |
| Reasoning markers | Parser name only | Explicit official start/end marker config | Match official server builder. |
| Agent sandbox tests | `enable_sandbox_testing=False` | Official default `True` | Pytest and unittest are available. |
| Agent instruction | Claimed pytest disabled; asked to keep reasoning brief | Correct testing instructions; brevity restriction removed | Remove our artificial behavioral restriction and obsolete environment claim. |
| Saved/reference patch verifier timeout | 180 seconds | Official default 300 seconds | Inherit `HarnessLimits`; no separate shorter override. |
| ADK context cache configuration | Omitted | 2,048 minimum tokens; 1,800-second TTL; interval 10 | Exact organizer notebook settings. This does not prove a cache hit on a particular backend. |
| Repetition/no-edit/nudge monitor thresholds | Automatically created pause gates | Alerts only by default | Observation should not silently stop task execution. `--pause-on-alert` opts in. |
| Monitor workspace-command timeout | 15 seconds | Inherit official command timeout (300 seconds) | Avoid a separate shorter local timeout that could trigger a false pause. |
| Benchmark argument ceiling | Refused output requests over 4,096 | Removed; validates against the chosen context | The prompt must also fit. Benchmark generation length remains user-selectable. |
| Benchmark server readiness timeout | 180 seconds | 1,200 seconds, configurable | Organizer notebook startup allowance; excluded from timing. |
| Old Kaggle generator/notebook | Could recreate the 20-turn/30-tool/10-minute HTTPX comparison | Archived and replaced by non-executing retirement notices | Prevent accidental reuse of the rejected experiment. |

## Why the local client omits a full-window output limit

The local vLLM API validates `prompt_tokens + requested_output_tokens` against
`max_model_len`. Sending an explicit 32,768 output request leaves no space for
even one input token and is rejected. Its own `get_max_tokens` function handles
an omitted output request by using `max_model_len - formatted_prompt_tokens`,
subject to model/server defaults. The pinned model has no `max_new_tokens` in
its generation config, and the launcher no longer overrides it.

[vllm_client.py](../scripts/vllm_client.py) therefore removes an output-limit
field only when it equals the configured full context. It preserves smaller
explicit limits, tools, messages, thinking controls and all other parameters.
The server counts the actual formatted prompt, including tool schemas. There
is no estimated token count, fixed reserved margin or newly invented cap.
For example, a 9,000-token formatted prompt leaves at most 23,768 tokens for
reasoning and output together.

The mapping explicitly opts into this behavior. The runner checks the live
server's advertised model/context before a task and records each request's
budget transformation in `request-budget.jsonl`. A model/version advertising a
different context fails the check. Summaries already omit an output ceiling and
now also receive the remaining context without the old server-wide cap.

This is **our local transport adaptation**, not code added to the official
wheelhouse or Metal runtime. It is not automatically included in a Kaggle
submission. Although 32,768 is accepted by the submission schema, a literal
full-window request can conflict with the scoring server's input/output
validation too. The scoring path must be checked separately before submitting
this maximum-budget policy. No claim of end-to-end Kaggle equivalence is made.

## Settings retained with their provenance

| Setting | Current behavior | Why retained |
|---|---|---|
| Context window | 32,768 | Official competition serving window. |
| Temperature / top-p | 0.2 / 0.95 | Official sample sampling settings. Other omitted parameters now use model defaults. |
| Include thoughts | `true` | Official sample; preserve provider-exposed reasoning in the harness history. |
| History compaction | Interval 15, overlap 2, token threshold 14,336, retain 5 events | Exact organizer notebook settings, not locally invented values. |
| Tool/verification command timeout | 300 seconds | Official `HarnessLimits` defaults. |
| Command output / file views | 5,000 chars; 150 lines and 10,000 chars per file read | Official harness defaults. |
| Session time / tool calls | No added cap | Earlier explicit user request to let the local task finish; supported by `EvaluationBudget(None, ...)`. Official default time is 60 minutes and scorer tool default is 100, both configurable rather than hard maxima. |
| Turns | No custom override; upstream fallback 500 | Existing official loop behavior. |
| Text-only continuation nudges | Official maximum 3 | Upstream loop behavior, distinct from our now-observational monitor. |
| Error retries | Official CLI model-registry and harness retry behavior | Not changed by this audit. CLI registry retries and harness plugin retries are separate layers. |
| Tool set | All nine standard execution/file/graph tools | No tool was removed or limited by this audit. |
| Model and agent structure | User-selected 12B; current single-agent prompt/tool list | User's experiment choice. The supplied sample additionally has an analyzer agent tool; ours is not a byte-identical copy of that sample. |
| Manual supervision | Explicit `pause.json` and recorded steering | Retained for user-requested intervention. No automatic pause by default. |

`--minutes` remains available only as an explicit user-supplied override. The
monitor's heartbeat and alert counts are observation settings, not generation
or task budgets. Effective settings are saved at each run's start and can be
inspected with `scripts/run_supervised.py --print-config` without starting a run.

## Remaining runtime and experiment differences

- **Thinking enforcement:** Metal ignores the separate numeric thinking budget.
  It can spend all remaining generation capacity thinking. No upstream patch was
  made; the earlier instruction to keep the runtime unchanged still applies.
- **Hardware/weights:** local MLX affine quantization on Apple Silicon differs
  from Google's compressed-tensors QAT checkpoint on the competition's CUDA GPUs.
  The text-only local serving switch and checkpoint path remain compatibility
  settings. CUDA tensor parallelism and GPU dtype settings cannot be copied
  literally onto this Mac.
- **Sandbox:** retain the local Docker backend and source-import-priority
  overlay. Its container limits (4 GiB, two CPUs) come from the official Docker
  backend. The existing Podman VM has 12 GiB RAM and ten CPUs; it was stopped
  during this audit and was neither started nor resized. These host resources
  are not a reproduction of Kaggle's machine.
- **Server allocation:** the 0.90 memory fraction now matches the official
  configuration, but Metal uses unified memory. Effective batching/concurrency
  is selected by upstream. The revised allocation has not been tested under
  load; old throughput measurements use the previous settings.
- **CLIProxyAPI:** remote access ignores output limits, temperature and top-p.
  Its synthetic test's low reasoning, tiny output probes, zero client retries
  and 90-second request timeout remain explicitly test-only. They do not feed
  the Gemma task runner or define the future teacher policy.
- **Benchmarks/chat:** 1,024 output tokens, thinking off, seed 42, three repetitions
  and other speed-test controls remain confined to the benchmark. The casual
  Ollama chat alias keeps its separate 8,192 context. These are not task limits.
  Small 8/16/32/256-token diagnostic probes deliberately test specific behavior;
  raising them to competition maxima would defeat those tests.

## Validation

Ten offline tests pass against the downloaded organizer assets and installed
compiler/client. They cover maximum generation values, all standard tools,
tool limits/testing availability, compaction/cache parity, absence of added
task budgets, the real compiler/client request and summary path, server-context
validation, launch-argument parity, and observational versus opt-in pauses.
Every revised launcher flag is accepted by the installed vLLM CLI help.
The actual vLLM validation and `get_max_tokens` functions were also exercised
offline: a literal full-window output request rejects a nonempty prompt, while
omitting the request yields exactly the remaining context at multiple prompt sizes.

```sh
.venv/bin/python scripts/run_supervised.py --print-config
.venv/bin/python -m unittest discover -s tests -p 'test_challenge_configuration.py' -v
```

No model weights were loaded, no model answer was generated, no sandbox was
started and no Kaggle GPU job was launched during this audit. Apply the revised
launcher on the next deliberate server start; source/config validation is not
a claim of successful inference under the new settings.
