# Stronger-model experiment

Status: CLIProxyAPI selected for the first trial. Release 8.0.29 is installed and
authenticated; five synthetic ADK/LiteLLM checks with GPT-6.1 Sol passed, and an
output-limit probe confirmed that this route does not enforce the requested
ceiling. Phoenix is selected and installed; shared tracing has passed scripted
and remote transport checks. The task model and budget policy remain open.
No teacher task, fine-tuning job or new Kaggle GPU run has been started. Notes
reflect the project state on 2026-10-14; see [proxy trial](dev/cliproxyapi.md).

The subsequent [configuration audit](dev/challenge-configuration-audit.md) removed
our artificial Gemma output cap, restored sandbox tests and selected maximum
permitted generation budgets. Input plus generation still shares a 32,768-token
context. The old interrupted runs used different settings; they must not be
presented as results under the audited configuration. Review pauses are now
explicitly opt-in, and the teacher's effective budget still needs documenting.

## Question

Can a substantially stronger model solve the first dataset task using the same
agent harness, tools and repository environment that local Gemma 4 12B struggled
with? Which observable choices make the difference, and which can be transferred
to Gemma through instructions, examples, tools or later fine-tuning?

Start with one task as a diagnostic case study. It will not establish a general
success rate or prove that a training method works.

## Preserve the harness

Use the existing official SWE-Gemma/ADK execution loop with the same local runner,
agent instruction, tool schemas, graph/embedding assets, sandbox image, fresh
repository snapshot, compaction settings and external verifier for both models.
Change the model endpoint and the necessary protocol adapter. Capture all adapter
transformations and any settings the new backend cannot honor.

```text
Same SWE-Gemma/ADK harness and sandbox
  ├─ local vLLM-Metal → Gemma 4 12B
  └─ selected local proxy → stronger remote model

Both → full local event log → selected trace viewer
Both → saved patch → fresh external verification container
```

Use the first entry in the pinned task dataset, not the previously selected HTTPX
smoke task. Record its identifier, dataset hash and repository base revision in
the private run manifest. Keep reference patches, verifier-only material and the
other model's answer outside the model's context.

The current harness tools include file reading/writing/editing, shell commands,
status/submission, code neighbors, semantic code search and code subgraphs. Both
models must receive the same set. Adding tools or changing prompts comes in later,
separately labeled experiments.

Do not substitute Codex, another coding agent or a different agent framework for
the loop: that would compare complete agent systems rather than model behavior
inside this harness. The proxy transports model requests; the harness executes
the declared tools in its sandbox. Do not enable extra proxy-side tools.

## GPTMock versus CLIProxyAPI: CLIProxyAPI trial selected

Both candidates can expose a local API while inference happens remotely. Account
authentication is not evidence that the exact model or tool workflow works.
Verify the actual model identity and pin the chosen proxy revision.

| Candidate | Why consider it | Main questions for this experiment |
|---|---|---|
| [GPTMock](https://github.com/rapidrabbit76/GPTMock) | Focused ChatGPT access; Python implementation; OpenAI-compatible routes. | Does the chosen model complete a real multi-turn tool cycle through our ADK/LiteLLM client? Its documented output-token policy can omit unenforceable limits with a warning or reject the request. |
| [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) | Broader multi-provider gateway; Codex OAuth and OpenAI-compatible APIs. | Can we keep one explicit provider/model, disable fallback or rotation that obscures identity, and inspect transformations, retries and output-budget behavior? |

Selection should depend on faithful tool/message transport and debuggability,
not merely a successful chat response or a model name in a registry. CLIProxyAPI
8.0.29 passed synthetic streaming, structured tools, tool-result/error and ADK
summary checks. Its Codex route drops output ceilings, temperature and top-p;
an 8-token request returned 122 completion tokens. It does not solve GPTMock's
budget limitation. The full task loop and observability still need preparation.

Before the full task, exercise a small synthetic request sequence:

1. System instruction plus user request; confirm the exact served model.
2. Structured tool call, argument parsing, tool result and follow-up response.
3. Multiple tool calls, errors and streaming termination.
4. A deliberately small output ceiling and the actual resulting finish reason.
5. A compaction/summary request using the real client path.

Record whether limits apply to reasoning, visible output, or both. Model
tokenizers and reasoning controls differ, so equal numeric settings do not imply
equal compute or identical effective context. Preserve the same harness policy
where supported, and label any mismatch. If the teacher has a larger effective
budget, the run is an upper-bound demonstration, not a strictly matched-budget
comparison. Decide and document that before running.

## Observability requirements

Keep durable raw logs locally, independent of the viewer. Create a trace per task
run, with linked spans/events for every model call, tool execution, compaction and
verification step. Record:

- Complete model-visible messages, instruction and tool schemas, plus the request
  after client/proxy transformation where accessible.
- Responses, structured tool calls, arguments, returned content, errors, retries
  and provider-exposed reasoning or summaries. Hidden reasoning is unavailable.
- Model/provider identity, generation settings, settings rejected or ignored,
  token usage and its source, latency, time to first output and finish reason.
- Workspace revision, file changes, patch snapshots, tool duration and verification
  outcome. Record graph calls separately from ordinary file inspection.
- Compaction inputs/output and trigger; human pauses, interventions and steering.
- Model/runtime/proxy/harness versions and hashes, context settings, and machine
  telemetry for local inference. Remote and local wall time measure different
  deployments and should be labeled accordingly.

Separate the verifier trace from agent-visible history. Redact credentials from
all exports; keep competition-derived content and raw traces out of the public
GitHub repository.

Phoenix is selected and running locally with SQLite. A scoped OpenTelemetry
provider captures model, transport, tool, summary and verification operations
without upgrading the pinned harness. Durable JSONL artifacts remain independent
of the viewer. See [setup, validated boundaries and limitations](dev/phoenix.md).

## Execution and review

1. Agree on the proxy, exact model/reasoning setting and comparison limits.
2. Make run metadata provider-neutral; the current runner hard-codes vLLM labels.
   Instrument both paths, including summaries, while preserving harness behavior.
3. Validate the synthetic tool cycle and trace completeness before using task data.
4. Verify the task fixture separately, then start from a clean snapshot per run.
5. Run the teacher and Gemma under the documented protocol. Keep both independent
   and report the existing interrupted Gemma run separately from a completed run.
6. Let each model finish; no arbitrary 20-turn or hidden time cap. Keep review
   pauses visible. The upstream 500-call fallback must be reported, not described
   as unlimited execution. Do not silently add steering when comparing runs.
7. Evaluate the saved patches externally. A convincing final message is not success.
8. Review the traces together: useful localization, repeated reads, graph use,
   tool failures, time to first edit, hypothesis testing, patch quality, targeted
   verification, and total tool/model calls and tokens.

If a provider fails, distinguish infrastructure failure from model failure. If
the teacher also fails, investigate the harness, task and available tools rather
than assuming a stronger model guarantees a solution.

## Transfer to Gemma

First try improvements supported by the trace: clearer tool descriptions,
instructions, short worked examples and changes to tools. Change one factor at
a time, then evaluate on other tasks.

LoRA is a later option. One successful trajectory is not a sufficient training
dataset. Collect multiple independently verified solutions, remove credentials
and verifier-only information, check permission to use provider outputs for
training, and preserve held-out repositories/tasks. Train on observable actions
and valid tool transitions; do not invent hidden teacher reasoning. Compare the
adapter against the original Gemma model with the same harness and budget.

## Open decisions

- Continue with CLIProxyAPI 8.0.29 for the teacher, accepting its documented
  parameter limitations, or choose another route for enforced output ceilings?
- Which exact teacher model and reasoning effort are available to the account?
- Matched-budget comparison or explicitly documented upper-bound demonstration?
- How should we annotate and compare completed task traces in Phoenix?
- After the first case study, which held-out tasks will test whether changes help?

Competition GPU use remains a separate, explicitly approved action.
