# CLIProxyAPI local trial

CLIProxyAPI **8.0.29**, source revision
`5fc580b1d72d87a7bcfe5cfb6f479c1aea5625b3`, was installed and authenticated on
2026-10-14. Five synthetic requests using `gpt-6.1-sol` with low reasoning passed
through the installed ADK/LiteLLM client. No competition task was run.

This is a local HTTP gateway to a **remote** ChatGPT/Codex model. The model does
not run on this Mac. Authentication uses a separate OAuth login stored inside
this project; it does not read or modify Codex's existing login.

## Start and use

Use the existing Python 3.13 environment. CLIProxyAPI itself is a native Go
binary and does not need a new Python environment.

```sh
# Already completed on this machine. Needed for a fresh checkout:
.venv/bin/python scripts/cliproxy.py setup
.venv/bin/python scripts/cliproxy.py login

# Keep this running in one terminal; Ctrl-C stops it:
.venv/bin/python scripts/cliproxy.py serve
```

In another terminal:

```sh
.venv/bin/python scripts/cliproxy.py models
.venv/bin/python scripts/check_cliproxy.py --model gpt-6.1-sol --reasoning low
```

The second command makes five small remote inference requests using the signed-in
account. It loads no competition tasks, containers, graphs or reference patches.
The advertised model catalog alone is not proof of account access. The trial
successfully requested `gpt-6.1-sol`; other listed models have not been tested.
The exact teacher model and reasoning effort remain undecided.

## Configuration and local files

- Endpoint: `http://127.0.0.1:8317/v1`, with a randomly generated local client key.
- Private configuration: `.cache/cliproxyapi/config.yaml` (mode `0600`).
- OAuth credentials: `.cache/cliproxyapi/auth/` inside a mode `0700` directory;
  the credential file is mode `0600`.
- Binary and archive: `downloads/tools/cliproxyapi-8.0.29/`.
- Release URL and verified archive SHA256: [cliproxyapi-runtime.json](cliproxyapi-runtime.json).
- Public template: [cliproxyapi.example.yaml](cliproxyapi.example.yaml).
- Synthetic inputs, responses and checks: `results/*-cliproxy-preflight/`.

All runtime files, credentials and raw results are ignored by Git. Setup preserves
an existing private configuration and login. The server is not installed as an
autostart service. The initial test server was stopped after validation.

The configuration binds to loopback, disables management/panel downloads, LAN
discovery, plugins and image-generation injection. It uses one credential and
the release's embedded model catalog, with no configured aliases or alternate
providers. Proxy retries and streaming bootstrap retries are disabled. The
preflight also sets client retries to zero and a 90-second request timeout; these
are explicit preflight settings, not changes to the competition runner. The
official model registry currently sets client retries to five.

Unauthenticated `/v1/models` returned 401; management routes returned 404.
Routine proxy request logging is disabled. The preflight logs ADK and LiteLLM SDK
inputs/outputs with an allowlist for request settings that excludes credentials.
These are not a capture of the final upstream HTTP payload or a complete agent
trace. LiteLLM may attach estimated API costs; those are not ChatGPT billing data.
Server access logs showed that LiteLLM selected `/v1/responses` for the two
requests containing tools and `/v1/chat/completions` for the other three. Both
paths therefore matter when inspecting protocol transformations.

## What the trial verified

Local result directory: `results/20261014T210713727377Z-cliproxy-preflight/`.

| Check | Observed result |
|---|---|
| System instruction and streaming | Followed the system instruction; completed the stream successfully. |
| Two structured tool calls | Returned both requested tool calls with valid JSON arguments and distinct IDs. |
| Tool results and error | Preserved the supplied value and the synthetic error in the final answer. |
| ADK history summary | The real `LlmEventSummarizer` preserved both facts. |
| Requested output ceiling of 8 | Returned **122 completion tokens**, finish reason `stop`; the ceiling was not enforced. |

All five client responses reported `gpt-6.1-sol`. This is the identity exposed by
the proxy, not independent verification of the provider's internal routing.
These checks validate the client/proxy path, not end-to-end task solving or
reliability over a long agent session.

## Consequence for the teacher experiment

The pinned [Chat Completions translator](https://github.com/router-for-me/CLIProxyAPI/blob/5fc580b1d72d87a7bcfe5cfb6f479c1aea5625b3/internal/translator/codex/openai/chat-completions/codex_openai_request.go#L52)
and [Responses translator](https://github.com/router-for-me/CLIProxyAPI/blob/5fc580b1d72d87a7bcfe5cfb6f479c1aea5625b3/internal/translator/codex/openai/responses/codex_openai-responses_request.go#L30)
omit output-token limits, temperature and top-p for this provider. The chat path maps
`reasoning_effort` to the upstream reasoning setting. The trial explicitly used
`low`; it did not establish equivalence to Gemma's thinking settings.

The original local Gemma comparison used a custom 4,096-token response ceiling;
that was not a competition requirement and has now been removed. The
[configuration audit](challenge-configuration-audit.md) records the user's choice
of maximum permitted budgets: local Gemma can use the remaining part of its
32,768-token context. CLIProxyAPI still drops output-token settings, so this route
does not establish equal effective context/output budgets. A prompt asking for
brevity is not a hard limit. Document that mismatch before a teacher task run.

Phoenix is now installed, and the runner has an explicit `--backend cliproxy`
path with provider-specific authentication and metadata. The five synthetic
checks were repeated with tracing; all five model calls and both HTTP routes
were captured. See [observability setup](phoenix.md).

Next: choose the teacher model, reasoning effort and comparison policy before
running the first task. The proxy's budget limitations remain unchanged. See
[ideas.md](../ideas.md) for the experiment protocol.
