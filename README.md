# Gemma 4 Developer Agent experiments

Local development for the [Google – The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).

The project runs the official SWE-Gemma/ADK harness with a local Gemma 4 12B
endpoint, inspects agent behavior, verifies saved patches, and benchmarks local
inference. The next experiment is to run a stronger model through the same
harness and learn from its observable actions: see [ideas.md](ideas.md).

## Current state

- The local Apple Silicon runtime is vLLM-Metal; Ollama utilities are also included.
- The first-task Gemma experiment was stopped after repeated inspections without
  edits. No successful solution is claimed.
- The supervised runner defaults to the first dataset entry. It imposes no custom
  turn/tool/time limit unless `--minutes` is supplied. The upstream 500-call
  fallback still exists. Monitoring reports alerts; automatic review pauses are opt-in.
- The agent declares the maximum permitted output and thinking budgets: 32,768
  tokens each, sharing a 32,768-token context with the input. The local vLLM client
  lets the server use the remaining context for output; there is no custom global
  output cap. Metal still **cannot enforce a separate thinking budget**.
- Pytest/unittest are enabled, and command/verification limits use the official
  harness defaults. See the [configuration audit](dev/challenge-configuration-audit.md).
- CLIProxyAPI 8.0.29 is installed and authenticated locally. Synthetic checks
  with GPT-6.1 Sol passed through ADK/LiteLLM, but this route ignores output-token
  ceilings. See [setup and findings](dev/cliproxyapi.md). No teacher task has run.
- Phoenix 20.21.4 is installed locally. The shared runner records both model
  paths, tools, summaries and verification. Scripted and live remote tracing
  preflights passed; full task runs remain untested. See [observability](dev/phoenix.md).
- Kaggle GPU jobs require explicit approval. The commands below are local.

## What belongs in Git

This repository contains our scripts, agent configuration, dependency lockfile,
runtime manifests and documentation. Downloaded wheels, model checkpoints,
competition tasks/snapshots/graphs/embeddings, environments, credentials and raw
run traces stay local and are ignored. The thermal chart is a curated aggregate
figure, not a competition trace.

Obtain competition assets through your own authenticated Kaggle account after
accepting the [official rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules).
The downloaded rules snapshot and the abandoned capped Kaggle notebook/generator
are also excluded. No notebook upload or cloud execution runs during setup.

## Local setup

The tested machine is an Apple Silicon Mac. Install `uv`, authenticate with Kaggle,
and provide a Docker-compatible container engine for harness execution. Podman
can be used when its Docker API socket is exposed to the Python Docker SDK.

### 1. Official harness: Python 3.13

The lockfile references four official wheels that must exist before `uv sync`.
Download only those files from wheelhouse version 31 and verify their recorded
hashes:

```sh
uv run --no-project --python 3.13 --with kagglehub python - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import kagglehub

manifest = json.loads(Path("dev/harness-wheelhouse.json").read_text())
target = Path("downloads/harness")
target.mkdir(parents=True, exist_ok=True)
for entry in manifest["files"]:
    source = Path(kagglehub.dataset_download(
        "metric/gemma-4-developer-agent-wheelhouse/versions/31",
        path=entry["file"],
    ))
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {entry['file']}")
    shutil.copy2(source, target / entry["file"])
PY

uv sync --frozen
.venv/bin/python scripts/fetch_smoke_assets.py
```

The asset script downloads the task index and the first task's supporting assets;
it does not run inference. `--task-id` selects another public dataset task.

Build the official sandbox and the small local source-import overlay:

```sh
docker build -f downloads/competition/docker/Dockerfile.sandbox \
  -t localhost/gemma-swebench-sandbox:2026-10-14 downloads/competition/docker
docker build -f dev/Dockerfile.local \
  -t localhost/gemma-swebench-sandbox-local:2026-10-14 dev
```

The overlay prioritizes `/workspace` sources so an installed dependency wheel
cannot mask the code being edited. Local ARM64 execution is not identical to
Kaggle's evaluation environment.

### 2. Local vLLM-Metal: Python 3.12

The separate environment matches the available macOS runtime wheels and keeps
their dependencies separate from the Python 3.13 competition harness.

```sh
uv venv --python 3.12 .venv-vllm-metal
uv pip install --python .venv-vllm-metal/bin/python -r dev/requirements.vllm-metal.txt

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/gemma-4-12B-it-qat-4bit",
    revision="e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6",
    local_dir="models/vllm/gemma-4-12B-it-qat-4bit",
)
PY

sh scripts/start_vllm.sh
```

The launcher binds to `127.0.0.1:8000`, enables Gemma tool/reasoning parsers and
configures a 32,768-token context. [dev/vllm-runtime.json](dev/vllm-runtime.json)
records package versions, current configuration and known limitations. The
audited launcher settings have passed offline checks but have not yet been run
with a loaded model. This is not a complete lockfile for the separate runtime.

The official evaluation uses Google's QAT W4A16 compressed-tensors checkpoint on
CUDA. Our MLX conversion and Ollama's NVFP4 artifact differ from it and from each
other. Local speed comparisons include those differences.

### 3. Run and inspect

With the assets, container engine and local model server ready:

```sh
# Inspect configuration without inference or containers.
.venv/bin/python scripts/run_supervised.py --print-config

# No model calls: verify the unpatched task in the local sandbox.
.venv/bin/python scripts/run_supervised.py --baseline

# Runs local inference on the first dataset entry.
.venv/bin/python scripts/run_supervised.py

# Verify an existing generated patch in a fresh verification container.
.venv/bin/python scripts/verify_patch.py results/YOUR_RUN/agent.patch
```

Results, full local traces and progress events go under `results/`, including
`effective-config.json` and `request-budget.jsonl`. Repeated failures, calls or
missing edits produce alerts without stopping the agent. `--pause-on-alert`
explicitly opts into automatic review pauses. Creating `pause.json` manually
still pauses at the next model-call boundary; remove it after review to resume.
Optional human guidance is recorded separately.

`check_reference_patch.py` is a separate fixture check for the published reference
patch. Its data must stay outside the agent's view and outside Git.

The runner's endpoint guard accepts localhost URLs. `--backend cliproxy` forwards
inference remotely and requires an explicit `--remote-model` and
`--reasoning-effort`; manifests and trace metadata label that route. See the
[Phoenix guide](dev/phoenix.md) for both paths and the remaining budget differences.

## Other utilities and notes

- [Local speed benchmarks](dev/benchmark_12b.md): identical short prompts,
  streaming traces and server/client timing comparisons.
- [Thermal slowdown investigation](dev/slowdown-investigation-2026-10-14.md):
  thermal throttling was reproduced; control cooling when comparing results.
- `scripts/monitor_local_inference.py`: macOS counters and optional `macmon`
  hardware sensors; it does not change fan settings.
- `scripts/check_vllm.py` and `scripts/diagnose_vllm_budget.py`: tool-call/output
  limit checks and a diagnostic for the unsupported thinking budget.
- Offline checks against the downloaded official assets:
  `.venv/bin/python -m unittest discover -s tests -p 'test_challenge_configuration.py'`.
- Ollama scripts expect its app binary under
  `downloads/tools/Ollama.app/Contents/Resources/ollama`. Pull `gemma4:12b` using
  that local server and create the chat alias from
  `dev/gemma4-12b-chat.Modelfile`; `configure_ollama.py` creates the separate harness
  alias and makes a tool-call probe. Downloads are not included here.

Neither observability setup nor a stronger-model run is part of repository setup.
