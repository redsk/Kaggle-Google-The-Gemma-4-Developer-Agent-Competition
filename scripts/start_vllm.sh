#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
# Match the official starter's serving settings where supported on Metal.
# Sampling comes from the request/model configuration, with no global token cap.
# See dev/challenge-configuration-audit.md for unavoidable runtime differences.
exec "$PROJECT_ROOT/.venv-vllm-metal/bin/vllm" serve \
  "$PROJECT_ROOT/models/vllm/gemma-4-12B-it-qat-4bit" \
  --served-model-name gemma4-12b-local \
  --host 127.0.0.1 --port 8000 \
  --language-model-only \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}' \
  --no-scheduler-reserve-full-isl
