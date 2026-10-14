#!/bin/sh
# Interactive local chat; --verbose prints generation speed after each response.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OLLAMA_HOST=127.0.0.1:11434
exec "$PROJECT_ROOT/downloads/tools/Ollama.app/Contents/Resources/ollama" \
  run gemma4-12b-chat --think=false --verbose "$@"
