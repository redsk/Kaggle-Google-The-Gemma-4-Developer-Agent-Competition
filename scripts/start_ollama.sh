#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OLLAMA_MODELS="$PROJECT_ROOT/models"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=32768
exec "$PROJECT_ROOT/downloads/tools/Ollama.app/Contents/Resources/ollama" serve
