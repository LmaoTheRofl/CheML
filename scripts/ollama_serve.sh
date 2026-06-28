#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/bin/ollama}"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Ollama executable not found: $OLLAMA_BIN" >&2
    exit 1
fi

export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"

if [[ -n "${OLLAMA_MODELS:-}" ]]; then
    mkdir -p "$OLLAMA_MODELS"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m chemx.ollama_adapter \
    --listen "${CHEMX_OLLAMA_ADAPTER_URL:-127.0.0.1:11434}" \
    --upstream "${OLLAMA_UPSTREAM_URL:-127.0.0.1:11435}" \
    --ollama-bin "$OLLAMA_BIN"
