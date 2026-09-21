#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/wikipedia-en-2K-s42_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

cd "$PROJECT_ROOT"

uv run python src/lora_training_xlmr.py \
    --config-name=wikipedia-en-2K-s42 \
    2>&1 | tee "$LOG_FILE"
