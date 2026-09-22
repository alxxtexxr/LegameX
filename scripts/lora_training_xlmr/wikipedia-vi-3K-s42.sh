#!/usr/bin/env bash
set -euo pipefail

BACKGROUND=false
for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND=true ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

SESSION_NAME="$(basename "$0" .sh)"

if [ "$BACKGROUND" = true ]; then
  screen -dmS "$SESSION_NAME" uv run python src/lora_training_xlmr.py \
    --config-name=wikipedia-vi-3K-s42
  echo "Started in screen session: $SESSION_NAME"
  echo "Re-attach with: screen -r $SESSION_NAME"
else
  uv run python src/lora_training_xlmr.py \
    --config-name=wikipedia-vi-3K-s42
fi
