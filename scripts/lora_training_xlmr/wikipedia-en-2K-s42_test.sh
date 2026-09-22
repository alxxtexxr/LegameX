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
  tmux new-session -d -s "$SESSION_NAME" "uv run python src/lora_training_xlmr.py --config-name=wikipedia-en-2K-s42_test; exec bash"
  echo "Started in tmux session: $SESSION_NAME"
  echo "Re-attach with: tmux attach -t $SESSION_NAME"
else
  uv run python src/lora_training_xlmr.py \
    --config-name=wikipedia-en-2K-s42_test
fi
