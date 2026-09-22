#!/usr/bin/env bash
set -euo pipefail

BACKGROUND=false
VASTAI_ID=""
for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND=true ;;
    --vastai-id) shift; VASTAI_ID="$1" ;;
    --vastai-id=*) VASTAI_ID="${arg#*=}" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

SESSION_NAME="$(basename "$0" .sh)"

stop_vastai_instance() {
  if [ -n "$VASTAI_ID" ]; then
    echo "Stopping vastai instance $VASTAI_ID..."
    output=$(uv run vastai stop instance "$VASTAI_ID" 2>&1)
    echo "$output"
    if echo "$output" | grep -qi 'error\|failed'; then
      echo "Failed to stop instance $VASTAI_ID."
    else
      echo "Instance $VASTAI_ID stopped."
    fi
  fi
}

if [ "$BACKGROUND" = true ]; then
  STOP_CMD=""
  if [ -n "$VASTAI_ID" ]; then
    STOP_CMD="output=\$(uv run vastai stop instance $VASTAI_ID 2>&1); echo \"\$output\"; if echo \"\$output\" | grep -qi 'error|failed'; then echo 'Failed to stop instance $VASTAI_ID.'; else echo 'Instance $VASTAI_ID stopped.'; fi"
  fi
  tmux new-session -d -s "$SESSION_NAME" "uv run python src/lora_training_xlmr.py --config-name=wikipedia-vi-3K-s42_test; $STOP_CMD; exec bash"
  echo "Started in tmux session: $SESSION_NAME"
  echo "Re-attach with: tmux attach -t $SESSION_NAME"
else
  uv run python src/lora_training_xlmr.py \
    --config-name=wikipedia-vi-3K-s42_test
  stop_vastai_instance
fi
