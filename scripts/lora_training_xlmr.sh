#!/usr/bin/env bash
set -euo pipefail

CONFIGS=()
BACKGROUND=false
VASTAI_ID=""

for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND=true ;;
    --vastai-id) shift; VASTAI_ID="$1" ;;
    --vastai-id=*) VASTAI_ID="${arg#*=}" ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *)  CONFIGS+=("$arg") ;;
  esac
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "Usage: $0 <config-name>... [--background|-b] [--vastai-id <id>]" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 squad-en-15K-s42" >&2
  echo "  $0 wikipedia-en-2K-s42 --background" >&2
  echo "  $0 wikipedia-vi-3K-s42 -b --vastai-id=12345" >&2
  echo "  $0 squad-en-15K-s42 wikipedia-en-2K-s42 wikipedia-vi-3K-s42" >&2
  echo "  $0 squad-en-15K-s42 wikipedia-en-2K-s42 wikipedia-vi-3K-s42 -b" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

SESSION_NAME="${CONFIGS[0]}"
[ ${#CONFIGS[@]} -gt 1 ] && SESSION_NAME="all-configs"

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

run_config() {
  local config="$1"
  echo "=== Running config: $config ==="
  uv run python src/lora_training_xlmr.py \
    --config-name="$config"
}

if [ "$BACKGROUND" = true ]; then
  STOP_CMD=""
  if [ -n "$VASTAI_ID" ]; then
    STOP_CMD="output=\$(uv run vastai stop instance $VASTAI_ID 2>&1); echo \"\$output\"; if echo \"\$output\" | grep -qi 'error|failed'; then echo 'Failed to stop instance $VASTAI_ID.'; else echo 'Instance $VASTAI_ID stopped.'; fi"
  fi

  # Build the command chain for tmux
  CMDS=""
  for config in "${CONFIGS[@]}"; do
    CMDS+="uv run python src/lora_training_xlmr.py --config-name=$config && "
  done
  CMDS="${CMDS% && }"

  tmux new-session -d -s "$SESSION_NAME" "
    $CMDS
    echo '---'
    echo 'All configs finished.'
    $STOP_CMD
    exec bash
  "
  echo "Started in tmux session: $SESSION_NAME"
  echo "Re-attach with: tmux attach -t $SESSION_NAME"
else
  for config in "${CONFIGS[@]}"; do
    run_config "$config"
  done
  echo "---"
  echo "All configs finished."
  stop_vastai_instance
fi
