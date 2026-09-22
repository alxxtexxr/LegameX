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

stop_vastai_instance() {
  if [ -n "$VASTAI_ID" ]; then
    echo "Stopping vastai instance $VASTAI_ID..."
    output=$(vastai stop instance "$VASTAI_ID" 2>&1)
    echo "$output"
    if echo "$output" | grep -qi 'error\|failed'; then
      echo "Failed to stop instance $VASTAI_ID."
    else
      echo "Instance $VASTAI_ID stopped."
    fi
  fi
}

if [ "$BACKGROUND" = true ]; then
  # Build stop command inline for tmux subshell
  STOP_CMD=""
  if [ -n "$VASTAI_ID" ]; then
    STOP_CMD="output=\$(vastai stop instance $VASTAI_ID 2>&1); echo \"\$output\"; if echo \"\$output\" | grep -qi 'error|failed'; then echo 'Failed to stop instance $VASTAI_ID.'; else echo 'Instance $VASTAI_ID stopped.'; fi"
  fi
  tmux new-session -d -s "all-test-configs" "
    echo 'Running all test configs sequentially...'
    \"$SCRIPT_DIR/squad-en-15K-s42_test.sh\"
    \"$SCRIPT_DIR/wikipedia-en-2K-s42_test.sh\"
    \"$SCRIPT_DIR/wikipedia-vi-3K-s42_test.sh\"
    echo '---'
    echo 'All test configs finished.'
    $STOP_CMD
    exec bash
  "
  echo "Started in tmux session: all-test-configs"
  echo "Re-attach with: tmux attach -t all-test-configs"
else
  echo "Starting all test configs..."
  echo "---"
  "$SCRIPT_DIR/squad-en-15K-s42_test.sh"
  "$SCRIPT_DIR/wikipedia-en-2K-s42_test.sh"
  "$SCRIPT_DIR/wikipedia-vi-3K-s42_test.sh"
  echo "---"
  echo "All test configs finished."
  stop_vastai_instance
fi
