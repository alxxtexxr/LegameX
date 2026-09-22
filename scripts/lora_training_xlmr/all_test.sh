#!/usr/bin/env bash
set -euo pipefail

BACKGROUND=false
for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND=true ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$BACKGROUND" = true ]; then
  tmux new-session -d -s "all-test-configs" "
    echo 'Running all test configs sequentially...'
    "$SCRIPT_DIR/squad-en-15K-s42_test.sh"
    "$SCRIPT_DIR/wikipedia-en-2K-s42_test.sh"
    "$SCRIPT_DIR/wikipedia-vi-3K-s42_test.sh"
    echo '---'
    echo 'All test configs finished.'
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
fi
