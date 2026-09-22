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
  tmux new-session -d -s "all-configs" "
    echo 'Running all configs sequentially...'
    "$SCRIPT_DIR/squad-en-15K-s42.sh"
    "$SCRIPT_DIR/wikipedia-en-2K-s42.sh"
    "$SCRIPT_DIR/wikipedia-vi-3K-s42.sh"
    echo '---'
    echo 'All configs finished.'
    exec bash
  "
  echo "Started in tmux session: all-configs"
  echo "Re-attach with: tmux attach -t all-configs"
else
  echo "Starting all configs..."
  echo "---"
  "$SCRIPT_DIR/squad-en-15K-s42.sh"
  "$SCRIPT_DIR/wikipedia-en-2K-s42.sh"
  "$SCRIPT_DIR/wikipedia-vi-3K-s42.sh"
  echo "---"
  echo "All configs finished."
fi
