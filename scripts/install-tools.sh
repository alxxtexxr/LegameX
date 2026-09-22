#!/usr/bin/env bash
set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Detect package manager ──────────────────────────────────────────────────
install_pkg() {
    local pkg="$1"
    if command -v brew &>/dev/null; then
        brew install "$pkg"
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y "$pkg"
    else
        error "No supported package manager found. Install $pkg manually."
        exit 1
    fi
}

# ─── Step 1: Install tmux ────────────────────────────────────────────────────
info "Installing tmux..."
if command -v tmux &>/dev/null; then
    info "tmux already installed ($(tmux -V))"
else
    install_pkg tmux
    info "tmux installed"
fi

# ─── Step 2: Install fish ────────────────────────────────────────────────────
info "Installing fish shell..."
if command -v fish &>/dev/null; then
    info "fish already installed ($(fish --version))"
else
    install_pkg fish
    info "fish installed"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
info "Tools installed successfully!"
