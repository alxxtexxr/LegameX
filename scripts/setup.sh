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

# ─── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --hf-token <HF_TOKEN> [OPTIONS]

Required:
  --hf-token <HF_TOKEN>       Hugging Face token

Optional:
  --with-tools                Also install tmux and fish (runs install-tools.sh)
  --wandb-key <WANDB_KEY>     Weights & Biases API key
  --vastai-key <VASTAI_KEY>   Vast.ai API key
  -h, --help                  Show this help message
EOF
    exit 1
}

HF_TOKEN=""
WANDB_KEY=""
VASTAI_KEY=""
INSTALL_TOOLS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token)    HF_TOKEN="$2";   shift 2 ;;
        --with-tools)  INSTALL_TOOLS=true; shift ;;
        --wandb-key)   WANDB_KEY="$2";  shift 2 ;;
        --vastai-key)  VASTAI_KEY="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *)             error "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$HF_TOKEN" ]]; then
    error "Missing required argument: --hf-token"
    usage
fi

# ─── Step 1: Install tools (optional) ───────────────────────────────────────
if [[ "$INSTALL_TOOLS" == true ]]; then
    info "Running install-tools.sh..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    bash "$SCRIPT_DIR/install-tools.sh"
else
    warn "Skipping tool installation (pass --with-tools to include)"
fi

# ─── Step 2: Install Python dependencies with uv ─────────────────────────────
info "Installing Python dependencies with uv..."
if ! command -v uv &>/dev/null; then
    error "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
uv sync
info "Python dependencies installed"

# ─── Step 3: Login Hugging Face ──────────────────────────────────────────────
info "Configuring Hugging Face..."
git config --global credential.helper store
uv run hf auth login --add-to-git-credential --token "$HF_TOKEN"
info "Hugging Face logged in"

# ─── Step 4: Login wandb (optional) ──────────────────────────────────────────
if [[ -n "$WANDB_KEY" ]]; then
    info "Logging in to Weights & Biases..."
    uv run wandb login "$WANDB_KEY"
    info "wandb logged in"
else
    warn "Skipping wandb login (no --wandb-key provided)"
fi

# ─── Step 5: Set Vast.ai API key (optional) ──────────────────────────────────
if [[ -n "$VASTAI_KEY" ]]; then
    info "Setting Vast.ai API key..."
    uv run vastai set api-key "$VASTAI_KEY"
    info "Vast.ai API key set"
else
    warn "Skipping Vast.ai API key (no --vastai-key provided)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
info "Setup complete! You're ready to go."
