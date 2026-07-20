#!/usr/bin/env bash
#
# SA3 MLX installer — uv-based.
#
# Creates a project-local .venv/ with the right Python and runtime deps,
# then hands off to install.py for the interactive weight-download prompt.
#
# Usage:
#   ./install.sh                  # auto-detect uv, prompt to install if missing
#   ./install.sh -y               # assume yes to "install uv?" prompt
#   ./install.sh --python VER     # pin a specific Python (default: 3.11)
#
# After install:
#   source .venv/bin/activate
#   python sa3_mlx.py --prompt "lofi house" --dit medium --decoder same-l
#   # or, without activating:
#   .venv/bin/python sa3_mlx.py --prompt "lofi house" --dit medium --decoder same-l
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PY_VERSION_DEFAULT="3.11"

# ── colours ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; RED=$'\033[1;31m'
    YELLOW=$'\033[1;33m'; GREEN=$'\033[1;32m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    BOLD=""; CYAN=""; RED=""; YELLOW=""; GREEN=""; DIM=""; RESET=""
fi
step() { printf '\n%s→ %s%s\n' "$CYAN" "$1" "$RESET"; }
fail() { printf '%serror%s: %s\n' "$RED" "$RESET" "$1" >&2; }
warn() { printf '%swarning%s: %s\n' "$YELLOW" "$RESET" "$1" >&2; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }

# ── arg parsing ─────────────────────────────────────────────────────────────
ASSUME_YES=0
PY_VERSION="$PY_VERSION_DEFAULT"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1; shift ;;
        --python) PY_VERSION="$2"; shift 2 ;;
        --python=*) PY_VERSION="${1#--python=}"; shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── platform sanity ─────────────────────────────────────────────────────────
OS="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$OS" != "Darwin" || "$ARCH" != "arm64" ]]; then
    warn "this stack is Apple-Silicon-only (MLX is Metal-backed). Detected $OS/$ARCH — installs will probably fail."
fi

# ── ensure uv is installed ──────────────────────────────────────────────────
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        ok "uv $(uv --version 2>/dev/null | awk '{print $2}') already installed at $(command -v uv)"
        return 0
    fi

    step "uv not found — uv is required (much faster than pip, also manages Python versions)"
    if [[ "$ASSUME_YES" -ne 1 ]]; then
        printf '  Install uv now via the official installer? (curl + sh) %s[Y/n]%s ' "$DIM" "$RESET"
        read -r REPLY < /dev/tty
        case "$REPLY" in
            ""|y|Y|yes|YES) ;;
            *)
                fail "install aborted — install uv manually then re-run:"
                printf '    curl -LsSf https://astral.sh/uv/install.sh | sh\n' >&2
                printf '  or: brew install uv\n' >&2
                exit 1 ;;
        esac
    fi

    step "Installing uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        fail "uv installer failed. Try a manual install:"
        printf '    brew install uv\n' >&2
        exit 1
    fi
    # The installer drops uv at ~/.local/bin/uv (or $XDG_BIN_HOME)
    export PATH="$HOME/.local/bin:${XDG_BIN_HOME:-}:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        fail "uv was installed but isn't on PATH. Add ~/.local/bin to PATH, restart your shell, and re-run install.sh."
        exit 1
    fi
    ok "uv $(uv --version | awk '{print $2}') installed"
}

ensure_uv

# ── create venv (uv auto-installs the requested Python if missing) ──────────
step "Creating virtual environment at .venv/ with Python $PY_VERSION"
if [[ -d "$VENV_DIR" ]]; then
    EXISTING_PY=$("$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "unknown")
    if [[ "$EXISTING_PY" == "$PY_VERSION"* ]]; then
        ok "reusing existing .venv (Python $EXISTING_PY)"
    else
        warn "existing .venv uses Python $EXISTING_PY (wanted $PY_VERSION) — recreating"
        rm -rf "$VENV_DIR"
        uv venv --seed --python "$PY_VERSION" "$VENV_DIR"
    fi
else
    uv venv --seed --python "$PY_VERSION" "$VENV_DIR"
fi

# ── install runtime deps ────────────────────────────────────────────────────
step "Installing dependencies (uv pip install -r requirements.txt)"
VIRTUAL_ENV="$VENV_DIR" uv pip install -r "$SCRIPT_DIR/requirements.txt"

# ── ffmpeg (optional) ────────────────────────────────────────────────────────
# Soft dependency: dataset decode already works via libsndfile (bundled with the
# soundfile wheel just installed), and training demos fall back to WAV without
# ffmpeg. It's only needed for compact MP3 demos + sa3_mlx.py's exotic-WAV
# fallback path — so a missing ffmpeg is a warning, never a failure.
step "Checking for ffmpeg (optional)"
if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg found at $(command -v ffmpeg)"
else
    warn "ffmpeg not found — demos will save as WAV (larger); exotic-format audio"
    warn "  loading is disabled. Install it (fast, prebuilt bottle on Apple Silicon):"
    printf '    %sbrew install ffmpeg%s\n' "$BOLD" "$RESET" >&2
fi

# ── hand off to install.py ──────────────────────────────────────────────────
# Any unrecognized args we collected (e.g. --download medium,sm-music) get
# forwarded to install.py via EXTRA_ARGS.
step "Bundle picker"
INSTALL_SKIP_PIP=1 exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/scripts/install.py" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
