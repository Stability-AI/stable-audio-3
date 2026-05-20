#!/usr/bin/env bash
#
# sa3_mlx bootstrap — Stable Audio 3 inference on Apple Silicon in one command.
#
# Hosted at:
#   https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash -s -- --prompt "Death Metal" --dit medium --decoder same-l
#
# Default demo prompt is "Epic orchestral buildup".
#
# What it does:
#   1. Verifies you're on Apple Silicon.
#   2. Downloads the optimized/mlx/ subdir of github.com/Stability-AI/stable-audio-3
#      via a tarball (curl + tar — no git, no Xcode CLT needed). Sibling
#      optimized/{tensorrt,coreml,...} subdirs are skipped.
#   3. Runs ./install.sh -y inside it (uv + Python 3.11 + venv + weight downloads).
#   4. Runs ./sa3 with whatever args you passed (default: "Epic orchestral buildup" demo + --play).
#
set -euo pipefail

REPO_OWNER="Stability-AI"
REPO_NAME="stable-audio-3"
BRANCH="main"
SUBDIR_IN_REPO="optimized/mlx"
LOCAL_DIR="sa3_mlx"
DEFAULT_ARGS=(--prompt "Epic orchestral buildup" --dit sm-music --decoder same-s --seconds 120 --play)

TAR_URL="https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$BRANCH.tar.gz"
TAR_INNER="$REPO_NAME-$BRANCH/$SUBDIR_IN_REPO"

# ── colours ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; RED=$'\033[1;31m'
    YELLOW=$'\033[1;33m'; GREEN=$'\033[1;32m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    BOLD=""; CYAN=""; RED=""; YELLOW=""; GREEN=""; DIM=""; RESET=""
fi
step() { printf '\n%s→ %s%s\n' "$CYAN" "$1" "$RESET"; }
fail() { printf '\n%serror%s: %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '%swarning%s: %s\n' "$YELLOW" "$RESET" "$1" >&2; }

# ── 1. platform sanity ──────────────────────────────────────────────────────
OS="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$OS" != "Darwin" || "$ARCH" != "arm64" ]]; then
    fail "this stack is Apple-Silicon-only (MLX is Metal-backed). Detected $OS/$ARCH."
fi
ok "platform: $OS/$ARCH"

# ── 2. preflight: curl + tar (preinstalled on macOS — should always pass) ───
for tool in curl tar; do
    command -v "$tool" >/dev/null 2>&1 || \
        fail "$tool not found on PATH. (It ships with macOS — something's unusual about your install.)"
done
ok "curl + tar present"

# ── 3. download the optimized/mlx subdir via tarball ────────────────────────
if [[ -d "$LOCAL_DIR" && -x "$LOCAL_DIR/install.sh" ]]; then
    step "Reusing existing $LOCAL_DIR/ (delete it to re-download)"
else
    if [[ -e "$LOCAL_DIR" ]]; then
        fail "./$LOCAL_DIR exists but doesn't look like a sa3_mlx checkout — remove or rename it."
    fi
    step "Downloading $REPO_OWNER/$REPO_NAME ($BRANCH) → ./$LOCAL_DIR"

    TMP_TAR="$(mktemp -t sa3_repo.XXXXXX).tar.gz"
    TMP_EXTRACT="$(mktemp -d -t sa3_extract.XXXXXX)"
    trap 'rm -rf "$TMP_TAR" "$TMP_EXTRACT"' EXIT

    # --progress-bar writes to stderr; -f makes 404/5xx a real curl error
    curl -fL --progress-bar "$TAR_URL" -o "$TMP_TAR"

    # BSD tar (macOS) extracts only paths matching the pattern.
    tar -xz -f "$TMP_TAR" -C "$TMP_EXTRACT" "$TAR_INNER"

    SRC="$TMP_EXTRACT/$TAR_INNER"
    [[ -d "$SRC" ]] || fail "Expected '$TAR_INNER' inside the tarball but didn't find it."
    mv "$SRC" "$LOCAL_DIR"
    ok "extracted $(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ') files to ./$LOCAL_DIR"
fi

# ── 4. install ──────────────────────────────────────────────────────────────
cd "$LOCAL_DIR"
[[ -x ./install.sh ]] || fail "install.sh missing or not executable in ./$LOCAL_DIR."
step "Running ./install.sh -y"
./install.sh -y

# ── 5. inference ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    step "Running ./sa3 $*"
    exec ./sa3 "$@"
else
    step "Running demo: ./sa3 ${DEFAULT_ARGS[*]}"
    printf '  %s(pass your own args via:  curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash -s -- --prompt "..." ...)%s\n' "$DIM" "$RESET"
    exec ./sa3 "${DEFAULT_ARGS[@]}"
fi
