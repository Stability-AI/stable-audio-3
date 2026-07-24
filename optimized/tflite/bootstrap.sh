#!/usr/bin/env bash
#
# sa3_tflite bootstrap — Stable Audio 3 inference on CPU (LiteRT/TFLite) in one command.
#
# Hosted at:
#   https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash -s -- --prompt "Death Metal" --dit medium --decoder same-l
#
# Default demo prompt is "Impending tribal, epic orchestral buildup".
#
# What it does:
#   1. Verifies curl + tar are present (portable CPU stack — runs on macOS/Linux, x86/ARM).
#   2. Fetches the project:
#        - If git is installed → `git clone --depth=1` into ./stable-audio-3/,
#          then cd into optimized/tflite/ (real repo; pullable, modifiable).
#        - If not → tarball pull via curl + tar, extracting only optimized/tflite/
#          into ./sa3_tflite/ (no git needed).
#   3. Runs ./install.sh -y inside it (uv + Python 3.11 + venv + weight downloads).
#   4. Runs ./sa3 with whatever args you passed (default: "Impending tribal, epic orchestral buildup" demo + --play).
#
set -euo pipefail

# uv's curl installer drops the binary at $XDG_BIN_HOME (~/.local/bin by default)
# and updates the user's shell profile — but that profile only takes effect in
# *new* shells. We pre-emptively put both locations on PATH so the just-installed
# uv (and anything else from this run) is findable in the current process tree.
export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.local/bin:$PATH"

REPO_OWNER="Stability-AI"
REPO_NAME="stable-audio-3"
BRANCH="main"
SUBDIR_IN_REPO="optimized/tflite"
LOCAL_DIR="sa3_tflite"
DEFAULT_ARGS=(--prompt "Impending tribal, epic orchestral buildup" --dit sm-music --decoder same-s --seconds 30 --play)

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

# ── 1. platform note (portable — no gate) ───────────────────────────────────
OS="$(uname -s)"; ARCH="$(uname -m)"
ok "platform: $OS/$ARCH  (LiteRT/TFLite is CPU-portable)"

# ── 2. preflight: curl + tar (preinstalled on macOS/most Linux) ─────────────
for tool in curl tar; do
    command -v "$tool" >/dev/null 2>&1 || \
        fail "$tool not found on PATH. Install it and re-run."
done
ok "curl + tar present"

# ── 3. fetch the project ────────────────────────────────────────────────────
# Prefer `git clone` if git is on the machine — the user gets a real repo
# they can pull updates from / navigate sibling subdirs in (mlx/, tensorRT/).
# Falls back to a tarball pull (curl + tar) if git is missing.

if command -v git >/dev/null 2>&1; then
    GIT_DIR="$REPO_NAME"
    WORK_DIR="$GIT_DIR/$SUBDIR_IN_REPO"

    if [[ -d "$GIT_DIR/.git" ]]; then
        step "Reusing existing ./$GIT_DIR (git pull --ff-only)"
        git -C "$GIT_DIR" pull --ff-only
    elif [[ -e "$GIT_DIR" ]]; then
        fail "./$GIT_DIR exists but isn't a git repo — remove or rename it."
    else
        step "git clone https://github.com/$REPO_OWNER/$REPO_NAME → ./$GIT_DIR"
        git clone --depth=1 "https://github.com/$REPO_OWNER/$REPO_NAME" "$GIT_DIR"
    fi

    [[ -d "$WORK_DIR" ]] || \
        fail "Expected '$SUBDIR_IN_REPO' inside the repo but didn't find it."
    ok "ready at ./$WORK_DIR"
else
    # No git — pull a tarball and extract only optimized/tflite/.
    WORK_DIR="$LOCAL_DIR"

    if [[ -d "$LOCAL_DIR" && -x "$LOCAL_DIR/install.sh" ]]; then
        step "Reusing existing $LOCAL_DIR/ (delete it to re-download)"
    else
        if [[ -e "$LOCAL_DIR" ]]; then
            fail "./$LOCAL_DIR exists but doesn't look like a sa3_tflite checkout — remove or rename it."
        fi
        step "git not installed — downloading $REPO_OWNER/$REPO_NAME ($BRANCH) tarball → ./$LOCAL_DIR"

        TMP_TAR="$(mktemp -t sa3_repo.XXXXXX).tar.gz"
        TMP_EXTRACT="$(mktemp -d -t sa3_extract.XXXXXX)"
        trap 'rm -rf "$TMP_TAR" "$TMP_EXTRACT"' EXIT

        # --progress-bar writes to stderr; -f makes 404/5xx a real curl error
        curl -fL --progress-bar "$TAR_URL" -o "$TMP_TAR"

        # BSD tar (macOS) / GNU tar both extract only paths matching the pattern.
        tar -xz -f "$TMP_TAR" -C "$TMP_EXTRACT" "$TAR_INNER"

        SRC="$TMP_EXTRACT/$TAR_INNER"
        [[ -d "$SRC" ]] || fail "Expected '$TAR_INNER' inside the tarball but didn't find it."
        mv "$SRC" "$LOCAL_DIR"
        ok "extracted $(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ') files to ./$LOCAL_DIR"
    fi
fi

# ── 4. install ──────────────────────────────────────────────────────────────
cd "$WORK_DIR"
[[ -x ./install.sh ]] || fail "install.sh missing or not executable in ./$WORK_DIR."
step "Running ./install.sh -y"
./install.sh -y

# ── 5. inference ────────────────────────────────────────────────────────────
# Run as a subprocess (not `exec`) so we can drop the user into an
# interactive shell here when it finishes.
if [[ $# -gt 0 ]]; then
    step "Running ./sa3 $*"
    ./sa3 "$@" || true
else
    step "Running demo: ./sa3 ${DEFAULT_ARGS[*]}"
    printf '  %s(pass your own args via:  curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash -s -- --prompt "..." ...)%s\n' "$DIM" "$RESET"
    ./sa3 "${DEFAULT_ARGS[@]}" || true
fi

# ── 6. drop user into an interactive shell sitting in the project dir ──────
# A subprocess can't change its parent shell's cwd — but we CAN replace
# ourselves with a fresh interactive shell, leaving the user at a prompt
# inside ./$WORK_DIR. `exit` (or Ctrl-D) returns them to their original
# shell, at their original cwd, just like a normal subshell.
#
# `< /dev/tty` is essential when bootstrap.sh was invoked via curl|bash:
# stdin at this point is the (closed) pipe; an interactive shell needs a
# real terminal. /dev/tty always refers to the user's controlling TTY.

if [[ ! -e /dev/tty ]]; then
    # Headless / scripted invocation — skip the shell drop.
    exit 0
fi

USER_SHELL="${SHELL:-/bin/bash}"
printf '\n%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n' "$BOLD" "$RESET"
printf '  %s✓ you are now in%s %s%s%s\n' "$GREEN" "$RESET" "$BOLD" "$(pwd)" "$RESET"
printf '    %stype %s./sa3 --help%s for options, or %sexit%s to return to your previous shell%s\n' \
    "$DIM" "$RESET$BOLD" "$RESET$DIM" "$RESET$BOLD" "$RESET$DIM" "$RESET"
printf '%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n' "$BOLD" "$RESET"

exec "$USER_SHELL" -i < /dev/tty
