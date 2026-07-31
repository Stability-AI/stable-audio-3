#!/usr/bin/env bash
#
# sa3_trt bootstrap — Stable Audio 3 inference on NVIDIA GPUs in one command.
#
# Hosted at:
#   https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh | bash
#   curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh | bash -s -- --prompt "Death Metal" --dit medium --decoder same-l
#
# Default demo prompt is "Death Metal".
#
# What it does:
#   1. Verifies you're on Linux with an NVIDIA GPU.
#   2. Installs `git` via the system package manager if it's not already present,
#      then shallow-clones github.com/Stability-AI/stable-audio-3.
#   3. cd's into optimized/tensorRT/ and runs ./install.sh -y (uv + Python +
#      venv + arch-aware TRT engine downloads, or build-from-ONNX if no
#      prebuilt engines exist for this arch).
#   4. Runs ./sa3 with whatever args you passed (default: "Death Metal" demo,
#      medium DiT + SAME-L decoder, 120s).
#
set -euo pipefail

REPO_OWNER="Stability-AI"
REPO_NAME="stable-audio-3"
BRANCH="main"
SUBDIR_IN_REPO="optimized/tensorRT"
CLONE_DIR="$REPO_NAME"                    # full repo cloned here
WORKDIR="$CLONE_DIR/$SUBDIR_IN_REPO"      # where install.sh + sa3 live
DEFAULT_ARGS=(--prompt "Death Metal" --dit medium --decoder same-l --seconds 120)

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
if [[ "$OS" != "Linux" ]]; then
    printf '\n%serror%s: this stack is Linux-only (TensorRT requires the NVIDIA Linux CUDA toolchain). Detected %s/%s.\n' "$RED" "$RESET" "$OS" "$ARCH" >&2
    printf '\n  For your platform use one of the other optimized routes:\n' >&2
    printf '    CPU (macOS/Linux/Windows, x86/ARM):\n' >&2
    printf '      curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash\n' >&2
    printf '      (Windows: irm https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.ps1 | iex)\n' >&2
    printf '    Apple Silicon GPU (fastest on Mac):\n' >&2
    printf '      curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash\n' >&2
    exit 1
fi
if [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
    warn "untested architecture $ARCH — TensorRT typically supports x86_64 and aarch64."
fi
ok "platform: $OS/$ARCH"

# ── 2. preflight: nvidia-smi ────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found on PATH."
GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
[[ -n "$GPU_INFO" ]] || fail "nvidia-smi ran but reported no GPUs."
ok "GPU: $GPU_INFO"

# ── 2b. shortcut: already inside a checkout? skip clone, run in place ──────
# If the user ran bootstrap from inside optimized/tensorRT/ of an existing
# clone (e.g. for a re-install or to pick up fresh code after a `git pull`),
# don't try to clone the repo into a subdir — just use the current dir.
if [[ -f ./install.sh && -x ./sa3 && -d ./build && -d ./scripts ]]; then
    step "Already inside an optimized/tensorRT/ checkout — using ./ in place"
    cd "$(pwd)"   # no-op; just makes the path absolute for later messages
    if [[ ! -f ../../stable_audio_3/audio_output.py && ! -f ./scripts/audio_output.py ]]; then
        command -v curl >/dev/null 2>&1 || \
            fail "curl is required to complete this standalone TensorRT checkout."
        step "Fetching shared audio output helper for the standalone checkout"
        curl -fL --progress-bar \
            "https://raw.githubusercontent.com/$REPO_OWNER/$REPO_NAME/$BRANCH/stable_audio_3/audio_output.py" \
            -o ./scripts/audio_output.py
    fi
    [[ -x ./install.sh ]] || fail "install.sh not executable in $(pwd)."
    ok "running install.sh in $(pwd)"
    ./install.sh -y

    if [[ $# -gt 0 ]]; then
        step "Running ./sa3 $*"
        exec ./sa3 "$@"
    else
        step "Running demo: ./sa3 ${DEFAULT_ARGS[*]}"
        exec ./sa3 "${DEFAULT_ARGS[@]}"
    fi
fi

# ── 3. ensure git (with tarball fallback) ──────────────────────────────────
# Try every common Linux package manager; if all fail (no sudo, locked-down
# container, exotic distro, network issue, etc.) fall back to a curl+tar
# download of the GitHub source tarball. Either path ends with the repo
# checked out at $CLONE_DIR/.
if ! command -v git >/dev/null 2>&1; then
    step "git not found — trying system package managers"
    SUDO=""; [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
    INSTALLED=0
    for pm in apt-get dnf yum apk pacman zypper; do
        command -v "$pm" >/dev/null 2>&1 || continue
        case "$pm" in
            apt-get) cmd="$SUDO apt-get update -qq && $SUDO apt-get install -y git" ;;
            dnf)     cmd="$SUDO dnf install -y git" ;;
            yum)     cmd="$SUDO yum install -y git" ;;
            apk)     cmd="$SUDO apk add git" ;;
            pacman)  cmd="$SUDO pacman -Sy --noconfirm git" ;;
            zypper)  cmd="$SUDO zypper install -y git" ;;
        esac
        printf "  %s%s%s\n" "$DIM" "trying: $cmd" "$RESET"
        # Run under bash -c so the && chain works without trying to shell-parse it twice.
        if bash -c "$cmd"; then
            INSTALLED=1
            break
        else
            warn "$pm couldn't install git — trying next"
        fi
    done
    if [[ "$INSTALLED" -ne 1 ]] || ! command -v git >/dev/null 2>&1; then
        warn "couldn't install git via package manager — falling back to tarball download"
    fi
fi

HAVE_GIT=0
if command -v git >/dev/null 2>&1; then
    HAVE_GIT=1
    ok "git: $(git --version | awk '{print $3}')"
else
    # Need curl + tar for the fallback path.
    for tool in curl tar; do
        command -v "$tool" >/dev/null 2>&1 || \
            fail "$tool not found — required for the tarball fallback. Install git OR ($tool) manually."
    done
    ok "tarball fallback path (curl + tar both present)"
fi

# ── 4. download the repo (clone if we have git, else tarball) ──────────────
if [[ -d "$WORKDIR" && -x "$WORKDIR/install.sh" ]]; then
    if [[ "$HAVE_GIT" -eq 1 ]]; then
        step "Reusing existing $CLONE_DIR/ (delete or 'cd $CLONE_DIR && git pull' to refresh)"
    else
        step "Reusing existing $CLONE_DIR/ (delete to re-download)"
    fi
else
    if [[ -e "$CLONE_DIR" ]]; then
        fail "./$CLONE_DIR exists but doesn't have $SUBDIR_IN_REPO/install.sh — remove or rename it."
    fi
    if [[ "$HAVE_GIT" -eq 1 ]]; then
        step "Cloning $REPO_OWNER/$REPO_NAME ($BRANCH) → ./$CLONE_DIR"
        git clone --depth 1 -b "$BRANCH" \
            "https://github.com/$REPO_OWNER/$REPO_NAME.git" "$CLONE_DIR"
    else
        step "Downloading $REPO_OWNER/$REPO_NAME ($BRANCH) tarball → ./$CLONE_DIR"
        TAR_URL="https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$BRANCH.tar.gz"
        TMP_TAR="$(mktemp -t sa3_repo.XXXXXX).tar.gz"
        TMP_EXTRACT="$(mktemp -d -t sa3_extract.XXXXXX)"
        trap 'rm -rf "$TMP_TAR" "$TMP_EXTRACT"' EXIT
        curl -fL --progress-bar "$TAR_URL" -o "$TMP_TAR"
        tar -xz -f "$TMP_TAR" -C "$TMP_EXTRACT"
        SRC="$TMP_EXTRACT/$REPO_NAME-$BRANCH"
        [[ -d "$SRC" ]] || fail "expected $REPO_NAME-$BRANCH/ inside the tarball — repo layout may have changed."
        mv "$SRC" "$CLONE_DIR"
        warn "tarball checkout — updates require re-running bootstrap (no 'git pull')."
    fi
    [[ -d "$WORKDIR" ]] || fail "didn't find $SUBDIR_IN_REPO/ inside ./$CLONE_DIR."
    ok "ready at ./$WORKDIR"
fi

# ── 5. install ──────────────────────────────────────────────────────────────
cd "$WORKDIR"
[[ -x ./install.sh ]] || fail "install.sh missing or not executable in ./$WORKDIR."
step "Running ./install.sh -y"
./install.sh -y

# ── 6. inference ────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    step "Running ./sa3 $*"
    exec ./sa3 "$@"
else
    step "Running demo: ./sa3 ${DEFAULT_ARGS[*]}"
    printf '  %s(pass your own args via:  curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh | bash -s -- --prompt "..." ...)%s\n' "$DIM" "$RESET"
    exec ./sa3 "${DEFAULT_ARGS[@]}"
fi
