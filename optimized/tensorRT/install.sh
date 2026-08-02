#!/usr/bin/env bash
# Install sa3_trt.py dependencies (via uv) + download the TRT engine sets you want.
#
# Usage:
#   bash install.sh                  # interactive: picks engines + installs deps
#   bash install.sh --no-deps        # skip dependency install (just download engines)
#   bash install.sh --engines medium # non-interactive download (medium | sm-music | sm-sfx | all | skip)
#   bash install.sh --help

set -e

REPO_ID="stabilityai/stable-audio-3-optimized"
HF_SUBDIR=""  # Filled in by GPU-architecture detection below (e.g., "tensorRT/sm_90").
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
VENV_DIR="${SCRIPT_DIR}/.venv"
DEPS=1
PICK=""
ASSUME_YES=0

# ─── Colors (auto-detect TTY) ──────────────────────────────────────────────
if [ -t 1 ] && [ -n "${TERM:-}" ] && [ "${TERM}" != "dumb" ]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_MAGENTA=$'\033[35m'
    C_CYAN=$'\033[36m'
else
    C_RESET= C_BOLD= C_DIM= C_RED= C_GREEN= C_YELLOW= C_BLUE= C_MAGENTA= C_CYAN=
fi
bold()    { printf "%s%s%s" "$C_BOLD" "$*" "$C_RESET"; }
dim()     { printf "%s%s%s" "$C_DIM" "$*" "$C_RESET"; }
cyan()    { printf "%s%s%s" "$C_CYAN" "$*" "$C_RESET"; }
green()   { printf "%s%s%s" "$C_GREEN" "$*" "$C_RESET"; }
yellow()  { printf "%s%s%s" "$C_YELLOW" "$*" "$C_RESET"; }
magenta() { printf "%s%s%s" "$C_MAGENTA" "$*" "$C_RESET"; }
red()     { printf "%s%s%s" "$C_RED" "$*" "$C_RESET"; }
RULE_W=64
RULE=$(printf '━%.0s' $(seq 1 $RULE_W))
rule()    { printf "%s%s%s\n" "$C_CYAN" "$RULE" "$C_RESET"; }
banner()  { rule; printf "  %s\n" "$(bold "$*")"; rule; }
section() { printf "\n%s %s\n\n" "$(cyan '━━━')" "$(bold "$*")"; }
check()   { printf "  %s %s\n" "$(green '✓')" "$(dim "$*")"; }
arrow()   { printf "  %s %s\n" "$(cyan '→')" "$*"; }
errmsg()  { printf "  %s %s\n" "$(red '✗')" "$*" >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --no-deps) DEPS=0; shift;;
        --engines) PICK="$2"; shift 2;;
        -y|--yes) ASSUME_YES=1; shift;;
        -h|--help)
            cat <<USAGE
$(bold "install.sh") — set up SA3 TRT inference.

  $(yellow "--no-deps")              Skip dependency install (uv venv + python packages)
  $(yellow "--engines GROUP")        Pre-select engine set: $(magenta "medium | sm-music | sm-sfx | all | skip")
                         (otherwise asks interactively)
  $(yellow "-y, --yes")              Unattended mode: default --engines=all, auto-pick first
                         option in any arch chooser (used by bootstrap.sh)
  $(yellow "-h, --help")             Show this message

$(dim "After install, run with:")
  $(cyan ./.venv/bin/python) sa3_trt.py --prompt "..." --dit sm-music --decoder same-s
$(dim "or (with uv):")
  $(cyan uv run) sa3_trt.py --prompt "..." --dit sm-music --decoder same-s
$(dim "Quantized decoders/encoders (auto-download from HF):") $(yellow "--dec-precision") $(magenta "fp8 | fp8_fast")
  $(dim "fp8 ~1.14x near-transparent · fp8_fast SAME-S only ~1.22x lossier · --precision fp32 for max fidelity")
  $(dim "build tier engines locally with quantize/build_tiers.py; tier ONNX + sm_90/sm_120 engines live on HF")
USAGE
            exit 0;;
        *) errmsg "unknown arg: $1"; exit 2;;
    esac
done

# In unattended mode, default engine set to "all" if not pre-selected.
if [ "$ASSUME_YES" = "1" ] && [ -z "$PICK" ]; then
    PICK="all"
fi

echo
banner "SA3 TRT Installer"

# ─── 1. uv + venv + deps ───────────────────────────────────────────────────
if [ "$DEPS" = "1" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        section "Installing uv (Python package manager)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        if ! command -v uv >/dev/null 2>&1; then
            errmsg "uv install seemingly succeeded but 'uv' not on PATH."
            errmsg "  try: source \$HOME/.local/bin/env  (or restart your shell)"
            exit 1
        fi
    fi
    arrow "uv: $(bold "$(uv --version | awk '{print $2}')")"

    section "Setting up venv"
    arrow "$(dim "$VENV_DIR")"
    if [ ! -d "${VENV_DIR}" ]; then
        uv venv "${VENV_DIR}" 2>&1 | sed "s/^/    /"
    else
        check "existing venv reused"
    fi

    section "Installing Python deps"
    UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        uv pip install -r "${SCRIPT_DIR}/requirements.txt" --python "${VENV_DIR}/bin/python" \
        2>&1 | sed "s/^/    /"
fi

# Verify huggingface_hub is importable in the venv (or system Python if --no-deps)
PY="${VENV_DIR}/bin/python"
if [ ! -x "${PY}" ]; then
    PY="python3"
fi
"${PY}" -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
    errmsg "huggingface_hub not installed in ${PY}. Re-run without --no-deps."
    exit 1
}

# ─── 1.5. Detect GPU architecture(s) + pick matching HF subdir ────────────
section "GPU architecture"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    errmsg "nvidia-smi not found. SA3 TRT inference requires an NVIDIA GPU."
    exit 1
fi

declare -A SEEN_ARCH
LOCAL_ARCHES=()
while IFS=, read -r name cc; do
    name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
    cc="${cc#"${cc%%[![:space:]]*}"}";       cc="${cc%"${cc##*[![:space:]]}"}"
    arch="sm_${cc//./}"
    arrow "$(bold "$name") $(dim "(compute capability $cc → $arch)")"
    if [ -z "${SEEN_ARCH[$arch]:-}" ]; then
        LOCAL_ARCHES+=("$arch")
        SEEN_ARCH[$arch]=1
    fi
done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)

if [ ${#LOCAL_ARCHES[@]} -eq 0 ]; then
    errmsg "No NVIDIA GPUs detected."
    exit 1
fi

# Query HF for available sm_* subdirs under tensorRT/. Uses urllib (stdlib) —
# works whether or not huggingface_hub is installed.
HF_ARCHES_RAW=$("${PY}" - "$REPO_ID" 2>/dev/null <<'PYHF'
import sys, json, urllib.request
repo = sys.argv[1]
url = f"https://huggingface.co/api/models/{repo}/tree/main/tensorRT"
try:
    data = json.load(urllib.request.urlopen(url, timeout=10))
    arches = sorted({d["path"].rsplit("/", 1)[-1] for d in data
                     if d["type"] == "directory" and d["path"].rsplit("/", 1)[-1].startswith("sm_")})
    print(" ".join(arches))
except Exception:
    sys.exit(1)
PYHF
) || {
    arrow "$(yellow "couldn't reach HF — defaulting to sm_90.")"
    HF_ARCHES_RAW="sm_90"
}
read -ra HF_ARCHES <<< "$HF_ARCHES_RAW"
if [ ${#HF_ARCHES[@]} -eq 0 ]; then
    arrow "$(yellow "no sm_* subdirs found on HF (repo may not be updated yet) — defaulting to sm_90.")"
    HF_ARCHES=("sm_90")
fi
arrow "$(dim "available on $REPO_ID:") $(bold "${HF_ARCHES[*]}")"

MATCHES=()
for a in "${LOCAL_ARCHES[@]}"; do
    for b in "${HF_ARCHES[@]}"; do
        [ "$a" = "$b" ] && MATCHES+=("$a")
    done
done

ARCH=""
BUILD_FROM_ONNX=0
case ${#MATCHES[@]} in
    1)
        ARCH="${MATCHES[0]}"
        check "match: $(bold "$ARCH")"
        ;;
    0)
        echo
        arrow "$(yellow "No prebuilt TRT engines on HF for your arch.")"
        arrow "your GPU(s):  $(bold "${LOCAL_ARCHES[*]}")"
        arrow "available:    $(bold "${HF_ARCHES[*]}")"
        echo
        printf "  %s\n" "$(bold "Options:")"
        printf "    %s %s   %s\n" "$(bold "[B]")" "$(green build)"    "$(dim "Compile fresh engines from HF-hosted ONNX — ~30 min, recommended")"
        printf "    %s %s    %s\n" "$(bold "[D]")" "$(yellow download)" "$(dim "Download a non-matching arch anyway (engine may not load)")"
        printf "    %s %s        %s\n" "$(bold "[S]")" "$(dim skip)"   "$(dim "Skip engine setup; do it manually later")"
        echo
        if [ "$ASSUME_YES" = "1" ]; then
            choice="b"
            arrow "$(dim "-y: auto-picking [B] build from ONNX")"
        else
            read -rp "  Choose [$(bold B/D/S)]: " choice
        fi
        case "$choice" in
            b|B)
                BUILD_FROM_ONNX=1
                ARCH="${LOCAL_ARCHES[0]}"
                check "will compile engines for $(bold "$ARCH") from ONNX after deps install"
                ;;
            d|D)
                if [ ${#HF_ARCHES[@]} -gt 1 ]; then
                    echo
                    printf "  %s\n" "$(yellow "Pick an HF arch to download (won't match your GPU; load may fail):")"
                    for i in "${!HF_ARCHES[@]}"; do
                        printf "    %s %s\n" "$(bold "[$((i+1))]")" "${HF_ARCHES[$i]}"
                    done
                    read -rp "  Choose [1-${#HF_ARCHES[@]}]: " idx
                    if ! [[ "$idx" =~ ^[0-9]+$ ]] || [ "$idx" -lt 1 ] || [ "$idx" -gt "${#HF_ARCHES[@]}" ]; then
                        errmsg "invalid choice"; exit 2
                    fi
                    ARCH="${HF_ARCHES[$((idx-1))]}"
                else
                    ARCH="${HF_ARCHES[0]}"
                fi
                arrow "$(yellow "! installing $ARCH on ${LOCAL_ARCHES[*]} — engine may not load.")"
                ;;
            s|S|"")
                arrow "$(dim "skipping engine setup; build later with: cd build && python build_from_onnx.py all")"
                ARCH="${LOCAL_ARCHES[0]}"
                PICK="skip"
                ;;
            *)
                errmsg "invalid choice: $choice"; exit 2
                ;;
        esac
        ;;
    *)
        echo
        arrow "multiple matching arches available"
        printf "  %s\n" "$(bold "Pick one:")"
        for i in "${!MATCHES[@]}"; do
            printf "    %s %s\n" "$(bold "[$((i+1))]")" "${MATCHES[$i]}"
        done
        if [ "$ASSUME_YES" = "1" ]; then
            idx=1
            arrow "$(dim "-y: auto-picking [1] ${MATCHES[0]}")"
        else
            read -rp "  Choose [1-${#MATCHES[@]}]: " idx
        fi
        if ! [[ "$idx" =~ ^[0-9]+$ ]] || [ "$idx" -lt 1 ] || [ "$idx" -gt "${#MATCHES[@]}" ]; then
            errmsg "invalid choice"; exit 2
        fi
        ARCH="${MATCHES[$((idx-1))]}"
        ;;
esac

HF_SUBDIR="tensorRT/${ARCH}"
mkdir -p "${MODELS_DIR}"
echo "$ARCH" > "${MODELS_DIR}/.arch"

# ─── 2. Pick engine sets ───────────────────────────────────────────────────
if [ -z "$PICK" ]; then
    section "Engine set"
    cat <<INTRO
  Download to $(dim "$MODELS_DIR"):

    $(bold "[1] medium")      SA3-medium DiT + SAME-L codec       $(yellow "~6.6 GB")
    $(bold "[2] sm-music")    SA3-small-music DiT + SAME-S codec  $(yellow "~1.7 GB")
    $(bold "[3] sm-sfx")      SA3-small-sfx DiT + SAME-S codec    $(yellow "~1.7 GB")
    $(bold "[a] all")         all three (shared files dedupe)     $(yellow "~9.0 GB")
    $(bold "[s] skip")        sa3_trt.py will fetch on first use

INTRO
    read -rp "  Choose [$(bold "1/2/3/a/s")]: " choice
    case "$choice" in
        1) PICK="medium";;
        2) PICK="sm-music";;
        3) PICK="sm-sfx";;
        a) PICK="all";;
        s|"") PICK="skip";;
        *) errmsg "invalid choice: $choice"; exit 2;;
    esac
fi

# ─── 3a. Build from ONNX (if no prebuilt engines for this arch on HF) ─────
if [ "$BUILD_FROM_ONNX" = "1" ]; then
    section "Compiling TRT engines from ONNX for $(bold "$ARCH")"
    arrow "$(dim "ONNX is pulled from $(magenta "$REPO_ID/onnx/") and compiled in place.")"
    arrow "$(dim "Per-engine build time: ~2-3 min small, ~4 min medium DiT, ~10+ min same-l.")"
    echo

    BUILD_TARGETS=()
    case "$PICK" in
        skip)     ;;  # build nothing
        medium)   BUILD_TARGETS=(t5gemma same-l-encoder same-l-decoder sa3-m);;
        sm-music) BUILD_TARGETS=(t5gemma same-s-encoder same-s-decoder sa3-sm-music);;
        sm-sfx)   BUILD_TARGETS=(t5gemma same-s-encoder same-s-decoder sa3-sm-sfx);;
        all)      BUILD_TARGETS=(all);;
        *) errmsg "internal error: unknown PICK=$PICK"; exit 2;;
    esac

    for t in "${BUILD_TARGETS[@]}"; do
        arrow "build_from_onnx.py $(magenta "$t")"
        (cd "${SCRIPT_DIR}/build" && "${PY}" build_from_onnx.py "$t") || {
            errmsg "build failed for target: $t"
            exit 3
        }
    done

    if [ ${#BUILD_TARGETS[@]} -gt 0 ]; then
        arrow "$(green "✓") engines compiled to $(dim "$MODELS_DIR/$ARCH/")"
        arrow "$(dim "Consider uploading to HF so others on $ARCH can skip the build:")"
        arrow "$(dim "  cp -r $MODELS_DIR/$ARCH <hf-repo>/tensorRT/$ARCH && cd <hf-repo> && git add tensorRT/$ARCH && git commit && git push")"
    fi

    # Skip the HF download path below — we built locally.
    FILES=()
    SKIP_DOWNLOAD=1
fi

# ─── 3b. Download via huggingface_hub ──────────────────────────────────────

SHARED=(
    # T5Gemma engine — arch-specific. (tokenizer.json is arch-agnostic and
    # ships bundled with the repo at scripts/tokenizer.json — no download.)
    "${HF_SUBDIR}/t5gemma/t5gemma_fp16mixed.trt"
)
# Only TRT engines are downloaded from HF.
MEDIUM=(
    "${HF_SUBDIR}/sa3-m/dit_fp16mixed.trt"
    "${HF_SUBDIR}/same-l/enc_dynamic_triton_swa.trt"
    "${HF_SUBDIR}/same-l/dec_dynamic_triton_swa.trt"
)
SM_MUSIC=(
    "${HF_SUBDIR}/sa3-sm-music/dit_fp16mixed.trt"
    "${HF_SUBDIR}/same-s/enc_dynamic_bf16.trt"
    "${HF_SUBDIR}/same-s/dec_dynamic_bf16.trt"
)
SM_SFX=(
    "${HF_SUBDIR}/sa3-sm-sfx/dit_fp16mixed.trt"
    "${HF_SUBDIR}/same-s/enc_dynamic_bf16.trt"
    "${HF_SUBDIR}/same-s/dec_dynamic_bf16.trt"
)

FILES=()
if [ "$PICK" != "skip" ] && [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    FILES+=("${SHARED[@]}")
    case "$PICK" in
        medium)   FILES+=("${MEDIUM[@]}");;
        sm-music) FILES+=("${SM_MUSIC[@]}");;
        sm-sfx)   FILES+=("${SM_SFX[@]}");;
        all)      FILES+=("${MEDIUM[@]}" "${SM_MUSIC[@]}" "${SM_SFX[@]}");;
        *) errmsg "internal error: unknown PICK=$PICK"; exit 2;;
    esac
fi

if [ ${#FILES[@]} -eq 0 ]; then
    section "Engine downloads skipped"
    arrow "$(dim "sa3_trt.py will fetch missing files on first run.")"
else
    section "Downloading engines from $(magenta "$REPO_ID")"
    arrow "$(dim "into $MODELS_DIR")"
    echo

    # Dedup the file list
    declare -A SEEN
    DEDUP=()
    for f in "${FILES[@]}"; do
        if [ -z "${SEEN[$f]:-}" ]; then
            DEDUP+=("$f"); SEEN[$f]=1
        fi
    done

    mkdir -p "${MODELS_DIR}"
    NEW=0
    SKIPPED=0
    for hf_path in "${DEDUP[@]}"; do
        # Strip only "tensorRT/" so the arch stays in the local path:
        # tensorRT/sm_90/sa3-m/dit_fp16mixed.trt → models/sm_90/sa3-m/dit_fp16mixed.trt.
        local_rel="${hf_path#tensorRT/}"
        local_path="${MODELS_DIR}/${local_rel}"
        if [ -f "${local_path}" ] && [ -s "${local_path}" ]; then
            check "$local_rel"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        mkdir -p "$(dirname "${local_path}")"
        arrow "$(magenta "$local_rel")"
        "${PY}" - "$REPO_ID" "$hf_path" "$local_path" <<'PYDL'
import sys, shutil
from huggingface_hub import hf_hub_download
repo, hf_path, local_path = sys.argv[1], sys.argv[2], sys.argv[3]
cached = hf_hub_download(repo_id=repo, filename=hf_path)
shutil.copyfile(cached, local_path)
PYDL
        NEW=$((NEW + 1))
    done
    echo
    arrow "$(green "$NEW") downloaded, $(dim "$SKIPPED already present")"
fi

# ─── 4. Done — show examples ───────────────────────────────────────────────
echo
banner "$(green Done)"
arrow "Models root: $(dim "$MODELS_DIR")"

SA3="${SCRIPT_DIR}/sa3"
# In the example output, show "./sa3" — assumes the user `cd`s into the repo first
RUN_PREFIX="$(cyan "./sa3")"

# Pretty-print one example with colored argument parts.
ex() {
    local dit="$1" decoder="$2" prompt="$3" tag="$4" comment="$5"
    printf "    %s\n" "$(dim "# $comment")"
    printf "    %s %s \"%s\" %s\n" \
        "${RUN_PREFIX}" "$(yellow "--prompt")" "$(green "$prompt")" "\\"
    printf "        %s %s %s %s %s %s %s %s\n" \
        "$(yellow "--dit")" "$(magenta "$dit")" \
        "$(yellow "--decoder")" "$(magenta "$decoder")" \
        "$(yellow "--seconds")" "30" \
        "$(yellow "--out")" "$(green "${tag}.wav")"
    printf "\n"
}

section "Example commands"

case "$PICK" in
    medium)
        ex medium same-l "A beautiful piano arpeggio grows into a cinematic climax" piano \
            "text-to-audio (high-quality music)"
        ;;
    sm-music)
        ex sm-music same-s "lofi house loop" lofi \
            "text-to-audio (fast music)"
        ;;
    sm-sfx)
        ex sm-sfx same-s "footsteps on gravel" steps \
            "text-to-audio (fast sfx)"
        ;;
    all|skip)
        ex sm-music same-s "lofi house loop" lofi \
            "1. small music — fast"
        ex sm-sfx same-s "footsteps on gravel" steps \
            "2. small sound effects"
        ex medium same-l "A beautiful piano arpeggio grows into a cinematic climax" piano \
            "3. medium — higher quality, slower"
        ;;
esac

# Cross-cutting examples
printf "    %s\n" "$(dim "# audio-to-audio variation (init audio + lower σmax)")"
printf "    %s %s \"%s\" %s\n" \
    "${RUN_PREFIX}" "$(yellow "--prompt")" "$(green "jazz fusion")" "\\"
printf "        %s %s %s %s %s %s %s %s %s %s\n" \
    "$(yellow "--dit")" "$(magenta "sm-music")" \
    "$(yellow "--decoder")" "$(magenta "same-s")" \
    "$(yellow "--init-audio")" "$(green "lofi.wav")" \
    "$(yellow "--init-noise-level")" "0.7" \
    "$(yellow "--out")" "$(green "lofi_jazz.wav")"
printf "\n"

printf "    %s\n" "$(dim "# inpainting (regenerate seconds 4-7, keep rest)")"
printf "    %s %s \"%s\" %s\n" \
    "${RUN_PREFIX}" "$(yellow "--prompt")" "$(green "explosive drum break")" "\\"
printf "        %s %s %s %s %s %s %s %s %s %s\n" \
    "$(yellow "--dit")" "$(magenta "sm-music")" \
    "$(yellow "--decoder")" "$(magenta "same-s")" \
    "$(yellow "--init-audio")" "$(green "lofi.wav")" \
    "$(yellow "--inpaint-range")" "$(green "\"4,7\"")" \
    "$(yellow "--out")" "$(green "lofi_drums.wav")"
printf "\n"

printf "    %s\n" "$(dim "# CFG with negative prompt")"
printf "    %s %s \"%s\" %s %s %s \"%s\" %s\n" \
    "${RUN_PREFIX}" "$(yellow "--prompt")" "$(green "ambient drone")" \
    "$(yellow "--cfg")" "3.0" \
    "$(yellow "--negative-prompt")" "$(green "drums vocals")" "\\"
printf "        %s %s %s %s %s %s\n" \
    "$(yellow "--dit")" "$(magenta "sm-music")" \
    "$(yellow "--decoder")" "$(magenta "same-s")" \
    "$(yellow "--out")" "$(green "drone.wav")"
printf "\n"

printf "    %s\n" "$(dim "# interactive picker (omit --dit / --decoder)")"
printf "    %s %s \"%s\" %s %s\n\n" \
    "${RUN_PREFIX}" "$(yellow "--prompt")" "$(green "your prompt here")" \
    "$(yellow "--out")" "$(green "out.wav")"

printf "    %s\n" "$(dim "# show all flags")"
printf "    %s %s\n\n" "${RUN_PREFIX}" "$(yellow "--help")"
