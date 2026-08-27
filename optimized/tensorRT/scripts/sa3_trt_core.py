"""SA3 text-to-audio inference via TensorRT (CUDA / H200).

Pipeline:
    prompt → T5Gemma TRT → conditioning (cross_attn + global) → DiT pingpong (8 steps)
           → SAME-S / SAME-L decoder TRT → patched-pretransform unpatch → WAV

Usage:
    python sa3_trt.py --prompt "lofi house loop" --dit sm-music --decoder same-s --seconds 30 --out out.wav

If --dit or --decoder is omitted, the script prompts the user interactively.
"""
from __future__ import annotations
import argparse, math, os, random, sys, termios, time, tty, wave
from pathlib import Path

import numpy as np

# sa3_trt.py lives in scripts/; models/ lives one level up at the repo root.
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

# torch + tensorrt are imported LAZILY in main() (after CLI parsing) so that
# `sa3 --help` doesn't pay the ~5 s of import cost. The silence_fd helper is
# used at plugin-import time to suppress TRT's "experimental tensorrt.plugin
# module" stdout warning (printed from C++, not Python — so we redirect the
# OS-level file descriptor for the duration of the plugin import).
import contextlib, os
@contextlib.contextmanager
def _silence_fd(fileno: int):
    saved = os.dup(fileno)
    try:
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), fileno)
        yield
    finally:
        os.dup2(saved, fileno)
        os.close(saved)


def _import_heavy():
    """Import torch + tensorrt + the SAME-L Triton plugin into module globals.

    Called from main() after CLI parsing. Takes ~5 s cold (torch 3.5 s + trt
    1.8 s + CUDA context init). Plugin must be registered BEFORE any TRT
    engine deserialization.
    """
    global torch, trt
    import torch as _torch
    import tensorrt as _trt
    torch = _torch
    trt = _trt
    with _silence_fd(1), _silence_fd(2):
        import diff_attn_nocast_plugin  # noqa: F401  registers samel::diff_attn_swa

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
IO_CHANNELS = 256
T5_MAX_LEN = 256
COND_DIM = 768

MODELS_DIR = REPO / "models"
OUTPUT_DIR = REPO / "output"

# HuggingFace repo + subdir where engine files live (used for lazy-download).
HF_REPO_ID = "stabilityai/stable-audio-3-optimized"


def _detect_gpu_arch() -> str:
    """Return the sm_XX subdir name for this machine's GPU.

    Preference: an .arch file written by install.sh (so install-time choices —
    e.g., "install sm_90 even though my GPU is sm_100" — persist). Falls back
    to live nvidia-smi detection, then to sm_90 (the only published arch today).
    """
    arch_file = MODELS_DIR / ".arch"
    if arch_file.exists():
        return arch_file.read_text().strip()
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip().splitlines()
        if out:
            return f"sm_{out[0].strip().replace('.', '')}"
    except Exception:
        pass
    return "sm_90"


# Engines live under models/<arch>/<engine_subdir>/<file>.trt so a single
# install can hold multiple architectures side-by-side. The HF repo mirrors
# this: tensorRT/<arch>/<engine_subdir>/<file>.trt.
ARCH = _detect_gpu_arch()
ARCH_DIR = MODELS_DIR / ARCH
HF_SUBDIR = f"tensorRT/{ARCH}"
T5GEMMA_PATH = ARCH_DIR / "t5gemma" / "t5gemma_fp16.trt"

# What engine files each DiT choice needs (relative to MODELS_DIR, mirroring HF repo layout).
# Each DiT engine bundles its conditioner tensors (padding_embedding + seconds_total
# Linear) as graph Constants, so no sidecar weight files are needed.
DIT_ENGINE_FILES = {
    "sm-music": ["sa3-sm-music/dit_fp16.trt"],
    "sm-sfx":   ["sa3-sm-sfx/dit_fp16.trt"],
    "medium":   ["sa3-m/dit_fp16.trt"],   # fp16 is the medium default
}
DECODER_FILES = {
    "same-s": [
        "same-s/dec_bf16_chunkable_limiter.trt",
    ],
    "same-l": [
        "same-l/dec_dynamic_triton_swa.trt",
    ],
}
ENCODER_FILES = {
    "same-s": ["same-s/enc_bf16_chunkable.trt"],
    "same-l": ["same-l/enc_fp16_chunkable.trt"],
}
SHARED_FILES = [
    # T5Gemma engine — downloaded from HF per-arch. (The tokenizer.json is
    # arch-agnostic and ships bundled with the repo at scripts/tokenizer.json,
    # so it's NOT in this list.)
    "t5gemma/t5gemma_fp16.trt",
]

DIT_CHOICES = {
    "sm-music": {"engine": ARCH_DIR / "sa3-sm-music" / "dit_fp16.trt",
                 "default_decoder": "same-s"},
    "sm-sfx":   {"engine": ARCH_DIR / "sa3-sm-sfx" / "dit_fp16.trt",
                 "default_decoder": "same-s"},
    "medium":   {"engine": ARCH_DIR / "sa3-m" / "dit_fp16.trt",  # medium default
                 "default_decoder": "same-l"},
}
# SAME-L now points at the CHUNKABLE engines. Each carries two optimization profiles -- a
# 256-latent (decoder) / 64-latent (encoder) low band for minimum VRAM, and a 4096 wide band for
# single-shot -- plus, on the decoder, the shipped limiter with its ceiling as a runtime input.
# Scratch is committed per profile, so the low band costs 509 MB where the old single-profile
# engine reserved 8143 MB regardless of render length. The previous engines remain on the model
# repo and are reachable through the "legacy" tier below; nothing was moved or deleted, so an
# older checkout still resolves its own filenames.
DECODER_PATHS = {
    "same-s": ARCH_DIR / "same-s" / "dec_bf16_chunkable_limiter.trt",
    "same-l": ARCH_DIR / "same-l" / "dec_fp16_chunkable_limiter.trt",
}
ENCODER_PATHS = {
    "same-s": ARCH_DIR / "same-s" / "enc_bf16_chunkable.trt",
    "same-l": ARCH_DIR / "same-l" / "enc_fp16_chunkable.trt",
}


# ─── Precision-keyed engine maps ─────────────────────────────────────────
#
# Three DiT precisions (fp16mixed was renamed → fp16 — every tier is mixed, so the
# qualifier was noise; the medium-only bf16 tier was retired. Both old tokens alias
# to fp16, see _PRECISION_ALIAS):
#   fp16       — canonical AND the default for every DiT. FP16 trunk with FP32
#                islands around RMSNorm and the RoPE *generation*; the attention
#                core (QK^T → Softmax → P·V) runs FP16 so TRT's FMHA fuser fires
#                (96 fused nodes on medium). Bounding that island is what lets the
#                core fuse — before it, the whole O(L²) core sat in FP32 unfused
#                (which is what made the retired bf16 tier look 4.7× faster). Now:
#                teacher-forced velocity cos 1.0000 vs the FP32 engine at every
#                length, ~3% off the old bf16 speed, and accurate at long sequence
#                (bf16 drifted — it evaluated RoPE's angle too coarsely past ~2048
#                rad, ~4155 rad @L4092 vs bf16 spacing 32 rad, clipping 2–3% of a
#                6-min render). Requires STRONGLY_TYPED (see build_dit_fp16.py).
#   fp8        — all DiTs, fp8 E4M3 on the linear GEMMs (attention + RoPE kept
#                higher precision). On MEDIUM it's a max-speed clean tier (~1.3×
#                over fp16) via the recipe described below. On sm-music /
#                sm-sfx it is a CLEAN WEIGHT-HALVING tier (engine 479 vs 936 MB),
#                only marginally faster (~1.10–1.17×): those DiTs' ~5 ms forward is
#                overhead-bound at batch 1, so fp8's GEMM-math savings barely show.
#                Their fp8 is an fp8-QDQ graft onto the fp16 graph (fp8 linears
#                + fp16 fused attention + the fp16 fp32 islands, STRONGLY_TYPED),
#                velocity-cos ~0.99 vs eager, clip% at/below fp16. medium recipe:
#                MAX-SPEED clean tier. fp8 E4M3 on the 176 linear
#                GEMMs + bf16 fused FMHA (96 nodes) + a BAKED fp32 RoPE constant
#                table: position cos/sin are computed host-side at build time and
#                frozen as a graph Constant (no in-graph trig), so the island is
#                precision-policy-robust and cross-runtime-stable — a constant
#                can't be re-fused to bf16 the way a runtime pin can. ~1.3× faster
#                than fp16 at every length (measured H200, same-run round-
#                robin: 1.40× @L129 / 1.32× @L1292 / 1.30× @L4092) and CLEAN at
#                long sequence (latent std 0.86 vs eager 0.95, 0.000% clip @2min &
#                full) — the baked RoPE dodges bf16's long-angle drift. It is a
#                SPEED tier over an already-good default, NOT a fidelity upgrade:
#                single-step velocity cos vs fp32 is ~0.92–0.97 (below fp16's
#                ~1.0) but the 8-step render stays coherent. Built weakly-typed
#                EXPLICIT_BATCH + BF16 + FP8 + OBEY_PRECISION_CONSTRAINTS (the
#                opposite of fp16's STRONGLY_TYPED — see build_from_onnx.py).
#                The baked table is sized to the profile max L=4096 (= the SAME-L
#                decoder's own cap), so within the shipped range no re-bake is
#                needed; L>4096 is rejected (a re-bake would be required).
#   fp32       — pure-FP32, bit-equivalent to PyTorch eager. ~2× size/latency.
#
# The lookup tables below resolve the engine filename per (dit/decoder,
# precision). Decoders/encoders have no DiT-specific recipe, so they reuse their
# canonical fp16 engine for every precision. Encoders are fp16 only.
DIT_ENGINE_FILENAME = {
    "fp8":  "dit_fp8.trt",   # all DiTs; fp8 linears (medium: +baked RoPE; small: graft on fp16)
    "fp16": "dit_fp16.trt",  # canonical + default for every DiT (fp16 trunk + fp32 RMSNorm/RoPE islands)
    "fp32": "dit_fp32.trt",  # pure fp32, bit-equivalent to eager
}
# DiT precisions actually built per model. fp8 is available for all three (medium
# via baked-RoPE/bf16-attn; sm-music/sm-sfx via an fp8-QDQ graft onto their fp16
# graph — fp8 linears + fp16 fused attn + fp32 islands). The retired medium-only
# bf16 tier (drifted at long sequence) now aliases to fp16 — see _PRECISION_ALIAS.
_DIT_PRECISIONS = {
    "sm-music": ("fp16", "fp8", "fp32"),
    "sm-sfx":   ("fp16", "fp8", "fp32"),
    "medium":   ("fp8", "fp16", "fp32"),
}
# Per-DiT default precision — fp16 everywhere. Medium moved off bf16 once
# fp16's attention core was fused (4.3x faster than the old fp16 engine,
# ~3% off bf16, and fp32-accurate at every sequence length).
DIT_DEFAULT_PRECISION = {"sm-music": "fp16", "sm-sfx": "fp16",
                         "medium": "fp16"}
_DIT_SUBDIR = {"sm-music": "sa3-sm-music", "sm-sfx": "sa3-sm-sfx", "medium": "sa3-m"}
# Both SAME decoders' latent profiles start at L=32 (verified against the engines:
# min=(1,256,32) for same-l and same-s; the DiT's starts at 1). Below 32,
# set_input_shape() REFUSES the shape — it returns False and TRT logs
# "does not satisfy any optimization profiles" — after which get_tensor_shape("pcm")
# resolves to (1, -1, 2). No caller checks that return value, so instead of a clear
# message the failure surfaces downstream as a bogus allocation or silence.
DECODER_MIN_L = 32


DECODER_ENGINE_FILENAME = {
    "same-l": {
        # The DiT precision selects the DECODER only through this map. Both fp8 and fp16 DiTs
        # get the fp16 chunkable decoder; an fp8 DECODER is a separate choice made with
        # --dec-precision fp8 (see DECODER_TIER_FILENAME), because it is a quantisation of the
        # decoder itself rather than a consequence of the DiT's precision.
        "fp8":  "dec_fp16_chunkable_limiter.trt",
        "fp16": "dec_fp16_chunkable_limiter.trt",
        "fp32": "dec_dynamic_fp32.trt",
    },
    "same-s": {
        "fp8":  "dec_dynamic_bf16.trt",
        "fp16": "dec_dynamic_bf16.trt",
        "fp32": "dec_dynamic_fp32.trt",
    },
}
PRECISIONS = ("fp8", "fp16", "fp32")
# Back-compat: retired precision tokens resolve to their replacement.
#   fp16mixed → fp16 : every tier is mixed-precision, so the qualifier was noise.
#   bf16      → fp16 : the medium-only bf16 DiT drifted at long sequence; retired.
_PRECISION_ALIAS = {"fp16mixed": "fp16", "bf16": "fp16"}
def normalize_precision(precision):
    """Map a (possibly retired) precision token to its canonical name."""
    return _PRECISION_ALIAS.get(precision, precision) if precision else precision

# ── Decoder / encoder quantization TIERS (orthogonal to the DiT --precision) ──
# Train-free fp8 tiers grafted onto the bf16 export (see quantize/README.md).
# "canonical" = the shipped default decoder engine. The others are downloaded from HF on demand
# (tensorRT/<arch>/<same-*>/dec_*.trt) — the same wide-profile engines built by quantize/build_tiers.py.
#   fp8       fp8 FFN GEMMs (near-transparent, ~1.14×) — the speed pick
#   fp8_fast  SAME-S only: fp8 on the attention projections too (~1.22×, lossier)
# (int8-weight-only "w8_bf16" was retired: its DequantizeLinear constant-folds to bf16 at build, so
#  the engine was byte-for-byte the size/speed of the bf16 baseline with slightly lossier weights —
#  strictly dominated. For a max-fidelity decoder use --precision fp32.)
DECODER_TIER_FILENAME = {
    # ⚠ SAME-S is BF16 -- SAME-L is the fp16 one. Its chunkable decoder is also the DETERMINISTIC
    # rebuild: the legacy engine has a RandomNormalLike at its bottleneck, so two runs of it differ
    # by ~0.9989 and any A/B against it is floored there.
    "same-s": {"canonical": "dec_bf16_chunkable_limiter.trt",
               "fp8": "dec_fp8.trt", "fp8_fast": "dec_fp8_fast.trt",
               "legacy": "dec_dynamic_bf16.trt"},
    # ⚠ "legacy" is the pre-chunkable engine, kept reachable so renders made before the limiter
    # landed can be reproduced exactly -- the limiter CHANGES the audio, so a same-name swap
    # would silently alter old output. It is still on the model repo; nothing was moved.
    "same-l": {"canonical": "dec_fp16_chunkable_limiter.trt",
               "fp8": "dec_fp8_chunkable_limiter.trt",
               "legacy": "dec_dynamic_triton_swa.trt"},
}
ENCODER_TIER_FILENAME = {
    "same-s": {"canonical": "enc_bf16_chunkable.trt",
               "fp8": "enc_fp8.trt", "fp8_fast": "enc_fp8_fast.trt",
               "legacy": "enc_dynamic_bf16.trt"},
    # ⚠ the old same-l enc_fp8.trt was REMOVED from the model repo: its activation quantisers
    # were calibrated with p99.9-within-clip → p90-across-clips → a 1e-4 floor, putting the clip
    # points at 0.04-1.22 where the decoder's plain amax puts them at 22.4. That cost up to
    # 2.16 dB of round trip on clean acoustic material. enc_fp8_chunkable.trt is the same weights
    # recalibrated with amax (build/recalib_enc_fp8.py) and measures 30.9 dB where the old one
    # measured 18.3.
    "same-l": {"canonical": "enc_fp16_chunkable.trt",
               "fp8": "enc_fp8_chunkable.trt",
               "legacy": "enc_dynamic_triton_swa.trt"},
}
DECODER_TIERS = ("canonical", "fp8", "fp8_fast", "legacy")


def get_decoder_tier_path(decoder_name: str, tier: str = "canonical") -> Path:
    tiers = DECODER_TIER_FILENAME.get(decoder_name)
    if tiers is None:
        raise ValueError(f"unknown decoder={decoder_name!r}; valid: {list(DECODER_TIER_FILENAME)}")
    if tier not in tiers:
        raise ValueError(f"decoder tier {tier!r} not available for {decoder_name}; valid: {list(tiers)}")
    return ARCH_DIR / decoder_name / tiers[tier]


def get_encoder_tier_path(decoder_name: str, tier: str = "canonical") -> Path:
    tiers = ENCODER_TIER_FILENAME[decoder_name]
    return ARCH_DIR / decoder_name / tiers.get(tier, tiers["canonical"])   # enc falls back if a tier is decoder-only


def decoder_tier_from_filename(decoder_name: str, filename: str) -> str:
    """Reverse-map a decoder engine filename (e.g. 'dec_fp8.trt') to its tier name
    ('fp8'); used by the gradio variant picker. Falls back to 'canonical'."""
    for tier, fname in DECODER_TIER_FILENAME.get(decoder_name, {}).items():
        if fname == filename:
            return tier
    return "canonical"


def resolve_decoder_engine(decoder_name: str, precision: str = None, dec_tier: str = "canonical") -> Path:
    """dec_tier 'canonical' → the precision-driven canonical/fp32 decoder (back-compat with
    --precision, e.g. fp32 still yields dec_dynamic_fp32.trt); any other tier → its engine."""
    if dec_tier == "canonical":
        return get_decoder_engine_path(decoder_name, precision)
    return get_decoder_tier_path(decoder_name, dec_tier)


def resolve_encoder_engine(decoder_name: str, dec_tier: str = "canonical") -> Path:
    if dec_tier == "canonical":
        return ENCODER_PATHS[decoder_name]
    return get_encoder_tier_path(decoder_name, dec_tier)


def default_precision(dit_name: str) -> str:
    """Default DiT precision: fp16 for every model.

    (fp16 = FP16 trunk + FP32 RMSNorm/RoPE islands + an FMHA-fused FP16 attention
    core. Medium used the now-retired bf16 tier until 2026-07, when bounding the
    fp16 RoPE island let its attention core fuse — 4.3x faster than before and
    accurate at long sequence, where bf16 drifted.)
    """
    return DIT_DEFAULT_PRECISION.get(dit_name, "fp16")


def get_dit_engine_path(dit_name: str, precision: str = None) -> Path:
    if dit_name not in _DIT_SUBDIR:
        raise ValueError(f"unknown dit={dit_name!r}; valid: {list(_DIT_SUBDIR)}")
    if precision is None:
        precision = default_precision(dit_name)
    precision = normalize_precision(precision)
    if precision not in DIT_ENGINE_FILENAME:
        raise ValueError(f"unknown precision={precision!r}; valid: {PRECISIONS}")
    # fp8 is available for all three DiTs (medium: baked-RoPE + bf16 attn; sm-music/
    # sm-sfx: fp8-QDQ graft on their fp16 graph).
    return ARCH_DIR / _DIT_SUBDIR[dit_name] / DIT_ENGINE_FILENAME[precision]


def get_decoder_engine_path(decoder_name: str, precision: str = None) -> Path:
    if decoder_name not in DECODER_ENGINE_FILENAME:
        raise ValueError(f"unknown decoder={decoder_name!r}; valid: {list(DECODER_ENGINE_FILENAME)}")
    if precision is None:
        precision = "fp16"   # decoders track the canonical (fp16) engine
    precision = normalize_precision(precision)
    if precision not in DECODER_ENGINE_FILENAME[decoder_name]:
        raise ValueError(f"unknown precision={precision!r}; valid: {PRECISIONS}")
    return ARCH_DIR / decoder_name / DECODER_ENGINE_FILENAME[decoder_name][precision]


def get_engine_files(dit_name: str, decoder_name: str, precision: str = None,
                       with_encoder: bool = False, dec_tier: str = "canonical") -> list[str]:
    """Relative paths (under ARCH_DIR) needed for the chosen pipeline. Pass this
    list to _ensure_files() to auto-download anything missing from HF.
    precision=None resolves to the per-model default (fp16); dec_tier picks the
    decoder/encoder quantization tier (canonical / fp8 / fp8_fast)."""
    if precision is None:
        precision = default_precision(dit_name)
    precision = normalize_precision(precision)
    files = list(SHARED_FILES)
    files.append(f"{_DIT_SUBDIR[dit_name]}/{DIT_ENGINE_FILENAME[precision]}")
    if dec_tier == "canonical":   # precision-driven canonical/fp32 decoder (back-compat)
        files.append(f"{decoder_name}/{DECODER_ENGINE_FILENAME[decoder_name][precision]}")
        if with_encoder:
            files.append(f"{decoder_name}/{ENCODER_PATHS[decoder_name].name}")
    else:                          # a quantization tier
        files.append(f"{decoder_name}/{DECODER_TIER_FILENAME[decoder_name][dec_tier]}")
        if with_encoder:
            enc = ENCODER_TIER_FILENAME[decoder_name]
            files.append(f"{decoder_name}/{enc.get(dec_tier, enc['canonical'])}")
    return files

# ─── Display helpers (ANSI color when stdout is a TTY) ───────────────────
_USE_COLOR = sys.stdout.isatty()
_RULE_W = 64

def _c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _USE_COLOR else s
def bold(s):    return _c("1", s)
def dim(s):     return _c("2", s)
def cyan(s):    return _c("36", s)
def yellow(s):  return _c("33", s)
def green(s):   return _c("32", s)
def magenta(s): return _c("35", s)
def red(s):     return _c("31", s)

def rule(char="━", color=cyan):
    print(color(char * _RULE_W))

def banner(title: str):
    rule()
    print(f"  {bold(title)}")
    rule()

def _fmt_mem(b: int) -> str:
    if b >= 1024**3: return f"{b/1024**3:5.2f} GB"
    return f"{b/1024**2:5.0f} MB"


def stage(idx_total: str, label: str, ms: float | None = None):
    """Right-aligned timing column with dotted fill."""
    head = f"  {cyan(idx_total)} {bold(label)}"
    if ms is None:
        print(head); return
    visible = len(f"  {idx_total} {label}")
    fill = max(2, _RULE_W - visible - 9)
    dots = dim("·" * fill)
    print(f"{head} {dots} {yellow(f'{ms:>5.0f} ms')}")

def sub(text: str):
    print(f"        {dim(text)}")


# ─── Lazy download from HuggingFace ──────────────────────────────────────
def _ensure_files(rel_paths: list[str]) -> None:
    """Download any of these (relative to ARCH_DIR) that don't exist locally.

    Files come from {HF_REPO_ID}/{HF_SUBDIR}/<rel_path> (HF_SUBDIR already
    includes the arch) and land at {ARCH_DIR}/<rel_path>.
    """
    missing = [p for p in rel_paths if not (ARCH_DIR / p).exists() or
                                          (ARCH_DIR / p).stat().st_size == 0]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("error: huggingface_hub not installed. "
                 "Re-run install.sh (or pip install huggingface-hub).")
    import shutil
    print(f"  {dim('downloading')} {len(missing)} missing file(s) from {HF_REPO_ID}/{HF_SUBDIR}")
    for rel in missing:
        hf_path = f"{HF_SUBDIR}/{rel}"
        local_path = ARCH_DIR / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    → {rel}")
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_path)
        shutil.copyfile(cached, local_path)


# ─── Process-local VRAM tracking (sees TRT engine memory, not just torch's) ──
_NVML_HANDLE = None
_MY_PID = os.getpid()

def _init_nvml():
    """Best-effort NVML init. Falls back to torch.cuda.* if unavailable."""
    global _NVML_HANDLE
    if _NVML_HANDLE is not None:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        # When CUDA_VISIBLE_DEVICES is set we want the FIRST listed device (CUDA index 0
        # from the process's perspective, which NVML knows under the original UUID).
        # Use UUID lookup for correctness when devices are remapped.
        if torch.cuda.is_available():
            uuid = torch.cuda.get_device_properties(0).uuid
            handle = pynvml.nvmlDeviceGetHandleByUUID(b"GPU-" + str(uuid).encode("ascii"))
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _NVML_HANDLE = handle
    except Exception:
        _NVML_HANDLE = False  # mark "tried, failed"

def _my_vram_bytes() -> int:
    """Bytes of GPU memory used by THIS process (includes TRT engines)."""
    _init_nvml()
    if _NVML_HANDLE is False or _NVML_HANDLE is None:
        return int(torch.cuda.memory_allocated())
    try:
        import pynvml
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_NVML_HANDLE)
        for p in procs:
            if p.pid == _MY_PID:
                return int(p.usedGpuMemory)
        return 0
    except Exception:
        return int(torch.cuda.memory_allocated())

_STAGE_VRAMS: list[tuple[str, int]] = []   # [(stage_label, peak_bytes)]

def _stage_vram(label: str) -> int:
    """Snapshot current VRAM and log it under the given stage label."""
    b = _my_vram_bytes()
    _STAGE_VRAMS.append((label, b))
    return b


# ─── TRT engine wrapper ─────────────────────────────────────────────────
class TRTRunner:
    # _DT is populated lazily on the first __init__ — can't be a class
    # attribute because torch/trt aren't imported until main() runs.
    _DT = None

    def __init__(self, engine_path: Path, logger_level=None, user_managed: bool = False,
                 profile: int = 0):
        """user_managed=True creates the execution context WITHOUT its own scratch allocation.

        TensorRT commits a context's scratch at create_execution_context(), sized from the
        selected profile -- so on a multi-profile engine a DEFAULT context reserves the MAX
        ACROSS profiles and a low-memory band saves nothing. USER_MANAGED lets the caller
        allocate exactly get_device_memory_size_for_profile_v2(profile) and, where several
        engines run sequentially on one stream, share a single buffer between them.

        profile selects the optimization profile (band). The SAME-L engines ship two: 0 is the
        low-ceiling chunked band, 1 is the wide single-shot band.
        """
        if logger_level is None:
            logger_level = trt.Logger.ERROR
        if TRTRunner._DT is None:
            TRTRunner._DT = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF:  torch.float16,
                trt.DataType.BF16:  torch.bfloat16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT64: torch.int64,
                trt.DataType.BOOL:  torch.bool,
                trt.DataType.INT8:  torch.int8,
                trt.DataType.UINT8: torch.uint8,
            }
        runtime = trt.Runtime(trt.Logger(logger_level))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.profile = int(profile or 0)
        if user_managed:
            self.context = self.engine.create_execution_context(
                trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        else:
            self.context = self.engine.create_execution_context()
        if self.profile and self.engine.num_optimization_profiles > 1:
            st = torch.cuda.current_stream()
            self.context.set_optimization_profile_async(self.profile, st.cuda_stream)
            st.synchronize()
        self.stream = torch.cuda.Stream()
        # I/O dtype/name maps
        self.in_dtype = {}
        self.out_dtype = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            dt = self._DT[self.engine.get_tensor_dtype(n)]
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_dtype[n] = dt
            else:
                self.out_dtype[n] = dt

    def free(self):
        """Best-effort engine teardown — frees device memory back to CUDA."""
        del self.context
        del self.engine
        torch.cuda.empty_cache()


def _run_engine(runner: "TRTRunner", ctx, what: str) -> None:
    """Enqueue TRT on the runner's private stream, correctly ordered against the caller.

    Every input buffer these helpers bind is produced on the CURRENT stream — a
    `.to(dtype)` cast, a `.contiguous()` copy, a `.cuda()` transfer, or the caller's
    own compute — while TRT is enqueued on `runner.stream`. CUDA gives no ordering
    between two streams unless you ask for it, so without the `wait_stream()` below
    the engine may start reading those buffers before the producing work has
    finished. That is a data race, and its failure mode here is silent: the engine
    consumes whatever happens to be in memory and returns a plausible-looking
    tensor, so a partially-written latent decodes to subtly (or completely) wrong
    audio with no error anywhere.

    The trailing host-side `synchronize()` is what makes the OUTPUT safe to read.
    It also means the caching allocator cannot recycle either buffer while the
    engine is still using it — both stay referenced by the calling frame until
    after the sync returns — so `record_stream()` is not needed on top.

    The return value of `execute_async_v3` is checked because a failed enqueue
    otherwise leaves the output buffer untouched, which reads downstream as
    digital silence rather than as an error.
    """
    runner.stream.wait_stream(torch.cuda.current_stream())
    if not ctx.execute_async_v3(runner.stream.cuda_stream):
        raise RuntimeError(
            f"{what}: TRT execute_async_v3 failed to enqueue. The output buffer is "
            f"unwritten — treat any audio produced from it as invalid.")
    runner.stream.synchronize()



def _enqueue_captured(ctx, stream, what: str) -> None:
    """execute_async_v3 for a CAPTURE region, with its return value checked.

    _run_engine's wait_stream/synchronize pair cannot be used under stream capture —
    a host-side sync would abort the capture — so this is the stripped-down variant:
    launch on the given stream, and raise if TRT refuses.

    Checking matters more here than anywhere else. Outside capture a failed enqueue
    leaves the output buffer unwritten, which at least reads as silence. Inside
    capture nothing is recorded into the graph at all, no exception is raised, and
    every later replay re-reads whatever that buffer held from the pre-capture
    warmup — so the stage silently vanishes from the pipeline while each call still
    looks like it succeeded. That is how the SAME-L decoder failed on sm_120: its
    engine was not stream-capturable there, so the graph held T5 and all 8 DiT steps
    but no decode, and every render returned the warmup decode of zero latents — a
    constant wash, byte-identical across seeds and prompts, with exit code 0.
    """
    if not ctx.execute_async_v3(stream.cuda_stream):
        raise RuntimeError(
            f"{what}: TRT refused to enqueue. Under CUDA-graph capture this omits the "
            f"stage silently, so any resulting audio is invalid rather than merely "
            f"slow. If this is a SAME-L engine on a new architecture it is most likely "
            f"not stream-capturable — see 'Choosing the SAME-L attention kernel' in "
            f"build/README.md, or rerun with --no-mega-graph.")

# ─── TRT call helpers (one per engine type) ──────────────────────────────
def t5gemma_encode(runner: TRTRunner, tokenizer, prompt: str):
    """Run T5Gemma TRT. Returns (embeds (1, 256, 768), mask (1, 256))."""
    tokens = tokenizer(prompt, return_tensors="pt", max_length=T5_MAX_LEN,
                       padding="max_length", truncation=True)
    ids = tokens["input_ids"].cuda()
    mask = tokens["attention_mask"].cuda()
    ctx = runner.context
    ctx.set_input_shape("input_ids", tuple(ids.shape))
    ctx.set_input_shape("attention_mask", tuple(mask.shape))
    out_shape = tuple(ctx.get_tensor_shape("hidden_states"))
    out = torch.empty(out_shape, dtype=runner.out_dtype["hidden_states"], device="cuda")
    ctx.set_tensor_address("input_ids", ids.data_ptr())
    ctx.set_tensor_address("attention_mask", mask.data_ptr())
    ctx.set_tensor_address("hidden_states", out.data_ptr())
    _run_engine(runner, ctx, "t5gemma_encode")
    return out.float(), mask


def profile_bounds(runner: "TRTRunner", tensor: str) -> tuple[int, int]:
    """(min, max) for `tensor` on the runner's ACTIVE profile, not profile 0.

    With one band per setting, reading profile 0 would report the low band's ceiling even when the
    wide band is selected -- and the windowing decision is derived from that number.
    """
    idx = getattr(runner, "profile", 0) or 0
    lo, _, hi = runner.engine.get_tensor_profile_shape(tensor, idx)
    return int(lo[-1]), int(hi[-1])


def _window_owners(starts, sizes, N):
    """Which window should supply each position: the one holding it FURTHEST from its own edges.

    The fixed half-overlap trim (keep [s+half, s+chunk-half)) assumes every window overlaps its
    neighbour by exactly `overlap`. When the final hop is irregular -- covering N=646 with 50-latent
    windows at hop 42 leaves a last hop of 8 -- the tail window overwrites a wide region using
    near-edge estimates, discarding better central ones from the previous window. That showed up as
    a single latent (index 126 at T_lat=645) with 0.364 relative error against a 0.0158 median.

    Centrality is min(pos, size-1-pos) within a window; the winner per position is unimodal per
    window, so each window ends up owning ONE contiguous run and the writes stay slice-shaped.
    Window STARTS are untouched, so the even-alignment (SAME-S pair grid) still holds.

    Returns [(dst_lo, dst_hi, src_lo, src_hi)] per window, empty runs dropped.
    """
    best = [-1] * N
    owner = [-1] * N
    for i, (st, sz) in enumerate(zip(starts, sizes)):
        for j in range(sz):
            x = st + j
            if x >= N:
                break
            c = j if j < sz - 1 - j else sz - 1 - j
            if c > best[x]:
                best[x] = c
                owner[x] = i
    runs = []
    for i, (st, sz) in enumerate(zip(starts, sizes)):
        idx = [x for x in range(st, min(st + sz, N)) if owner[x] == i]
        if not idx:
            runs.append(None)
            continue
        lo, hi = idx[0], idx[-1] + 1
        assert hi - lo == len(idx), f"window {i} owns a non-contiguous run"
        runs.append((lo, hi, lo - st, hi - st))
    return runs


def encoder_encode(runner: TRTRunner, audio: torch.Tensor) -> torch.Tensor:
    """SAME-S/L encoder, single-shot. audio: (1, 2, T_samples). Returns (1, 256, L).

    WARNING: the canonical SAME-L (Triton SWA) and SAME-S (BF16) encoder engines
    diverge from PyTorch eager at long sequence lengths — SAME-L produces NaN
    past ~T_lat=200 on real music, SAME-S settles at cos ≈ 0.4 vs PT past
    T_lat=108. Engine output is also sensitive to prior shape/state.

    Use `encode_chunked()` instead for any audio with T_lat > ~100. It splits
    the input into safe-length chunks (cos ≥ 0.998 vs PT eager up to 37 s
    tested) and stitches the latents.
    """
    ctx = runner.context
    in_dt = runner.in_dtype["audio"]
    out_dt = runner.out_dtype["latent"]
    a = audio.to(in_dt).contiguous()
    if not ctx.set_input_shape("audio", tuple(a.shape)):
        # Same silent-failure class as the decoder: a refused shape leaves the context on whatever
        # it last held, so the call returns the previous request's latents instead of raising.
        # A capped encoder is the normal case now -- its ceiling is deliberately near
        # encode_chunked's 50-latent window, because single-shot long encodes are ALREADY wrong
        # (see the warning above: SAME-L NaNs past ~200 latents, SAME-S drifts to cos ~0.4 past
        # ~108). So this guard mostly catches callers that should be using encode_chunked().
        idx = getattr(ctx, "active_optimization_profile", 0) or 0
        lo, _, hi = runner.engine.get_tensor_profile_shape("audio", idx)
        raise ValueError(
            f"encoder: audio length {a.shape[-1]} samples "
            f"({a.shape[-1] // SAMPLES_PER_LATENT} latents) is outside the engine profile "
            f"[{lo[-1] // SAMPLES_PER_LATENT}, {hi[-1] // SAMPLES_PER_LATENT}] latents "
            f"(profile {idx}). Use encode_chunked() -- it windows to a safe length, which long "
            f"single-shot encodes require for CORRECTNESS as well as memory.")
    out_shape = tuple(ctx.get_tensor_shape("latent"))
    out = torch.empty(out_shape, dtype=out_dt, device="cuda")
    ctx.set_tensor_address("audio", a.data_ptr())
    ctx.set_tensor_address("latent", out.data_ptr())
    _run_engine(runner, ctx, "encoder_encode")
    return out.float()


# SAME-L and SAME-S encoder engines lose accuracy / produce NaN past short
# audio lengths. PT eager is essentially "local" (cos(PT_full[:N], PT(short_N))
# > 0.999), so we can stitch chunked outputs and match PT to cos > 0.998.
DEFAULT_ENCODER_CHUNK_LAT = 50    # safe length: cos ≥ 0.998 vs PT for both arches
DEFAULT_ENCODER_OVERLAP_LAT = 8   # 4 latents trimmed from each interior edge


def encode_chunked(runner: TRTRunner, audio: torch.Tensor, *,
                    chunk_lat: int = DEFAULT_ENCODER_CHUNK_LAT,
                    overlap_lat: int = DEFAULT_ENCODER_OVERLAP_LAT,
                    warmup_passes: int = 2,
                    balance: bool = False, max_chunk: int = 0) -> torch.Tensor:
    """Chunked SAME-S/L encode. Equivalent to encoder_encode for short audio,
    but reliably accurate at any length.

    Splits `audio` into `chunk_lat`-latent windows that overlap by `overlap_lat`,
    encodes each (all at the same shape — no in-loop shape transitions, which
    the encoders' BF16/FP16-mixed trunks don't always handle cleanly), and
    stitches the resulting latents by keeping the interior of each chunk.

    Args:
      audio:        (1, 2, T_samples), T_samples must be a multiple of 4096.
      chunk_lat:    latents per chunk; default 50 (≈ 1.86 s) — verified to
                    match PT eager at cos ≥ 0.998 for both encoders.
      overlap_lat:  overlap between adjacent chunks; default 8 (4 latents
                    trimmed from each interior edge).
      warmup_passes: zero-audio calls at chunk_lat before the real chunks,
                    to stabilise engine state. Default 2.
      balance:      shrink/grow every window to the smallest uniform width that still covers the
                    request. Total work is T_lat + (n-1)*overlap_lat, so the cheapest plan is the
                    one with the FEWEST windows -- and since every stitch is a small error, it is
                    also the most accurate. Measured on SAME-L: bit-exact wherever the request
                    fits in one call (fixed-50 splits the same request in two and lands 50-59 dB
                    away), and +0.02 to +0.04 dB of round trip beyond that.
      max_chunk:    widest window balancing may use. 0 = read the engine's profile ceiling, which
                    is the right answer and needs no constant.

    Returns: (1, 256, T_lat) where T_lat = T_samples // 4096.
    """
    SAMPLES_PER_LATENT_ = 4096
    if audio.shape[-1] % SAMPLES_PER_LATENT_ != 0:
        raise ValueError(f"audio length {audio.shape[-1]} not divisible by {SAMPLES_PER_LATENT_}")
    T_lat = audio.shape[-1] // SAMPLES_PER_LATENT_
    if T_lat <= chunk_lat:
        # Single shot — still warm the engine at this exact shape first
        warm = torch.zeros_like(audio)
        for _ in range(warmup_passes):
            _ = encoder_encode(runner, warm)
        return encoder_encode(runner, audio)

    if balance:
        # Every interior boundary costs one `overlap_lat` of re-encoded audio, so total work is
        # T_lat + (n-1)*overlap_lat and the cheapest plan is simply the one with the FEWEST
        # windows -- i.e. the widest window allowed. Balancing then shrinks that width back to the
        # smallest value that still covers the request, which changes nothing about the cost but
        # spreads the work evenly and avoids a nearly-empty final window.
        #
        # `max_chunk` is the allowance and defaults to chunk_lat, so by default balancing only
        # REDISTRIBUTES. Raising it to the engine ceiling also cuts the window count -- but
        # chunk_lat=50 was chosen for ACCURACY (cos >= 0.998 vs PT eager), not for the profile, so
        # a wider window is a quality decision and has to be measured, not assumed.
        #
        # Balance BEFORE the pre-warm below: warming at a width the real windows do not use
        # reintroduces exactly the in-loop shape transition the warm-up exists to prevent.
        # Derive the allowance from the ENGINE rather than a constant. The widest window an
        # engine can take is its profile ceiling, and that is exactly the width that minimises
        # window count -- so there is nothing left to hardcode. (The old default, 50 latents, was
        # picked when windows were fixed-width; balancing supersedes it, and for every realistic
        # length the balanced width lands at 58-64 on a 64-ceiling encoder, never 50.)
        if max_chunk:
            cap = max(chunk_lat, int(max_chunk))
        else:
            try:
                cap = max(chunk_lat, profile_bounds(runner, "audio")[1] // SAMPLES_PER_LATENT_)
            except Exception:
                cap = chunk_lat
        T_pad = T_lat + (T_lat & 1)
        for n in range(1, T_pad + 1):
            w = -(-(T_pad + (n - 1) * overlap_lat) // n)
            w += w & 1                      # even width keeps every start even
            if w <= cap:
                chunk_lat = max(w, overlap_lat + 2)
                break

    # Pre-warm at chunk_lat (zero audio, same shape as real chunks)
    warm = torch.zeros(1, 2, chunk_lat * SAMPLES_PER_LATENT_,
                        device=audio.device, dtype=torch.float32)
    for _ in range(warmup_passes):
        _ = encoder_encode(runner, warm)

    # Chunk start positions. EVERY start must be EVEN: the encoder is the same
    # TransformerResamplingBlock as the decoder (stride 16, chunk_size 32, chunk_midpoint_shift,
    # 17 tokens per latent => 34-token = 2-latent transformer chunks), so a window starting on an
    # odd latent has that entire grid shifted by one and encodes measurably worse -- 0.988-0.990
    # against 0.9985-0.9996 for even starts.
    #
    # `step` is even (50-8=42), so interior starts are even; the TAIL is the trap. Anchoring it at
    # T_lat - chunk_lat is odd whenever T_lat is odd, which degraded the final ~46 latents (~4 s)
    # of every odd-length encode to cos 0.9922 vs 0.9998. Rounding the tail start DOWN to even and
    # letting that one window grow to reach the end restores 0.9996.
    # Pad an ODD latent count to even, so every window is exactly chunk_lat and the engine sees
    # ONE shape. Growing the tail to chunk_lat+1 instead (the previous fix for the odd-start parity
    # problem) keeps starts even but introduces an in-loop SHAPE TRANSITION, and that costs +74 ms
    # on the SAME-L encoder (107.0 ms vs 33.1 at T_lat=323) and +2.9 ms on SAME-S. Same failure the
    # docstring's "all chunks same shape => stable engine state" is guarding against. One
    # replicated latent of audio is cheaper than the transition, and the extra latent is sliced off.
    T_req = T_lat
    if T_lat % 2:
        audio = torch.cat([audio, audio[..., -SAMPLES_PER_LATENT_:]], dim=-1)
        T_lat += 1
    step = chunk_lat - overlap_lat
    starts = list(range(0, T_lat - chunk_lat, step))
    last = T_lat - chunk_lat               # even now, since T_lat and chunk_lat both are
    if not starts or starts[-1] != last:
        starts.append(last)
    assert last % 2 == 0, f"odd tail start {last} would break SAME-S pair alignment"
    tail_len = chunk_lat                   # uniform: one shape for every window

    # Run all chunks (all same shape ⇒ stable engine state)
    chunks = []
    for i, s in enumerate(starts):
        sz = tail_len if i == len(starts) - 1 else chunk_lat
        chunk_audio = audio[..., s * SAMPLES_PER_LATENT_ : (s + sz) * SAMPLES_PER_LATENT_].contiguous()
        chunks.append((s, encoder_encode(runner, chunk_audio)))

    # Stitch: take the interior of each chunk (first/last include their outer edge)
    out = torch.zeros(1, 256, T_lat, device=audio.device, dtype=torch.float32)
    runs = _window_owners([s for s, _ in chunks], [chunk_lat]*len(chunks), T_lat)
    for (s, z), run in zip(chunks, runs):
        if run is None:
            continue
        dlo, dhi, slo, shi = run
        out[..., dlo:dhi] = z[..., slo:shi]
    return out[..., :T_req]                # drop the replicated latent


# The limiter ceiling, when the decoder exposes it as a runtime input. TensorRT has NO default
# for a graph input -- an unbound address makes enqueue fail outright ("Address is not set for
# input tensor limiter_ceiling") -- so the default lives here and decoder_decode binds it unless
# the caller overrides. 0.977 = -0.2021 dBFS, the setting decided by the 21,600-cell sweep.
# A value far above 1 makes the gain identically 1.0, bypassing the limiter and letting the
# graph's +-1.0 clamp reproduce the old hard-clip behaviour.
DEFAULT_LIMITER_CEILING = 0.977


DEFAULT_DECODER_CHUNK_LAT = 256   # 0.36 GB of scratch for SAME-S, 0.51 GB for SAME-L
DEFAULT_DECODER_TRIM_LAT = 16     # dropped from each interior edge; must be EVEN


def decode_chunked(runner: TRTRunner, latents: torch.Tensor, *,
                   chunk_lat: int = DEFAULT_DECODER_CHUNK_LAT,
                   trim_lat: int = DEFAULT_DECODER_TRIM_LAT,
                   warmup_passes: int = 2, balance: bool = True,
                   limiter_ceiling: float | None = None) -> torch.Tensor:
    """Chunked SAME-S/L decode, so the engine's PROFILE CEILING can be lowered.

    Decoder scratch is linear in the profile ceiling (SAME-S 1.40 MB per latent of ceiling,
    SAME-L 1.99) and is committed at create_execution_context() -- before any shape is bound.
    A 4096-ceiling decoder therefore reserves 5.72 GB (SAME-S) / 8.14 GB (SAME-L) whether you
    decode 6 minutes or 5 seconds. Decoding in windows lets the engine be built with a ceiling
    of `chunk_lat` instead, which is the only thing that reduces that reservation.

    ⚠ EVERY CHUNK START MUST BE EVEN. The latent->audio path carries a stride-2 structure:
    a window starting at an even latent decodes at the noise floor, and one starting at an ODD
    latent collapses to cos 0.978 against the single-shot decode. The failure is binary (no
    gradation between even and odd) and CANNOT be recovered by overlap or by left warm-up
    context -- 0 through 192 latents of warm-up all measure identically wrong. The natural way
    to hit it is the final chunk: anchoring the tail at N-chunk_lat gives an odd start whenever
    N is odd, so the tail start is rounded UP to even here and the last window is simply shorter.

    Measured against a single-shot decode, chunked output lands on SAME-S's own noise floor
    (~0.9989, set by its RandomNormalLike bottleneck -- i.e. as close as two decodes of the same
    latent get to each other) at every length from 12 s to 6 min and every chunk size 64..2048.
    SAME-L is bit-deterministic and chunks to ~0.9998; its residual is a sliding-window edge
    effect that more trim does not remove (trim 16/32/64 all measure the same).

    Args:
      latents:       (1, 256, N)
      chunk_lat:     the engine's profile ceiling -- the LARGEST window allowed. With
                     balance=True the actual window shrinks to spread the work evenly, which
                     never costs VRAM and avoids decoding up to 1.59x the needed latents.
      trim_lat:      dropped from each interior edge (half per side). Must be even to keep
                     every start even.
      warmup_passes: zero-latent calls at chunk_lat before the real windows, matching
                     encode_chunked -- these engines dislike in-loop shape transitions.

    Returns: same flavor as decoder_decode -- (1, L*4096, 2) int32 pcm, or (1, 2, L*4096) audio.
    """
    if trim_lat % 2:
        raise ValueError(f"trim_lat must be even to keep chunk starts even, got {trim_lat}")
    N_req = latents.shape[-1]
    if N_req <= chunk_lat:
        # Pass the ceiling THROUGH. Dropping it here silently re-enabled the limiter for every
        # request short enough to fit one window, so a caller asking for LIMITER_BYPASS got
        # bypass above the chunk size and 0.977 below it -- which reads as a length-dependent
        # quality cliff in any A/B that spans the boundary.
        return decoder_decode(runner, latents, limiter_ceiling=limiter_ceiling)

    # Pad an ODD length to even before planning. With N odd the balanced plan cannot give every
    # window the same size -- L=1291 yields five windows of 230 and a tail of 229 -- and that single
    # in-loop SHAPE TRANSITION costs ~54 ms on SAME-L (126.8 ms vs 72.2 ms for L=1292, same window
    # count). It is the instability the repo's encode_chunked already guards against by holding one
    # shape and pre-warming. One replicated latent makes every window identical; the extra audio is
    # sliced off at the end. (A single odd-length window is NOT slow on its own -- 169 vs 170
    # latents measures 8.82 vs 8.72 ms -- so this is the transition, not the shape.)
    if N_req % 2:
        latents = torch.cat([latents, latents[..., -1:]], dim=-1)
    N = latents.shape[-1]

    if balance:
        # chunk_lat is the engine's CEILING; the actual window may be smaller. Anchoring a
        # fixed-size tail wastes work whenever N sits just above a multiple of the window:
        # covering N=646 with 512-windows decodes 1024 latents (1.59x the work) because the
        # last window overlaps the first by 378. Spreading the same number of windows evenly
        # costs 1.03x instead -- same chunk count, same VRAM, ~35% less decoding. Chunk count
        # is what the ceiling forces; only the size within it is free.
        n = max(1, -(-(N - trim_lat) // (chunk_lat - trim_lat)))
        chunk_lat = min(chunk_lat, -(-(N + (n - 1) * trim_lat) // n // 2) * 2)
        if chunk_lat <= trim_lat:
            raise ValueError(f"balanced chunk {chunk_lat} <= trim {trim_lat}")

    step = (chunk_lat - trim_lat) // 2 * 2
    starts = list(range(0, N - chunk_lat, step))
    last = -(-(N - chunk_lat) // 2) * 2          # round UP -> stays even, covers the final latent
    if not starts or starts[-1] != last:
        starts.append(last)

    warm = torch.zeros(1, IO_CHANNELS, chunk_lat, device=latents.device, dtype=latents.dtype)
    for _ in range(warmup_passes):
        decoder_decode(runner, warm, limiter_ceiling=limiter_ceiling)

    first = decoder_decode(runner, latents[..., starts[0]:starts[0] + chunk_lat].contiguous(),
                           limiter_ceiling=limiter_ceiling)
    pcm_flavor = (first.dim() == 3 and first.shape[1] >= first.shape[2])   # (1, T, 2) int32
    # Accumulate PCM in int16, not the engine's int32. The graph already clamps to +-32767 before
    # its int32 cast (TRT cannot emit int16), and every caller narrows to int16 immediately, so the
    # full-length int32 buffer is 2x larger than anything needs. It is also the ONLY term here that
    # scales with length: at L=16384 it is 537 MB of a 2818 MiB peak, halved to 268 MB. Narrowing
    # per window keeps the wide dtype confined to one chunk. `.to(torch.int16)` in existing callers
    # stays valid -- it is a no-op on an int16 tensor.
    acc_dtype = torch.int16 if (pcm_flavor and first.dtype == torch.int32) else first.dtype
    out = torch.zeros((1, N * SAMPLES_PER_LATENT, 2) if pcm_flavor
                      else (1, 2, N * SAMPLES_PER_LATENT),
                      dtype=acc_dtype, device=first.device)

    def _place(dst_lo, dst_hi, src, src_lo, src_hi):
        a, b = dst_lo * SAMPLES_PER_LATENT, dst_hi * SAMPLES_PER_LATENT
        c, d = src_lo * SAMPLES_PER_LATENT, src_hi * SAMPLES_PER_LATENT
        if pcm_flavor:
            out[:, a:b, :] = src[:, c:d, :].to(acc_dtype)
        else:
            out[:, :, a:b] = src[:, :, c:d]

    sizes = [(N - st) if i == len(starts) - 1 else chunk_lat for i, st in enumerate(starts)]
    runs = _window_owners(starts, sizes, N)
    for i, s in enumerate(starts):
        end = s + sizes[i]
        piece = (first if i == 0 else
                 decoder_decode(runner, latents[..., s:end].contiguous(),
                                limiter_ceiling=limiter_ceiling))
        run = runs[i]
        if run is None:
            continue
        dlo, dhi, slo, shi = run
        _place(dlo, dhi, piece, slo, shi)
    if N != N_req:                       # drop the replicated latent's audio
        keep = N_req * SAMPLES_PER_LATENT
        out = out[:, :keep, :] if pcm_flavor else out[:, :, :keep]
    return out


def decoder_decode(runner: TRTRunner, latents: torch.Tensor, *,
                   limiter_ceiling: float | None = None) -> torch.Tensor:
    """SAME-S/L decoder.

    Two engine flavors are supported (auto-detected by output tensor name):

      - Legacy (output name "audio", fp32/bf16): shape (1, 2, L*4096) audio
        in [-1, 1]. The caller is responsible for clip + scale + cast to
        int16 and transposing to (T, 2) interleaved PCM.
      - PCM-baked (output name "pcm", int32): shape (1, L*4096, 2) PCM
        already clipped + scaled to int16 range and transposed. The caller
        only needs `.to(torch.int16)` to finish the conversion. Saves ~18 ms
        per inference by letting TRT fuse the postprocess tail.

    Both engines accept any L in [32, 4096] (odd or even); SAME-S decoder at
    odd L matches PT eager at cos ≥ 0.99 on in-distribution latents — no
    chunking needed.

    Returns whatever the engine emits (caller branches on .dtype to decide
    what postprocessing — if any — is still needed in Stage 5).
    """
    ctx = runner.context
    in_dt = runner.in_dtype["latent"]
    out_name = "pcm" if "pcm" in runner.out_dtype else "audio"
    out_dt = runner.out_dtype[out_name]
    lat = latents.to(in_dt).contiguous()
    if "limiter_ceiling" in runner.in_dtype:
        # Engines built with the ceiling promoted to an input. Bind the shipped default so these
        # engines are drop-in for callers that know nothing about the parameter; pass
        # limiter_ceiling=LIMITER_BYPASS to disable limiting (the clamp still bounds the output).
        cv = DEFAULT_LIMITER_CEILING if limiter_ceiling is None else float(limiter_ceiling)
        if not hasattr(runner, "_ceil_buf") or float(runner._ceil_buf.item()) != cv:
            runner._ceil_buf = torch.tensor(cv, dtype=torch.float32, device="cuda")
        ctx.set_input_shape("limiter_ceiling", ())
        ctx.set_tensor_address("limiter_ceiling", runner._ceil_buf.data_ptr())
    elif limiter_ceiling is not None:
        raise ValueError("this decoder engine has the limiter ceiling baked in; rebuild with "
                         "build_ship_capped.py --ceiling-input to make it settable")
    if not ctx.set_input_shape("latent", tuple(lat.shape)):
        # An out-of-profile shape is REFUSED but does not raise: the context keeps whatever shape
        # it last held, so the decode returns the previous request's length filled with plausible
        # audio. Measured: asking a 4096-ceiling engine for 5168 latents returned 4,194,304 samples
        # (the prior 1024-latent shape) at absmax 32767, and asking for L=1 returned 995,328
        # samples (the prior window). Both look like healthy audio and are wrong, so this has to be
        # an error rather than a silent truncation.
        idx = getattr(ctx, "active_optimization_profile", 0) or 0
        lo, _, hi = runner.engine.get_tensor_profile_shape("latent", idx)
        raise ValueError(
            f"decoder: latent length {lat.shape[-1]} is outside the engine profile "
            f"[{lo[-1]}, {hi[-1]}] (profile {idx}). Above the ceiling, decode in windows with "
            f"decode_chunked(); below the floor, rebuild the engine with a lower profile minimum "
            f"(build_capped.py --min-l 1) -- padding up to the floor is out-of-distribution and "
            f"measures 0.82-0.97 against a native short decode.")
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    out = torch.empty(out_shape, dtype=out_dt, device="cuda")
    ctx.set_tensor_address("latent", lat.data_ptr())
    ctx.set_tensor_address(out_name, out.data_ptr())
    _run_engine(runner, ctx, "decoder_decode")
    if out.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return out.float()
    return out


class DiTRunner:
    """Persistent DiT runner — sets shapes once per (L), reuses buffers across the 8 steps.

    The DiT engine bundles the conditioner (padding_embedding + seconds_total Linear) as
    graph Constants, so its inputs are raw T5Gemma outputs + the duration scalar rather
    than the precomputed cross_attn_cond/global_cond pair.
    """

    def __init__(self, runner: TRTRunner):
        self.runner = runner
        self._L = None
        self._vel_buf = None
        self._t_buf = torch.empty(1, dtype=torch.float32, device="cuda")
        self._sec_buf = torch.empty(1, dtype=torch.float32, device="cuda")
        # Persistent input buffers for graph-capture mode. Populated by
        # bind_persistent() once; thereafter callers MUST copy_ into these
        # buffers instead of calling step() with fresh tensors (which would
        # invalidate the captured pointers).
        self._x_buf = None
        self._t5_hidden_buf = None
        self._t5_mask_buf = None
        self._local_add_cond_buf = None
        self._persistent_bound = False

    def _setup(self, L: int):
        if L == self._L:
            return
        ctx = self.runner.context
        ctx.set_input_shape("x",              (1, IO_CHANNELS, L))
        ctx.set_input_shape("t",              (1,))
        ctx.set_input_shape("t5_hidden",      (1, 256, COND_DIM))
        ctx.set_input_shape("t5_mask",        (1, 256))
        ctx.set_input_shape("seconds_total",  (1,))
        ctx.set_input_shape("local_add_cond", (1, 257, L))
        out_shape = tuple(ctx.get_tensor_shape("velocity"))
        self._vel_buf = torch.empty(out_shape,
                                     dtype=self.runner.out_dtype["velocity"], device="cuda")
        self._L = L

    def step(self, x, t, t5_hidden, t5_mask, seconds, local_add_cond):
        """Single DiT forward. All FP32 IO. Returns velocity (same shape as x)."""
        L = x.shape[-1]
        self._setup(L)
        ctx = self.runner.context
        # scalar inputs → preallocated 1-element buffers
        self._t_buf[0] = float(t.item() if torch.is_tensor(t) else t)
        self._sec_buf[0] = float(seconds.item() if torch.is_tensor(seconds) else seconds)
        ctx.set_tensor_address("x",              x.float().contiguous().data_ptr())
        ctx.set_tensor_address("t",              self._t_buf.data_ptr())
        ctx.set_tensor_address("t5_hidden",      t5_hidden.float().contiguous().data_ptr())
        ctx.set_tensor_address("t5_mask",        t5_mask.float().contiguous().data_ptr())
        ctx.set_tensor_address("seconds_total",  self._sec_buf.data_ptr())
        ctx.set_tensor_address("local_add_cond", local_add_cond.float().contiguous().data_ptr())
        ctx.set_tensor_address("velocity",       self._vel_buf.data_ptr())
        if not ctx.execute_async_v3(self.runner.stream.cuda_stream):
            raise RuntimeError(
                f"DiT execute_async_v3 failed (L={L}). A failed enqueue otherwise "
                f"returns the previous buffer contents as if it had succeeded.")
        self.runner.stream.synchronize()
        # .clone(), NOT .float(): _vel_buf is already FP32, so .float() is a no-op
        # that hands back the SAME tensor every call. Any caller holding two
        # results at once then sees one aliased buffer — which silently turned CFG
        # into a no-op, because
        #     v_cond   = dit.step(..., embeds,      ...)
        #     v_uncond = dit.step(..., null_embeds, ...)
        # left v_cond pointing at the unconditional velocity, so the guided render
        # followed the NEGATIVE prompt. Returning a copy is the contract callers
        # already assume. (The graph-capture path is unaffected: it reads
        # _out_buf under an explicit capture protocol.)
        return self._vel_buf.clone()

    def bind_persistent(self, L: int):
        """Allocate persistent input buffers + bind tensor addresses once.

        Used by the graph-capture path. After this call:
          - Callers must copy_ into self._x_buf, self._t5_hidden_buf, etc.
            (rather than passing fresh tensors to step_captured)
          - self._t_buf / self._sec_buf are written via copy_ from a 1-element
            tensor on the CAPTURED stream (so the write becomes a graph node)
          - step_captured() launches execute_async_v3 on the captured stream
            without re-binding addresses (graph-safe)
        """
        self._setup(L)
        # Allocate persistent input buffers (fp32 throughout).
        self._x_buf = torch.zeros(1, IO_CHANNELS, L, dtype=torch.float32, device="cuda")
        self._t5_hidden_buf = torch.zeros(1, T5_MAX_LEN, COND_DIM,
                                          dtype=torch.float32, device="cuda")
        self._t5_mask_buf = torch.zeros(1, T5_MAX_LEN, dtype=torch.float32, device="cuda")
        self._local_add_cond_buf = torch.zeros(1, 257, L, dtype=torch.float32, device="cuda")
        ctx = self.runner.context
        ctx.set_tensor_address("x",              self._x_buf.data_ptr())
        ctx.set_tensor_address("t",              self._t_buf.data_ptr())
        ctx.set_tensor_address("t5_hidden",      self._t5_hidden_buf.data_ptr())
        ctx.set_tensor_address("t5_mask",        self._t5_mask_buf.data_ptr())
        ctx.set_tensor_address("seconds_total",  self._sec_buf.data_ptr())
        ctx.set_tensor_address("local_add_cond", self._local_add_cond_buf.data_ptr())
        ctx.set_tensor_address("velocity",       self._vel_buf.data_ptr())
        self._persistent_bound = True

    def step_captured(self, stream):
        """Launch TRT execute_async_v3 on `stream` (no address re-binding).

        Assumes bind_persistent() was called and the persistent buffers already
        hold the desired inputs (including self._t_buf for the current sigma
        and self._sec_buf for the duration). The output lands in self._vel_buf.
        Returns self._vel_buf (NOT a copy — caller must read it before the
        next step_captured overwrites it).
        """
        assert self._persistent_bound, "call bind_persistent() before step_captured()"
        _enqueue_captured(self.runner.context, stream, "dit (captured step)")
        return self._vel_buf


# ─── Pingpong sampler ────────────────────────────────────────────────────
def build_pingpong_schedule(steps, sigma_max=1.0, dist_shift=None, latent_len=None):
    """Match SAT build_schedule for RF/RF-denoiser: linspace(sigma_max, 0, steps+1),
    optionally dist-shifted, with t[0] forced back to sigma_max so the schedule's
    starting point aligns with the init-mix's t."""
    t = torch.linspace(sigma_max, 0.0, steps + 1, device="cuda")
    if dist_shift is not None and latent_len is not None:
        t = dist_shift.shift(t, latent_len)
        t[0] = sigma_max
    return t


def sample_flow_pingpong(model_fn, x, sigmas, seed=None, paste_back=None, on_step=None):
    """Pingpong sampler for rf_denoiser. Matches SAT sample_flow_pingpong."""
    ns = sigmas.shape[0] - 1
    g = torch.Generator(device="cuda")
    if seed is not None:
        g.manual_seed(int(seed))
    for i in range(ns):
        t_curr = sigmas[i]; t_next = sigmas[i + 1]
        t = t_curr.unsqueeze(0).contiguous()
        v = model_fn(x, t)
        denoised = x - t_curr * v
        if i < ns - 1:
            noise = torch.randn(*x.shape, device="cuda", dtype=x.dtype, generator=g)
            x = (1.0 - t_next) * denoised + t_next * noise
        else:
            x = denoised
        if paste_back is not None:
            init_lat, keep_mask = paste_back   # keep_mask: 1 = preserve init, 0 = regenerated
            x = init_lat.to(x.dtype) * keep_mask + x * (1.0 - keep_mask)
        if on_step:
            on_step(i + 1, ns)
    return x


# ─── CUDA-graph captured pingpong sampler ────────────────────────────────
class GraphPingpongSampler:
    """Captures the 8-step DiT pingpong loop in a CUDA Graph.

    Trade-offs:
      - Static T_lat / steps / paste_back-or-not / cfg=1.0 only (no CFG dual-pass).
      - Pre-generated noise: 7 buffers are filled with fresh noise BEFORE each
        replay (the graph reads from them by pointer). This keeps per-inference
        sampling stochastic while avoiding the "captured RNG state replays
        identical noise" trap.
      - dit.step() runs inside the graph via execute_async_v3 (TRT 10 supports
        stream capture). The persistent input buffers are bound ONCE before
        capture; the graph replays the same kernel sequence with the same
        pointers.

    Usage:
        sampler = GraphPingpongSampler(dit, T_lat, steps, paste_back=None)
        sampler.build(sigmas, embeds, mask, seconds, local_add_cond)
        x = sampler.sample(initial_noise, seed=42)   # fresh noise each call
    """

    def __init__(self, dit, L, steps, paste_back=None):
        self.dit = dit
        self.L = L
        self.steps = steps
        self.paste_back = paste_back   # (init_lat, keep_mask) or None
        # Persistent IO buffers (created in build()).
        self._x_buf = None             # current latent state (also dit's x input buffer)
        self._noise_bufs = None        # list[Tensor] of (steps-1) noise tensors, fp32
        self._sigma_curr_bufs = None   # list of 0-d tensors, one per step
        self._sigma_next_bufs = None   # list of 0-d tensors, one per step (last unused)
        self._out_buf = None           # final x copied here (so subsequent replays don't clobber)
        self._init_lat_buf = None
        self._keep_mask_buf = None
        self._graph = None
        self._built = False

    def build(self, sigmas, embeds, mask, seconds, local_add_cond):
        """Allocate buffers, prime TRT, and capture the 8-step loop graph.

        sigmas: 1-D tensor of length steps+1.
        embeds: (1, 256, COND_DIM) fp32  — copied into dit's persistent t5_hidden buf.
        mask:   (1, 256) fp32            — copied into dit's persistent t5_mask buf.
        seconds: python float            — written into dit._sec_buf.
        local_add_cond: (1, 257, L) fp32 — copied into dit's persistent local_add_cond buf.

        After build(), call sample(initial_noise) repeatedly.
        """
        dit = self.dit
        L = self.L
        ns = self.steps

        # 1) Bind DiT's persistent buffers + addresses once (this also runs
        #    set_input_shape inside _setup).
        dit.bind_persistent(L)

        # 2) Copy fixed conditioning into DiT's persistent input buffers.
        #    These don't change across the 8 steps, so they're written ONCE
        #    here (outside the captured graph).
        dit._t5_hidden_buf.copy_(embeds.float().contiguous())
        dit._t5_mask_buf.copy_(mask.float().contiguous())
        dit._sec_buf[0] = float(seconds)
        dit._local_add_cond_buf.copy_(local_add_cond.float().contiguous())

        # 3) Allocate per-step scalar buffers for sigma values. We can't read
        #    `float(sigmas[i])` inside the graph because that's a host-side
        #    operation. Instead, capture the sigmas as their scalar fp32
        #    values directly into persistent 0-D tensors used in the math.
        self._sigma_curr_bufs = [
            torch.tensor(float(sigmas[i]), dtype=torch.float32, device="cuda")
            for i in range(ns)
        ]
        self._sigma_next_bufs = [
            torch.tensor(float(sigmas[i + 1]), dtype=torch.float32, device="cuda")
            for i in range(ns)
        ]

        # 4) Allocate noise buffers (one per renoise step; last step doesn't renoise).
        self._noise_bufs = [
            torch.empty(1, IO_CHANNELS, L, dtype=torch.float32, device="cuda")
            for _ in range(ns - 1)
        ]

        # 5) Allocate output buffer. The graph's last op copies x_buf -> out_buf
        #    so subsequent replays (which overwrite x_buf during step 0) don't
        #    invalidate the previous result.
        self._out_buf = torch.empty(1, IO_CHANNELS, L, dtype=torch.float32, device="cuda")

        # 6) (Optional) inpaint paste-back buffers.
        if self.paste_back is not None:
            init_lat, keep_mask = self.paste_back
            self._init_lat_buf = init_lat.float().contiguous().clone()
            self._keep_mask_buf = keep_mask.float().contiguous().clone()
            # Broadcast keep_mask to the latent shape so paste-back is one add.
            if self._keep_mask_buf.shape != self._init_lat_buf.shape:
                self._keep_mask_buf = self._keep_mask_buf.expand_as(self._init_lat_buf).contiguous()

        # The dit runner has its own torch.cuda.Stream — we capture on it.
        capture_stream = dit.runner.stream

        # 7) Warmup pass on the capture stream so kernel JIT etc. is done
        #    BEFORE we capture. Run a couple iterations of the actual logic.
        with torch.cuda.stream(capture_stream):
            # x_buf starts as zeros (won't matter; sample() overwrites it).
            for _ in range(2):
                dit._x_buf.zero_()
                for i in range(ns):
                    dit._t_buf[0] = float(sigmas[i])
                    _enqueue_captured(dit.runner.context, capture_stream,
                                      "dit (pingpong warmup)")
                    v = dit._vel_buf.float()
                    denoised = dit._x_buf - self._sigma_curr_bufs[i] * v
                    if i < ns - 1:
                        nb = self._noise_bufs[i]
                        nb.normal_()   # fresh noise during warmup
                        new_x = (1.0 - self._sigma_next_bufs[i]) * denoised + self._sigma_next_bufs[i] * nb
                    else:
                        new_x = denoised
                    if self.paste_back is not None:
                        new_x = self._init_lat_buf * self._keep_mask_buf + new_x * (1.0 - self._keep_mask_buf)
                    dit._x_buf.copy_(new_x)
        capture_stream.synchronize()

        # 8) Capture the loop. NOTE: torch.cuda.graph(g) needs torch.cuda.current_stream()
        #    to be the capture stream. We use the dit runner's stream throughout.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            # Write t_buf for step 0 INSIDE the capture so it's a graph node
            # (the engine's t input lives at t_buf.data_ptr() — bound once
            # already via bind_persistent).
            with torch.cuda.graph(self._graph, stream=capture_stream):
                for i in range(ns):
                    # Write the current sigma into the engine's scalar input.
                    # Use a 0-D copy_ so this becomes a graph kernel.
                    dit._t_buf[0] = self._sigma_curr_bufs[i]
                    # Launch TRT.
                    _enqueue_captured(dit.runner.context, capture_stream,
                                      "dit (pingpong graph)")
                    v = dit._vel_buf.float()
                    denoised = dit._x_buf - self._sigma_curr_bufs[i] * v
                    if i < ns - 1:
                        nb = self._noise_bufs[i]
                        new_x = (1.0 - self._sigma_next_bufs[i]) * denoised + self._sigma_next_bufs[i] * nb
                    else:
                        new_x = denoised
                    if self.paste_back is not None:
                        new_x = self._init_lat_buf * self._keep_mask_buf + new_x * (1.0 - self._keep_mask_buf)
                    if i < ns - 1:
                        dit._x_buf.copy_(new_x)
                    else:
                        self._out_buf.copy_(new_x)

        self._built = True

    def sample(self, initial_noise, seed=None):
        """Replay the captured graph with `initial_noise` and fresh randn noise.

        initial_noise: (1, IO_CHANNELS, L) fp32 — copied into dit._x_buf.
        seed: int or None. If provided, the (steps-1) noise tensors are filled
              from a fresh Generator seeded with `seed` (matches baseline
              sample_flow_pingpong behavior for parity).

        Returns the final latent (fp32, shape matching initial_noise).
        """
        assert self._built, "call build() before sample()"
        # All input writes must happen on the SAME stream that the graph
        # replays on, otherwise the replay races the input copies. The graph
        # was captured on dit.runner.stream — use that for all writes here too.
        stream = self.dit.runner.stream
        with torch.cuda.stream(stream):
            # Fill the initial x_buf.
            self.dit._x_buf.copy_(initial_noise.float().contiguous())
            # Fill noise buffers (the graph reads from them by pointer).
            if seed is not None:
                g = torch.Generator(device="cuda")
                g.manual_seed(int(seed))
                for nb in self._noise_bufs:
                    nb.normal_(generator=g)
            else:
                for nb in self._noise_bufs:
                    nb.normal_()
            # Replay the graph (all 8 dit.step + math + noise consumption).
            self._graph.replay()
        # Sync because the rest of the pipeline runs on the default stream
        # and may read self._out_buf.
        stream.synchronize()
        return self._out_buf.clone()


# ─── WAV I/O ─────────────────────────────────────────────────────────────
def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """audio: (channels, T) float32 in [-1, 1]. Writes 16-bit PCM stereo WAV."""
    if not np.isfinite(audio).all():
        n_bad = int((~np.isfinite(audio)).sum())
        raise RuntimeError(f"refusing to write WAV — audio contains {n_bad} non-finite samples (NaN/Inf)")
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).T   # (T, channels) interleaved
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> np.ndarray:
    """Read 16-bit PCM @ 44.1 kHz. Returns (2, T) float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        nch, sw, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"{path}: sample rate {sr} Hz; need {SAMPLE_RATE}. "
                f"Resample: ffmpeg -i {path} -ar {SAMPLE_RATE} -ac 2 -sample_fmt s16 out.wav"
            )
        if sw != 2:
            raise ValueError(f"{path}: {sw*8}-bit WAV; need 16-bit PCM.")
        raw = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float32) / 32767.0
    if nch == 1:
        return np.stack([raw, raw], axis=0)
    return raw.reshape(-1, nch).T[:2]


# ─── Arrow-key picker ────────────────────────────────────────────────────
def _arrow_pick(prompt: str, options: list[str], default: str | None = None) -> str:
    """Tiny arrow-key picker — termios, no external deps. Falls back to numeric on non-TTY."""
    if not sys.stdin.isatty():
        print(prompt)
        for i, o in enumerate(options):
            mark = "*" if o == default else " "
            print(f"  {mark} [{i}] {o}")
        s = input(f"Choose [0-{len(options)-1}] (Enter for default): ").strip()
        if s == "":
            return default or options[0]
        if s.isdigit() and 0 <= int(s) < len(options):
            return options[int(s)]
        return s if s in options else (default or options[0])
    idx = options.index(default) if default in options else 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt)
    for _ in options:
        print()
    try:
        tty.setcbreak(fd)
        while True:
            sys.stdout.write(f"\x1b[{len(options)}A")
            for i, o in enumerate(options):
                if i == idx: sys.stdout.write(f"\x1b[2K\x1b[36m▶ {o}\x1b[0m\n")
                else:        sys.stdout.write(f"\x1b[2K  {o}\n")
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A": idx = (idx - 1) % len(options)
                elif seq == "[B": idx = (idx + 1) % len(options)
            elif ch in ("\n", "\r"):
                return options[idx]
            elif ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_user_if_missing(args):
    """Fill in --dit / --decoder / --precision / --seed interactively if missing."""
    if args.dit is None:
        args.dit = _arrow_pick("Choose DiT model:", list(DIT_CHOICES.keys()), default="medium")
        print(f"  → {args.dit}")
    if args.decoder is None:
        suggested = DIT_CHOICES[args.dit]["default_decoder"]
        args.decoder = _arrow_pick("Choose audio decoder:", list(DECODER_PATHS.keys()), default=suggested)
        print(f"  → {args.decoder}")
    # Resolve DiT precision: default fp16 for every model. Retired tokens
    # (fp16mixed, bf16) alias to fp16 with a one-line note.
    if getattr(args, "precision", None) is None:
        args.precision = default_precision(args.dit)
    elif args.precision in _PRECISION_ALIAS:
        _canon = _PRECISION_ALIAS[args.precision]
        print(f"  note: --precision {args.precision} is retired → using {_canon}", file=sys.stderr)
        args.precision = _canon
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    return args


# ─── Helpful argparser ────────────────────────────────────────────────────
class _HelpfulParser(argparse.ArgumentParser):
    """argparse that prints full help (not just usage) when a flag is unknown / invalid."""
    def error(self, message):
        sys.stderr.write(f"\nerror: {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    global MODELS_DIR, ARCH_DIR, T5GEMMA_PATH
    ap = _HelpfulParser(
        description="SA3 text-to-audio (+ audio-to-audio + inpainting) via TensorRT (CUDA/H200)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes\n"
            "  text-to-audio    --prompt P\n"
            "  audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level σ]\n"
            "  inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END\n"
            "  negative CFG     --prompt P --cfg N --negative-prompt P_NEG\n"
            "\nrun `sa3_trt.py --help` for per-flag details."
        ),
    )
    # ── Inputs ──
    ap.add_argument("--prompt", default=None,
                    help="Text prompt. Empty string = unconditional. If omitted, asked interactively via stdin.")
    ap.add_argument("--negative-prompt", default=None,
                    help="Negative prompt for CFG uncond branch. Ignored when --cfg=1.0.")
    ap.add_argument("--init-audio", default=None,
                    help="WAV (44.1 kHz 16-bit) for audio-to-audio (with --init-noise-level) or "
                         "inpainting (with --inpaint-range). Trimmed or zero-padded to --seconds.")
    ap.add_argument("--inpaint-range", default=None,
                    help="Inpaint window 'START,END' (seconds). Requires --init-audio. "
                         "The region is regenerated while outside is paste-backed verbatim.")
    # ── Models ──
    ap.add_argument("--dit", choices=list(DIT_CHOICES.keys()), default=None,
                    help="DiT model. 'sm-music' / 'sm-sfx' = small (faster). 'medium' = sa3-medium (higher quality). "
                         "Interactive picker if omitted.")
    ap.add_argument("--decoder", choices=list(DECODER_PATHS.keys()), default=None,
                    help="Audio decoder. 'same-s' pairs with sm-* (110 MB engine). "
                         "'same-l' pairs with medium (1.2 GB engine). Interactive picker if omitted.")
    ap.add_argument("--dec-precision", choices=list(DECODER_TIERS), default="canonical",
                    help="Decoder/encoder quantization tier (orthogonal to --precision, the DiT): "
                         "canonical (bf16) | fp8 (~1.14x, near-transparent) | fp8_fast (SAME-S only, "
                         "~1.22x). Auto-downloads from HF. (For max-fidelity decode use --precision fp32.)")
    ap.add_argument("--precision", choices=list(PRECISIONS) + list(_PRECISION_ALIAS), default=None,
                    help="DiT engine precision (default 'fp16' for every model). "
                         "'fp16' = FP16 trunk + FP32 RMSNorm/RoPE islands with an FMHA-fused "
                         "FP16 attention core (canonical; fp32-accurate at every length). "
                         "'fp8' = fp8 E4M3 linears (attention + RoPE kept higher precision). On MEDIUM "
                         "it's a max-speed clean tier (bf16 fused FMHA + baked fp32 RoPE, ~1.3× faster, "
                         "clean at long sequence, capped at L<=4096). On sm-music/sm-sfx it's a clean "
                         "weight-halving tier (479 vs 936 MB engine, velocity-cos ~0.99 vs eager) that is "
                         "only marginally faster (~1.1×; their small forward is overhead-bound at batch 1). "
                         "'fp32' = pure FP32, matches PyTorch eager bit-for-bit but ~2× slower "
                         "and ~2× the VRAM. ('fp16mixed' and the retired medium-only 'bf16' are accepted "
                         "as deprecated aliases for 'fp16'.) Engines auto-download from HF if missing.")
    ap.add_argument("--models-dir", default=str(MODELS_DIR),
                    help=f"Directory containing the TRT engines. Default: {MODELS_DIR}")
    # ── Sampling ──
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="Output length. T_lat = ceil(seconds * 44100 / 4096). "
                         "Final WAV trimmed to exact --seconds.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Pingpong sampling steps. 1 = single forward (fastest). 8 = the distilled "
                         "sweet spot for rf_denoiser. >8 is diminishing returns.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed. If omitted, a random seed is chosen and printed at the end.")
    ap.add_argument("--init-noise-level", type=float, default=1.0,
                    help="σmax — schedule starting noise. 1.0 = full regen / pure text-to-audio. "
                         "0.4-0.8 typical for variations. Minimum 0.01 (rf_denoiser undefined at t≈0).")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Classifier-Free Guidance scale. 1.0 = off (default). >1 toward prompt, <1 toward uncond. "
                         "Costs 2× per step when ≠ 1 (sequential cond+uncond passes — TRT engines are static batch=1).")
    ap.add_argument("--apg", type=float, default=1.0,
                    help="APG scale [0..1]. Only relevant when --cfg≠1. 1.0 = full APG (orthogonal projection). "
                         "0.0 = vanilla CFG. rf_denoiser default is 1.0.")
    # ── Memory / runtime ──
    ap.add_argument("--free-models", action=argparse.BooleanOptionalAction, default=False,
                    help="Free each TRT engine's CUDA memory after its last use, to lower peak VRAM. "
                         "Default: off (engines stay resident → faster, since freeing 2.9 GB DiT releases workspace "
                         "and can cost ~100-400 ms). Use --free-models when VRAM is tight.")
    ap.add_argument("--pinned-copy", action=argparse.BooleanOptionalAction, default=True,
                    help="Use a pre-allocated pinned host buffer + non-blocking DMA for the Stage-5 "
                         "GPU→CPU PCM copy. Default: on (saves ~3 ms vs the default `.cpu()` path). "
                         "Use --no-pinned-copy to fall back to the blocking copy (frees ~67 MB of "
                         "page-locked RAM at the cost of slightly slower WAV save).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-stage prints, VRAM/NVML probes, and the sampling progress bar. "
                         "Use to measure pure inference cost without instrumentation overhead. "
                         "Still prints the final 'done   Inference ...' summary line.")
    # ── Output ──
    ap.add_argument("--out", default="out.wav",
                    help=f"Output WAV path. Relative paths are saved under {OUTPUT_DIR}/; "
                         f"absolute paths are used as-is. Always 16-bit PCM stereo @ 44.1 kHz, "
                         f"trimmed to --seconds.")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error(f"--steps must be ≥ 1 (got {args.steps})")

    # --quiet: stub out stage/sub prints, VRAM probes, and the sampling progress
    # bar so we can measure pure inference cost without instrumentation overhead.
    if args.quiet:
        global stage, sub, _stage_vram, _USE_COLOR
        def stage(idx_total, label, ms=None): pass
        def sub(text): pass
        def _stage_vram(label): return 0
        _USE_COLOR = False   # also kills the in-loop sampling progress bar

    # Update engine dir if user overrode (arch stays the same — re-anchor at new root)
    if args.models_dir != str(MODELS_DIR):
        new_root = Path(args.models_dir).resolve()
        new_arch_dir = new_root / ARCH
        T5GEMMA_PATH = new_arch_dir / "t5gemma" / "t5gemma_fp16.trt"
        for kk in DIT_CHOICES:
            DIT_CHOICES[kk]["engine"] = new_arch_dir / DIT_CHOICES[kk]["engine"].relative_to(ARCH_DIR)
        for kk in DECODER_PATHS:
            DECODER_PATHS[kk] = new_arch_dir / DECODER_PATHS[kk].relative_to(ARCH_DIR)
        for kk in ENCODER_PATHS:
            ENCODER_PATHS[kk] = new_arch_dir / ENCODER_PATHS[kk].relative_to(ARCH_DIR)
        MODELS_DIR = new_root
        ARCH_DIR = new_arch_dir

    args = prompt_user_if_missing(args)
    if args.prompt is None:
        args.prompt = input("Prompt: ").strip()

    # ── Compute T_lat ──
    T_lat = max(1, math.ceil(args.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    target_dur = T_lat * SAMPLES_PER_LATENT / SAMPLE_RATE

    # DiT engines support L=1..4096 (profile min=1 since the 2026-05-29 rebuild).
    # The model was trained on L≥256 (~23.8s); outputs below that are
    # out-of-distribution and likely musically garbage but the engine runs.
    DIT_MIN_L = 1
    DIT_MAX_L = 4096
    DIT_TRAINED_MIN_L = 256
    if T_lat < DIT_MIN_L:
        sys.exit(f"error: T_lat={T_lat} (= {args.seconds}s) is below engine minimum ({DIT_MIN_L}).")
    if T_lat > DIT_MAX_L:
        sys.exit(f"error: T_lat={T_lat} (= {args.seconds}s) is above the DiT's trained "
                 f"maximum length ({DIT_MAX_L} ≈ {DIT_MAX_L*SAMPLES_PER_LATENT/SAMPLE_RATE:.1f}s).")
    if T_lat < DECODER_MIN_L:
        sys.exit(f"error: T_lat={T_lat} (= {args.seconds}s) is below the {args.decoder} "
                 f"decoder's profile minimum ({DECODER_MIN_L} ≈ "
                 f"{DECODER_MIN_L*SAMPLES_PER_LATENT/SAMPLE_RATE:.2f}s). The DiT accepts it, "
                 f"but the decoder would refuse the shape and the render would come back "
                 f"empty rather than erroring.")
    if T_lat < DIT_TRAINED_MIN_L and not args.quiet:
        print(f"  warning: T_lat={T_lat} (= {args.seconds}s) is below the DiT's trained "
              f"minimum ({DIT_TRAINED_MIN_L} ≈ {DIT_TRAINED_MIN_L*SAMPLES_PER_LATENT/SAMPLE_RATE:.1f}s). "
              f"Engine runs; output quality is undefined.")

    # ── Inpaint window validation + parameter mapping ──
    inpaint_range = None
    inp_start_sec = inp_end_sec = None
    if args.inpaint_range is not None:
        if args.init_audio is None:
            sys.exit("error: --inpaint-range requires --init-audio (the audio to inpaint into)")
        try:
            s_str, e_str = args.inpaint_range.split(",")
            inp_start_sec = float(s_str.strip()); inp_end_sec = float(e_str.strip())
        except ValueError:
            sys.exit(f"error: --inpaint-range must be 'START,END' in seconds; got {args.inpaint_range!r}")
        if not (0 <= inp_start_sec < inp_end_sec <= args.seconds):
            sys.exit(f"error: invalid inpaint range {inp_start_sec}-{inp_end_sec}s "
                     f"(must satisfy 0 <= start < end <= {args.seconds}s)")
        inp_start_lat = max(0, int(round(inp_start_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inp_end_lat   = min(T_lat, int(round(inp_end_sec   * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_range = (inp_start_lat, inp_end_lat)

    sigma_max = float(args.init_noise_level)
    mode = ("inpaint" if inpaint_range else
            "audio-to-audio" if args.init_audio else "text-to-audio")

    MIN_SIGMA = 0.01
    if sigma_max < MIN_SIGMA:
        sys.exit(
            f"error: --init-noise-level={sigma_max} is too low (σmax min is {MIN_SIGMA}). "
            f"The rf_denoiser model is undefined at t≈0; below {MIN_SIGMA} the schedule "
            f"collapses and the model emits NaN."
        )

    # ── Banner ──
    _STAGE_VRAMS.clear()

    print()
    banner(f"SA3 → TRT  {mode}")
    k = lambda s: dim(f"{s:>10}")
    v = lambda s, w=10: f"{s:<{w}}"
    print(f"  {k('prompt')}  {bold(repr(args.prompt))}")
    if args.negative_prompt:
        suffix = "" if args.cfg != 1.0 else dim("  (ignored: --cfg=1.0, no uncond branch)")
        print(f"  {k('neg prompt')}  {bold(repr(args.negative_prompt))}{suffix}")
    dit_decoder_line = f"  {k('dit')}  {magenta(v(args.dit))}   {k('decoder')}  {magenta(v(args.decoder))}"
    if args.init_audio:
        dit_decoder_line += f"   {k('encoder')}  {magenta(v(args.decoder))}"
    print(dit_decoder_line)
    if args.init_audio:
        print(f"  {k('init audio')}  {bold(args.init_audio)}")
        if inpaint_range:
            s0, s1 = inpaint_range
            print(f"  {k('inpaint')}  {bold(f'{inp_start_sec:.2f}s..{inp_end_sec:.2f}s')} "
                  f"{dim(f'(latent {s0}..{s1} of {T_lat})')}")
    print(f"  {k('σmax')}  {bold(f'{sigma_max:.2f}')}")
    print(f"  {k('seconds')}  {v(f'{args.seconds}s')}   {k('steps')}  {v(args.steps)}   {k('seed')}  {args.seed}")
    cfg_label = f"{args.cfg}"
    if args.cfg != 1.0:
        cfg_label += f" (apg={args.apg})"
    print(f"  {k('cfg')}  {v(cfg_label)}   {k('free models')}  {v('on' if args.free_models else 'off')}")
    print(f"  {k('T_lat')}  {T_lat} {dim(f'({target_dur:.2f}s → trimmed to {args.seconds}s)')}")
    print()

    # ── Lazy-download any missing engine files for the chosen (dit, decoder) combo ──
    needed = get_engine_files(args.dit, args.decoder, args.precision,
                                with_encoder=bool(args.init_audio), dec_tier=args.dec_precision)
    _ensure_files(needed)

    # ── Heavy imports (torch + tensorrt + plugin) ──
    # Deferred from module-load to here so `--help` and CLI errors stay snappy.
    # Costs ~5 s cold (torch 3.5 s + trt 1.8 s + CUDA ctx 1.2 s).
    sub(dim("Loading..."))
    t0 = time.time()
    _import_heavy()
    sub(f"{dim('heavy imports')} {(time.time()-t0)*1000:.0f} ms")

    _init_nvml()  # uses pynvml — needs CUDA context (now alive)

    # ── Load pipeline state (tokenizer + dist_shift) ──
    t0 = time.time()
    import runtime as rt
    rt.MODELS_DIR = str(MODELS_DIR)
    rt.ARCH_DIR = str(ARCH_DIR)
    state = rt.load()
    tokenizer = state["tokenizer"]
    dist_shift = state["dist_shift"]
    sub(f"{dim('tokenizer + dist-shift')} {(time.time()-t0)*1000:.0f} ms")

    # ── Pre-load all TRT engines in parallel (the slow part) ──
    import concurrent.futures
    engine_specs = {
        "t5":  T5GEMMA_PATH,
        "dit": get_dit_engine_path(args.dit, args.precision),
        "dec": resolve_decoder_engine(args.decoder, args.precision, args.dec_precision),
    }
    if args.init_audio:
        engine_specs["enc"] = resolve_encoder_engine(args.decoder, args.dec_precision)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engine_specs)) as ex:
        futs = {name: ex.submit(TRTRunner, path) for name, path in engine_specs.items()}
        runners = {name: fut.result() for name, fut in futs.items()}
    sub(f"{dim(f'engine load (parallel, {len(runners)} engines)')} {(time.time()-t0)*1000:.0f} ms")

    # ── Warmup: prime kernel JIT + caches at the requested L ──
    # 3 passes per engine: first call settles kernel selection / Triton PTX
    # compile, subsequent passes pull the GPU into steady-state boost clocks
    # and prime L1/L2 caches.
    WARMUP_PASSES = 3
    t0 = time.time()
    dit = DiTRunner(runners["dit"])
    _w_x  = torch.zeros(1, IO_CHANNELS, T_lat, device="cuda")
    _w_t  = torch.tensor([0.5], device="cuda")
    _w_h  = torch.zeros(1, T5_MAX_LEN, COND_DIM, device="cuda")
    _w_m  = torch.zeros(1, T5_MAX_LEN, device="cuda")
    _w_l  = torch.zeros(1, 257, T_lat, device="cuda")
    _w_lat   = torch.zeros(1, IO_CHANNELS, T_lat, device="cuda")
    # Encoder warmup is at the chunk_lat shape that encode_chunked actually
    # uses; encoding at the full T_lat shape diverges past T_lat≈100-200 on
    # both encoders (see encoder_encode docstring), so we never feed the
    # engine that shape — encode_chunked stitches chunk_lat=50 windows.
    _w_audio = torch.zeros(1, 2, DEFAULT_ENCODER_CHUNK_LAT * SAMPLES_PER_LATENT,
                            device="cuda") if "enc" in runners else None
    # Optional: pre-allocate a pinned-memory destination buffer for the
    # Stage-5 narrow + DtoH path. With pinned dst + non_blocking=True the DMA
    # goes straight from GPU into RAM without the usual pageable→pinned
    # staging hop. Size to the engine's full audio output
    # (T_lat * SAMPLES_PER_LATENT samples, stereo int16). Disabled via
    # --no-pinned-copy if you want to free up the ~67 MB of page-locked RAM.
    if args.pinned_copy:
        _pinned_pcm = torch.empty((T_lat * SAMPLES_PER_LATENT, 2),
                                   dtype=torch.int16, pin_memory=True)
    else:
        _pinned_pcm = None

    # Also do the cheap setup-between-stages ops once during warmup to JIT
    # their CUDA kernels (torch.linspace + the distribution-shift math + a
    # torch.randn). First call cold-starts at ~20 ms; subsequent are <1 ms.
    _w_sigmas = build_pingpong_schedule(args.steps, sigma_max=sigma_max,
                                         dist_shift=dist_shift, latent_len=T_lat)
    _ = torch.randn(1, IO_CHANNELS, T_lat, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP_PASSES):
        _ = t5gemma_encode(runners["t5"], tokenizer, " ")
        _ = dit.step(_w_x, _w_t, _w_h, _w_m, 30.0, _w_l)
        dec_out = decoder_decode(runners["dec"], _w_lat)
        # Also warm the Stage-5 narrow+DtoH path — first inference used to
        # spend ~10 ms here on cold-launch overhead. Mirror the production
        # path exactly.
        if dec_out.dtype == torch.int32:
            pcm_w = dec_out[0].to(torch.int16)
            if _pinned_pcm is not None:
                _pinned_pcm.copy_(pcm_w, non_blocking=True)
            else:
                _ = pcm_w.cpu().numpy()
        else:
            _ = dec_out.cpu().numpy()
        if _w_audio is not None:
            _ = encoder_encode(runners["enc"], _w_audio)

    # One full sampling-loop pass: primes sample_flow_pingpong's per-step body
    # (t_curr.unsqueeze, the in-loop torch.randn(generator=g) variant, the
    # denoise/renoise arithmetic). DiT engine itself is already warm from the
    # passes above; cost here is ~args.steps × dit_step_ms of overhead-warming.
    _ = sample_flow_pingpong(
        lambda x, t: dit.step(x, t, _w_h, _w_m, 30.0, _w_l),
        _w_x, _w_sigmas, seed=0, paste_back=None, on_step=None,
    )
    torch.cuda.synchronize()

    # ── Build the CUDA-graph captured sampler at warmup time (CFG=1.0 only) ──
    # The graph captures persistent-pointer kernel launches; we update the
    # contents of those buffers (embeds/mask/local_add_cond/initial_noise/
    # noise_bufs) before each inference's g.replay().
    #
    # Inpaint mode and CFG≠1.0 fall back to the eager Python sampler.
    graph_sampler = None
    if args.cfg == 1.0 and args.inpaint_range is None:
        # Use the warmup sigmas (will be overwritten if inference uses a
        # different schedule — but in practice --steps / σmax / dist_shift /
        # T_lat are CLI-fixed for a process lifetime).
        graph_sampler = GraphPingpongSampler(dit, T_lat, args.steps, paste_back=None)
        # Build with zero placeholders for embeds/mask/local_add_cond — actual
        # values get copy_()'d in at inference time.
        graph_sampler.build(_w_sigmas, _w_h, _w_m, args.seconds, _w_l)
        torch.cuda.synchronize()

    sub(f"{dim(f'warmup ({WARMUP_PASSES} passes)')} {(time.time()-t0)*1000:.0f} ms")
    print()

    # ── Inference wall clock starts NOW (post-load, post-warmup) ──
    t_wall_start = time.time()

    # ── 1. T5Gemma encode ──
    t0 = time.time()
    embeds, mask = t5gemma_encode(runners["t5"], tokenizer, args.prompt)
    null_embeds = None
    null_mask = None
    if args.cfg != 1.0:
        neg_prompt = args.negative_prompt if args.negative_prompt is not None else ""
        null_embeds, null_mask = t5gemma_encode(runners["t5"], tokenizer, neg_prompt)
    if args.free_models:
        runners["t5"].free(); del runners["t5"]
    stage("[1/5]", "T5Gemma encode", (time.time() - t0) * 1000)
    sub(f"embeds {tuple(embeds.shape)} {embeds.dtype}  (padding sub deferred to DiT engine)")
    _stage_vram("T5Gemma encode")

    # ── 2. Conditioning ──
    stage("[2/5]", "Conditioning", 0.0)
    sub(f"t5 {tuple(embeds.shape)} mask {tuple(mask.shape)} sec={args.seconds:.2f}s"
        + (f"   neg ready ({'prompt' if args.negative_prompt else 'zeros'})"
           if null_embeds is not None else ""))
    _stage_vram("Conditioning")

    # ── 3a. (optional) Encode init_audio → latents ──
    init_latents = None
    if args.init_audio:
        stage("[3a]", "Encoding init audio → latents")
        t0 = time.time()
        audio_np = read_wav(args.init_audio)
        target_samples = T_lat * SAMPLES_PER_LATENT
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
            init_action = f"trimmed to {target_samples} samples"
        else:
            pad = target_samples - audio_np.shape[-1]
            audio_np = np.pad(audio_np, ((0, 0), (0, pad)), mode="constant")
            init_action = f"zero-padded by {pad} samples"
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).cuda()   # (1, 2, T)
        sub(f"read+prep ({init_action})  {(time.time() - t0) * 1000:.0f} ms")
        t0 = time.time()
        init_latents = encode_chunked(runners["enc"], audio_t)
        sub(f"encode  {(time.time() - t0) * 1000:.0f} ms   latents {tuple(init_latents.shape)}")
        if args.free_models:
            runners["enc"].free(); del runners["enc"]
        _stage_vram("Init audio encode")

    # ── 3b. DiT pingpong sampling ──
    sigmas = build_pingpong_schedule(args.steps, sigma_max=sigma_max,
                                      dist_shift=dist_shift, latent_len=T_lat)
    sched_str = " · ".join(f"{float(x):.3f}" for x in sigmas)

    # Initial noise / latent
    g = torch.Generator(device="cuda"); g.manual_seed(int(args.seed))
    pure_noise = torch.randn(1, IO_CHANNELS, T_lat, device="cuda", dtype=torch.float32, generator=g)
    if init_latents is not None and inpaint_range is None:
        # Linear init mix (RF/RF-denoiser convention)
        noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
        sub(f"init: latent * {1-sigma_max:.2f} + noise * {sigma_max:.2f}")
    else:
        noise = pure_noise

    # Build local_add_cond (always 1, 257, T_lat — inpaint vs zero)
    if inpaint_range is not None:
        s0, s1 = inpaint_range
        # keep_mask: 1 = preserve init region, 0 = regenerate
        keep_mask = torch.ones((1, 1, T_lat), device="cuda", dtype=torch.float32)
        keep_mask[:, :, s0:s1] = 0.0
        masked_input = init_latents * keep_mask
        local_add_cond = torch.cat([keep_mask, masked_input], dim=1).contiguous()  # (1, 257, T_lat)
        paste_back = (init_latents, keep_mask)
        sub(f"local_add_cond {tuple(local_add_cond.shape)}  inpaint mask: {s0}..{s1} of {T_lat} "
            f"({(s1-s0)/T_lat*100:.0f}% regenerated)")
    else:
        local_add_cond = torch.zeros((1, 257, T_lat), device="cuda", dtype=torch.float32)
        paste_back = None

    # Unconditional pass: use null_embeds + null_mask if a neg prompt was provided,
    # otherwise an all-zero T5 hidden + all-zero mask (every position becomes the
    # learned padding_embedding inside the engine — same effect as the old "zero
    # cross_attn" path, but expressed at the input to the bundled conditioner).
    if args.cfg != 1.0 and null_embeds is None:
        null_embeds = torch.zeros_like(embeds)
        null_mask = torch.zeros_like(mask)

    def model_fn(x, t):
        if args.cfg == 1.0:
            return dit.step(x, t, embeds, mask, args.seconds, local_add_cond)
        # Sequential dual-pass CFG (engine is static batch=1)
        v_cond   = dit.step(x, t, embeds,      mask,      args.seconds, local_add_cond)
        v_uncond = dit.step(x, t, null_embeds, null_mask, args.seconds, local_add_cond)
        sigma = t.reshape(-1, 1, 1).float()
        cond_d   = x.float() - v_cond.float()   * sigma
        uncond_d = x.float() - v_uncond.float() * sigma
        diff = cond_d - uncond_d
        if args.apg <= 0.0:
            cfg_diff = diff
        else:
            norm = torch.sqrt((cond_d * cond_d).sum(dim=(-2, -1), keepdim=True))
            unit = cond_d / torch.clamp(norm, min=1e-8)
            parallel = (diff * unit).sum(dim=(-2, -1), keepdim=True) * unit
            diff_orth = diff - parallel
            cfg_diff = diff_orth if args.apg >= 1.0 else (args.apg * diff_orth + (1.0 - args.apg) * diff)
        cfg_d = cond_d + (args.cfg - 1.0) * cfg_diff
        cfg_v = (x.float() - cfg_d) / sigma
        return cfg_v.to(x.dtype)

    t_step_prev = [time.time()]
    def _on_step(i: int, total: int):
        if not _USE_COLOR:
            return
        now = time.time(); elapsed = (now - t_step_prev[0]) * 1000; t_step_prev[0] = now
        bar_w = 20
        filled = int(round(bar_w * i / total))
        bar = cyan("█" * filled) + dim("·" * (bar_w - filled))
        sys.stdout.write(f"\r\x1b[K        {dim('sampling')} {bar} "
                         f"{bold(f'step {i}/{total}')}  {yellow(f'{elapsed:.0f} ms')}")
        sys.stdout.flush()

    t0 = time.time()
    if graph_sampler is not None:
        # Graph-captured path: copy fresh conditioning into the persistent
        # buffers that the captured graph reads, then replay. The graph
        # captures all 8 dit.step calls + the pingpong math + noise reads.
        # All copies must run on the same stream the graph captured on, so the
        # replay sees the new data (otherwise it races the default stream).
        _stream = dit.runner.stream
        # Make the runner stream wait for any pending work on the default
        # stream (the upstream embed tensors were produced there).
        _stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_stream):
            dit._t5_hidden_buf.copy_(embeds.float().contiguous())
            dit._t5_mask_buf.copy_(mask.float().contiguous())
            dit._sec_buf[0] = float(args.seconds)
            dit._local_add_cond_buf.copy_(local_add_cond.float().contiguous())
        latents = graph_sampler.sample(noise, seed=int(args.seed) + 1)
        # The captured graph doesn't fire on_step (no host callbacks possible).
    else:
        latents = sample_flow_pingpong(model_fn, noise, sigmas,
                                        seed=int(args.seed) + 1, paste_back=paste_back, on_step=_on_step)
    # Include free() in stage timing — for big engines (medium DiT is 2.9 GB)
    # the CUDA memory release isn't free and would otherwise show up as a
    # "missing" 100-400 ms in the Inference total.
    if args.free_models:
        runners["dit"].free(); del dit, runners["dit"]
    sample_ms = (time.time() - t0) * 1000
    if _USE_COLOR:
        sys.stdout.write("\r\x1b[K")
    stage("[3/5]", f"DiT sample ({args.steps} steps, σmax={sigma_max:.2f})", sample_ms)
    sub(f"schedule  {sched_str}")
    _stage_vram("DiT sample")

    # ── 4. Decoder ──
    t0 = time.time()
    audio = decoder_decode(runners["dec"], latents)
    if args.free_models:
        runners["dec"].free(); del runners["dec"]
    decode_ms = (time.time() - t0) * 1000
    _pcm_baked = audio.dtype == torch.int32
    stage("[4/5]", f"Decoder ({args.decoder})", decode_ms)
    sub(f"audio {tuple(audio.shape)} {audio.dtype}"
        f"{'  (pcm baked-in)' if _pcm_baked else ''}")
    _stage_vram("Decode")

    # ── End of inference wall clock (WAV save excluded — that's I/O) ──
    t_inference = time.time() - t_wall_start

    # ── 5. Trim + save WAV ──
    # Resolve --out: relative paths land in OUTPUT_DIR; absolute paths used as-is.
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_display = out_path.relative_to(REPO).as_posix()
    except ValueError:
        out_display = str(out_path)

    t0_total = time.time()

    # PCM conversion. Two paths:
    #   - PCM-baked engines (output "pcm", int32 (1, T_full, 2)): clip + scale
    #     + transpose are already done inside the decoder graph. The only
    #     remaining work is narrow int32→int16 and trim → done as part of
    #     the GPU→CPU copy (one fast kernel + DtoH).
    #   - Legacy engines (output "audio", fp32 (1, 2, T_full)): we still do
    #     the old clip + scale + cast + transpose on the GPU before the copy.
    requested_samples = int(round(args.seconds * SAMPLE_RATE))
    t0 = time.time()
    if _pcm_baked:
        # (1, T_full, 2) int32 → (T, 2) int16 on GPU
        pcm_gpu = audio[0]                                      # (T_full, 2) int32
        if pcm_gpu.shape[0] > requested_samples:
            pcm_gpu = pcm_gpu[:requested_samples]
        # Engine output isn't clipped — values > ±32767 wrap when cast to int16
        # (audible clicks). Clamp first.
        pcm_gpu = pcm_gpu.clamp(-32767, 32767).to(torch.int16)
        n = pcm_gpu.shape[0]
        if _pinned_pcm is not None:
            # Non-blocking DMA straight into the pre-allocated pinned host
            # buffer, then sync — avoids the implicit staging-buffer hop of
            # `.cpu()` and lands ~2-3 ms below the blocking path.
            _pinned_pcm[:n].copy_(pcm_gpu, non_blocking=True)
            torch.cuda.synchronize()
            pcm = _pinned_pcm[:n].numpy()
        else:
            pcm = pcm_gpu.contiguous().cpu().numpy()             # blocking fallback
    else:
        # legacy fp32 (1, 2, T_full): clip + scale + cast + transpose on GPU
        audio_gpu = audio[0]                                    # (2, T_full) fp32
        if audio_gpu.shape[-1] > requested_samples:
            audio_gpu = audio_gpu[..., :requested_samples]
        pcm_gpu = (audio_gpu.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).T.contiguous()  # (T, 2)
        pcm = pcm_gpu.cpu().numpy()
    t_gpu2cpu = (time.time() - t0) * 1000

    t0 = time.time()
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    t_disk = (time.time() - t0) * 1000

    stage("[5/5]", f"WAV → {out_display}", (time.time() - t0_total) * 1000)
    if _pcm_baked:
        sub(f"cast int32→int16 + GPU→CPU {t_gpu2cpu:.0f} ms  ·  disk write {t_disk:.0f} ms")
    else:
        sub(f"clip/cast/transpose + GPU→CPU {t_gpu2cpu:.0f} ms  ·  disk write {t_disk:.0f} ms")

    t_full = time.time() - t_wall_start
    # ── Per-stage VRAM table ──
    audio_dur = pcm.shape[0] / SAMPLE_RATE   # pcm is (T, 2) int16
    peak_vram = max((b for _, b in _STAGE_VRAMS), default=0)

    def _fmt_t(s: float) -> str:
        return f"{s*1000:.0f} ms" if s < 1 else f"{s:.2f} s"

    print()
    print(f"  {bold('VRAM by stage (process-local)')}")
    if _STAGE_VRAMS:
        name_w = max(len(n) for n, _ in _STAGE_VRAMS)
        peak_b = max(b for _, b in _STAGE_VRAMS)
        for name, b in _STAGE_VRAMS:
            bar_units = int(round(b / max(peak_b, 1) * 24))
            bar = cyan("█" * bar_units) + dim("·" * (24 - bar_units))
            mark = bold(" ←  peak") if b == peak_b else ""
            print(f"    {dim(name.ljust(name_w))}  {bar}  {_fmt_mem(b)}{mark}")

    print()
    rule()
    # Two-column layout: each timing with its own realtime ratio below it,
    # aligned by visible (ANSI-stripped) string width.
    import re
    _ANSI = re.compile(r'\x1b\[[0-9;]*m')
    def _vlen(s): return len(_ANSI.sub('', s))
    col1     = f"Inference {bold(_fmt_t(t_inference))}"
    col2     = f"Inference + Saving the WAV {bold(_fmt_t(t_full))}"
    col1_rt  = dim(f"{audio_dur/t_inference:.0f}× realtime")
    col2_rt  = dim(f"{audio_dur/t_full:.0f}× realtime")
    gap      = max(_vlen(col1), _vlen(col1_rt)) + 4
    indent   = "  " + " " * len("done   ")
    print(f"  {bold(green('done'))}   {col1}{' ' * (gap - _vlen(col1))}{col2}")
    print(f"{indent}{col1_rt}{' ' * (gap - _vlen(col1_rt))}{col2_rt}")
    rule()
    print(f"  {bold(green('▸ saved'))}  {bold(cyan(out_display))}   "
          f"{dim(f'({out_path.resolve()})')}")


if __name__ == "__main__":
    main()
