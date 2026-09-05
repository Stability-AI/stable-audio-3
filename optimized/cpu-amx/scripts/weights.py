"""Pull the cpu-amx engines (.so + weight blobs) from HF on first use.

The big binaries live on HF at stabilityai/stable-audio-3-optimized/cpu-amx/ (flat), NOT in git.
`ensure(group)` downloads an engine's files and places them where its ctypes shim expects them, so
`backends.py` can load unchanged. Idempotent: skips files already present with the right size.

Engine home
-----------
Everything is placed under ``$SA3_CPUAMX_HOME`` (default ``~/.cache/stable-audio-3/cpu-amx``),
one directory per engine. ``backends.py`` resolves the same base, so the two never disagree.

Portability caveat
------------------
All engines resolve their kernel/weight paths from ``$SA3_CPUAMX_HOME`` at load time (the C++
reads the same variable), so the tree is relocatable. NOTE: binaries published before this
change have the old absolute prefix baked in — rebuild from ``build/`` to get a portable one.
"""
from __future__ import annotations
import os, shutil, tarfile

HF_REPO = "stabilityai/stable-audio-3-optimized"
HF_DIR  = "cpu-amx"
# Single source of truth for where engines live; backends.py imports this.
HOME = os.environ.get("SA3_CPUAMX_HOME",
                      os.path.expanduser("~/.cache/stable-audio-3/cpu-amx"))
_C = HOME

# HF flat filename  ->  local destination the loader reads
FILES = {
 "t5gemma_bf16.so":                          f"{_C}/t5gemma_cpu_amx/t5gemma_cpu_amx.so",
 "t5gemma_bf16_weights.bin":                 f"{_C}/t5gemma_cpu_amx/weights.bin",
 "t5gemma_bf16_weights_manifest.txt":        f"{_C}/t5gemma_cpu_amx/weights_manifest.txt",
 "same_s_decoder_bf16.so":              f"{_C}/same_s_cpu_amx/same_s_cpu_amx.so",
 "same_s_decoder_bf16_weights.bin":     f"{_C}/same_s_cpu_amx/weights.bin",
 "same_s_decoder_bf16_weights_manifest.txt": f"{_C}/same_s_cpu_amx/weights_manifest.txt",
 "same_l_decoder_bf16.so":              f"{_C}/same_l_cpu_amx/same_l_cpu_amx.so",
 "same_l_decoder_bf16_weights.bin":     f"{_C}/same_l_cpu_amx/weights.bin",
 "same_l_decoder_bf16_weights_manifest.txt": f"{_C}/same_l_cpu_amx/weights_manifest.txt",
 "same_s_decoder_int8.so":              f"{_C}/same_s_int8fused_cpu_amx/same_s_int8fused_cpu_amx.so",
 "same_s_decoder_int8_weights.bin":     f"{_C}/same_s_int8fused_cpu_amx/weights.bin",
 "same_s_decoder_int8_weights_manifest.txt": f"{_C}/same_s_int8fused_cpu_amx/weights_manifest.txt",
 "same_l_decoder_int8.so":              f"{_C}/same_l_int8fused_cpu_amx/same_l_int8fused_cpu_amx.so",
 "same_l_decoder_int8_weights.bin":     f"{_C}/same_l_int8fused_cpu_amx/weights.bin",
 "same_l_decoder_int8_weights_manifest.txt": f"{_C}/same_l_int8fused_cpu_amx/weights_manifest.txt",
 "same_s_encoder_bf16.so":                   f"{_C}/same_s_encoder_cpu_amx/same_s_encoder_cpu_amx.so",
 "same_s_encoder_bf16_weights.bin":          f"{_C}/same_s_encoder_cpu_amx/weights.bin",
 "same_s_encoder_bf16_weights_manifest.txt": f"{_C}/same_s_encoder_cpu_amx/weights_manifest.txt",
 "same_l_encoder_bf16.so":                   f"{_C}/same_l_encoder_cpu_amx/same_l_encoder_cpu_amx.so",
 "same_l_encoder_bf16_weights.bin":          f"{_C}/same_l_encoder_cpu_amx/weights.bin",
 "same_l_encoder_bf16_weights_manifest.txt": f"{_C}/same_l_encoder_cpu_amx/weights_manifest.txt",
 "same_l_encoder_bf16_weights_f32.bin":      f"{_C}/same_l_encoder_cpu_amx/weights_f32.bin",
 "same_l_encoder_bf16_weights_f32_manifest.txt": f"{_C}/same_l_encoder_cpu_amx/weights_f32_manifest.txt",
 "same_s_encoder_int8.so":                   f"{_C}/same_s_encoder_int8fused_cpu_amx/same_s_encoder_int8fused_cpu_amx.so",
 "same_s_encoder_int8_weights.bin":          f"{_C}/same_s_encoder_int8fused_cpu_amx/weights.bin",
 "same_s_encoder_int8_weights_manifest.txt": f"{_C}/same_s_encoder_int8fused_cpu_amx/weights_manifest.txt",
 "same_l_encoder_int8.so":                   f"{_C}/same_l_encoder_int8fused_cpu_amx/same_l_encoder_int8fused_cpu_amx.so",
 "same_l_encoder_int8_weights.bin":          f"{_C}/same_l_encoder_int8fused_cpu_amx/weights.bin",
 "same_l_encoder_int8_weights_manifest.txt": f"{_C}/same_l_encoder_int8fused_cpu_amx/weights_manifest.txt",
 "dit_medium_int8.so":                       f"{_C}/dit_medium_cpu_amx/dit_cpu_amx.so",
 "dit_medium_int8_core.bin":                 f"{_C}/dit_medium_cpu_amx/core_L320.bin",
 "dit_medium_int8_core_manifest.txt":        f"{_C}/dit_medium_cpu_amx/core_L320_manifest.txt",
 "dit_medium_bf16.so":                       f"{_C}/dit_medium_cpu_amx/dit_cpu_amx_bf16.so",
 "dit_medium_bf16_core.bin":                 f"{_C}/dit_medium_cpu_amx/core_bf16.bin",
 "dit_medium_bf16_core_manifest.txt":        f"{_C}/dit_medium_cpu_amx/core_bf16_manifest.txt",
 "dit_medium_bf16_pin_fp32.npz":             f"{_C}/dit_medium_cpu_amx/pin_fp32.npz",
 "dit_medium_bf16_flash.so":                 f"{_C}/dit_medium_cpu_amx/so_flash/_flash_diff_bf16_bm128.so",
}
# dit_medium_kernels.tar.gz is extracted (not a single dst); handled specially.
_DIT_KERNELS_HF = "dit_medium_int8_kernels.tar.gz"
_DIT_KERNELS_SENTINEL = f"{_C}/dit_medium_cpu_amx/aot_stage2/cpp_kernels.txt"

GROUPS = {
 "t5gemma": ["t5gemma_bf16.so", "t5gemma_bf16_weights.bin", "t5gemma_bf16_weights_manifest.txt"],
 "dit": ["dit_medium_int8.so", "dit_medium_int8_core.bin", "dit_medium_int8_core_manifest.txt", _DIT_KERNELS_HF],
 "dit_bf16": ["dit_medium_bf16.so", "dit_medium_bf16_core.bin", "dit_medium_bf16_core_manifest.txt",
              "dit_medium_bf16_pin_fp32.npz", "dit_medium_bf16_flash.so", _DIT_KERNELS_HF],
 "same_s_decoder_bf16": ["same_s_decoder_bf16.so", "same_s_decoder_bf16_weights.bin", "same_s_decoder_bf16_weights_manifest.txt"],
 "same_l_decoder_bf16": ["same_l_decoder_bf16.so", "same_l_decoder_bf16_weights.bin", "same_l_decoder_bf16_weights_manifest.txt"],
 "same_s_decoder_int8": ["same_s_decoder_int8.so", "same_s_decoder_int8_weights.bin", "same_s_decoder_int8_weights_manifest.txt"],
 "same_l_decoder_int8": ["same_l_decoder_int8.so", "same_l_decoder_int8_weights.bin", "same_l_decoder_int8_weights_manifest.txt"],
 "same_s_encoder_bf16": ["same_s_encoder_bf16.so", "same_s_encoder_bf16_weights.bin", "same_s_encoder_bf16_weights_manifest.txt"],
 "same_l_encoder_bf16": ["same_l_encoder_bf16.so", "same_l_encoder_bf16_weights.bin", "same_l_encoder_bf16_weights_manifest.txt",
                    "same_l_encoder_bf16_weights_f32.bin", "same_l_encoder_bf16_weights_f32_manifest.txt"],
 "same_s_encoder_int8": ["same_s_encoder_int8.so", "same_s_encoder_int8_weights.bin", "same_s_encoder_int8_weights_manifest.txt"],
 "same_l_encoder_int8": ["same_l_encoder_int8.so", "same_l_encoder_int8_weights.bin", "same_l_encoder_int8_weights_manifest.txt"],
}
_DISABLE = os.environ.get("SA3_CPUAMX_NO_HF", "") not in ("", "0", "false")

def _fetch(hf_name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, f"{HF_DIR}/{hf_name}")

def _place(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp); os.replace(tmp, dst)      # atomic; no shell rm

def ensure(group: str):
    """Download + place one engine's files from HF (idempotent). No-op if SA3_CPUAMX_NO_HF is set
    or the files are already present (local build)."""
    if _DISABLE:
        return
    for hf in GROUPS[group]:
        if hf == _DIT_KERNELS_HF:
            if os.path.exists(_DIT_KERNELS_SENTINEL):
                continue
            tarpath = _fetch(hf)
            with tarfile.open(tarpath, "r:gz") as t:
                t.extractall(f"{_C}/dit_medium_cpu_amx")   # recreates aot_stage2/{so,cpp_kernels.txt} + so_flash
            continue
        dst = FILES[hf]
        if os.path.exists(dst):
            continue
        _place(_fetch(hf), dst)

def ensure_all():
    for g in GROUPS:
        ensure(g)
