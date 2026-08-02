"""Pull the cpu-amx engines (.so + weight blobs) from HF on first use.

The big binaries live on HF at stabilityai/stable-audio-3-optimized/cpu-amx/ (flat), NOT in git.
`ensure(group)` downloads an engine's files and places them where its ctypes shim expects them, so
`backends.py` can load unchanged. Idempotent: skips files already present with the right size.

Portability note: the DiT `.so` has baked-in absolute paths (`/weka2/cj/clod/tritoncpu_sa3/...`), so its
files are placed there; the other seven engines load relative to their own dir. Making everything
relocatable is a follow-up (rebuild the DiT `.so` with env-configurable kernel/core paths).
"""
from __future__ import annotations
import os, shutil, tarfile

HF_REPO = "stabilityai/stable-audio-3-optimized"
HF_DIR  = "cpu-amx"
_C = os.environ.get("SA3_CPUAMX_HOME", "/weka2/cj/clod")   # base the shims + DiT .so expect

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
 "dit_medium_int8.so":                       f"{_C}/tritoncpu_sa3/aot_speedprove/dit_cpu_amx.so",
 "dit_medium_int8_core.bin":                 f"{_C}/tritoncpu_sa3/aot_speedprove/core_L320.bin",
 "dit_medium_int8_core_manifest.txt":        f"{_C}/tritoncpu_sa3/aot_speedprove/core_L320_manifest.txt",
}
# dit_medium_kernels.tar.gz is extracted (not a single dst); handled specially.
_DIT_KERNELS_HF = "dit_medium_int8_kernels.tar.gz"
_DIT_KERNELS_SENTINEL = f"{_C}/tritoncpu_sa3/aot_stage2/cpp_kernels.txt"

GROUPS = {
 "t5gemma": ["t5gemma_bf16.so", "t5gemma_bf16_weights.bin", "t5gemma_bf16_weights_manifest.txt"],
 "dit": ["dit_medium_int8.so", "dit_medium_int8_core.bin", "dit_medium_int8_core_manifest.txt", _DIT_KERNELS_HF],
 "same_s_decoder_bf16": ["same_s_decoder_bf16.so", "same_s_decoder_bf16_weights.bin", "same_s_decoder_bf16_weights_manifest.txt"],
 "same_l_decoder_bf16": ["same_l_decoder_bf16.so", "same_l_decoder_bf16_weights.bin", "same_l_decoder_bf16_weights_manifest.txt"],
 "same_s_decoder_int8": ["same_s_decoder_int8.so", "same_s_decoder_int8_weights.bin", "same_s_decoder_int8_weights_manifest.txt"],
 "same_l_decoder_int8": ["same_l_decoder_int8.so", "same_l_decoder_int8_weights.bin", "same_l_decoder_int8_weights_manifest.txt"],
 "same_s_encoder": ["same_s_encoder_bf16.so", "same_s_encoder_bf16_weights.bin", "same_s_encoder_bf16_weights_manifest.txt"],
 "same_l_encoder": ["same_l_encoder_bf16.so", "same_l_encoder_bf16_weights.bin", "same_l_encoder_bf16_weights_manifest.txt",
                    "same_l_encoder_bf16_weights_f32.bin", "same_l_encoder_bf16_weights_f32_manifest.txt"],
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
                t.extractall(f"{_C}/tritoncpu_sa3")        # recreates aot_stage2/so, cpp_kernels.txt, aot_speedprove/so_flash
            continue
        dst = FILES[hf]
        if os.path.exists(dst):
            continue
        _place(_fetch(hf), dst)

def ensure_all():
    for g in GROUPS:
        ensure(g)
