#!/usr/bin/env python3
"""Quality gate: audio-PSNR of the torch-free C++ AMX-BF16 decode vs the fp32 torch decode,
on the fixed real sm-music latent. Also checks it matches the Python-Triton bf16 SHIP tier.

Gate (from same_s_bf16_amx/RESULTS.md): audio-PSNR vs fp32 ~ 50.9 dB @seq320 / 54.3 @seq1292.
"""
import sys, os
sys.path.insert(0, "/weka2/cj/clod/same_s_bf16_amx")
import numpy as np
import torch
import ref_common as R
from same_s_cpu_backend import SamesCPU

torch.set_num_threads(16)
Ls = [int(x) for x in sys.argv[1:]] or [320, 1292]

m = SamesCPU(threads=16)
tm = R.load_torch_fp32(output_audio=False)

# optional: the Python-Triton bf16 SHIP reference (per-op parity check)
try:
    from model_bf16 import SAMESBf16
    SHIP = dict(lin_backend="onednn", attn_backend="sdpa", conv_backend="onednn")
    py = SAMESBf16(tm, num_threads=16, **SHIP)
except Exception as e:
    py = None
    print("(python bf16 ref unavailable:", e, ")")

print(f"{'L':>6} {'audio-s':>8} | {'C++ patch dB':>12} {'C++ audio dB':>12} {'cos':>9} | "
      f"{'pyBF16 audio':>12} {'C++ vs pyBF16':>13}")
for L in Ls:
    lat = R.make_latent(L)
    with torch.no_grad():
        ref_patch = tm(torch.from_numpy(lat)).numpy()
    ref_audio = R.unpatch(ref_patch).numpy()

    cpp_patch = m.forward(lat)
    cpp_audio = R.unpatch(cpp_patch).numpy()

    dpp = R.psnr(ref_patch, cpp_patch)
    dpa = R.psnr(ref_audio, cpp_audio)
    cs = R.cos(ref_audio, cpp_audio)

    if py is not None:
        py_patch = py.forward(lat)
        py_audio = R.unpatch(py_patch).numpy()
        dpy = R.psnr(ref_audio, py_audio)
        dcp = R.psnr(py_audio, cpp_audio)   # C++ vs python-bf16 (per-op parity tier)
        s_py, s_cp = f"{dpy:12.2f}", f"{dcp:13.2f}"
    else:
        s_py, s_cp = f"{'-':>12}", f"{'-':>13}"

    print(f"{L:>6} {L*4096/44100:>8.1f} | {dpp:12.2f} {dpa:12.2f} {cs:9.5f} | {s_py} {s_cp}")
