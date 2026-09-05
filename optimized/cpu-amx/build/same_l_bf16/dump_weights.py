#!/usr/bin/env python3
"""Dump SAME-L fp32 npz -> flat {weights.bin + weights_manifest.txt} for the C++ engine.

SAME-L = native 426M medium decoder: 12 blocks, dim=1536, 24 heads, hd=64, banded
SWA attention (window +/-17), sin-gate FF from block 5, plain Linear(1536->512) output map.

Layout choices (mirror same_s_cpu_amx/dump_weights.py):
  * Linears: stored **bf16** in oneDNN matmul weight layout [K=in, N=out] (= W.T of the
    npz [out,in]). AMX-BF16 GEMM src[M,K]bf16 x wei[K,N]bf16 -> dst[M,N]f32.
  * Output map = plain Conv1d(k=1) == Linear: mapping.weight npz is [out=512,in=1536,k=1]
    (weight_norm ALREADY fused in export). Reshape -> [512,1536], store bf16 W.T = [1536,512].
    SIMPLER than SAME-S (no k=3 im2col conv).
  * DyT (alpha scalar, gamma/beta), biases, new_tokens, running_std: fp32 (the
    cancellation-fragile differential-attention elementwise stays fp32).

bf16 conversion uses torch's round-to-nearest-even (build-time tool; the *runtime* is
torch-free). Manifest line:  name dtype byte_offset nelem d0 d1 ...
"""

import os

# Paths come from the environment so nothing local is baked in.
#   SA3_CPUAMX_HOME  where the engine dirs live (default ./engines)
#   SA3_REPO         checkout providing the reference weights to dump
HOME = os.environ.get("SA3_CPUAMX_HOME", os.path.abspath("engines"))
REPO = os.environ.get("SA3_REPO", os.path.abspath("."))
import os
import numpy as np
import torch

SA3 = REPO
WEIGHTS = os.path.join(SA3, "models", "mlx", "same_l_decoder_f32.npz")
OUT = os.path.join(HOME, "same_l_cpu_amx")
NB = 12

raw = dict(np.load(WEIGHTS))
def a(k):  # fp32 numpy
    return raw[k].astype(np.float32)

blob = bytearray()
lines = []

def put(name, arr, dt):
    """dt in {'bf16','f32'}. arr is numpy fp32 (any shape); stored C-contiguous."""
    global blob
    arr = np.ascontiguousarray(arr.astype(np.float32))
    off = len(blob)
    if dt == "bf16":
        t = torch.from_numpy(arr).to(torch.bfloat16)
        u = t.view(torch.uint16).numpy().ravel()
        blob += u.tobytes()
    elif dt == "f32":
        blob += arr.ravel().tobytes()
    else:
        raise ValueError(dt)
    shp = " ".join(str(s) for s in arr.shape)
    lines.append(f"{name} {dt} {off} {arr.size} {shp}")

def put_lin(name, w_oi):
    """w_oi = npz [out,in]; store bf16 W.T = [in,out] for oneDNN wei[K,N]."""
    put(name, np.ascontiguousarray(w_oi.T), "bf16")

# ---- top-level ----
put("running_std", a("running_std"), "f32")
put_lin("project_in.w", a("project_in.weight"))          # [256,1536]
put("project_in.b", a("project_in.bias"), "f32")         # [1536]
put("new_tokens", a("new_tokens").reshape(-1), "f32")    # [1536]

for b in range(NB):
    p = f"blocks.{b}"
    put(f"b{b}.pre.alpha", a(f"{p}.pre_norm.alpha"), "f32")
    put(f"b{b}.pre.gamma", a(f"{p}.pre_norm.gamma"), "f32")
    put(f"b{b}.pre.beta",  a(f"{p}.pre_norm.beta"),  "f32")
    put_lin(f"b{b}.qkv.w", a(f"{p}.attn.to_qkv.weight"))   # [1536,7680]
    put(f"b{b}.qn.alpha", a(f"{p}.attn.q_norm.alpha"), "f32")
    put(f"b{b}.qn.gamma", a(f"{p}.attn.q_norm.gamma"), "f32")  # [64]
    put(f"b{b}.qn.beta",  a(f"{p}.attn.q_norm.beta"),  "f32")
    put(f"b{b}.kn.alpha", a(f"{p}.attn.k_norm.alpha"), "f32")
    put(f"b{b}.kn.gamma", a(f"{p}.attn.k_norm.gamma"), "f32")
    put(f"b{b}.kn.beta",  a(f"{p}.attn.k_norm.beta"),  "f32")
    put_lin(f"b{b}.out.w", a(f"{p}.attn.to_out.weight"))   # [1536,1536]
    put(f"b{b}.ff.alpha", a(f"{p}.ff_norm.alpha"), "f32")
    put(f"b{b}.ff.gamma", a(f"{p}.ff_norm.gamma"), "f32")
    put(f"b{b}.ff.beta",  a(f"{p}.ff_norm.beta"),  "f32")
    put_lin(f"b{b}.glu.w", a(f"{p}.ff.glu_proj.weight"))   # [1536,9216]
    put(f"b{b}.glu.b", a(f"{p}.ff.glu_proj.bias"), "f32")  # [9216]
    put_lin(f"b{b}.proj.w", a(f"{p}.ff.proj_out.weight"))  # [4608,1536]
    put(f"b{b}.proj.b", a(f"{p}.ff.proj_out.bias"), "f32") # [1536]

# output map: plain Linear(1536->512). mapping.weight [512,1536,1] -> [512,1536] -> W.T [1536,512]
put_lin("map.w", a("mapping.weight").reshape(512, 1536))   # -> bf16 [1536,512]
put("map.b", a("mapping.bias"), "f32")                     # [512]

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights_manifest.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote weights.bin ({len(blob)/1e6:.1f} MB), {len(lines)} arrays")
print("first/last few manifest lines:")
for l in lines[:5] + ["..."] + lines[-4:]:
    print(" ", l)
