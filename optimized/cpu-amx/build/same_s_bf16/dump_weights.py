#!/usr/bin/env python3
"""Dump SAME-S fp32 npz -> flat {weights.bin + weights_manifest.txt} for the C++ engine.

Layout choices (match model_bf16.py SHIP config exactly):
  * Linears: stored **bf16** in oneDNN matmul weight layout [K=in, N=out] (= W.T of the
    npz [out,in]). AMX-BF16 GEMM src[M,K]bf16 x wei[K,N]bf16 -> dst[M,N]f32.
  * Conv (WNConv1d 768->512 k=3, weight_norm PRE-FUSED in the npz): stored bf16 as
    [K=768*3, N=512] = conv_w.permute(1,2,0).reshape(2304,512) — exactly model_bf16's
    im2col weight `cw`, whose col index is in*3+k (in-major, k-minor).
  * DyT (alpha scalar, gamma/beta), biases, new_tokens, running_std: fp32 (the
    cancellation-fragile elementwise stays fp32).

bf16 conversion uses torch's round-to-nearest-even (build-time tool; the *runtime* is
torch-free). Manifest line:  name dtype byte_offset nelem d0 d1 ...
"""
import os, sys
import numpy as np
import torch

SA3 = "/weka2/cj/clod/q4/sa3-w4-cluster"
WEIGHTS = os.path.join(SA3, "models", "mlx", "same_s_decoder_f32.npz")
OUT = "/weka2/cj/clod/same_s_cpu_amx"
NB = 6

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
put_lin("project_in.w", a("project_in.weight"))       # [256,768]
put("project_in.b", a("project_in.bias"), "f32")      # [768]
put("new_tokens", a("new_tokens").reshape(-1), "f32") # [768]

for b in range(NB):
    p = f"blocks.{b}"
    put(f"b{b}.pre.alpha", a(f"{p}.pre_norm.alpha"), "f32")
    put(f"b{b}.pre.gamma", a(f"{p}.pre_norm.gamma"), "f32")
    put(f"b{b}.pre.beta",  a(f"{p}.pre_norm.beta"),  "f32")
    put_lin(f"b{b}.qkv.w", a(f"{p}.attn.to_qkv.weight"))   # [768,3840]
    put(f"b{b}.qn.alpha", a(f"{p}.attn.q_norm.alpha"), "f32")
    put(f"b{b}.qn.gamma", a(f"{p}.attn.q_norm.gamma"), "f32")  # [64]
    put(f"b{b}.qn.beta",  a(f"{p}.attn.q_norm.beta"),  "f32")
    put(f"b{b}.kn.alpha", a(f"{p}.attn.k_norm.alpha"), "f32")
    put(f"b{b}.kn.gamma", a(f"{p}.attn.k_norm.gamma"), "f32")
    put(f"b{b}.kn.beta",  a(f"{p}.attn.k_norm.beta"),  "f32")
    put_lin(f"b{b}.out.w", a(f"{p}.attn.to_out.weight"))   # [768,768]
    put(f"b{b}.ff.alpha", a(f"{p}.ff_norm.alpha"), "f32")
    put(f"b{b}.ff.gamma", a(f"{p}.ff_norm.gamma"), "f32")
    put(f"b{b}.ff.beta",  a(f"{p}.ff_norm.beta"),  "f32")
    put_lin(f"b{b}.glu.w", a(f"{p}.ff.glu_proj.weight"))   # [768,4608]
    put(f"b{b}.glu.b", a(f"{p}.ff.glu_proj.bias"), "f32")  # [4608]
    put_lin(f"b{b}.proj.w", a(f"{p}.ff.proj_out.weight"))  # [2304,768]
    put(f"b{b}.proj.b", a(f"{p}.ff.proj_out.bias"), "f32") # [768]

# conv: [512,768,3]=[out,in,k] -> [in,k,out]=[768,3,512] -> [2304,512] (row=in*3+k)
cw = np.ascontiguousarray(a("mapping.weight").transpose(1, 2, 0).reshape(768 * 3, 512))
put("conv.w", cw, "bf16")           # already [K=2304, N=512]
put("conv.b", a("mapping.bias"), "f32")  # [512]

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights_manifest.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote weights.bin ({len(blob)/1e6:.1f} MB), {len(lines)} arrays")
print("first/last few manifest lines:")
for l in lines[:4] + ["..."] + lines[-4:]:
    print(" ", l)
