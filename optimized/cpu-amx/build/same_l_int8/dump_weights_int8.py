#!/usr/bin/env python3
"""Dump SAME-L fp32 npz -> flat {weights.bin + weights_manifest.txt} for the INT8 C++ engine.

Mirrors ../same_l_cpu_amx/dump_weights.py, but Linears + the plain Linear(1536->512) output
map are stored **w8: per-output-channel symmetric int8** (name.q int8 [K,N] + name.scale f32 [N]),
exactly the DiT/SAME-S-int8 scheme. Runtime: per-row dynamic int8 activation x per-channel int8
weight -> s8s8->s32 AMX GEMM -> deq. Banded differential attention stays fp32 (Stage 1).

DyT/biases/new_tokens/running_std: fp32. Optional SmoothQuant (SQ=1) folds per-in-channel s[K]
into `to_qkv` (+ 1/s into pre_norm gamma/beta), read from sq_scales.npz (calibrate_sq.py).
"""

import os

# Paths come from the environment so nothing local is baked in.
#   SA3_CPUAMX_HOME  where the engine dirs live (default ./engines)
#   SA3_REPO         checkout providing the reference weights to dump
HOME = os.environ.get("SA3_CPUAMX_HOME", os.path.abspath("engines"))
REPO = os.environ.get("SA3_REPO", os.path.abspath("."))
import os
import numpy as np

SA3 = REPO
WEIGHTS = os.path.join(SA3, "models", "mlx", "same_l_decoder_f32.npz")
OUT = os.path.join(HOME, "same_l_int8_cpu_amx")
NB = 12
USE_SQ = os.environ.get("SQ", "0") == "1"
SQ_NPZ = os.path.join(OUT, "sq_scales.npz")

raw = dict(np.load(WEIGHTS))
def a(k):
    return raw[k].astype(np.float32)

sq = dict(np.load(SQ_NPZ)) if (USE_SQ and os.path.exists(SQ_NPZ)) else {}
if USE_SQ:
    print(f"SmoothQuant ON: {'loaded '+SQ_NPZ if sq else 'NO sq_scales.npz -> plain'}")

blob = bytearray()
lines = []

def put(name, arr, dt):
    global blob
    off = len(blob)
    if dt == "i8":
        arr = np.ascontiguousarray(arr.astype(np.int8)); blob += arr.tobytes()
    elif dt == "f32":
        arr = np.ascontiguousarray(arr.astype(np.float32)); blob += arr.ravel().tobytes()
    else:
        raise ValueError(dt)
    lines.append(f"{name} {dt} {off} {arr.size} " + " ".join(str(s) for s in arr.shape))

def quant_w(wt):
    amax = np.abs(wt).max(axis=0)
    scale = np.maximum(amax / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(wt / scale[None, :]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)

def put_lin_i8(name, w_oi, smooth_s=None):
    wt = np.ascontiguousarray(w_oi.T).astype(np.float32)   # [K=in, N=out]
    if smooth_s is not None:
        wt = wt * smooth_s[:, None]
    q, scale = quant_w(wt)
    put(name + ".q", q, "i8")
    put(name + ".scale", scale, "f32")

# ---- top-level ----
put("running_std", a("running_std"), "f32")
put_lin_i8("project_in", a("project_in.weight"))          # [256,1536]
put("project_in.b", a("project_in.bias"), "f32")          # [1536]
put("new_tokens", a("new_tokens").reshape(-1), "f32")     # [1536]

for b in range(NB):
    p = f"blocks.{b}"
    pg = a(f"{p}.pre_norm.gamma").copy()
    pb = a(f"{p}.pre_norm.beta").copy()
    s_qkv = sq.get(f"b{b}.qkv.s") if USE_SQ else None
    if s_qkv is not None:
        pg = pg / s_qkv
        pb = pb / s_qkv
    put(f"b{b}.pre.alpha", a(f"{p}.pre_norm.alpha"), "f32")
    put(f"b{b}.pre.gamma", pg, "f32")
    put(f"b{b}.pre.beta",  pb, "f32")
    put_lin_i8(f"b{b}.qkv", a(f"{p}.attn.to_qkv.weight"), smooth_s=s_qkv)  # [1536,7680]
    put(f"b{b}.qn.alpha", a(f"{p}.attn.q_norm.alpha"), "f32")
    put(f"b{b}.qn.gamma", a(f"{p}.attn.q_norm.gamma"), "f32")  # [64]
    put(f"b{b}.qn.beta",  a(f"{p}.attn.q_norm.beta"),  "f32")
    put(f"b{b}.kn.alpha", a(f"{p}.attn.k_norm.alpha"), "f32")
    put(f"b{b}.kn.gamma", a(f"{p}.attn.k_norm.gamma"), "f32")
    put(f"b{b}.kn.beta",  a(f"{p}.attn.k_norm.beta"),  "f32")
    put_lin_i8(f"b{b}.out", a(f"{p}.attn.to_out.weight"))   # [1536,1536]
    put(f"b{b}.ff.alpha", a(f"{p}.ff_norm.alpha"), "f32")
    put(f"b{b}.ff.gamma", a(f"{p}.ff_norm.gamma"), "f32")
    put(f"b{b}.ff.beta",  a(f"{p}.ff_norm.beta"),  "f32")
    put_lin_i8(f"b{b}.glu", a(f"{p}.ff.glu_proj.weight"))   # [1536,9216]
    put(f"b{b}.glu.b", a(f"{p}.ff.glu_proj.bias"), "f32")   # [9216]
    put_lin_i8(f"b{b}.proj", a(f"{p}.ff.proj_out.weight"))  # [4608,1536]  (NEVER smoothed)
    put(f"b{b}.proj.b", a(f"{p}.ff.proj_out.bias"), "f32")  # [1536]

# output map: plain Linear(1536->512). mapping.weight [512,1536,1] -> [512,1536] -> int8 W.T [1536,512]
put_lin_i8("map", a("mapping.weight").reshape(512, 1536))   # int8 [1536,512] + scale [512]
put("map.b", a("mapping.bias"), "f32")                      # [512]

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights_manifest.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote weights.bin ({len(blob)/1e6:.1f} MB), {len(lines)} arrays  (SQ={'on' if sq else 'off'})")
for l in lines[:4] + ["..."] + lines[-4:]:
    print(" ", l)
