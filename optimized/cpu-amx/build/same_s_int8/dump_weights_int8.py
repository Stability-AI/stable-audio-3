#!/usr/bin/env python3
"""Dump SAME-S fp32 npz -> flat {weights.bin + weights_manifest.txt} for the INT8 C++ engine.

Mirrors ../same_s_cpu_amx/dump_weights.py, but the Linears (and the WNConv1d) are stored
**w8: per-output-channel symmetric int8** (exactly the DiT engine's scheme, model.py W):
    wt = W.T  = [K=in, N=out]
    scale[N] = max(|wt|.amax(axis=0) / 127, 1e-12)   # per out-channel
    q[K,N]   = clip(round(wt / scale[None,:]), -127, 127).int8
Runtime: activation is quantized per-row (dynamic symmetric int8) and s8s8->s32 AMX GEMM,
then dequant o[m,n] = acc[m,n]*a_scale[m]*w_scale[n]. Attention stays fp32 (Stage 1).

DyT (alpha/gamma/beta), biases, new_tokens, running_std: fp32 (cancellation-fragile).

Optional SmoothQuant (SQ=1 env): fold per-input-channel smoothing s[K] into `to_qkv` weight
(W_hat[k,n]=s[k]*W[k,n]) and DIVIDE it out of the activation by folding 1/s[k] into the
pre_norm gamma/beta that produce the to_qkv input (h=pre_norm(x) feeds ONLY to_qkv, so this is
exact + runtime-free). s[k] read from sq_scales.npz (built by calibrate_sq.py). proj_out is
NEVER smoothed (214x outliers -> backfires, per smoothquant_test/SMOOTHQUANT_W8A8.md).

Manifest line:  name dtype byte_offset nelem d0 d1 ...    (dtype in {i8,f32})
"""
import os
import numpy as np

SA3 = "/weka2/cj/clod/q4/sa3-w4-cluster"
WEIGHTS = os.path.join(SA3, "models", "mlx", "same_s_decoder_f32.npz")
OUT = "/weka2/cj/clod/same_s_int8_cpu_amx"
NB = 6
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
        arr = np.ascontiguousarray(arr.astype(np.int8))
        blob += arr.tobytes()
    elif dt == "f32":
        arr = np.ascontiguousarray(arr.astype(np.float32))
        blob += arr.ravel().tobytes()
    else:
        raise ValueError(dt)
    shp = " ".join(str(s) for s in arr.shape)
    lines.append(f"{name} {dt} {off} {arr.size} {shp}")

def quant_w(wt):
    """wt [K,N] fp32 -> (q int8 [K,N], scale f32 [N]) per out-channel symmetric."""
    amax = np.abs(wt).max(axis=0)
    scale = np.maximum(amax / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(wt / scale[None, :]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)

def put_lin_i8(name, w_oi, smooth_s=None):
    """w_oi = npz [out,in]; store int8 W.T=[in,out] per-out-channel + f32 scale.
    smooth_s: optional per-input-channel s[K] to fold (W_hat[k,n]=s[k]*W[k,n])."""
    wt = np.ascontiguousarray(w_oi.T).astype(np.float32)   # [K=in, N=out]
    if smooth_s is not None:
        wt = wt * smooth_s[:, None]
    q, scale = quant_w(wt)
    put(name + ".q", q, "i8")
    put(name + ".scale", scale, "f32")

# ---- top-level ----
put("running_std", a("running_std"), "f32")
put_lin_i8("project_in", a("project_in.weight"))       # [256,768]
put("project_in.b", a("project_in.bias"), "f32")       # [768]
put("new_tokens", a("new_tokens").reshape(-1), "f32")  # [768]

for b in range(NB):
    p = f"blocks.{b}"
    pg = a(f"{p}.pre_norm.gamma").copy()
    pb = a(f"{p}.pre_norm.beta").copy()
    s_qkv = sq.get(f"b{b}.qkv.s") if USE_SQ else None
    if s_qkv is not None:
        pg = pg / s_qkv          # h' = h/s  -> fold 1/s into gamma,beta (h feeds ONLY to_qkv)
        pb = pb / s_qkv
    put(f"b{b}.pre.alpha", a(f"{p}.pre_norm.alpha"), "f32")
    put(f"b{b}.pre.gamma", pg, "f32")
    put(f"b{b}.pre.beta",  pb, "f32")
    put_lin_i8(f"b{b}.qkv", a(f"{p}.attn.to_qkv.weight"), smooth_s=s_qkv)  # [768,3840]
    put(f"b{b}.qn.alpha", a(f"{p}.attn.q_norm.alpha"), "f32")
    put(f"b{b}.qn.gamma", a(f"{p}.attn.q_norm.gamma"), "f32")
    put(f"b{b}.qn.beta",  a(f"{p}.attn.q_norm.beta"),  "f32")
    put(f"b{b}.kn.alpha", a(f"{p}.attn.k_norm.alpha"), "f32")
    put(f"b{b}.kn.gamma", a(f"{p}.attn.k_norm.gamma"), "f32")
    put(f"b{b}.kn.beta",  a(f"{p}.attn.k_norm.beta"),  "f32")
    put_lin_i8(f"b{b}.out", a(f"{p}.attn.to_out.weight"))   # [768,768]
    put(f"b{b}.ff.alpha", a(f"{p}.ff_norm.alpha"), "f32")
    put(f"b{b}.ff.gamma", a(f"{p}.ff_norm.gamma"), "f32")
    put(f"b{b}.ff.beta",  a(f"{p}.ff_norm.beta"),  "f32")
    put_lin_i8(f"b{b}.glu", a(f"{p}.ff.glu_proj.weight"))   # [768,4608]
    put(f"b{b}.glu.b", a(f"{p}.ff.glu_proj.bias"), "f32")   # [4608]
    put_lin_i8(f"b{b}.proj", a(f"{p}.ff.proj_out.weight"))  # [2304,768]  (NEVER smoothed)
    put(f"b{b}.proj.b", a(f"{p}.ff.proj_out.bias"), "f32")  # [768]

# conv: [512,768,3]=[out,in,k] -> [in,k,out]=[768,3,512] -> [2304,512] (row=in*3+k)
cw = np.ascontiguousarray(a("mapping.weight").transpose(1, 2, 0).reshape(768 * 3, 512))
q, scale = quant_w(cw)
put("conv.q", q, "i8")
put("conv.scale", scale, "f32")
put("conv.b", a("mapping.bias"), "f32")  # [512]

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights_manifest.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote weights.bin ({len(blob)/1e6:.1f} MB), {len(lines)} arrays  (SQ={'on' if sq else 'off'})")
for l in lines[:4] + ["..."] + lines[-5:]:
    print(" ", l)
