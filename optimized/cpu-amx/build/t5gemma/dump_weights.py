#!/usr/bin/env python3
"""Dump the T5Gemma-b-b-ul2 encoder npz -> {weights.bin + weights_manifest.txt}
for the torch-free C++ AMX engine. MATCHES the SAME-L dump format exactly so the
mmap loader (load_weights) is reused unchanged.

Layout choices (mirror same_l_cpu_amx/dump_weights.py):
  * The 7 linears/layer + embed table: stored **bf16**.
    - Linears (nn.Linear weight [out,in]): stored as W.T = [in=K, out=N] so the
      oneDNN gemm_bf16(src[M,K], wei[K,N]) consumes them directly (pre-transposed).
    - Embedding [256000,768]: stored as-is (a gather table; row = token id). bf16.
  * RMSNorm weights, rope_inv_freq: fp32 (the cancellation-fragile islands stay fp32).

bf16 conversion is round-to-nearest-even done in pure numpy, BIT-IDENTICAL to the
C++ runtime f2b() (r = x + 0x7fff + ((x>>16)&1); bf16 = r>>16). No torch needed.

Manifest line:  name dtype byte_offset nelem d0 d1 ...
"""
import os
import numpy as np

NPZ = "/weka2/cj/clod/t5gemma_cpu_amx/hf/MLX/t5gemma_f16.npz"
OUT = "/weka2/cj/clod/t5gemma_cpu_amx"
NB = 12


def f32_to_bf16_rne(arr):
    """f32 -> bf16 (uint16) round-to-nearest-even, bit-identical to C++ f2b()."""
    x = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = x + np.uint64(0x7FFF) + ((x >> np.uint64(16)) & np.uint64(1))
    return (r >> np.uint64(16)).astype(np.uint16)


z = np.load(NPZ, allow_pickle=True)
def a(k):  # fp32 numpy
    return z[k].astype(np.float32)

blob = bytearray()
lines = []


def put(name, arr, dt):
    """dt in {'bf16','f32'}. arr is numpy fp32 (any shape); stored C-contiguous."""
    global blob
    arr = np.ascontiguousarray(arr.astype(np.float32))
    off = len(blob)
    if dt == "bf16":
        blob += f32_to_bf16_rne(arr).tobytes()
    elif dt == "f32":
        blob += arr.ravel().tobytes()
    else:
        raise ValueError(dt)
    shp = " ".join(str(s) for s in arr.shape)
    lines.append(f"{name} {dt} {off} {arr.size} {shp}")


def put_lin(name, w_oi):
    """w_oi = npz [out,in]; store bf16 W.T = [in,out] for oneDNN wei[K,N]."""
    put(name, np.ascontiguousarray(w_oi.T), "bf16")


# ---- top level ----
put("embed", a("embed_tokens.weight"), "bf16")   # [256000,768] gather table, bf16
put("norm", a("norm.weight"), "f32")              # [768]
put("rope_inv", z["rope_inv_freq"].astype(np.float32), "f32")  # [32]

# ---- 12 layers ----
for i in range(NB):
    p = f"layers.{i}."
    put(f"L{i}.pre_a",  a(p + "pre_self_attn_layernorm.weight"),  "f32")
    put(f"L{i}.post_a", a(p + "post_self_attn_layernorm.weight"), "f32")
    put(f"L{i}.pre_f",  a(p + "pre_feedforward_layernorm.weight"), "f32")
    put(f"L{i}.post_f", a(p + "post_feedforward_layernorm.weight"), "f32")
    put_lin(f"L{i}.q", a(p + "self_attn.q_proj.weight"))   # [768,768] -> bf16 [768,768]
    put_lin(f"L{i}.k", a(p + "self_attn.k_proj.weight"))
    put_lin(f"L{i}.v", a(p + "self_attn.v_proj.weight"))
    put_lin(f"L{i}.o", a(p + "self_attn.o_proj.weight"))
    put_lin(f"L{i}.gate", a(p + "mlp.gate_proj.weight"))   # [2048,768] -> bf16 [768,2048]
    put_lin(f"L{i}.up",   a(p + "mlp.up_proj.weight"))     # [2048,768] -> bf16 [768,2048]
    put_lin(f"L{i}.down", a(p + "mlp.down_proj.weight"))   # [768,2048] -> bf16 [2048,768]

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights_manifest.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote weights.bin ({len(blob)/1e6:.1f} MB), {len(lines)} arrays")
print("first/last few manifest lines:")
for l in lines[:6] + ["  ..."] + lines[-3:]:
    print(" ", l)
