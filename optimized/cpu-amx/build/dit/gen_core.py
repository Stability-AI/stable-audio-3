#!/usr/bin/env python
"""Parameterized core.bin generator for the speedprove driver (read-only on the npz sources).
Flatten weights_int8_L{L}.npz + injected golden preamble (x_init, ctx_i8, ctx_s, gc) + golden
out -> raw core_L{L}.bin + core_L{L}_manifest.txt in aot_speedprove/. Same byte layout as
aot_stage2/dump_bin.py; cpp_kernels.txt/so/ are L-independent and reused from aot_stage2."""
import os, sys
import numpy as np

SRC = "/weka2/cj/clod/tritoncpu_sa3/aot_stage2"          # read-only npz source
DST = "/weka2/cj/clod/tritoncpu_sa3/aot_speedprove"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 1292
DT = {np.dtype("float32"): "f32", np.dtype("int8"): "i8", np.dtype("int32"): "i32"}

W = dict(np.load(f"{SRC}/weights_int8_L{L}.npz"))
G = dict(np.load(f"{SRC}/golden_L{L}.npz"))
arrays = {}
arrays.update({k: v for k, v in W.items()})
for k in ("x_init", "ctx_i8", "ctx_s", "gc", "out"):
    arrays[k] = G[k]

buf = bytearray()
man = []
for name, a in arrays.items():
    a = np.ascontiguousarray(a)
    off = len(buf)
    buf += a.tobytes()
    man.append((name, DT[a.dtype], off, a.size, list(a.shape)))
with open(f"{DST}/core_L{L}.bin", "wb") as f:
    f.write(buf)
with open(f"{DST}/core_L{L}_manifest.txt", "w") as f:
    for name, dt, off, n, shp in man:
        f.write(f"{name} {dt} {off} {n} {' '.join(map(str, shp))}\n")
print(f"core_L{L}.bin {len(buf)} bytes ({len(buf)/1e9:.2f} GB), {len(man)} arrays; x_init={arrays['x_init'].shape} out={arrays['out'].shape}")
