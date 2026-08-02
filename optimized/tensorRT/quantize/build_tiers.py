#!/usr/bin/env python3
"""Build a SA3 encoder/decoder tier engine with a WIDE dynamic profile.

    python build_tiers.py <tier.onnx> <out.trt> --arch {same-s,same-l} --kind {enc,dec} [--fp8]

Why "wide": the historical [32,4096]-frame profile was a conservative default, not a model limit.
With min=1 the models are correct at L=1..31 (verified vs eager — these can't be chunked), and long
files run natively (a ~12:40 decode / 6:20 encode is a single call, no chunking). See README.

Profiles built here:
  decoder : latent (1,256,1) .. (1,256,1292) .. (1,256,8192)             # 1 frame .. ~12:40
  encoder : audio  (1,2,1)   .. (1,2,2097152) .. (1,2,33554432)          # 1 samp  .. ~12:40

SAME-S  -> weakly-typed EXPLICIT_BATCH + BF16 (+ FP8 builder flag for fp8/fp8_fast tiers).
SAME-L  -> strongly-typed + diff_attn_swa plugin (PREFER_JIT) + NO FP8 flag
           (the in-graph fp8 QDQ nodes carry the precision themselves).
`--fp8` is required for fp8 / fp8_fast SAME-S tiers; harmless/ignored for SAME-L and w8/bf16.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import tensorrt as trt

DEC_PROFILE = ((1, 256, 1), (1, 256, 1292), (1, 256, 8192))
ENC_PROFILE = ((1, 2, 1), (1, 2, 2097152), (1, 2, 33554432))

def build(onnx_path, out_path, arch, kind, fp8):
    lg = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(lg, "")
    strong = arch == "same-l"
    if strong:
        import diff_attn_nocast_plugin  # registers samel::diff_attn_swa (AOT + JIT impls)  # noqa: F401
    flags = 0
    if strong:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS)
    else:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    b = trt.Builder(lg); net = b.create_network(flags); p = trt.OnnxParser(net, lg)
    if not p.parse_from_file(onnx_path):
        for i in range(p.num_errors): print(p.get_error(i))
        raise SystemExit("parse failed")
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 20 << 30)
    if not strong:
        cfg.set_flag(trt.BuilderFlag.BF16)
        if fp8: cfg.set_flag(trt.BuilderFlag.FP8)   # SAME-L carries fp8 in-graph; only SAME-S needs the flag
    name, prof = ("audio", ENC_PROFILE) if kind == "enc" else ("latent", DEC_PROFILE)
    pr = b.create_optimization_profile(); pr.set_shape(name, *prof); cfg.add_optimization_profile(pr)
    ser = b.build_serialized_network(net, cfg)
    if ser is None: raise SystemExit("build failed")
    with open(out_path, "wb") as f: f.write(ser)
    print(f"built {out_path} ({ser.nbytes/1e6:.0f} MB) — {arch} {kind} {'fp8 ' if fp8 else ''}wide", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx"); ap.add_argument("out")
    ap.add_argument("--arch", required=True, choices=["same-s", "same-l"])
    ap.add_argument("--kind", required=True, choices=["enc", "dec"])
    ap.add_argument("--fp8", action="store_true", help="required for SAME-S fp8/fp8_fast tiers")
    a = ap.parse_args()
    build(a.onnx, a.out, a.arch, a.kind, a.fp8)
