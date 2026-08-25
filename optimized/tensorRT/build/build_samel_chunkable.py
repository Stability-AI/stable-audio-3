#!/usr/bin/env python3
"""Build the CHUNKABLE SAME-L encoder / decoder — the canonical fp16 and fp8 engines.

Each engine carries TWO optimization profiles:

    band 0   decoder L=1..256   encoder L=1..64      509 MB / 130 MB of scratch
    band 1   decoder L=1..4096  encoder L=1..4096   8143 MB / 8330 MB

Why two bands rather than two files: TensorRT commits a context's scratch at
create_execution_context(), sized from the selected profile and *before any shape is bound*, so a
single 4096-ceiling engine reserves 8.1 GB whether it decodes five seconds or six minutes. Windowed
decode keeps every enqueue inside band 0's ceiling while the render length stays free.

⚠ The saving only materialises with USER_MANAGED contexts sized by
get_device_memory_size_for_profile_v2(band). `device_memory_size_v2` is the MAX ACROSS profiles, so
a DEFAULT context on a two-profile engine reserves the wide band and the entire benefit disappears
silently. scripts/sa3_trt_core.TRTRunner(..., profile=N) does this correctly.

`opt` is a SEPARATE lever from the ceiling and costs no memory: TRT tunes tactic selection around
it. Measured across L=32..4096, the inherited default of min(1292, ceiling) costs 17-23% at short L
on the wide band while buying nothing at long L. Tuned values, per band:

    decoder   256 -> opt 256   (fp16) / opt 128 (fp8)      4096 -> opt 1024
    encoder    64 -> opt  64                                4096 -> opt  256

fp8 differs only on the chunked decoder band, where fp8's faster kernels shift the balance toward
fixed overhead and pull the optimum down.

RECIPE. SAME-L is STRONGLY_TYPED with the diff_attn_swa plugin; the fp8 variants carry their
quantisation as in-graph QDQ nodes and must NOT get trt.BuilderFlag.FP8 (strongly-typed forbids
it). AOT plugin, not JIT: JIT runs Python in the enqueue path, is not stream-capturable, and was
measured to corrupt the heap on repeated enqueues in a multi-profile engine where AOT was bit-exact.

  fp16 decoder:  --kind dec --src onnx/same-l/dec_fp16_ship_ceil.onnx --bands 256,4096 --opts 256,1024
  fp8  decoder:  --kind dec --src onnx/same-l/dec_fp8_ship_ceil.onnx  --bands 256,4096 --opts 128,1024
  fp16 encoder:  --kind enc --src onnx/same-l/enc_dynamic_triton_swa.onnx --bands 64,4096 --opts 64,256
  fp8  encoder:  --kind enc --src onnx/same-l/enc_fp8_amax.onnx --bands 64,4096 --opts 64,256

The decoder ONNX must already carry the limiter with its ceiling promoted to a runtime input; the
fp8 encoder ONNX must be the amax-recalibrated one (see build/recalib_enc_fp8.py).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import tensorrt as trt

SPL = 4096                     # samples per latent
ROOT = Path(__file__).resolve().parent


def build(src, out, kind, bands, opts, min_l, plugin, workspace_gb=16):
    os.environ["SA3_SWA_PLUGIN"] = plugin
    sys.path.insert(0, str(ROOT.parent / "scripts"))
    import diff_attn_nocast_plugin  # noqa: F401  registers samel::diff_attn_swa

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS
                      if plugin == "aot" else
                      trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS)
    builder = trt.Builder(logger)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(str(src)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(2)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    # NO trt.BuilderFlag.FP8 — a strongly-typed network carries fp8 in its own QDQ nodes.
    for hi, opt in zip(bands, opts):
        prof = builder.create_optimization_profile()
        if kind == "enc":
            prof.set_shape("audio", (1, 2, min_l * SPL), (1, 2, opt * SPL), (1, 2, hi * SPL))
        else:
            prof.set_shape("latent", (1, 256, min_l), (1, 256, opt), (1, 256, hi))
            # limiter_ceiling is rank-0: a scalar input needs no profile entry.
        cfg.add_optimization_profile(prof)

    print(f"[build] same-l {kind}: " +
          " | ".join(f"L={min_l}..{hi} opt={opt}" for hi, opt in zip(bands, opts)) +
          f", STRONGLY_TYPED + {plugin.upper()} plugin", flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise SystemExit("BUILD FAILED")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(memoryview(ser))

    eng = trt.Runtime(logger).deserialize_cuda_engine(Path(out).read_bytes())
    print(f"[done] {Path(out).name}  {Path(out).stat().st_size / 1e6:.0f} MB  "
          f"in {time.time() - t0:.0f}s", flush=True)
    tname = "audio" if kind == "enc" else "latent"
    div = SPL if kind == "enc" else 1
    for i, hi in enumerate(bands):
        lo_s, opt_s, hi_s = eng.get_tensor_profile_shape(tname, i)
        print(f"    band {i}: L={lo_s[-1] // div}..{hi_s[-1] // div} opt={opt_s[-1] // div}  "
              f"scratch {eng.get_device_memory_size_for_profile_v2(i) / 1e6:.1f} MB")
    print(f"    ⚠ engine-wide device_memory_size_v2 = {eng.device_memory_size_v2 / 1e6:.0f} MB "
          f"— what a DEFAULT context would grab; use TRTRunner(profile=N)")
    print("    IO: " + " | ".join(
        f"{eng.get_tensor_name(j)} {eng.get_tensor_dtype(eng.get_tensor_name(j)).name}"
        for j in range(eng.num_io_tensors)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source ONNX")
    ap.add_argument("--out", required=True, help="output .trt")
    ap.add_argument("--kind", default="dec", choices=("dec", "enc"))
    ap.add_argument("--bands", default="", help="ceilings in LATENTS, e.g. 256,4096")
    ap.add_argument("--opts", default="", help="opt shape per band, in LATENTS")
    ap.add_argument("--min-l", type=int, default=1, help="profile floor in latents")
    ap.add_argument("--plugin", default="aot", choices=("aot", "jit"))
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",")] if a.bands else \
        ([256, 4096] if a.kind == "dec" else [64, 4096])
    opts = [int(x) for x in a.opts.split(",")] if a.opts else \
        [min(1292, hi) for hi in bands]
    if len(opts) != len(bands):
        ap.error(f"--opts has {len(opts)} entries for {len(bands)} bands")
    build(a.src, a.out, a.kind, bands, opts, a.min_l, a.plugin)


if __name__ == "__main__":
    main()
