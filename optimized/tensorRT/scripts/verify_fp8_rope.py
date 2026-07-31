#!/usr/bin/env python3
"""EngineInspector identity for the fp8-linears + native-FMHA + fp32-RoPE DiT engine.
From the DETAILED per-layer I/O Format/Datatype (baked into the serialized engine), NOT
the build flag.  Three make-or-break checks the task demands:

  CHECK 1  fp8 GEMMs fire on the Linears   — count gemm-type layers running FP8 E4M3 (~222).
  CHECK 2  native fused MHA fires          — count _gemm_mha_v2 / fmha nodes (expect 96,
                                             bf16 or fp16 — NOT 0/unfused, NOT fp8).
  CHECK 3  RoPE is BAKED, not computed at runtime — at most 2 cos/sin layers survive
                                             (timestep_features + seconds_total), and some
                                             fp32 remains. Boundary dtypes are reported but
                                             NOT asserted: they differ per arch by Myelin
                                             fusion (see the comment on check3).

usage: python verify_fp8_rope.py <engine.trt> [label]
"""
import sys, json, re
from collections import Counter
from pathlib import Path
import tensorrt as trt


def norm_dtype(s):
    if not s:
        return None
    t = str(s).lower()
    if "int8" in t: return "INT8"
    if "fp8" in t or "e4m3" in t or "e5m2" in t: return "FP8"
    if "bf16" in t or "bfloat" in t: return "BF16"
    if "fp16" in t or "half" in t or "float16" in t: return "FP16"
    if "fp32" in t or "float32" in t or "float " in t or t.endswith("float"): return "FP32"
    return None


def io_dtypes(ly):
    found = set(); raw = []
    for io_key in ("Inputs", "Outputs"):
        for io in ly.get(io_key, []) or []:
            if isinstance(io, dict):
                for fk in ("Format/Datatype", "Datatype", "Format", "Dtype"):
                    v = io.get(fk)
                    if v:
                        raw.append(str(v)); n = norm_dtype(v)
                        if n: found.add(n)
    for key in ("Precision", "ComputePrecision"):
        v = ly.get(key)
        if v:
            raw.append(f"{key}={v}"); n = norm_dtype(v)
            if n: found.add(n)
    return found, raw


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else Path(path).name
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
    if engine is None:
        raise SystemExit(f"deserialize failed: {path}")
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", info if isinstance(info, list) else [])

    print(f"\n{'='*80}\nENGINE: {label}\n  path: {path}\n"
          f"  size: {Path(path).stat().st_size/1e9:.2f} GB\n  total layers: {len(layers)}")

    # An engine built without ProfilingVerbosity.DETAILED serializes layer NAMES only:
    # the inspector hands back a list of strings, every per-layer lookup finds nothing,
    # and all three checks report False — which reads exactly like "fp8 did not fire"
    # when the truth is "nothing was recorded to look at". Say so instead of guessing.
    n_dict = sum(1 for ly in layers if isinstance(ly, dict))
    if layers and n_dict == 0:
        print(f"\n  {'!'*76}")
        print("  CANNOT VERIFY: this engine was built without ProfilingVerbosity.DETAILED,")
        print("  so it carries layer names only — no per-layer Format/Datatype to inspect.")
        print("  The checks below will all report False regardless of what the engine does.")
        print("  Rebuild for inspection with:")
        print("      cfg = builder.create_builder_config()")
        print("      cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED")
        print("  Verbosity is metadata-only — it changes neither kernel selection nor")
        print("  numerics (measured: identical velocity-cos, speed within 0.7%), so the")
        print("  DETAILED twin is a valid stand-in for the engine you ship.")
        print(f"  {'!'*76}")

    type_hist = Counter(); gemm_prec = Counter(); prec_any = Counter()
    gemm_examples = []; mha_nodes = []; mha_examples = []; fp8_layers = 0
    rope_fp32 = []; rope_seen = []
    MHA_RE = re.compile(r"mha|fmha", re.I)
    # timestep_features and seconds_total are the only Fourier chains evaluated at
    # runtime; RoPE's own cos/sin are baked constants and must not appear as layers.
    MAX_RUNTIME_COSSIN = 2
    ROPE_RE = re.compile(r"(cos|sin|q_norm|k_norm|rotary|inv_freq|angle)", re.I)

    for ly in layers:
        if not isinstance(ly, dict):
            continue
        name = ly.get("Name", "?")
        lt = str(ly.get("LayerType", ly.get("ParameterType", "?")))
        type_hist[lt] += 1
        found, raw = io_dtypes(ly)
        for p in found:
            prec_any[p] += 1
        if "FP8" in found:
            fp8_layers += 1
        if lt == "gemm":
            dom = ("FP8" if "FP8" in found else "INT8" if "INT8" in found else
                   "FP16" if "FP16" in found else "BF16" if "BF16" in found else
                   "FP32" if "FP32" in found else "UNKNOWN")
            gemm_prec[dom] += 1
            if dom == "FP8" and len(gemm_examples) < 4:
                gemm_examples.append((name, raw[:4]))
        blob = json.dumps(ly)
        if MHA_RE.search(name) or MHA_RE.search(blob):
            mha_nodes.append((name, lt, sorted(found)))
            if len(mha_examples) < 4:
                mha_examples.append((name, lt, raw[:6]))
        if ROPE_RE.search(name) and ("cos" in name.lower() or "sin" in name.lower()
                                     or "q_norm" in name.lower() or "k_norm" in name.lower()):
            rope_seen.append(name)
            if "FP32" in found:
                rope_fp32.append(name)

    n_gemm = sum(gemm_prec.values()); n_gemm_fp8 = gemm_prec.get("FP8", 0)
    mha_prec = Counter()
    for _, _, found in mha_nodes:
        d = ("FP8" if "FP8" in found else "BF16" if "BF16" in found else
             "FP16" if "FP16" in found else "FP32" if "FP32" in found else "OTHER")
        mha_prec[d] += 1

    print("\n--- layer-type histogram (top 14) ---")
    for lt, c in type_hist.most_common(14):
        print(f"    {c:6d}  {lt}")

    print(f"\n{'#'*80}\nCHECK 1 — FP8 GEMMs FIRE ON LINEARS?")
    for p in ("FP8", "INT8", "FP16", "BF16", "FP32", "UNKNOWN"):
        if gemm_prec.get(p):
            print(f"    {p:8s}: {gemm_prec[p]:5d}")
    print(f"  => {n_gemm_fp8}/{n_gemm} gemm layers run FP8 (E4M3). total FP8-carrying layers: {fp8_layers}")
    for nm, raw in gemm_examples[:3]:
        print(f"      {nm[:78]}"); [print(f"          {r[:86]}") for r in raw]
    # native-FMHA fp8 count matches the fp8_gemm reference (176), NOT the sage-plugin
    # variant (222): with native bf16-FMHA, TRT fuses attn-adjacent linears differently.
    check1 = n_gemm_fp8 >= 170

    print(f"\n{'#'*80}\nCHECK 2 — NATIVE FUSED MHA FIRES (NOT 0/unfused)?")
    print(f"  fused-MHA (_gemm_mha_v2/fmha) nodes: {len(mha_nodes)}   precision: {dict(mha_prec)}")
    for nm, lt, raw in mha_examples:
        print(f"      ({lt}) {nm[:74]}"); [print(f"          {r[:86]}") for r in raw]
    check2 = len(mha_nodes) >= 90 and mha_prec.get("FP8", 0) == 0

    # CHECK 3 rewritten 2026-07-31. The old condition was
    #     len(rope_fp32) > 0 and prec_any["FP32"] > 100
    # and it reported False on BOTH architectures for the engine it exists to validate.
    # Two reasons, both about what this metadata can actually tell you:
    #
    #  1. ">100 FP32 layers" describes the fp32-TRUNK recipe. The shipped engine BAKES
    #     RoPE's cos/sin as fp32 constants, so there is no fp32 trunk — only the two
    #     runtime Fourier chains (timestep_features, seconds_total) stay fp32, which is
    #     6 FP32 layers on sm_120 and 16 on sm_90. Neither will ever exceed 100.
    #
    #  2. The per-layer "FP32" here is the dtype at the FUSED BLOB BOUNDARY, not the
    #     precision of the arithmetic inside it, and Myelin fuses those boundaries
    #     differently per arch. Measured on the same ONNX and recipe:
    #         sm_120  __myl_MaxMinDivReshMulSinCastCosCastConc   I/O ['Float','BFloat16']
    #         sm_90   __myl_CastMaxMinDivReshMulCosSinCastCastConc I/O ['BFloat16','BFloat16']
    #     Same engine identity, same measured fidelity, opposite verdicts from a
    #     name-and-boundary check. So "is the island fp32 inside" is NOT decidable here.
    #
    # What IS decidable and does matter: RoPE must be BAKED, i.e. no runtime cos/sin
    # beyond those two Fourier chains. If the bake silently failed, RoPE would be
    # computed at runtime in reduced precision and drift at long sequence — the bug this
    # recipe exists to avoid. Fidelity itself belongs to a numerical test, not an
    # inspector: sweep velocity-cos vs the fp32 engine across t (calibration and RoPE
    # damage both show up at t~1.0, and are invisible mid-schedule).
    print(f"\n{'#'*80}\nCHECK 3 — RoPE BAKED (no runtime cos/sin beyond the 2 Fourier chains)?")
    print(f"  cos/sin-named layers: {len(rope_seen)} (expect <= {MAX_RUNTIME_COSSIN}: "
          f"timestep_features + seconds_total)")
    for nm in rope_seen[:6]:
        print(f"      {'FP32' if nm in rope_fp32 else 'reduced-precision boundary'}: {nm}")
    print(f"  precision histogram across ALL layers: {dict(prec_any)}")
    print(f"  note: boundary dtypes above are informational — they differ by arch "
          f"(Myelin fusion), so they do not decide the island. Use a velocity-cos sweep.")
    check3 = len(rope_seen) <= MAX_RUNTIME_COSSIN and prec_any.get("FP32", 0) > 0

    out = dict(engine=path, label=label, total_layers=len(layers),
               gemm_by_precision=dict(gemm_prec), n_gemm=n_gemm, n_gemm_fp8=n_gemm_fp8,
               total_fp8_layers=fp8_layers, precision_any=dict(prec_any),
               n_mha_fused=len(mha_nodes), mha_precision=dict(mha_prec),
               n_rope_fp32=len(rope_fp32),
               check1_fp8_gemms=bool(check1),
               check2_native_mha_fused=bool(check2),
               n_runtime_cossin=len(rope_seen),
               check3_rope_baked=bool(check3))
    Path(path).with_suffix(".verify.json").write_text(json.dumps(out, indent=2))
    print(f"\n  SUMMARY: CHECK1(fp8 gemms>=170)={check1}  "
          f"CHECK2(mha fused, no fp8)={check2}  CHECK3(rope baked)={check3}")
    print(f"  wrote {Path(path).with_suffix('.verify.json')}")


if __name__ == "__main__":
    main()
