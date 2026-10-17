#!/usr/bin/env python3
"""Build the SAME-L / SAME-S autoencoder engines for the local GPU.

Consumer flow: download build-ready ONNX from HuggingFace, compile to TensorRT, verify. No model
checkpoints, no calibration data, no stable-audio-tools -- just tensorrt + torch + huggingface-hub.

Everything that needs a checkpoint has already been baked into the published ONNX: the decoders
arrive deterministic, with the limiter grafted in and its ceiling baked as a constant, and the fp8
encoders arrive with amax-calibrated activation scales. So this script only has to compile.

    python build/build_autoencoders.py                       # all 8, auto-detect arch
    python build/build_autoencoders.py --model same-l        # one model
    python build/build_autoencoders.py --precision fp8       # one precision
    python build/build_autoencoders.py --no-verify           # skip the post-build checks
    python build/build_autoencoders.py --list                # show targets and exit

WHAT GETS BUILT (8 engines). Each carries TWO optimization profiles:

    band 0   L=1..256    the chunked / low-VRAM mode      decoder ~509 MB scratch (SAME-L)
    band 1   L=1..4096   single-shot                      decoder ~8143 MB

TensorRT commits a context's scratch at create_execution_context(), sized from the profile ceiling
and *before any shape is bound* -- so a single 4096-ceiling engine reserves the full amount whether
it decodes five seconds or six minutes. Two bands let one file serve both a low-memory deployment
and a fast single-shot one. ⚠ The saving only materialises with USER_MANAGED contexts sized by
get_device_memory_size_for_profile_v2(band): device_memory_size_v2 is the MAX across profiles, so a
DEFAULT context reserves the wide band and the benefit silently disappears. The runtime's
TRTRunner(..., profile=N) does this correctly.

`opt` is a separate lever from the ceiling and costs no memory -- TRT tunes tactic selection around
it. Measured across L=32..4096, the naive default of min(1292, ceiling) costs 17-23% at short L on
the wide band while buying nothing at long L. The values below are the measured optima.

ARCH. TensorRT bakes the compute capability into the engine, so run this ON the target GPU. The
arch you build on is the arch the engine runs on.
"""

import argparse
import os
import sys
import time
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD_DIR))
from _arch import detect_arch, arch_dir as _arch_dir  # noqa: E402

HF_REPO = "stabilityai/stable-audio-3-optimized"
SPL = 4096            # samples per latent

# ── targets ────────────────────────────────────────────────────────────────────────────────────
# recipe "samel": STRONGLY_TYPED + the diff_attn_swa plugin. fp8 rides in the graph's own QDQ
#                 nodes, so the FP8 builder flag must NOT be set (strongly-typed forbids it).
# recipe "sames": weakly-typed EXPLICIT_BATCH + BF16. Here the FP8 flag IS required for the fp8
#                 tiers -- it is what lets TRT honour the in-graph QDQ.
TARGETS = {
    "same-l/dec_fp16": dict(model="same-l", kind="dec", recipe="samel", fp8=False,
                            onnx="onnx/same-l/dec_fp16_chunkable_limiter.onnx",
                            out="same-l/dec_fp16_chunkable_limiter.trt",
                            bands=[256, 4096], opts=[256, 1024]),
    "same-l/enc_fp16": dict(model="same-l", kind="enc", recipe="samel", fp8=False,
                            onnx="onnx/same-l/enc_dynamic_triton_swa.onnx",
                            out="same-l/enc_fp16_chunkable.trt",
                            bands=[256, 4096], opts=[256, 256]),
    "same-l/dec_fp8":  dict(model="same-l", kind="dec", recipe="samel", fp8=False,
                            onnx="onnx/same-l/dec_fp8_chunkable_limiter.onnx",
                            out="same-l/dec_fp8_chunkable_limiter.trt",
                            bands=[256, 4096], opts=[256, 1024]),
    "same-l/enc_fp8":  dict(model="same-l", kind="enc", recipe="samel", fp8=False,
                            onnx="onnx/same-l/enc_fp8.onnx",
                            out="same-l/enc_fp8_chunkable.trt",
                            bands=[256, 4096], opts=[256, 256]),
    "same-s/dec_bf16": dict(model="same-s", kind="dec", recipe="sames", fp8=False,
                            onnx="onnx/same-s/dec_bf16_chunkable_limiter.onnx",
                            out="same-s/dec_bf16_chunkable_limiter.trt",
                            bands=[256, 4096], opts=[256, 512]),
    "same-s/enc_bf16": dict(model="same-s", kind="enc", recipe="sames", fp8=False,
                            onnx="onnx/same-s/enc_dynamic_bf16.onnx",
                            out="same-s/enc_bf16_chunkable.trt",
                            bands=[256, 4096], opts=[256, 256]),
    "same-s/dec_fp8":  dict(model="same-s", kind="dec", recipe="sames", fp8=True,
                            onnx="onnx/same-s/dec_fp8_chunkable_limiter.onnx",
                            out="same-s/dec_fp8_chunkable_limiter.trt",
                            bands=[256, 4096], opts=[256, 512]),
    "same-s/enc_fp8":  dict(model="same-s", kind="enc", recipe="sames", fp8=True,
                            onnx="onnx/same-s/enc_fp8.onnx",
                            out="same-s/enc_fp8_chunkable.trt",
                            bands=[256, 4096], opts=[256, 256]),
}
LIMITER_CEILING = 0.977      # -0.2021 dBFS, baked into every published decoder


def _fetch(rel: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, rel)


def build_one(key: str, spec: dict, out_root: Path, workspace_gb: int = 16) -> Path:
    import tensorrt as trt
    out = out_root / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n━━━ {key} → {spec['out']} ━━━", flush=True)
    print(f"  onnx: {spec['onnx']}", flush=True)
    src = _fetch(spec["onnx"])

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    flags = 0
    if spec["recipe"] == "samel":
        # AOT, not JIT: a JIT plugin runs Python inside the enqueue path, is not
        # stream-capturable (so the engine cannot go inside a CUDA graph), and was measured to
        # corrupt the heap on repeated enqueues in a multi-profile engine where AOT was bit-exact.
        os.environ.setdefault("SA3_SWA_PLUGIN", "aot")
        sys.path.insert(0, str(BUILD_DIR.parent / "scripts"))
        import diff_attn_nocast_plugin  # noqa: F401  registers samel::diff_attn_swa
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)

    builder = trt.Builder(logger)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(src):
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}", file=sys.stderr)
        raise SystemExit(f"ONNX parse failed for {key}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if spec["recipe"] == "sames":
        cfg.set_flag(trt.BuilderFlag.BF16)
        if spec["fp8"]:
            cfg.set_flag(trt.BuilderFlag.FP8)
    tname = "audio" if spec["kind"] == "enc" else "latent"
    for hi, opt in zip(spec["bands"], spec["opts"]):
        prof = builder.create_optimization_profile()
        if spec["kind"] == "enc":
            prof.set_shape(tname, (1, 2, SPL), (1, 2, opt * SPL), (1, 2, hi * SPL))
        else:
            prof.set_shape(tname, (1, 256, 1), (1, 256, opt), (1, 256, hi))
        cfg.add_optimization_profile(prof)
    print("  profiles: " + " | ".join(f"L=1..{hi} opt={o}"
                                      for hi, o in zip(spec["bands"], spec["opts"])), flush=True)
    print(f"  recipe:   {'STRONGLY_TYPED + AOT plugin' if spec['recipe']=='samel' else 'EXPLICIT_BATCH + BF16'}"
          f"{' + FP8' if spec['fp8'] else ''}", flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise SystemExit(f"BUILD FAILED for {key}")
    out.write_bytes(memoryview(ser))
    eng = trt.Runtime(logger).deserialize_cuda_engine(out.read_bytes())
    print(f"  built {out.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
    div = SPL if spec["kind"] == "enc" else 1
    for i in range(eng.num_optimization_profiles):
        lo, o, hi = eng.get_tensor_profile_shape(tname, i)
        print(f"    band {i}: L={lo[-1]//div}..{hi[-1]//div} opt={o[-1]//div}  "
              f"scratch {eng.get_device_memory_size_for_profile_v2(i)/1e6:>7.1f} MB", flush=True)
    return out


def _test_signal(n_lat: int):
    """A harmonic, envelope-shaped stereo signal at a realistic level.

    ⚠ This is a STRUCTURAL smoke test -- does the engine produce finite, correctly-shaped,
    non-silent audio and hold the limiter -- NOT a quality measurement. Autoencoder quality must
    be measured on real music; synthetic noise in particular decodes past the int16 rail and
    fabricates plausible-looking numbers.
    """
    import numpy as np
    t = np.arange(n_lat * SPL, dtype=np.float64) / 44100.0
    sig = sum(np.sin(2*np.pi*f*t) / (i + 1) for i, f in enumerate((110., 220., 330., 550., 880.)))
    env = 0.5 + 0.5 * np.sin(2*np.pi*0.7*t)
    x = (sig / np.abs(sig).max() * env * 0.7).astype(np.float32)
    return np.stack([x, np.roll(x, 977)]).copy()


def verify(built: dict, out_root: Path) -> bool:
    """Per-engine structural checks, plus the one cross-check that needs no reference: the two
    bands of the same engine must agree with each other."""
    import numpy as np, torch, tensorrt as trt
    sys.path.insert(0, str(BUILD_DIR.parent / "scripts"))
    import sa3_trt_core as canon
    canon._import_heavy()
    print("\n━━━ verify ━━━", flush=True)
    ok = True

    def run(path, prof, tensor, data):
        """Returns (array, is_pcm). is_pcm marks int32 output scaled to int16 range -- these
        decoders bake the clip+scale+transpose tail, so the caller must divide by 32767. Deciding
        that from the returned array's dtype does not work: it has already been cast to float."""
        r = canon.TRTRunner(str(path), None, True, prof)
        need = r.engine.get_device_memory_size_for_profile_v2(prof)
        buf = torch.empty(need, dtype=torch.uint8, device="cuda")
        r.context.set_device_memory(buf.data_ptr(), need)
        if tensor == "audio":
            return canon.encoder_encode(r, data).float().cpu().numpy(), False
        out = canon.decoder_decode(r, data.to(r.in_dtype["latent"]))
        return out.float().cpu().numpy(), ("pcm" in r.out_dtype)

    for key, spec in built.items():
        p = out_root / spec["out"]
        L = 128                                             # fits band 0 and band 1 alike
        try:
            if spec["kind"] == "enc":
                x = torch.from_numpy(_test_signal(L)).unsqueeze(0).cuda()
                a, _ = run(p, 0, "audio", x)
                b, _ = run(p, 1, "audio", x)
                fin = np.isfinite(a).all() and np.isfinite(b).all()
                shape_ok = a.shape[-1] == L
                cs = float((a.ravel() @ b.ravel()) /
                           (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                good = fin and shape_ok and cs > 0.99 and np.abs(a).max() > 1e-3
                print(f"  {spec['out']:>40s}  latents{a.shape[-1]:>5d}  band0/band1 cos {cs:.6f}"
                      f"  {'OK' if good else 'FAIL'}", flush=True)
            else:
                enc_key = key.replace("dec", "enc")
                ep = out_root / TARGETS[enc_key]["out"]
                if not ep.exists():
                    # The decoder check is a round trip, so it needs the matching encoder.
                    # Building with --kind dec alone is legitimate (the encoder may already
                    # be deployed elsewhere), so say what is missing instead of reporting a
                    # failure the engine is not responsible for. No synthetic latent stands
                    # in here: a fabricated signal would not exercise the decoder the way
                    # real content does, and a green tick from one would be worse than none.
                    print(f"  {spec['out']:>40s}  SKIP  needs {TARGETS[enc_key]['out']} for "
                          f"the round trip — build without --kind, or --kind enc first",
                          flush=True)
                    continue
                x = torch.from_numpy(_test_signal(L)).unsqueeze(0).cuda()
                z, _ = run(ep, 0, "audio", x)
                zt = torch.from_numpy(z).cuda()
                a, pcm_a = run(p, 0, "latent", zt)
                b, pcm_b = run(p, 1, "latent", zt)
                fa = np.clip(a[0] / 32767.0, -1, 1) if pcm_a else a[0]
                fb = np.clip(b[0] / 32767.0, -1, 1) if pcm_b else b[0]
                pk = float(np.abs(fa).max())
                fin = np.isfinite(fa).all() and np.isfinite(fb).all()
                cs = float((fa.ravel() @ fb.ravel()) /
                           (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-30))
                # the limiter is baked: output must never exceed the ceiling (+1 LSB slack)
                lim_ok = pk <= LIMITER_CEILING + 1.0 / 32767
                good = fin and cs > 0.99 and pk > 1e-3 and lim_ok
                print(f"  {spec['out']:>40s}  peak {20*np.log10(max(pk,1e-9)):>+7.2f} dBFS"
                      f"  band0/band1 cos {cs:.6f}  limiter {'held' if lim_ok else 'EXCEEDED'}"
                      f"  {'OK' if good else 'FAIL'}", flush=True)
            ok &= bool(good)
        except Exception as e:
            print(f"  {spec['out']:>40s}  FAIL  {type(e).__name__}: {str(e)[:90]}", flush=True)
            ok = False
        torch.cuda.empty_cache()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("same-l", "same-s"), help="build only this model")
    ap.add_argument("--precision", choices=("fp16", "bf16", "fp8"), help="build only this precision")
    ap.add_argument("--kind", choices=("enc", "dec"), help="build only encoders or only decoders")
    ap.add_argument("--out", default=None, help="output root (default ../models/<arch>/)")
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force", action="store_true", help="rebuild engines that already exist")
    ap.add_argument("--list", action="store_true", help="list targets and exit")
    a = ap.parse_args()

    sel = {k: v for k, v in TARGETS.items()
           if (not a.model or v["model"] == a.model)
           and (not a.kind or v["kind"] == a.kind)
           and (not a.precision or a.precision in k)}
    if not sel:
        print("no targets match those filters", file=sys.stderr); return 2
    arch = detect_arch()
    out_root = Path(a.out) if a.out else Path(_arch_dir(arch))
    print(f"  arch {arch}   →  {out_root}")
    if a.list:
        for k, v in sel.items():
            print(f"    {k:>16s}  {v['onnx']:>46s}  →  {v['out']}")
        return 0
    todo = {}
    for k, v in sel.items():
        p = out_root / v["out"]
        if p.exists() and not a.force:
            print(f"  skip {v['out']} (exists; --force to rebuild)")
        else:
            todo[k] = v
    for k, v in todo.items():
        build_one(k, v, out_root, a.workspace_gb)
    if a.verify:
        # verify everything selected, including engines that were already present
        if not verify(sel, out_root):
            print("\n  VERIFICATION FAILED", file=sys.stderr); return 1
        print("\n  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
