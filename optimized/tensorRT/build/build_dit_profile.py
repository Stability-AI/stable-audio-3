#!/usr/bin/env python3
"""Build a SA3 DiT TRT engine with a custom dynamic-shape profile.

Differs from build_from_onnx.py only in that the optimization-profile `min`
(and optionally `opt`) can be overridden — used for short-form audio
work where the canonical min=256 (~24 s) is too high, and for tactic-selection
experiments where the canonical opt=1292 is much larger than typical L.

Usage:
    python build_dit_profile.py sa3-sm-music 32 dit_bf16_min32.trt
    python build_dit_profile.py sa3-m      128 dit_bf16_min128.trt
    python build_dit_profile.py sa3-sm-music 1 dit_bf16_opt64.trt --opt 64
    python build_dit_profile.py sa3-m      1 dit_bf16_opt324.trt --opt 324

Args:
    <onnx_target>     Subdir under onnx/ (sa3-sm-music | sa3-sm-sfx | sa3-m).
                       The .onnx (plus .data sidecar) must already exist locally
                       under <repo>/stable-audio-3-optimized/onnx/<target>/.
    <profile_min>     Minimum L (e.g., 32, 128). max stays at 4096.
    <output_name>     Filename to write under models/<arch>/<target>/.

Optional flags:
    --opt N           Optimization point on L axis (default 1292). TRT picks
                       kernels to run fastest at this shape.
    --precision P     "bf16" (default; matches the sa3-m-bf16 recipe, NOT the
                       canonical fp16-mixed one — this script builds from the raw
                       fp32 dit.onnx) or "fp32" (omits the BF16 flag entirely —
                       forces full FP32 execution; engine ~2x larger, build
                       slower).

Build flags mirror sa3-sm-music / sa3-m in build_from_onnx.py: BF16 +
EXPLICIT_BATCH, 16 GB workspace. With --precision fp32 the BF16 flag is
omitted; this lets us test whether BF16 rounding (e.g. in RoPE) is the
source of the long-clip silence artifact at T_lat=1292.
"""
import os
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent / "scripts"))

from _arch import arch_dir, repo_root  # noqa: E402


T5_TOKENS = 256
T5_HIDDEN_DIM = 768


# Maps the target name (also used as the onnx subdir + the output subdir under
# models/<arch>/) to its on-disk ONNX path. The medium target has a sidecar
# .data file; the ONNX parser handles it transparently as long as both files
# sit in the same directory.
TARGETS = {
    "sa3-sm-music": {
        "onnx_rel":  "sa3-sm-music/dit.onnx",
        "out_subdir": "sa3-sm-music",
    },
    "sa3-sm-sfx": {
        "onnx_rel":  "sa3-sm-sfx/dit.onnx",
        "out_subdir": "sa3-sm-sfx",
    },
    "sa3-m": {
        "onnx_rel":  "sa3-m/dit.onnx",   # dit.onnx.data co-located
        "out_subdir": "sa3-m",
    },
}


def _local_onnx_root() -> Path:
    """Resolve the local ONNX root the same way _arch.onnx_dir() does, but
    without auto-creating directories — we just want a path that exists.

    Canonical layout:  <github>/stable-audio-3-optimized/onnx/<subdir>/
    """
    override = os.environ.get("SA3_ONNX_DIR")
    if override:
        return Path(override)
    # repo_root() = .../stable-audio-3/optimized/tensorRT
    # HF sibling  = .../stable-audio-3-optimized
    github_root = Path(repo_root()).parent.parent.parent
    return github_root / "stable-audio-3-optimized" / "onnx"


def build_dit(target_name: str, profile_min: int, output_name: str,
              profile_opt: int = 1292, profile_max: int = 4096,
              precision: str = "bf16") -> str:
    if target_name not in TARGETS:
        sys.exit(f"unknown target {target_name!r}; valid: {list(TARGETS)}")
    if precision not in ("bf16", "fp32"):
        sys.exit(f"precision={precision!r} must be one of: bf16, fp32")
    if not (profile_min <= profile_opt <= profile_max):
        sys.exit(f"profile_opt={profile_opt} must satisfy "
                 f"min={profile_min} <= opt <= max={profile_max}")

    rec = TARGETS[target_name]
    onnx_path = _local_onnx_root() / rec["onnx_rel"]
    if not onnx_path.exists():
        sys.exit(f"ONNX not found at {onnx_path}")

    print(f"\n━━━ build_dit_profile: {target_name} min={profile_min} opt={profile_opt} "
          f"precision={precision} → {output_name} ━━━")
    print(f"  onnx: {onnx_path}", flush=True)
    sz_b = onnx_path.stat().st_size
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if sidecar.exists():
        sz_b += sidecar.stat().st_size
        print(f"  sidecar: {sidecar.name} ({sidecar.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"  onnx total size: {sz_b/1e9:.2f} GB", flush=True)

    # Build profile dict: x and local_add_cond get the dynamic L; scalars/T5 stay fixed.
    profile_shapes = {
        "x":              [(1, 256, profile_min),  (1, 256, profile_opt),  (1, 256, profile_max)],
        "t":              [(1,),                   (1,),                   (1,)],
        "t5_hidden":      [(1, T5_TOKENS, T5_HIDDEN_DIM)] * 3,
        "t5_mask":        [(1, T5_TOKENS)] * 3,
        "seconds_total":  [(1,)] * 3,
        "local_add_cond": [(1, 257, profile_min),  (1, 257, profile_opt),  (1, 257, profile_max)],
    }

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}", flush=True)
        sys.exit(2)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    if precision == "bf16":
        cfg.set_flag(trt.BuilderFlag.BF16)
    else:
        # Pure FP32: do NOT set BF16 (or FP16). We additionally set
        # OBEY_PRECISION_CONSTRAINTS so TRT can't silently downcast layers to
        # lower precision if it judges it "safe" — we want full FP32 to test
        # the hypothesis that BF16 rounding (e.g. in RoPE) causes the
        # long-clip silence artifact.
        try:
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        except AttributeError:
            pass

    profile = builder.create_optimization_profile()
    for input_name, (lo, opt, hi) in profile_shapes.items():
        profile.set_shape(input_name, lo, opt, hi)
    cfg.add_optimization_profile(profile)
    print(f"  profile: x/local_add_cond L=({profile_min}, {profile_opt}, 4096), "
          f"workspace 16 GB, precision={precision.upper()}", flush=True)

    print(f"  building...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print(f"  BUILD FAILED", flush=True)
        sys.exit(3)
    build_s = time.time() - t0
    print(f"  built in {build_s:.0f}s ({serialized.nbytes/1e6:.0f} MB)", flush=True)

    out_dir = Path(arch_dir()) / rec["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / output_name
    with open(target, "wb") as f:
        f.write(serialized)
    print(f"  wrote {target}", flush=True)
    print(f"  build_time_s={build_s:.1f}  size_bytes={target.stat().st_size}", flush=True)
    return str(target)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("target_name",
                    help="One of sa3-sm-music, sa3-sm-sfx, sa3-m")
    ap.add_argument("profile_min", type=int,
                    help="Minimum L for the optimization profile")
    ap.add_argument("output_name",
                    help="Filename to write under models/<arch>/<target>/")
    ap.add_argument("--opt", dest="profile_opt", type=int, default=1292,
                    help="L value for the profile's opt point (default 1292)")
    ap.add_argument("--max", dest="profile_max", type=int, default=4096,
                    help="Maximum L for the profile (default 4096). Set to "
                         "match min for a fully-static engine.")
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                    help="Builder precision: bf16 (default; matches the "
                         "sa3-m-bf16 recipe, not the canonical fp16-mixed one) "
                         "or fp32 (omits BF16 flag, forces FP32 throughout).")
    args = ap.parse_args()
    build_dit(args.target_name, args.profile_min, args.output_name,
              profile_opt=args.profile_opt, profile_max=args.profile_max,
              precision=args.precision)


if __name__ == "__main__":
    main()
