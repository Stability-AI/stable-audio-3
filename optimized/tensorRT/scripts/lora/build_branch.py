"""Build the per-step-controllable LoRA engine: a runtime low-rank branch on the 182 linears.

    y = W0.x * (1 + pout)  +  Bp . ( srow * ( A . x ) )

pout is the per-output rescale that DoRA needs. TFLite's fold sets pchan_out = c-1 with
c = magnitude/||W0+dW||, and scales BOTH pout and Bp linearly by strength -- which makes the knob
exactly the linear fade  W(s) = W0 + s*(W_full - W0), exact at s=0 and s=1. Without pout the
branch silently computes plain LoRA for a DoRA checkpoint (measured: it misses 32% of the
adapter's effect at 1% magnitude drift, 86% at 5%, and no strength value recovers it).

A[1,R,in] and Bt[1,R,out] are NETWORK INPUTS (activations-as-weights), one pair per linear, so any
adapter is served without a rebuild and swaps are microseconds. srow[R] is a per-rank-BLOCK
strength vector: a stack concatenates along R, so scaling block k scales adapter k -- that is
per-step, per-adapter strength for the price of an R-element write, with the graph replayed
unchanged.

Exact for lora / lora-xs (their contribution is linear in strength). For dora/bora the
renormalisation is not linear in strength, so srow is an approximation for those -- the exact
route is to re-fold operands per step (not built here, by choice).

Design notes paid for in prototypes:
  * per-layer inputs, NOT one grouped tensor sliced per layer: a dynamic slice costs a real copy
    plus a fusion-blocking shape chain (measured +194% vs +16%).
  * Bt is stored RANK-MAJOR [R,out] and consumed with op NONE; feeding a column slice of an
    [out,R] buffer silently computes the wrong branch (fp16 hides it).
"""
import argparse, json, os, sys, time
from pathlib import Path
import tensorrt as trt

ROOT = Path(__file__).resolve().parent.parent
from branch_runtime import ADAPTER_PFX
from targets import (classify, TARGETS, LOCAL_TARGETS,
                     SECONDS_TARGET, EXTRA_TARGETS, CONV_TARGETS)

# ---------------------------------------------------------------------------
# Presets, provenance and preflight.
#
# The point of a preset is that a user with a fresh GPU types one short command and
# everything else -- which ONNX, where it comes from, how many targets to expect, what the
# rank ceiling is -- is already decided and checked. Everything stays overridable.
# ---------------------------------------------------------------------------
HF_REPO = "stabilityai/stable-audio-3-optimized"

PRESETS = {
    # sa3-medium: the public, shipping LoRA engine.
    "sa3-m": dict(
        onnx="sa3-m/dit_fp16.onnx",
        # the HF repo renamed fp16mixed -> fp16 (2026-08); older clones still have the old
        # name on disk. ⚠ They are NOT the same graph: fp16mixed runs softmax in an fp32
        # island, fp16 runs the attention core in fp16. We build from the current one.
        onnx_legacy="sa3-m/dit_fp16mixed.onnx",
        hf_files=("onnx/sa3-m/dit_fp16.onnx", "onnx/sa3-m/dit_fp16.onnx.data"),
        engine="engines/sa3-m/dit_fp16_lora.trt",
        map="lora/branch_map_medium_lora.json",
        targets="all", rank_max=512, rank_opt=32,
        expect_targets=229, expect_layers=24,
        blurb="SA3-medium (24 layers) -- the shipping LoRA engine, 229 targets",
    ),
}


def say(stage, msg, **kw):
    """One progress line. Stages are numbered so a long build reads as a checklist."""
    print(f"[{stage}] {msg}", flush=True, **kw)


def die(msg, fix=None):
    print(f"\nERROR: {msg}", file=sys.stderr)
    if fix:
        print(f"\n  how to fix:\n    {fix}\n", file=sys.stderr)
    sys.exit(2)


def gpu_info():
    """(name, 'sm_90', total_GiB) for device 0, or (None, None, None) if we cannot tell."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if not out:
            return (None, None, None)
        name, cc, mem = (f.strip() for f in out[0].split(","))
        major, minor = cc.split(".")
        return (name, f"sm_{major}{minor}", float(mem) / 1024.0)
    except Exception:
        return (None, None, None)


def sha256_head(path, nbytes=64 << 20):
    """Hash of the first nbytes. A full hash of a 2.7 GB .data costs ~6 s and buys nothing
    we need -- this is provenance, not integrity, and the size is recorded alongside."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def resolve_onnx(spec, explicit=None, allow_download=True):
    """Find the ONNX. Order: --onnx, the preset path under the resolved ONNX dir, the
    legacy name (with a loud note), then HuggingFace."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            die(f"--onnx {p} does not exist")
        return p, "explicit path"

    base = onnx_base_dir()
    p = base / spec["onnx"]
    if p.exists():
        return p, f"found in {base}"

    legacy = spec.get("onnx_legacy")
    if legacy and (base / legacy).exists():
        lp = base / legacy
        say("onnx", f"⚠ only the LEGACY name is present: {lp.name}")
        say("onnx", "  that is the older fp16mixed graph (fp32 softmax island), NOT the")
        say("onnx", f"  current {Path(spec['onnx']).name} (fp16 attention core).")
        say("onnx", "  Building from it anyway. Pass --onnx explicitly to silence this,")
        say("onnx", "  or delete nothing -- just fetch the current file with --download.")
        return lp, "legacy name (older tier)"

    if not allow_download or not spec.get("hf_files"):
        die(f"no ONNX found at {p}",
            f"fetch it:  python {Path(__file__).name} --model <m> --download")
    return download_onnx(spec, base), "downloaded from HuggingFace"


def onnx_base_dir():
    """$SA3_ONNX_DIR, else the HF model repo checked out beside this one, else ./onnx."""
    import os
    env = os.environ.get("SA3_ONNX_DIR")
    if env:
        return Path(env)
    sibling = ROOT.parent / "stable-audio-3-optimized" / "onnx"
    if sibling.is_dir():
        return sibling
    return ROOT / "onnx"


def download_onnx(spec, base):
    """Pull the ONNX + its external data from HF into `base`."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("huggingface_hub is not installed and the ONNX is missing",
            "pip install huggingface_hub")
    base.mkdir(parents=True, exist_ok=True)
    got = None
    for rel in spec["hf_files"]:
        say("onnx", f"downloading {HF_REPO}/{rel} ...")
        f = hf_hub_download(repo_id=HF_REPO, filename=rel,
                            local_dir=str(base.parent) if base.name == "onnx" else str(base))
        if rel.endswith(".onnx"):
            got = Path(f)
    if got is None:
        die("download finished but no .onnx file came back")
    say("onnx", f"ready: {got}")
    return got


def provenance(args, onnx_path, n_targets, build_s):
    """Everything needed to answer 'what is this engine and can I rebuild it?'.

    The six medium maps all went stale because the only thing recorded was args.engine,
    verbatim, at build time -- a rename downstream rotted it silently.
    """
    import datetime, os
    name, arch, _vram = gpu_info()
    data = Path(str(onnx_path) + ".data")
    return {
        "built_utc": datetime.datetime.now(datetime.timezone.utc)
                             .replace(microsecond=0).isoformat(),
        "built_by": "lora/build_branch.py",
        "build_seconds": round(build_s, 1),
        "onnx": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256_head64m": sha256_head(onnx_path),
        "onnx_data_bytes": data.stat().st_size if data.exists() else None,
        "trt_version": trt.__version__,
        "gpu": name, "arch": arch,
        "targets_mode": args.targets, "n_targets": n_targets,
        "rank_max": args.rank_max, "rank_opt": args.rank_opt,
        "host": os.uname().nodename,
    }


def check_maps(paths):
    """--check: every map's engine pointer resolves, and its layer table matches the
    engine it names. Exit non-zero on any failure so CI can gate on it."""
    import mmap
    bad = 0
    for mp in paths:
        mp = Path(mp)
        if not mp.exists():
            print(f"  MISSING MAP  {mp}"); bad += 1; continue
        d = json.loads(mp.read_text())
        eng, n = d.get("engine"), len(d.get("layers", {}))
        cand = [mp.parent / eng, ROOT / eng, Path(eng)] if eng else []
        hit = next((c for c in cand if c.exists()), None)
        if hit is None:
            print(f"  STALE        {mp.name}: engine {eng!r} does not resolve"); bad += 1; continue
        prov = "prov" if "provenance" in d else "NO-PROV"
        try:
            logger = trt.Logger(trt.Logger.ERROR)
            with open(hit, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                e = trt.Runtime(logger).deserialize_cuda_engine(mm)
            if e is None:
                print(f"  UNREADABLE   {mp.name}: {hit.name}"); bad += 1; continue
            io = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
            n_a = sum(1 for x in io if x.startswith("loraA::"))
            ok = (n_a == n)
            miss = [r["A"] for r in d["layers"].values() if r["A"] not in set(io)]
            if miss:
                ok = False
            # The rank ceiling is what distinguishes two maps with the same layer count
            # (229 @ r128 vs 229 @ r512 name different engines). Counts alone would pass
            # a mispairing; the profile max does not.
            eng_rank = None
            if not miss and n_a:
                a0 = next(x for x in io if x.startswith("loraA::"))
                eng_rank = int(e.get_tensor_profile_shape(a0, 0)[2][1])
                if eng_rank != d.get("rank_max"):
                    ok = False
            print(f"  {'OK   ' if ok else 'MISMATCH'}     {mp.name}: {n} layers, engine has "
                  f"{n_a} loraA inputs, rank_max {d.get('rank_max')}"
                  f"{'' if eng_rank is None or eng_rank == d.get('rank_max') else f' but ENGINE IS {eng_rank}'}"
                  f" [{prov}] -> {hit.name}")
            if miss:
                print(f"               first missing operand: {miss[0]} (+{len(miss)-1} more)")
            bad += (0 if ok else 1)
            del e
        except Exception as ex:
            print(f"  ERROR        {mp.name}: {ex}"); bad += 1
    print(f"\n{'all maps OK' if not bad else f'{bad} problem(s)'}")
    return 1 if bad else 0



def main():
    ap = argparse.ArgumentParser(
        prog="build_branch.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a hot-swappable LoRA/DoRA TensorRT engine for the SA3 DiT.",
        epilog="""
typical use
-----------
  # build the shipping SA3-medium LoRA engine for THIS GPU (downloads the ONNX if needed)
  python lora/build_branch.py --model sa3-m --download

  # check that every branch map still points at an engine that exists and matches
  python lora/build_branch.py --check

The engine is specific to the GPU architecture it is built on -- run this on the GPU you
intend to serve from. Everything the preset decides can be overridden with the flags below.
""")
    ap.add_argument("--model", choices=sorted(PRESETS), default=None,
                    help="preset that fills in onnx/engine/map/targets/rank (recommended)")
    ap.add_argument("--check", nargs="*", metavar="MAP", default=None,
                    help="validate branch maps instead of building; no args = all of them")
    ap.add_argument("--download", action="store_true",
                    help="fetch the ONNX from HuggingFace if it is not on disk")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output engine if it already exists")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--rank-max", type=int, default=None)
    ap.add_argument("--rank-opt", type=int, default=None)
    ap.add_argument("--workspace-gb", type=int, default=48)
    # ⚠ was hardcoded to lora/branch_map.json — building a second model's engine would
    # silently overwrite the first model's layer map, and the runtime reads it by name.
    # (the flag itself is declared below, after --targets, so the preset can fill it in)
    # core = 169: the 7 block linears per layer + the seconds_total embedder, which is what
    #        every adapter trained so far actually targets.
    # full = 228: every adapted tensor INSIDE the DiT -- 9 per layer (adding to_local_embed),
    #        the to_*_embed / global_cond_embedder / project_in/out stack, and both convs.
    #        Deliberately excludes the seconds_total embedder, which lives in the conditioner.
    # all  = 229: full + the seconds_total embedder, i.e. every tensor any sa3-medium adapter
    #        has ever targeted. Measured to cost nothing over 168 targets in time or VRAM
    #        (+7 MB engine, +5 MB operands at r16), so coverage here is free.
    ap.add_argument("--targets", choices=("core", "full", "all"), default=None)
    ap.add_argument("--map", dest="map", default=None)
    args = ap.parse_args()

    # ---- --check: validate maps and exit -----------------------------------------
    if args.check is not None:
        maps = args.check or sorted(str(q) for q in (ROOT / "lora").glob("branch_map*.json"))
        print(f"checking {len(maps)} branch map(s)\n")
        sys.exit(check_maps(maps))

    # ---- resolve the preset ------------------------------------------------------
    spec = PRESETS.get(args.model, {})
    if not args.model and not (args.onnx and args.engine and args.map):
        die("say which model to build",
            "python lora/build_branch.py --model sa3-m --download\n"
            "    (or pass --onnx/--engine/--map explicitly)")
    if args.model:
        say("model", spec["blurb"])
    for k, dflt in (("targets", "core"), ("rank_max", 128), ("rank_opt", 32)):
        if getattr(args, k) is None:
            setattr(args, k, spec.get(k, dflt))
    if args.engine is None:
        args.engine = str(ROOT / spec["engine"])
    if args.map is None:
        args.map = str(ROOT / spec["map"])

    onnx_path, how = resolve_onnx(spec, args.onnx, allow_download=args.download) \
        if spec else (Path(args.onnx), "explicit path")
    args.onnx = str(onnx_path)

    # ---- preflight ---------------------------------------------------------------
    name, arch, vram = gpu_info()
    say("gpu", f"{name or 'unknown'} ({arch or '?'}), {vram:.0f} GiB" if name
        else "no GPU detected via nvidia-smi -- TensorRT needs one to build")
    say("trt", f"TensorRT {trt.__version__}")
    say("onnx", f"{onnx_path}  ({onnx_path.stat().st_size/1e6:.0f} MB proto, {how})")
    data = Path(str(onnx_path) + ".data")
    if data.exists():
        say("onnx", f"external data {data.name}  ({data.stat().st_size/2**30:.2f} GiB)")
    elif onnx_path.stat().st_size < 50e6:
        die(f"{onnx_path.name} is small and {data.name} is missing -- the weights are in "
            "a separate file that must sit next to the proto",
            "re-run with --download, or copy the .onnx.data file alongside it")
    out = Path(args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        die(f"{out} already exists", "pass --force to overwrite, or --engine <other path>")
    say("plan", f"targets={args.targets}  rank 1..{args.rank_max} (opt {args.rank_opt})")
    say("plan", f"engine -> {out}")
    say("plan", f"map    -> {args.map}")
    if arch:
        say("plan", f"this engine will only run on {arch}; rebuild on each architecture")

    from trt_build import build_engine, profile_for

    added = {}

    def surgery(net):
        prod = {}
        for i in range(net.num_layers):
            L = net.get_layer(i)
            for o in range(L.num_outputs):
                prod[L.get_output(o).name] = (i, L)
        per_layer = dict(TARGETS)
        if args.targets == "core":
            extra = dict(SECONDS_TARGET)
        else:
            per_layer.update(LOCAL_TARGETS)
            extra = {**EXTRA_TARGETS, **CONV_TARGETS}
            if args.targets == "all":
                extra.update(SECONDS_TARGET)

        targets, n_block, got = [], 0, set()
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if L.type not in (trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION):
                continue
            nm, suf = (classify(L.name, per_layer)
                       if L.type == trt.LayerType.MATRIX_MULTIPLY else (None, None))
            if nm is not None:
                # block linears share one prefix, so the adapter key is derived
                targets.append((i, L, nm, suf, ADAPTER_PFX + suf)); n_block += 1
            elif L.name in extra:
                slug, bases_suf, akey = extra[L.name]
                targets.append((i, L, f"lora::{slug}", bases_suf, akey))
                got.add(slug)

        # Derived, not hardcoded, so a graph change surfaces as a count mismatch here
        # = 182, medium is 24 x 7 = 168. Asserting the large count is what stopped this
        # building for any other model. What must hold is that every layer contributes
        # the full set of per-layer targets -- a partial layer means the name match drifted.
        n_t = len(per_layer)
        assert n_block and n_block % n_t == 0, (
            f"found {n_block} block linears, not a multiple of the {n_t} per-layer targets "
            f"-- the ONNX node naming probably differs from what classify() expects")
        # An unmatched extra is silent otherwise: the engine builds, the adapter loads, and
        # that layer just never gets its branch.
        want = {v[0] for v in extra.values()}
        assert got == want, f"extra targets missing from the graph: {sorted(want - got)}"
        print(f"[build] {n_block} block linears = {n_block//n_t} layers x {n_t} targets"
              f"  + {len(targets)-n_block} non-block ({args.targets})")

        srow = net.add_input("lora_srow", trt.float16, (1, 1, -1))     # [1,1,R] broadcasts over S
        outputs = {net.get_output(i).name for i in range(net.num_outputs)}

        for i, L, nm, suf, akey in targets:
            x = L.get_input(0)
            T = L.get_output(0)
            key = nm[len("lora::"):]

            # Three activation layouts reach the same branch. The block linears are already
            # [1,S,D]. The seconds_total embedder is rank-2 [1,256]->[1,768] and sits in the
            # fp32 sinusoidal island. The two convs are NCHW [1,C,L,1] 1x1 kernels, i.e. a
            # per-position channel mix. Each is converted to [1,S,D] for the branch and back
            # afterwards, so the runtime's buffer writes and the optimisation profile stay
            # uniform at fp16 [1,R,*] for EVERY target -- no per-kind operand layout.
            kind = ("conv" if L.type == trt.LayerType.CONVOLUTION else
                    "flat" if len(x.shape) == 2 else "seq")
            in_d = int(x.shape[1] if kind == "conv" else x.shape[-1])
            out_d = int(T.shape[1] if kind == "conv" else T.shape[-1])

            def to_seq(t, d, tag, _k=key, _kind=kind):
                sh = net.add_shuffle(t); sh.name = f"lora/seq/{tag}/{_k}"
                if _kind == "conv":
                    sh.first_transpose = trt.Permutation([0, 2, 3, 1])   # NCHW -> N,H,W,C
                    sh.reshape_dims = (1, -1, d)
                else:
                    sh.reshape_dims = (1, 1, d)
                return sh.get_output(0)

            xb, Tb = ((x, T) if kind == "seq"
                      else (to_seq(x, in_d, "x"), to_seq(T, out_d, "y")))

            A = net.add_input(f"loraA::{key}", trt.float16, (1, -1, in_d))
            Bt = net.add_input(f"loraB::{key}", trt.float16, (1, -1, out_d))
            P = net.add_input(f"loraP::{key}", trt.float16, (1, 1, out_d))

            def match(t, tag, _key=key, _dt=Tb.dtype):
                """Operands enter as fp16; a target in an fp32 island needs them cast, or TRT
                builds a mismatched elementwise (it warns, then computes in the wrong type)."""
                if t.dtype == _dt:
                    return t
                c = net.add_cast(t, _dt); c.name = f"lora/cast/{tag}/{_key}"
                return c.get_output(0)

            A, Bt, P, sr = (match(A, "A"), match(Bt, "B"),
                            match(P, "P"), match(srow, "srow"))
            h = net.add_matrix_multiply(xb, trt.MatrixOperation.NONE,
                                        A, trt.MatrixOperation.TRANSPOSE)
            h.name = f"lora/down/{key}"
            hs = net.add_elementwise(h.get_output(0), sr, trt.ElementWiseOperation.PROD)
            hs.name = f"lora/strength/{key}"
            y = net.add_matrix_multiply(hs.get_output(0), trt.MatrixOperation.NONE,
                                        Bt, trt.MatrixOperation.NONE)
            y.name = f"lora/up/{key}"
            # per-output rescale: T + pout*T  (pout = 0 when no dora/bora is loaded)
            sc = net.add_elementwise(Tb, P, trt.ElementWiseOperation.PROD)
            sc.name = f"lora/pout/{key}"
            base_sc = net.add_elementwise(Tb, sc.get_output(0), trt.ElementWiseOperation.SUM)
            base_sc.name = f"lora/poutadd/{key}"
            add = net.add_elementwise(base_sc.get_output(0), y.get_output(0),
                                      trt.ElementWiseOperation.SUM)
            add.name = f"lora/add/{key}"
            new = add.get_output(0)
            if kind != "seq":                         # back to the shape consumers expect
                sb = net.add_shuffle(new); sb.name = f"lora/native/{key}"
                if kind == "conv":
                    sb.reshape_dims = (1, -1, 1, out_d)
                    sb.second_transpose = trt.Permutation([0, 3, 1, 2])  # N,H,W,C -> NCHW
                else:
                    sb.reshape_dims = (1, out_d)
                new = sb.get_output(0)
            # rewire every consumer of T (except our own add) onto the branched tensor
            for j in range(net.num_layers):
                C = net.get_layer(j)
                if C.name.startswith("lora/"):
                    continue
                for k in range(C.num_inputs):
                    if C.get_input(k) is not None and C.get_input(k).name == T.name:
                        C.set_input(k, new)
            assert T.name not in outputs, f"{L.name} output is a network output; rewire needed"
            added[nm] = {"in": in_d, "out": out_d, "ckpt_suffix": suf,
                         "adapter_key": akey,
                         "A": f"loraA::{key}", "B": f"loraB::{key}", "P": f"loraP::{key}"}
        print(f"[surgery] branch on {len(added)} linears "
              f"(+{3*len(added)+1} inputs, rank dynamic 1..{args.rank_max})", flush=True)

    def profiles(builder, cfg):
        for bs in (1,):
            for ropt in (args.rank_opt,):
                pr = builder.create_optimization_profile()
                for name, (lo, opt, hi) in profile_for(bs).items():
                    pr.set_shape(name, lo, opt, hi)
                pr.set_shape("lora_srow", (1, 1, 1), (1, 1, ropt), (1, 1, args.rank_max))
                for nm, rec in added.items():
                    pr.set_shape(rec["A"], (1, 1, rec["in"]), (1, ropt, rec["in"]),
                                 (1, args.rank_max, rec["in"]))
                    pr.set_shape(rec["B"], (1, 1, rec["out"]), (1, ropt, rec["out"]),
                                 (1, args.rank_max, rec["out"]))
                    pr.set_shape(rec["P"], (1, 1, rec["out"]), (1, 1, rec["out"]),
                                 (1, 1, rec["out"]))
                cfg.add_optimization_profile(pr)

    say("build", "compiling -- TensorRT tactic search takes a while (minutes, not seconds)")
    t0 = time.time()
    build_engine(args.onnx, args.engine, workspace_gb=args.workspace_gb, detailed=True,
                 batches=(1,), on_network=surgery, profile_hook=profiles)
    build_s = time.time() - t0

    # A preset knows how many targets it must end up with. Catching a shortfall here --
    # rather than at render time, as silently-unadapted layers -- is the whole point.
    want = spec.get("expect_targets")
    if want and len(added) != want:
        die(f"expected {want} targets for {args.model}, got {len(added)}",
            "the ONNX node naming has drifted; check classify() in lora/targets.py")

    # Store the engine path RELATIVE TO THE MAP, so moving the tree together keeps it valid,
    # and record enough provenance to rebuild this exact engine later.
    mp = Path(args.map)
    try:
        rel = os.path.relpath(Path(args.engine).resolve(), mp.resolve().parent)
    except ValueError:
        rel = str(args.engine)
    mp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"engine": rel, "rank_max": args.rank_max, "layers": added,
               "provenance": provenance(args, Path(args.onnx), len(added), build_s)},
              open(mp, "w"), indent=1)

    out = Path(args.engine)
    mins = build_s / 60.0
    print()
    say("done", f"{out}  ({out.stat().st_size/1e9:.2f} GB) in {mins:.1f} min")
    say("done", f"{len(added)} adapted layers, stack rank up to {args.rank_max}")
    say("done", f"map {mp}")
    print()
    print("  verify it:   python lora/build_branch.py --check")
    print("  use it:      python gradio/sa3_trt.py --dit medium --lora <adapter.safetensors>")


if __name__ == "__main__":
    main()
