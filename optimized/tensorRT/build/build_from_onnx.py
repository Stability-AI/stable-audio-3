#!/usr/bin/env python3
"""Compile a TRT engine from a pre-built ONNX file pulled from HuggingFace.

This is the "consumer" build path: no stable-audio-tools, no model checkpoints,
no PyTorch source — just TensorRT + the public ONNX files from
stabilityai/stable-audio-3-optimized/onnx/.

To rebuild engines for a new GPU arch (sm_100, sm_120, ...) you run this on
that GPU and TRT bakes the arch into the engine.

Usage:
    python build_from_onnx.py t5gemma
    python build_from_onnx.py same-s-encoder
    python build_from_onnx.py same-s-decoder
    python build_from_onnx.py same-l-encoder
    python build_from_onnx.py same-l-decoder
    # SA3 DiT engines use a different builder (FP16-mixed recipe):
    python build_dit_fp16mixed.py --input <onnx> --engine <out.trt>
    python build_from_onnx.py all          # build everything for this arch
"""
import os
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
# diff_attn_nocast_plugin + triton_swa_v2 (for SAME-L plugin) live in ../scripts/
sys.path.insert(0, str(SCRIPTS_DIR.parent / "scripts"))

from _arch import detect_arch, arch_dir  # noqa: E402


HF_REPO = "stabilityai/stable-audio-3-optimized"
HF_ONNX_PREFIX = "onnx"  # files at HF_REPO/onnx/<engine_subdir>/<file>.onnx

T5_TOKENS = 256
T5_HIDDEN_DIM = 768
SAMPLES_PER_LATENT = 4096


# DiT optimization profile — shared by all 3 DiT variants since they have the
# same input shapes (cond bakedin, dynamic L in [1, 4096]).
# Min lowered from 256 → 1 after benchmarking showed TRT picks identical
# tactics across the [1, 4096] range at opt=1292; the extended lower bound
# unlocks sub-trained short-form output (~93 ms minimum) at zero perf cost.
# Audio quality below L=256 (~23.8 s) is undefined — the model was trained on
# L≥256 — but the engine runs.
_DIT_PROFILE = {
    "x":              [(1, 256, 1),     (1, 256, 1292),   (1, 256, 4096)],
    "t":              [(1,),            (1,),             (1,)],
    "t5_hidden":      [(1, T5_TOKENS, T5_HIDDEN_DIM)] * 3,
    "t5_mask":        [(1, T5_TOKENS)] * 3,
    "seconds_total":  [(1,)] * 3,
    "local_add_cond": [(1, 257, 1),     (1, 257, 1292),   (1, 257, 4096)],
}

# Per-engine recipe: where the ONNX lives on HF, where the .trt goes locally,
# what TRT builder flags to use, and the optimization profile shapes.
TARGETS = {
    "t5gemma": {
        "onnx_hf":     ["t5gemma/encoder.onnx"],
        # tokenizer.json ships bundled with the repo at scripts/tokenizer.json
        # (arch-agnostic), so we don't fetch it here anymore.
        "trt_local":    "t5gemma/t5gemma_fp16mixed.trt",
        "flags":        set(),  # STRONGLY_TYPED carries the FP16/FP32 dtype hints
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 8,
        "profile":      None,  # static shapes
        "plugin":       False,
    },
    "same-s-encoder": {
        "onnx_hf":      ["same-s/enc_dynamic_bf16.onnx"],
        "trt_local":    "same-s/enc_dynamic_bf16.trt",
        "flags":        {"BF16"},
        "network":      "EXPLICIT_BATCH",
        "workspace_gb": 16,
        "profile":      {"audio": [(1, 2, 32 * SAMPLES_PER_LATENT),
                                    (1, 2, 1292 * SAMPLES_PER_LATENT),
                                    (1, 2, 4096 * SAMPLES_PER_LATENT)]},
        "plugin":       False,
    },
    "same-s-decoder": {
        "onnx_hf":      ["same-s/dec_dynamic_bf16.onnx"],
        "trt_local":    "same-s/dec_dynamic_bf16.trt",
        "flags":        {"BF16"},
        "network":      "EXPLICIT_BATCH",
        "workspace_gb": 16,
        "profile":      {"latent": [(1, 256, 32), (1, 256, 1292), (1, 256, 4096)]},
        "plugin":       False,
    },
    "same-l-encoder": {
        "onnx_hf":      ["same-l/enc_dynamic_triton_swa.onnx"],
        "trt_local":    "same-l/enc_dynamic_triton_swa.trt",
        "flags":        set(),  # STRONGLY_TYPED carries dtype hints
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      {"audio": [(1, 2, 32 * SAMPLES_PER_LATENT),
                                    (1, 2, 1292 * SAMPLES_PER_LATENT),
                                    (1, 2, 4096 * SAMPLES_PER_LATENT)]},
        "plugin":       True,
    },
    "same-l-decoder": {
        "onnx_hf":      ["same-l/dec_dynamic_triton_swa.onnx"],
        "trt_local":    "same-l/dec_dynamic_triton_swa.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      {"latent": [(1, 256, 32), (1, 256, 1292), (1, 256, 4096)]},
        "plugin":       True,
    },
    # SA3 DiT engines: build from the pre-processed FP16-mixed ONNX hosted on
    # HF. The producer (build_dit_fp16mixed.py) does the FP32-island surgery
    # once and uploads the result; consumers just compile with STRONGLY_TYPED
    # (no onnx-graphsurgeon dependency).
    "sa3-sm-music": {
        "onnx_hf":      ["sa3-sm-music/dit_fp16mixed.onnx"],
        "trt_local":    "sa3-sm-music/dit_fp16mixed.trt",
        "flags":        set(),         # STRONGLY_TYPED + ONNX dtypes carry precision
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    "sa3-sm-sfx": {
        "onnx_hf":      ["sa3-sm-sfx/dit_fp16mixed.onnx"],
        "trt_local":    "sa3-sm-sfx/dit_fp16mixed.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    "sa3-m": {
        # 2.9 GB external-data sidecar travels alongside.
        "onnx_hf":      ["sa3-m/dit_fp16mixed.onnx", "sa3-m/dit_fp16mixed.onnx.data"],
        "trt_local":    "sa3-m/dit_fp16mixed.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    # SA3 medium DiT in bf16 — the medium SPEED DEFAULT (fp16-mixed above stays
    # selectable). Reuses the SAME raw fp32 dit.onnx as the fp32 variant, but
    # builds with BuilderFlag.BF16 + EXPLICIT_BATCH (NOT STRONGLY_TYPED): bf16
    # carries fp32's dynamic range, so the FP32-softmax islands that fp16-mixed
    # kept vanish and TRT 10.15's FMHA fuser fires (0 → 96 fused _gemm_mha_v2
    # nodes) → measured ~1.76×@L=256 / 4.70×@L=4096 vs fp16-mixed, within the
    # perceptual re-seed floor (FAD 0.59× floor, CLAP within spread, n=128).
    # It is a build-time PRECISION RECIPE only — no new ONNX, no graph change.
    # medium-only: sm-music / sm-sfx use standard attention and already fuse in
    # their fp16-mixed engines. bf16 is NOT seed-reproducible vs fp16-mixed
    # (differential attention is cancellation-sensitive) — a different-but-equal
    # take per seed; use fp16-mixed for bit-reproducibility.
    # Same _DIT_PROFILE (batch=1, dynamic L∈[1,4096], opt=1292) as every DiT
    # engine → identical CLI/feature surface (varlen, CFG sequential dual-pass,
    # neg-prompt/APG, a2a, inpaint). The DiT ONNX bakes batch=1, so there is no
    # varbatch axis on ANY TRT DiT engine (CFG is a sequential dual-pass, as in
    # fp16-mixed) — fusion is verified to survive the full L profile at batch=1.
    "sa3-m-bf16": {
        # 5.8 GB external-data sidecar (fp32 weights) travels alongside.
        "onnx_hf":      ["sa3-m/dit.onnx", "sa3-m/dit.onnx.data"],
        "trt_local":    "sa3-m/dit_bf16.trt",
        "flags":        {"BF16"},
        "network":      "EXPLICIT_BATCH",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    # ── FP32 variants ────────────────────────────────────────────────────
    # DiT FP32: read the unsurgered FP32 ONNX directly (dit.onnx), build
    # STRONGLY_TYPED. ~2× the engine size of FP16-mixed, ~2× slower, but
    # matches PyTorch eager bit-for-bit.
    "sa3-sm-music-fp32": {
        "onnx_hf":      ["sa3-sm-music/dit.onnx"],
        "trt_local":    "sa3-sm-music/dit_fp32.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    "sa3-sm-sfx-fp32": {
        "onnx_hf":      ["sa3-sm-sfx/dit.onnx"],
        "trt_local":    "sa3-sm-sfx/dit_fp32.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    "sa3-m-fp32": {
        # 2.9 GB external-data sidecar travels alongside.
        "onnx_hf":      ["sa3-m/dit.onnx", "sa3-m/dit.onnx.data"],
        "trt_local":    "sa3-m/dit_fp32.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    # SAME-L FP32 decoder: the canonical ONNX is FP16-mixed; we upcast every
    # FP16 initializer/Constant/Cast to FP32 in-process before building. The
    # Triton SWA plugin already runs FP32 internally, so its contract is
    # unchanged. Output engine matches PyTorch FP32 eager closely.
    "same-l-decoder-fp32": {
        "onnx_hf":      ["same-l/dec_dynamic_triton_swa.onnx"],
        "trt_local":    "same-l/dec_dynamic_fp32.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      {"latent": [(1, 256, 32), (1, 256, 1292), (1, 256, 4096)]},
        "plugin":       True,
        "upcast_to_fp32": True,
    },
    # SAME-S FP32 decoder: the canonical ONNX is already FP32 throughout
    # (no FP16 ops to upcast). Just build STRONGLY_TYPED so the engine
    # honors the ONNX dtypes (FP32 instead of BF16).
    "same-s-decoder-fp32": {
        "onnx_hf":      ["same-s/dec_dynamic_bf16.onnx"],
        "trt_local":    "same-s/dec_dynamic_fp32.trt",
        "flags":        set(),
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      {"latent": [(1, 256, 32), (1, 256, 1292), (1, 256, 4096)]},
        "plugin":       False,
    },
}


def _upcast_onnx_to_fp32(input_onnx: str, output_onnx: str) -> str:
    """Walk an FP16-mixed ONNX and rewrite every FP16 entity to FP32.

    Used by recipes with `"upcast_to_fp32": True`. Walks initializers,
    value_info, graph inputs, Constant node `.t` attrs, and Cast nodes
    (to=FP16 → to=FP32). The result is a pure-FP32 trunk that, combined
    with STRONGLY_TYPED at build time, produces an FP32 engine.
    """
    import onnx, numpy as np
    from onnx import TensorProto, numpy_helper

    model = onnx.load(input_onnx)
    n_init = n_vi = n_inp = n_out = n_const = n_cast = 0

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init); n_init += 1
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT; n_vi += 1
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT; n_inp += 1
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = TensorProto.FLOAT; n_out += 1
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t); n_const += 1
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT; n_cast += 1

    print(f"  upcast FP16→FP32: {n_init} init, {n_vi} vi, {n_inp} inp, "
          f"{n_const} const, {n_cast} cast", flush=True)
    onnx.save(model, output_onnx, save_as_external_data=True,
               location=Path(output_onnx).name + ".data",
               size_threshold=1024 * 1024)
    return output_onnx


def _ensure_onnx(rel_paths):
    """Pull the ONNX (and any .data sidecar) from HF; cache on disk."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("error: huggingface_hub not installed. pip install huggingface-hub")
    local_paths = []
    for rel in rel_paths:
        hf_filename = f"{HF_ONNX_PREFIX}/{rel}"
        print(f"  hf_hub_download → {hf_filename}", flush=True)
        local = hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
        local_paths.append(local)
    # The proto path is the first listed; .data sidecars travel alongside.
    return local_paths[0]


def build_one(name: str) -> str:
    recipe = TARGETS[name]
    print(f"\n━━━ build_from_onnx: {name} ━━━")

    # 1. Pull ONNX (cached by huggingface_hub)
    onnx_path = _ensure_onnx(recipe["onnx_hf"])
    print(f"  onnx: {onnx_path}", flush=True)

    # 1b. Optional in-process FP16→FP32 upcast for FP32 variants of FP16-mixed
    # source ONNXes (currently only SAME-L decoder needs this — DiT FP32 reads
    # the pre-existing FP32 dit.onnx directly, SAME-S canonical ONNX is already
    # FP32 throughout).
    if recipe.get("upcast_to_fp32"):
        upcast_path = "/tmp/_build_from_onnx_fp32_upcast.onnx"
        onnx_path = _upcast_onnx_to_fp32(onnx_path, upcast_path)

    # 2. Optional plugin import (SAME-L only — registers samel::diff_attn_swa)
    if recipe["plugin"]:
        print(f"  registering Triton SWA plugin...", flush=True)
        import diff_attn_nocast_plugin  # noqa: F401

    # 3. Build the engine
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if recipe["network"] == "STRONGLY_TYPED":
        net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}", flush=True)
        sys.exit(2)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, recipe["workspace_gb"] << 30)
    if "BF16" in recipe["flags"]:
        cfg.set_flag(trt.BuilderFlag.BF16)

    if recipe["profile"]:
        profile = builder.create_optimization_profile()
        for input_name, (lo, opt, hi) in recipe["profile"].items():
            profile.set_shape(input_name, lo, opt, hi)
        cfg.add_optimization_profile(profile)
        print(f"  optimization profile: {len(recipe['profile'])} input(s)", flush=True)

    print(f"  building TRT (workspace {recipe['workspace_gb']} GB"
          f"{', BF16' if 'BF16' in recipe['flags'] else ''}"
          f"{', STRONGLY_TYPED' if recipe['network']=='STRONGLY_TYPED' else ''})...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print(f"  BUILD FAILED", flush=True)
        sys.exit(3)
    print(f"  built in {time.time()-t0:.0f}s ({serialized.nbytes/1e6:.0f} MB)", flush=True)

    # 4. Write .trt under models/<arch>/<engine>/
    out_dir = arch_dir()
    target = Path(out_dir) / recipe["trt_local"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(serialized)
    print(f"  wrote {target}", flush=True)

    return str(target)


def main():
    canonical = [k for k in TARGETS if not k.endswith("-fp32")]
    fp32      = [k for k in TARGETS if k.endswith("-fp32")]

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCanonical (FP16-mixed) targets:")
        for k in canonical:
            print(f"  {k}")
        print("\nFP32 variants (~2x slower, ~2x engine size, opt-in):")
        for k in fp32:
            print(f"  {k}")
        print("\nGroups:")
        print("  all       — every canonical target (default for shipping)")
        print("  all-fp32  — every FP32 target")
        print("  all-both  — both canonical and FP32")
        sys.exit(1)

    target = sys.argv[1]
    if target == "all":
        for name in canonical:
            build_one(name)
    elif target == "all-fp32":
        for name in fp32:
            build_one(name)
    elif target == "all-both":
        for name in canonical + fp32:
            build_one(name)
    elif target in TARGETS:
        build_one(target)
    else:
        print(f"unknown target: {target}")
        print(f"valid: {list(TARGETS)} + 'all' / 'all-fp32' / 'all-both'")
        sys.exit(1)


if __name__ == "__main__":
    main()
