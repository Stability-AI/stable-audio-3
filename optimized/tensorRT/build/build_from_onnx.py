#!/usr/bin/env python3
"""Compile a TRT engine from a pre-built ONNX file pulled from HuggingFace.

This is the "consumer" build path: no stable-audio-tools, no model checkpoints,
no PyTorch source — just TensorRT + the public ONNX files from
stabilityai/stable-audio-3-optimized/onnx/.

To rebuild engines for a new GPU arch (sm_100, sm_120, ...) you run this on
that GPU and TRT bakes the arch into the engine.

SAME-L attention kernel: the SWA plugin ships two implementations and the choice is
baked into the engine at build time. AOT (default) compiles a block-tiled tensor-core
PTX kernel in, so nothing re-enters Python at inference and the engine can be captured
into a CUDA graph. JIT dispatches to Triton through a Python callback on every enqueue.
If you are bringing up a new architecture and the decoder produces a constant wash of
noise, or silence, read "Choosing the SAME-L attention kernel" in build/README.md before
anything else — that is the signature of a non-capturable engine inside the mega-graph.

    SA3_SWA_PLUGIN=aot|jit   which implementation the engine uses (default aot)
    SA3_SWA_AOT=mma|ptx      which AOT kernel, if AOT (default mma)

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
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
# diff_attn_nocast_plugin + triton_swa_v2 (for SAME-L plugin) live in ../scripts/
sys.path.insert(0, str(SCRIPTS_DIR.parent / "scripts"))

from _arch import detect_arch, arch_dir  # noqa: E402
from decoder_output import rewrite_decoder_onnx  # noqa: E402


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
        "unbounded_audio": True,
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
        "unbounded_audio": True,
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
    # SA3 small DiTs in fp8 — SELECTABLE (default stays fp16mixed). fp8 E4M3 on the
    # 186 linear GEMMs, attention left fp16-fused and the fp32 RMSNorm/RoPE islands
    # intact — an fp8-QDQ graft onto the fp16mixed graph (build/make_dit_fp8_smalldit.py),
    # NOT the medium's baked-RoPE recipe (these DiTs never had the bf16 long-angle
    # problem). Built STRONGLY_TYPED: the QDQ nodes carry fp8; TRT fires fp8 tensor-core
    # GEMMs on the linears while the fp16 FMHA fuser still runs the attention. Same
    # _DIT_PROFILE (batch=1, dynamic L∈[1,4096]) → identical CLI/feature surface.
    # This is a CLEAN WEIGHT-HALVING tier (479 vs 936 MB, velocity-cos ~0.99 vs eager,
    # clip% at/below fp16mixed), only marginally faster (~1.1×): the small DiTs' ~5 ms
    # forward is overhead-bound at batch 1, so fp8's GEMM savings barely show. sm-* fp8
    # is NOT seed-reproducible vs fp16mixed.
    "sa3-sm-music-fp8": {
        "onnx_hf":      ["sa3-sm-music/dit_fp8.onnx", "sa3-sm-music/dit_fp8.onnx.data"],
        "trt_local":    "sa3-sm-music/dit_fp8.trt",
        "flags":        set(),         # STRONGLY_TYPED + the fp8 QDQ carry the precision
        "network":      "STRONGLY_TYPED",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
    },
    "sa3-sm-sfx-fp8": {
        "onnx_hf":      ["sa3-sm-sfx/dit_fp8.onnx", "sa3-sm-sfx/dit_fp8.onnx.data"],
        "trt_local":    "sa3-sm-sfx/dit_fp8.trt",
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
    # SA3 medium DiT in bf16 — SELECTABLE, no longer the medium default (see the
    # sa3-m entry above). Reuses the SAME raw fp32 dit.onnx as the fp32 variant,
    # but builds with BuilderFlag.BF16 + EXPLICIT_BATCH (NOT STRONGLY_TYPED): a
    # uniform bf16 trunk lets TRT 10.15's FMHA fuser fire (0 → 96 fused
    # _gemm_mha_v2 nodes). It is a build-time PRECISION RECIPE only — no new
    # ONNX, no graph change.
    #
    # The ~1.76×@L=256 / 4.70×@L=4096 speedup this engine was shipped for was
    # measured against an fp16-mixed engine whose attention core was stuck in
    # FP32; with that fixed (build_dit_fp16mixed.py's bound_attention_core) bf16
    # is only ~3% ahead, and it loses on accuracy: weakly-typed BF16 also lets
    # TRT evaluate RoPE's rotation angle in bf16, and that angle reaches ~4155
    # rad at L=4092 where bf16's spacing is 32 rad (> 2π), so position info for
    # the fast-rotating dims is destroyed — the latent inflates ~2.5× over 8
    # steps and a 6-min render clips 2–3% of samples. Clean at short lengths.
    # (The FAD 0.59× floor / CLAP-within-spread n=128 result that cleared this
    # engine predates the long-sequence finding.)
    # medium-only: sm-music / sm-sfx use standard attention and already fuse in
    # their fp16-mixed engines. bf16 is also NOT seed-reproducible vs fp16-mixed;
    # use fp16-mixed for bit-reproducibility.
    # Same _DIT_PROFILE (batch=1, dynamic L∈[1,4096], opt=1292) as every DiT
    # engine → identical CLI/feature surface (varlen, CFG sequential dual-pass,
    # neg-prompt/APG, a2a, inpaint). The DiT ONNX bakes batch=1, so there is no
    # varbatch axis on ANY TRT DiT engine (CFG is a sequential dual-pass, as in
    # fp16-mixed) — fusion is verified to survive the full L profile at batch=1.
    # SA3 medium DiT in bf16 — SELECTABLE (fp16-mixed is the medium default).
    # Reads the ROPE-BAKED ONNX from build_dit_bf16.py, NOT the raw dit.onnx.
    #
    # Baking is what makes a bf16 trunk safe here. RoPE's angle is t*inv_freq with
    # inv_freq[0] == 1.0 exactly, so it reaches ~4155 rad at L=4092 — and bf16's
    # spacing up there is 32 rad, more than a full 2*pi rotation. Computing that
    # angle in bf16 destroys 9/16 frequency pairs, the latent inflates ~2.5x over 8
    # sampling steps, and a 6-minute render clips 2-3% of samples. Building the raw
    # dit.onnx with BuilderFlag.BF16 — what shipped before 2026-07 — has exactly
    # that defect. Precomputing the tables on the host removes the angle arithmetic
    # from the graph entirely; the baked table VALUES live in [-1,1] where bf16 is
    # perfectly adequate (measured: latent std 0.9161 vs fp32's 0.9162).
    #
    # bf16 needs no fp32 RMSNorm islands (its exponent range covers the variance
    # that would overflow fp16), so the trunk is uniform and TRT's FMHA fuser fires:
    # 96 fused MHA nodes. Measured teacher-forced velocity cos vs the fp32 engine
    # 0.9993/0.9990/0.9983 at L=256/1292/4092 on sm_120 (0.9987/0.9990/0.9923 on
    # sm_90), and 380 s clipping 3.112% -> 0.014%. It is ~4-7% faster than
    # fp16-mixed but NOT fp32-exact, and it is not seed-reproducible against it.
    #
    # The 5.8 GB fp32 sidecar is shared with dit.onnx — no second weight copy.
    "sa3-m-bf16": {
        "onnx_hf":      ["sa3-m/dit_bf16.onnx", "sa3-m/dit.onnx.data"],
        "trt_local":    "sa3-m/dit_bf16.trt",
        "flags":        {"BF16", "OBEY_PRECISION_CONSTRAINTS"},
        "network":      "EXPLICIT_BATCH",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
        # The two Fourier chains take runtime inputs, so they cannot be baked and
        # still need a (tiny, ~0.1%) fp32 island. See _pin_fourier_fp32.
        "pin_fourier_fp32": True,
    },
    # SA3 medium DiT in fp8 — SELECTABLE, medium-only, the MAX-SPEED clean tier,
    # now CALIBRATED. Identical recipe to sa3-m-bf16 (same RoPE-baked ONNX flow from
    # build_dit_bf16.py, same EXPLICIT_BATCH + OBEY + Fourier fp32 island) PLUS
    # BuilderFlag.FP8: the ONNX carries fp8 E4M3 Quantize/Dequantize pairs on the 176
    # Linear GEMMs, so TRT fires fp8 tensor-core GEMMs there while the 96 attention
    # blocks stay bf16 fused-MHA (_gemm_mha_v2) and the RoPE cos/sin ride as baked fp32
    # constants. Measured on H200 (same-run round-robin): ~1.3x faster than fp16-mixed
    # at every length (1.40x@L129 / 1.32x@L1292 / 1.34x@L4092, also ahead of bf16) and
    # clean at long sequence (latent std 0.86 vs eager 0.95, 0.000% clip @6-min) — the
    # baked RoPE dodges bf16's long-angle drift, so it stays clean where bf16 clips.
    #
    # CALIBRATED SCALES (2026-07-31): the published dit_fp8.onnx now carries the
    # real-conditioning calibrated activation + per-channel weight scales from
    # @ryanontheinside's fp8 work (#47), grafted onto this bakedmin structure by
    # build/transplant_scales.py. Calibration is speed-free (31.2 vs the uncalibrated
    # 30.8 ms/fwd, within run noise) and lifts worst-step velocity-cos vs fp32 from
    # 0.52/0.57/0.64 to 0.92/0.94/0.92 on adversarial seeds; sampling steps 1-7 match
    # the fully-calibrated #47 reference within ~0.001. It is still a SPEED tier over the
    # fp16-mixed default rather than a fidelity upgrade over it (single-step velocity cos
    # ~0.92-0.94 < fp16mixed's ~1.0), but it no longer collapses at the highest-noise
    # first step the way the uncalibrated engine did. #47's fp16-attention variant buys
    # the last ~0.03 of step-0 fidelity for +2 ms/fwd; this tier keeps bf16 attention for
    # the speed. Identity of the shipped engine: 176 fp8 GEMMs + 96 bf16 fused MHA +
    # baked fp32 RoPE constant — check with scripts/verify_fp8_rope.py. Same _DIT_PROFILE
    # (batch=1, L in [1,4096]) → identical CLI/feature surface. medium-only, not
    # seed-reproducible vs fp16mixed.
    #
    # Producer: the fp8-linear ONNX (fp8 QDQ inserted; inv_freq inline) is RoPE-baked by
    # build_dit_bf16.py, the SAME baker as the bf16 tier (it handles inline OR external
    # inv_freq), then build/transplant_scales.py swaps in #47's calibrated scale VALUES
    # (the 5.8 GB fp32 weights are untouched — TRT re-quantizes them at build). A
    # from-scratch recalibration (model retrain) regenerates those scales with #47's full
    # pipeline: make_calib.py (real-conditioning capture, in this repo) -> build_dit_fp8.py
    # (max-PTQ + per-channel scales, lives in #47). The baked cos/sin tables are sized to
    # profile max L=4096; renders past 4096 (the SAME-L decoder's own cap → rejected by
    # sa3_trt) would need a re-bake (--max-t). The .onnx keeps its external-data reference
    # to `dit_fp8lin.onnx.data`, so that exact sidecar name is what ships alongside.
    "sa3-m-fp8": {
        "onnx_hf":      ["sa3-m/dit_fp8.onnx", "sa3-m/dit_fp8lin.onnx.data"],
        "trt_local":    "sa3-m/dit_fp8.trt",
        "flags":        {"BF16", "FP8", "OBEY_PRECISION_CONSTRAINTS"},
        "network":      "EXPLICIT_BATCH",
        "workspace_gb": 16,
        "profile":      _DIT_PROFILE,
        "plugin":       False,
        # Same tiny fp32 island as bf16 for the runtime Fourier chains; on the
        # RoPE-baked ONNX the only Cos/Sin left ARE those two chains.
        "pin_fourier_fp32": True,
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
        "unbounded_audio": True,
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
        "unbounded_audio": True,
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


def _pin_fourier_fp32(network, trt) -> int:
    """Pin the seconds_total / timestep Fourier angle chains to FP32 compute.

    Needed only by the bf16 recipe, and only for these two chains. Both compute
    cos/sin of an angle that reaches ~62,000 and ~55,000 radians respectively
    (seconds_total is s/384 * seconds_freqs, with seconds_freqs maxing at 2*pi*10000).
    bf16 has 8 significand bits, so its ULP passes 2*pi above ~2048 rad — one
    quantisation step becomes more than a full rotation and the feature is destroyed.
    Unlike RoPE these are functions of RUNTIME INPUTS, so they cannot be baked into
    constants and fp32 is the only option. The tensors are scalars and [1,128]
    vectors, so the island costs ~0.1%.

    Selection is STRUCTURAL, not by name. The input graph has already had RoPE's trig
    replaced by baked constants, so every Cos/Sin still present *is* one of these two
    chains — which stays true across re-exports, where a name regex would not. Then
    pin the whole ancestor closure, because one bf16 op anywhere upstream ruins the
    angle just as thoroughly as one at the trig itself. (An earlier name-based pin
    silently missed ONNX `/Clip`: the parser expands it into two *unnamed* ElementWise
    layers that no regex can match, and the resulting engine was a no-op fix.)

    CONSTANT layers are skipped: one holding Int64 weights rejects a FLOAT precision
    outright, and TRT materialises float constants at whatever precision the pinned
    consumer demands anyway.
    """
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    sinks = [ly for ly in layers
             if ly.type == trt.LayerType.UNARY and
             any(s in ly.name for s in ("Cos", "Sin"))]
    if not sinks:
        raise RuntimeError(
            "no Cos/Sin layers found — expected the two Fourier chains. Is this the "
            "RoPE-baked ONNX (build_dit_bf16.py)? Building the raw dit.onnx as bf16 "
            "reintroduces the long-sequence RoPE defect.")

    by_out = {}
    for ly in layers:
        for j in range(ly.num_outputs):
            by_out[ly.get_output(j).name] = ly
    seen, stack = {id(s): s for s in sinks}, list(sinks)
    while stack:
        ly = stack.pop()
        for j in range(ly.num_inputs):
            t = ly.get_input(j)
            if t is None:
                continue
            up = by_out.get(t.name)
            if up is not None and id(up) not in seen:
                seen[id(up)] = up
                stack.append(up)

    FLOATY = (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16)
    n = 0
    for ly in seen.values():
        if ly.type == trt.LayerType.CONSTANT:
            continue
        if not any(ly.get_output(j).dtype in FLOATY for j in range(ly.num_outputs)):
            continue
        ly.precision = trt.DataType.FLOAT
        n += 1
    print(f"  fourier fp32 island: {len(sinks)} trig sinks -> {len(seen)} ancestors, "
          f"{n} pinned", flush=True)
    return n


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
    transform_tmp = None
    if recipe.get("upcast_to_fp32") or recipe.get("unbounded_audio"):
        transform_tmp = tempfile.TemporaryDirectory(prefix=f"sa3-{name}-")
        transform_dir = Path(transform_tmp.name)

    # 1b. Optional in-process FP16→FP32 upcast for FP32 variants of FP16-mixed
    # source ONNXes (currently only SAME-L decoder needs this — DiT FP32 reads
    # the pre-existing FP32 dit.onnx directly, SAME-S canonical ONNX is already
    # FP32 throughout).
    if recipe.get("upcast_to_fp32"):
        upcast_path = transform_dir / "decoder_fp32.onnx"
        onnx_path = _upcast_onnx_to_fp32(onnx_path, str(upcast_path))

    # Decoder ONNXes historically baked `audio.clamp(-1, 1)`, PCM scaling,
    # and INT32 conversion into the output tail. Expose sample-major FP32 so
    # runtime peak protection sees non-finites and preserves sample ratios.
    if recipe.get("unbounded_audio"):
        unbounded_path = transform_dir / "decoder_unbounded_audio.onnx"
        onnx_path = rewrite_decoder_onnx(onnx_path, str(unbounded_path))

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
    if recipe["plugin"]:
        # The SAME-L SWA plugin registers BOTH an aot_impl (PTX compiled into the
        # engine) and an impl (Triton via a Python callback). TRT refuses to create the
        # plugin unless the network states which it prefers:
        #   "Plugin 'samel::diff_attn_swa' has both AOT and JIT implementations.
        #    PREFER_AOT_PYTHON_PLUGINS or PREFER_JIT_PYTHON_PLUGINS should be specified."
        #
        # Two implementations of the SAME-L SWA plugin exist and the engine bakes in
        # whichever this build prefers. Default AOT; override with SA3_SWA_PLUGIN=jit.
        # See "Choosing the SAME-L attention kernel" in build/README.md.
        want = os.environ.get("SA3_SWA_PLUGIN", "aot").lower()
        if want not in ("aot", "jit"):
            sys.exit(f"SA3_SWA_PLUGIN must be 'aot' or 'jit', got {want!r}")
        if want == "aot":
            net_flags |= 1 << int(
                trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)
            print("  plugin impl: AOT — graph-capturable, no Python in the enqueue "
                  "path (SA3_SWA_PLUGIN=jit to switch)", flush=True)
        else:
            net_flags |= 1 << int(
                trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS)
            print("  plugin impl: JIT — Triton at runtime. NOT stream-capturable on "
                  "sm_120; verify before shipping (see build/README.md)", flush=True)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(onnx_path)
    if transform_tmp is not None:
        transform_tmp.cleanup()
    if not parsed:
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}", flush=True)
        sys.exit(2)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, recipe["workspace_gb"] << 30)
    for flag in sorted(recipe["flags"]):
        cfg.set_flag(getattr(trt.BuilderFlag, flag))
    if recipe.get("pin_fourier_fp32"):
        _pin_fourier_fp32(network, trt)

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
