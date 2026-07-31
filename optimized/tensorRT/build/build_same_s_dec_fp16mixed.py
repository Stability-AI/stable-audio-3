#!/usr/bin/env python3
"""Build a SAME-S decoder TRT engine in FP16-mixed precision: FP16 trunk with
FP32 islands around the precision-sensitive ops (tanh-bounded norm chains,
attention Softmax, the differential-attention Sub, and the RoPE region).

Background:
- The shipped BF16 engine produces audible "crackling" on long outputs.
- Naive FP16 catastrophically cancels at the differential attention's
  `Sub` op (two large positive softmax-weighted projections whose
  difference is small) and overflows/underflows on long RoPE sequences.
- MLX explicitly notes: "Decoder always FP32 — SAME-S needs it (FP16
  catastrophically cancels at differential attention)".

The SAME-S decoder is structurally different from the SA3 DiT:
- Normalization is *NOT* RMSNorm. It's a tanh-bounded affine norm:
    out = beta + gamma * tanh(alpha * x)
  exporting as a 4-op chain  Mul(alpha,x) -> Tanh -> Mul(gamma,Tanh) -> Add(Mul,beta).
  24 instances total (6 layers * 4 norms/layer = q_norm + k_norm + pre_norm + ff_norm).
- Differential attention: each layer has TWO softmaxes whose
  attention outputs are subtracted: `Sub(MatMul1(softmax1,v), MatMul3(softmax2,v))`.
  The Sub is the catastrophic-cancellation site; the trunk-side residual
  Add downstream is where the cancellation surfaces.
- RoPE: 6 Cos + 6 Sin generated via Range -> Cast -> Div -> Einsum -> Cos/Sin
  (one pair per layer).

This script identifies all of those as FP32 islands and wraps them with
explicit Cast(FP16<->FP32) before doing the trunk FP16 conversion.

Inputs/outputs:
- Input: `latent` (FP32, shape [1, 256, L])  — keep FP32
- Output: `pcm_unbounded` (INT32, shape [1, T, 2]) — the output Clip is removed
  before conversion so runtime can apply no-boost attenuation before INT16 narrowing.
  The pre-Cast scale remains in the graph.

Usage:
    python build_same_s_dec_fp16mixed.py
      [--mode {minimal,rope,full}]      # default: rope
      [--input  /weka2/cj/clod/sa3s/stable-audio-3-optimized/onnx/same-s/dec_dynamic_bf16.onnx]
      [--onnx   /tmp/same_s_dec_fp16mixed.onnx]
      [--engine .../models/sm_90/same-s/dec_dynamic_fp16mixed.trt]
"""
import argparse
import os
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the helpers from the DiT FP16-mixed build. Most of them work as-is
# — only the FP32-island finder needs SAME-S-specific patterns, which we
# override below.
from build_dit_fp16mixed import (
    strip_noop_fp32_casts,
    wrap_islands_with_casts,
    fix_dtype_mismatches,
    manual_convert_to_fp16,
)
from decoder_output import remove_output_hard_clip


# SAME-S decoder profile — same as the canonical BF16 engine.
_SAMES_DEC_PROFILE = {
    "latent": [(1, 256, 32), (1, 256, 1292), (1, 256, 4096)],
}


def find_fp32_islands_same_s_dec(model, mode="minimal"):
    """SAME-S-specific FP32-island finder.

    Differences from the DiT version:
      - Normalization chain is tanh-bounded (Mul -> Tanh -> Mul -> Add)
        instead of RMSNorm (Pow -> ReduceMean -> ... -> Mul -> Mul).
      - Differential attention has a Sub op combining two attention paths;
        both Softmaxes feeding it AND the Sub itself must be FP32.

    mode=minimal: norm chains + Softmax + diff-attn Sub.
    mode=rope:    minimal + RoPE generation / apply_rotary regions (default).
    mode=full:    every Cast(to=FP32) downstream region — biggest islands.
    """
    out2node = {}
    in2nodes = {}
    for n in model.graph.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2nodes.setdefault(inp, []).append(n)

    blocked = set()

    def bfs_forward_until_cast(seed):
        queue = list(seed.output)
        seen = set([seed.name])
        while queue:
            out_name = queue.pop(0)
            for consumer in in2nodes.get(out_name, []):
                if consumer.name in seen:
                    continue
                seen.add(consumer.name)
                if consumer.op_type == "Cast":
                    continue
                blocked.add(consumer.name)
                queue.extend(consumer.output)

    # (A) Original Cast(to=FP32) seeds — varies by mode.
    fp32_cast_seeds = []
    for c in model.graph.node:
        if c.op_type != "Cast":
            continue
        to_val = None
        for a in c.attribute:
            if a.name == "to":
                to_val = a.i
        if to_val != 1:  # FP32
            continue
        fp32_cast_seeds.append(c)

    def is_rope_island_seed(seed):
        queue = [(seed, 0)]
        seen = set([seed.name])
        while queue:
            n, d = queue.pop(0)
            if d > 6:
                continue
            for o in n.output:
                for c in in2nodes.get(o, []):
                    if c.name in seen:
                        continue
                    seen.add(c.name)
                    if c.op_type in ("Cos", "Sin", "Einsum"):
                        return True
                    queue.append((c, d + 1))
        return False

    def is_self_attn_seed(seed):
        # SAME-S has 72 of these Cast(to=FP32) markers inside the
        # self_attn submodule that protect Sqrt/Div/Mul chains for
        # attention scale + diff-attn lambda mixing. Catching them in
        # addition to the RoPE-leading seeds restores full PyTorch parity
        # without dragging in the irrelevant shape-manipulation casts.
        return "self_attn" in seed.name

    rope_seeds = []
    other_seeds = []
    if mode in ("rope", "attention", "full"):
        for s in fp32_cast_seeds:
            include = False
            if mode == "full":
                include = True
            elif mode == "attention":
                include = is_rope_island_seed(s) or is_self_attn_seed(s)
            elif mode == "rope":
                include = is_rope_island_seed(s)
            if include:
                blocked.add(s.name)
                bfs_forward_until_cast(s)
                rope_seeds.append(s)
            else:
                other_seeds.append(s)

    # (B) Tanh-bounded norm chains.
    # Pattern: Mul(alpha, x) -> Tanh -> Mul(gamma, Tanh) -> Add(Mul, beta)
    # We identify them by walking from each Tanh node: its predecessor is
    # the first Mul (alpha * x), its successor is the second Mul
    # (gamma * tanh), whose successor is the Add (+ beta).
    norm_chains = 0
    for n in model.graph.node:
        if n.op_type != "Tanh":
            continue
        # Predecessor must be a Mul (alpha * x). Walk back via input[0].
        pre_mul = out2node.get(n.input[0]) if n.input else None
        if pre_mul is None or pre_mul.op_type != "Mul":
            continue
        # Successor must be a Mul (gamma * tanh).
        succ = in2nodes.get(n.output[0], [])
        post_mul = next((c for c in succ if c.op_type == "Mul"), None)
        if post_mul is None:
            continue
        # Then Add (+ beta).
        add_succ = in2nodes.get(post_mul.output[0], [])
        add_node = next((c for c in add_succ if c.op_type == "Add"), None)
        if add_node is None:
            continue
        chain = [pre_mul, n, post_mul, add_node]
        for c in chain:
            blocked.add(c.name)
        norm_chains += 1

    # (C) Softmax — all of them.
    softmax_count = 0
    for n in model.graph.node:
        if n.op_type == "Softmax":
            blocked.add(n.name)
            softmax_count += 1

    # (D) Differential-attention Sub:
    # In SAME-S, each transformer layer has a self_attn/Sub that combines
    # two MatMul outputs whose other inputs are Softmax results. The Sub
    # is the catastrophic-cancellation site at FP16.
    # We also pull in the MatMul nodes (Softmax @ V) on both sides — these
    # produce the values that get subtracted, and their precision matters
    # for the subtraction.
    diff_sub_count = 0
    for n in model.graph.node:
        if n.op_type != "Sub":
            continue
        src0 = out2node.get(n.input[0]) if len(n.input) > 0 else None
        src1 = out2node.get(n.input[1]) if len(n.input) > 1 else None
        if src0 is None or src1 is None:
            continue
        if src0.op_type != "MatMul" or src1.op_type != "MatMul":
            continue
        # Verify both MatMuls have a Softmax feeding them.
        def has_softmax_input(matmul_node):
            for inp in matmul_node.input:
                p = out2node.get(inp)
                if p is not None and p.op_type == "Softmax":
                    return True
            return False
        if not (has_softmax_input(src0) and has_softmax_input(src1)):
            continue
        blocked.add(n.name)
        blocked.add(src0.name)
        blocked.add(src1.name)
        diff_sub_count += 1

    # (E) Bottleneck variational sampling chain:
    # The decoder graph starts with a small reparameterization that
    # combines the latent with noise:
    #   sliced = Slice(Pad(latent))
    #   Mul_2 = sliced * running_std         <- shape provider for noise
    #   noise = RandomNormalLike(Mul_2)      <- TRT only supports FP32 out
    #   Mul_3 = noise * running_std
    #   Mul_4 = Mul_3 * Constant_20
    #   Add   = Mul_2 + Mul_4                <- final bottleneck output
    # The whole region must stay FP32 because:
    #   - TRT 10.15 has no FP16 kernel for RandomNormalLike.
    #   - `running_std` is shared by Mul_2 and Mul_3; if we keep it FP32 for
    #     one we must keep both Muls FP32.
    # We walk forward AND backward from RandomNormalLike, plus we sweep up
    # any consumer of `running_std` / `running_mean` initializers.
    bottleneck_count = 0

    def walk_forward(seed, depth_cap=6):
        queue = [(seed, 0)]
        seen = set([seed.name])
        while queue:
            cur, depth = queue.pop(0)
            if depth >= depth_cap:
                continue
            for o in cur.output:
                for c in in2nodes.get(o, []):
                    if c.name in seen:
                        continue
                    seen.add(c.name)
                    # Stop at hard boundaries.
                    if c.op_type in ("MatMul", "Conv", "Cast", "Transpose"):
                        continue
                    blocked.add(c.name)
                    queue.append((c, depth + 1))

    def walk_backward(seed, depth_cap=4):
        queue = [(seed, 0)]
        seen = set([seed.name])
        while queue:
            cur, depth = queue.pop(0)
            if depth >= depth_cap:
                continue
            for inp in cur.input:
                if not inp:
                    continue
                p = out2node.get(inp)
                if p is None or p.name in seen:
                    continue
                seen.add(p.name)
                if p.op_type in ("Cast", "Slice", "Pad", "Shape", "Constant",
                                 "Unsqueeze", "Gather", "Concat", "Reshape"):
                    continue
                blocked.add(p.name)
                queue.append((p, depth + 1))

    for n in model.graph.node:
        if n.op_type != "RandomNormalLike":
            continue
        blocked.add(n.name)
        bottleneck_count += 1
        walk_forward(n)
        walk_backward(n)

    # Also: any Mul/Add that consumes the bottleneck running_std/_mean
    # initializers must be FP32.
    bottleneck_init_names = [init.name for init in model.graph.initializer
                              if "bottleneck.running" in init.name]
    for init_name in bottleneck_init_names:
        for c in in2nodes.get(init_name, []):
            blocked.add(c.name)

    # (F) Identity-aliases of FP32 initializers:
    # In SAME-S, the per-layer RoPE `inv_freq` initializer is shared across
    # all 6 layers via Identity nodes:
    #   model.decoder.layers.3.transformers.0.rope.inv_freq (FP32 init)
    #     -> Identity_947 -> .transformers.5.rope.inv_freq -> Einsum (BLOCKED)
    #     -> Identity_948 -> .transformers.4.rope.inv_freq -> Einsum (BLOCKED)
    #     ...
    #     -> Einsum /decoder/layers.3/transformers.0/Einsum (BLOCKED)
    # If any Identity nodes aren't themselves blocked, manual_convert_to_fp16
    # will convert the initializer to FP16 (its consumer set isn't 100%
    # blocked), which then mismatches with the blocked Einsum's dtype.
    # We mark any Identity whose downstream consumers are ALL blocked (or
    # eventually feed only blocked nodes through more Identities) as blocked
    # itself. Iterate to a fixed point since Identity chains may be deeper.
    n_identity_added = 0
    for _ in range(8):
        added_this_pass = 0
        for n in model.graph.node:
            if n.op_type != "Identity":
                continue
            if n.name in blocked:
                continue
            consumers = []
            for o in n.output:
                consumers.extend(in2nodes.get(o, []))
            if not consumers:
                continue
            if all(c.name in blocked for c in consumers):
                blocked.add(n.name)
                added_this_pass += 1
        n_identity_added += added_this_pass
        if added_this_pass == 0:
            break

    print(f"  FP32 islands (mode={mode}): "
          f"{len(rope_seeds)} RoPE/island Cast seeds (of {len(fp32_cast_seeds)}), "
          f"{norm_chains} tanh-norm chains, "
          f"{softmax_count} Softmax, "
          f"{diff_sub_count} diff-attn Sub, "
          f"{bottleneck_count} bottleneck RandomNormalLike + chain, "
          f"{n_identity_added} Identity aliases")
    print(f"  total blocked nodes: {len(blocked)}")

    from collections import Counter
    name2node = {n.name: n for n in model.graph.node}
    op_counts = Counter(name2node[name].op_type for name in blocked
                        if name in name2node)
    print(f"  blocked op-type top10:")
    for k, v in op_counts.most_common(10):
        print(f"    {k}: {v}")

    return blocked


def fix_extra_dtype_mismatches(model):
    """Second-pass fix for dtype mismatches that the DiT helper's
    fix_dtype_mismatches doesn't catch. SAME-S has a few extra trouble
    spots:

    1. Clip(input, min, max): the DecWrap clamp(-1, 1) outputs FP16 in
       the trunk but its min/max come from Cast(to=FP32) of Constants.
       Retarget those Casts to emit FP16 instead so all three inputs
       agree.

    2. Pad / Resize / Range / etc — not explicitly added here yet, but
       Clip is the only one we've hit.

    Returns the modified model.
    """
    from onnx import TensorProto

    out2node = {}
    in2nodes = {}
    for n in model.graph.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2nodes.setdefault(inp, []).append(n)

    # Forward-propagate dtypes one more time, this time only enough to
    # know what feeds each Clip.
    init_dtype = {init.name: init.data_type for init in model.graph.initializer}
    graph_input_dtype = {gi.name: gi.type.tensor_type.elem_type
                          for gi in model.graph.input}

    def cast_to(node):
        for a in node.attribute:
            if a.name == "to":
                return a.i
        return None

    tensor_dtype = {}
    tensor_dtype.update(init_dtype)
    tensor_dtype.update(graph_input_dtype)

    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    for o in node.output:
                        tensor_dtype[o] = a.t.data_type
        elif node.op_type == "Cast":
            t = cast_to(node)
            if t is not None:
                for o in node.output:
                    tensor_dtype[o] = t
        elif node.op_type == "Shape":
            for o in node.output:
                tensor_dtype[o] = TensorProto.INT64
        elif node.op_type == "ConstantOfShape":
            dt = TensorProto.FLOAT
            for a in node.attribute:
                if a.name == "value":
                    dt = a.t.data_type
            for o in node.output:
                tensor_dtype[o] = dt
        elif node.op_type == "RandomNormalLike":
            dt = TensorProto.FLOAT
            for a in node.attribute:
                if a.name == "dtype":
                    dt = a.i
            for o in node.output:
                tensor_dtype[o] = dt
        elif node.op_type == "Range":
            dt = tensor_dtype.get(node.input[0]) if node.input else TensorProto.INT64
            for o in node.output:
                tensor_dtype[o] = dt
        elif node.op_type in ("Equal", "Greater", "Less", "And", "Or", "Not",
                              "GreaterOrEqual", "LessOrEqual"):
            for o in node.output:
                tensor_dtype[o] = TensorProto.BOOL
        else:
            picked = None
            for inp in node.input:
                if not inp:
                    continue
                dt = tensor_dtype.get(inp)
                if dt in (TensorProto.FLOAT, TensorProto.FLOAT16,
                          TensorProto.BFLOAT16, TensorProto.DOUBLE):
                    picked = dt
                    break
            if picked is None and node.input:
                picked = tensor_dtype.get(node.input[0])
            if picked is not None:
                for o in node.output:
                    tensor_dtype.setdefault(o, picked)

    # Find Clip nodes with dtype-mismatched inputs.
    n_retargeted = 0
    for node in model.graph.node:
        if node.op_type != "Clip":
            continue
        if len(node.input) < 1:
            continue
        main_dt = tensor_dtype.get(node.input[0])
        if main_dt not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for inp_idx in range(1, len(node.input)):
            inp = node.input[inp_idx]
            if not inp:
                continue
            dt = tensor_dtype.get(inp)
            if dt == main_dt:
                continue
            # Find the producer; if it's a Cast, retarget its `to`.
            prod = out2node.get(inp)
            if prod is None or prod.op_type != "Cast":
                # Can't trivially fix — leave it (TRT will complain).
                continue
            for a in prod.attribute:
                if a.name == "to":
                    a.i = main_dt
                    tensor_dtype[inp] = main_dt
                    n_retargeted += 1
                    break

    if n_retargeted:
        print(f"  retargeted {n_retargeted} Cast(to=*) nodes feeding Clip min/max")
    return model


def convert_to_fp16mixed(input_onnx, output_onnx, mode="minimal"):
    """Load FP32 ONNX, identify FP32 islands, convert everything else to
    FP16, and save."""
    import onnx

    print(f"  loading {input_onnx} ({os.path.getsize(input_onnx)/1e6:.0f} MB)")
    model = onnx.load(input_onnx)
    # Inline any external-data initializers/constants.
    from onnx import numpy_helper as _nh
    _onnx_TP = onnx.TensorProto

    def _inline_tensor(t):
        if t.data_location != _onnx_TP.EXTERNAL:
            return
        arr = _nh.to_array(t, base_dir=os.path.dirname(input_onnx))
        t.raw_data = arr.tobytes()
        t.data_location = _onnx_TP.DEFAULT
        t.ClearField("external_data")

    for init in model.graph.initializer:
        _inline_tensor(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    _inline_tensor(attr.t)
    print(f"  inlined external-data tensors")

    removed_clips = remove_output_hard_clip(model)
    print(f"  decoder peak policy: removed {removed_clips} hard Clip")

    # Find FP32 islands BEFORE stripping no-op casts (so RoPE seeds are
    # still in the graph).
    blocked_names = find_fp32_islands_same_s_dec(model, mode=mode)

    model = strip_noop_fp32_casts(model)
    model = wrap_islands_with_casts(model, blocked_names)

    print(f"  manual conversion to FP16 (keeping {len(blocked_names)} nodes FP32)...")
    t0 = time.time()
    fp16_model = manual_convert_to_fp16(model, blocked_names)
    print(f"  converted in {time.time()-t0:.0f}s")

    # NOTE: we no longer retarget RandomNormalLike to FP16 — TRT 10.15
    # doesn't have an FP16 kernel for that op ("No supported formats
    # for /RandomNormalLike"). Instead, the find_fp32_islands_same_s_dec
    # function above already blocks the entire bottleneck chain
    # (RandomNormalLike + downstream Muls/Adds) as FP32.

    print(f"  fixing dtype mismatches with autocast insertion...")
    fp16_model = fix_dtype_mismatches(fp16_model, blocked_names)

    # Retain compatibility with input graphs that have other Clip nodes.
    fp16_model = fix_extra_dtype_mismatches(fp16_model)

    print(f"  saving to {output_onnx}")
    try:
        onnx.save(fp16_model, output_onnx)
    except Exception as e:
        if "Failed to serialize" not in str(e) and "EncodeError" not in type(e).__name__:
            raise
        print(f"  (proto >2GB; using external-data save)")
        onnx.save_model(fp16_model, output_onnx, save_as_external_data=True,
                         all_tensors_to_one_file=True,
                         location=os.path.basename(output_onnx) + ".data",
                         size_threshold=1024, convert_attribute=False)
    print(f"  saved {os.path.getsize(output_onnx)/1e6:.0f} MB")

    from collections import Counter
    init_dtypes = Counter()
    for init in fp16_model.graph.initializer:
        init_dtypes[init.data_type] += 1
    print(f"  initializer dtype counts (1=FP32, 10=FP16, 7=INT64): {dict(init_dtypes)}")

    cast_to_dtypes = Counter()
    for n in fp16_model.graph.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to":
                    cast_to_dtypes[a.i] += 1
    print(f"  Cast.to counts (1=FP32, 10=FP16, 6=INT32): {dict(cast_to_dtypes)}")

    return output_onnx


def build_trt_engine(onnx_path, engine_path, workspace_gb=16):
    """Build TRT engine with STRONGLY_TYPED + SAME-S decoder profile."""
    import tensorrt as trt
    print(f"\n  building TRT engine -> {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}")
        sys.exit(2)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    profile = builder.create_optimization_profile()
    for input_name, (lo, opt, hi) in _SAMES_DEC_PROFILE.items():
        profile.set_shape(input_name, lo, opt, hi)
    cfg.add_optimization_profile(profile)
    print(f"  optimization profile: {len(_SAMES_DEC_PROFILE)} input(s)")

    print(f"  building TRT (workspace {workspace_gb} GB, STRONGLY_TYPED, FP16-mixed)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print(f"  BUILD FAILED")
        sys.exit(3)
    print(f"  built in {time.time()-t0:.0f}s ({serialized.nbytes/1e6:.0f} MB)")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  wrote {engine_path}")
    return engine_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",
                    default="/weka2/cj/clod/sa3s/stable-audio-3-optimized/"
                            "onnx/same-s/dec_dynamic_bf16.onnx",
                    help="Input FP32 ONNX (the `_bf16` suffix refers to the "
                         "eventual engine flavor; the ONNX itself is FP32)")
    ap.add_argument("--onnx",
                    default="/tmp/same_s_dec_fp16mixed.onnx",
                    help="Output FP16-mixed ONNX (intermediate)")
    ap.add_argument("--engine",
                    default="/weka2/cj/clod/sa3s/stable-audio-3/optimized/"
                            "tensorRT/models/sm_90/same-s/"
                            "dec_dynamic_fp16mixed.trt",
                    help="Output TRT engine path")
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--mode", choices=("minimal", "rope", "attention", "full"),
                    default="attention",
                    help="FP32 island coverage: "
                         "minimal=norm+Softmax+diff-attn-Sub only, "
                         "rope=+RoPE region, "
                         "attention=+RoPE+all self_attn Cast(FP32) seeds (default), "
                         "full=every Cast(FP32) downstream region.")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip ONNX conversion (reuse existing --onnx file)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip TRT build (just produce the ONNX)")
    args = ap.parse_args()

    if not args.skip_convert:
        print(f"━━━ Convert FP32 ONNX -> FP16-mixed ONNX (mode={args.mode}) ━━━")
        convert_to_fp16mixed(args.input, args.onnx, mode=args.mode)

    if not args.skip_build:
        print("\n━━━ Build TRT engine ━━━")
        build_trt_engine(args.onnx, args.engine, args.workspace_gb)


if __name__ == "__main__":
    main()
