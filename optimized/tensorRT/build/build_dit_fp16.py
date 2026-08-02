#!/usr/bin/env python3
"""Build a SA3 DiT TRT engine in FP16-mixed precision: FP16 trunk with FP32
islands around RMSNorm and (by default) the rotary position embedding
*generation* that PyTorch marks @autocast(enabled=False). The attention core
(QK^T -> Softmax -> P.V) runs FP16 so TRT's FMHA fuser can fire — step 6 below.

Background:
- The BF16 engine drifts at long sequence, and the cause is narrower than
  "bf16 accumulates error over the 8 sampling steps": a per-op ablation showed
  bf16 arithmetic everywhere else in the DiT is clean (GEMMs, RMSNorm, the
  differential-attention subtraction, even the cos/sin tables). The single
  culprit is the RoPE rotation ANGLE, t * inv_freq, which reaches ~4155 rad at
  L=4092 — where bf16's spacing is 32 rad, five whole rotations — so cos/sin of
  it is uncorrelated with the true value and position information for the
  fast-rotating dims is destroyed. The axis that separates a good engine from a
  bad one is therefore "fp32 angle island present", not fp16-vs-bf16.
- Naive FP16 (just BuilderFlag.FP16) catastrophically diverges: TRT promotes
  RMSNorm's Pow/ReduceMean/Sqrt to FP16, which overflows (variance > 65504).
  The Softmax is a different story — see bound_attention_core(): FP16 softmax
  is fine, and an FP32 one costs 4.3x at L=4096 for nothing.
- MLX gets away with FP16 because it casts those subgraphs to FP32 internally.

This script reproduces that recipe on the ONNX side:

  1. Identify FP32 "islands":
       - Every RMSNorm chain (Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Mul)
       - Every Softmax (transient — step 6 hands the attention core back to FP16)
       - (mode=rope, default) Every region reachable from a Cast(to=FP32) that
         leads to Cos/Sin or Einsum — i.e. the rotary position embedding
         (@autocast(enabled=False) in PyTorch source)
  2. Strip the 569 no-op Cast(to=FP32) nodes that the PyTorch ONNX export
     left as semantic markers — we already have the islands explicitly.
  3. Wrap each island input/output with Cast(to=FP32)/Cast(to=FP16). These
     are no-ops in the still-FP32 graph; they become real boundaries after
     step 4.
  4. Manually convert non-island initializers/Constant/value_info to FP16.
     RMSNorm gamma weights and other initializers feeding islands stay FP32.
  5. Post-process: walk the graph inserting autocast nodes wherever two
     operands of a Mul/MatMul/Add/etc disagree on dtype (this handles the
     boundary between FP32 t5_hidden input and FP16 trunk).
  6. bound_attention_core(): terminate the RoPE island before QK^T so the
     attention core (QK^T -> Softmax -> P.V) runs FP16 and TRT's FMHA fuser can
     fire. Mirrors `apply_rotary_pos_emb`'s closing `t.to(out_dtype)`, which the
     island passes above do not reproduce. WITHOUT this step the whole O(L^2)
     attention core stays FP32: 0 fused MHA nodes and 4.3x slower at L=4096, for
     no accuracy gain (teacher-forced cos vs FP32 is 1.0000 WITH the pass and
     0.9998 without). See the function docstring. Disable with --no-bound-attn
     only to reproduce the pre-2026-07 engines.
  7. Build a TRT engine with NetworkDefinitionCreationFlag.STRONGLY_TYPED
     so TRT respects the per-tensor dtypes exactly (no auto-promotion).
     STRONGLY_TYPED is mandatory, not stylistic: weakly-typed + BuilderFlag.FP16
     lets TRT re-cast the FP32 RMSNorm islands and reintroduces the overflow.

Inputs/outputs stay FP32 so the runtime can swap engines transparently.

Validated results (sa3-sm-music, L=1292):
  - Single-step cos vs PT FP32: 0.99997
  - 8-step final-latent cos vs PT FP32: 0.998
  - Decoded audio RMS-curve correlation vs PT FP32: 0.999
  - Inference time per 8-step: ~44 ms (BF16: 42 ms, FP32: 116 ms)
  - Engine size: 926 MB (BF16: 935 MB, FP32: 1842 MB)

Usage:
    python build_dit_fp16.py
        [--mode {minimal,rope,full}]      # default: rope
        [--input  /tmp/dit_sm-music_fixed_v2.onnx]
        [--onnx   /tmp/dit_sm-music_fp16.onnx]
        [--engine .../models/sm_90/sa3-sm-music/dit_fp16.trt]
"""
import argparse
import os
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


T5_TOKENS = 256
T5_HIDDEN_DIM = 768

# Optimization profile — same as sa3-sm-music canonical build.
# Min=1 lets sub-trained short forms run; opt=1292 is the canonical tactic point.
_DIT_PROFILE = {
    "x":              [(1, 256, 1),     (1, 256, 1292),   (1, 256, 4096)],
    "t":              [(1,),            (1,),             (1,)],
    "t5_hidden":      [(1, T5_TOKENS, T5_HIDDEN_DIM)] * 3,
    "t5_mask":        [(1, T5_TOKENS)] * 3,
    "seconds_total":  [(1,)] * 3,
    "local_add_cond": [(1, 257, 1),     (1, 257, 1292),   (1, 257, 4096)],
}


def find_fp32_islands(model, mode="minimal"):
    """Identify nodes that must stay in FP32.

    mode="minimal": only RMSNorm chains + Softmax. Smallest island set —
        produces fastest engine but may not preserve all the precision-
        sensitive paths PyTorch's autocast(enabled=False) was protecting.
    mode="rope": minimal + the rotary embedding generation + apply_rotary
        islands (matches PyTorch's @autocast(enabled=False) on RoPE).
    mode="full": every node reachable from a Cast(to=FP32) seed in the
        original FP32 ONNX. Largest island set — preserves every PyTorch
        .float() intent but most of the network ends up FP32.

    All modes additionally block:
      - RMSNorm chains (Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Mul)
      - Softmax
    """
    out2node = {}
    in2nodes = {}
    for n in model.graph.node:
        for o in n.output:
            out2node[o] = n
        for inp in n.input:
            in2nodes.setdefault(inp, []).append(n)

    blocked = set()

    # Walk forward from a node, stopping at any Cast op. Add every node
    # we visit to `blocked`. Used for tracing FP32 islands from Cast(to=FP32)
    # seeds.
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
                    continue  # boundary
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

    # Heuristic: classify Cast(to=FP32) seeds by what they're protecting.
    # RoPE-island casts are those near Range, Einsum, Cos, Sin.
    # RMSNorm-pre casts are those right before a Pow (and the chain head).
    # apply_rotary casts are those right before Mul/Add inside attention.
    def is_rope_island_seed(seed):
        # The 2 rope seeds (one for q, one for k) lead toward Cos/Sin via
        # Range -> Cast -> Div -> Einsum -> Cos/Sin.
        # Also each apply_rotary call casts q, k to FP32 before the rotation
        # multiply/add. Those are 2 per layer.
        # Cheap heuristic: walk downstream up to 3 steps and check if we
        # encounter Cos/Sin or hit the Q/K projection chain.
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

    rope_seeds = []
    other_seeds = []
    if mode in ("rope", "full"):
        for s in fp32_cast_seeds:
            if mode == "full" or is_rope_island_seed(s):
                blocked.add(s.name)
                bfs_forward_until_cast(s)
                rope_seeds.append(s)
            else:
                other_seeds.append(s)

    # (B) Explicit RMSNorm chains
    RMS_STEPS = ["ReduceMean", "Add", "Sqrt", "Div", "Mul", "Mul"]
    rmsnorm_chains = 0
    for n in model.graph.node:
        if n.op_type != "Pow":
            continue
        chain = [n]
        cur = n
        ok = True
        for step in RMS_STEPS:
            ops = in2nodes.get(cur.output[0], [])
            match = [c for c in ops if c.op_type == step]
            if not match:
                ok = False
                break
            cur = match[0]
            chain.append(cur)
        if not ok:
            continue
        rmsnorm_chains += 1
        for c in chain:
            blocked.add(c.name)

    # (C) Explicit Softmax
    softmax_count = 0
    for n in model.graph.node:
        if n.op_type == "Softmax":
            blocked.add(n.name)
            softmax_count += 1

    print(f"  FP32 islands (mode={mode}): "
          f"{len(rope_seeds)} RoPE/island Cast seeds (of {len(fp32_cast_seeds)}), "
          f"{rmsnorm_chains} RMSNorm chains, "
          f"{softmax_count} Softmax")
    print(f"  total blocked nodes: {len(blocked)}")

    # Op-type breakdown for sanity
    from collections import Counter
    name2node = {n.name: n for n in model.graph.node}
    op_counts = Counter(name2node[name].op_type for name in blocked
                        if name in name2node)
    print(f"  blocked op-type top10:")
    for k, v in op_counts.most_common(10):
        print(f"    {k}: {v}")

    return blocked


def strip_noop_fp32_casts(model):
    """Remove Cast(to=FP32) nodes whose input is already FP32. The PyTorch
    ONNX export inserts these to mirror `x.float()` calls in FP32 modules
    (where they're no-ops). Removing them before the FP16 conversion lets
    the converter correctly insert FP32->FP16 transitions; without removal,
    these stale casts produce dtype contradictions (TRT: "A is Float, B is
    Half"). The leading Cast(to=FP32) for RMSNorm pre_norm chains is one
    such example."""
    import onnx as _onnx
    from onnx import shape_inference, TensorProto
    # In-memory shape_inference fails on models >2GB (protobuf serialization
    # limit) — happens for sa3-m. The fallback (file-based inference + tempdir
    # external data) is fragile because inferred initializers retain refs to
    # the tempdir even after explicit clearing. Simpler escape: skip the
    # no-op-Cast-stripping pass entirely. It's an optimization (slightly
    # smaller graph, easier TRT inspection), not required for correctness —
    # the downstream conversion handles redundant Casts fine.
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as e:
        if "Failed to serialize" in str(e) or "EncodeError" in type(e).__name__:
            print(f"  shape_inference failed (model >2GB, protobuf limit) — "
                  f"skipping no-op-Cast stripping. Engine will still build correctly.")
            return model
        raise
    vi_dtype = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        vi_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        vi_dtype[init.name] = init.data_type

    # Collect no-op Casts to remove
    to_remove = []
    rename_map = {}  # output -> input (rewire downstream)
    for n in model.graph.node:
        if n.op_type != "Cast":
            continue
        to_attr = None
        for a in n.attribute:
            if a.name == "to":
                to_attr = a.i
        if to_attr != TensorProto.FLOAT:  # only FP32 casts
            continue
        in_dtype = vi_dtype.get(n.input[0])
        if in_dtype == TensorProto.FLOAT:
            to_remove.append(n)
            rename_map[n.output[0]] = n.input[0]
    print(f"  stripping {len(to_remove)} no-op Cast(to=FP32) nodes")

    # Transitively close rename_map so chains like Cast_6 -> Cast_8 -> consumer
    # rewire all the way back to the pre-Cast_6 source. Otherwise we end up
    # with consumers referring to the intermediate Cast_6_output_0 which has
    # also been removed.
    def resolve(name):
        while name in rename_map:
            name = rename_map[name]
        return name
    rename_map = {k: resolve(v) for k, v in rename_map.items()}

    # Rewire downstream consumers
    for n in model.graph.node:
        for i, inp in enumerate(n.input):
            if inp in rename_map:
                n.input[i] = rename_map[inp]
    # Also fix graph outputs (rare)
    for o in model.graph.output:
        if o.name in rename_map:
            o.name = rename_map[o.name]

    # Remove the dead Cast nodes
    for n in to_remove:
        model.graph.node.remove(n)

    return model


def wrap_islands_with_casts(model, blocked_names):
    """Surround every FP32-island (Pow/ReduceMean/.../Mul) and Softmax node
    with explicit Cast(FP16->FP32) on inputs and Cast(FP32->FP16) on outputs
    BEFORE the FP16 conversion runs. This ensures the converter sees clear
    dtype boundaries — every input to a blocked node is preceded by a Cast,
    every output is followed by a Cast.

    Why pre-insert: the converter's auto-Cast-insertion has fragile
    interactions with shape inference; pre-inserting our own Casts (whose
    `to` attribute we'll later rewrite via convert_float_to_float16) gives
    deterministic boundaries.

    Strategy:
      - For each blocked-island INPUT edge (input of a blocked node whose
        source is NOT another blocked node), insert
        Cast(in -> in__fp32) and rewire the blocked node to consume in__fp32.
        Initially in__fp32 is Cast(to=FP32) — a no-op in the FP32 graph;
        after convert_float_to_float16, surrounding becomes FP16 and this
        Cast performs the actual upcast.
      - For each blocked-island OUTPUT edge (a non-blocked consumer of a
        blocked node), insert Cast(out -> out__fp16) and rewire that consumer.
    """
    from onnx import TensorProto, helper

    in2nodes = {}
    for n in model.graph.node:
        for inp in n.input:
            in2nodes.setdefault(inp, []).append(n)
    out2node = {}
    for n in model.graph.node:
        for o in n.output:
            out2node[o] = n

    new_casts = []
    n_input_wraps = 0
    n_output_wraps = 0

    # We need to know each tensor's dtype to skip non-float inputs (INT64
    # axes/indices/etc). Build a dtype map from initializers + Constant
    # outputs + graph inputs, then do forward inference for the rest.
    tensor_dtype = {}
    for init in model.graph.initializer:
        tensor_dtype[init.name] = init.data_type
    for gi in model.graph.input:
        tensor_dtype[gi.name] = gi.type.tensor_type.elem_type
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    for o in node.output:
                        tensor_dtype[o] = a.t.data_type
        elif node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to":
                    for o in node.output:
                        tensor_dtype[o] = a.i
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
        elif node.op_type == "Where":
            # Where(condition, x, y) — output dtype matches x/y (inputs[1])
            if len(node.input) >= 2:
                dt = tensor_dtype.get(node.input[1])
                if dt is not None:
                    for o in node.output:
                        tensor_dtype.setdefault(o, dt)
        elif node.op_type in ("Equal", "Greater", "Less", "And", "Or", "Not",
                              "GreaterOrEqual", "LessOrEqual"):
            # Boolean output
            for o in node.output:
                tensor_dtype.setdefault(o, TensorProto.BOOL)
        else:
            # Default: scan inputs for the first FP-typed input dtype, fall
            # back to first input dtype. This avoids leaking BOOL from
            # boolean conditions etc.
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

    # === Inputs ===
    # For each blocked node, every FLOAT input not coming from another
    # blocked node needs a Cast(to=FP32) preface. We skip non-float inputs
    # (INT64 axes/indices/etc) and initializer inputs (those stay FP32
    # directly when feeding an island).
    init_names = {init.name for init in model.graph.initializer}
    input_cast_map = {}
    for node in model.graph.node:
        if node.name not in blocked_names:
            continue
        for i, inp in enumerate(node.input):
            if not inp:
                continue
            src = out2node.get(inp)
            if src is not None and src.name in blocked_names:
                continue  # interior edge
            if inp in init_names:
                continue  # initializer — let it stay FP32, no Cast needed
            # Skip non-float inputs (INT64 axes, etc.)
            dt = tensor_dtype.get(inp)
            if dt is not None and dt not in (TensorProto.FLOAT,
                                              TensorProto.FLOAT16,
                                              TensorProto.BFLOAT16):
                continue
            if inp in input_cast_map:
                node.input[i] = input_cast_map[inp]
                continue
            cast_out = f"{inp}__island_in_fp32"
            cast_name = f"{inp}__island_in_cast"
            new_casts.append(helper.make_node(
                "Cast", inputs=[inp], outputs=[cast_out],
                name=cast_name, to=TensorProto.FLOAT))
            input_cast_map[inp] = cast_out
            node.input[i] = cast_out
            n_input_wraps += 1

    # === Outputs ===
    # For each blocked node, every consumer not in blocked needs a
    # Cast(to=FP16) wrapper, with the blocked node's output rewired
    # through the new cast for that consumer set.
    # Dedupe: one cast per output that has FP16-consumers.
    output_cast_map = {}  # original output name -> new fp16 cast output name
    for node in model.graph.node:
        if node.name not in blocked_names:
            continue
        for out_name in node.output:
            consumers = in2nodes.get(out_name, [])
            non_blocked = [c for c in consumers if c.name not in blocked_names]
            if not non_blocked:
                continue
            cast_out = f"{out_name}__island_out_fp16"
            cast_name = f"{out_name}__island_out_cast"
            new_casts.append(helper.make_node(
                "Cast", inputs=[out_name], outputs=[cast_out],
                name=cast_name, to=TensorProto.FLOAT16))
            output_cast_map[out_name] = cast_out
            for c in non_blocked:
                for i, inp in enumerate(c.input):
                    if inp == out_name:
                        c.input[i] = cast_out
            n_output_wraps += 1

    # Also rewire graph outputs that reference blocked node outputs.
    for o in model.graph.output:
        if o.name in output_cast_map:
            # We can't really rename graph outputs, but the cast was added,
            # so we leave it. The downstream consumer chain has been rewired.
            pass

    model.graph.node.extend(new_casts)
    print(f"  wrapped {len(blocked_names)} blocked nodes: "
          f"{n_input_wraps} input casts, {n_output_wraps} output casts")
    return model


def bound_attention_core(model):
    """Terminate the RoPE island before QK^T so the attention core runs FP16.

    Why this pass exists
    --------------------
    find_fp32_islands() blocks every Softmax (section C) *and* the RoPE region.
    Because RoPE emits q/k in FP32 and nothing casts them back,
    fix_dtype_mismatches then keeps the QK^T MatMul in FP32 to match its operands.
    Net effect: the entire O(L^2) attention core — QK^T, the scale Mul, and Softmax
    — runs FP32, and TensorRT's FMHA fuser cannot fire on it. Measured on the
    medium DiT: **0 fused MHA nodes and 4.3x slower at L=4096** than it needs to be.

    Neither FP32 region is actually required:

      * PyTorch does not keep the attention core in FP32. `apply_rotary_pos_emb`
        ends with `t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)` —
        it computes in FP32 and casts *back* to the autocast dtype — and attention
        is then `F.scaled_dot_product_attention(q, k, v)`, a fused kernel taking
        FP16 inputs. The FP32 softmax island has no counterpart in the source.
      * TRT's FMHA kernels accumulate softmax in FP32 internally, so the explicit
        FP32 softmax buys nothing the kernel does not already do.

    So for each attention this rewrites

        Mul(fp32) -> MatMul(fp32,fp32) -> Cast(fp32)[no-op] -> Softmax(fp32) -> Cast(fp16) -> MatMul(P.V)
    to
        Mul(fp32) -> Cast(fp16) x2 -> MatMul(fp16) ->            Softmax(fp16) ->            MatMul(P.V)

    mirroring the source's cast-back. The RMSNorm islands are untouched: those
    guard variance *overflow* (FP16's 5-bit exponent), a different failure mode
    from precision, and they are still required.

    Verified on sa3-m: 96/96 attentions converted, 96 fused MHA nodes at FP16,
    teacher-forced velocity cos 1.0000 vs the FP32 engine at L=256/1292/4092
    (the unbounded graph matches it at 256/1292 but drops to 0.9998 at 4092), and
    4.3x faster at L=4096.

    Scope: only the medium DiT has the unbounded island. In sa3-sm-music /
    sa3-sm-sfx the RoPE island already terminates before QK^T (verified: their
    QK^T operands are FP16, so their published engines fuse 40/40 MHA), and what
    the pass converts there is the Softmax island alone — a real change to those
    graphs, but not the fix this exists for. Their published ONNX/engines predate
    the pass; --no-bound-attn skips it and reproduces the graph they were built
    from.

    NOTE the resulting graph must be built STRONGLY_TYPED. Building it weakly-typed
    with BuilderFlag.FP16 lets TRT re-cast the FP32 RMSNorm islands too, which is
    the naive-FP16 overflow this recipe exists to prevent (measured: teacher-forced
    cos collapses to 0.88).
    """
    from collections import defaultdict
    from onnx import TensorProto, helper

    g = model.graph
    producer = {}
    consumers = defaultdict(list)
    for n in g.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            if i:
                consumers[i].append(n)
    graph_outputs = {o.name for o in g.output}

    def cast_to(node):
        for a in node.attribute:
            if a.name == "to":
                return a.i
        return None

    drop, rename, new_casts = set(), {}, []
    converted, converted_softmax = 0, []
    skipped = defaultdict(int)

    for sm in [n for n in g.node if n.op_type == "Softmax"]:
        pre = producer.get(sm.input[0])
        if pre is None or pre.op_type != "Cast" or cast_to(pre) != TensorProto.FLOAT:
            skipped["no_fp32_cast_before_softmax"] += 1
            continue
        if len(consumers[sm.input[0]]) != 1:
            skipped["pre_cast_output_shared"] += 1
            continue
        qk = producer.get(pre.input[0])
        if qk is None or qk.op_type != "MatMul":
            skipped["upstream_not_matmul"] += 1
            continue
        posts = consumers[sm.output[0]]
        if not posts or any(p.op_type != "Cast" or cast_to(p) != TensorProto.FLOAT16
                            for p in posts):
            skipped["no_fp16_cast_after_softmax"] += 1
            continue
        if sm.output[0] in graph_outputs or any(p.output[0] in graph_outputs for p in posts):
            skipped["touches_graph_output"] += 1
            continue
        if len(consumers[qk.output[0]]) != 1:
            skipped["qk_output_shared"] += 1
            continue

        for idx, src in enumerate(list(qk.input)):
            out_name = f"{src}__attnfp16_{converted}_{idx}"
            new_casts.append(helper.make_node(
                "Cast", [src], [out_name], to=TensorProto.FLOAT16,
                name=f"{src}__attnfp16_cast_{converted}_{idx}"))
            qk.input[idx] = out_name
        sm.input[0] = qk.output[0]
        drop.add(id(pre))
        for p in posts:
            rename[p.output[0]] = sm.output[0]
            drop.add(id(p))
        converted += 1
        converted_softmax.append(sm)

    print(f"  bound_attention_core: {converted} attention(s) -> FP16 core")
    if skipped:
        print(f"    skipped: {dict(skipped)}")
    if converted == 0:
        raise RuntimeError("bound_attention_core converted nothing — the island "
                           "passes upstream must have changed shape")

    def resolve(name):
        seen = set()
        while name in rename and name not in seen:
            seen.add(name)
            name = rename[name]
        return name

    kept = [n for n in g.node if id(n) not in drop]
    for n in kept:
        for i, t in enumerate(n.input):
            r = resolve(t)
            if r != t:
                n.input[i] = r

    # Each new Cast must precede its consumer to keep the graph topologically sorted.
    cast_by_output = {c.output[0]: c for c in new_casts}
    ordered = []
    for n in kept:
        for t in n.input:
            c = cast_by_output.pop(t, None)
            if c is not None:
                ordered.append(c)
        ordered.append(n)
    if cast_by_output:
        raise RuntimeError(f"{len(cast_by_output)} inserted casts could not be placed")
    del g.node[:]
    g.node.extend(ordered)

    # Retype/drop stale value_info: QK^T and Softmax outputs are FP16 now, and the
    # spliced Cast outputs no longer exist. A stale FP32 declaration on an FP16
    # tensor makes STRONGLY_TYPED reject the graph. Only the attentions we actually
    # converted move — a skipped Softmax keeps its FP32 declaration.
    retype = set()
    for sm in converted_softmax:
        retype.add(sm.output[0])
        retype.add(sm.input[0])     # rewired above to the QK^T MatMul output
    dead = set(rename)
    kept_vi = []
    for vi in g.value_info:
        if vi.name in dead:
            continue
        if vi.name in retype and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        kept_vi.append(vi)
    del g.value_info[:]
    g.value_info.extend(kept_vi)
    return model


def fix_dtype_mismatches(model, blocked_names, low_dt=None):
    """Insert Cast nodes wherever two operands of an op disagree on float dtype.

    Forward dtype propagation from graph inputs and initializers, computing each
    tensor's dtype as it would appear at runtime. For binary/multi-input ops
    (MatMul, Add, Mul, Div, Sub, Pow, Where, ...) all float operands are aligned
    onto one dtype: FP32 when the consuming node is in `blocked_names` (an FP32
    island — never downcast those), otherwise the trunk dtype, and the mismatched
    operands get a Cast.

    `low_dt` is that trunk dtype: FLOAT16 for the fp16-mixed recipe (the default
    when None) or BFLOAT16 for a bf16 trunk. Both recipes need the same graph-wide
    reconciliation — a bf16 trunk with fp32 islands hits exactly the same class of
    mismatch (`ElementWiseOperation DIV must have same input types`,
    `IMatrixMultiplyLayer must have same input types`) — so this pass is shared
    rather than duplicated.
    """
    from onnx import TensorProto, helper

    if low_dt is None:
        low_dt = TensorProto.FLOAT16    # default trunk dtype: the fp16-mixed recipe

    init_dtype = {init.name: init.data_type for init in model.graph.initializer}
    graph_input_dtype = {gi.name: gi.type.tensor_type.elem_type for gi in model.graph.input}

    def cast_to(node):
        for a in node.attribute:
            if a.name == "to":
                return a.i
        return None

    # Forward dtype propagation. Returns dtype for each tensor.
    # Topological order: process nodes in graph order (ONNX guarantees topo).
    tensor_dtype = {}
    tensor_dtype.update(init_dtype)
    tensor_dtype.update(graph_input_dtype)
    # Constant nodes — their outputs are determined by their .t value
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    for o in n.output:
                        tensor_dtype[o] = a.t.data_type

    def get_dt(name):
        return tensor_dtype.get(name, None)

    def set_dt(name, dt):
        tensor_dtype[name] = dt

    # Walk nodes in order, infer their output dtype.
    new_casts = []
    n_inserted = 0

    # Ops that require all float inputs to share dtype
    SHARED_FLOAT_DT_OPS = {
        "MatMul", "Gemm", "Add", "Mul", "Div", "Sub", "Pow", "Min", "Max",
        "Where", "Equal", "Greater", "Less", "And", "Or",
        "Conv", "ConvTranspose",
    }
    # Ops where the output dtype = first float input's dtype
    PRESERVE_OPS = {
        "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Slice", "Concat",
        "Gather", "Identity", "Tile", "Expand", "Pad", "Split",
        "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
        "Softmax", "Sigmoid", "Tanh", "Relu", "Gelu", "Sqrt", "Sin", "Cos",
        "Neg", "Abs", "Exp", "Log", "Conv", "ConvTranspose",
    }

    nodes_list = list(model.graph.node)
    for node in nodes_list:
        if node.op_type == "Cast":
            for o in node.output:
                set_dt(o, cast_to(node))
            continue
        if node.op_type == "Constant":
            continue
        if node.op_type == "Shape":
            # output is int64
            for o in node.output:
                set_dt(o, TensorProto.INT64)
            continue
        if node.op_type == "ConstantOfShape":
            # output dtype = attr value dtype (default FP32)
            dt = TensorProto.FLOAT
            for a in node.attribute:
                if a.name == "value":
                    dt = a.t.data_type
            for o in node.output:
                set_dt(o, dt)
            continue
        if node.op_type == "Range":
            # output is whatever the start dtype is
            dt = get_dt(node.input[0]) if node.input else TensorProto.INT64
            for o in node.output:
                set_dt(o, dt)
            continue
        if node.op_type in ("Equal", "Greater", "Less", "And", "Or", "Not",
                            "GreaterOrEqual", "LessOrEqual"):
            for o in node.output:
                set_dt(o, TensorProto.BOOL)
            continue

        # For shared-dtype ops, find the dominant float dtype and cast
        # any mismatched inputs.
        float_input_dts = []
        for inp in node.input:
            if not inp:
                continue
            dt = get_dt(inp)
            if dt in (TensorProto.FLOAT, low_dt):
                float_input_dts.append((inp, dt))
        if node.op_type in SHARED_FLOAT_DT_OPS and len(float_input_dts) > 1:
            unique_dts = set(dt for _, dt in float_input_dts)
            if len(unique_dts) > 1:
                # Mismatch — need to align. Rule: if blocked node, force FP32.
                # Otherwise prefer low_dt (the trunk dtype, FP16 by default).
                if node.name in blocked_names:
                    target = TensorProto.FLOAT
                else:
                    target = low_dt
                for inp, dt in float_input_dts:
                    if dt == target:
                        continue
                    # Insert a Cast(to=target) before this input
                    cast_out = f"{inp}__autocast_{n_inserted}"
                    cast_node = helper.make_node(
                        "Cast", inputs=[inp], outputs=[cast_out],
                        name=f"{inp}__autocast_node_{n_inserted}",
                        to=target)
                    new_casts.append(cast_node)
                    set_dt(cast_out, target)
                    # Rewire this node's input
                    for i, ninp in enumerate(node.input):
                        if ninp == inp:
                            node.input[i] = cast_out
                    n_inserted += 1

        # Compute output dtype
        if node.op_type in SHARED_FLOAT_DT_OPS or node.op_type in PRESERVE_OPS:
            # Output dtype: first float input dtype (after possible casts)
            for inp in node.input:
                if not inp:
                    continue
                dt = get_dt(inp)
                if dt in (TensorProto.FLOAT, low_dt):
                    for o in node.output:
                        set_dt(o, dt)
                    break
        else:
            # Default: preserve first input dtype
            if node.input:
                dt = get_dt(node.input[0])
                if dt is not None:
                    for o in node.output:
                        set_dt(o, dt)

    model.graph.node.extend(new_casts)
    print(f"  fix_dtype_mismatches: inserted {n_inserted} autocast nodes")
    return model


def manual_convert_to_fp16(model, blocked_names):
    """Convert FP32 ONNX to FP16 trunk + FP32 islands.

    Algorithm:
    1. Compute the set of "FP32 tensors" — outputs of blocked nodes, plus
       outputs of our Cast(to=FP32) wrappers.
    2. For every initializer NOT consumed exclusively by blocked nodes,
       convert from FP32 to FP16. (If exclusively in islands, keep FP32.)
    3. For every value_info NOT in the FP32-tensor set, set dtype FP16.
    4. Convert Constant nodes (op_type=Constant) outside blocked: change
       value tensor to FP16.

    Inputs of the graph stay FP32 (so the runtime doesn't need to change).
    Output of the graph: we'll Cast to FP32 at the very end so the runtime
    sees FP32.
    """
    from onnx import TensorProto, numpy_helper, helper
    import numpy as np

    in2nodes = {}
    out2node = {}
    for n in model.graph.node:
        for inp in n.input:
            in2nodes.setdefault(inp, []).append(n)
        for o in n.output:
            out2node[o] = n

    # Tensors that must stay FP32: every output of a blocked node + every
    # output of our wrap_islands Cast(to=FP32) nodes.
    fp32_tensors = set()
    for node in model.graph.node:
        if node.name in blocked_names:
            for o in node.output:
                fp32_tensors.add(o)
        # Cast(to=FP32) outputs — these are the island input wrappers
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT:
                    for o in node.output:
                        fp32_tensors.add(o)

    print(f"  FP32 tensors: {len(fp32_tensors)}")

    # === Convert initializers ===
    # An initializer "feeds an FP32 island" if all of its consumers are
    # either blocked themselves, or are Cast(to=FP32) wrappers whose
    # eventual consumers are all blocked. We walk through Cast(to=FP32)
    # nodes transparently.
    def initializer_feeds_island_only(name):
        consumers = in2nodes.get(name, [])
        if not consumers:
            return False
        for c in consumers:
            if c.name in blocked_names:
                continue
            # Allow walking through Cast(to=FP32) wrapper to see what
            # ultimately consumes it.
            if c.op_type == "Cast":
                to_val = None
                for a in c.attribute:
                    if a.name == "to":
                        to_val = a.i
                if to_val == TensorProto.FLOAT:
                    # All downstream of this Cast must also be blocked
                    cast_outputs = c.output
                    deep_consumers = []
                    for o in cast_outputs:
                        deep_consumers.extend(in2nodes.get(o, []))
                    if all(dc.name in blocked_names for dc in deep_consumers) and deep_consumers:
                        continue
            return False
        return True

    init_consumed_by_blocked = {init.name: initializer_feeds_island_only(init.name)
                                 for init in model.graph.initializer}

    n_init_fp16 = 0
    n_init_fp32_keep = 0
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if init_consumed_by_blocked.get(init.name, False):
            n_init_fp32_keep += 1
            continue
        # Convert FP32 -> FP16
        arr = numpy_helper.to_array(init).astype(np.float16)
        new_init = numpy_helper.from_array(arr, name=init.name)
        init.CopyFrom(new_init)
        n_init_fp16 += 1
    print(f"  initializers: {n_init_fp16} converted to FP16, "
          f"{n_init_fp32_keep} kept FP32")

    # === Convert Constant nodes (these are inline tensor literals) ===
    def constant_feeds_island_only(consumers):
        """Like initializer_feeds_island_only but for Constant outputs."""
        if not consumers:
            return False
        for c in consumers:
            if c.name in blocked_names:
                continue
            if c.op_type == "Cast":
                to_val = None
                for a in c.attribute:
                    if a.name == "to":
                        to_val = a.i
                if to_val == TensorProto.FLOAT:
                    deep_consumers = []
                    for o in c.output:
                        deep_consumers.extend(in2nodes.get(o, []))
                    if all(dc.name in blocked_names for dc in deep_consumers) and deep_consumers:
                        continue
            return False
        return True

    n_const_fp16 = 0
    n_const_fp32_keep = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        # Find the value attribute
        for a in node.attribute:
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                # Where is this Constant consumed?
                consumers = []
                for o in node.output:
                    consumers.extend(in2nodes.get(o, []))
                if constant_feeds_island_only(consumers):
                    n_const_fp32_keep += 1
                    continue
                arr = numpy_helper.to_array(a.t).astype(np.float16)
                new_t = numpy_helper.from_array(arr)
                a.t.CopyFrom(new_t)
                n_const_fp16 += 1
                break
    print(f"  Constant nodes: {n_const_fp16} converted to FP16, "
          f"{n_const_fp32_keep} kept FP32")

    # === Update value_info dtypes ===
    n_vi_fp16 = 0
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        if vi.name in fp32_tensors:
            continue
        vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        n_vi_fp16 += 1
    print(f"  value_info: {n_vi_fp16} set to FP16")

    # === Update graph output dtype to match interior dtype ===
    # The original graph has output "velocity" as FP32. After our changes,
    # the graph's last layer (likely a MatMul or Add) is FP16 (it's not
    # blocked). So the actual output is FP16. We cast back to FP32 to
    # preserve the runtime contract.
    for go in model.graph.output:
        if go.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        # Find producer
        prod = out2node.get(go.name)
        if prod is None:
            continue
        if prod.name in blocked_names:
            continue  # output is FP32 from blocked, no cast needed
        if go.name in fp32_tensors:
            continue  # already considered FP32
        # The output is now FP16 (produced by an FP16 trunk node). Insert
        # Cast(to=FP32) at the very end: rename the producer's output and
        # add a Cast node feeding the original output name.
        renamed_out = f"{go.name}__pre_final_fp32cast"
        cast_name = f"{go.name}__final_fp32cast"
        # Rewire the producer
        for i, oname in enumerate(prod.output):
            if oname == go.name:
                prod.output[i] = renamed_out
        # Add a value_info for the renamed (still FP16) output
        new_vi = helper.make_tensor_value_info(
            renamed_out, TensorProto.FLOAT16,
            [d.dim_param if d.HasField("dim_param") else d.dim_value
             for d in go.type.tensor_type.shape.dim])
        model.graph.value_info.append(new_vi)
        # Add the Cast node
        cast_node = helper.make_node(
            "Cast", inputs=[renamed_out], outputs=[go.name],
            name=cast_name, to=TensorProto.FLOAT)
        model.graph.node.append(cast_node)
        print(f"  added final Cast(FP16->FP32) for output {go.name}")

    return model


def convert_to_fp16(input_onnx, output_onnx, mode="minimal", bound_attn=True):
    """Load FP32 ONNX, identify FP32 islands, convert everything else to
    FP16, and save."""
    import onnx

    print(f"  loading {input_onnx} ({os.path.getsize(input_onnx)/1e6:.0f} MB)")
    model = onnx.load(input_onnx)
    # Materialise any external-data tensors INLINE so subsequent
    # numpy_helper.to_array calls don't try to re-read the .data sidecar
    # (which may be deleted, or located elsewhere from a tempdir-based
    # shape_inference fallback below). Walks initializers + Constant nodes.
    from onnx import numpy_helper as _nh
    _onnx_TP = onnx.TensorProto
    def _inline_tensor(t):
        if t.data_location != _onnx_TP.EXTERNAL:
            return
        # Read bytes via the existing external_data ref, then promote to raw_data.
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

    # Find FP32 islands FIRST (so RoPE mode can use the original Cast(to=FP32)
    # seeds that mark PyTorch's @autocast(enabled=False) regions).
    blocked_names = find_fp32_islands(model, mode=mode)

    # The original FP32 ONNX has 652 Cast(to=FP32) nodes — all no-ops
    # because the entire graph is already FP32. These were inserted by
    # PyTorch ONNX export to mark `.float()` and @autocast(enabled=False)
    # call sites. We strip them now (after finding islands by them).
    model = strip_noop_fp32_casts(model)

    # Pre-insert Cast(FP32) at every island input edge and Cast(FP16) at
    # every island output edge. These Casts are still no-ops in the
    # currently-FP32 graph; they become real boundaries after manual_convert.
    model = wrap_islands_with_casts(model, blocked_names)

    print(f"  manual conversion to FP16 (keeping {len(blocked_names)} nodes FP32)...")
    t0 = time.time()
    fp16_model = manual_convert_to_fp16(model, blocked_names)
    print(f"  converted in {time.time()-t0:.0f}s")

    # Walk the graph and insert Cast nodes wherever two operands of a
    # node disagree on dtype (FP16 vs FP32). This handles all the random
    # places where t5_hidden (FP32 input) feeds into an op alongside an
    # FP16 tensor, or where Cast(to=FP32) outputs reach FP16 consumers, etc.
    print(f"  fixing dtype mismatches with autocast insertion...")
    fp16_model = fix_dtype_mismatches(fp16_model, blocked_names)

    # The island passes above leave the whole O(L^2) attention core in FP32 —
    # see bound_attention_core() for why that is wrong and what it costs.
    if bound_attn:
        fp16_model = bound_attention_core(fp16_model)

    print(f"  saving to {output_onnx}")
    # Try a plain save first; if the protobuf exceeds 2GB, fall back to
    # external-data save (needed for sa3-m: its FP16-mixed proto is ~3GB
    # because most FP16 weights are inline but FP32 island weights doubled
    # them in size).
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

    # Quick dtype audit
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
    print(f"  Cast.to counts (1=FP32, 10=FP16): {dict(cast_to_dtypes)}")

    return output_onnx


def build_trt_engine(onnx_path, engine_path, workspace_gb=16):
    """Build TRT engine with STRONGLY_TYPED + sa3-sm-music profile."""
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
    for input_name, (lo, opt, hi) in _DIT_PROFILE.items():
        profile.set_shape(input_name, lo, opt, hi)
    cfg.add_optimization_profile(profile)
    print(f"  optimization profile: {len(_DIT_PROFILE)} input(s)")

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
    ap.add_argument("--input",  default="/tmp/dit_sm-music_fixed_v2.onnx",
                    help="Input FP32 ONNX")
    ap.add_argument("--onnx",   default="/tmp/dit_sm-music_fp16.onnx",
                    help="Output FP16-mixed ONNX (intermediate)")
    ap.add_argument("--engine", default="/weka2/cj/clod/sa3s/stable-audio-3/"
                                       "optimized/tensorRT/models/sm_90/"
                                       "sa3-sm-music/dit_fp16.trt",
                    help="Output TRT engine path")
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--mode", choices=("minimal", "rope", "full"),
                    default="rope",
                    help="FP32 island coverage: minimal=RMSNorm+Softmax only, "
                         "rope=+RoPE freq/apply_rotary, full=every Cast(FP32) "
                         "downstream region from the original ONNX.")
    ap.add_argument("--no-bound-attn", action="store_true",
                    help="Skip bound_attention_core(). NOT recommended: leaving the "
                         "RoPE island unbounded keeps QK^T + Softmax in FP32, which "
                         "blocks TRT's FMHA fuser (0 fused MHA nodes, 4.3x slower at "
                         "L=4096) for no accuracy gain. Only for reproducing the "
                         "pre-2026-07 engines.")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip ONNX conversion (reuse existing --onnx file)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip TRT build (just produce the ONNX)")
    args = ap.parse_args()

    if not args.skip_convert:
        print(f"━━━ Convert FP32 ONNX -> FP16-mixed ONNX (mode={args.mode}) ━━━")
        convert_to_fp16(args.input, args.onnx, mode=args.mode,
                             bound_attn=not args.no_bound_attn)

    if not args.skip_build:
        print("\n━━━ Build TRT engine ━━━")
        build_trt_engine(args.onnx, args.engine, args.workspace_gb)


if __name__ == "__main__":
    main()
