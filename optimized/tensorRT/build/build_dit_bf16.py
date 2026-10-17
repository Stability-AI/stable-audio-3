#!/usr/bin/env python3
"""Bake RoPE's cos/sin tables as ONNX constants, deleting the runtime angle chain.

Why
---
The medium DiT's RoPE angle is `t (x) inv_freq` with `inv_freq[0] == 1.0` exactly,
so the angle equals the position index for the fastest-rotating pair and reaches
~4159 rad at the 6-minute anchor. bf16's ULP up there is 32 rad — five whole
rotations — so a bf16 angle destroys 16/32 frequency pairs, the latent inflates
2.5x over 8 sampling steps, and the decoder clips 2-3%. See ROPE_PRECISION.md.

Every published fix fences that region into an FP32 island. But the tables depend
on NOTHING except position — not activations, not the timestep, not the prompt —
so they are compile-time constants. And the same ablation shows the table VALUES
are bf16-safe (rounding cos/sin outputs to bf16: latent std 0.9161 vs fp32's
0.9162, cos 0.9998) because they live in [-1, 1] where bf16's ULP is ~0.004. Only
the *pre-trig magnitude* needs precision, and precision is free on the host.

So: precompute cos/sin in fp64 on the host, embed as fp32 initializers, slice by
the dynamic sequence length, rewire all 96 trig sites to the two shared slices,
and let dead-code elimination delete the angle chain. No fp32 island is then
needed anywhere in the RoPE region — and bf16 (unlike fp16) needs no fp32 RMSNorm
islands either — so the whole graph can be uniformly bf16: maximum fusion, no
reformat boundaries, and one fewer fencing constraint blocking future fp8/NVFP4.

Graph facts (verified against the real ONNX, not assumed)
--------------------------------------------------------
  inv_freq = 1/(10000**(arange(0,32,2)/32))  -> 16 values (NOT 32), inv_freq[0]=1.0
  /transformer/Range(0, T, 1) -> Cast(f32) -> Div(/1.0) -> Einsum('i,j->ij')
      -> Concat_2(axis=-1, [freqs, freqs])  ->  [T, 32]   (NOT [T, 64])
  per layer i: Cast_14 <- (layer i-1)'s Cast_17, i.e. one long no-op fp32 Cast chain
      Cast_14 -> {Cast_17 -> Slice_5 -> {Cos, Sin},
                  Mul_5(x T_q/T_k) -> Cast_23 -> Slice_8 -> {Cos_1, Sin_1},
                  Shape_18}        Mul_5 -> Shape_24
  Slice_5/Slice_8 are `angle[-T_q:]` / `angle[-T_k:]` on a table that already has
      exactly T rows, i.e. identity. T_q == T_k == T (self-attention), so the
      Mul by T_q/T_k is a multiply by exactly 1.0 and BOTH sites see the SAME
      angle -> one shared cos slice + one shared sin slice serves all 96.
  Shape_18 / Shape_24 read the angle tensor's SHAPE (to find rot_dim=32 for the
      q/k partial-rotary split), so they are rewired to the baked slice, which is
      [T, 32] as well. Only then does the angle chain become fully dead.
  T = L + 64 (64 prepended global tokens); profile max L=4096 -> max T=4160.
  There are 4 further Cos/Sin nodes in the graph (`/Cos`, `/Sin`,
      `/timestep_features/Cos`, `/timestep_features/Sin`) for the seconds_total /
      timestep Fourier features. Those are NOT RoPE and are left alone; sites are
      selected by reachability from /transformer/Einsum, not by op type.

This is the shared RoPE-baking producer for BOTH medium RoPE-baked tiers — it only
touches the RoPE angle chain, so it is agnostic to what else the graph carries:
  bf16 : --input dit.onnx        (raw fp32; inv_freq is EXTERNAL)
  fp8  : --input dit_fp8lin.onnx (fp8 E4M3 Q/DQ already on the Linears; inv_freq INLINE)
Downstream compile: build_from_onnx.py sa3-m-bf16 / sa3-m-fp8.

usage:
    python build_dit_bf16.py \
        --input  /path/to/onnx/sa3-m/dit.onnx \
        --output onnx/sa3-m/dit_bf16.onnx [--max-t 4160]

The input's 5.8 GB weights are NEVER loaded or rewritten: initializers keep their
external-data references and the output .onnx is written beside a link to the
original `<input>.onnx.data` sidecar.
"""
import argparse
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP32 = TensorProto.FLOAT
INT64_MAX = 2 ** 63 - 1
EINSUM = "/transformer/Einsum"
INV_FREQ = "dit.transformer.rotary_pos_emb.inv_freq"
T_SOURCE = "/transformer/Cast_1_output_0"   # int64 scalar T; the Range's limit input


def read_external(tensor, base_dir):
    """Read one external initializer's bytes directly.

    onnx's own loader rejects this repo's sidecar ("not regular file") on the weka
    mount even though it is one, so go to the bytes.
    """
    meta = {d.key: d.value for d in tensor.external_data}
    path = os.path.join(base_dir, meta["location"])
    with open(path, "rb") as f:
        f.seek(int(meta.get("offset", 0)))
        raw = f.read(int(meta["length"]))
    dt = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return np.frombuffer(raw, dtype=dt).reshape([d for d in tensor.dims]).copy()


def const_of(tname, producer, inits):
    """Resolve a tensor to a numpy constant, or None."""
    if tname in inits:
        t = inits[tname]
        if t.data_location == TensorProto.EXTERNAL:
            return None
        return numpy_helper.to_array(t)
    p = producer.get(tname)
    if p is None:
        return None
    if p.op_type == "Constant":
        return numpy_helper.to_array(p.attribute[0].t)
    if p.op_type == "Unsqueeze":
        a = const_of(p.input[0], producer, inits)
        if a is None:
            return None
        ax = const_of(p.input[1], producer, inits) if len(p.input) > 1 else None
        return np.expand_dims(a, int(ax.ravel()[0]) if ax is not None else 0)
    return None


def attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/path/to/sa3s/stable-audio-3-optimized/"
                                      "onnx/sa3-m/dit.onnx")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-t", type=int, default=4160,
                    help="rows to bake = max sequence length at self-attention "
                         "(= profile max L + 64 global tokens)")
    ap.add_argument("--link-data", action="store_true", default=True,
                    help="symlink the input's external-data sidecar next to the output")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    t0 = time.time()

    print(f"loading graph metadata from {src} (weights NOT loaded) ...", flush=True)
    model = onnx.load(str(src), load_external_data=False)
    g = model.graph
    print(f"  {len(g.node)} nodes · {len(g.initializer)} initializers · "
          f"opset {[o.version for o in model.opset_import]} · ir {model.ir_version}",
          flush=True)

    producer, consumers = {}, defaultdict(list)
    for n in g.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            if i:
                consumers[i].append(n)
    inits = {i.name: i for i in g.initializer}
    graph_outputs = {o.name for o in g.output}

    # ── 1. the real inv_freq ────────────────────────────────────────────────
    if INV_FREQ not in inits:
        raise SystemExit(f"initializer {INV_FREQ} not found")
    _ivf = inits[INV_FREQ]
    if _ivf.data_location == TensorProto.EXTERNAL:
        inv_freq = read_external(_ivf, str(src.parent))   # external in fp32 dit.onnx
    else:
        inv_freq = numpy_helper.to_array(_ivf)            # inline in the fp8-linear ONNX
    n_half = inv_freq.size
    print(f"\ninv_freq: {n_half} values, dtype={inv_freq.dtype}, "
          f"[0]={inv_freq[0]!r}, [-1]={inv_freq[-1]!r}", flush=True)
    expect = (1.0 / (10000.0 ** (np.arange(0, 2 * n_half, 2, dtype=np.float32)
                                 / (2 * n_half)))).astype(np.float32)
    print(f"  bit-identical to 1/(10000**(arange(0,{2*n_half},2)/{2*n_half})): "
          f"{np.array_equal(inv_freq, expect)}", flush=True)
    if float(inv_freq[0]) != 1.0:
        print("  NOTE: inv_freq[0] != 1.0 — the angle no longer equals the position "
              "index; the bf16 hazard is milder than documented.", flush=True)

    # ── 2. bake the tables ─────────────────────────────────────────────────
    T = args.max_t
    rot = 2 * n_half                     # Concat_2 width
    # Reproduce the fp32 graph's angle exactly (Einsum outer product = one exactly
    # rounded fp32 multiply; Cast int64->fp32 of i<=4159 is exact; Div by 1.0 is
    # exact), then evaluate the trig in float64 and round the RESULT to fp32.
    pos32 = np.arange(T, dtype=np.float32)
    ang32 = (pos32[:, None] * inv_freq[None, :]).astype(np.float32)     # [T, n_half]
    ang64 = ang32.astype(np.float64)
    cos_t = np.cos(ang64).astype(np.float32)
    sin_t = np.sin(ang64).astype(np.float32)
    cos_t = np.concatenate([cos_t, cos_t], axis=-1)                    # [T, rot]
    sin_t = np.concatenate([sin_t, sin_t], axis=-1)
    assert cos_t.shape == (T, rot)

    # how much did fp32 angle rounding cost vs an exact-in-fp64 angle?
    ang_exact = np.arange(T, dtype=np.float64)[:, None] * inv_freq.astype(np.float64)[None, :]
    d_ang = np.abs(ang64 - ang_exact).max()
    d_cos = np.abs(np.cos(ang64) - np.cos(ang_exact)).max()
    # what bf16 would have done to the angle, for the record
    def to_bf16(a):
        u = a.astype(np.float32).view(np.uint32)
        u = ((u + 0x8000 + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return u.view(np.float32)
    d_bf16 = np.abs(np.cos(to_bf16(ang32).astype(np.float64)) - np.cos(ang64)).max()
    print(f"\nbaked tables: cos/sin {cos_t.shape} fp32 "
          f"({cos_t.nbytes/1e6:.2f} MB each)", flush=True)
    print(f"  max angle          : {ang32.max():.1f} rad (T={T}, pair 0 uses inv_freq=1)",
          flush=True)
    print(f"  fp32-angle rounding: max |dangle|={d_ang:.2e} rad -> max |dcos|={d_cos:.2e}",
          flush=True)
    print(f"  bf16-angle (avoided): max |dcos|={d_bf16:.4f}", flush=True)

    # ── 3. locate the 96 RoPE trig sites by reachability from the Einsum ────
    if EINSUM not in {n.name for n in g.node}:
        raise SystemExit(f"node {EINSUM} not found")
    einsum = next(n for n in g.node if n.name == EINSUM)
    reach, q = set(), deque(einsum.output)
    while q:
        for c in consumers[q.popleft()]:
            if c.name in reach:
                continue
            reach.add(c.name)
            q.extend(c.output)
    all_trig = [n for n in g.node if n.op_type in ("Cos", "Sin")]
    rope_trig = [n for n in all_trig if n.name in reach]
    other_trig = [n for n in all_trig if n.name not in reach]
    print(f"\ntrig nodes: {len(all_trig)} total · {len(rope_trig)} reachable from "
          f"{EINSUM} (RoPE) · {len(other_trig)} not (left alone)", flush=True)
    print(f"  left alone: {sorted(n.name for n in other_trig)}", flush=True)
    if len(rope_trig) != 96:
        print(f"  WARNING: expected 96 RoPE trig nodes, found {len(rope_trig)}", flush=True)

    # ── 4. verify every site really is cos/sin(angle[0:T]) ─────────────────
    def is_angle(t, seen=None):
        """tensor carries the [T, rot] angle (identity Casts / x1.0 Mul / identity Slice)"""
        seen = seen or set()
        if t in einsum.output:
            return True
        if t in seen:
            return False
        seen.add(t)
        p = producer.get(t)
        if p is None:
            return False
        if p.op_type in ("Cast", "Slice"):
            return is_angle(p.input[0], seen)
        if p.op_type == "Concat":
            return all(is_angle(i, seen) for i in p.input)
        if p.op_type == "Mul":
            return any(is_angle(i, seen) for i in p.input)
        return False

    problems = []
    for n in rope_trig:
        sl = producer.get(n.input[0])
        why = []
        if sl is None or sl.op_type != "Slice":
            why.append(f"input producer is {sl.op_type if sl else None}, not Slice")
        else:
            ends = const_of(sl.input[2], producer, inits)
            axes = const_of(sl.input[3], producer, inits) if len(sl.input) > 3 else None
            steps = const_of(sl.input[4], producer, inits) if len(sl.input) > 4 else None
            if ends is None or int(ends.ravel()[0]) != INT64_MAX:
                why.append(f"ends != INT64_MAX ({ends})")
            if axes is None or int(axes.ravel()[0]) != 0:
                why.append(f"axes != [0] ({axes})")
            if steps is None or int(steps.ravel()[0]) != 1:
                why.append(f"steps != [1] ({steps})")
            # starts must be -Gather(Shape(q_or_k), 3) == -T, making the slice identity
            sp = producer.get(sl.input[1])
            ng = producer.get(sp.input[0]) if sp is not None and sp.op_type == "Unsqueeze" else None
            if ng is None or ng.op_type != "Neg":
                why.append("starts is not Unsqueeze(Neg(...)) i.e. not angle[-T:]")
            else:
                gt = producer.get(ng.input[0])
                if gt is None or gt.op_type != "Gather":
                    why.append("Neg source is not a Gather")
                else:
                    ix = const_of(gt.input[1], producer, inits)
                    sh = producer.get(gt.input[0])
                    if ix is None or int(ix.ravel()[0]) != 3:
                        why.append(f"Gather index != 3 ({ix}) — not the seq dim of [B,2,H,T,D]")
                    if sh is None or sh.op_type != "Shape":
                        why.append("Gather source is not a Shape")
            if not is_angle(sl.input[0]):
                why.append("sliced tensor does not trace to the RoPE Einsum via "
                           "identity Cast/Mul/Concat only")
        if len(consumers[n.output[0]]) == 0 or n.output[0] in graph_outputs:
            why.append("output is a graph output or unused")
        if why:
            problems.append((n.name, why))
    print(f"  structural verification: {len(rope_trig)-len(problems)}/{len(rope_trig)} OK",
          flush=True)
    for nm, why in problems[:8]:
        print(f"    ✗ {nm}: {why}", flush=True)
    if problems:
        raise SystemExit("aborting: not every RoPE site matches the verified pattern")

    # every Mul on the angle path must be by exactly T_q/T_k == 1.0, else the two
    # sites do not share a table
    n_ratio_mul = 0
    for n in g.node:
        if n.op_type != "Mul" or n.name not in reach:
            continue
        if not any(is_angle(i) for i in n.input):
            continue
        other = [i for i in n.input if not is_angle(i)]
        if len(other) != 1:
            raise SystemExit(f"{n.name}: ambiguous angle Mul {list(n.input)}")
        d = producer.get(other[0])
        ok = False
        if d is not None and d.op_type == "Div":
            ca, cb = producer.get(d.input[0]), producer.get(d.input[1])
            if ca is not None and cb is not None and ca.op_type == cb.op_type == "Cast":
                ga, gb = producer.get(ca.input[0]), producer.get(cb.input[0])
                if (ga is not None and gb is not None and ga.op_type == gb.op_type == "Gather"
                        and int(const_of(ga.input[1], producer, inits).ravel()[0]) == 3
                        and int(const_of(gb.input[1], producer, inits).ravel()[0]) == 3):
                    sa, sb = producer.get(ga.input[0]), producer.get(gb.input[0])
                    ok = (sa is not None and sb is not None
                          and sa.op_type == sb.op_type == "Shape")
        if not ok:
            raise SystemExit(f"{n.name}: angle scaled by something other than T_q/T_k")
        n_ratio_mul += 1
    print(f"  angle Mul nodes, all by T_q/T_k (== 1.0 in self-attention): {n_ratio_mul}",
          flush=True)

    # ── 5. build the baked subgraph ────────────────────────────────────────
    if T_SOURCE not in producer:
        raise SystemExit(f"{T_SOURCE} (dynamic T) not found")
    new_inits = [
        numpy_helper.from_array(cos_t, "rope_cos_table"),
        numpy_helper.from_array(sin_t, "rope_sin_table"),
        numpy_helper.from_array(np.array([0], np.int64), "rope_start0"),
        numpy_helper.from_array(np.array([0], np.int64), "rope_axis0"),
        numpy_helper.from_array(np.array([1], np.int64), "rope_step1"),
        numpy_helper.from_array(np.array([0], np.int64), "rope_unsq_axis"),
    ]
    new_nodes = [
        helper.make_node("Unsqueeze", [T_SOURCE, "rope_unsq_axis"], ["rope_T_1d"],
                         name="rope_bake/T_1d"),
        helper.make_node("Slice", ["rope_cos_table", "rope_start0", "rope_T_1d",
                                   "rope_axis0", "rope_step1"], ["rope_cos"],
                         name="rope_bake/cos_slice"),
        helper.make_node("Slice", ["rope_sin_table", "rope_start0", "rope_T_1d",
                                   "rope_axis0", "rope_step1"], ["rope_sin"],
                         name="rope_bake/sin_slice"),
    ]

    # ── 6. rewire ──────────────────────────────────────────────────────────
    rename = {}
    for n in rope_trig:
        rename[n.output[0]] = "rope_cos" if n.op_type == "Cos" else "rope_sin"
    drop = {id(n) for n in rope_trig}

    # Shape consumers of angle tensors must move to the baked slice (identical
    # [T, rot] shape) or the angle chain stays alive purely to be measured.
    shape_rewired = 0
    for n in g.node:
        if n.op_type == "Shape" and n.input[0] not in rename and is_angle(n.input[0]):
            n.input[0] = "rope_cos"
            shape_rewired += 1
    print(f"\nrewired {len(rename)} trig outputs -> rope_cos/rope_sin "
          f"({sum(1 for v in rename.values() if v=='rope_cos')} cos / "
          f"{sum(1 for v in rename.values() if v=='rope_sin')} sin)", flush=True)
    print(f"rewired {shape_rewired} Shape nodes off the angle chain onto rope_cos",
          flush=True)

    kept = [n for n in g.node if id(n) not in drop]
    n_rewired = 0
    for n in kept:
        for i, t in enumerate(n.input):
            if t in rename:
                n.input[i] = rename[t]
                n_rewired += 1
    print(f"  {n_rewired} node inputs repointed", flush=True)

    # No node outside the angle chain itself may still read an angle tensor —
    # otherwise DCE cannot remove the chain. (Chain-internal nodes consume each
    # other by construction; those go away together.)
    def all_outputs_angle(n):
        return all(is_angle(o) for o in n.output)
    escapes = [(n.name, n.op_type) for n in kept
               if not all_outputs_angle(n)
               and any(is_angle(i) for i in n.input if i)]
    if escapes:
        raise SystemExit(f"angle values still escape the chain at {len(escapes)} nodes: "
                         f"{escapes[:6]}")
    print(f"  angle values escape the chain at 0 nodes (chain is now self-contained)",
          flush=True)

    # ── 7. dead-code elimination from the graph outputs ────────────────────
    kept_by_out = {}
    for n in kept + new_nodes:
        for o in n.output:
            kept_by_out[o] = n
    live, q = set(), deque(o.name for o in g.output)
    while q:
        t = q.popleft()
        p = kept_by_out.get(t)
        if p is None or id(p) in live:
            continue
        live.add(id(p))
        q.extend(i for i in p.input if i)
    final = [n for n in kept + new_nodes if id(n) in live]
    dead = [n for n in kept if id(n) not in live]
    dead_hist = defaultdict(int)
    for n in dead:
        dead_hist[n.op_type] += 1
    print(f"\nDCE: {len(g.node)} -> {len(final)} nodes "
          f"({len(rope_trig)} trig removed, {len(dead)} newly dead, "
          f"{len(new_nodes)} added)", flush=True)
    print(f"  dead by op: {dict(sorted(dead_hist.items(), key=lambda kv: -kv[1]))}", flush=True)
    for probe in ("/transformer/Range", EINSUM, "/transformer/Concat_2", "/transformer/Cast_2",
                  "/transformer/Div", "/transformer/layers.0/self_attn/Slice_5",
                  "/transformer/layers.23/self_attn/Mul_5"):
        gone = probe in {n.name for n in dead}
        print(f"  {'removed' if gone else 'STILL PRESENT'}: {probe}", flush=True)
    survivors = [(n.name, n.op_type) for n in final
                 if not n.name.startswith("rope_bake/")
                 and any(is_angle(i) for i in n.input if i)]
    if survivors:
        raise SystemExit(f"angle chain survived DCE at {len(survivors)} nodes: {survivors[:8]}")
    print(f"  angle chain fully eliminated: 0 surviving nodes touch it", flush=True)

    # topological order: the new nodes only depend on T_SOURCE + initializers, so
    # splice each immediately before its first consumer.
    pending = {n.output[0]: n for n in new_nodes}
    ordered, emitted = [], set()

    def emit(n):
        if id(n) in emitted:
            return
        emitted.add(id(n))
        for i in n.input:
            if i in pending:
                emit(pending[i])
        ordered.append(n)

    for n in final:
        if id(n) in {id(x) for x in new_nodes}:
            continue
        for i in n.input:
            if i in pending and id(pending[i]) not in emitted:
                emit(pending[i])
        emit(n)
    for n in new_nodes:                      # any not yet placed (shouldn't happen)
        emit(n)
    assert len(ordered) == len(final), f"{len(ordered)} != {len(final)}"

    del g.node[:]
    g.node.extend(ordered)

    # ── 8. prune initializers that nothing reads any more ──────────────────
    used = {i for n in g.node for i in n.input if i}
    keep_i = [i for i in g.initializer if i.name in used]
    pruned = [i.name for i in g.initializer if i.name not in used]
    del g.initializer[:]
    g.initializer.extend(keep_i + new_inits)
    print(f"\ninitializers: {len(keep_i)} kept + {len(new_inits)} new "
          f"(pruned {len(pruned)}: {pruned})", flush=True)

    # ── 9. save: metadata only; external weights stay in the original sidecar
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))                # new tables ride inline (1.1 MB)
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    locs = {d.value for i in g.initializer if i.data_location == TensorProto.EXTERNAL
            for d in i.external_data if d.key == "location"}
    for loc in sorted(locs):
        link = out.parent / loc
        target = src.parent / loc
        if link.exists() or link.is_symlink():
            print(f"  external data {loc}: already present -> "
                  f"{os.path.realpath(link)}", flush=True)
        elif args.link_data:
            link.symlink_to(target)
            print(f"  external data {loc}: symlinked -> {target}", flush=True)
        else:
            print(f"  external data {loc}: MISSING, expected at {link}", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
