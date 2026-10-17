#!/usr/bin/env python3
"""Splice the limiter subgraph into a SAME-S / SAME-L decoder ONNX.

Both decoders end in the same six nodes:

    ... -> Cast(FLOAT) -> /Clip(-1,+1) -> /Mul(32767) -> /Clip_1(+-32767) -> Cast(int32)
        -> Transpose -> pcm

The graft removes `/Clip` — which is what currently destroys the overshoot before anyone can
see it — and puts the limiter in its place, so the engine emits already-limited int32 PCM and
every caller gets it for free. `/Clip_1` stays as the belt-and-braces int16 bound.

The limiter nodes come from limiter.onnx, exported from the torch reference, so the semantics
are the reference implementation's rather than a hand-rebuild.

A second mode, `--noclip`, removes `/Clip` and inserts nothing. That engine emits the decoder's
RAW float scaled to int32 — no clipping, no limiting — which is the only way to verify the
limiter exactly inside the real engine rather than against a differently-numbered decoder.

usage: graft_limiter.py <in.onnx> <out.onnx> [--noclip]
"""
import os
import sys
import onnx
from onnx import helper

LIM = "limiter.onnx"
PREFIX = "lim/"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    noclip = "--noclip" in sys.argv[3:]
    print(f"loading {src} ...", flush=True)
    m = onnx.load(src)
    g = m.graph
    lim = onnx.load(LIM).graph

    clip = [n for n in g.node if n.op_type == "Clip" and n.name.endswith("/Clip")]
    assert len(clip) == 1, f"expected exactly one /Clip, found {[n.name for n in clip]}"
    clip = clip[0]
    pre, post = clip.input[0], clip.output[0]
    consumers = [(n, i) for n in g.node for i, t in enumerate(n.input) if t == post]
    assert len(consumers) == 1 and consumers[0][0].op_type == "Mul", \
        f"/Clip output goes somewhere unexpected: {[(n.op_type, n.name) for n, _ in consumers]}"
    mul = consumers[0][0]
    print(f"  splice point: {pre}  ->  [{'nothing (--noclip)' if noclip else 'limiter'}]  "
          f"->  {mul.name}")
    if noclip:
        # Emit the decoder's RAW FLOAT instead of int32 PCM: drop Mul/Clip_1/Cast and hand the
        # pre-clip tensor straight to the final Transpose. Going via int32 would quantise the
        # reference to a 1/32767 grid before the limiter saw it, which shifts the gain by ~1 LSB
        # and muddies an exactness check that should be clean.
        tr = [n for n in g.node if n.op_type == "Transpose"
              and n.output[0] == g.output[0].name][0]
        # walk back from the Transpose to the splice point, dropping exactly the tail.
        # Stopping on op_type is not enough — the node that PRODUCES `pre` is itself a Cast,
        # so the walk would step straight past the splice and delete it.
        drop = []
        cur = tr.input[0]
        while cur != pre:
            n = [x for x in g.node if cur in x.output]
            assert n, f"walked off the graph at {cur}"
            drop.append(n[0])
            cur = n[0].input[0]
            assert len(drop) < 8, f"tail longer than expected: {[x.op_type for x in drop]}"
        for n in drop:
            g.node.remove(n)
        tr.input[0] = pre
        g.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        del g.value_info[:]
        print(f"  raw-float mode: dropped {[n.op_type for n in drop]}, output is FLOAT")
        onnx.checker.check_model(m, full_check=False)
        print(f"saving {dst} ...", flush=True)
        onnx.save(m, dst)
        print("ok")
        return
        g.node.remove(clip)
        mul.input[0] = pre
        del g.value_info[:]
        # the +-32767 clamp downstream would re-impose a bound, so widen it to int32 range
        for n in g.node:
            if n.op_type == "Clip" and n.name.endswith("/Clip_1"):
                import numpy as np
                from onnx import numpy_helper
                lo = helper.make_tensor(PREFIX + "lo", onnx.TensorProto.FLOAT, [],
                                        [-2.0e9])
                hi = helper.make_tensor(PREFIX + "hi", onnx.TensorProto.FLOAT, [], [2.0e9])
                g.initializer.extend([lo, hi])
                n.input[1], n.input[2] = lo.name, hi.name
        onnx.checker.check_model(m, full_check=False)
        print(f"saving {dst} ...", flush=True)
        onnx.save(m, dst)
        print("ok")
        return

    # ── rename everything in the limiter graph so it cannot collide ──
    ren = {}

    def r(t):
        if t in ("", None):
            return t
        if t == "lim_in":
            return pre                      # feed it the decoder's pre-clip tensor
        if t == "lim_out":
            return PREFIX + "out"
        ren.setdefault(t, PREFIX + t)
        return ren[t]

    inits = []
    for w in lim.initializer:
        w.name = r(w.name)
        inits.append(w)
    nodes = []
    for n in lim.node:
        n.name = PREFIX + (n.name or n.op_type)
        n.input[:] = [r(t) for t in n.input]
        n.output[:] = [r(t) for t in n.output]
        nodes.append(n)

    # ── rewire and insert, keeping topological order ──
    g.node.remove(clip)
    idx = list(g.node).index(mul)
    for k, n in enumerate(nodes):
        g.node.insert(idx + k, n)
    mul.input[0] = PREFIX + "out"
    g.initializer.extend(inits)
    # limiter value_info would be stale (it was inferred for a standalone graph)
    del g.value_info[:]

    # ── round-to-nearest instead of the graph's truncating Cast ──────────────
    # Cast(float -> int32) truncates toward zero, a biased quantiser: measured -79.6 dB
    # error-to-signal versus -90.4 dB for round-to-nearest, with a 3.4x larger DC term. TRT has
    # no INT16 type, so int32 is the narrowest the engine can emit and the host's .to(int16) is
    # lossless — which means rounding HERE fully determines the final sample values.
    if os.environ.get("GRAFT_NO_ROUND") != "1":
        tr = [n for n in g.node if n.output and n.output[0] == g.output[0].name][0]
        cast = [n for n in g.node if tr.input[0] in n.output][0]
        to = {at.name: at.i for at in cast.attribute}.get("to")
        assert cast.op_type == "Cast" and to == onnx.TensorProto.INT32, \
            f"expected a Cast->INT32 before the output Transpose, got {cast.op_type} to={to}"
        rnd = helper.make_node("Round", [cast.input[0]], [PREFIX + "rounded"],
                               name=PREFIX + "round_to_nearest")
        cast.input[0] = PREFIX + "rounded"
        g.node.insert(list(g.node).index(cast), rnd)
        print(f"  + Round before {cast.name} (round-to-nearest, was truncation)")

    print(f"  +{len(nodes)} nodes, +{len(inits)} initializers, -1 Clip", flush=True)
    onnx.checker.check_model(m, full_check=False)
    print(f"saving {dst} ...", flush=True)
    onnx.save(m, dst)
    print("ok")


if __name__ == "__main__":
    main()
