#!/usr/bin/env python3
"""Graft #47's calibrated fp8 scales onto OUR RoPE-baked bakedmin ONNX.

This is the transplant step that produces the *calibrated* medium-DiT fp8 tier
(`dit_fp8.onnx` / `dit_fp8.trt`). It keeps our fast "bakedmin" structure — baked
fp32 RoPE constant + bf16 fused MHA, built weakly-typed — while swapping in the
calibrated fp8 scale VALUES from @ryanontheinside's fp8 work (#47, feat/dit-fp8).

Pipeline (medium-DiT fp8 "calibrated bakedmin" tier)
----------------------------------------------------
  1. make_calib.py            capture a real-conditioning calibration .npz
                              (@ryanontheinside, #47)
  2. build_dit_fp8.py         max-PTQ + per-channel weight scales -> a
                              *calibrated* fp8 ONNX (@ryanontheinside, #47;
                              full pipeline lives in that PR, not this repo)
  3. build_dit_bf16.py        RoPE-bake OUR fp8-linear ONNX -> the bakedmin
                              structure (baked fp32 RoPE constant; this is the
                              SHARED baker for the bf16 AND fp8 tiers)
  4. transplant_scales.py     <-- THIS SCRIPT: copy step 2's calibrated scale
                              VALUES onto step 3's graph, so the shipped fp8
                              engine keeps bakedmin's bf16 fused-MHA speed AND
                              gains #47's worst-step fidelity.
  5. build_from_onnx.py sa3-m-fp8   compile the calibrated ONNX to dit_fp8.trt

Why transplant instead of shipping #47's engine directly: #47's calibrated ONNX
is fp16-derived and its engine uses fp16 attention + a runtime fp32 RoPE
island (+~2 ms/fwd). Our bakedmin structure keeps the faster bf16 fused MHA and
a baked fp32 RoPE constant. Calibration is a property of the SCALE VALUES, not of
the structure, so grafting #47's scales onto the bakedmin graph gives #47's
worst-step fidelity at bakedmin's speed (measured H200: 31.2 vs 30.8 ms/fwd, i.e.
speed-free; worst-step velocity-cos vs fp32 0.52->0.92 on adversarial seeds). A
from-scratch recalibration (model retrain) reruns steps 1-2 with #47's full
pipeline; everyday consumers just pull the published `dit_fp8.onnx`.

What it does
------------
OUR ropebaked ONNX has fp32 trunk, baked RoPE constant, and per-MatMul dedicated
fp8 Q/DQ with PER-TENSOR scales. Match every quantized Linear by weight-initializer
name (identical across both graphs) and swap ONLY the fp8 scale VALUES:
  * activation : per-tensor scalar  <- #47's real-conditioning amax/448
  * weight     : per-tensor -> PER-CHANNEL axis=0 (pre-transpose N/out axis)
                 <- #47's per-channel vector. New per-channel fp8-zero vectors are
                 added (the graph's global scalar `fp8_zero` is shared, so weight
                 Q/DQ are repointed to fresh vectors; activations keep the scalar).
The 5.8 GB fp32 weights (`dit_fp8lin.onnx.data`) are UNTOUCHED — only scales change;
TRT re-quantizes the weights at build with the new scales. The 2 `to_global_embed`
linears #47 leaves unquantized are kept as-is (bakedmin's original fp8), preserving
bakedmin's 176-fp8-GEMM structure. Verify the result with
`../scripts/verify_fp8_rope.py` (expect 176 fp8 GEMMs + 96 bf16 fused MHA).

Calibration approach + scale VALUES by @ryanontheinside (#47).

usage:
    python transplant_scales.py \
        --ours-onnx  onnx/sa3-m/dit_fp8lin_ropebaked.onnx \
        --calib-onnx <build_dit_fp8.py output>/dit_fp8_calib.onnx \
        --out        onnx/sa3-m/dit_fp8.onnx
"""
import os, sys, argparse
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, helper
from collections import defaultdict

E4M3_MAX = 448.0


def read_ext(tensor, base):
    meta = {d.key: d.value for d in tensor.external_data}
    path = os.path.join(base, meta["location"])
    with open(path, "rb") as f:
        f.seek(int(meta.get("offset", 0)))
        raw = f.read(int(meta["length"]))
    dt = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    return np.frombuffer(raw, dtype=dt).reshape([d for d in tensor.dims]).copy()


class G:
    def __init__(self, path, load_data=False):
        self.path = path; self.base = os.path.dirname(os.path.abspath(path))
        self.m = onnx.load(path, load_external_data=False)
        self.g = self.m.graph
        self.inits = {i.name: i for i in self.g.initializer}
        self.producer = {}; self.consumers = defaultdict(list)
        for n in self.g.node:
            for o in n.output: self.producer[o] = n
            for i in n.input:
                if i: self.consumers[i].append(n)

    def arr(self, name):
        if name in self.inits:
            t = self.inits[name]
            return read_ext(t, self.base) if t.data_location == TensorProto.EXTERNAL \
                else numpy_helper.to_array(t)
        p = self.producer.get(name)
        if p is not None and p.op_type == "Constant":
            return numpy_helper.to_array(p.attribute[0].t)
        return None

    def dims(self, name):
        return list(self.inits[name].dims) if name in self.inits else None

    def axis(self, node):
        for a in node.attribute:
            if a.name == "axis": return a.i
        return None


def scale_dtype(g, name):
    if name in g.inits: return g.inits[name].data_type
    return None


def trace_weight_init(g, tensor, hops=6):
    cur = tensor
    for _ in range(hops):
        if cur in g.inits: return cur
        p = g.producer.get(cur)
        if p is None: return None
        if p.op_type in ("Transpose", "DequantizeLinear", "QuantizeLinear", "Cast"):
            cur = p.input[0]
        else: return None
    return None


def build_map(g):
    out = {}
    for mm in g.g.node:
        if mm.op_type not in ("MatMul", "Gemm"): continue
        w_side = a_side = None
        for i in mm.input:
            wi = trace_weight_init(g, i)
            if wi is not None: w_side = (i, wi)
            else: a_side = i
        if w_side is None or a_side is None: continue
        w_in, w_init = w_side

        def find_dq(t, hops=4):
            cur = t
            for _ in range(hops):
                p = g.producer.get(cur)
                if p is None: return None
                if p.op_type == "DequantizeLinear": return p
                if p.op_type in ("Transpose", "Cast"): cur = p.input[0]
                else: return None
            return None

        wdq = find_dq(w_in)
        pa = g.producer.get(a_side); adq = pa if (pa and pa.op_type == "DequantizeLinear") else None
        if wdq is None or adq is None: continue
        wq = g.producer.get(wdq.input[0]); aq = g.producer.get(adq.input[0])
        out[w_init] = dict(matmul=mm.name, wq=wq, wdq=wdq, aq=aq, adq=adq)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours-onnx", required=True,
                    help="OUR RoPE-baked fp8-linear ONNX (build_dit_bf16.py output): "
                         "the bakedmin graph whose per-tensor scales are overwritten")
    ap.add_argument("--calib-onnx", required=True,
                    help="#47's calibrated fp8 ONNX (build_dit_fp8.py output): the "
                         "SOURCE of the calibrated activation + per-channel weight scales")
    ap.add_argument("--out", required=True, help="output calibrated bakedmin ONNX")
    a = ap.parse_args()

    print(f"loading OURS ({a.ours_onnx}) + #47 calib ({a.calib_onnx}) metadata ...", flush=True)
    go = G(a.ours_onnx); gc = G(a.calib_onnx)
    mo = build_map(go); mc = build_map(gc)
    both = sorted(set(mo) & set(mc)); only_ours = sorted(set(mo) - set(mc))
    print(f"matched linears: {len(both)}  | ours-only (kept as-is): {len(only_ours)} "
          f"-> {only_ours}", flush=True)

    # index #47 scale arrays by weight_init
    c47scale = {}
    for w in both:
        cwdq = mc[w]['wdq']; cadq = mc[w]['adq']
        wscale = gc.arr(cwdq.input[1]); waxis = gc.axis(cwdq)
        ascale = gc.arr(cadq.input[1])
        c47scale[w] = dict(wscale=np.asarray(wscale), waxis=waxis, ascale=np.asarray(ascale))

    # sanity: #47 per-channel weight scale ~= amax(our fp32 weight channel)/448
    checked = 0; maxrel = 0.0
    for w in both[:6]:
        W = go.arr(w)  # our fp32 weight [N, K]
        if W is None: continue
        amax = np.abs(W.astype(np.float64)).max(axis=1)  # per output channel N
        want = amax / E4M3_MAX
        got = c47scale[w]['wscale'].astype(np.float64).ravel()
        if got.size == want.size:
            rel = np.abs(got - want) / (want + 1e-12)
            maxrel = max(maxrel, float(np.median(rel)))
            checked += 1
    print(f"weight-scale amax cross-check on {checked} layers: median rel-diff ~ {maxrel:.3f} "
          f"(~0 means #47 scales fit OUR fp32 weights and axis=0 is right)", flush=True)

    # ---- edit OUR graph in place ----
    inits_by_name = {i.name: i for i in go.g.initializer}

    def set_init_array(name, arr):
        """Replace an existing (inline) initializer's data with arr, same name."""
        new = numpy_helper.from_array(np.ascontiguousarray(arr), name=name)
        old = inits_by_name.get(name)
        if old is not None: old.CopyFrom(new)
        else: go.g.initializer.append(new); inits_by_name[name] = new

    def add_fp8_zeros(name, n):
        """fp8 E4M3 zero = byte 0x00; build via raw bytes (robust vs ml_dtypes)."""
        t = helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, [n], vals=b"\x00" * n, raw=True)
        old = inits_by_name.get(name)
        if old is not None: old.CopyFrom(t)
        else: go.g.initializer.append(t); inits_by_name[name] = t

    def set_axis(node, ax):
        found = False
        for at in node.attribute:
            if at.name == "axis": at.i = ax; found = True
        if not found:
            node.attribute.append(helper.make_attribute("axis", ax))

    n_w = n_a = 0
    for w in both:
        o = mo[w]; c = c47scale[w]
        N = go.dims(w)[0]
        # weight scale is fp32 (matches bakedmin's fp32 scale -> DQ output dtype unchanged)
        wscale = c['wscale'].ravel().astype(np.float32)
        assert wscale.size == N, f"{w}: wscale {wscale.size} != N {N}"
        ascale = np.asarray(c['ascale']).ravel().astype(np.float32)[0]  # scalar fp32
        # --- weight Q + DQ: per-channel axis=0, new per-channel fp8 zero-point vector ---
        wzp_name = o['wdq'].input[1].replace("wscale", "wzpvec")
        if wzp_name == o['wdq'].input[1]:
            wzp_name = o['wdq'].input[1] + "_zpvec"
        add_fp8_zeros(wzp_name, N)
        for nd in (o['wq'], o['wdq']):
            set_axis(nd, 0)
            set_init_array(nd.input[1], wscale)          # scalar->vector, same name, fp32
            if len(nd.input) > 2 and nd.input[2]:
                nd.input[2] = wzp_name                    # repoint off shared scalar fp8_zero
        # --- activation Q + DQ: scalar value swap, keep shared scalar fp8_zero zp ---
        for nd in (o['aq'], o['adq']):
            set_init_array(nd.input[1], np.float32(ascale))
        n_w += 1; n_a += 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    onnx.save(go.m, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)  "
          f"edited {n_w} weight + {n_a} act quantizers", flush=True)

    # link the external-data sidecar(s) next to the output (weights are shared, untouched)
    locs = {d.value for i in go.g.initializer if i.data_location == TensorProto.EXTERNAL
            for d in i.external_data if d.key == "location"}
    for loc in sorted(locs):
        link = os.path.join(os.path.dirname(os.path.abspath(a.out)), loc)
        tgt = os.path.realpath(os.path.join(go.base, loc))
        if os.path.exists(link) or os.path.islink(link):
            print(f"external data {loc}: already present -> {os.path.realpath(link)}", flush=True)
        else:
            os.symlink(tgt, link)
            print(f"external data {loc}: symlinked -> {tgt}", flush=True)


if __name__ == "__main__":
    main()
