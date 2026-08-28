#!/usr/bin/env python3
"""Recompute the SAME-S fp8 ENCODER activation scales with AMAX.

quantize/enc_quant_sames.py sets each scale to

    max(percentile(|activations|, 99.9) / 448, 1e-4)

where the percentile is taken over a 400,000-element RANDOM SUBSAMPLE per calibration clip — so
the tail is discarded twice: once by the percentile and once by the subsampling. The SAME-S
DECODER's own quantiser uses amax, and the two disagree by 26x across layers.

Same ORT capture as the original (same base ONNX, same 6 calibration clips, same MatMul selection
rule) with `amax` in place of the percentile, computed exactly over the full tensor. Only the 13
activation-scale initializers are rewritten; the quantised WEIGHTS are untouched.

Mapping is exact, not positional: the quantiser writes its scale as `{pfx}_asc` where
pfx = matmul_node_name.strip("/").replace("/", "_").

usage: recalib_enc_sames_fp8.py --fp8 enc_sames_fp8_pad.onnx --out enc_sames_fp8_amax.onnx
"""
import argparse, os
import numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper as nh
E4M3 = 448.0
WD = "/weka2/cj/clod/sames_fp8"

def warr(inits, prod, n):
    w = n.input[1]
    if w in inits: return w, nh.to_array(inits[w])
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits:
        return p.input[0], nh.to_array(inits[p.input[0]])
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", required=True, help="the fp8 encoder ONNX to correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="base ONNX to capture on (default: HF bf16 encoder)")
    ap.add_argument("--calib", default=f"{WD}/calib_audio_sames.npz")
    ap.add_argument("--work", default="/weka2/cj/tmp/_enc_amax.onnx")
    a = ap.parse_args()
    base = a.base
    if not base:
        from huggingface_hub import hf_hub_download
        base = hf_hub_download("stabilityai/stable-audio-3-optimized",
                               "onnx/same-s/enc_bf16.onnx")
    m = onnx.load(base, load_external_data=True); g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    tin = {}
    for n in g.node:
        if n.op_type == "MatMul":
            ws, W = warr(inits, prod, n)
            if ws is not None and W.ndim == 2: tin[n.name] = n.input[0]
    have = {o.name for o in g.output}
    for t in set(tin.values()):
        if t not in have: g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(m, a.work, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(a.work) + ".data", size_threshold=1024)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(a.work, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ks = list(tin); ts = [tin[k] for k in ks]
    aud = np.load(a.calib)["audio"].astype(np.float32)
    amax = {k: 0.0 for k in ks}
    for ci in range(aud.shape[0]):
        for k, act in zip(ks, sess.run(ts, {iname: aud[ci:ci + 1]})):
            amax[k] = max(amax[k], float(np.abs(act).max()))     # exact, whole tensor
        print(f"  captured clip {ci}", flush=True)
    del sess
    mq = onnx.load(a.fp8, load_external_data=True)
    qi = {i.name: i for i in mq.graph.initializer}
    new = {k.strip("/").replace("/", "_") + "_asc": v / E4M3 for k, v in amax.items()}
    hits, ratios = 0, []
    for name, init in qi.items():
        if name in new:
            old = float(np.asarray(nh.to_array(init)).ravel()[0])
            ratios.append(new[name] / old); hits += 1
            arr = nh.to_array(init)
            init.CopyFrom(nh.from_array(np.full(arr.shape, new[name], dtype=arr.dtype), name))
    if hits == 0:
        raise SystemExit("no activation scales matched — check the pfx mapping")
    r = np.array(ratios)
    print(f"\n  {hits} activation scales rewritten with amax/448")
    print(f"  amax/old ratio: min {r.min():.1f}x  median {np.median(r):.1f}x  max {r.max():.1f}x")
    print(f"  clip points now: {min(new.values())*E4M3:.3f} .. {max(new.values())*E4M3:.3f}")
    onnx.save(mq, a.out, save_as_external_data=False)
    print(f"  wrote {a.out}")

if __name__ == "__main__":
    main()
