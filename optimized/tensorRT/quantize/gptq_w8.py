#!/usr/bin/env python3
"""GPTQ int8 per-channel weight-only quantization of a decoder onnx. Captures each linear's input
activations (ORT), builds the Hessian H = AᵀA, and quantizes weights column-by-column with
inverse-Hessian error compensation (the OBQ/GPTQ update). Writes a W8 onnx. argv: <fp32_onnx> <out>
[Lslice]."""
import os, sys
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
DEC, OUT = sys.argv[1], sys.argv[2]
LSL = int(sys.argv[3]) if len(sys.argv) > 3 else 768
LATS = np.load("/weka2/cj/clod/sames_fp8/calib_latents.npz")["latents"]
m = onnx.load(DEC, load_external_data=True); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
def warr(n):
    w = n.input[1]
    if w in inits: return w, numpy_helper.to_array(inits[w]), False, None
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits: return p.input[0], numpy_helper.to_array(inits[p.input[0]]), True, p
    return None, None, None, None
linears = []
for n in g.node:
    if n.op_type == "MatMul":
        ws, W, vt, tn = warr(n)
        if ws is not None and W.ndim == 2: linears.append((n, ws, W, vt, tn))
print(f"{len(linears)} linears; capturing activations (L={LSL}) ...", flush=True)
# --- capture inputs via ORT (add as temp outputs, then restore) ---
in_names = [n.input[0] for n, *_ in linears]
n0 = len(g.output)
existing = {o.name for o in g.output}
for t in in_names:
    if t not in existing: g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
onnx.save(m, "/weka2/cj/clod/sames_fp8/_gptq_cap.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="_gptq_cap.onnx.data", size_threshold=1024)
del g.output[n0:]
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("/weka2/cj/clod/sames_fp8/_gptq_cap.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
Hs = {n.name: None for n, *_ in linears}
for li in range(LATS.shape[0]):
    acts = sess.run(in_names, {iname: LATS[li:li + 1, :, :LSL].astype(np.float32)})
    for (n, *_), a in zip(linears, acts):
        A = a.reshape(-1, a.shape[-1]).astype(np.float64)
        Hs[n.name] = A.T @ A if Hs[n.name] is None else Hs[n.name] + A.T @ A
    print(f"  latent {li} done", flush=True)
# --- GPTQ ---
def gptq(W, H, scale, pd=0.01):
    W = W.astype(np.float64).copy(); cin = W.shape[1]; H = H.copy()
    dead = np.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0.0
    H[np.diag_indices(cin)] += pd * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T                      # upper-tri Cholesky of H⁻¹
    Qi = np.zeros(W.shape, dtype=np.int8); sc = scale
    for i in range(cin):
        w = W[:, i]; d = Hinv[i, i]
        qi = np.clip(np.round(w / sc), -127, 127); q = qi * sc
        Qi[:, i] = qi.astype(np.int8)
        W[:, i:] -= np.outer((w - q) / d, Hinv[i, i:])
    return Qi
new_nodes, new_inits = [], []
for (n, ws, W, vt, tn) in linears:
    Wg = W if vt else W.T                                              # -> [out, in]
    scale = np.maximum(np.abs(Wg).max(axis=1) / 127.0, 1e-9).astype(np.float32)   # per-output [out]
    Qi = gptq(Wg, Hs[n.name], scale)                                  # [out, in] int8
    inits[ws].CopyFrom(numpy_helper.from_array(Qi if vt else Qi.T, ws))
    pfx = n.name.strip("/").replace("/", "_"); qax = 0 if vt else 1
    new_inits += [helper.make_tensor(f"{pfx}_wsc", TensorProto.FLOAT, [scale.size], scale.tolist()),
                  helper.make_tensor(f"{pfx}_wzp", TensorProto.INT8, [scale.size], [0] * scale.size)]
    dq = f"{pfx}_wdq"
    new_nodes.append(helper.make_node("DequantizeLinear", [ws, f"{pfx}_wsc", f"{pfx}_wzp"], [dq], name=f"{pfx}_DQw", axis=qax))
    if vt:
        for i, inp in enumerate(tn.input):
            if inp == ws: tn.input[i] = dq
    else: n.input[1] = dq
g.initializer.extend(new_inits); g.node.extend(new_nodes)
avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {""}
order, rem = [], list(g.node)
while rem:
    nx, prog = [], False
    for nd in rem:
        if all(i in avail for i in nd.input): order.append(nd); [avail.add(o) for o in nd.output]; prog = True
        else: nx.append(nd)
    rem = nx
    if not prog: raise RuntimeError("topo stuck")
del g.node[:]; g.node.extend(order)
for f in (OUT, OUT + ".data"):
    if os.path.exists(f): os.remove(f)
onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(OUT) + ".data", size_threshold=1024)
print(f"GPTQ int8 on {len(linears)} linears -> {OUT}", flush=True)
