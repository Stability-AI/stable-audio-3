#!/usr/bin/env python3
"""fp8-STORED + GPTQ decoder tier — self-contained, every linear 8-bit.
ALL linears: per-output-channel fp8 weight, GPTQ-compensated, STORED as fp8 (small onnx).
  - FFN / non-attention: ALSO per-tensor fp8 activation QDQ  -> fp8 GEMM (speed, 1.14x)
  - attention proj (to_qkv/to_out): weight-only fp8, DQ to bf16 -> bf16 compute (quality-safe)
Fresh reload before grafting so the output .data is self-contained. argv: <fp32_onnx> <out_onnx>."""
import sys, os, json, re
import numpy as np, onnx, torch
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
torch.set_grad_enabled(False)
DEC, OUT = sys.argv[1], sys.argv[2]; WD = "/path/to/sames_fp8"
AS = json.load(open(f"{WD}/dec_fp8_act_scales_real.json")); nsc = AS["node_scale"]; gs = AS["global_scale"]
LATS = np.load(f"{WD}/calib_latents.npz")["latents"]
NOACT = re.compile(os.environ.get("NOACT_RE", r"to_qkv|to_out"))   # weight-only fp8 (bf16); set "$^" for all-linears fp8
E4M3 = 448.0
def warr(inits, prod, n):
    w = n.input[1]
    if w in inits: return w, numpy_helper.to_array(inits[w]), False, None
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits: return p.input[0], numpy_helper.to_array(inits[p.input[0]]), True, p
    return None, None, None, None
# ---- capture Hessians for ALL linears (ORT), on a temp copy ----
m = onnx.load(DEC, load_external_data=True); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
tin = {n.name: n.input[0] for n in g.node if n.op_type == "MatMul" and warr(inits, prod, n)[0] is not None and warr(inits, prod, n)[1].ndim == 2}
n0 = len(g.output); ex = {o.name for o in g.output}
for t in set(tin.values()):
    if t not in ex: g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
for _f in (f"{WD}/_fpg.onnx", f"{WD}/_fpg.onnx.data"):
    if os.path.exists(_f): os.remove(_f)
onnx.save(m, f"{WD}/_fpg.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="_fpg.onnx.data", size_threshold=1024)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(f"{WD}/_fpg.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
iname = sess.get_inputs()[0].name; ks = list(tin.keys()); ts = [tin[k] for k in ks]; Hs = {k: None for k in ks}
for li in range(LATS.shape[0]):
    acts = sess.run(ts, {iname: LATS[li:li + 1, :, :768].astype(np.float32)})
    for k, a in zip(ks, acts):
        A = a.reshape(-1, a.shape[-1]).astype(np.float64); Hs[k] = A.T @ A if Hs[k] is None else Hs[k] + A.T @ A
    print(f"  captured {li}", flush=True)
del sess
# ---- GPTQ (torch, blocked, per-output-channel fp8 weight) ----
def fp8_gptq(Wnp, Hnp, sc_np, bs=128, pd=0.05):
    dev = "cuda"; W = torch.tensor(np.asarray(Wnp, np.float32), device=dev); H = torch.tensor(np.asarray(Hnp, np.float32), device=dev)
    sc = torch.tensor(sc_np.astype(np.float32), device=dev); cin = W.shape[1]; idx = torch.arange(cin, device=dev)
    H[~torch.isfinite(H)] = 0; d = torch.diag(H).clone(); dead = d <= 0; d[dead] = 1; H[idx, idx] = d; W[:, dead] = 0
    H = (H + H.T) / 2; damp = pd * d.mean(); Hinv = None
    for k in range(6):
        try: L = torch.linalg.cholesky(H + torch.eye(cin, device=dev) * (damp * 4.0 ** k)); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); break
        except Exception: continue
    if Hinv is None: Hinv = torch.diag(1.0 / torch.sqrt(d + damp))
    Q = torch.zeros_like(W)
    for i1 in range(0, cin, bs):
        i2 = min(i1 + bs, cin); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]; dd = Hi[j, j].clamp_min(1e-12)
            qf = (w / sc).to(torch.float8_e4m3fn).to(torch.float32); q = qf * sc
            Q1[:, j] = qf; err = (w - q) / dd; Err[:, j] = err; W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        if i2 < cin: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q
# ---- fresh reload (all raw_data present -> self-contained output), then graft ----
m = onnx.load(DEC, load_external_data=True); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
g.initializer.append(helper.make_tensor("fp8_zero", TensorProto.FLOAT8E4M3FN, [], [0.0]))
new_nodes, new_inits, drop, nact = [], [], set(), 0
for n in [x for x in g.node if x.op_type == "MatMul" and x.name in Hs]:
    ws, W, vt, tn = warr(inits, prod, n)
    if ws is None: continue
    Wg = W if vt else W.T
    scale = np.maximum(np.abs(Wg).max(axis=1) / E4M3, 1e-8).astype(np.float32)
    Qs = fp8_gptq(Wg, Hs[n.name], scale); Qs = Qs if vt else Qs.T.contiguous()
    raw = Qs.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy().tobytes()
    pfx = n.name.strip("/").replace("/", "_"); qax = 0 if vt else 1
    new_inits.append(helper.make_tensor(f"{pfx}_w8", TensorProto.FLOAT8E4M3FN, list(Qs.shape), raw, raw=True))
    new_inits.append(numpy_helper.from_array(scale, f"{pfx}_wsc"))
    wdq = f"{pfx}_wdq"
    new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", "fp8_zero"], [wdq], name=f"{pfx}_DQw", axis=qax))
    if not NOACT.search(n.name):                            # FFN -> fp8 activation too (fp8 GEMM)
        a = float(max(nsc.get(n.name, gs), 1e-4)); new_inits.append(helper.make_tensor(f"{pfx}_asc", TensorProto.FLOAT, [], [a]))
        aq, adq = f"{pfx}_aq", f"{pfx}_adq"
        new_nodes.append(helper.make_node("QuantizeLinear", [n.input[0], f"{pfx}_asc", "fp8_zero"], [aq], name=f"{pfx}_Qa"))
        new_nodes.append(helper.make_node("DequantizeLinear", [aq, f"{pfx}_asc", "fp8_zero"], [adq], name=f"{pfx}_DQa"))
        n.input[0] = adq; nact += 1
    if vt:
        for i, inp in enumerate(tn.input):
            if inp == ws: tn.input[i] = wdq
    else: n.input[1] = wdq
    drop.add(ws)
g.initializer.extend(new_inits); g.node.extend(new_nodes)
_keep = [i for i in g.initializer if i.name not in drop]; del g.initializer[:]; g.initializer.extend(_keep)
avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {""}; order, rem = [], list(g.node)
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
print(f"fp8-stored: {len(drop)} linears 8-bit ({nact} with fp8 activation/GEMM, {len(drop)-nact} weight-only) -> {OUT}", flush=True)
