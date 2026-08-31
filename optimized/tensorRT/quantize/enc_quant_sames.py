#!/usr/bin/env python3
"""SAME-S ENCODER quant tiers. ORT-capture FFN/attn-proj activations from calibration AUDIO (Hessian +
p99.9 clip scale), then emit 3 self-contained onnx: w8 (int8 weight-only), fp8 (FFN fp8-compute +
attn weight-only fp8), fast (all-linears fp8-compute). Per-channel weight scales (weakly-typed OK)."""
import os, re
import numpy as np, onnx, torch
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
torch.set_grad_enabled(False)
WD = "/path/to/sames_fp8"; E4M3 = 448.0
SRC = os.environ.get("ENC_ONNX")
if not SRC:
    from huggingface_hub import hf_hub_download
    SRC = hf_hub_download("stabilityai/stable-audio-3-optimized", "onnx/same-s/enc_dynamic_bf16.onnx")
AUD = np.load(f"{WD}/calib_audio_sames.npz")["audio"].astype(np.float32)               # [6,2,N]
def warr(inits, prod, n):
    w = n.input[1]
    if w in inits: return w, numpy_helper.to_array(inits[w]), False, None
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits: return p.input[0], numpy_helper.to_array(inits[p.input[0]]), True, p
    return None, None, None, None
# ---- capture (ORT): Hessian H=AtA + p99.9 abs per linear ----
m = onnx.load(SRC, load_external_data=True); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
tin = {}
for n in g.node:
    if n.op_type == "MatMul":
        ws, W, vt, tn = warr(inits, prod, n)
        if ws is not None and W.ndim == 2: tin[n.name] = n.input[0]
n0 = len(g.output); ex = {o.name for o in g.output}
for t in set(tin.values()):
    if t not in ex: g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
for f in (f"{WD}/_enc.onnx", f"{WD}/_enc.onnx.data"):
    if os.path.exists(f): os.remove(f)
onnx.save(m, f"{WD}/_enc.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="_enc.onnx.data", size_threshold=1024)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(f"{WD}/_enc.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
iname = sess.get_inputs()[0].name; ks = list(tin.keys()); ts = [tin[k] for k in ks]
Hs = {k: None for k in ks}; P99 = {k: [] for k in ks}
for ci in range(AUD.shape[0]):
    acts = sess.run(ts, {iname: AUD[ci:ci + 1]})
    for k, a in zip(ks, acts):
        A = a.reshape(-1, a.shape[-1]).astype(np.float64); Hs[k] = A.T @ A if Hs[k] is None else Hs[k] + A.T @ A
        av = np.abs(a).ravel(); P99[k].append(av if av.size < 400000 else np.random.default_rng(ci).choice(av, 400000, replace=False))
    print(f"  captured clip {ci}", flush=True)
del sess
ASCALE = {k: max(float(np.percentile(np.concatenate(v), 99.9)) / E4M3, 1e-4) for k, v in P99.items()}   # per-linear clip scale
print(f"  clip scales: min {min(ASCALE.values()):.4f} max {max(ASCALE.values()):.4f}", flush=True)
# ---- GPTQ (int8 or fp8), blocked, per-output-channel ----
def gptq(Wnp, Hnp, sc_np, fmt, bs=128, pd=0.05):
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
            qf = torch.clamp(torch.round(w / sc), -127, 127) if fmt == "int8" else (w / sc).to(torch.float8_e4m3fn).to(torch.float32)
            q = qf * sc; Q1[:, j] = qf; err = (w - q) / dd; Err[:, j] = err; W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        if i2 < cin: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q
# ---- graft one tier (fresh load -> self-contained) ----
def build_tier(mode, out):
    m = onnx.load(SRC, load_external_data=True); g = m.graph
    inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
    fp8 = mode in ("fp8", "fast"); noact = re.compile(r"to_qkv|to_out") if mode == "fp8" else re.compile(r"$^")
    if fp8: g.initializer.append(helper.make_tensor("fp8_zero", TensorProto.FLOAT8E4M3FN, [], [0.0]))
    new_nodes, new_inits, drop = [], [], set()
    for n in [x for x in g.node if x.op_type == "MatMul" and x.name in Hs]:
        ws, W, vt, tn = warr(inits, prod, n)
        if ws is None: continue
        Wg = W if vt else W.T; div = E4M3 if fp8 else 127.0
        scale = np.maximum(np.abs(Wg).max(axis=1) / div, 1e-8).astype(np.float32)
        Qi = gptq(Wg, Hs[n.name], scale, "fp8" if fp8 else "int8"); Qs = Qi if vt else Qi.T.contiguous()
        pfx = n.name.strip("/").replace("/", "_"); qax = 0 if vt else 1; wdq = f"{pfx}_wdq"
        do_act = fp8 and not noact.search(n.name)
        if fp8:
            raw = Qs.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy().tobytes()
            new_inits.append(helper.make_tensor(f"{pfx}_w8", TensorProto.FLOAT8E4M3FN, list(Qs.shape), raw, raw=True))
            new_inits.append(numpy_helper.from_array(scale, f"{pfx}_wsc"))
            new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", "fp8_zero"], [wdq], name=f"{pfx}_DQw", axis=qax))
            if do_act:
                a = ASCALE[n.name]; new_inits.append(helper.make_tensor(f"{pfx}_asc", TensorProto.FLOAT, [], [a]))
                aq, adq = f"{pfx}_aq", f"{pfx}_adq"
                new_nodes.append(helper.make_node("QuantizeLinear", [n.input[0], f"{pfx}_asc", "fp8_zero"], [aq], name=f"{pfx}_Qa"))
                new_nodes.append(helper.make_node("DequantizeLinear", [aq, f"{pfx}_asc", "fp8_zero"], [adq], name=f"{pfx}_DQa"))
                n.input[0] = adq
        else:
            new_inits.append(numpy_helper.from_array(Qs.to(torch.int8).cpu().numpy(), f"{pfx}_w8"))
            new_inits.append(numpy_helper.from_array(scale, f"{pfx}_wsc"))
            new_inits.append(helper.make_tensor(f"{pfx}_wzp", TensorProto.INT8, [scale.size], [0] * scale.size))
            new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", f"{pfx}_wzp"], [wdq], name=f"{pfx}_DQw", axis=qax))
        if vt:
            for i, inp in enumerate(tn.input):
                if inp == ws: tn.input[i] = wdq
        else: n.input[1] = wdq
        drop.add(ws)
    g.initializer.extend(new_inits); g.node.extend(new_nodes)
    keep = [i for i in g.initializer if i.name not in drop]; del g.initializer[:]; g.initializer.extend(keep)
    avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {""}; order, rem = [], list(g.node)
    while rem:
        nx, prog = [], False
        for nd in rem:
            if all(i in avail for i in nd.input): order.append(nd); [avail.add(o) for o in nd.output]; prog = True
            else: nx.append(nd)
        rem = nx
        if not prog: raise RuntimeError("topo stuck")
    del g.node[:]; g.node.extend(order)
    for f in (out, out + ".data"):
        if os.path.exists(f): os.remove(f)
    onnx.save(m, out, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(out) + ".data", size_threshold=1024)
    print(f"  {mode:5s} -> {out}", flush=True)
for mode, out in [("w8", f"{WD}/enc_sames_w8.onnx"), ("fp8", f"{WD}/enc_sames_fp8.onnx"), ("fast", f"{WD}/enc_sames_fast.onnx")]:
    build_tier(mode, out)
print("done", flush=True)
