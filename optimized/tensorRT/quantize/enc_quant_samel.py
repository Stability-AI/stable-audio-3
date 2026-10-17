#!/usr/bin/env python3
"""SAME-L ENCODER quant tiers (w8 + fp8). Eager-capture the encoder FFN activations (Hessian + p99.9
clip scale) via hooks (SWA plugin blocks ORT), map eager<->onnx FFN linears by order+weight-value,
graft int8 weight-only (w8) and fp8 FFN GEMM (fp8, per-tensor weight, per-linear clip scale). Attention
proj stay fp32 islands. Self-contained."""
import sys, os, re, json, glob
import numpy as np, torch, onnx
from onnx import TensorProto, helper, numpy_helper
sys.path.insert(0, "/path/to/sa3s/stable-audio-3/optimized/tensorRT/scripts"); sys.path.insert(0, "/path/to/fp8_calib/build")
from stable_audio_3.factory import create_autoencoder_from_config
from stable_audio_3.loading_utils import copy_state_dict
torch.set_grad_enabled(False)
WD = "/path/to/sames_fp8"; E4M3 = 448.0
cfg = json.load(open("/path/to/sa3s/models/SAME-L/SAME-L.json"))
ae = create_autoencoder_from_config(cfg["model"], cfg["sample_rate"])
ck = torch.load("/path/to/sa3s/models/SAME-L/SAME-L.ckpt", map_location="cpu", weights_only=False)
copy_state_dict(ae, ck.get("state_dict", ck) if isinstance(ck, dict) else ck); ae = ae.cuda().eval()
# eager encoder FFN linears, in module order
eproj = [m for nm, m in ae.named_modules() if re.search(r"encoder\..*\.ff\.ff\.0\.proj$", nm)]
eout  = [m for nm, m in ae.named_modules() if re.search(r"encoder\..*\.ff\.ff\.2$", nm)]
print(f"eager encoder FFN: {len(eproj)} proj + {len(eout)} out", flush=True)
HP = {}; PP = {}                                                    # keyed by (role, idx)
def mk(role, i):
    def hook(m, inp):
        A = inp[0].reshape(-1, inp[0].shape[-1]).double(); h = (A.T @ A).cpu().numpy()
        HP[(role, i)] = h if (role, i) not in HP else HP[(role, i)] + h
        p = float(np.percentile(np.abs(inp[0].detach().float().cpu().numpy()), 99.9)); PP.setdefault((role, i), []).append(p)
    return hook
H = [m.register_forward_pre_hook(mk("proj", i)) for i, m in enumerate(eproj)] + [m.register_forward_pre_hook(mk("out", i)) for i, m in enumerate(eout)]
AUD = np.load(f"{WD}/calib_audio_samel.npz")["audio"].astype(np.float32)
for ci in range(AUD.shape[0]):
    ae.encode(torch.tensor(AUD[ci:ci + 1], device="cuda")); print(f"  captured clip {ci}", flush=True)
for h in H: h.remove()
ASC = {k: max(float(np.percentile(v, 90)) / E4M3, 1e-4) for k, v in PP.items()}   # per-linear clip scale (robust over clips)
def fp8_gptq(Wnp, Hnp, sc, bs=128, pd=0.05):
    dev = "cuda"; W = torch.tensor(np.asarray(Wnp, np.float32), device=dev); Hh = torch.tensor(np.asarray(Hnp, np.float32), device=dev)
    cin = W.shape[1]; idx = torch.arange(cin, device=dev); Hh[~torch.isfinite(Hh)] = 0
    d = torch.diag(Hh).clone(); dead = d <= 0; d[dead] = 1; Hh[idx, idx] = d; W[:, dead] = 0; Hh = (Hh + Hh.T) / 2; damp = pd * d.mean(); Hinv = None
    for k in range(6):
        try: L = torch.linalg.cholesky(Hh + torch.eye(cin, device=dev) * (damp * 4.0 ** k)); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); break
        except Exception: continue
    if Hinv is None: Hinv = torch.diag(1.0 / torch.sqrt(d + damp))
    Q = torch.zeros_like(W)
    for i1 in range(0, cin, bs):
        i2 = min(i1 + bs, cin); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]; dd = Hi[j, j].clamp_min(1e-12); qf = (w / sc).to(torch.float8_e4m3fn).to(torch.float32)
            q = qf * sc; Q1[:, j] = qf; err = (w - q) / dd; Err[:, j] = err; W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        if i2 < cin: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q
def warr(inits, prod, n):
    w = n.input[1]
    if w in inits: return w, numpy_helper.to_array(inits[w]), False, None
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits: return p.input[0], numpy_helper.to_array(inits[p.input[0]]), True, p
    return None, None, None, None
SRC = os.environ["ENC_ONNX"]
def onnx_ffn(g, inits, prod):
    pr, ou = {}, {}
    for n in g.node:
        if n.op_type != "MatMul": continue
        mp = re.search(r"blocks\.(\d+)/ff/proj", n.name); mo = re.search(r"blocks\.(\d+)/ff/out", n.name)
        if mp: pr[int(mp.group(1))] = n
        elif mo: ou[int(mo.group(1))] = n
    return [pr[i] for i in sorted(pr)], [ou[i] for i in sorted(ou)]
def build_tier(mode, out):
    m = onnx.load(SRC, load_external_data=True); g = m.graph
    inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
    oproj, oout = onnx_ffn(g, inits, prod)
    assert len(oproj) == len(eproj) and len(oout) == len(eout), f"{len(oproj)}/{len(eproj)} {len(oout)}/{len(eout)}"
    fp8 = mode == "fp8"
    if fp8: g.initializer.append(helper.make_tensor("fp8_zero", TensorProto.FLOAT8E4M3FN, [], [0.0]))
    new_nodes, new_inits, drop = [], [], set()
    for role, elist, olist in [("proj", eproj, oproj), ("out", eout, oout)]:
        for i, (em, n) in enumerate(zip(elist, olist)):
            ws, W, vt, tn = warr(inits, prod, n)
            Wg = W if vt else W.T                                          # [out,in]
            Weag = em.weight.detach().double().cpu().numpy()
            assert np.allclose(Wg, Weag, atol=1e-2), f"map mismatch {role}{i} (L2 {np.abs(Wg-Weag).mean():.4f})"
            pfx = n.name.strip("/").replace("/", "_"); qax = 0 if vt else 1; wdq = f"{pfx}_wdq"
            if fp8:
                scale = np.float32(max(np.abs(Wg).max() / E4M3, 1e-8))     # per-tensor
                Qi = fp8_gptq(Wg, HP[(role, i)], scale); Qs = Qi if vt else Qi.T.contiguous()
                raw = Qs.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy().tobytes()
                new_inits.append(helper.make_tensor(f"{pfx}_w8", TensorProto.FLOAT8E4M3FN, list(Qs.shape), raw, raw=True))
                new_inits.append(numpy_helper.from_array(np.array(scale, np.float16), f"{pfx}_wsc"))
                new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", "fp8_zero"], [wdq], name=f"{pfx}_DQw"))
                a = ASC[(role, i)]; new_inits.append(numpy_helper.from_array(np.array(a, np.float16), f"{pfx}_asc"))
                aq, adq = f"{pfx}_aq", f"{pfx}_adq"
                new_nodes.append(helper.make_node("QuantizeLinear", [n.input[0], f"{pfx}_asc", "fp8_zero"], [aq], name=f"{pfx}_Qa"))
                new_nodes.append(helper.make_node("DequantizeLinear", [aq, f"{pfx}_asc", "fp8_zero"], [adq], name=f"{pfx}_DQa"))
                n.input[0] = adq
            else:
                scale = np.maximum(np.abs(Wg).max(axis=1) / 127.0, 1e-4).astype(np.float32)   # per-channel int8
                sc_b = scale.reshape(-1, 1) if not vt else scale.reshape(-1, 1)
                Wq = np.round((Wg / scale.reshape(-1, 1))).clip(-127, 127).astype(np.int8); Qs = Wq if vt else Wq.T
                new_inits.append(numpy_helper.from_array(Qs, f"{pfx}_w8"))
                new_inits.append(numpy_helper.from_array(scale.astype(np.float16), f"{pfx}_wsc"))
                new_inits.append(helper.make_tensor(f"{pfx}_wzp", TensorProto.INT8, [scale.size], [0] * scale.size))
                new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", f"{pfx}_wzp"], [wdq], name=f"{pfx}_DQw", axis=qax))
            if vt:
                for k, inp in enumerate(tn.input):
                    if inp == ws: tn.input[k] = wdq
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
    print(f"  {mode} -> {out}", flush=True)
build_tier("w8", f"{WD}/enc_samel_w8.onnx")
build_tier("fp8", f"{WD}/enc_samel_fp8.onnx")
print("done", flush=True)
