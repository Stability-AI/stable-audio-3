#!/usr/bin/env python3
"""SAME-L fp8-STORED + GPTQ. Eager Hessian+amax capture (ORT can't run the SWA plugin), fp8-GPTQ the
FFN weights, graft fp8-stored weight + per-tensor fp8 activation QDQ onto the SAME-L onnx. Attention
proj stay fp32 islands (fp8 compute breaks them). Self-contained. Writes samel_fp8_gptq.onnx."""
import sys, os, re, json
from pathlib import Path
import numpy as np, torch, onnx
from onnx import TensorProto, helper, numpy_helper
sys.path.insert(0, "/weka2/cj/clod/sa3s/stable-audio-3/optimized/tensorRT/scripts")
sys.path.insert(0, "/weka2/cj/clod/fp8_calib/build")
from stable_audio_3.factory import create_autoencoder_from_config
from stable_audio_3.loading_utils import copy_state_dict
torch.set_grad_enabled(False)
ML = "/weka2/cj/clod/sa3s/models/SAME-L"; AB = "/weka2/cj/clod/sames_fp8/decoder_ab"; E4M3 = 448.0
cfg = json.load(open(f"{ML}/SAME-L.json"))
ae = create_autoencoder_from_config(cfg["model"], cfg["sample_rate"])
ck = torch.load(f"{ML}/SAME-L.ckpt", map_location="cpu", weights_only=False)
copy_state_dict(ae, ck.get("state_dict", ck) if isinstance(ck, dict) else ck); ae = ae.to("cuda").eval()
def ekey(nm):
    m = re.search(r"transformers\.(\d+)\.ff\.ff\.0\.proj$", nm);  x = f"b{m.group(1)}.proj" if m else None
    if x: return x
    m = re.search(r"transformers\.(\d+)\.ff\.ff\.2$", nm)
    if m: return f"b{m.group(1)}.out"
    return "latent_proj" if re.search(r"latent_proj$", nm) else None
def okey(nm):
    m = re.search(r"blocks\.(\d+)/ff/proj", nm);  x = f"b{m.group(1)}.proj" if m else None
    if x: return x
    m = re.search(r"blocks\.(\d+)/ff/out", nm)
    if m: return f"b{m.group(1)}.out"
    return "latent_proj" if re.search(r"latent_proj", nm) else None
mods = {}
for nm, mod in ae.named_modules():
    if hasattr(mod, "weight") and getattr(mod.weight, "ndim", 0) == 2:
        k = ekey(nm)
        if k: mods[k] = mod
print(f"hooked {len(mods)} FFN linears", flush=True)
H = {k: None for k in mods}; AMAX = {k: 0.0 for k in mods}
def mk_hook(k):
    def hook(m, inp):
        A = inp[0].reshape(-1, inp[0].shape[-1]).double(); h = (A.T @ A).cpu().numpy()
        H[k] = h if H[k] is None else H[k] + h; AMAX[k] = max(AMAX[k], float(inp[0].abs().max()))
    return hook
handles = [m.register_forward_pre_hook(mk_hook(k)) for k, m in mods.items()]
lats = [np.load("/weka2/cj/clod/fp8_listening/latents_2min.npz")["bf16"]]
lats += [np.load(f) for f in sorted(Path(AB).glob("samel_lat_*.npy"))]
for i, L in enumerate(lats):
    ae.decode(torch.tensor(L, device="cuda", dtype=torch.float32)); print(f"  captured {i} ({L.shape[-1]})", flush=True)
for h in handles: h.remove()
def fp8_gptq(Wnp, Hnp, sc_np, bs=128, pd=0.05):
    dev = "cuda"; W = torch.tensor(np.asarray(Wnp, np.float32), device=dev); Hh = torch.tensor(np.asarray(Hnp, np.float32), device=dev)
    sc = torch.tensor(sc_np.astype(np.float32), device=dev); cin = W.shape[1]; idx = torch.arange(cin, device=dev)
    Hh[~torch.isfinite(Hh)] = 0; d = torch.diag(Hh).clone(); dead = d <= 0; d[dead] = 1; Hh[idx, idx] = d; W[:, dead] = 0
    Hh = (Hh + Hh.T) / 2; damp = pd * d.mean(); Hinv = None
    for k in range(6):
        try: L = torch.linalg.cholesky(Hh + torch.eye(cin, device=dev) * (damp * 4.0 ** k)); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); break
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
# ---- graft onto the SAME-L onnx (fresh load -> self-contained) ----
from huggingface_hub import hf_hub_download
DEC = hf_hub_download("stabilityai/stable-audio-3-optimized", "onnx/same-l/dec_dynamic_triton_swa.onnx")
m = onnx.load(DEC, load_external_data=True); g = m.graph
inits = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
def warr(n):
    w = n.input[1]
    if w in inits: return w, numpy_helper.to_array(inits[w]), False, None
    p = prod.get(w)
    if p and p.op_type == "Transpose" and p.input and p.input[0] in inits: return p.input[0], numpy_helper.to_array(inits[p.input[0]]), True, p
    return None, None, None, None
g.initializer.append(helper.make_tensor("fp8_zero", TensorProto.FLOAT8E4M3FN, [], [0.0]))
new_nodes, new_inits, drop, done = [], [], set(), 0
for n in [x for x in g.node if x.op_type == "MatMul"]:
    key = okey(n.name)
    if key not in mods: continue                                                   # skip attn proj (fp32 islands) / non-FFN
    ws, W, vt, tn = warr(n)
    if ws is None: continue
    Wg = mods[key].weight.detach().double().cpu().numpy()                            # eager [out,in]
    scale = np.float32(max(np.abs(Wg).max() / E4M3, 1e-8))                            # per-TENSOR scalar (strongly-typed fp8 GEMM needs per-tensor weight scale)
    Qi = fp8_gptq(Wg, H[key], scale); Qstore = Qi if vt else Qi.T.contiguous()       # onnx orientation
    raw = Qstore.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy().tobytes()
    pfx = n.name.strip("/").replace("/", "_")
    new_inits.append(helper.make_tensor(f"{pfx}_w8", TensorProto.FLOAT8E4M3FN, list(Qstore.shape), raw, raw=True))
    new_inits.append(numpy_helper.from_array(np.array(scale, np.float16), f"{pfx}_wsc"))         # scalar fp16 (fp16 trunk)
    a = 0.05; new_inits.append(numpy_helper.from_array(np.array(a, np.float16), f"{pfx}_asc"))   # repo-calibrated FFN activation CLIPPING scale
    wdq, aq, adq = f"{pfx}_wdq", f"{pfx}_aq", f"{pfx}_adq"
    new_nodes.append(helper.make_node("DequantizeLinear", [f"{pfx}_w8", f"{pfx}_wsc", "fp8_zero"], [wdq], name=f"{pfx}_DQw"))
    new_nodes.append(helper.make_node("QuantizeLinear", [n.input[0], f"{pfx}_asc", "fp8_zero"], [aq], name=f"{pfx}_Qa"))
    new_nodes.append(helper.make_node("DequantizeLinear", [aq, f"{pfx}_asc", "fp8_zero"], [adq], name=f"{pfx}_DQa"))
    n.input[0] = adq
    if vt:
        for i, inp in enumerate(tn.input):
            if inp == ws: tn.input[i] = wdq
    else: n.input[1] = wdq
    drop.add(ws); done += 1
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
OUT = "/weka2/cj/clod/sames_fp8/samel_fp8_gptq.onnx"
for f in (OUT, OUT + ".data"):
    if os.path.exists(f): os.remove(f)
onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(OUT) + ".data", size_threshold=1024)
print(f"SAME-L fp8-stored+GPTQ on {done} FFN linears -> {OUT}", flush=True)
