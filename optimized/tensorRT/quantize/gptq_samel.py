#!/usr/bin/env python3
"""GPTQ int8 for the SAME-L decoder FFN linears. ORT can't run the SWA plugin, so capture the
per-linear input activations from the EAGER SAME-L autoencoder (hooks), build Hessians, GPTQ, and
graft int8 (fp16 scale) onto the SAME-L onnx (attention proj kept fp32). Writes samel_w8_gptq.onnx."""
import sys, os, re
from pathlib import Path
import numpy as np, torch, onnx
from onnx import TensorProto, helper, numpy_helper
sys.path.insert(0, "/path/to/sa3s/stable-audio-3/optimized/tensorRT/scripts")
sys.path.insert(0, "/path/to/fp8_calib/build")
from stable_audio_3.factory import create_autoencoder_from_config
from stable_audio_3.loading_utils import copy_state_dict
import json
torch.set_grad_enabled(False)
ML = "/path/to/sa3s/models/SAME-L"; AB = "/path/to/sames_fp8/decoder_ab"
cfg = json.load(open(f"{ML}/SAME-L.json"))
ae = create_autoencoder_from_config(cfg["model"], cfg["sample_rate"])
ck = torch.load(f"{ML}/SAME-L.ckpt", map_location="cpu", weights_only=False)
copy_state_dict(ae, ck.get("state_dict", ck) if isinstance(ck, dict) else ck)
ae = ae.to("cuda").eval()
# hook the decoder FFN linears (exclude attention proj). Normalize eager+onnx names to (block, role).
def ekey(nm):
    m = re.search(r"transformers\.(\d+)\.ff\.ff\.0\.proj$", nm)
    if m: return f"b{m.group(1)}.proj"
    m = re.search(r"transformers\.(\d+)\.ff\.ff\.2$", nm)
    if m: return f"b{m.group(1)}.out"
    if re.search(r"latent_proj$", nm): return "latent_proj"
    return None
def okey(nm):
    m = re.search(r"blocks\.(\d+)/ff/proj", nm)
    if m: return f"b{m.group(1)}.proj"
    m = re.search(r"blocks\.(\d+)/ff/out", nm)
    if m: return f"b{m.group(1)}.out"
    if re.search(r"latent_proj", nm): return "latent_proj"
    return None
mods = {}
for nm, mod in ae.named_modules():
    if hasattr(mod, "weight") and getattr(mod.weight, "ndim", 0) == 2:
        k = ekey(nm)
        if k: mods[k] = mod
print(f"hooked {len(mods)} eager FFN linears: {sorted(mods)[:3]} ...", flush=True)
H = {k: None for k in mods}
def mk_hook(k):
    def hook(m, inp):
        A = inp[0].reshape(-1, inp[0].shape[-1]).double()
        h = (A.T @ A).cpu().numpy()
        H[k] = h if H[k] is None else H[k] + h
    return hook
handles = [m.register_forward_pre_hook(mk_hook(k)) for k, m in mods.items()]
# calibration latents: medium 2-min + 380s + the 10 real-song SAME-L latents
lats = [np.load("/path/to/fp8_listening/latents_2min.npz")["bf16"]]   # 1292 samples ...
lats += [np.load(f) for f in sorted(Path(AB).glob("samel_lat_*.npy"))]      # + 10×376 = ample for the Hessian
for i, L in enumerate(lats):
    ae.decode(torch.tensor(L, device="cuda", dtype=torch.float32)); print(f"  captured latent {i} ({L.shape[-1]})", flush=True)
for h in handles: h.remove()
def gptq(Wnp, Hnp, scale_np, pd=0.05, bs=128):
    dev = "cuda"
    W = torch.tensor(np.asarray(Wnp, np.float32), device=dev)   # [out, in]
    H = torch.tensor(np.asarray(Hnp, np.float32), device=dev)
    sc = torch.tensor(scale_np.astype(np.float32), device=dev)  # [out]
    cin = W.shape[1]; idx = torch.arange(cin, device=dev)
    H[~torch.isfinite(H)] = 0.0
    d = torch.diag(H).clone(); dead = d <= 0; d[dead] = 1.0; H[idx, idx] = d; W[:, dead] = 0.0
    H = (H + H.T) / 2; damp = pd * d.mean(); Hinv = None
    for k in range(6):                                          # damped Cholesky of H⁻¹ (upper)
        try:
            L = torch.linalg.cholesky(H + torch.eye(cin, device=dev) * (damp * 4.0 ** k))
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); break
        except Exception:
            continue
    if Hinv is None: Hinv = torch.diag(1.0 / torch.sqrt(d + damp))
    Q = torch.zeros_like(W)
    for i1 in range(0, cin, bs):                                # BLOCKED: small updates in-block, 1 matmul for the rest
        i2 = min(i1 + bs, cin); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]; dd = Hi[j, j].clamp_min(1e-12)
            qi = torch.clamp(torch.round(w / sc), -127, 127); err = (w - qi * sc) / dd
            Q1[:, j] = qi; Err[:, j] = err; W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        if i2 < cin: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return torch.clamp(torch.round(Q), -127, 127).to(torch.int8).cpu().numpy()
# graft onto the onnx
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
new_nodes, new_inits, done = [], [], 0
for n in [x for x in g.node if x.op_type == "MatMul"]:
    key = okey(n.name)
    if key not in mods: continue                                                            # skip attn proj / non-FFN
    ws, W, vt, tn = warr(n)
    if ws is None: continue
    Wg = mods[key].weight.detach().double().cpu().numpy()                                    # eager [out,in]
    scale = np.maximum(np.abs(Wg).max(axis=1) / 127.0, 1e-4).astype(np.float32)
    Qi = gptq(Wg, H[key], scale)                                                             # [out,in]
    qax = 0 if vt else 1
    inits[ws].CopyFrom(numpy_helper.from_array(Qi if vt else Qi.T, ws))
    pfx = n.name.strip("/").replace("/", "_")
    new_inits += [numpy_helper.from_array(scale.astype(np.float16), f"{pfx}_wsc"),
                  helper.make_tensor(f"{pfx}_wzp", TensorProto.INT8, [scale.size], [0] * scale.size)]
    dq = f"{pfx}_wdq"
    new_nodes.append(helper.make_node("DequantizeLinear", [ws, f"{pfx}_wsc", f"{pfx}_wzp"], [dq], name=f"{pfx}_DQw", axis=qax))
    if vt:
        for i, inp in enumerate(tn.input):
            if inp == ws: tn.input[i] = dq
    else: n.input[1] = dq
    done += 1
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
OUT = "/path/to/sames_fp8/samel_w8_gptq.onnx"
for f in (OUT, OUT + ".data"):
    if os.path.exists(f): os.remove(f)
onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(OUT) + ".data", size_threshold=1024)
print(f"GPTQ int8 grafted on {done} SAME-L FFN linears -> {OUT}", flush=True)
