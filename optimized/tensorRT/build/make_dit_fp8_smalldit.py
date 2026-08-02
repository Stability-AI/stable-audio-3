#!/usr/bin/env python3
"""Produce the fp8 ONNX for the SMALL DiTs (sm-music / sm-sfx) by grafting fp8 E4M3 Q/DQ onto
the LINEAR GEMMs of the model's fp16 ONNX — leaving attention fp16-fused and the fp32
RMSNorm/RoPE islands intact. This is deliberately NOT the medium's fp8 recipe (baked RoPE +
bf16 attention, build_dit_bf16.py + build_dit_fp8.py): the small DiTs never had the bf16
long-angle RoPE problem, so their fp8 is a straight graft on the known-good fp16 graph.

Why a dedicated script (vs ModelOpt/build_dit_fp8.py): running ModelOpt PTQ on these models
flattens the fp32 islands and its island-reapply (tuned for the medium graph) does NOT restore
them correctly here — the resulting engine drops to velocity-cos ~0.69 with clipping. Grafting
onto the fp16 ONNX keeps the islands correct by construction (velocity-cos ~0.99 vs eager).

Two fp16-trunk specifics vs dit_fp8_max/make_fp8_onnx.py (which targets the fp32-trunk medium):
  * Q/DQ scales are FLOAT16 (the trunk is fp16 → DequantizeLinear must output fp16, else TRT
    sees Half-vs-Float at the residual Adds).
  * scales floored at 1e-4 (fp16 underflows anything <~6e-5 to 0, and TRT rejects a non-positive
    scale; only bites zero-input layers e.g. to_local_embed when local_add_cond=0 — harmless).

Activation scales are calibrated from the eager model on the model's own-domain few-shot prompts
(Music for sm-music, SFX for sm-sfx) plus one full-length render, with a margin so nothing clips.
Scale VALUES affect accuracy only, not whether fp8 fires or the latency.

    python make_dit_fp8_smalldit.py \
        --model-config <ckpt>/model_config.json --checkpoint <ckpt>/model.safetensors \
        --fp16-onnx onnx/sa3-sm-music/dit_fp16.onnx \
        --domain Music --out onnx/sa3-sm-music/dit_fp8.onnx

Then compile with build_from_onnx.py sa3-sm-music-fp8 (STRONGLY_TYPED; the QDQ carry fp8).
"""
import argparse, os, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

E4M3_MAX = 448.0
SCALE_DT = TensorProto.FLOAT16
SCALE_FLOOR = 1e-4


def calibrate_act_scales(model_config, checkpoint, fp16_onnx, domain, margin, device):
    """Per-linear activation max|x| from the eager model → {onnx_node_name: scale}. Maps ONNX
    linear nodes to torch modules by weight name (ONNX 'dit.<path>.weight' vs torch 'model.<path>'
    → match on the suffix after the first dotted component)."""
    import torch
    import torch.nn.functional as F
    import make_calib as MK  # repo sibling; wraps load_diffusion_cond + StableAudioModel
    from stable_audio_3.interface.reprompt import SYSTEM_PROMPTS, _extract_examples
    torch.set_grad_enabled(False)

    sa3 = MK._load_model(Path(model_config), Path(checkpoint), device)
    dit = sa3.dit
    suffix = lambda nm: nm.split(".", 1)[1] if "." in nm else nm
    mods = {}

    class Q:
        def __init__(s, l): s.l = l; s.amax = 1e-9
        def __call__(s, x): s.amax = max(s.amax, float(x.abs().amax())); return F.linear(x, s.l.weight, s.l.bias)
    for name, mod in dit.named_modules():
        if isinstance(mod, torch.nn.Linear) and min(mod.in_features, mod.out_features) >= 128:
            mod._q = Q(mod); mod.forward = (lambda m: (lambda x: m._q(x)))(mod); mods[suffix(name)] = mod._q

    prompts = _extract_examples(SYSTEM_PROMPTS[domain])[:14]
    render = {"Music": "Genre: House, Subgenre: Deep House, BPM: 122 BPM, Tempo: Medium, "
                       "VocalType: Instrumental, TrackType: Music, Grade: Neutral",
              "SFX": "Heavy rain on a tin roof with distant thunder, steady continuous downpour"}.get(domain, prompts[0])
    for i, p in enumerate(prompts):
        sa3.generate(prompt=p, duration=MK.DEFAULT_DURATION_S, steps=8, cfg_scale=1.0,
                     sampler_type="pingpong", seed=MK.DEFAULT_SEED + i, duration_padding_sec=0.0, return_latents=True)
    sa3.generate(prompt=render, duration=1292 * 4096 / 44100.0, steps=8, cfg_scale=1.0,
                 sampler_type="pingpong", seed=6000, duration_padding_sec=0.0, return_latents=True)
    print(f"  calibrated {len(mods)} linears on {len(prompts)} {domain} prompts + one full render", flush=True)

    m = onnx.load(fp16_onnx, load_external_data=False); g = m.graph
    inits = {i.name for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
    def wsrc(n):
        w = n.input[1]
        if w in inits: return w
        p = prod.get(w)
        return p.input[0] if (p is not None and p.op_type == "Transpose" and p.input and p.input[0] in inits) else None
    node_scale = {}
    for n in g.node:
        if n.op_type != "MatMul": continue
        w = wsrc(n)
        if w is None: continue
        q = mods.get(suffix(w[:-len(".weight")] if w.endswith(".weight") else w))
        if q is not None:
            node_scale[n.name] = max(q.amax * margin / E4M3_MAX, SCALE_FLOOR)
    gscale = max(max(q.amax for q in mods.values()) * margin / E4M3_MAX, SCALE_FLOOR)
    return node_scale, gscale


def topo_sort(g):
    avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {""}
    remaining, result = list(g.node), []
    while remaining:
        nxt, prog = [], False
        for n in remaining:
            if all(i in avail for i in n.input):
                result.append(n); [avail.add(o) for o in n.output]; prog = True
            else:
                nxt.append(n)
        remaining = nxt
        if not prog: raise RuntimeError(f"topo stuck: {len(remaining)}")
    del g.node[:]; g.node.extend(result)


def graft_fp8(fp16_onnx, node_scale, gscale, out):
    model = onnx.load(fp16_onnx, load_external_data=True); g = model.graph
    have = False
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            have = True
            if op.version < 19: op.version = 19
    if not have: model.opset_import.append(helper.make_opsetid("", 19))
    if model.ir_version < 9: model.ir_version = 9
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    g.initializer.append(helper.make_tensor("fp8_zero", TensorProto.FLOAT8E4M3FN, [], [0.0]))
    new_nodes, new_inits, made, skipped = [], [], 0, 0
    for node in [n for n in g.node if n.op_type == "MatMul"]:
        Wname = node.input[1]; via_t = tnode = w_src = Warr = None; via_t = False
        if Wname in inits:
            w_src = Wname; Warr = numpy_helper.to_array(inits[Wname])
        else:
            p = prod.get(Wname)
            if p is not None and p.op_type == "Transpose" and p.input and p.input[0] in inits:
                tnode = p; w_src = p.input[0]; via_t = True; Warr = numpy_helper.to_array(inits[w_src])
        if Warr is None or Warr.ndim != 2:
            skipped += 1; continue   # attention BMM (no weight initializer)
        pfx = node.name.strip("/").replace("/", "_")
        w_scale = float(max(np.abs(Warr.astype(np.float32)).max() / E4M3_MAX, SCALE_FLOOR))
        a_scale = float(max(node_scale.get(node.name, gscale), SCALE_FLOOR))
        new_inits += [helper.make_tensor(f"{pfx}_wscale", SCALE_DT, [], [w_scale]),
                      helper.make_tensor(f"{pfx}_ascale", SCALE_DT, [], [a_scale])]
        aq, adq = f"{pfx}_aq", f"{pfx}_adq"
        new_nodes += [helper.make_node("QuantizeLinear", [node.input[0], f"{pfx}_ascale", "fp8_zero"], [aq], name=f"{pfx}_Qa"),
                      helper.make_node("DequantizeLinear", [aq, f"{pfx}_ascale", "fp8_zero"], [adq], name=f"{pfx}_DQa")]
        node.input[0] = adq
        wq, wdq = f"{pfx}_wq", f"{pfx}_wdq"
        new_nodes += [helper.make_node("QuantizeLinear", [w_src, f"{pfx}_wscale", "fp8_zero"], [wq], name=f"{pfx}_Qw"),
                      helper.make_node("DequantizeLinear", [wq, f"{pfx}_wscale", "fp8_zero"], [wdq], name=f"{pfx}_DQw")]
        if via_t:
            for i, inp in enumerate(tnode.input):
                if inp == w_src: tnode.input[i] = wdq
        else:
            node.input[1] = wdq
        made += 1
    g.initializer.extend(new_inits); g.node.extend(new_nodes); topo_sort(g)
    if os.path.exists(out): os.remove(out)
    if os.path.exists(out + ".data"): os.remove(out + ".data")
    onnx.save(model, out, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(out) + ".data", size_threshold=1024)
    c = Counter(n.op_type for n in g.node)
    print(f"  fp8 Q/DQ on {made} linear MatMuls ({skipped} attention BMMs skipped); "
          f"Q={c.get('QuantizeLinear',0)} DQ={c.get('DequantizeLinear',0)} Softmax={c.get('Softmax',0)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fp16-onnx", required=True, help="the model's canonical dit_fp16.onnx")
    ap.add_argument("--out", required=True, help="output dit_fp8.onnx (a .data sidecar is written alongside)")
    ap.add_argument("--domain", default="Music", choices=["Music", "SFX", "Instrument", "One-shot"],
                    help="reprompt few-shot domain for activation calibration (Music for sm-music, SFX for sm-sfx)")
    ap.add_argument("--margin", type=float, default=1.35, help="activation-scale headroom so nothing clips")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[make_dit_fp8_smalldit] calibrating ({a.domain}) ...", flush=True)
    node_scale, gscale = calibrate_act_scales(a.model_config, a.checkpoint, a.fp16_onnx, a.domain, a.margin, a.device)
    print(f"  {len(node_scale)} node scales, global={gscale:.5f}; grafting fp8 ...", flush=True)
    graft_fp8(a.fp16_onnx, node_scale, gscale, a.out)
    print(f"DONE -> {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
