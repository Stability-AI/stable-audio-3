"""Export the medium DiT (stable_audio_3.models.dit.DiffusionTransformer) to a fixed-length tflite
rung, from the repo's OWN model + the ARC checkpoint — the baked-I/O 7-input form the tflite runtime
(rung_dit.py) feeds:
    x[1,256,R] t[1] t5_hidden[1,256,768] t5_mask[1,256] seconds[1] local_add_cond[1,257,R] attn_mask[1,1,1,64+R]
The conditioner (T5 padding + seconds NumberEmbedder) is baked in; attn_mask masks the (R-L) pad KEYS so a
render of real length L on rung R is EXACT. Dense masked SDPA (flash_attn absent) → exports cleanly.

Usage:  export_dit.py <R>  [--verify]      (--verify: torch-only mask check, no tflite export)
"""
import os, sys, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch, torch.nn as nn, torch.nn.functional as F

MEM = 64
GCE_OUT = 9216       # global_cond_embedder output = 6 * embed_dim (6*1536 for medium); the adaLN global gc
_ADD_MASK = [None]   # the rung's ADDITIVE self-attn key mask [B,1,1,S] (0 valid / -1e9 pad); set per forward

def patch_attention_for_export(debug=False):
    """Replace Attention.apply_attn with a plain DENSE SDPA that applies an ADDITIVE -inf key mask from
    padding_mask (True=valid). The repo's default uses V-zeroing (zeros pad V but keeps pad keys in the
    softmax denominator -> inexact); the true model path (flash-varlen) UNPADS (exact). Additive key
    masking reproduces the exact/unpadded result and exports cleanly to tflite (BATCH_MATMUL+SOFTMAX)."""
    from stable_audio_3.models import transformer as T
    def apply_attn(self, q, k, v, causal=None, padding_mask=None, **kw):
        if self.num_heads != self.kv_heads:
            r = self.num_heads // self.kv_heads
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        # Explicit dense attention (BATCH_MATMUL + SOFTMAX) instead of F.sdpa — ai_edge_torch's sdpa
        # lowering emits a tfl.less (illegal here); this is export-clean and identical math.
        scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        am = _ADD_MASK[0]
        if am is not None and am.shape[-1] == k.shape[2]:   # self-attn (static shapes) -> add the rung mask
            scores = scores + am
        return torch.matmul(torch.softmax(scores, dim=-1), v)
    T.Attention.apply_attn = apply_attn

    # RoPE rotate_half uses `rearrange('... (j d) -> ... j d', j=2)` which makes a 6-D tensor for the
    # differential 5-D q/k -> tflite's strided_slice rejects >5-D. `(j d)` with j=2 is a block split
    # (first half / second half), so slicing the last dim is numerically IDENTICAL and stays <=5-D.
    def rotate_half(x):
        d = x.shape[-1] // 2
        return torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    T.rotate_half = rotate_half

    # Timestep logsnr conditioning uses .clamp() (decomposes to tfl.less). Replace with minimum/maximum.
    from stable_audio_3.models import dit as D
    def _t_to_logsnr_cond(self, t):
        tf = t.float()
        def mm(x, lo, hi):
            return torch.minimum(torch.maximum(x, x.new_full((), float(lo))), x.new_full((), float(hi)))
        tc = mm(tf, 1e-7, 1 - 1e-7)
        logsnr = mm(torch.log((1 - tc) / tc), self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).to(t.dtype)
    D.DiffusionTransformer._t_to_logsnr_cond = _t_to_logsnr_cond

    # ExpoFourierFeatures uses torch.linspace (converter lowers it to a tfl.less range). The freqs are
    # constant, so precompute them as a buffer on the first (eager) call -> no linspace in the trace.
    import math as _math
    from stable_audio_3.models import blocks as B
    def expo_forward(self, t):
        in_dtype = t.dtype; t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if not hasattr(self, "_freqs_cache"):
            half = self.dim // 2
            ramp = torch.linspace(0, 1, half, dtype=torch.float32)
            self.register_buffer("_freqs_cache",
                torch.exp(ramp * (_math.log(self.max_freq) - _math.log(self.min_freq)) + _math.log(self.min_freq)),
                persistent=False)
        args = t * self._freqs_cache * 2 * _math.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(in_dtype)
    B.ExpoFourierFeatures.forward = expo_forward

    # RMSNorm/LayerNorm via F.rms_norm/F.layer_norm emit tfl.pow (x**2 for variance), illegal here.
    # Compute manually with x*x instead of pow.
    def rms_forward(self, x):
        dt = x.dtype; xf = x.float()
        n = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + self.eps)
        return (n * self.gamma.float()).to(dt)
    T.RMSNorm.forward = rms_forward
    def ln_forward(self, x):
        dt = x.dtype; xf = x.float()
        mu = xf.mean(dim=-1, keepdim=True); xc = xf - mu
        n = xc * torch.rsqrt((xc * xc).mean(dim=-1, keepdim=True) + self.eps)
        return (n * self.gamma.float() + self.beta.float()).to(dt)
    T.LayerNorm.forward = ln_forward
_HF_REPO = "stabilityai/stable-audio-3-medium"
_CFG_NAME = "stable-audio-3-medium-ARC.json"
_CKPT_NAME = "stable-audio-3-medium-ARC.safetensors"
def _medium_files():
    """Resolve (config, checkpoint) for the medium ARC portably — no machine-specific path baked in. In
    order: (1) $SA3_CKPT_MEDIUM's directory if set (the build_all.sh convention); (2) a glob of the HF
    cache snapshots under the PORTABLE cache root (HF_HUB_CACHE / $HF_HOME / ~/.cache/huggingface); (3)
    hf_hub_download. The DiT weights (model.model.*) + the seconds conditioner live in the ckpt."""
    import glob
    p = os.environ.get("SA3_CKPT_MEDIUM")
    if p and os.path.exists(p):
        cfg = os.path.join(os.path.dirname(os.path.abspath(p)), _CFG_NAME)
        if os.path.exists(cfg):
            return cfg, p
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as cache
    except Exception:
        cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = os.path.join(cache, "models--" + _HF_REPO.replace("/", "--"), "snapshots")
    def g(name):
        hits = sorted(glob.glob(os.path.join(snaps, "*", name)))
        return hits[0] if hits else None
    cfg, ckpt = g(_CFG_NAME), g(_CKPT_NAME)
    if cfg and ckpt:
        return cfg, ckpt
    from huggingface_hub import hf_hub_download
    return (hf_hub_download(repo_id=_HF_REPO, filename=_CFG_NAME),
            hf_hub_download(repo_id=_HF_REPO, filename=_CKPT_NAME))
CFG, CKPT = _medium_files()

def load_model():
    """Build JUST the DiT + the seconds NumberConditioner from the repo config, loading weights from the
    ARC checkpoint. Skips the T5Gemma encoder (never run for the export — t5_hidden is an input) and the AE."""
    from stable_audio_3.models.diffusion import DiTWrapper
    from stable_audio_3.models.conditioners import NumberConditioner
    from safetensors import safe_open
    mc = json.load(open(CFG))["model"]
    dcfg = mc["diffusion"]
    dit = DiTWrapper(dcfg.get("diffusion_objective", "v"), **dcfg["config"]).model.eval()
    cond_dim = mc["conditioning"]["cond_dim"]
    secc = next(c for c in mc["conditioning"]["configs"] if c["type"] == "number")["config"]
    seccond = NumberConditioner(cond_dim, **secc).eval()
    dit_sd, sec_sd, pad = {}, {}, None
    with safe_open(CKPT, "pt") as f:
        for k in f.keys():
            if k.startswith("model.model."):
                dit_sd[k[len("model.model."):]] = f.get_tensor(k)
            elif k.startswith("conditioner.conditioners.seconds_total."):
                sec_sd[k[len("conditioner.conditioners.seconds_total."):]] = f.get_tensor(k)
            elif k == "conditioner.conditioners.prompt.padding_embedding":
                pad = f.get_tensor(k)
    md = dit.load_state_dict(dit_sd, strict=False)
    ms = seccond.load_state_dict(sec_sd, strict=False)
    print(f"DiT load: {len(dit_sd)} tensors, missing={len(md.missing_keys)} unexpected={len(md.unexpected_keys)}", flush=True)
    print(f"sec load: {len(sec_sd)} tensors, missing={len(ms.missing_keys)} unexpected={len(ms.unexpected_keys)}", flush=True)
    return dit, seccond, pad.float()

class GlobalCondBaked(nn.Module):
    """Computes the DiT's adaLN global vector gc[1,9216] = global_cond_embedder(to_global_embed(seconds_embed)
    + timestep_embed(t)), exported as a SEPARATE small fp32 tflite. This keeps the 24x-inlined (M=1 [9216,1536])
    global_cond_embedder FC OUT of the int8 rung graph — inside it, under the XNNPACK weight cache, that shared
    FC corrupts multi-rung int8 (memory sa3-rung-weightcache-int8-bug). The runtime runs this per diffusion step
    and feeds gc into the rung. Mirrors dit._forward lines 200-245 + transformer.py:1211 exactly. Captures the
    submodules by reference at construction so a later apply_gce_outside(dit) mutation can't disturb it."""
    def __init__(self, dit, seccond):
        super().__init__()
        self.seccond = seccond
        self.to_global_embed = dit.to_global_embed            # 768 -> 1536 -> 1536
        self.gce = dit.transformer.global_cond_embedder       # 1536 -> 1536 -> 9216 (the trigger FC)
        self.to_timestep_embed = dit.to_timestep_embed
        self.timestep_features = dit.timestep_features
        self._logsnr = dit._t_to_logsnr_cond                  # class-patched method (no clamp)
        self.use_logsnr = dit.timestep_features_logsnr

    def _seconds_embed(self, seconds):
        c = self.seccond
        s = (seconds - c.min_val) / (c.max_val - c.min_val)
        emb = c.embedder(s)
        return emb if emb.dim() == 3 else emb.unsqueeze(1)    # [B,1,768]

    def forward(self, seconds, t):
        se = self._seconds_embed(seconds)[:, 0, :]            # [B,768]  (== DiTBaked gcond)
        ge = self.to_global_embed(se)                         # [B,1536]
        t_cond = self._logsnr(t) if self.use_logsnr else t
        t_cond = t_cond.to(ge.dtype)
        te = self.to_timestep_embed(self.timestep_features(t_cond[:, None]))   # [B,1536]
        return self.gce(ge + te)                              # [B,9216]  (timestep_cond_type=="global")


def apply_gce_outside(dit):
    """Move the global-cond preproc OUT of the rung graph (it lives in GlobalCondBaked / gcond.tflite): the
    rung then receives the pre-computed gc[1,9216] as `global_embed` and feeds it straight to the blocks'
    adaLN. Keeps the 24x-inlined global_cond_embedder FC out of the int8 rung (the XNNPACK-cache corruption
    trigger). _forward: to_global_embed=Identity -> global_embed stays = gc; timestep_cond_type!='global' ->
    no timestep add; global_cond_embedder=None -> blocks get gc directly (see memory file)."""
    dit.to_global_embed = nn.Identity()
    dit.timestep_cond_type = "DISABLED"
    dit.transformer.global_cond_embedder = None


class DiTBaked(nn.Module):
    def __init__(self, dit, seccond, pad):
        super().__init__()
        apply_gce_outside(dit)               # rung takes pre-computed gc[1,9216]; gce lives in gcond.tflite
        self.dit = dit
        self.seccond = seccond
        self.register_buffer("pad", pad)

    def _seconds_embed(self, seconds):
        c = self.seccond
        # no clamp: `seconds` is always in [min,max] at inference; a runtime min/max lowers to tfl.less
        # (illegal in this converter config). The runtime driver feeds a valid seconds value.
        s = (seconds - c.min_val) / (c.max_val - c.min_val)
        emb = c.embedder(s)                            # NumberEmbedder -> [B, 768] (or [B,1,768])
        return emb if emb.dim() == 3 else emb.unsqueeze(1)   # [B,1,768]

    def forward(self, x, gc, t5_hidden, t5_mask, seconds, local_add_cond, attn_mask):
        _ADD_MASK[0] = attn_mask                                       # [B,1,1,MEM+R] additive self-attn key mask
        m = t5_mask[..., None]                                          # [B,256,1] float 0/1 (no bool cast -> no tfl.less)
        padded = t5_hidden * m + self.pad.view(1, 1, -1) * (1 - m)      # [B,256,768]  (== where(m,t5,pad))
        se = self._seconds_embed(seconds)                              # [B,1,768]  (still needed for cross)
        cross = torch.cat([padded, se], dim=1)                         # [B,257,768]
        # gc[B,9216] is the pre-computed adaLN global (from gcond.tflite); dit mutated (apply_gce_outside) so
        # _forward passes it straight to the blocks. t is dropped from the rung (it only fed timestep->global,
        # now in gcond.tflite); a constant dummy keeps _forward's positional signature, its timestep_embed is
        # unused (timestep_cond_type=="DISABLED") and constant-folds away.
        dummy_t = x.new_full((x.shape[0],), 0.5)                       # [B]  (unused; prunes)
        return self.dit._forward(
            x, dummy_t, cross_attn_cond=cross, global_embed=gc,
            local_add_cond=local_add_cond, padding_mask=None)


def _inputs(R, L=None, g=None):
    L = L or R
    g = torch.Generator().manual_seed(0) if g is None else g
    x = torch.randn(1, 256, R, generator=g)
    gc = torch.randn(1, GCE_OUT, generator=g)                        # pre-computed adaLN global (gcond.tflite)
    t5 = torch.randn(1, 256, 768, generator=g)
    t5m = torch.ones(1, 256)
    lac = torch.randn(1, 257, R, generator=g)
    am = torch.zeros(1, 1, 1, MEM + R)
    am[..., MEM + L:] = -1e9                                          # mask the (R-L) pad keys
    return (x, gc, t5, t5m, torch.tensor([120.0]), lac, am)


def _gcond_inputs(g=None):
    return (torch.tensor([120.0]), torch.tensor([0.5]))              # (seconds, t) -> gc[1,9216]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("R", type=int, nargs="?", default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--gcond", action="store_true", help="export gcond.tflite: (seconds,t) -> gc[1,9216]")
    a = ap.parse_args()
    patch_attention_for_export(debug=True)
    print("loading DiT + seconds from ARC checkpoint…", flush=True)
    dit, seccond, pad = load_model()

    if a.gcond:                                   # the global-cond preamble (BEFORE any DiTBaked mutation)
        gcb = GlobalCondBaked(dit, seccond).eval()
        gsample = _gcond_inputs()
        with torch.no_grad():
            gfull = gcb(*gsample)
        print(f"gcond forward out={tuple(gfull.shape)} OK", flush=True)
        import ai_edge_torch
        work = os.environ.get("SA3_BUILD_WORK", os.path.dirname(__file__))
        os.makedirs(work, exist_ok=True)
        out = os.path.join(work, "gcond.tflite")
        print("converting gcond -> tflite …", flush=True)
        ai_edge_torch.convert(gcb.eval(), gsample).export(out)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=out); it.allocate_tensors()
        det = sorted(it.get_input_details(), key=lambda d: d["name"])   # args_0=seconds, args_1=t
        for d_, arr in zip(det, gsample):
            it.set_tensor(d_["index"], arr.numpy().astype(d_["dtype"]))
        it.invoke()
        tfl = it.get_tensor(it.get_output_details()[0]["index"])
        err = float((torch.from_numpy(tfl) - gfull).abs().max())
        print(f"EXPORTED gcond -> {out} ({os.path.getsize(out)/1e6:.0f}MB)  tflite-vs-torch max|d|={err:.2e} "
              f"{'OK' if err < 1e-2 else 'CHECK'}", flush=True)
        print("DONE", flush=True); sys.exit(0)

    baked = DiTBaked(dit, seccond, pad).eval()    # NOTE: mutates dit (apply_gce_outside) — do gcond FIRST
    with torch.no_grad():
        full = baked(*_inputs(a.R))
        print(f"R={a.R} baked forward out={tuple(full.shape)} OK", flush=True)
        if a.verify:
            # mask check: rung-R output on real length L, trimmed to L, vs a plain length-L forward
            L = max(8, a.R // 2)
            gi = torch.Generator().manual_seed(1)
            xr = _inputs(a.R, L=L, g=torch.Generator().manual_seed(1))
            # build a matching length-L input from the same seed-1 draw (draw order == _inputs: x, gc, t5, lac)
            gL = torch.Generator().manual_seed(1)
            xL = torch.randn(1, 256, a.R, generator=gL)[:, :, :L]
            gcL = torch.randn(1, GCE_OUT, generator=gL)                   # gc is length-independent (same draw)
            t5L = torch.randn(1, 256, 768, generator=gL)
            lacL = torch.randn(1, 257, a.R, generator=gL)[:, :, :L]
            amL = torch.zeros(1, 1, 1, MEM + L)
            ref = baked(xL, gcL, t5L, torch.ones(1, 256), torch.tensor([120.0]), lacL, amL)
            got = baked(*xr)[:, :, :L]
            d = (ref - got).abs().max().item()
            print(f"MASK CHECK R={a.R} L={L}: max|rungR[:L] - plainL| = {d:.2e}  {'OK' if d < 1e-3 else 'MISMATCH'}", flush=True)
    if a.probe:
        import torch.export as TE
        ep = TE.export(baked.eval(), _inputs(a.R))
        print("=== first 30 graph nodes ===", flush=True)
        for i, n in enumerate(ep.graph.nodes):
            if i >= 30:
                break
            st = n.meta.get("stack_trace", "") or ""
            src = [l.strip() for l in st.splitlines() if "stable_audio_3" in l or "export_dit" in l]
            print(f"  #{i} {n.name}: {n.op}/{n.target}  <- {src[-1] if src else ''}", flush=True)
        print("timestep_features_logsnr =", getattr(dit, "timestep_features_logsnr", "?"), flush=True)
        hits = 0
        for n in ep.graph.nodes:
            tn = str(n.target)
            if any(s in tn.lower() for s in ("lt.", "less", ".gt.", ".ge.", ".le.", "where", "_to_copy", ".ne.", ".eq.", "clamp", "sign", "floor", "ceil", "round", "minimum", "maximum", "amin", "amax", "aten.min", "aten.max")):
                st = n.meta.get("stack_trace", "") or ""
                src = [l.strip() for l in st.splitlines() if "stable_audio_3" in l or "export_dit" in l]
                print(f"  {n.name}: {tn}  <-  {src[-1] if src else '?'}", flush=True)
                hits += 1
        print(f"PROBE_DONE ({hits} suspect ops)", flush=True)
    if not a.verify and not a.probe:
        import ai_edge_torch
        work = os.environ.get("SA3_BUILD_WORK", os.path.dirname(__file__))
        os.makedirs(work, exist_ok=True)
        out = os.path.join(work, f"dit_fp32_{a.R}.tflite")
        sample = _inputs(a.R)
        print(f"converting R={a.R} -> tflite …", flush=True)
        ai_edge_torch.convert(baked.eval(), sample).export(out)
        # correctness: tflite output vs torch (same inputs)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=out); it.allocate_tensors()
        # get_input_details() is NOT in forward/args order — sort by name (args_0..6 == the forward
        # arg order that `sample` is in) so each tensor gets its matching input.
        det = sorted(it.get_input_details(), key=lambda d: d["name"])
        for d_, arr in zip(det, sample):
            it.set_tensor(d_["index"], arr.numpy().astype(d_["dtype"]))
        it.invoke()
        tfl = it.get_tensor(it.get_output_details()[0]["index"])
        err = float((torch.from_numpy(tfl) - full).abs().max())
        print(f"EXPORTED R={a.R} -> {out} ({os.path.getsize(out)/1e6:.0f}MB)  tflite-vs-torch max|d|={err:.2e} {'OK' if err < 1e-2 else 'CHECK'}", flush=True)
    print("DONE", flush=True)
