#!/usr/bin/env python3
"""Bake the strength-independent fold quantities into a DoRA / -xs adapter.

Why: the runtime fold needs ``c = magnitude / ||W0 + s*LR||_row``. Computing that denominator
means loading all 10.5 GB of base weights (4.6-21 s and ~10 GB of VRAM) purely to take a row
norm — the TRT engine already carries its own copy of the weights, so it is a second, redundant,
double-precision copy. Baking the norms into the adapter removes it entirely.

Differences from the TFLite bake_norms.py this is compatible with:

  * SVD IS NEVER COMPUTED. -xs bases are sliced from the frozen svd_bases.pt built by
    underfit/utils/compute_svd.py. bake_norms recomputes them through numpy, which on this
    venv's numpy 1.23.5/OpenBLAS takes 80-90 s per layer WHEN IT WORKS and hangs or raises
    "SVD did not converge" on (4096,2048), (2048,8192), (10240,2048), (16384,2048) — most of
    the DiT. torch does all 182 in ~43 s, and the frozen file needs 0.
  * The self-check handles -xs. bake_norms' KeyErrors on `lora_B` for dora-*-xs adapters,
    which store M_xs instead.
  * Base vs arc is chosen from the adapter's own metadata, overridable, and NOT sha-enforced
    (quantisation differences would break a hash without meaning anything).

Key/metadata layout is byte-compatible with bake_norms.py, so the TFLite runtime reads these:
``<layer>.parametrizations.weight.0.baked_vnorm_row`` + a __metadata__ block.

    python bake_dora.py <adapter.safetensors> [--out PATH] [--force]
    python bake_dora.py <adapter.safetensors> --norm-source arc     # advanced: ARC weights
    python bake_dora.py <adapter.safetensors> --check      # exit 0 if already baked
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import paths
# The published onnx/sa3-m/dit_fp16.onnx is built from ARC, so row norms must be
# taken against ARC: folding against the other variant reads as a 27% fold error.
ENGINE_VARIANT = "arc"

PSUF = ".parametrizations.weight.0."
BAKED = "baked_vnorm_row"


def is_baked(path) -> bool:
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        return any(BAKED in k for k in f.keys())


def baked_path(path) -> Path:
    p = Path(path)
    return p if p.name.endswith(".normbaked.safetensors") else \
        p.with_suffix("").with_suffix(".normbaked.safetensors") if p.suffixes[-2:] == [".normbaked", ".safetensors"] \
        else p.parent / (p.stem + ".normbaked.safetensors")


def pick_variant(path, override=None, model="sa3-medium") -> str:
    """The variant the model's engines were built from, unless explicitly overridden.

    Deliberately does NOT read the adapter's metadata. An adapter that merely *claims* a
    variant would silently redirect the bake, and a wrong pairing is invisible: base and arc
    reconstruct each other's norms at rel 2.2e-3 with nothing raising.

    It comes from the engine registry instead. A baked norm is only meaningful against the
    W0 the engine actually carries -- the fold is c = magnitude / ||W0 + s.LR||_row -- so
    "which weights" is a property of the engine, never a user preference. Hardcoding 'base'
    here is what made the medium branch engine (built from the ARC onnx/sa3-m/dit_fp16.onnx)
    read as a 27% fold error.
    """
    return override or ENGINE_VARIANT


def bake(adapter, variant=None, out=None, force=False, quiet=False, model="sa3-medium"):
    import numpy as np, torch
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import lora_core as lc

    adapter = Path(adapter)
    out = Path(out) if out else baked_path(adapter)
    if out.exists() and not force:
        if not quiet: print(f"  already baked: {out.name}  (--force to redo)")
        return out
    variant = variant or ENGINE_VARIANT
    wpath, bpath = paths.checkpoint(variant), paths.SVD_BASES

    atype, scaling, layers = lc.parse_adapter(str(adapter))
    stem = atype[:-3] if atype.endswith("-xs") else atype
    is_xs = atype.endswith("-xs")
    if not quiet:
        print(f"  {adapter.name}\n    type={atype} scaling={scaling:.4f} layers={len(layers)} "
              f"variant={variant}")
    if stem == "lora":
        if not quiet: print("    plain lora — fold is already trivial, nothing to bake")
        return None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bases = torch.load(bpath, map_location="cpu", weights_only=True, mmap=True) if is_xs else None
    with safe_open(str(adapter), framework="numpy") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = dict(f.metadata() or {})
    wf = safe_open(str(wpath), framework="pt")
    wkeys = set(wf.keys())

    n, worst, nneg = 0, 0.0, 0
    for lid, p in layers.items():
        # `conditioner.` covers the one non-DiT layer real adapters touch
        # (conditioners.seconds_total.embedder.embedding.1), which the checkpoint
        # stores under a different prefix from the transformer blocks.
        wk = next((k for k in (f"model.{lid}.weight", f"{lid}.weight",
                               f"conditioner.{lid}.weight") if k in wkeys), None)
        if wk is None:
            raise KeyError(f"no base weight for {lid} in {wpath.name}")
        W0 = wf.get_tensor(wk).to(dev, torch.float32)
        if is_xs:
            bk = next((k for k in bases if k.endswith(lid.split("model.", 1)[-1] + ".weight")
                       or k.endswith(lid + ".weight")), None)
            if bk is None:
                raise KeyError(f"no frozen SVD basis for {lid} in {bpath.name}")
            r = p["M_xs"].shape[0]
            U = bases[bk]["U"][:, :r].to(dev, torch.float32)
            V = bases[bk]["V"][:, :r].to(dev, torch.float32)
            LR = U @ torch.as_tensor(p["M_xs"], device=dev, dtype=torch.float32) @ V.T
        else:
            LR = (torch.as_tensor(p["lora_B"], device=dev, dtype=torch.float32)
                  @ torch.as_tensor(p["lora_A"], device=dev, dtype=torch.float32))
        vn = torch.linalg.norm(W0 + scaling * LR, dim=1)          # ||W0 + s.LR||_row
        # Self-check on a real invariant: the DoRA weight is  W = mag ⊙ (W0+s.LR)/vn , so its
        # row norms must come back as `magnitude`. -xs aware, unlike bake_norms' own check.
        mag = torch.as_tensor(np.squeeze(p["magnitude"]).astype(np.float32), device=dev)
        c = mag / (vn + 1e-12)
        rows = torch.linalg.norm(c.unsqueeze(1) * (W0 + scaling * LR), dim=1)
        # `magnitude` is a free nn.Parameter initialised to ||W0||_row, so training can drive
        # rows NEGATIVE. The DoRA weight is c.V with c signed, so the recovered row norm is
        # |magnitude|, not magnitude -- comparing against the signed value reports a rel error
        # of exactly 2.0 for every negative row and hides real errors behind it.
        worst = max(worst, float(((rows - mag.abs()) / mag.abs().clamp_min(1e-6)).abs().max()))
        nneg += int((mag < 0).sum())
        tensors[f"{lid}{PSUF}{BAKED}"] = vn.cpu().numpy().astype(np.float32)
        n += 1

    meta.update({
        "base_model": f"{model}-{variant}", "norm_source": variant,
        "norm_base_sha": hashlib.sha256(wpath.name.encode()).hexdigest()[:32],   # recorded, not enforced
        "baked_variant": atype, "baked_norms": str(n),
        "baked_xs_svd": ("external:" + bpath.name) if is_xs else "none",
        "baked_by": "bake_dora.py",
        "baked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    if not quiet:
        print(f"    → {out.name}  (+{n} baked norms, {len(tensors)} tensors, "
              f"{out.stat().st_size/1e6:.1f} MB)")
        print(f"    self-check ||mag⊙(W0+s·LR)/vn||_row vs |magnitude|: max rel {worst:.2e}"
              f"   ({nneg} negative-magnitude rows)")
    return out


def ensure_baked(adapter, variant=None, interactive=True, quiet=False, model="sa3-medium"):
    """Runtime hook: refuse an unbaked DoRA, offer to bake it, return the path to load.

    Never silently falls back to loading the base weights — that fallback is precisely what
    hides 10 GB and 20 s behind a working-looking load.
    """
    adapter = Path(adapter)
    if is_baked(adapter):
        return adapter
    cand = baked_path(adapter)
    if cand.exists() and is_baked(cand):
        return cand
    from safetensors import safe_open
    with safe_open(str(adapter), framework="numpy") as f:
        md = f.metadata() or {}
    cfg = json.loads(md.get("lora_config", "{}") or "{}")
    if (cfg.get("adapter_type", "lora")).startswith("lora"):
        return adapter                      # plain lora needs nothing
    v = pick_variant(adapter, variant, model)
    msg = (f"{adapter.name} is a {cfg.get('adapter_type')} adapter with no baked row norms.\n"
           f"  Without them the fold must load the {v} base weights: ~10 GB VRAM and 5-20 s.\n"
           f"  Bake now against '{v}'? [Y/n] ")
    if not interactive or not sys.stdin.isatty():
        raise RuntimeError(
            f"{adapter.name} is unbaked. Run:  python bake_dora.py {adapter} --variant {v}")
    if (input(msg).strip().lower() or "y") not in ("y", "yes"):
        raise RuntimeError("declined; adapter cannot be loaded without baked norms")
    return bake(adapter, v, quiet=quiet)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", nargs="+")
    # Defaults to whichever variant the model's engines were built from. Overriding it is
    # opt-in through a deliberately less-discoverable flag so nobody reaches it by accident.
    ap.add_argument("--norm-source", dest="variant", choices=list(paths.VARIANTS), default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=("sa3-medium",), default="sa3-medium")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if any adapter is unbaked")
    a = ap.parse_args()
    if a.check:
        bad = [p for p in a.adapter if not is_baked(p) and not baked_path(p).exists()]
        for p in bad: print(f"  UNBAKED: {p}")
        return 1 if bad else 0
    for p in a.adapter:
        bake(p, pick_variant(p, a.variant, a.model), a.out, a.force, model=a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
