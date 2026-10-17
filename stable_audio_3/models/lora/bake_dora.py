"""Bake the strength-independent fold quantities into a DoRA / -xs adapter.

Why: the runtime fold needs ``c = magnitude / ||W0 + s*LR||_row``. Computing that denominator
means loading the full base weights purely to take a row norm -- for a deployment runtime that
already carries its own copy of the weights, that is a second, redundant, double-precision copy.
Baking the norms into the adapter removes it entirely.

Training saves these automatically (``save_lora_safetensors(..., model=...)``). This script is
the retrofit path for adapters trained before that existed, and produces the identical key and
metadata layout:

``<layer>.parametrizations.weight.<i>.baked_vnorm_row`` (float32) + a ``__metadata__`` block.

SVD IS NEVER COMPUTED. ``-xs`` bases are sliced from the frozen ``svd_bases.pt`` that the
adapter was trained against; recomputing them per layer is minutes of work and numerically
unstable on the DiT's larger matrices.

Runs as a module (it imports from its own package), or through the bake_dora.sh
wrapper next to this file, which does that for you from any working directory:

    ./stable_audio_3/models/lora/bake_dora.sh <adapter.safetensors> \
        --base-weights /path/to/model.safetensors [--out PATH] [--force]

    python -m stable_audio_3.models.lora.bake_dora <adapter.safetensors> \
        --base-weights /path/to/model.safetensors [--out PATH] [--force]
    python -m stable_audio_3.models.lora.bake_dora <adapter.safetensors> --check
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

from .utils import BAKED_VNORM_KEY as BAKED   # one definition of the key, shared with the save path

PSUF = ".parametrizations.weight."


def is_baked(path) -> bool:
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        return any(BAKED in k for k in f.keys())


def baked_path(path) -> Path:
    p = Path(path)
    return p if p.name.endswith(".normbaked.safetensors") else \
        p.parent / (p.stem + ".normbaked.safetensors")


def parse_adapter(path):
    """``(adapter_type, scaling, layers)``; ``layers`` maps ``(layer_name, lora_index)`` to
    its param dict. Mirrors the grouping the loader does, from the file alone."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = dict(f.metadata() or {})
    layers = {}
    for k, v in tensors.items():
        if PSUF not in k:
            continue
        layer, _, rest = k.partition(PSUF)
        idx, _, param = rest.partition(".")
        if not idx.isdigit():
            continue
        layers.setdefault((layer, int(idx)), {})[param] = v
    if not layers:
        raise KeyError(f"{Path(path).name!r}: no SA3 LoRA parametrization keys")
    cfg = json.loads(meta.get("lora_config", "{}") or "{}")
    rank = int(cfg.get("rank") or _infer_rank(layers))
    alpha = float(cfg.get("alpha", rank))
    atype = cfg.get("adapter_type", "lora")
    atype = "dora-rows" if atype == "dora" else atype   # legacy alias, mirrors resolve_adapter_type
    return atype, alpha / rank, layers, tensors, meta


def _infer_rank(layers):
    for p in layers.values():
        if "lora_A" in p:
            return p["lora_A"].shape[0]
        if "M_xs" in p:
            return p["M_xs"].shape[0]
    raise KeyError("cannot infer LoRA rank (no lora_A / M_xs tensors)")


def _find_base_key(lid, wkeys, wpath):
    """The adapter stores layer ids relative to the sub-module it was applied to; a checkpoint
    wraps them in a prefix whose depth varies by model (``model.``, ``model.model.``, and
    ``conditioner.`` for the non-DiT layers adapters touch, e.g.
    conditioners.seconds_total.embedder.embedding.1). Try the known prefixes, then fall back
    to a unique suffix match so an unforeseen wrapper does not silently fail to bake.
    """
    for pre in ("", "model.", "model.model.", "conditioner.", "model.conditioner."):
        k = f"{pre}{lid}.weight"
        if k in wkeys:
            return k
    hits = [k for k in wkeys if k.endswith(f".{lid}.weight")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise KeyError(f"{lid} is ambiguous in {wpath.name}: {sorted(hits)[:4]}")
    raise KeyError(f"no base weight for {lid} in {wpath.name}")


def bake(adapter, base_weights, out=None, force=False, quiet=False,
         svd_bases=None, norm_source="base", model="stable-audio-3"):
    import numpy as np, torch
    from safetensors import safe_open
    from safetensors.numpy import save_file

    adapter, wpath = Path(adapter), Path(base_weights)
    out = Path(out) if out else baked_path(adapter)
    if out.exists() and not force:
        if not quiet: print(f"  already baked: {out.name}  (--force to redo)")
        return out

    atype, scaling, layers, tensors, meta = parse_adapter(adapter)
    is_xs = atype.endswith("-xs")
    stem = atype[:-3] if is_xs else atype
    if not quiet:
        print(f"  {adapter.name}\n    type={atype} scaling={scaling:.4f} layers={len(layers)} "
              f"norm_source={norm_source}")
    if stem.startswith("lora"):
        if not quiet: print("    plain lora -- fold is already trivial, nothing to bake")
        return None
    if is_xs and not svd_bases:
        raise ValueError(f"{atype} needs --svd-bases (the frozen bases it was trained against)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bases = torch.load(svd_bases, map_location="cpu", weights_only=True, mmap=True) if is_xs else None
    wf = safe_open(str(wpath), framework="pt")
    wkeys = set(wf.keys())

    n, worst, nneg = 0, 0.0, 0
    for (lid, idx), p in layers.items():
        wk = _find_base_key(lid, wkeys, wpath)
        W0 = wf.get_tensor(wk).to(dev, torch.float32)
        W0 = W0.reshape(W0.shape[0], -1)
        if is_xs:
            bk = next((k for k in bases if k.endswith(lid.split("model.", 1)[-1] + ".weight")
                       or k.endswith(lid + ".weight")), None)
            if bk is None:
                raise KeyError(f"no frozen SVD basis for {lid}")
            r = p["M_xs"].shape[0]
            U = bases[bk]["U"][:, :r].to(dev, torch.float32)
            V = bases[bk]["V"][:, :r].to(dev, torch.float32)
            LR = U @ torch.as_tensor(p["M_xs"], device=dev, dtype=torch.float32) @ V.T
        else:
            LR = (torch.as_tensor(p["lora_B"], device=dev, dtype=torch.float32)
                  @ torch.as_tensor(p["lora_A"], device=dev, dtype=torch.float32))
        vn = torch.linalg.norm(W0 + scaling * LR, dim=1)          # ||W0 + s.LR||_row
        # Self-check on a real invariant: the DoRA weight is  W = mag ⊙ (W0+s.LR)/vn , so its
        # row norms must come back as `magnitude`.
        mag = torch.as_tensor(np.squeeze(p["magnitude"]).astype(np.float32), device=dev)
        c = mag / (vn + 1e-12)
        rows = torch.linalg.norm(c.unsqueeze(1) * (W0 + scaling * LR), dim=1)
        # `magnitude` is a free nn.Parameter initialised to ||W0||_row, so training can drive
        # rows NEGATIVE. The DoRA weight is c·V with c signed, so the recovered row norm is
        # |magnitude|, not magnitude -- comparing against the signed value reports a rel error
        # of exactly 2.0 for every negative row and hides real errors behind it.
        worst = max(worst, float(((rows - mag.abs()) / mag.abs().clamp_min(1e-6)).abs().max()))
        nneg += int((mag < 0).sum())
        tensors[f"{lid}{PSUF}{idx}.{BAKED}"] = vn.cpu().numpy().astype(np.float32)
        n += 1

    cfg = json.loads(meta.get("lora_config", "{}") or "{}")
    cfg["baked_norms"] = n
    cfg["norm_base"] = f"{model}-{norm_source}"
    meta["lora_config"] = json.dumps(cfg)
    meta.update({
        "base_model": f"{model}-{norm_source}", "norm_source": norm_source,
        "norm_base_sha": hashlib.sha256(wpath.name.encode()).hexdigest()[:32],  # recorded, not enforced
        "baked_variant": atype, "baked_norms": str(n),
        "baked_xs_svd": ("external:" + Path(svd_bases).name) if is_xs else "none",
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


def ensure_baked(adapter, base_weights=None, interactive=True, quiet=False, **kw):
    """Runtime hook: refuse an unbaked DoRA, offer to bake it, return the path to load.

    Never silently falls back to loading the base weights -- that fallback is precisely what
    hides the redundant weight copy behind a working-looking load.
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
    if not base_weights:
        raise RuntimeError(f"{adapter.name} is unbaked and no --base-weights was given")
    msg = (f"{adapter.name} is a {cfg.get('adapter_type')} adapter with no baked row norms.\n"
           f"  Without them the fold must load {Path(base_weights).name} to take a row norm.\n"
           f"  Bake now? [Y/n] ")
    if not interactive or not sys.stdin.isatty():
        raise RuntimeError(f"{adapter.name} is unbaked. Run:  python -m "
                           f"stable_audio_3.models.lora.bake_dora {adapter} "
                           f"--base-weights {base_weights}")
    if (input(msg).strip().lower() or "y") not in ("y", "yes"):
        raise RuntimeError("declined; adapter cannot be loaded without baked norms")
    return bake(adapter, base_weights, quiet=quiet, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", nargs="+")
    ap.add_argument("--base-weights", default=None,
                    help="safetensors holding the W0 the adapter was trained against")
    ap.add_argument("--svd-bases", default=None, help="frozen svd_bases.pt (dora-*-xs only)")
    # A baked norm is only meaningful against the W0 the runtime actually carries -- base and
    # arc reconstruct each other's norms at rel ~2e-3 with nothing raising -- so this label is
    # recorded in the metadata to make a wrong pairing visible after the fact.
    ap.add_argument("--norm-source", default="base", help="label recorded in the metadata")
    ap.add_argument("--model", default="stable-audio-3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if any adapter is unbaked")
    a = ap.parse_args()
    if a.check:
        bad = [p for p in a.adapter if not is_baked(p) and not baked_path(p).exists()]
        for p in bad: print(f"  UNBAKED: {p}")
        return 1 if bad else 0
    if not a.base_weights:
        ap.error("--base-weights is required to bake")
    for p in a.adapter:
        bake(p, a.base_weights, a.out, a.force, svd_bases=a.svd_bases,
             norm_source=a.norm_source, model=a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
