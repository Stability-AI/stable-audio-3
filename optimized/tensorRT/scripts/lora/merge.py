"""Merge a LoRA-family stack into the adapted linears of the SA3-medium DiT.

Semantics are taken from the shipped MLX path (optimized/mlx/models/defs/lora.py
:func:`_apply_checkpoint_layer` + :func:`apply_lora_checkpoints`), which is the product's ground
truth, not invented here:

  * strength scales the LOW-RANK DELTA INSIDE V, before any DoRA/BoRA renormalisation
    (``value = W + (alpha/rank)*strength*delta``), so strength is not a linear fade of the whole
    adapter for the renormalising variants;
  * a stack is applied SEQUENTIALLY and is ORDER-DEPENDENT — each adapter takes the previous
    adapter's output as its base, which for the -xs variants also means its SVD bases are taken
    from the running weight.

Both fall out for free when you merge. Note a folded runtime branch CANNOT reproduce the second
one: it superposes per-adapter deltas instead of composing them, so it is only exact for stacks
of pure lora/lora-xs. That is a correctness argument for merging, on top of the speed one.
"""
from pathlib import Path
import numpy as np

import paths  # noqa: F401  -- puts the shared lora_core on sys.path
import lora_core as lc          # reference merge math, all 8 adapter variants


def _install_torch_svd():
    """Route the -xs SVD through torch.

    ⚠ Some numpy/OpenBLAS builds cannot SVD the DiT's widest weight matrices: they raise
    "DLASCL parameter number 4 had an illegal value" / "SVD did not converge" on e.g. (16384,2048).
    Same OpenBLAS root cause as its mis-computed float64 matmuls at this size. torch's
    LAPACK handles it, on GPU when available. Sign canonicalisation is kept identical to
    lora_core._canonicalize_svd_signs so the -xs bases match the reference exactly.
    """
    import torch

    def _svd_bases(W0, rank):
        t = torch.as_tensor(np.ascontiguousarray(W0, dtype=np.float32))
        if torch.cuda.is_available():
            t = t.cuda()
        U, _S, Vh = torch.linalg.svd(t, full_matrices=False)
        U = U.cpu().numpy(); Vh = Vh.cpu().numpy()
        U, Vh = lc._canonicalize_svd_signs(U, Vh)
        return U[:, :rank], Vh[:rank, :].T

    lc._svd_bases = _svd_bases


_install_torch_svd()

TARGET_SUFFIXES = ("self_attn.to_qkv", "self_attn.to_out", "cross_attn.to_q",
                   "cross_attn.to_kv", "cross_attn.to_out", "ff.ff.0.proj", "ff.ff.2")


def refit_name(layer_key):
    """'model.transformer.layers.7.ff.ff.0.proj' -> the engine weight name used by build_refit.py"""
    k = layer_key
    for p in ("model.transformer.", "transformer.", "model."):
        if k.startswith(p):
            k = k[len(p):]
            break
    return "lora::" + k


# ── layers no adapter may touch ──────────────────────────────────────────────────────────────
# `conditioners.seconds_total.embedder.embedding.1` is the SECONDS EMBEDDER: a [256]->[768]
# linear whose output is GLOBAL conditioning -- it feeds the adaLN modulation of all 24 blocks
# -- rather than a residual-stream projection like every other target. Two things follow, and
# both say adapters should leave it alone:
#
#   * it encodes DURATION, not style, so there is nothing there for a style adapter to learn;
#   * a stack SUMS its rescales (merge_loras_into_base_model semantics: every delta is taken
#     against the same W0), so two adapters that each pull this row to c ~ 0.2 give
#     1 + sum(c-1) ~ -0.6 -- sign-flipped. Because the row is global, that inverted gain
#     multiplies every token in every block, the activations blow past the fp16 range, and the
#     render comes back ALL-NaN with no error raised anywhere.
#
# Measured 2026-10-07 on the 229-target branch engine: forcing 1+pout at this one layer to
# -0.20 triples max||v|| (5.4 -> 17.2); <= -0.50 NaNs all 330752 outputs. Enabling it alone
# reproduces the failure; all 228 other targets together are clean. The 228-target engine,
# which never served it, was immune the whole time.
#
# CJ's call (2026-10-07): adapters should never train this layer in the first place. Ones that
# already did are warned about and have it dropped, here, for every consumer -- branch engine,
# merge/refit engine and the reference merge alike.
EXCLUDED_LAYERS = ("conditioners.seconds_total.embedder.embedding.1",)

# Escape hatch for A/B only: SA3_LORA_NO_EXCLUDE=1 restores the pre-2026-10-07 behaviour, so
# the cost of the exclusion can be heard rather than inferred from a norm. It will happily
# return NaN audio for a two-DoRA stack -- that is the point of it.
import os as _os
if _os.environ.get("SA3_LORA_NO_EXCLUDE", "") not in ("", "0", "false", "False"):
    EXCLUDED_LAYERS = ()

_warned_excluded = set()


def drop_excluded(stack, warn=True):
    """Remove EXCLUDED_LAYERS from every adapter in `stack`, in place.

    Returns {adapter path: [dropped keys]}. Warned once per adapter path per process so a UI
    that swaps repeatedly does not spam, which is the same rule warn_unserved uses.
    """
    import warnings
    dropped = {}
    for a in stack:
        hit = sorted(k for k in a["layers"] if k in EXCLUDED_LAYERS)
        if not hit:
            continue
        for k in hit:
            del a["layers"][k]
        dropped[a["path"]] = hit
        if warn and a["path"] not in _warned_excluded:
            _warned_excluded.add(a["path"])
            warnings.warn(
                f"{Path(a['path']).name}: adapts {len(hit)} EXCLUDED layer(s) -- "
                f"{', '.join(hit)} -- which are IGNORED. The seconds embedder is global "
                f"conditioning (adaLN into every block), not a residual-stream projection; "
                f"stacking two adapters that both rescale it sign-flips the gain and NaNs the "
                f"render. Retrain without it; its effect here was measured at 0.6-17.75% of "
                f"an adapter.", RuntimeWarning, stacklevel=3)
    return dropped


def load_stack(specs, drop_excluded_layers=True):
    """specs: [(path, strength), ...] -> [{type, scaling, strength, layers}] in stack order.

    EXCLUDED_LAYERS are stripped by default (see above). Pass drop_excluded_layers=False only
    to reproduce the old, unfiltered behaviour -- the verification harnesses use it to price
    what the exclusion costs against a torch reference that still applies the layer.
    """
    out = []
    for path, strength in specs:
        atype, scaling, layers = lc.parse_adapter(str(path))
        out.append({"path": str(path), "type": atype, "scaling": float(scaling),
                    "strength": float(strength), "layers": layers})
    if drop_excluded_layers:
        drop_excluded(out)
    return out


def stack_layer_keys(stack):
    keys = set()
    for a in stack:
        keys.update(a["layers"].keys())
    return keys


def merge_layer(W0, stack, layer_key, strict=False):
    """Apply the stack, in order, to one layer's weight. Returns fp32 (out,in) or None if no
    adapter in the stack touches this layer."""
    W = None
    for a in stack:
        p = a["layers"].get(layer_key)
        if p is None:
            continue
        if a["strength"] == 0.0:
            continue
        cur = W0 if W is None else W
        if strict:
            lc.check_shapes(layer_key, cur, p, a["type"])
        # scaling*strength reproduces MLX's `(alpha/rank)*strength*delta` exactly
        W = lc.merged_weight(np.asarray(cur, np.float32), p, a["type"],
                             a["scaling"] * a["strength"])
    return W


def merge_layer_additive(W0, stack, layer_key):
    """`merge_loras_into_base_model` semantics: every delta taken against the SAME W0, summed.

    This is what the BRANCH engine computes, and it is not what merge_layer() above computes.
    Two differences, both deliberate:

      * composition -- summed here, CHAINED there (chaining is what load_and_apply_loras, the
        MLX path and our own refit engine do);
      * strength    -- an `application_weight` on the finished delta here (a linear fade,
        matching the branch's set_strengths), INSIDE V there (so it is inside the DoRA
        renormalisation and the fade is not linear).

    Kept so a verification harness can compare the branch against the semantics it actually
    implements, instead of against the chained reference -- which for a stack of drifted DoRAs
    differs by more than the adapters' own effect (rel ~1.0-1.3 measured) and swamps any real
    wiring error.
    """
    W0 = np.asarray(W0, np.float32)
    acc = None
    for a in stack:
        p = a["layers"].get(layer_key)
        if p is None or a["strength"] == 0.0:
            continue
        d = a["strength"] * (lc.merged_weight(W0, p, a["type"], a["scaling"]) - W0)
        acc = d if acc is None else acc + d
    return None if acc is None else W0 + acc


def merge_all(base_weights, stack, strict=True):
    """base_weights: {layer_key: (out,in) fp32}. Returns {layer_key: merged fp32} for the layers
    the stack actually touches AND that exist in the base (others are reported, not merged)."""
    merged, skipped = {}, []
    for key in sorted(stack_layer_keys(stack)):
        if key not in base_weights:
            skipped.append(key)
            continue
        W = merge_layer(base_weights[key], stack, key, strict=strict)
        if W is not None:
            merged[key] = np.ascontiguousarray(W, dtype=np.float32)
    return merged, skipped


def is_target(layer_key):
    return any(layer_key.endswith(s) for s in TARGET_SUFFIXES)


# ── GPU merge ────────────────────────────────────────────────────────────────
# The reference path is numpy on CPU, which costs tens of seconds across the adapted layers (a [out,r]@[r,in]
# product per layer on a bad BLAS). The same math in torch on device is ~100x faster and the
# weights are already there. Mirrors lora_core.merged_weight exactly, variant for variant.

def merged_weight_torch(W0, p, atype, scaling):
    import torch
    dev = W0.device
    T = lambda k: torch.as_tensor(p[k], dtype=torch.float32, device=dev)
    if atype.endswith("-xs"):
        r = p["M_xs"].shape[0]
        U, _S, Vh = torch.linalg.svd(W0, full_matrices=False)
        # same deterministic sign convention as lora_core._canonicalize_svd_signs
        idx = U.abs().argmax(dim=0)
        sg = torch.sign(U[idx, torch.arange(U.shape[1], device=dev)])
        sg[sg == 0] = 1.0
        U = U * sg[None, :]; Vh = Vh * sg[:, None]
        Ur, Vr = U[:, :r], Vh[:r, :].T
        V = W0 + scaling * (Ur @ T("M_xs") @ Vr.T)
    else:
        V = W0 + scaling * (T("lora_B") @ T("lora_A"))
    base = atype[:-3] if atype.endswith("-xs") else atype
    if base == "lora":
        return V
    if base in ("dora-rows", "dora-cols"):
        dim = 1 if base == "dora-rows" else 0
        Vh_ = V / (V.norm(dim=dim, keepdim=True) + 1e-12)
        mag = T("magnitude").reshape(-1, 1) if dim == 1 else T("magnitude").reshape(1, -1)
        return Vh_ * mag
    if base == "bora":
        Vr_ = V / (V.norm(dim=1, keepdim=True) + 1e-12)
        inter = T("magnitude_r").reshape(-1, 1) * Vr_
        Hc = inter / (inter.norm(dim=0, keepdim=True) + 1e-12)
        return Hc * T("magnitude_c").reshape(1, -1)
    raise ValueError(f"unknown adapter_type {atype!r}")


def merge_layer_additive_torch(W0, stack, layer_key):
    """merge_layer_additive on the device. W0: (out,in) float32 cuda tensor, or None.

    Same semantics as the numpy twin -- every delta taken against the SAME W0 and summed,
    with strength as an application_weight on the finished delta -- so a harness can compare
    the branch engine against what it actually computes.

    Exists because the numpy path is not viable as a gate: it costs ~5 s per layer on the
    widest weights (the DoRA row norms over [7680,1536] on CPU), and a caller holding the
    weights on the GPU pays a device->host->device round trip per layer on top. Measured on a
    168-layer reference build: 7.7 min -> seconds.
    """
    acc = None
    for a in stack:
        p = a["layers"].get(layer_key)
        if p is None or a["strength"] == 0.0:
            continue
        d = a["strength"] * (merged_weight_torch(W0, p, a["type"], a["scaling"]) - W0)
        acc = d if acc is None else acc + d
    return None if acc is None else W0 + acc


def merge_layer_torch(W0, stack, layer_key):
    """Device-resident sequential merge. W0: (out,in) float32 cuda tensor."""
    W = None
    for a in stack:
        p = a["layers"].get(layer_key)
        if p is None or a["strength"] == 0.0:
            continue
        cur = W0 if W is None else W
        W = merged_weight_torch(cur, p, a["type"], a["scaling"] * a["strength"])
    return W
