"""LoRA merge-at-load for the MLX SA3 DiT.

Adds LoRA inference support to the MLX path, which the `sa3_mlx.py` CLI otherwise
lacks. The LoRA delta is **merged into the DiT weight dict at load time**, before
the model is built — no runtime parametrization and no extra per-step forward
cost. A strength of 0 is a bit-exact bypass.

Trust boundary: only `.safetensors` adapters are accepted. The legacy pickle
`.ckpt`/`.pt`/`.bin` path — which `torch.load` would execute arbitrary code from —
is refused outright; this module never calls `torch.load`.

Two on-disk conventions are supported:

  * **SA3-native** (`scripts/train_lora.py` output): tensor keys
    ``<layer>.parametrizations.weight.0.{lora_A,lora_B,M_xs,magnitude,
    magnitude_r,magnitude_c}`` with the adapter config
    (``adapter_type``/``rank``/``alpha``/``include``/``exclude``) JSON-encoded in
    the safetensors **metadata** under ``"lora_config"``. Covers all nine adapter
    types (lora, dora-rows/cols, bora, and the four -xs variants). Underfit
    (github.com/dada-bots/underfit) checkpoints are this convention with
    full-wrapper layer names (``model.…`` / ``conditioners.…``) — handled by
    ``_layer_to_npz_key``.
  * **PEFT** (huggingface `peft`): keys ``base_model.model.<layer>.lora_{A,B}.weight``
    with ``r``/``lora_alpha`` in a sibling ``adapter_config.json``. Standard LoRA,
    plus DoRA when ``use_dora`` is set.

The per-adapter-type math mirrors ``LoRAParametrization.*_forward`` in
``stable_audio_3/models/lora/model.py`` (and the accumulate-deltas-against-the-
original-weight semantics of ``merge_loras_into_base_model``), computed in
float32 and cast back to the DiT dtype. `-xs` adapters do not store their frozen
SVD bases, so they are recomputed from the base weight here, matching the
reference (`torch.linalg.svd` + a deterministic sign convention).
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np

# Pickle-backed extensions we refuse to load (the trust boundary).
_PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin")

# Adapter param names per type (mirrors utils._get_adapter_param_names).
_PARAMS_FOR = {
    "lora": ("lora_A", "lora_B"),
    "dora-rows": ("lora_A", "lora_B", "magnitude"),
    "dora-cols": ("lora_A", "lora_B", "magnitude"),
    "bora": ("lora_A", "lora_B", "magnitude_r", "magnitude_c"),
    "lora-xs": ("M_xs",),
    "dora-rows-xs": ("M_xs", "magnitude"),
    "dora-cols-xs": ("M_xs", "magnitude"),
    "bora-xs": ("M_xs", "magnitude_r", "magnitude_c"),
}


class LoraError(Exception):
    """An adapter could not be loaded or applied."""


# ── safetensors reading (no torch, no safetensors pkg — MLX reads it) ──────────

def _np(arr) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32)


def _load_safetensors(path: str):
    """Return ``(tensors: dict[str, np.ndarray], metadata: dict)``. Refuses pickle."""
    lower = path.lower()
    if lower.endswith(_PICKLE_EXTS):
        raise LoraError(
            f"refusing to load pickle-format adapter {os.path.basename(path)!r} — "
            f"only .safetensors adapters are accepted (a .ckpt/.pt is unpickled by "
            f"torch.load and can execute arbitrary code)"
        )
    if not lower.endswith(".safetensors"):
        raise LoraError(f"not a .safetensors adapter: {path!r}")
    arrs, meta = mx.load(path, return_metadata=True)
    return {k: _np(v) for k, v in arrs.items()}, (meta or {})


# ── SVD bases for -xs adapters (recomputed; mirrors model.py) ──────────────────

def _canonicalize_svd_signs(U: np.ndarray, Vh: np.ndarray):
    """Deterministic sign convention: largest-magnitude element of each U column
    is positive (mirrors model._canonicalize_svd_signs)."""
    max_abs_idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs[None, :], Vh * signs[:, None]


def _svd_bases(W0: np.ndarray, rank: int):
    """Return ``(U[:, :rank], V[:, :rank])`` from the SVD of ``W0`` (fan_out, fan_in),
    with V such that ``U @ diag(S) @ V.T`` reconstructs W0 (mirrors model.py)."""
    U_full, _S, Vh_full = np.linalg.svd(W0, full_matrices=False)
    U_full, Vh_full = _canonicalize_svd_signs(U_full, Vh_full)
    U = U_full[:, :rank]
    V = Vh_full[:rank, :].T
    return U, V


# ── per-type merge math (numpy, float32) ───────────────────────────────────────

def _merged_weight(W0: np.ndarray, p: dict, adapter_type: str, scaling: float) -> np.ndarray:
    """Return the LoRA-merged weight for one layer at full strength (lora_strength=1).

    ``W0`` is (fan_out, fan_in) float32; ``p`` holds the adapter tensors for this
    layer. Mirrors the matching ``*_forward`` in model.py.
    """
    if adapter_type == "lora":
        delta = p["lora_B"] @ p["lora_A"]
        return W0 + scaling * delta

    if adapter_type in ("dora-rows", "dora-cols"):
        norm_dim = 1 if adapter_type == "dora-rows" else 0
        delta = p["lora_B"] @ p["lora_A"]
        V = W0 + scaling * delta
        V_hat = V / (np.linalg.norm(V, axis=norm_dim, keepdims=True) + 1e-12)
        mag = _mag_2d(p["magnitude"], norm_dim)
        return V_hat * mag

    if adapter_type == "bora":
        delta = p["lora_B"] @ p["lora_A"]
        V = W0 + scaling * delta
        V_r = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        inter = p["magnitude_r"].reshape(-1, 1) * V_r
        H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
        return H_c * p["magnitude_c"].reshape(1, -1)

    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        U, V = _svd_bases(W0, rank)
        delta = U @ p["M_xs"] @ V.T
        Vfull = W0 + scaling * delta
        if adapter_type == "lora-xs":
            return Vfull
        if adapter_type in ("dora-rows-xs", "dora-cols-xs"):
            norm_dim = 1 if adapter_type == "dora-rows-xs" else 0
            V_hat = Vfull / (np.linalg.norm(Vfull, axis=norm_dim, keepdims=True) + 1e-12)
            mag = _mag_2d(p["magnitude"], norm_dim)
            return V_hat * mag
        if adapter_type == "bora-xs":
            V_r = Vfull / (np.linalg.norm(Vfull, axis=1, keepdims=True) + 1e-12)
            inter = p["magnitude_r"].reshape(-1, 1) * V_r
            H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
            return H_c * p["magnitude_c"].reshape(1, -1)

    raise LoraError(f"unknown adapter_type {adapter_type!r}")


def _check_shapes(layer: str, W0: np.ndarray, p: dict, adapter_type: str) -> None:
    """Fail with a clear message when the adapter doesn't fit the base weight —
    almost always because the adapter was trained for a different base than
    ``--dit`` (e.g. a medium adapter on sm-music). Without this the mismatch
    surfaces as a raw numpy broadcasting error deep in the merge."""
    fan_out, fan_in = W0.shape
    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        if rank > min(fan_out, fan_in):
            raise LoraError(
                f"{layer}: LoRA-XS rank {rank} exceeds base min-dim "
                f"{min(fan_out, fan_in)} for weight {W0.shape} — wrong base for --dit?"
            )
        return
    b_out, b_rank = p["lora_B"].shape
    a_rank, a_in = p["lora_A"].shape
    if b_out != fan_out or a_in != fan_in or a_rank != b_rank:
        raise LoraError(
            f"{layer}: adapter lora_B{p['lora_B'].shape}·lora_A{p['lora_A'].shape} "
            f"does not fit base weight {W0.shape} — wrong base for --dit?"
        )


def _mag_2d(mag: np.ndarray, norm_dim: int) -> np.ndarray:
    """Reshape a (possibly 2D) magnitude vector to broadcast against the weight on
    ``norm_dim`` (mirrors `magnitude.unsqueeze(norm_dim)` after a squeeze).
    ``atleast_1d`` guards the degenerate (1, 1) case where squeeze yields a
    0-d array (no real DiT layer has a single output, but keep it total)."""
    mag = np.atleast_1d(np.squeeze(mag))
    return mag.reshape(-1, 1) if norm_dim == 1 else mag.reshape(1, -1)


# ── checkpoint parsing → normalized per-layer adapter ──────────────────────────

# Underfit (github.com/dada-bots/underfit) saves adapters with full-wrapper
# layer names: DiT layers as ``model.transformer...`` and the seconds
# conditioner Linear as ``conditioners.seconds_total.embedder.embedding.1``.
# The MLX npz bakes that conditioner Linear as ``cond.seconds_total_weight``.
_COND_SECONDS_LAYER = "conditioners.seconds_total.embedder.embedding.1"


def _layer_to_npz_key(layer: str) -> str:
    """Map a checkpoint layer name to its DiT npz weight key. Strips the
    full-model ``model.`` prefix (underfit / torch-wrapper checkpoints; bare-DiT
    names pass through), maps the seconds-conditioner Linear onto its baked npz
    key, and renames ``to_local_embed.{0,2}`` → ``to_local_embed.seq.{0,2}``
    (dit_mlx.py); every other Linear/Conv1d name passes through unchanged."""
    if layer == _COND_SECONDS_LAYER:
        return "cond.seconds_total_weight"
    if layer.startswith("model."):
        layer = layer[len("model."):]
    layer = layer.replace(".to_local_embed.0", ".to_local_embed.seq.0")
    layer = layer.replace(".to_local_embed.2", ".to_local_embed.seq.2")
    return f"{layer}.weight"


def _resolve_path(path: str) -> str:
    """Accept a .safetensors file or a PEFT adapter directory (resolve to the
    adapter_model.safetensors inside it)."""
    if os.path.isdir(path):
        cand = os.path.join(path, "adapter_model.safetensors")
        if os.path.isfile(cand):
            return cand
        hits = [f for f in os.listdir(path) if f.lower().endswith(".safetensors")]
        if len(hits) == 1:
            return os.path.join(path, hits[0])
        raise LoraError(
            f"{path!r}: expected one .safetensors adapter in the directory, found {hits}"
        )
    return path


def _parse_adapter(path: str):
    """Load one adapter and return ``(adapter_type, scaling, layers)`` where
    ``layers`` maps a checkpoint layer name → its param dict (numpy float32)."""
    tensors, meta = _load_safetensors(path)

    native_marker = ".parametrizations.weight.0."
    is_native = any(native_marker in k for k in tensors)

    if is_native:
        cfg = json.loads(meta.get("lora_config", "{}")) if meta else {}
        layers = _group_native(tensors)
        rank = int(cfg.get("rank") or _infer_rank(layers))
        alpha = float(cfg.get("alpha", rank))
        adapter_type = _resolve_native_type(cfg.get("adapter_type", "lora"))
        scaling = alpha / rank
        return adapter_type, scaling, layers

    # PEFT — config lives in a sibling adapter_config.json
    peft_marker = ".lora_A.weight"
    if any(k.endswith(peft_marker) for k in tensors):
        cfg = _read_peft_config(path)
        rank = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha", rank))
        use_dora = bool(cfg.get("use_dora", False))
        use_rslora = bool(cfg.get("use_rslora", False))
        adapter_type = "dora-rows" if use_dora else "lora"
        scaling = alpha / (np.sqrt(rank) if use_rslora else rank)
        layers = _group_peft(tensors)
        return adapter_type, scaling, layers

    raise LoraError(
        f"{os.path.basename(path)!r}: not a recognised LoRA (no SA3-native "
        f"parametrization keys and no PEFT lora_A/lora_B keys)"
    )


def _group_native(tensors: dict) -> dict:
    marker = ".parametrizations.weight.0."
    layers: dict[str, dict] = {}
    for k, v in tensors.items():
        if marker not in k:
            continue
        layer, _, param = k.partition(marker)
        layers.setdefault(layer, {})[param] = v
    return layers


def _group_peft(tensors: dict) -> dict:
    prefix = "base_model.model."
    layers: dict[str, dict] = {}
    for k, v in tensors.items():
        name = k[len(prefix):] if k.startswith(prefix) else k
        for suffix, param in ((".lora_A.weight", "lora_A"),
                              (".lora_B.weight", "lora_B"),
                              (".lora_magnitude_vector.weight", "magnitude")):
            if name.endswith(suffix):
                layers.setdefault(name[: -len(suffix)], {})[param] = v
                break
    return layers


def _read_peft_config(path: str) -> dict:
    base = os.path.dirname(path)
    cfg_path = os.path.join(base, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise LoraError(
            f"PEFT adapter at {path!r} is missing its adapter_config.json sibling"
        )
    with open(cfg_path) as fh:
        return json.load(fh)


def _infer_rank(layers: dict) -> int:
    for params in layers.values():
        if "lora_A" in params:
            return params["lora_A"].shape[0]
        if "M_xs" in params:
            return params["M_xs"].shape[0]
    raise LoraError("cannot infer LoRA rank (no lora_A / M_xs tensors)")


def _resolve_native_type(adapter_type: str) -> str:
    """Legacy 'dora' → 'dora-rows' (the paper-correct default; mirrors
    utils.resolve_adapter_type, minus the 2D-magnitude shape sniff we don't need
    because saved magnitudes are 1D)."""
    return "dora-rows" if adapter_type == "dora" else adapter_type


# ── public entry point ─────────────────────────────────────────────────────────

def merge_loras_into_weights(weights: dict, lora_paths, strength: float = 1.0,
                             log=lambda _m: None) -> dict:
    """Merge one or more LoRA adapters into ``weights`` in place.

    ``weights`` is the DiT weight dict as loaded from the npz (str → mx.array).
    ``strength`` is the application weight applied to every adapter's delta (the
    `--lora-strength` knob; matches ``application_weight`` in
    ``merge_loras_into_base_model``). Deltas are accumulated against the original
    weight, then applied once, so stacking is order-independent for linear LoRA.

    Returns a stats dict ``{"merged": int, "skipped": list[str], "adapters": int}``.
    """
    if not lora_paths:
        return {"merged": 0, "skipped": [], "adapters": 0}

    parsed = []
    for raw in lora_paths:
        path = _resolve_path(raw)
        adapter_type, scaling, layers = _parse_adapter(path)
        parsed.append((path, adapter_type, scaling, layers))
        log(f"lora: {os.path.basename(path)} — {adapter_type}, "
            f"scaling={scaling:.3f}, {len(layers)} target layers")

    # Accumulate deltas per npz key against the *original* weight. Each entry is
    # [summed_delta, restore] — the layout restorer is the same across repeats.
    accum: dict[str, list] = {}
    skipped: list[str] = []
    for path, adapter_type, scaling, layers in parsed:
        need = _PARAMS_FOR.get(adapter_type, ())
        for layer, params in layers.items():
            key = _layer_to_npz_key(layer)
            if key not in weights:
                skipped.append(layer)
                continue
            missing = [n for n in need if n not in params]
            if missing:
                raise LoraError(f"{layer}: adapter is {adapter_type} but missing {missing}")
            W0, restore = _weight_as_2d(weights[key])
            _check_shapes(layer, W0, params, adapter_type)
            merged = _merged_weight(W0, params, adapter_type, scaling)
            delta = strength * (merged - W0)
            if key in accum:
                accum[key][0] += delta
            else:
                accum[key] = [delta, restore]

    for key, (delta, restore) in accum.items():
        W0, _ = _weight_as_2d(weights[key])
        weights[key] = mx.array(restore(W0 + delta))

    if skipped:
        log(f"lora: skipped {len(skipped)} layer(s) not in this DiT "
            f"(e.g. {skipped[0]})")
    if not accum:
        log("lora: WARNING — merged 0 layers; the adapter targets nothing in this "
            "DiT (wrong base for --dit, or unsupported target modules)")
    return {"merged": len(accum), "skipped": skipped, "adapters": len(parsed)}


def _weight_as_2d(arr):
    """Return ``(W2d, restore)`` where ``W2d`` is the PyTorch-layout 2D weight
    (fan_out, fan_in) as numpy float32, and ``restore(W2d)`` rebuilds the MLX
    layout. Linear weights are already 2D == PyTorch layout; Conv1d weights are
    stored MLX-style (out, k, in) and round-trip through PyTorch (out, in, k)."""
    np_arr = _np(arr)
    if np_arr.ndim == 2:
        return np_arr, lambda w: w.astype(np.float32)
    if np_arr.ndim == 3:
        out, k, cin = np_arr.shape
        w2d = np_arr.transpose(0, 2, 1).reshape(out, cin * k)  # (out, in*k), PyTorch order

        def restore(w):
            return w.reshape(out, cin, k).transpose(0, 2, 1).astype(np.float32)

        return w2d, restore
    raise LoraError(f"unexpected weight rank {np_arr.ndim} for a LoRA target")
