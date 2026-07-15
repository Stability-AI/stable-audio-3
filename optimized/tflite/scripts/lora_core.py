"""LoRA adapter parsing + per-layer merge math — torch/mlx-free (numpy only).

The TFLite route ships standalone (its own bootstrap, its own venv, runs on
Windows), so it can't import the MLX runtime. This module is a faithful vendored
copy of the *backend-agnostic* half of
``optimized/mlx/models/defs/lora_merge.py`` — the safetensors parsing and the
per-adapter-type ``W' = f(W0, adapter)`` math — with the two MLX touch-points
replaced by pure numpy:

  * safetensors is read with a ~25-line numpy reader (JSON header + a raw blob),
    instead of ``mx.load``;
  * everything else (``_parse_adapter``, ``_group_native``/``_group_peft``,
    ``_merged_weight`` for all nine adapter types, ``_check_shapes``,
    ``_svd_bases`` for the ``-xs`` variants, ``parse_lora_spec``) is copied
    verbatim, since it was already numpy-fp32.

KEEP IN SYNC with optimized/mlx/models/defs/lora_merge.py — the merge math must
match the MLX path bit-for-bit (both mirror ``LoRAParametrization.*_forward`` in
``stable_audio_3/models/lora/model.py``). What is intentionally NOT here: the
MLX weight-dict merge (``merge_loras_into_weights``) and the per-step
``LoraStepPlan`` — TFLite patches frozen FlatBuffer weight buffers instead (see
``lora_patch.py``), and per-step gating is not feasible on a frozen graph.

Trust boundary: only ``.safetensors`` adapters are accepted; a pickle
``.ckpt``/``.pt``/``.bin`` (which ``torch.load`` would execute code from) is
refused — this module never imports torch.
"""

from __future__ import annotations

import json
import os
import struct

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

# Underfit saves the seconds conditioner Linear under this checkpoint name; the
# baked TFLite DiT carries it in-graph as the unique (768, 256) FULLY_CONNECTED
# (lora_patch maps it there).
COND_SECONDS_LAYER = "conditioners.seconds_total.embedder.embedding.1"


class LoraError(Exception):
    """An adapter could not be loaded or applied."""


# ── numpy safetensors reader (no torch, no safetensors pkg) ────────────────────

_ST_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": None,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _load_safetensors(path: str):
    """Return ``(tensors: dict[str, np.ndarray fp32], metadata: dict)``.

    Refuses pickle. bf16 is widened to fp32 via a uint16<<16 bit-cast (numpy has
    no native bf16). Every returned tensor is fp32 so the merge math is uniform.
    """
    lower = path.lower()
    if lower.endswith(_PICKLE_EXTS):
        raise LoraError(
            f"refusing to load pickle-format adapter {os.path.basename(path)!r} — "
            f"only .safetensors adapters are accepted (a .ckpt/.pt is unpickled by "
            f"torch.load and can execute arbitrary code)"
        )
    if not lower.endswith(".safetensors"):
        raise LoraError(f"not a .safetensors adapter: {path!r}")
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
        blob = f.read()
    meta = hdr.pop("__metadata__", {}) or {}
    out: dict[str, np.ndarray] = {}
    for k, v in hdr.items():
        if v["dtype"] not in _ST_DTYPES:
            raise LoraError(f"{os.path.basename(path)}: unsupported tensor dtype "
                            f"{v['dtype']!r} for {k}")
        b0, b1 = v["data_offsets"]
        n = 1
        for d in v["shape"]:
            n *= d
        if v["dtype"] == "BF16":
            u = np.frombuffer(blob, np.uint16, count=n or 1, offset=b0)
            arr = (u.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(blob[b0:b1], _ST_DTYPES[v["dtype"]])
        out[k] = arr.reshape(v["shape"]).astype(np.float32, copy=False)
    return out, meta


# ── SVD bases for -xs adapters (recomputed; mirrors model.py) ──────────────────

def _canonicalize_svd_signs(U: np.ndarray, Vh: np.ndarray):
    """Deterministic sign convention: largest-magnitude element of each U column
    is positive (mirrors model._canonicalize_svd_signs)."""
    max_abs_idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs[None, :], Vh * signs[:, None]


def _svd_bases(W0: np.ndarray, rank: int):
    """``(U[:, :rank], V[:, :rank])`` from the SVD of W0 (fan_out, fan_in), with
    V such that ``U @ diag(S) @ V.T`` reconstructs W0 (mirrors model.py)."""
    U_full, _S, Vh_full = np.linalg.svd(W0, full_matrices=False)
    U_full, Vh_full = _canonicalize_svd_signs(U_full, Vh_full)
    return U_full[:, :rank], Vh_full[:rank, :].T


# ── per-type merge math (numpy, float32) — mirrors lora_merge._merged_weight ───

def _mag_2d(mag: np.ndarray, norm_dim: int) -> np.ndarray:
    mag = np.atleast_1d(np.squeeze(mag))
    return mag.reshape(-1, 1) if norm_dim == 1 else mag.reshape(1, -1)


def merged_weight(W0: np.ndarray, p: dict, adapter_type: str, scaling: float) -> np.ndarray:
    """LoRA-merged weight for one layer at full strength (lora_strength=1).
    ``W0`` is (fan_out, fan_in) float32; ``p`` holds the adapter tensors."""
    if adapter_type == "lora":
        return W0 + scaling * (p["lora_B"] @ p["lora_A"])

    if adapter_type in ("dora-rows", "dora-cols"):
        norm_dim = 1 if adapter_type == "dora-rows" else 0
        V = W0 + scaling * (p["lora_B"] @ p["lora_A"])
        V_hat = V / (np.linalg.norm(V, axis=norm_dim, keepdims=True) + 1e-12)
        return V_hat * _mag_2d(p["magnitude"], norm_dim)

    if adapter_type == "bora":
        V = W0 + scaling * (p["lora_B"] @ p["lora_A"])
        V_r = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        inter = p["magnitude_r"].reshape(-1, 1) * V_r
        H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
        return H_c * p["magnitude_c"].reshape(1, -1)

    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        U, V = _svd_bases(W0, rank)
        Vfull = W0 + scaling * (U @ p["M_xs"] @ V.T)
        if adapter_type == "lora-xs":
            return Vfull
        if adapter_type in ("dora-rows-xs", "dora-cols-xs"):
            norm_dim = 1 if adapter_type == "dora-rows-xs" else 0
            V_hat = Vfull / (np.linalg.norm(Vfull, axis=norm_dim, keepdims=True) + 1e-12)
            return V_hat * _mag_2d(p["magnitude"], norm_dim)
        if adapter_type == "bora-xs":
            V_r = Vfull / (np.linalg.norm(Vfull, axis=1, keepdims=True) + 1e-12)
            inter = p["magnitude_r"].reshape(-1, 1) * V_r
            H_c = inter / (np.linalg.norm(inter, axis=0, keepdims=True) + 1e-12)
            return H_c * p["magnitude_c"].reshape(1, -1)

    raise LoraError(f"unknown adapter_type {adapter_type!r}")


def check_shapes(layer: str, W0: np.ndarray, p: dict, adapter_type: str) -> None:
    """Clear error when the adapter doesn't fit the base weight — almost always a
    wrong-base-model mismatch. Without it the failure is a raw numpy broadcast."""
    fan_out, fan_in = W0.shape
    if adapter_type.endswith("-xs"):
        rank = p["M_xs"].shape[0]
        if rank > min(fan_out, fan_in):
            raise LoraError(
                f"{layer}: LoRA-XS rank {rank} exceeds base min-dim "
                f"{min(fan_out, fan_in)} for weight {W0.shape} — wrong base model?"
            )
        return
    b_out, b_rank = p["lora_B"].shape
    a_rank, a_in = p["lora_A"].shape
    if b_out != fan_out or a_in != fan_in or a_rank != b_rank:
        raise LoraError(
            f"{layer}: adapter lora_B{p['lora_B'].shape}·lora_A{p['lora_A'].shape} "
            f"does not fit base weight {W0.shape} — wrong base model?"
        )


# ── checkpoint parsing → normalized per-layer adapter ──────────────────────────

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


def parse_adapter(path: str):
    """Load one adapter → ``(adapter_type, scaling, layers)`` where ``layers``
    maps a checkpoint layer name → its param dict (numpy float32)."""
    tensors, meta = _load_safetensors(path)

    native_marker = ".parametrizations.weight.0."
    if any(native_marker in k for k in tensors):
        cfg = json.loads(meta.get("lora_config", "{}")) if meta else {}
        layers = _group_native(tensors)
        rank = int(cfg.get("rank") or _infer_rank(layers))
        alpha = float(cfg.get("alpha", rank))
        adapter_type = _resolve_native_type(cfg.get("adapter_type", "lora"))
        return adapter_type, alpha / rank, layers

    peft_marker = ".lora_A.weight"
    if any(k.endswith(peft_marker) for k in tensors):
        cfg = _read_peft_config(path)
        rank = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha", rank))
        use_dora = bool(cfg.get("use_dora", False))
        use_rslora = bool(cfg.get("use_rslora", False))
        adapter_type = "dora-rows" if use_dora else "lora"
        scaling = alpha / (np.sqrt(rank) if use_rslora else rank)
        return adapter_type, scaling, _group_peft(tensors)

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
    cfg_path = os.path.join(os.path.dirname(path), "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise LoraError(f"PEFT adapter at {path!r} is missing its adapter_config.json sibling")
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
    """Legacy 'dora' → 'dora-rows' (mirrors utils.resolve_adapter_type)."""
    return "dora-rows" if adapter_type == "dora" else adapter_type


# ── CLI spec parsing ────────────────────────────────────────────────────────

def parse_lora_spec(tokens, default_strength: float = 1.0) -> dict:
    """Parse one ``--lora`` group: ``PATH [strength=S]``.

    Mirrors the MLX CLI's flag, minus ``steps=`` — per-step LoRA gating is not
    feasible on a frozen TFLite graph (weights are packed into XNNPACK's private
    buffers at allocate_tensors; there is no per-step re-merge), so ``steps=`` is
    rejected with a pointer to the MLX backend.
    """
    if not tokens:
        raise LoraError("--lora needs an adapter path")
    spec = {"path": tokens[0], "strength": float(default_strength)}
    for tok in tokens[1:]:
        key, eq, val = tok.partition("=")
        if key == "steps":
            raise LoraError(
                "--lora steps=… (per-step gating) is not supported on the TFLite "
                "backend — the frozen graph's weights are merged once at load. "
                "Use the MLX backend (optimized/mlx) for step-gated LoRA."
            )
        if not eq or key != "strength":
            raise LoraError(
                f"--lora: unexpected token {tok!r} — one adapter per --lora flag; "
                f"the only option after the path is strength=S "
                f"(e.g. --lora a.safetensors strength=0.8)"
            )
        try:
            spec["strength"] = float(val)
        except ValueError:
            raise LoraError(f"--lora: bad strength {val!r}") from None
    return spec
