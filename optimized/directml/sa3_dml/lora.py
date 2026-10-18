"""LoRA adapters merged into the DiT graph weights when its session is created.

The merge follows `LoRAParametrization` of the reference: every adapter type, the strength
applied inside the adapter (with the DoRA magnitude blend) and stacked adapters applied in
order, each on the weight produced by the previous one. The merged tensors replace the graph's
externally stored weights in memory, so the prepared graph on disk is never modified.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from .specs import checkpoint_key

ADAPTER_PARAMS = {
    "lora": ("lora_A", "lora_B"),
    "dora-rows": ("lora_A", "lora_B", "magnitude"),
    "dora-cols": ("lora_A", "lora_B", "magnitude"),
    "bora": ("lora_A", "lora_B", "magnitude_r", "magnitude_c"),
    "lora-xs": ("M_xs",),
    "dora-rows-xs": ("M_xs", "magnitude"),
    "dora-cols-xs": ("M_xs", "magnitude"),
    "bora-xs": ("M_xs", "magnitude_r", "magnitude_c"),
}
NATIVE_MARKER = ".parametrizations.weight.0."
PEFT_PREFIX = "base_model.model."
PEFT_PARAMS = {".lora_A.weight": "lora_A", ".lora_B.weight": "lora_B",
               ".lora_magnitude_vector.weight": "magnitude", ".lora_magnitude_vector": "magnitude"}
DIT_SCOPE = "model."
CONDITIONER_SCOPE = "conditioners."
_EPS = 1e-12
# NumPy's bundled OpenBLAS LAPACK crashes when eigh runs concurrently from several threads.
_LAPACK_LOCK = threading.Lock()


class LoraError(ValueError):
    pass


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    strength: float = 1.0


@dataclass
class Adapter:
    """One adapter checkpoint: layer name (relative to the DiT wrapper or the conditioner) -> tensors."""

    name: str
    adapter_type: str
    scaling: float
    layers: dict[str, dict[str, np.ndarray]]
    include: list[str] | None = None
    exclude: list[str] | None = None

    @classmethod
    def load(cls, path: Path | str) -> Adapter:
        path = _resolve(Path(path))
        tensors, metadata = read_safetensors(path)
        if any(NATIVE_MARKER in k for k in tensors):
            return cls._native(path, tensors, metadata)
        if any(k.endswith(".lora_A.weight") for k in tensors):
            return cls._peft(path, tensors)
        raise LoraError(f"{path.name}: no Stable Audio LoRA parametrization keys and no PEFT lora_A/lora_B keys")

    @classmethod
    def _native(cls, path: Path, tensors: dict, metadata: dict) -> Adapter:
        config = json.loads(metadata.get("lora_config") or "{}")
        layers: dict[str, dict[str, np.ndarray]] = {}
        for key, value in tensors.items():
            layer, marker, param = key.partition(NATIVE_MARKER)
            if marker and param in _ALL_PARAMS:
                layers.setdefault(layer, {})[param] = value
        adapter_type = config.get("adapter_type", "lora")
        if adapter_type == "dora":
            adapter_type = _dora_variant(layers)
        rank = int(config.get("rank") or _infer_rank(layers))
        alpha = float(config.get("alpha", rank))
        return cls(path.stem, adapter_type, alpha / rank, layers, config.get("include"), config.get("exclude"))

    @classmethod
    def _peft(cls, path: Path, tensors: dict) -> Adapter:
        config_path = path.parent / "adapter_config.json"
        if not config_path.is_file():
            raise LoraError(f"PEFT adapter {path} has no adapter_config.json beside it")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("rank_pattern") or config.get("alpha_pattern") or config.get("modules_to_save"):
            raise LoraError(f"{path.name}: PEFT rank_pattern, alpha_pattern and modules_to_save are not supported")
        layers: dict[str, dict[str, np.ndarray]] = {}
        for key, value in tensors.items():
            name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
            for suffix, param in PEFT_PARAMS.items():
                if name.endswith(suffix):
                    layers.setdefault(DIT_SCOPE + name[:-len(suffix)], {})[param] = value
                    break
        rank = int(config["r"])
        alpha = float(config.get("lora_alpha", rank))
        scaling = alpha / (np.sqrt(rank) if config.get("use_rslora") else rank)
        adapter_type = "dora-rows" if config.get("use_dora") else "lora"
        return cls(path.parent.name if path.name == "adapter_model.safetensors" else path.stem,
                   adapter_type, float(scaling), layers)

    def applies_to(self, layer: str) -> bool:
        if self.include is not None and not _matches_any(layer, self.include):
            return False
        return not (self.exclude is not None and _matches_any(layer, self.exclude))


_ALL_PARAMS = {p for params in ADAPTER_PARAMS.values() for p in params}
_SAFETENSORS_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Floating-point tensors of a .safetensors file as float32 (bfloat16 included) and its metadata."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    size = int(raw[:8].view("<u8")[0])
    header = json.loads(bytes(raw[8:8 + size]))
    metadata = header.pop("__metadata__", None) or {}
    tensors = {}
    for name, info in header.items():
        dtype = _SAFETENSORS_DTYPES.get(info["dtype"])
        if dtype is None:
            continue
        start, end = (8 + size + o for o in info["data_offsets"])
        data = np.frombuffer(raw[start:end], dtype=dtype)
        if info["dtype"] == "BF16":
            data = (data.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = data.astype(np.float32).reshape(info["shape"])
    return tensors, metadata


def _resolve(path: Path) -> Path:
    if path.is_dir():
        inner = path / "adapter_model.safetensors"
        if inner.is_file():
            return inner
        found = sorted(path.glob("*.safetensors"))
        if len(found) != 1:
            raise LoraError(f"{path}: expected one .safetensors adapter, found {len(found)}")
        return found[0]
    if path.suffix.lower() != ".safetensors":
        raise LoraError(f"{path}: only .safetensors adapters are supported")
    if not path.is_file():
        raise LoraError(f"{path} not found")
    return path


def _infer_rank(layers: dict) -> int:
    for params in layers.values():
        if "lora_A" in params:
            return int(params["lora_A"].shape[0])
        if "M_xs" in params:
            return int(params["M_xs"].shape[0])
    raise LoraError("cannot infer the LoRA rank (no lora_A or M_xs tensors)")


def _dora_variant(layers: dict) -> str:
    """Legacy "dora": a [1, fan_in] magnitude is the column variant, anything else the row one."""
    for params in layers.values():
        mag = params.get("magnitude")
        if mag is not None and mag.ndim == 2 and mag.shape[0] == 1:
            return "dora-cols"
        if mag is not None:
            return "dora-rows"
    return "dora-rows"


def _expand(pattern: str) -> list[str]:
    parts = re.split(r"\[(\d+)-(\d+)\]", pattern)
    if len(parts) == 1:
        return [pattern]
    literals, starts, ends = parts[0::3], parts[1::3], parts[2::3]
    pools = []
    for a, b in zip(starts, ends):
        lo, hi = int(a), int(b)
        step = 1 if hi >= lo else -1
        pools.append([str(n) for n in range(lo, hi + step, step)])
    out = []
    for combo in product(*pools):
        pieces = []
        for i, literal in enumerate(literals):
            pieces.append(literal)
            if i < len(combo):
                pieces.append(combo[i])
        out.append("".join(pieces))
    return out


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(expanded in name for pattern in patterns for expanded in _expand(pattern.strip()))


def checkpoint_weight(layer: str) -> str | None:
    """Full checkpoint key of the weight a LoRA layer adapts."""
    if layer.startswith(DIT_SCOPE):
        return f"model.{layer}.weight"
    if layer.startswith(CONDITIONER_SCOPE):
        return f"conditioner.{layer}.weight"
    return None


def svd_bases(w0: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Leading `rank` left/right singular vectors of the base weight, with the reference sign convention.

    Computed in float64 from the eigenvectors of the smaller Gram matrix, which yields the
    leading singular pairs of a full SVD at a fraction of its cost.
    """
    w = w0.astype(np.float64)
    wide = w.shape[0] <= w.shape[1]
    gram = w @ w.T if wide else w.T @ w
    with _LAPACK_LOCK:
        values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:rank]
    sigma = np.sqrt(np.maximum(values[order], np.finfo(np.float64).tiny))
    if wide:
        u = vectors[:, order]
        v = (w.T @ u) / sigma
    else:
        v = vectors[:, order]
        u = (w @ v) / sigma
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    return (u * signs).astype(np.float32), (v * signs).astype(np.float32)


def _norm(x: np.ndarray, axis: int) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=True)


def _blend(magnitude: np.ndarray, base: np.ndarray, strength: float) -> np.ndarray:
    return magnitude if strength == 1.0 else base + strength * (magnitude - base)


def adapt(w: np.ndarray, w0: np.ndarray, params: dict[str, np.ndarray], adapter_type: str, scaling: float,
          strength: float) -> np.ndarray:
    """One `LoRAParametrization` forward on a 2-D weight `w`; `w0` is the unadapted weight.

    Returns float32; `w` and `w0` may be float16 graph weights.
    """
    if adapter_type not in ADAPTER_PARAMS:
        raise LoraError(f"unknown adapter type {adapter_type!r}")
    if adapter_type.endswith("-xs"):
        m = params["M_xs"]
        u, v = svd_bases(w0, m.shape[0])
        left, right = u @ m, v.T
    else:
        left, right = params["lora_B"], params["lora_A"]
    if strength == 0 and adapter_type not in ("lora", "lora-xs"):
        return w
    adapted = (left * np.float32(scaling * strength)) @ right
    adapted += w
    if adapter_type in ("lora", "lora-xs"):
        return adapted
    w = w.astype(np.float32, copy=False)
    if adapter_type.startswith("dora"):
        axis = 1 if adapter_type.startswith("dora-rows") else 0
        magnitude = np.ravel(params["magnitude"])
        magnitude = magnitude[:, None] if axis == 1 else magnitude[None, :]
        return adapted / (_norm(adapted, axis) + _EPS) * _blend(magnitude, _norm(w, axis), strength)
    rows = adapted / (_norm(adapted, 1) + _EPS)
    rows = _blend(np.ravel(params["magnitude_r"])[:, None], _norm(w, 1), strength) * rows
    return rows / (_norm(rows, 0) + _EPS) * _blend(np.ravel(params["magnitude_c"])[None, :], _norm(w, 0), strength)


def _check(layer: str, w0: np.ndarray, params: dict, adapter_type: str) -> None:
    missing = [p for p in ADAPTER_PARAMS[adapter_type] if p not in params]
    if missing:
        raise LoraError(f"{layer}: {adapter_type} adapter without {missing}")
    fan_out, fan_in = w0.shape
    if adapter_type.endswith("-xs"):
        if params["M_xs"].shape[0] > min(fan_out, fan_in):
            raise LoraError(f"{layer}: rank {params['M_xs'].shape[0]} exceeds weight {w0.shape}; wrong base model?")
        return
    b, a = params["lora_B"], params["lora_A"]
    if b.shape[0] != fan_out or a.shape[1] != fan_in or a.shape[0] != b.shape[1]:
        raise LoraError(f"{layer}: lora_B{b.shape} x lora_A{a.shape} does not fit weight {w0.shape}; "
                        "wrong base model?")


class GraphWeights:
    """Reads the externally stored initializers of a prepared graph without loading the graph data."""

    def __init__(self, graph: Path):
        self.graph = Path(graph)
        model = onnx.load(str(self.graph), load_external_data=False)
        self.by_checkpoint = {}
        self.inits = {}
        for init in model.graph.initializer:
            key = checkpoint_key(init.name)
            if key is not None:
                self.by_checkpoint[key] = init.name
                self.inits[init.name] = init

    def read(self, name: str) -> np.ndarray:
        init = self.inits[name]
        if init.data_location != onnx.TensorProto.EXTERNAL:
            raise LoraError(f"graph weight {name} is not stored externally and cannot be replaced")
        info = {e.key: e.value for e in init.external_data}
        dtype = helper.tensor_dtype_to_np_dtype(init.data_type)
        count = int(np.prod(init.dims))
        data = np.fromfile(self.graph.parent / info["location"], dtype=dtype, count=count,
                           offset=int(info.get("offset", 0)))
        return data.reshape(tuple(init.dims))


@dataclass
class MergeReport:
    adapter: str
    adapter_type: str
    scaling: float
    strength: float
    layers: int
    skipped: int


def merge(graph: Path, adapters: list[tuple[Adapter, float]],
          log: Callable[[str], None] | None = None) -> tuple[dict[str, np.ndarray], list[MergeReport]]:
    """Merged DiT weights {graph initializer name: array in the graph dtype} for stacked adapters."""
    weights = GraphWeights(graph)
    plan: dict[str, list[tuple[int, str, dict]]] = {}
    reports = []
    for index, (adapter, strength) in enumerate(adapters):
        matched = skipped = 0
        for layer, params in adapter.layers.items():
            target = weights.by_checkpoint.get(checkpoint_weight(layer) or "")
            if target is None or not adapter.applies_to(layer):
                skipped += 1
                continue
            plan.setdefault(target, []).append((index, layer, params))
            matched += 1
        if not matched:
            raise LoraError(f"{adapter.name}: no adapter layer matches this DiT; wrong base model?")
        reports.append(MergeReport(adapter.name, adapter.adapter_type, adapter.scaling, strength, matched, skipped))
        if log is not None:
            log(f"lora {adapter.name}: {adapter.adapter_type}, scaling {adapter.scaling:g}, strength {strength:g}, "
                f"{matched} layers" + (f" ({skipped} without a DiT target ignored)" if skipped else ""))

    def merge_one(target: str) -> np.ndarray:
        base = weights.read(target)
        w0 = base.reshape(base.shape[0], -1)
        w = w0
        for index, layer, params in plan[target]:
            adapter, strength = adapters[index]
            _check(layer, w0, params, adapter.adapter_type)
            w = adapt(w, w0, params, adapter.adapter_type, adapter.scaling, strength)
        out = w.reshape(base.shape).astype(base.dtype)
        if not np.isfinite(out).all():
            raise LoraError(f"{target}: the merged weight overflows {base.dtype}")
        return out

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        merged = dict(zip(plan, pool.map(merge_one, plan)))
    return merged, reports


def identity(specs: list[LoraSpec]) -> tuple:
    """Hashable identity of a LoRA configuration, for session and cache reuse."""
    out = []
    for spec in specs:
        path = _resolve(Path(spec.path))
        stat = path.stat()
        out.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, float(spec.strength)))
    return tuple(out)
