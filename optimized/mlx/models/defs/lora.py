"""MLX LoRA-family adapters compatible with Stable Audio 3 checkpoints.

This module intentionally depends only on MLX and NumPy so it can be used by
the standalone optimized MLX runtime without pulling in PyTorch or safetensors.
"""

from __future__ import annotations

import json
import math
import re
import typing as tp
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+)\.parametrizations\.weight\.(?P<index>\d+)\."
    r"(?P<param>lora_A|lora_B|M_xs|magnitude|magnitude_r|magnitude_c|U|V)$"
)
_XS_ADAPTER_TYPES = {
    "lora-xs",
    "dora-rows-xs",
    "dora-cols-xs",
    "bora-xs",
}
_SUPPORTED_ADAPTER_TYPES = {
    "lora",
    "dora-rows",
    "dora-cols",
    "bora",
    *_XS_ADAPTER_TYPES,
}
_FULL_WEIGHT_ADAPTER_TYPES = _SUPPORTED_ADAPTER_TYPES - {"lora"}


@dataclass(frozen=True)
class LoRAInjectionReport:
    layer_names: tuple[str, ...]
    trainable_parameters: int
    adapter_type: str

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)


@dataclass(frozen=True)
class LoRAApplyReport:
    path: str
    adapter_type: str
    loaded_layers: int
    applied_layers: int
    missing_targets: tuple[str, ...] = ()
    skipped_layers: tuple[str, ...] = ()


class LoRALinear(nn.Module):
    """Trainable LoRA-family wrapper for an MLX Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.checkpoint_name = _checkpoint_layer_name(self.source_name)
        self.adapter_type = canonical_adapter_type(adapter_type)

        fan_out, fan_in = (int(value) for value in base.weight.shape)
        source_weight = _linear_source_weight_2d(base.weight)
        _validate_rank(
            self.rank,
            fan_out=fan_out,
            fan_in=fan_in,
            source_name=self.source_name,
        )
        _initialize_adapter(self, source_weight, fan_out=fan_out, fan_in=fan_in)

    def __call__(self, x):
        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            adapted_weight = _adapted_weight_2d(
                _linear_source_weight_2d(self.base.weight),
                adapter_type=self.adapter_type,
                layer=self,
            )
            output = x.astype(mx.float32) @ adapted_weight.T
            bias = getattr(self.base, "bias", None)
            if bias is not None:
                output = output + bias.astype(mx.float32)
            return output.astype(x.dtype)

        base_output = self.base(x)
        adapter_output = (x.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
        return base_output + (adapter_output * self.scaling).astype(base_output.dtype)


class LoRAConv1d(nn.Module):
    """Trainable LoRA-family wrapper for an MLX Conv1d layer."""

    def __init__(
        self,
        base: nn.Conv1d,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.checkpoint_name = _checkpoint_layer_name(self.source_name)
        self.adapter_type = canonical_adapter_type(adapter_type)

        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in base.weight.shape
        )
        fan_in = fan_in_per_group * kernel_size
        source_weight = _conv1d_source_weight_2d(base.weight)
        _validate_rank(
            self.rank,
            fan_out=fan_out,
            fan_in=fan_in,
            source_name=self.source_name,
        )
        _initialize_adapter(self, source_weight, fan_out=fan_out, fan_in=fan_in)

    def __call__(self, x):
        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in self.base.weight.shape
        )

        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            adapted_source = _adapted_weight_2d(
                _conv1d_source_weight_2d(self.base.weight),
                adapter_type=self.adapter_type,
                layer=self,
            )
            adapted_weight = _conv1d_weight_from_source_2d(
                adapted_source,
                fan_out=fan_out,
                fan_in_per_group=fan_in_per_group,
                kernel_size=kernel_size,
            )
            output = mx.conv1d(
                x.astype(mx.float32),
                adapted_weight,
                self.base.stride,
                self.base.padding,
                self.base.dilation,
                self.base.groups,
            )
            bias = getattr(self.base, "bias", None)
            if bias is not None:
                output = output + bias.astype(mx.float32)
            return output.astype(x.dtype)

        base_output = self.base(x)
        delta_weight = _conv1d_weight_from_source_2d(
            self.lora_B @ self.lora_A,
            fan_out=fan_out,
            fan_in_per_group=fan_in_per_group,
            kernel_size=kernel_size,
        )
        adapter_output = mx.conv1d(
            x.astype(mx.float32),
            delta_weight,
            self.base.stride,
            self.base.padding,
            self.base.dilation,
            self.base.groups,
        )
        return base_output + (adapter_output * self.scaling).astype(base_output.dtype)


TrainableLoRALayer = LoRALinear | LoRAConv1d


def inject_trainable_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    adapter_type: str = "lora",
) -> LoRAInjectionReport:
    """Freeze an MLX model and replace selected Linear/Conv1d layers."""

    alpha = float(rank if alpha is None else alpha)
    adapter_type = canonical_adapter_type(adapter_type)
    model.freeze()

    replacements: list[tuple[str, TrainableLoRALayer]] = []
    for name, layer in model.named_modules():
        if not name or not _name_is_selected(name, include=include, exclude=exclude):
            continue
        if isinstance(layer, nn.Linear):
            replacement = LoRALinear(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
            )
        elif isinstance(layer, nn.Conv1d):
            replacement = LoRAConv1d(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
            )
        else:
            continue
        replacements.append((name, replacement))

    if not replacements:
        raise ValueError("No MLX Linear or Conv1d layers matched the LoRA filters.")

    model.update_modules(tree_unflatten(replacements))
    trainable_parameters = sum(
        int(value.size) for _, value in tree_flatten(model.trainable_parameters())
    )
    return LoRAInjectionReport(
        layer_names=tuple(name for name, _ in replacements),
        trainable_parameters=trainable_parameters,
        adapter_type=adapter_type,
    )


def iter_trainable_lora_layers(
    model: nn.Module,
) -> tp.Iterator[TrainableLoRALayer]:
    for _, layer in model.named_modules():
        if isinstance(layer, (LoRALinear, LoRAConv1d)):
            yield layer


def save_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    extra_config: dict[str, tp.Any] | None = None,
) -> Path:
    """Save trainable adapters using the official SA3 safetensors contract."""

    layers = list(iter_trainable_lora_layers(model))
    if not layers:
        raise ValueError("The model has no trainable MLX LoRA layers to save.")

    ranks = {layer.rank for layer in layers}
    alphas = {layer.alpha for layer in layers}
    adapter_types = {layer.adapter_type for layer in layers}
    if len(ranks) != 1 or len(alphas) != 1 or len(adapter_types) != 1:
        raise ValueError("A checkpoint must use one rank, alpha, and adapter type.")

    rank = next(iter(ranks))
    alpha = next(iter(alphas))
    adapter_type = next(iter(adapter_types))
    state_dict: dict[str, mx.array] = {}
    for layer in layers:
        prefix = f"{layer.checkpoint_name}.parametrizations.weight.0"
        if adapter_type in _XS_ADAPTER_TYPES:
            state_dict[f"{prefix}.M_xs"] = layer.M_xs.astype(mx.float16)
        else:
            state_dict[f"{prefix}.lora_A"] = layer.lora_A.astype(mx.float16)
            state_dict[f"{prefix}.lora_B"] = layer.lora_B.astype(mx.float16)

        if adapter_type in {
            "dora-rows",
            "dora-cols",
            "dora-rows-xs",
            "dora-cols-xs",
        }:
            state_dict[f"{prefix}.magnitude"] = layer.magnitude.astype(mx.float16)
        elif adapter_type in {"bora", "bora-xs"}:
            state_dict[f"{prefix}.magnitude_r"] = layer.magnitude_r.astype(mx.float16)
            state_dict[f"{prefix}.magnitude_c"] = layer.magnitude_c.astype(mx.float16)

    config: dict[str, tp.Any] = {
        "rank": rank,
        "alpha": alpha,
        "adapter_type": adapter_type,
        "include": list(include) if include else None,
        "exclude": list(exclude) if exclude else None,
    }
    if extra_config:
        protected = {"rank", "alpha", "adapter_type", "include", "exclude"}
        overlap = protected.intersection(extra_config)
        if overlap:
            raise ValueError(
                "extra_config cannot override checkpoint fields: "
                + ", ".join(sorted(overlap))
            )
        config.update(extra_config)

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output_path),
        state_dict,
        metadata={"lora_config": json.dumps(config)},
    )
    return output_path


def load_lora_checkpoint(
    path: str | Path,
) -> tuple[dict[str, mx.array], dict[str, tp.Any]]:
    """Load an SA3 LoRA safetensors checkpoint without PyTorch."""

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("The standalone MLX runtime supports .safetensors LoRAs.")
    state_dict, metadata = mx.load(str(checkpoint_path), return_metadata=True)
    config = {}
    if metadata and metadata.get("lora_config"):
        config = json.loads(metadata["lora_config"])
    return dict(state_dict), config


def apply_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strength: float = 1.0,
) -> LoRAApplyReport:
    """Materialize one checkpoint into a loaded MLX model at a fixed strength.

    Only the model's in-memory weights are changed; the base checkpoint on disk
    is untouched. The operation is not reversible because the original target
    weights are not retained. Reload the base model before applying a different
    strength instead of calling this function repeatedly on the same instance.
    """

    state_dict, config = load_lora_checkpoint(path)
    adapter_type = _adapter_type_from_state(
        config.get("adapter_type", "lora"),
        state_dict,
    )
    if adapter_type not in _SUPPORTED_ADAPTER_TYPES:
        raise ValueError(f"Unsupported MLX LoRA adapter type: {adapter_type!r}")

    grouped = _group_lora_state_dict(state_dict)
    target_params = dict(tree_flatten(model.parameters()))
    target_keys = tuple(target_params)
    missing_targets: list[str] = []
    skipped_layers: list[str] = []
    applied_layers = 0

    global_rank = int(config.get("rank") or _infer_global_rank(grouped) or 0)
    alpha_value = config.get("alpha", config.get("lora_alpha"))
    alpha = float(alpha_value if alpha_value is not None else (global_rank or 1))

    for source_name, params in grouped.items():
        target_key = _resolve_target_key(f"{source_name}.weight", target_keys)
        if target_key is None:
            missing_targets.append(source_name)
            continue
        try:
            adapted = _apply_checkpoint_layer(
                target_params[target_key],
                params,
                adapter_type=adapter_type,
                alpha=alpha,
                strength=float(strength),
            )
        except ValueError as exc:
            skipped_layers.append(f"{source_name}: {exc}")
            continue

        target_dtype = target_params[target_key].dtype
        updated = mx.array(adapted)
        if updated.dtype != target_dtype:
            updated = updated.astype(target_dtype)
        model.update(tree_unflatten([(target_key, updated)]))
        target_params[target_key] = updated
        applied_layers += 1

    if applied_layers:
        mx.eval(model.parameters())

    return LoRAApplyReport(
        path=str(Path(path).expanduser().resolve()),
        adapter_type=adapter_type,
        loaded_layers=len(grouped),
        applied_layers=applied_layers,
        missing_targets=tuple(sorted(set(missing_targets))),
        skipped_layers=tuple(skipped_layers),
    )


def apply_lora_checkpoints(
    model: nn.Module,
    paths: tp.Sequence[str | Path],
    *,
    strengths: float | tp.Sequence[float] = 1.0,
) -> tuple[LoRAApplyReport, ...]:
    """Materialize an ordered checkpoint stack into a loaded MLX model.

    This is a fixed-strength, in-place operation. In particular, DoRA and BoRA
    composition is order-dependent. Callers that need mutable strengths should
    retain canonical base weights and rebuild the complete ordered stack from
    those values rather than applying updates cumulatively.
    """

    if isinstance(strengths, (int, float)):
        values = [float(strengths)] * len(paths)
    else:
        values = [float(value) for value in strengths]
        if len(values) == 1:
            values *= len(paths)
        if len(values) != len(paths):
            raise ValueError(
                f"Expected 1 or {len(paths)} strengths, got {len(values)}."
            )

    return tuple(
        apply_lora_checkpoint(model, path, strength=strength)
        for path, strength in zip(paths, values, strict=True)
    )


def canonical_adapter_type(adapter_type: str) -> str:
    adapter_type = str(adapter_type or "lora").strip().lower()
    aliases = {
        "dora": "dora-rows",
        "dora-xs": "dora-rows-xs",
        "xs": "lora-xs",
    }
    adapter_type = aliases.get(adapter_type, adapter_type)
    if adapter_type not in _SUPPORTED_ADAPTER_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_ADAPTER_TYPES))
        raise ValueError(
            f"Unsupported MLX adapter type {adapter_type!r}. Expected one of: "
            f"{supported}."
        )
    return adapter_type


def _initialize_adapter(layer, source_weight, *, fan_out: int, fan_in: int) -> None:
    if layer.adapter_type in _XS_ADAPTER_TYPES:
        layer.U, layer.V = _svd_bases(source_weight, layer.rank)
        layer.M_xs = mx.zeros((layer.rank, layer.rank), dtype=mx.float32)
        layer.freeze(keys=["U", "V"], recurse=False)
    else:
        init_scale = 1.0 / math.sqrt(fan_in)
        layer.lora_A = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(layer.rank, fan_in),
            dtype=mx.float32,
        )
        layer.lora_B = mx.zeros((fan_out, layer.rank), dtype=mx.float32)

    if layer.adapter_type in {"dora-rows", "dora-rows-xs"}:
        layer.magnitude = _row_norms(source_weight)
    elif layer.adapter_type in {"dora-cols", "dora-cols-xs"}:
        layer.magnitude = _column_norms(source_weight)
    elif layer.adapter_type in {"bora", "bora-xs"}:
        layer.magnitude_r = _row_norms(source_weight)
        layer.magnitude_c = _column_norms(source_weight)


def _apply_checkpoint_layer(
    target_weight,
    params: dict[str, np.ndarray],
    *,
    adapter_type: str,
    alpha: float,
    strength: float,
) -> np.ndarray:
    target = np.asarray(target_weight, dtype=np.float32)
    if strength == 0:
        return target

    if adapter_type in _XS_ADAPTER_TYPES:
        source_shape = _source_shape_for_xs(tuple(target.shape), params)
    else:
        delta, _ = _lora_delta_2d(params)
        source_shape = _source_shape_for_delta(tuple(target.shape), delta.shape)

    source = _target_to_source_weight(target, source_shape)
    base_2d = source.reshape(source_shape[0], -1).astype(np.float32, copy=False)
    if adapter_type in _XS_ADAPTER_TYPES:
        delta, rank = _xs_delta_2d(params, base_2d)
    else:
        delta, rank = _lora_delta_2d(params)

    value = base_2d + (float(alpha) / rank) * strength * delta
    if adapter_type in {"lora", "lora-xs"}:
        adapted = value
    elif adapter_type in {"dora-rows", "dora-rows-xs"}:
        adapted = _dora_weight_2d(
            value,
            magnitude=_require_param(params, "magnitude").reshape(-1),
            norm_dim=1,
        )
    elif adapter_type in {"dora-cols", "dora-cols-xs"}:
        adapted = _dora_weight_2d(
            value,
            magnitude=_require_param(params, "magnitude").reshape(-1),
            norm_dim=0,
        )
    else:
        adapted = _bora_weight_2d(
            value,
            magnitude_r=_require_param(params, "magnitude_r").reshape(-1),
            magnitude_c=_require_param(params, "magnitude_c").reshape(-1),
        )

    return _source_to_target_weight(adapted.reshape(source_shape), target.shape)


def _adapted_weight_2d(weight_2d, *, adapter_type: str, layer):
    value = weight_2d.astype(mx.float32) + _adapter_delta_2d(layer) * float(
        layer.scaling
    )
    if adapter_type in {"lora", "lora-xs"}:
        return value
    if adapter_type in {"dora-rows", "dora-rows-xs"}:
        return _dora_weight_2d(value, magnitude=layer.magnitude, norm_dim=1)
    if adapter_type in {"dora-cols", "dora-cols-xs"}:
        return _dora_weight_2d(value, magnitude=layer.magnitude, norm_dim=0)
    return _bora_weight_2d(
        value,
        magnitude_r=layer.magnitude_r,
        magnitude_c=layer.magnitude_c,
    )


def _adapter_delta_2d(layer):
    if layer.adapter_type in _XS_ADAPTER_TYPES:
        return layer.U @ layer.M_xs.astype(mx.float32) @ layer.V.T
    return layer.lora_B @ layer.lora_A


def _linear_source_weight_2d(weight):
    return weight.astype(mx.float32)


def _conv1d_source_weight_2d(weight):
    fan_out, kernel_size, fan_in_per_group = (int(value) for value in weight.shape)
    return (
        weight.astype(mx.float32)
        .transpose(0, 2, 1)
        .reshape(
            fan_out,
            fan_in_per_group * kernel_size,
        )
    )


def _conv1d_weight_from_source_2d(
    source,
    *,
    fan_out: int,
    fan_in_per_group: int,
    kernel_size: int,
):
    return source.reshape(
        fan_out,
        fan_in_per_group,
        kernel_size,
    ).transpose(0, 2, 1)


def _row_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=1)).astype(mx.float32)


def _column_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=0)).astype(mx.float32)


def _dora_weight_2d(value, *, magnitude, norm_dim: int):
    if isinstance(value, np.ndarray):
        norms = np.linalg.norm(value, axis=norm_dim, keepdims=True)
        value_hat = value / np.maximum(norms, 1e-12)
        if norm_dim == 1:
            if magnitude.shape[0] != value.shape[0]:
                raise ValueError("DoRA row magnitude does not match the weight.")
            return value_hat * magnitude[:, None]
        if magnitude.shape[0] != value.shape[1]:
            raise ValueError("DoRA column magnitude does not match the weight.")
        return value_hat * magnitude[None, :]

    norms = mx.sqrt(mx.sum(value**2, axis=norm_dim, keepdims=True))
    value_hat = value / mx.maximum(norms, 1e-12)
    if norm_dim == 1:
        return value_hat * magnitude.astype(mx.float32)[:, None]
    return value_hat * magnitude.astype(mx.float32)[None, :]


def _bora_weight_2d(value, *, magnitude_r, magnitude_c):
    if isinstance(value, np.ndarray):
        if magnitude_r.shape[0] != value.shape[0]:
            raise ValueError("BoRA row magnitude does not match the weight.")
        if magnitude_c.shape[0] != value.shape[1]:
            raise ValueError("BoRA column magnitude does not match the weight.")
        row_norms = np.linalg.norm(value, axis=1, keepdims=True)
        row_scaled = value / np.maximum(row_norms, 1e-12)
        row_scaled *= magnitude_r[:, None]
        column_norms = np.linalg.norm(row_scaled, axis=0, keepdims=True)
        return (row_scaled / np.maximum(column_norms, 1e-12)) * magnitude_c[None, :]

    row_norms = mx.sqrt(mx.sum(value**2, axis=1, keepdims=True))
    row_scaled = (value / mx.maximum(row_norms, 1e-12)) * magnitude_r.astype(
        mx.float32
    )[:, None]
    column_norms = mx.sqrt(mx.sum(row_scaled**2, axis=0, keepdims=True))
    return (row_scaled / mx.maximum(column_norms, 1e-12)) * magnitude_c.astype(
        mx.float32
    )[None, :]


def _group_lora_state_dict(
    state_dict: dict[str, tp.Any],
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in state_dict.items():
        match = _LORA_KEY_RE.match(key)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {})[match.group("param")] = (
            np.asarray(value, dtype=np.float32)
        )
    return grouped


def _adapter_type_from_state(
    adapter_type: str,
    state_dict: dict[str, tp.Any],
) -> str:
    raw_type = str(adapter_type or "lora").strip().lower()
    keys = tuple(state_dict)
    has_xs = any(key.endswith(".M_xs") for key in keys)
    if has_xs:
        if raw_type in {"bora", "bora-xs"} or any(
            key.endswith((".magnitude_r", ".magnitude_c")) for key in keys
        ):
            return "bora-xs"
        if raw_type in {"dora-cols", "dora-cols-xs"}:
            return "dora-cols-xs"
        if raw_type in {"dora", "dora-rows", "dora-rows-xs"} or any(
            key.endswith(".magnitude") for key in keys
        ):
            return "dora-rows-xs"
        return "lora-xs"
    if raw_type == "lora":
        if any(key.endswith((".magnitude_r", ".magnitude_c")) for key in keys):
            return "bora"
        if any(key.endswith(".magnitude") for key in keys):
            return "dora-rows"
    return canonical_adapter_type(raw_type)


def _infer_global_rank(grouped: dict[str, dict[str, np.ndarray]]) -> int:
    ranks = {
        rank for params in grouped.values() if (rank := _rank_from_params(params)) > 0
    }
    if not ranks:
        return 0
    if len(ranks) > 1:
        raise ValueError(f"Multiple adapter ranks found: {sorted(ranks)}.")
    return next(iter(ranks))


def _rank_from_params(params: dict[str, np.ndarray]) -> int:
    core = params.get("M_xs")
    if core is not None and core.ndim == 2 and core.shape[0] == core.shape[1]:
        return int(core.shape[0])
    adapter_a = params.get("lora_A")
    adapter_b = params.get("lora_B")
    if adapter_a is None or adapter_b is None:
        return 0
    if adapter_b.shape[-1] == adapter_a.shape[0]:
        return int(adapter_a.shape[0])
    if adapter_a.shape[-1] == adapter_b.shape[0]:
        return int(adapter_a.shape[-1])
    return 0


def _lora_delta_2d(
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    adapter_a = _require_param(params, "lora_A").astype(np.float64)
    adapter_b = _require_param(params, "lora_B").astype(np.float64)
    if adapter_b.shape[-1] == adapter_a.shape[0]:
        delta = adapter_b @ adapter_a
        rank = adapter_a.shape[0]
    elif adapter_a.shape[-1] == adapter_b.shape[0]:
        delta = adapter_a @ adapter_b
        rank = adapter_a.shape[-1]
    else:
        raise ValueError(
            "Unable to multiply LoRA matrices with shapes "
            f"A={adapter_a.shape}, B={adapter_b.shape}."
        )
    if not np.isfinite(delta).all():
        raise ValueError("LoRA delta contains non-finite values.")
    return delta.astype(np.float32), int(rank)


def _xs_delta_2d(
    params: dict[str, np.ndarray],
    base_2d: np.ndarray,
) -> tuple[np.ndarray, int]:
    core = _require_param(params, "M_xs")
    if core.ndim != 2 or core.shape[0] != core.shape[1]:
        raise ValueError(f"LoRA-XS core must be square, got {core.shape}.")
    rank = int(core.shape[0])
    if rank > min(base_2d.shape):
        raise ValueError(
            f"LoRA-XS rank {rank} exceeds base weight shape {base_2d.shape}."
        )

    u = params.get("U")
    v = params.get("V")
    if u is None or v is None:
        u, v = _svd_bases_numpy(base_2d, rank)
    delta = u.astype(np.float64) @ core.astype(np.float64) @ v.astype(np.float64).T
    if not np.isfinite(delta).all():
        raise ValueError("LoRA-XS delta contains non-finite values.")
    return delta.astype(np.float32), rank


def _require_param(params: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = params.get(name)
    if value is None:
        raise ValueError(f"Adapter layer is missing {name}.")
    return value.astype(np.float32, copy=False)


def _resolve_target_key(
    source_weight_key: str,
    target_keys: tuple[str, ...],
) -> str | None:
    target_key_set = set(target_keys)
    candidates = _target_key_candidates(source_weight_key)
    for candidate in candidates:
        if candidate in target_key_set:
            return candidate

    suffix_matches = {
        target_key
        for candidate in candidates
        for target_key in target_keys
        if target_key.endswith(candidate)
    }
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches))
    return None


def _target_key_candidates(source_weight_key: str) -> tuple[str, ...]:
    prefixes = ("model.model.", "model.")
    candidates = [source_weight_key]
    for prefix in prefixes:
        if source_weight_key.startswith(prefix):
            candidates.append(source_weight_key[len(prefix) :])

    candidates.extend(
        candidate.replace("to_local_embed.0.", "to_local_embed.seq.0.").replace(
            "to_local_embed.2.", "to_local_embed.seq.2."
        )
        for candidate in tuple(candidates)
    )
    return tuple(dict.fromkeys(candidates))


def _checkpoint_layer_name(name: str) -> str:
    return name.replace("to_local_embed.seq.0", "to_local_embed.0").replace(
        "to_local_embed.seq.2", "to_local_embed.2"
    )


def _source_shape_for_delta(
    target_shape: tuple[int, ...],
    delta_shape: tuple[int, int],
) -> tuple[int, ...]:
    if len(target_shape) == 2:
        candidates = (target_shape, (target_shape[1], target_shape[0]))
    elif len(target_shape) == 3:
        candidates = (
            (target_shape[0], target_shape[2], target_shape[1]),
            target_shape,
        )
    else:
        candidates = (target_shape,)

    for candidate in candidates:
        if (
            candidate[0] == delta_shape[0]
            and int(np.prod(candidate[1:])) == delta_shape[1]
        ):
            return candidate
    raise ValueError(
        f"Unable to map LoRA delta {delta_shape} to target shape {target_shape}."
    )


def _source_shape_for_xs(
    target_shape: tuple[int, ...],
    params: dict[str, np.ndarray],
) -> tuple[int, ...]:
    u = params.get("U")
    v = params.get("V")
    if u is not None and v is not None:
        return _source_shape_for_delta(
            target_shape,
            (int(u.shape[0]), int(v.shape[0])),
        )
    if len(target_shape) == 3:
        return (target_shape[0], target_shape[2], target_shape[1])
    return target_shape


def _target_to_source_weight(
    target: np.ndarray,
    source_shape: tuple[int, ...],
) -> np.ndarray:
    if target.shape == source_shape:
        return target
    if target.ndim == 2 and target.T.shape == source_shape:
        return target.T
    if target.ndim == 3:
        candidate = target.transpose(0, 2, 1)
        if candidate.shape == source_shape:
            return candidate
    raise ValueError(
        f"Unable to map target shape {target.shape} to source shape {source_shape}."
    )


def _source_to_target_weight(
    source: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    if source.shape == target_shape:
        return source
    if source.ndim == 2 and source.T.shape == target_shape:
        return source.T
    if source.ndim == 3:
        candidate = source.transpose(0, 2, 1)
        if candidate.shape == target_shape:
            return candidate
    raise ValueError(
        f"Unable to map source shape {source.shape} to target shape {target_shape}."
    )


def _validate_rank(
    rank: int,
    *,
    fan_out: int,
    fan_in: int,
    source_name: str,
) -> None:
    max_rank = min(fan_out, fan_in)
    if rank > max_rank:
        raise ValueError(
            f"Adapter rank {rank} exceeds maximum rank {max_rank} for "
            f"{source_name!r} with shape ({fan_out}, {fan_in})."
        )


def _svd_bases(weight_2d, rank: int):
    u, v = _svd_bases_numpy(np.asarray(weight_2d, dtype=np.float32), rank)
    return mx.array(u, dtype=mx.float32), mx.array(v, dtype=mx.float32)


def _svd_bases_numpy(
    weight_2d: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    u, _, vh = np.linalg.svd(
        weight_2d.astype(np.float32, copy=False),
        full_matrices=False,
    )
    u, vh = _canonicalize_svd_signs(u, vh)
    return (
        u[:, :rank].astype(np.float32, copy=False),
        vh[:rank, :].T.astype(np.float32, copy=False),
    )


def _canonicalize_svd_signs(
    u: np.ndarray,
    vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_abs_indices = np.abs(u).argmax(axis=0)
    signs = np.sign(u[max_abs_indices, np.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs[None, :], vh * signs[:, None]


def _name_is_selected(
    name: str,
    *,
    include: tp.Sequence[str] | None,
    exclude: tp.Sequence[str] | None,
) -> bool:
    if include and not _matches_any(name, include):
        return False
    return not (exclude and _matches_any(name, exclude))


def _matches_any(name: str, patterns: tp.Sequence[str]) -> bool:
    return any(
        expanded in name for pattern in patterns for expanded in _expand(pattern)
    )


def _expand(pattern: str) -> list[str]:
    parts = re.split(r"\[(\d+)-(\d+)\]", pattern)
    if len(parts) == 1:
        return [pattern]

    literals = parts[0::3]
    starts = parts[1::3]
    ends = parts[2::3]
    ranges = []
    for start, end in zip(starts, ends, strict=True):
        start_value = int(start)
        end_value = int(end)
        step = 1 if end_value >= start_value else -1
        ranges.append(
            [str(value) for value in range(start_value, end_value + step, step)]
        )

    expanded = []
    for values in product(*ranges):
        pieces = []
        for index, literal in enumerate(literals):
            pieces.append(literal)
            if index < len(values):
                pieces.append(values[index])
        expanded.append("".join(pieces))
    return expanded
