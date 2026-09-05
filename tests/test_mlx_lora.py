import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from optimized.mlx.models.defs import dit_mlx, dit_mlx_medium
from optimized.mlx.models.defs import lora as lora_module
from optimized.mlx.models.defs.lora import (
    TrainableSecondsEmbedder,
    apply_lora_checkpoint,
    apply_lora_checkpoints,
    inject_from_lora_config,
    inject_trainable_lora,
    iter_trainable_lora_layers,
    load_lora_checkpoint,
    load_trainable_lora_state,
    save_lora_checkpoint,
    underfit_lora_config,
)
from optimized.mlx.models.defs.sa3_pipeline import SecondsTotalEmbedder
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    get_lora_state_dict,
    load_lora_checkpoint as load_torch_lora_checkpoint,
    save_lora_safetensors,
)

# A real underfit-trained checkpoint (medium DiT, dora-rows rank 16) used to
# verify byte-convention key-naming parity with underfit's saver. Opt-in:
# point $SA3_TEST_LORA_ADAPTER at a .safetensors adapter to run this check.
_LORA_ADAPTER_ENV = os.environ.get("SA3_TEST_LORA_ADAPTER", "")
UNDERFIT_REFERENCE_CHECKPOINT = Path(_LORA_ADAPTER_ENV or "nonexistent-adapter.safetensors")
needs_underfit_reference = pytest.mark.skipif(
    not (_LORA_ADAPTER_ENV and UNDERFIT_REFERENCE_CHECKPOINT.exists()),
    reason="real underfit reference checkpoint not available (set $SA3_TEST_LORA_ADAPTER)",
)


class TinyMLXLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2, bias=False)
        self.layer.weight = mx.array(
            [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]],
            dtype=mx.float32,
        )

    def __call__(self, x):
        return self.layer(x)


class TinyMLXRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 4, bias=False)
        self.output = nn.Linear(4, 2, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)

    def __call__(self, x):
        return self.output(nn.silu(self.input(x)))


class TinyMLXConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv1d(2, 3, kernel_size=3, bias=False)
        source_weight = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 20
        self.layer.weight = mx.array(source_weight.transpose(0, 2, 1))


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (TinyMLXLinear(), mx.ones((1, 3))),
        (TinyMLXConv1d(), mx.ones((1, 5, 2))),
    ],
)
def test_trainable_dora_supports_bias_free_mlx_layers(model, inputs):
    inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="dora",
    )

    output = model.layer(inputs)
    mx.eval(output)

    assert bool(mx.all(mx.isfinite(output)))


@pytest.mark.parametrize(
    ("model_factory", "minimum_layer_count"),
    [
        (dit_mlx.DiT, 190),
        (dit_mlx_medium.DiT, 220),
    ],
)
def test_trainable_lora_targets_optimized_small_and_medium_dits(
    model_factory,
    minimum_layer_count: int,
):
    model = model_factory(T_lat=8)
    target_names = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.Linear, nn.Conv1d))
    ]
    include = [
        "transformer.project_in",
        "preprocess_conv",
        "transformer.layers.0.to_local_embed.seq.0",
    ]

    assert len(target_names) >= minimum_layer_count
    assert all(name in target_names for name in include)

    report = inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="lora",
        include=include,
    )

    assert set(report.layer_names) == set(include)
    assert report.adapter_type == "lora"


class TinyTorchLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2, bias=False)


class TinyTorchConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv1d(2, 3, kernel_size=3, bias=False)


def test_trainable_lora_updates_only_adapter_parameters():
    mx.random.seed(7)
    model = TinyMLXRegressor()
    report = inject_trainable_lora(
        model,
        rank=2,
        alpha=2,
        include=["output"],
    )
    base_before = mx.array(model.output.base.weight)
    inputs = mx.array(
        [[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]],
        dtype=mx.float32,
    )
    target = mx.array(
        [[0.5, -1.0], [-0.25, 0.75]],
        dtype=mx.float32,
    )

    def loss_fn(local_model, values, expected):
        return mx.mean((local_model(values) - expected) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.1)
    initial_loss = float(loss_fn(model, inputs, target))
    for _ in range(40):
        loss, grads = loss_and_grad(model, inputs, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    assert report.layer_names == ("output",)
    assert report.trainable_parameters == 12
    assert [name for name, _ in tree_flatten(model.trainable_parameters())] == [
        "output.lora_A",
        "output.lora_B",
    ]
    assert mx.array_equal(model.output.base.weight, base_before)
    assert float(loss_fn(model, inputs, target)) < initial_loss * 0.1


def test_mlx_checkpoint_round_trips_through_official_torch_loader(
    tmp_path: Path,
):
    model = TinyMLXLinear()
    inject_trainable_lora(model, rank=1, alpha=1, adapter_type="dora")
    model.layer.lora_A = mx.array([[0.25, -0.5, 1.0]])
    model.layer.lora_B = mx.array([[0.5], [-0.25]])
    model.layer.magnitude = mx.array([3.0, 2.0])

    checkpoint = save_lora_checkpoint(
        model,
        tmp_path / "mlx-dora.safetensors",
        extra_config={"step": 20, "epoch": 3, "base_model": "sa3-medium"},
    )
    mlx_state, mlx_config = load_lora_checkpoint(checkpoint)
    torch_state, torch_config = load_torch_lora_checkpoint(checkpoint)

    # Underfit key convention: "model." + the bare runtime name. Underfit
    # loads with strict=False into model.model/model.conditioner, so the
    # model.-prefixed keys are what the official stack expects.
    assert sorted(mlx_state) == [
        "model.layer.parametrizations.weight.0.lora_A",
        "model.layer.parametrizations.weight.0.lora_B",
        "model.layer.parametrizations.weight.0.magnitude",
    ]
    assert sorted(mlx_state) == sorted(torch_state)
    assert mlx_config == torch_config
    assert mlx_config["adapter_type"] == "dora-rows"
    assert mlx_config["step"] == 20
    assert mlx_config["epoch"] == 3
    assert mlx_config["base_model"] == "sa3-medium"


@pytest.mark.parametrize(
    "adapter_type",
    [
        "lora",
        "dora-rows",
        "dora-cols",
        "bora",
        "lora-xs",
        "dora-rows-xs",
        "dora-cols-xs",
        "bora-xs",
    ],
)
def test_mlx_inference_matches_official_torch_adapter_math(
    tmp_path: Path,
    adapter_type: str,
):
    base_weight = torch.tensor(
        [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]],
        dtype=torch.float32,
    )
    torch_model = TinyTorchLinear()
    torch_model.layer.weight.data.copy_(base_weight)
    config = {
        torch.nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=1,
                lora_alpha=1,
                adapter_type=adapter_type,
            )
        }
    }
    add_lora(torch_model, config)
    adapter = torch_model.layer.parametrizations.weight[0]
    if adapter_type.endswith("-xs"):
        adapter.M_xs.data.fill_(0.5)
    else:
        adapter.lora_A.data.copy_(torch.tensor([[0.25, -0.5, 1.0]]))
        adapter.lora_B.data.copy_(torch.tensor([[0.5], [-0.25]]))

    if adapter_type in {"dora-rows", "dora-rows-xs"}:
        adapter.magnitude.data.copy_(torch.tensor([3.0, 2.0]))
    elif adapter_type in {"dora-cols", "dora-cols-xs"}:
        adapter.magnitude.data.copy_(torch.tensor([1.5, 2.5, 3.5]))
    elif adapter_type in {"bora", "bora-xs"}:
        adapter.magnitude_r.data.copy_(torch.tensor([3.0, 2.0]))
        adapter.magnitude_c.data.copy_(torch.tensor([1.5, 2.5, 3.5]))

    checkpoint = tmp_path / f"{adapter_type}.safetensors"
    save_lora_safetensors(
        get_lora_state_dict(torch_model),
        {"rank": 1, "alpha": 1, "adapter_type": adapter_type},
        checkpoint,
    )

    mlx_model = TinyMLXLinear()
    report = apply_lora_checkpoint(mlx_model, checkpoint)
    expected = torch_model.layer.weight.detach().numpy()

    assert report.adapter_type == adapter_type
    assert report.applied_layers == 1
    assert report.missing_targets == ()
    assert report.skipped_layers == ()
    assert np.allclose(np.asarray(mlx_model.layer.weight), expected, atol=2e-3)


def test_multiple_lora_checkpoints_apply_with_independent_strengths(
    tmp_path: Path,
):
    base_weight = np.array(
        [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]],
        dtype=np.float32,
    )
    checkpoints = []
    deltas = []
    for index, (lora_a, lora_b) in enumerate(
        (
            (
                [[0.25, -0.5, 1.0]],
                [[0.5], [-0.25]],
            ),
            (
                [[-0.75, 0.5, 0.25]],
                [[0.2], [0.4]],
            ),
        )
    ):
        checkpoint = tmp_path / f"lora-{index}.safetensors"
        mx.save_safetensors(
            str(checkpoint),
            {
                "layer.parametrizations.weight.0.lora_A": mx.array(lora_a),
                "layer.parametrizations.weight.0.lora_B": mx.array(lora_b),
            },
            metadata={"lora_config": '{"rank": 1, "alpha": 1, "adapter_type": "lora"}'},
        )
        checkpoints.append(checkpoint)
        deltas.append(np.asarray(lora_b, dtype=np.float32) @ np.asarray(lora_a))

    model = TinyMLXLinear()
    strengths = (0.25, 0.75)
    reports = apply_lora_checkpoints(
        model,
        checkpoints,
        strengths=strengths,
    )
    expected = base_weight + strengths[0] * deltas[0] + strengths[1] * deltas[1]

    assert [report.applied_layers for report in reports] == [1, 1]
    assert np.allclose(np.asarray(model.layer.weight), expected, atol=2e-3)


def test_checkpoint_names_map_to_optimized_local_embed_layout(tmp_path: Path):
    class LocalEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_local_embed = type("LocalEmbedSeq", (nn.Module,), {})()
            self.to_local_embed.seq = [
                nn.Linear(3, 2, bias=False),
                None,
                nn.Linear(2, 2, bias=False),
            ]

    model = LocalEmbed()
    original = np.asarray(model.to_local_embed.seq[0].weight).copy()
    checkpoint = tmp_path / "local-embed.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "to_local_embed.0.parametrizations.weight.0.lora_A": mx.array(
                [[1.0, 0.0, 0.0]]
            ),
            "to_local_embed.0.parametrizations.weight.0.lora_B": mx.array(
                [[0.5], [-0.25]]
            ),
        },
        metadata={"lora_config": ('{"rank": 1, "alpha": 1, "adapter_type": "lora"}')},
    )

    report = apply_lora_checkpoint(model, checkpoint)

    assert report.applied_layers == 1
    assert not np.array_equal(
        np.asarray(model.to_local_embed.seq[0].weight),
        original,
    )


def test_saved_local_embed_name_maps_back_to_pytorch_layout(tmp_path: Path):
    class LocalEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_local_embed = type("LocalEmbedSeq", (nn.Module,), {})()
            self.to_local_embed.seq = [
                nn.Linear(3, 2, bias=False),
                None,
                nn.Linear(2, 2, bias=False),
            ]

    model = LocalEmbed()
    inject_trainable_lora(
        model,
        rank=1,
        include=["to_local_embed.seq.0"],
    )
    checkpoint = save_lora_checkpoint(
        model,
        tmp_path / "local-embed-save.safetensors",
    )
    state_dict, _ = load_torch_lora_checkpoint(checkpoint)

    assert sorted(state_dict) == [
        "model.to_local_embed.0.parametrizations.weight.0.lora_A",
        "model.to_local_embed.0.parametrizations.weight.0.lora_B",
    ]


def test_conv1d_checkpoint_maps_pytorch_weight_layout_to_mlx(tmp_path: Path):
    torch_model = TinyTorchConv1d()
    torch_model.layer.weight.data.copy_(
        torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 20
    )
    config = {
        torch.nn.Conv1d: {
            "weight": partial(
                LoRAParametrization.from_conv1d,
                rank=1,
                lora_alpha=1,
                adapter_type="lora",
            )
        }
    }
    add_lora(torch_model, config)
    adapter = torch_model.layer.parametrizations.weight[0]
    adapter.lora_A.data.copy_(torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]]))
    adapter.lora_B.data.copy_(torch.tensor([[0.5], [-0.25], [0.75]]))
    checkpoint = tmp_path / "conv1d.safetensors"
    save_lora_safetensors(
        get_lora_state_dict(torch_model),
        {"rank": 1, "alpha": 1, "adapter_type": "lora"},
        checkpoint,
    )

    mlx_model = TinyMLXConv1d()
    report = apply_lora_checkpoint(mlx_model, checkpoint)
    expected = torch_model.layer.weight.detach().numpy().transpose(0, 2, 1)

    assert report.applied_layers == 1
    assert np.allclose(np.asarray(mlx_model.layer.weight), expected, atol=2e-3)


def test_underfit_lora_config_matches_dashboard_defaults():
    assert underfit_lora_config() == {
        "adapter_type": "dora-rows",
        "rank": 16,
        "alpha": 16,
        "include": None,
        "exclude": [
            "to_timestep_embed",
            "to_cond_embed",
            "to_global_embed",
            "to_local_embed",
            "global_cond_embedder",
            "project_in",
            "project_out",
            "preprocess_conv",
            "postprocess_conv",
        ],
    }


class WideMLXRegressor(nn.Module):
    """Layers wide enough for underfit's config-layer default rank (8)."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 12, bias=False)
        self.output = nn.Linear(12, 8, bias=False)

    def __call__(self, x):
        return self.output(nn.silu(self.input(x)))


def test_inject_from_lora_config_applies_underfit_config_layer_fallbacks():
    model = WideMLXRegressor()
    report, saved_config = inject_from_lora_config(model, {})

    assert report.adapter_type == "lora"
    assert saved_config == {
        "rank": 8,
        "alpha": 8,
        "adapter_type": "lora",
        "include": None,
        "exclude": None,
    }
    for layer in iter_trainable_lora_layers(model):
        assert layer.rank == 8
        assert layer.alpha == 8.0
        assert layer.checkpoint_name == f"model.{layer.source_name}"

    legacy = WideMLXRegressor()
    report, saved_config = inject_from_lora_config(
        legacy,
        {"adapter_type": "dora", "rank": 4, "include": ["output"]},
    )

    assert report.adapter_type == "dora-rows"
    assert report.layer_names == ("output",)
    assert saved_config == {
        "rank": 4,
        "alpha": 4,
        "adapter_type": "dora-rows",
        "include": ["output"],
        "exclude": None,
    }


def test_include_filters_match_checkpoint_convention_names():
    model = dit_mlx.DiT(T_lat=8)
    # Underfit dashboard filter strings are written in torch/checkpoint
    # naming: "model." prefix and "to_local_embed.0" (no ".seq").
    report = inject_trainable_lora(
        model,
        rank=1,
        include=[
            "model.transformer.layers.[0-1].self_attn.to_qkv",
            "model.transformer.layers.0.to_local_embed.0",
        ],
        exclude=["model.transformer.layers.1."],
    )

    assert set(report.layer_names) == {
        "transformer.layers.0.self_attn.to_qkv",
        "transformer.layers.0.to_local_embed.seq.0",
    }


def test_trainable_seconds_embedder_matches_pipeline_conditioner():
    rng = np.random.default_rng(2)
    weight = mx.array(rng.standard_normal((768, 256)).astype(np.float32))
    bias = mx.array(rng.standard_normal(768).astype(np.float32))
    reference = SecondsTotalEmbedder(weight, bias)
    trainable = TrainableSecondsEmbedder(weight, bias)

    for seconds in ([30.0], [0.0, 1.5, 47.25, 285.0, 384.0, 500.0]):
        expected = reference(seconds)
        actual = trainable(seconds)
        assert actual.shape == (len(seconds), 1, 768)
        assert bool(mx.all(actual == expected))
    # mx.array input path (what the trainer passes) is bit-identical too.
    assert bool(mx.all(trainable(mx.array([12.5, 380.0])) == reference([12.5, 380.0])))

    report, _ = inject_from_lora_config(
        trainable,
        underfit_lora_config(),
        checkpoint_prefix="conditioners.seconds_total.",
    )
    layer = next(iter(iter_trainable_lora_layers(trainable)))

    assert report.layer_names == ("embedder.embedding.1",)
    assert layer.checkpoint_name == ("conditioners.seconds_total.embedder.embedding.1")
    # Base Linear is frozen — only the adapter trains.
    assert sorted(
        name for name, _ in tree_flatten(trainable.trainable_parameters())
    ) == [
        "embedder.embedding.1.lora_A",
        "embedder.embedding.1.lora_B",
        "embedder.embedding.1.magnitude",
    ]
    assert trainable([30.0]).shape == (1, 1, 768)


def test_load_trainable_lora_state_round_trips_saved_adapters(tmp_path: Path):
    def build_model():
        mx.random.seed(11)
        model = TinyMLXRegressor()
        inject_trainable_lora(model, rank=2, alpha=2, adapter_type="dora-rows")
        return model

    trained = build_model()
    rng = np.random.default_rng(5)
    for layer in iter_trainable_lora_layers(trained):
        for name in ("lora_A", "lora_B", "magnitude"):
            shape = tuple(getattr(layer, name).shape)
            # fp16-representable values so the fp16 checkpoint is lossless.
            values = rng.standard_normal(shape).astype(np.float16)
            setattr(layer, name, mx.array(values.astype(np.float32)))

    checkpoint = save_lora_checkpoint(trained, tmp_path / "resume.safetensors")
    state_dict, _ = load_lora_checkpoint(checkpoint)
    inputs = mx.array([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]], dtype=mx.float32)

    fresh = build_model()
    assert not np.array_equal(np.asarray(trained(inputs)), np.asarray(fresh(inputs)))

    restored = load_trainable_lora_state(fresh, state_dict)

    assert restored == 2
    assert np.array_equal(np.asarray(trained(inputs)), np.asarray(fresh(inputs)))
    for layer in iter_trainable_lora_layers(fresh):
        assert layer.lora_A.dtype == mx.float32
        assert layer.magnitude.dtype == mx.float32

    # strict=False semantics: unmatched checkpoint layers are tolerated, and
    # 2-D DoRA magnitudes are squeezed like prepare_dora_state_dict.
    partial = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("model.output.")
    }
    magnitude_key = "model.output.parametrizations.weight.0.magnitude"
    partial[magnitude_key] = partial[magnitude_key].reshape(-1, 1)
    partial["model.missing.parametrizations.weight.0.lora_A"] = mx.zeros((2, 3))

    assert load_trainable_lora_state(build_model(), partial) == 1


# ---------------------------------------------------------------------------
# Reformulated (no-full-weight-materialization) DoRA forward
# ---------------------------------------------------------------------------

# lora-xs is included: it moved off the full-weight path onto the cheap
# low-rank path (delta == (U @ M_xs) @ V.T needs no materialization either).
# bora/bora-xs are included in their default "speed" mode, which caches W0²
# once at init to expand colnorm(diag(α)·V) without materializing V
# ("memory" mode keeps the naive full-weight forward).
REFORMULATED_ADAPTER_TYPES = (
    "dora-rows",
    "dora-cols",
    "dora-rows-xs",
    "dora-cols-xs",
    "lora-xs",
    "bora",
    "bora-xs",
)


def _build_reform_layer(
    adapter_type: str,
    *,
    bias: bool,
    dtype,
    fan_in: int = 24,
    fan_out: int = 18,
    rank: int = 4,
    alpha: float = 6.0,
    seed: int = 3,
):
    mx.random.seed(seed)
    base = nn.Linear(fan_in, fan_out, bias=bias)
    base.weight = (mx.random.normal((fan_out, fan_in)) * 0.5).astype(dtype)
    if bias:
        base.bias = mx.random.normal((fan_out,)).astype(dtype)
    layer = lora_module.LoRALinear(
        base,
        rank=rank,
        alpha=alpha,
        source_name="layer",
        adapter_type=adapter_type,
    )
    # Non-trivial adapter state so both the delta and the norms move.
    if adapter_type.endswith("-xs"):
        layer.M_xs = mx.random.normal((rank, rank)) * 0.2
    else:
        layer.lora_A = mx.random.normal((rank, fan_in)) * 0.4
        layer.lora_B = mx.random.normal((fan_out, rank)) * 0.4
    if "dora" in adapter_type:
        layer.magnitude = layer.magnitude * (
            1.0 + 0.1 * mx.random.normal(layer.magnitude.shape)
        )
    elif "bora" in adapter_type:
        layer.magnitude_r = layer.magnitude_r * (
            1.0 + 0.1 * mx.random.normal(layer.magnitude_r.shape)
        )
        layer.magnitude_c = layer.magnitude_c * (
            1.0 + 0.1 * mx.random.normal(layer.magnitude_c.shape)
        )
    return layer


@pytest.mark.parametrize("adapter_type", REFORMULATED_ADAPTER_TYPES)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_reformulated_dora_forward_matches_naive(
    monkeypatch, adapter_type, bias, dtype
):
    layer = _build_reform_layer(adapter_type, bias=bias, dtype=dtype)
    x = mx.random.normal((5, 24)).astype(dtype)

    assert not lora_module._NAIVE_DORA  # reformulation is the default
    reformed_output = layer(x)
    assert reformed_output.dtype == dtype
    reformed = np.asarray(reformed_output, dtype=np.float32)

    monkeypatch.setattr(lora_module, "_NAIVE_DORA", True)
    naive = np.asarray(layer(x), dtype=np.float32)

    # bora needs NO looser fp16 tolerance: its colnorm term-1 matvec runs in
    # the cache's native dtype (fp16 for an fp16 base), but the resulting
    # error stays ~4e-3 max abs — within the shared 2e-2 fp16 tolerance.
    tol = 1e-4 if dtype == mx.float32 else 2e-2
    np.testing.assert_allclose(reformed, naive, rtol=tol, atol=tol)


@pytest.mark.parametrize("adapter_type", REFORMULATED_ADAPTER_TYPES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_reformulated_dora_gradients_match_naive(monkeypatch, adapter_type, dtype):
    layer = _build_reform_layer(adapter_type, bias=True, dtype=dtype)
    x = mx.random.normal((5, 24)).astype(dtype)

    def loss_fn(model, values):
        return mx.mean(model(values).astype(mx.float32) ** 2)

    loss_and_grad = nn.value_and_grad(layer, loss_fn)
    loss_new, grads_new = loss_and_grad(layer, x)
    mx.eval(loss_new, grads_new)

    monkeypatch.setattr(lora_module, "_NAIVE_DORA", True)
    loss_old, grads_old = loss_and_grad(layer, x)
    mx.eval(loss_old, grads_old)

    new_flat = dict(tree_flatten(grads_new))
    old_flat = dict(tree_flatten(grads_old))
    expected_params = {"M_xs"} if adapter_type.endswith("-xs") else {"lora_A", "lora_B"}
    if "dora" in adapter_type:
        expected_params = expected_params | {"magnitude"}
    elif "bora" in adapter_type:
        expected_params = expected_params | {"magnitude_r", "magnitude_c"}
    assert expected_params.issubset(new_flat)
    assert set(new_flat) == set(old_flat)

    tol = 1e-3 if dtype == mx.float32 else 3e-2
    np.testing.assert_allclose(float(loss_new), float(loss_old), rtol=tol, atol=tol)
    for name in sorted(new_flat):
        np.testing.assert_allclose(
            np.asarray(new_flat[name], dtype=np.float32),
            np.asarray(old_flat[name], dtype=np.float32),
            rtol=tol,
            atol=tol,
            err_msg=f"gradient mismatch for {name}",
        )


def test_reformulated_dora_forward_backward_is_faster(monkeypatch):
    """Micro-timing: one medium-sized dora-rows layer, fwd+bwd, 20 iters.

    Direction-only check (the dedicated ablation benchmarks live elsewhere):
    the reformulated path must beat materializing the full 12288x1536 weight.
    """

    fan_in, fan_out = 1536, 12288
    mx.random.seed(0)
    base = nn.Linear(fan_in, fan_out, bias=True)
    layer = lora_module.LoRALinear(
        base,
        rank=16,
        alpha=16.0,
        source_name="layer",
        adapter_type="dora-rows",
    )
    layer.lora_B = mx.random.normal((fan_out, 16)) * 0.02
    x = mx.random.normal((64, fan_in))

    def loss_fn(model, values):
        return mx.mean(model(values) ** 2)

    loss_and_grad = nn.value_and_grad(layer, loss_fn)

    def average_step_seconds(iterations: int = 20) -> float:
        for _ in range(3):  # warmup
            loss, grads = loss_and_grad(layer, x)
            mx.eval(loss, grads)
        start = time.perf_counter()
        for _ in range(iterations):
            loss, grads = loss_and_grad(layer, x)
            mx.eval(loss, grads)
        return (time.perf_counter() - start) / iterations

    reformed = average_step_seconds()
    monkeypatch.setattr(lora_module, "_NAIVE_DORA", True)
    naive = average_step_seconds()

    print(
        f"\ndora-rows 12288x1536 fwd+bwd: reformed {reformed * 1e3:.2f} ms/iter"
        f" vs naive {naive * 1e3:.2f} ms/iter ({naive / reformed:.1f}x)"
    )
    assert reformed < naive, (
        f"reformed {reformed * 1e3:.2f} ms/iter is not faster than naive "
        f"{naive * 1e3:.2f} ms/iter"
    )


@pytest.mark.parametrize("adapter_type", ["bora", "bora-xs"])
def test_bora_memory_mode_is_naive_and_mode_threads_from_config(
    monkeypatch, adapter_type
):
    """bora_mode="memory" IS the naive full-weight code path (bit-identical),
    allocates no W0² cache, and the mode threads through
    inject_from_lora_config → inject_trainable_lora → LoRALinear."""

    def build(bora_mode):
        mx.random.seed(9)
        model = WideMLXRegressor()
        report, _ = inject_from_lora_config(
            model,
            {"adapter_type": adapter_type, "rank": 4},
            bora_mode=bora_mode,
        )
        assert report.adapter_type == adapter_type
        for layer in iter_trainable_lora_layers(model):
            if not adapter_type.endswith("-xs"):
                layer.lora_B = mx.random.normal(layer.lora_B.shape) * 0.3
            else:
                layer.M_xs = mx.random.normal(layer.M_xs.shape) * 0.3
        return model

    x = mx.random.normal((3, 16))

    memory_model = build("memory")
    for layer in iter_trainable_lora_layers(memory_model):
        assert layer.bora_mode == "memory"
        assert not hasattr(layer, "_w0_sq_full")  # no cache allocated
        assert not hasattr(layer, "_w0_sq")
    memory_output = memory_model(x)

    # Bit-identical to SA3_LORA_NAIVE_DORA=1 — memory mode dispatches to the
    # very same _full_weight_forward, not a numerically-close reimplementation.
    monkeypatch.setattr(lora_module, "_NAIVE_DORA", True)
    assert bool(mx.array_equal(memory_output, memory_model(x)))
    monkeypatch.setattr(lora_module, "_NAIVE_DORA", False)

    # Default mode is "speed": cache allocated, same math within fp32 tol.
    speed_model = build("speed")
    for layer in iter_trainable_lora_layers(speed_model):
        assert layer.bora_mode == "speed"
        assert layer._w0_sq_full.dtype == layer.base.weight.dtype
        assert layer._w0_sq_full.shape == layer.base.weight.shape
    np.testing.assert_allclose(
        np.asarray(speed_model(x), dtype=np.float32),
        np.asarray(memory_output, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )

    # The env ablation toggle still overrides bora_mode="speed" too.
    monkeypatch.setattr(lora_module, "_NAIVE_DORA", True)
    assert bool(mx.array_equal(speed_model(x), memory_output))

    with pytest.raises(ValueError, match="bora_mode"):
        build("fast")


def test_bora_w0_sq_cache_stays_out_of_parameters_and_checkpoints(
    tmp_path: Path,
):
    mx.random.seed(13)
    model = WideMLXRegressor()
    inject_trainable_lora(model, rank=4, alpha=4, adapter_type="bora")

    param_names = [name for name, _ in tree_flatten(model.parameters())]
    assert any(name.endswith(".lora_A") for name in param_names)
    assert not any("_w0_sq" in name for name in param_names)  # incl. _w0_sq_full
    assert not any(
        "_w0_sq" in name for name, _ in tree_flatten(model.trainable_parameters())
    )

    checkpoint = save_lora_checkpoint(model, tmp_path / "bora.safetensors")
    state_dict, config = load_lora_checkpoint(checkpoint)

    assert config["adapter_type"] == "bora"
    assert not any("_w0_sq" in key for key in state_dict)
    assert sorted(key.rsplit(".", 1)[1] for key in state_dict) == sorted(
        ["lora_A", "lora_B", "magnitude_r", "magnitude_c"] * 2
    )


def test_reformulated_bora_forward_backward_is_faster():
    """Micro-timing: one medium-sized bora layer, fwd+bwd, 20 iters.

    Direction-only check like the dora-rows one: speed mode (reformulated,
    cached W0²) must beat memory mode (materializing the full 12288x1536
    weight). Expected in the same ballpark as the dora-rows reform (7.2x)
    minus the extra α² @ W0² matvec.
    """

    fan_in, fan_out = 1536, 12288
    mx.random.seed(0)
    base = nn.Linear(fan_in, fan_out, bias=True)
    layer = lora_module.LoRALinear(
        base,
        rank=16,
        alpha=16.0,
        source_name="layer",
        adapter_type="bora",
    )
    layer.lora_B = mx.random.normal((fan_out, 16)) * 0.02
    x = mx.random.normal((64, fan_in))

    def loss_fn(model, values):
        return mx.mean(model(values) ** 2)

    loss_and_grad = nn.value_and_grad(layer, loss_fn)

    def average_step_seconds(iterations: int = 20) -> float:
        for _ in range(3):  # warmup
            loss, grads = loss_and_grad(layer, x)
            mx.eval(loss, grads)
        start = time.perf_counter()
        for _ in range(iterations):
            loss, grads = loss_and_grad(layer, x)
            mx.eval(loss, grads)
        return (time.perf_counter() - start) / iterations

    assert layer.bora_mode == "speed"
    speed = average_step_seconds()
    layer.bora_mode = "memory"  # dispatch-only switch; cache is just unused
    memory = average_step_seconds()

    print(
        f"\nbora 12288x1536 fwd+bwd: speed {speed * 1e3:.2f} ms/iter"
        f" vs memory {memory * 1e3:.2f} ms/iter ({memory / speed:.1f}x)"
    )
    assert speed < memory, (
        f"speed mode {speed * 1e3:.2f} ms/iter is not faster than memory "
        f"mode {memory * 1e3:.2f} ms/iter"
    )


@needs_underfit_reference
def test_checkpoint_key_naming_matches_real_underfit_checkpoint(tmp_path: Path):
    reference_state, reference_config = load_lora_checkpoint(
        UNDERFIT_REFERENCE_CHECKPOINT
    )
    config = underfit_lora_config()

    # The reference checkpoint's lora_config metadata is exactly the product
    # defaults (plus the injected step/epoch fields).
    assert {key: reference_config[key] for key in config} == config

    # Injecting the medium DiT with the product config selects exactly the
    # layer set underfit trained, with byte-identical checkpoint keys.
    model = dit_mlx_medium.DiT(T_lat=8)
    report, _ = inject_from_lora_config(model, config)
    produced_keys = set()
    for layer in iter_trainable_lora_layers(model):
        root = f"{layer.checkpoint_name}.parametrizations.weight.0"
        produced_keys.update(
            f"{root}.{param}" for param in ("lora_A", "lora_B", "magnitude")
        )
    reference_dit_keys = {key for key in reference_state if key.startswith("model.")}

    assert produced_keys == reference_dit_keys
    assert report.layer_count == len(reference_dit_keys) // 3

    # Resume: the real underfit checkpoint loads into the injected model.
    assert load_trainable_lora_state(model, reference_state) == report.layer_count

    # The saver reproduces a real medium DiT layer's keys AND shapes exactly.
    single = dit_mlx_medium.DiT(T_lat=8)
    inject_from_lora_config(single, dict(config, include=["layers.0.self_attn.to_qkv"]))
    saved_state, _ = load_lora_checkpoint(
        save_lora_checkpoint(single, tmp_path / "single-layer.safetensors")
    )
    layer_root = "model.transformer.layers.0.self_attn.to_qkv"
    expected = {
        key: tuple(value.shape)
        for key, value in reference_state.items()
        if key.startswith(f"{layer_root}.")
    }

    assert len(expected) == 3
    assert {key: tuple(value.shape) for key, value in saved_state.items()} == expected

    # The conditioner layer comes out at underfit's exact key root too.
    rng = np.random.default_rng(3)
    seconds_module = TrainableSecondsEmbedder(
        mx.array(rng.standard_normal((768, 256)).astype(np.float32)),
        mx.array(rng.standard_normal(768).astype(np.float32)),
    )
    inject_from_lora_config(
        seconds_module, config, checkpoint_prefix="conditioners.seconds_total."
    )
    saved_cond, _ = load_lora_checkpoint(
        save_lora_checkpoint(seconds_module, tmp_path / "conditioner.safetensors")
    )
    expected_cond = {
        key: tuple(value.shape)
        for key, value in reference_state.items()
        if key.startswith("conditioners.")
    }

    assert len(expected_cond) == 3
    assert {
        key: tuple(value.shape) for key, value in saved_cond.items()
    } == expected_cond
    assert load_trainable_lora_state(seconds_module, reference_state) == 1
