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

from optimized.mlx.models.defs.lora import (
    apply_lora_checkpoint,
    apply_lora_checkpoints,
    inject_trainable_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from stable_audio_3.models.lora import (
    LoRAParametrization,
    add_lora,
    get_lora_state_dict,
    load_lora_checkpoint as load_torch_lora_checkpoint,
    save_lora_safetensors,
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
        extra_config={"step": 20},
    )
    mlx_state, mlx_config = load_lora_checkpoint(checkpoint)
    torch_state, torch_config = load_torch_lora_checkpoint(checkpoint)

    assert sorted(mlx_state) == sorted(torch_state)
    assert mlx_config == torch_config
    assert mlx_config["adapter_type"] == "dora-rows"
    assert mlx_config["step"] == 20


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
        "to_local_embed.0.parametrizations.weight.0.lora_A",
        "to_local_embed.0.parametrizations.weight.0.lora_B",
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
