"""MPS (Apple Silicon) LoRA-training smoke test.

Verifies the device-neutral AMP plumbing added for MPS training support:

  - stable_audio_3.utils.device: resolve_device / autocast_context /
    make_grad_scaler on "mps"
  - fp32 islands (RoPE, ExpoFourierFeatures) actually stay fp32 under an
    active MPS autocast (the old @autocast("cuda", enabled=False) decorators
    silently did nothing on MPS)
  - a real (tiny) DiffusionTransformer with LoRA adapters runs 3
    forward/backward/AdamW steps on "mps" under fp16 autocast + GradScaler,
    using underfit's rectified-flow signal-only masked-MSE loss shape;
    LoRA params move, base params don't.

Skipped entirely when MPS is unavailable. Runnable standalone:
    python tests/test_mps_training_smoke.py
"""

import os
import sys
from functools import partial

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HAS_MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not HAS_MPS, reason="MPS not available on this machine")

DEVICE = "mps"


# ---------------------------------------------------------------------------
# underfit loss (real module when importable, faithful mirror otherwise)
# ---------------------------------------------------------------------------


def _underfit_loss_fns():
    try:
        from underfit.training.loss import compute_masked_loss, compute_normalized_mse

        return compute_normalized_mse, compute_masked_loss
    except ImportError:
        pass

    # Mirror of underfit/training/loss.py (loss_normalization="none",
    # mask_padding_attention=True → signal-only masked MSE).
    def compute_normalized_mse(
        pred, target, loss_mask, loss_normalization="none", loss_norm_eps=1e-6
    ):
        return (pred - target) ** 2

    def compute_masked_loss(
        loss_full, loss_mask, mask_padding_attention, mask_loss_weight=0.0
    ):
        signal = torch.where(loss_mask.unsqueeze(1), loss_full, 0.0)
        signal_sum = signal.sum(dim=(1, 2))
        n_channels = loss_full.shape[1]
        signal_count = loss_mask.sum(dim=1) * n_channels
        per_sample_loss = signal_sum / (signal_count + 1e-8)
        loss = per_sample_loss.mean()
        signal_mean = signal.sum() / (signal_count.sum() + 1e-8)
        return loss, signal_mean.detach(), torch.zeros(())

    return compute_normalized_mse, compute_masked_loss


# ---------------------------------------------------------------------------
# Device helper probes
# ---------------------------------------------------------------------------


def test_resolve_device_prefers_mps_without_cuda():
    from stable_audio_3.utils.device import resolve_device

    expected = "cuda" if torch.cuda.is_available() else "mps"
    assert resolve_device() == expected
    assert resolve_device("cpu") == "cpu"


def test_autocast_context_fp16_on_mps():
    from stable_audio_3.utils.device import autocast_context

    a = torch.randn(8, 8, device=DEVICE)
    with autocast_context(DEVICE, dtype=torch.float16):
        assert torch.is_autocast_enabled(DEVICE)
        out = a @ a
    assert out.dtype == torch.float16


def test_grad_scaler_on_mps_full_cycle():
    from stable_audio_3.utils.device import autocast_context, make_grad_scaler

    scaler = make_grad_scaler(DEVICE)
    assert scaler.is_enabled(), "torch 2.13 GradScaler('mps') should be enabled"

    lin = torch.nn.Linear(4, 4).to(DEVICE)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    with autocast_context(DEVICE, dtype=torch.float16):
        loss = lin(torch.randn(2, 4, device=DEVICE)).square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() > 0


def test_fp32_islands_hold_under_mps_autocast():
    """RoPE + ExpoFourierFeatures must compute in fp32 inside MPS autocast."""
    from stable_audio_3.models.blocks import ExpoFourierFeatures
    from stable_audio_3.models.transformer import RotaryEmbedding

    rope = RotaryEmbedding(16).to(DEVICE)
    eff = ExpoFourierFeatures(32).to(DEVICE)
    with torch.amp.autocast(DEVICE, dtype=torch.float16):
        freqs, _ = rope.forward_from_seq_len(8)
        feats = eff(torch.rand(4, device=DEVICE))
    # The einsum in RotaryEmbedding.forward is autocast-eligible; fp32 output
    # proves the disable_autocast island took effect.
    assert freqs.dtype == torch.float32
    assert feats.dtype == torch.float32


# ---------------------------------------------------------------------------
# End-to-end: tiny DiT + LoRA, 3 training steps on MPS
# ---------------------------------------------------------------------------


def _build_tiny_dit_with_lora():
    from stable_audio_3.models.dit import DiffusionTransformer
    from stable_audio_3.models.lora import LoRAParametrization, add_lora

    torch.manual_seed(0)
    model = DiffusionTransformer(
        io_channels=8,
        embed_dim=64,
        depth=2,
        num_heads=2,
        cond_token_dim=32,
        global_cond_dim=16,
        transformer_type="continuous_transformer",
        diffusion_objective="rectified_flow",
    ).to(DEVICE)

    lora_cfg = {
        torch.nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=4,
                lora_alpha=4.0,
                adapter_type="lora",
            ),
        },
    }
    add_lora(model, lora_cfg)

    lora_params, base_params = [], []
    for name, p in model.named_parameters():
        if "lora_" in name:
            lora_params.append((name, p))
        else:
            base_params.append((name, p))
    assert lora_params, "add_lora attached no LoRA parameters"

    # Mirror underfit/training/loop.py: only LoRA params train.
    for _, p in base_params:
        p.requires_grad_(False)
    for _, p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)
    return model, lora_params, base_params


def test_lora_training_steps_on_mps():
    from stable_audio_3.utils.device import autocast_context, make_grad_scaler

    compute_normalized_mse, compute_masked_loss = _underfit_loss_fns()
    model, lora_params, base_params = _build_tiny_dit_with_lora()

    lora_before = {n: p.detach().clone() for n, p in lora_params}
    base_before = {n: p.detach().clone() for n, p in base_params}

    opt = torch.optim.AdamW([p for _, p in lora_params], lr=1e-2)
    scaler = make_grad_scaler(DEVICE)

    # Fixed batch so loss trajectory is meaningful across steps.
    torch.manual_seed(1)
    B, C, T = 2, 8, 32
    x = torch.randn(B, C, T, device=DEVICE)
    cross = torch.randn(B, 24, 32, device=DEVICE)
    glob = torch.randn(B, 16, device=DEVICE)
    t = torch.rand(B, device=DEVICE) * 0.8 + 0.1
    noise = torch.randn_like(x)
    loss_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    # rectified_flow noising (as in underfit/training/loop.py)
    alphas, sigmas = (1 - t)[:, None, None], t[:, None, None]
    noised = x * alphas + noise * sigmas
    target = noise - x

    losses = []
    for _step in range(3):
        with autocast_context(DEVICE, dtype=torch.float16):
            out = model(noised, t, cross_attn_cond=cross, global_embed=glob)
            mse_full = compute_normalized_mse(out, target, loss_mask)
            loss, _sig, _pad = compute_masked_loss(
                mse_full, loss_mask, mask_padding_attention=True
            )
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

    assert all(x == x and abs(x) != float("inf") for x in losses), (
        f"non-finite loss: {losses}"
    )
    assert losses[-1] != losses[0], f"loss did not change over 3 steps: {losses}"
    # On a fixed batch with lr=1e-2 the loss should trend down.
    assert losses[-1] < losses[0] * 1.05, f"loss did not decrease: {losses}"

    lora_changed = any(
        not torch.equal(p.detach(), lora_before[n]) for n, p in lora_params
    )
    assert lora_changed, "no LoRA parameter changed after 3 optimizer steps"

    for n, p in base_params:
        assert torch.equal(p.detach(), base_before[n]), f"base param {n} changed"


def test_full_wrapper_training_step_on_mps():
    """The exact call shape underfit's loop uses, on the full SA3 wrapper:
    conditioner(metadata, device) -> model(noised, t, cond=..., cfg_dropout_prob=...).

    Uses a NumberConditioner (no checkpoint downloads) and pretransform=None
    (underfit's pre_encoded path never calls pretransform.encode).
    """
    from stable_audio_3.models.conditioners import MultiConditioner, NumberConditioner
    from stable_audio_3.models.diffusion import (
        ConditionedDiffusionModelWrapper,
        DiTWrapper,
    )
    from stable_audio_3.models.lora import LoRAParametrization, add_lora
    from stable_audio_3.utils.device import autocast_context, make_grad_scaler

    compute_normalized_mse, compute_masked_loss = _underfit_loss_fns()

    torch.manual_seed(0)
    dit = DiTWrapper(
        diffusion_objective="rectified_flow",
        io_channels=8,
        embed_dim=64,
        depth=2,
        num_heads=2,
        cond_token_dim=32,
        global_cond_dim=32,
        transformer_type="continuous_transformer",
    )
    conditioner = MultiConditioner(
        {"seconds_total": NumberConditioner(output_dim=32, min_val=0, max_val=512)}
    )
    model = ConditionedDiffusionModelWrapper(
        dit,
        conditioner,
        io_channels=8,
        sample_rate=44100,
        min_input_length=1,
        diffusion_objective="rectified_flow",
        pretransform=None,
        cross_attn_cond_ids=["seconds_total"],
        global_cond_ids=["seconds_total"],
    ).to(DEVICE)

    add_lora(
        model.model,
        {
            torch.nn.Linear: {
                "weight": partial(
                    LoRAParametrization.from_linear,
                    rank=4,
                    lora_alpha=4.0,
                    adapter_type="lora",
                ),
            },
        },
    )
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora_params
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)

    opt = torch.optim.AdamW(lora_params, lr=1e-2)
    scaler = make_grad_scaler(DEVICE)

    torch.manual_seed(2)
    B, C, T = 2, 8, 32
    metadata = [{"seconds_total": 30.0}, {"seconds_total": 47.5}]
    diffusion_input = torch.randn(B, C, T, device=DEVICE)
    loss_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    losses = []
    for _step in range(3):
        with autocast_context(DEVICE, dtype=torch.float16):
            conditioning = model.conditioner(metadata, DEVICE)
            t = torch.rand(B, device=DEVICE) * 0.8 + 0.1
            alphas, sigmas = (1 - t)[:, None, None], t[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised = diffusion_input * alphas + noise * sigmas
            target = noise - diffusion_input
            output = model(noised, t, cond=conditioning, cfg_dropout_prob=0.1)
            mse_full = compute_normalized_mse(output, target, loss_mask)
            loss, _sig, _pad = compute_masked_loss(
                mse_full, loss_mask, mask_padding_attention=True
            )
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

    assert all(x == x and abs(x) != float("inf") for x in losses), (
        f"non-finite loss: {losses}"
    )
    assert len(set(losses)) > 1, f"loss frozen across steps: {losses}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
