"""Fast (reformulated) LoRA/DoRA forward vs the naive materialized parametrization.

The fast path (the fast-lora forward (models/lora/model.py)) must
be a drop-in for torch.nn.utils.parametrize's materialize-W'-per-access
forward: same outputs, same gradients, same state_dict layout, and graceful
fallback everywhere the reformulation doesn't apply.
"""

import copy
from functools import partial

import pytest
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from stable_audio_3.models.lora.model import (
    LoRAParametrization,
    add_lora,
    _uninstall_fast_lora_forward,
    _install_fast_lora_forward,
    merge_lora,
    remove_lora,
    set_lora_strength,
)

ADAPTERS = ["lora", "dora-rows", "dora-cols"]
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def make_model(bias=True, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(64, 96, bias=bias),
        nn.GELU(),
        nn.Linear(96, 48, bias=bias),
    ).to(device)
    model.requires_grad_(False)
    return model


def lora_config(adapter_type, rank=8, alpha=16):
    return {
        nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                adapter_type=adapter_type,
            ),
        },
    }


def randomize_lora(model, seed=1, magnitude_jitter=0.3):
    """add_lora inits lora_B to zeros (delta == 0) — make the test non-trivial."""
    g = torch.Generator().manual_seed(seed)
    for mod in model.modules():
        if not isinstance(mod, LoRAParametrization):
            continue
        with torch.no_grad():
            for name in ("lora_A", "lora_B"):
                p = getattr(mod, name)
                p.copy_(torch.randn(p.shape, generator=g).to(p) * 0.05)
            for name in ("magnitude",):
                if hasattr(mod, name):
                    p = getattr(mod, name)
                    p.mul_(
                        1.0 + magnitude_jitter * torch.rand(p.shape, generator=g).to(p)
                    )


def lora_trainables(model):
    out = {}
    for name, p in model.named_parameters():
        if name.split(".")[-1] in ("lora_A", "lora_B", "magnitude"):
            out[name] = p
    return out


def run_fwd_bwd(model, x, proj, grads=True):
    for p in lora_trainables(model).values():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None
    y = model(x)
    if grads:
        (y * proj).sum().backward()
        gs = {k: p.grad.detach().clone() for k, p in lora_trainables(model).items()}
        return y.detach(), gs
    return y.detach(), None


def rel_err(a, b):
    denom = b.norm().item()
    if denom == 0:
        return a.norm().item()
    return (a - b).norm().item() / denom


def build_pair(adapter, bias, device, strength=None):
    """Same model twice: fast-enabled and naive (fast disabled)."""
    model = make_model(bias=bias, device=device)
    add_lora(model, lora_config(adapter))  # add_lora auto-enables the fast path
    randomize_lora(model)
    if strength is not None:
        set_lora_strength(model, strength)
    naive = copy.deepcopy(model)
    _uninstall_fast_lora_forward(naive)
    _install_fast_lora_forward(model)  # idempotent; explicit for clarity
    return model, naive


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("adapter", ADAPTERS)
def test_output_parity_fp32(adapter, bias, device):
    fast, naive = build_pair(adapter, bias, device)
    x = torch.randn(4, 7, 64, device=device)
    y_fast = fast(x)
    y_naive = naive(x)
    assert fast[0].__dict__.get("_fast_lora_wrapped"), "fast wrapper not installed"
    assert not naive[0].__dict__.get("_fast_lora_wrapped", False)
    torch.testing.assert_close(y_fast, y_naive, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adapter", ADAPTERS)
def test_gradient_parity_fp32(adapter, device):
    fast, naive = build_pair(adapter, bias=True, device=device)
    x = torch.randn(4, 7, 64, device=device)
    proj = torch.randn(4, 7, 48, device=device)
    _, g_fast = run_fwd_bwd(fast, x, proj)
    _, g_naive = run_fwd_bwd(naive, x, proj)
    assert set(g_fast) == set(g_naive) and len(g_fast) > 0
    worst = max(rel_err(g_fast[k], g_naive[k]) for k in g_naive)
    print(f"[grad-parity fp32 {adapter} {device}] worst rel err = {worst:.3e}")
    assert worst <= 1e-4, f"{adapter}/{device}: worst grad rel err {worst:.3e} > 1e-4"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("adapter", ADAPTERS)
def test_gradient_parity_mps_fp16_autocast(adapter):
    device = "mps"
    fast, naive = build_pair(adapter, bias=True, device=device)
    x = torch.randn(4, 7, 64, device=device)
    proj = torch.randn(4, 7, 48, device=device)
    with torch.amp.autocast(device, dtype=torch.float16):
        y_fast, g_fast = run_fwd_bwd(fast, x, proj)
        y_naive, g_naive = run_fwd_bwd(naive, x, proj)
    out_err = rel_err(y_fast.float(), y_naive.float())
    worst = max(rel_err(g_fast[k].float(), g_naive[k].float()) for k in g_naive)
    print(
        f"[autocast fp16 {adapter}] output rel err = {out_err:.3e}, worst grad rel err = {worst:.3e}"
    )
    # fp16 GEMM noise dominates; ~1e-2-class is the expected scale, this bound is a regression guard
    assert out_err <= 3e-2
    assert worst <= 5e-2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adapter", ADAPTERS)
def test_strength_buffer(adapter, device):
    # strength 0.5: fast == naive at the same strength
    fast, naive = build_pair(adapter, bias=True, device=device, strength=0.5)
    x = torch.randn(3, 5, 64, device=device)
    torch.testing.assert_close(fast(x), naive(x), atol=1e-5, rtol=1e-5)

    # strength 0: exactly the frozen base
    set_lora_strength(fast, 0.0)
    base = make_model(bias=True, device=device)  # same seed -> identical base weights
    y_fast = fast(x)
    y_base = base(x)
    assert torch.equal(y_fast, y_base), (
        f"strength=0 must be bit-exact to the base ({adapter}/{device})"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_stacked_parametrization_falls_back(device):
    fast, _ = build_pair("dora-rows", bias=True, device=device)
    naive = copy.deepcopy(fast)
    _uninstall_fast_lora_forward(naive)

    # stack a second adapter at lora_index=1 on BOTH copies (same init seed)
    for m in (fast, naive):
        torch.manual_seed(7)
        for mod in m.modules():
            if isinstance(mod, nn.Linear) and parametrize.is_parametrized(
                mod, "weight"
            ):
                p2 = LoRAParametrization.from_linear(
                    mod, rank=4, lora_alpha=8, adapter_type="dora-rows", lora_index=1
                )
                with torch.no_grad():
                    p2.lora_B.copy_(torch.randn_like(p2.lora_B) * 0.05)
                parametrize.register_parametrization(mod, "weight", p2, unsafe=True)
    _install_fast_lora_forward(fast)

    for mod in fast.modules():
        if isinstance(mod, nn.Linear):
            assert len(mod.parametrizations["weight"]) == 2

    x = torch.randn(2, 9, 64, device=device)
    y_fast = fast(x)  # wrapper must detect the stack and fall back per-forward
    y_naive = naive(x)
    assert torch.equal(y_fast, y_naive), "stacked modules must use the exact naive path"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adapter", ADAPTERS)
def test_merge_and_remove_with_wrapper(adapter, device):
    # NOTE: fresh models per case — deepcopying a parametrized module shares the
    # injected ParametrizedLinear class, so un-parametrizing one copy breaks the
    # other (pre-existing torch.nn.utils.parametrize behavior, wrapper-unrelated).
    x = torch.randn(2, 6, 64, device=device)

    merged, _ = build_pair(adapter, bias=True, device=device)
    y_before = merged(x)
    merge_lora(merged)
    assert not parametrize.is_parametrized(merged[0], "weight")
    assert not merged[0].__dict__.get("_fast_lora_wrapped", False)
    torch.testing.assert_close(merged(x), y_before, atol=1e-5, rtol=1e-5)

    removed, _ = build_pair(adapter, bias=True, device=device)
    remove_lora(removed)
    assert not parametrize.is_parametrized(removed[0], "weight")
    base = make_model(bias=True, device=device)
    torch.testing.assert_close(removed(x), base(x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_state_dict_layout_unchanged(adapter):
    fast = make_model()
    add_lora(fast, lora_config(adapter))
    randomize_lora(fast)

    naive = make_model()
    add_lora(naive, lora_config(adapter))
    _uninstall_fast_lora_forward(naive)  # naive reference: class-level forward
    assert not naive[0].__dict__.get("_fast_lora_wrapped", False)

    # populate the norm-constant cache before snapshotting
    fast(torch.randn(2, 4, 64))
    assert fast[0].__dict__.get("_fast_lora_w0_sq") is not None or adapter == "lora"

    k_fast, k_naive = set(fast.state_dict()), set(naive.state_dict())
    assert k_fast == k_naive
    assert not any("fast" in k or "w0_sq" in k for k in k_fast)
    # canonical parametrize key layout intact
    assert any(".parametrizations.weight.original" in k for k in k_fast)
    assert any(".parametrizations.weight.0.lora_A" in k for k in k_fast)


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_shared_between_direction_and_norm(device):
    """With lora_dropout_p > 0 in train mode, fast and naive draw dropout once per
    forward; under a fixed RNG seed both paths must consume the same draws."""
    cfg = {
        nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=8,
                lora_alpha=16,
                adapter_type="dora-rows",
                lora_dropout_p=0.5,
            ),
        },
    }
    model = make_model(device=device)
    add_lora(model, cfg)
    randomize_lora(model)
    naive = copy.deepcopy(model)
    _uninstall_fast_lora_forward(naive)
    model.train(), naive.train()

    x = torch.randn(2, 5, 64, device=device)
    torch.manual_seed(123)
    y_fast = model(x)
    torch.manual_seed(123)
    y_naive = naive(x)
    torch.testing.assert_close(y_fast, y_naive, atol=1e-5, rtol=1e-5)
