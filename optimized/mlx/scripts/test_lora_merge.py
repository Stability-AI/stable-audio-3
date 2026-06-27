"""Unit tests for models/defs/lora_merge.py — the LoRA merge-at-load math.

Pure-numpy/MLX, no model weights or torch needed. Run either way:

    python scripts/test_lora_merge.py          # standalone (no pytest needed)
    pytest scripts/test_lora_merge.py

Correctness is established by:
  * the zero-init -> identity invariant for every adapter type (a strong guard on
    axis / reshape / transpose bugs: with B/M_xs zeroed and magnitudes at their
    init values, every forward must return W0);
  * an exact independent reconstruction for standard LoRA and PEFT;
  * the Conv1d (out,k,in) <-> PyTorch (out,in,k) layout round-trip;
  * the to_local_embed name remap;
  * --lora-strength scaling and the bit-exact strength-0 bypass;
  * the trust boundary (pickle refusal) and base-mismatch / 0-merge handling.
"""
import json
import os
import sys
import tempfile

import numpy as np
import mlx.core as mx

# Import lora_merge the same way the CLI does (REPO = optimized/mlx on sys.path).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.defs import lora_merge as lm  # noqa: E402

rng = np.random.default_rng(0)


def _save_native(path, layer, params, cfg):
    d = {f"{layer}.parametrizations.weight.0.{k}": mx.array(v.astype(np.float32))
         for k, v in params.items()}
    mx.save_safetensors(path, d, metadata={"lora_config": json.dumps(cfg)})


def _init_params(adapter_type, W0, rank, nonzero):
    """Init so a zeroed core -> identity; if nonzero, give a real core."""
    fo, fi = W0.shape
    p = {}
    if not adapter_type.endswith("-xs"):
        p["lora_A"] = rng.standard_normal((rank, fi)).astype(np.float32)
        p["lora_B"] = (rng.standard_normal((fo, rank)) if nonzero
                       else np.zeros((fo, rank))).astype(np.float32)
    else:
        p["M_xs"] = (rng.standard_normal((rank, rank)) if nonzero
                     else np.zeros((rank, rank))).astype(np.float32)
    if "magnitude" in lm._PARAMS_FOR[adapter_type]:
        nd = 1 if "rows" in adapter_type else 0
        p["magnitude"] = np.linalg.norm(W0, axis=nd).astype(np.float32)
    if "magnitude_r" in lm._PARAMS_FOR[adapter_type]:
        p["magnitude_r"] = np.linalg.norm(W0, axis=1).astype(np.float32)
        p["magnitude_c"] = np.linalg.norm(W0, axis=0).astype(np.float32)
    return p


def _np(arr):
    return np.array(arr.astype(mx.float32))


def test_trust_boundary_refuses_pickle():
    for bad in ("evil.ckpt", "evil.pt", "evil.bin", "weights.npz"):
        try:
            lm._load_safetensors(bad)
            assert False, f"should have refused {bad}"
        except lm.LoraError:
            pass


def test_all_types_zero_init_identity_and_nonzero_delta():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for atype in lm._PARAMS_FOR:
            path = os.path.join(tmp, f"{atype}.safetensors")
            # zero-init -> identity
            _save_native(path, "foo", _init_params(atype, W0, 4, nonzero=False),
                         {"adapter_type": atype, "rank": 4, "alpha": 4})
            w = {"foo.weight": mx.array(W0)}
            lm.merge_loras_into_weights(w, [path])
            assert np.allclose(_np(w["foo.weight"]), W0, atol=1e-4), f"{atype} identity"
            # nonzero core -> finite, changed
            _save_native(path, "foo", _init_params(atype, W0, 4, nonzero=True),
                         {"adapter_type": atype, "rank": 4, "alpha": 4})
            w = {"foo.weight": mx.array(W0)}
            lm.merge_loras_into_weights(w, [path])
            got = _np(w["foo.weight"])
            assert np.isfinite(got).all() and not np.allclose(got, W0), f"{atype} delta"


def test_standard_lora_exact_and_strength():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((6, 4)).astype(np.float32)
    expect = W0 + (8.0 / 4.0) * (B @ A)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "std.safetensors")
        _save_native(path, "foo", {"lora_A": A, "lora_B": B},
                     {"adapter_type": "lora", "rank": 4, "alpha": 8})
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path])
        assert np.allclose(_np(w["foo.weight"]), expect, atol=1e-4)
        # strength 0.5 halves the delta; strength 0 is the original
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path], strength=0.5)
        assert np.allclose(_np(w["foo.weight"]), W0 + 0.5 * (8.0 / 4.0) * (B @ A), atol=1e-4)
        w = {"foo.weight": mx.array(W0)}
        lm.merge_loras_into_weights(w, [path], strength=0.0)
        assert np.allclose(_np(w["foo.weight"]), W0, atol=1e-6)


def test_conv1d_layout_round_trip():
    out_c, k_c, in_c = 4, 1, 3
    Wc = rng.standard_normal((out_c, k_c, in_c)).astype(np.float32)   # MLX layout
    Ac = rng.standard_normal((2, in_c * k_c)).astype(np.float32)
    Bc = rng.standard_normal((out_c, 2)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "conv.safetensors")
        _save_native(path, "conv", {"lora_A": Ac, "lora_B": Bc},
                     {"adapter_type": "lora", "rank": 2, "alpha": 2})
        w = {"conv.weight": mx.array(Wc)}
        lm.merge_loras_into_weights(w, [path])
        got = _np(w["conv.weight"])
        expect = Wc + (Bc @ Ac).reshape(out_c, in_c, k_c).transpose(0, 2, 1)
        assert got.shape == Wc.shape and np.allclose(got, expect, atol=1e-4)


def test_to_local_embed_remap():
    Wt = rng.standard_normal((5, 5)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tle.safetensors")
        _save_native(path, "transformer.layers.0.to_local_embed.0",
                     {"lora_A": rng.standard_normal((2, 5)).astype(np.float32),
                      "lora_B": rng.standard_normal((5, 2)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 2, "alpha": 2})
        w = {"transformer.layers.0.to_local_embed.seq.0.weight": mx.array(Wt)}
        stats = lm.merge_loras_into_weights(w, [path])
        assert stats["merged"] == 1 and not stats["skipped"]


def test_peft_format_exact():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((6, 4)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "adapter_config.json"), "w") as fh:
            json.dump({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "use_dora": False}, fh)
        mx.save_safetensors(
            os.path.join(tmp, "adapter_model.safetensors"),
            {"base_model.model.foo.lora_A.weight": mx.array(A),
             "base_model.model.foo.lora_B.weight": mx.array(B)},
            metadata={"format": "pt"})
        w = {"foo.weight": mx.array(W0)}
        stats = lm.merge_loras_into_weights(w, [tmp])   # directory input
        assert stats["merged"] == 1
        assert np.allclose(_np(w["foo.weight"]), W0 + (8.0 / 4.0) * (B @ A), atol=1e-4)


def test_unknown_layers_skipped_base_untouched():
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "std.safetensors")
        _save_native(path, "foo",
                     {"lora_A": rng.standard_normal((4, 8)).astype(np.float32),
                      "lora_B": rng.standard_normal((6, 4)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 4, "alpha": 4})
        w = {"other.weight": mx.array(W0)}
        msgs = []
        stats = lm.merge_loras_into_weights(w, [path], log=msgs.append)
        assert stats["merged"] == 0 and stats["skipped"] == ["foo"]
        assert np.allclose(_np(w["other.weight"]), W0)
        # a 0-merge run must warn, not look like a successful no-op
        assert any("WARNING" in m and "0 layers" in m for m in msgs)


def test_base_mismatch_raises_clear_error():
    # base is (6, 8) but the adapter was trained for an (10, 8) layer
    W0 = rng.standard_normal((6, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mismatch.safetensors")
        _save_native(path, "foo",
                     {"lora_A": rng.standard_normal((4, 8)).astype(np.float32),
                      "lora_B": rng.standard_normal((10, 4)).astype(np.float32)},
                     {"adapter_type": "lora", "rank": 4, "alpha": 4})
        w = {"foo.weight": mx.array(W0)}
        try:
            lm.merge_loras_into_weights(w, [path])
            assert False, "should have raised on base mismatch"
        except lm.LoraError as e:
            assert "foo" in str(e) and "base" in str(e).lower()


def test_svd_bases_orthonormal():
    Wsvd = rng.standard_normal((6, 8)).astype(np.float32)
    U, V = lm._svd_bases(Wsvd, rank=6)
    assert np.allclose(U.T @ U, np.eye(6), atol=1e-4)
    assert np.allclose(V.T @ V, np.eye(6), atol=1e-4)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
    print("\n" + ("ALL PASS" if not failed else f"FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
