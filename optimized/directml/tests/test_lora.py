import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from sa3_dml import lora
from sa3_dml.session import RuntimeConfig, Session
from safetensors.numpy import save_file

PROVIDERS = [p for p in ("dml", "cpu") if p == "cpu" or "DmlExecutionProvider" in ort.get_available_providers()]
RNG = np.random.default_rng(0)


def _params(out=6, inp=5, rank=2):
    return {"lora_A": RNG.standard_normal((rank, inp)).astype(np.float32),
            "lora_B": RNG.standard_normal((out, rank)).astype(np.float32)}


def test_lora_strength_scales_the_low_rank_update():
    w = RNG.standard_normal((6, 5)).astype(np.float32)
    p = _params()
    np.testing.assert_allclose(lora.adapt(w, w, p, "lora", 2.0, 0.5), w + p["lora_B"] @ p["lora_A"], rtol=1e-5)
    np.testing.assert_array_equal(lora.adapt(w, w, p, "lora", 2.0, 0.0), w)


@pytest.mark.parametrize("kind,axis", [("dora-rows", 1), ("dora-cols", 0)])
def test_dora_magnitude_and_strength_blend(kind, axis):
    w = RNG.standard_normal((6, 5)).astype(np.float32)
    p = _params()
    size = w.shape[0] if axis == 1 else w.shape[1]
    p["magnitude"] = np.abs(RNG.standard_normal(size)).astype(np.float32) + 0.5
    full = lora.adapt(w, w, p, kind, 1.0, 1.0)
    np.testing.assert_allclose(np.linalg.norm(full, axis=axis), p["magnitude"], rtol=1e-5)
    half = lora.adapt(w, w, p, kind, 1.0, 0.5)
    base = np.linalg.norm(w, axis=axis)
    np.testing.assert_allclose(np.linalg.norm(half, axis=axis), base + 0.5 * (p["magnitude"] - base), rtol=1e-5)
    np.testing.assert_array_equal(lora.adapt(w, w, p, kind, 1.0, 0.0), w)


def test_bora_scales_rows_then_columns():
    w = RNG.standard_normal((6, 5)).astype(np.float32)
    p = {**_params(), "magnitude_r": np.full(6, 2.0, np.float32), "magnitude_c": np.arange(1, 6, dtype=np.float32)}
    np.testing.assert_allclose(np.linalg.norm(lora.adapt(w, w, p, "bora", 1.0, 1.0), axis=0), p["magnitude_c"],
                               rtol=1e-5)


@pytest.mark.parametrize("shape", [(7, 12), (12, 7)])
def test_svd_bases_match_full_svd_with_reference_signs(shape):
    w = RNG.standard_normal(shape).astype(np.float32)
    u, v = lora.svd_bases(w, 3)
    full_u, s, full_vh = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    for i in range(3):
        assert abs(abs(u[:, i] @ full_u[:, i]) - 1) < 1e-5 and abs(abs(v[:, i] @ full_vh[i]) - 1) < 1e-5
        assert u[np.argmax(np.abs(u[:, i])), i] > 0
    np.testing.assert_allclose(u.T @ w @ v, np.diag(s[:3]), atol=1e-4)
    m = RNG.standard_normal((3, 3)).astype(np.float32)
    np.testing.assert_allclose(lora.adapt(w, w, {"M_xs": m}, "lora-xs", 2.0, 1.0), w + 2.0 * u @ m @ v.T, atol=1e-5)


def test_layer_filters_and_names():
    adapter = lora.Adapter("a", "lora", 1.0, {}, include=["layers.[0-2].ff"], exclude=["layers.1."])
    assert adapter.applies_to("model.transformer.layers.0.ff.ff.2")
    assert not adapter.applies_to("model.transformer.layers.1.ff.ff.2")
    assert not adapter.applies_to("model.transformer.layers.3.ff.ff.2")
    dit_weight = "model.model.transformer.layers.0.ff.ff.2.weight"
    assert lora.checkpoint_weight("model.transformer.layers.0.ff.ff.2") == dit_weight
    assert lora.checkpoint_weight("conditioners.seconds_total.embedder.embedding.1") == \
        "conditioner.conditioners.seconds_total.embedder.embedding.1.weight"


def test_native_and_peft_checkpoints(tmp_path):
    key = "model.transformer.layers.0.ff.ff.2.parametrizations.weight.0."
    tensors = {key + "lora_A": np.ones((4, 3), np.float16), key + "lora_B": np.ones((5, 4), np.float16),
               key + "magnitude": np.ones((1, 3), np.float16), key + "baked_vnorm_row": np.ones(5, np.float32)}
    save_file(tensors, str(tmp_path / "native.safetensors"), metadata={"lora_config": json.dumps(
        {"alpha": 8, "adapter_type": "dora", "exclude": ["to_timestep_embed"]})})
    native = lora.Adapter.load(tmp_path / "native.safetensors")
    assert (native.adapter_type, native.scaling, native.exclude) == ("dora-cols", 2.0, ["to_timestep_embed"])
    assert set(native.layers["model.transformer.layers.0.ff.ff.2"]) == {"lora_A", "lora_B", "magnitude"}

    peft = tmp_path / "peft"
    peft.mkdir()
    save_file({"base_model.model.transformer.layers.0.ff.ff.2.lora_A.weight": np.ones((4, 3), np.float32),
               "base_model.model.transformer.layers.0.ff.ff.2.lora_B.weight": np.ones((5, 4), np.float32)},
              str(peft / "adapter_model.safetensors"))
    (peft / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "use_rslora": True}))
    adapter = lora.Adapter.load(peft)
    assert adapter.adapter_type == "lora" and adapter.scaling == pytest.approx(4.0)
    assert list(adapter.layers) == ["model.transformer.layers.0.ff.ff.2"]
    with pytest.raises(lora.LoraError):
        lora.Adapter.load(tmp_path / "missing.ckpt")


def _graph(tmp_path, weights: dict[str, np.ndarray]):
    nodes, inputs, outputs = [], [], []
    for i, (name, w) in enumerate(weights.items()):
        x, y = f"x{i}", f"y{i}"
        nodes += [helper.make_node("Transpose", [name], [f"{name}_t"]),
                  helper.make_node("MatMul", [x, f"{name}_t"], [y])]
        inputs.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, w.shape[1]]))
        outputs.append(helper.make_tensor_value_info(y, TensorProto.FLOAT, [1, w.shape[0]]))
    inits = [numpy_helper.from_array(w, n) for n, w in weights.items()]
    model = helper.make_model(helper.make_graph(nodes, "g", inputs, outputs, inits),
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    path = tmp_path / "model.onnx"
    onnx.save_model(model, str(path), save_as_external_data=True, location="model.onnx.data", size_threshold=0)
    return path


@pytest.mark.parametrize("provider", PROVIDERS)
def test_merged_weights_replace_graph_weights(tmp_path, provider):
    dit = RNG.standard_normal((6, 5)).astype(np.float32)
    seconds = RNG.standard_normal((4, 3)).astype(np.float32)
    path = _graph(tmp_path, {"dit.blocks.0.proj.weight": dit, "seconds_weight": seconds})
    first, second = _params(), {**_params(), "magnitude": np.full(6, 0.5, np.float32)}
    cond = _params(4, 3)
    a = lora.Adapter("a", "lora", 1.0, {"model.blocks.0.proj": first,
                                        "conditioners.seconds_total.embedder.embedding.1": cond})
    b = lora.Adapter("b", "dora-rows", 1.0, {"model.blocks.0.proj": second, "model.unknown": second})
    merged, reports = lora.merge(path, [(a, 0.5), (b, 1.0)])
    chained = lora.adapt(lora.adapt(dit, dit, first, "lora", 1.0, 0.5), dit, second, "dora-rows", 1.0, 1.0)
    np.testing.assert_allclose(merged["dit.blocks.0.proj.weight"], chained, rtol=1e-5)
    np.testing.assert_allclose(merged["seconds_weight"], lora.adapt(seconds, seconds, cond, "lora", 1.0, 0.5),
                               rtol=1e-5)
    assert [(r.layers, r.skipped) for r in reports] == [(2, 0), (1, 1)]

    x0, x1 = np.ones((1, 5), np.float32), np.ones((1, 3), np.float32)
    session = Session(path, RuntimeConfig(provider=provider), initializers=merged)
    y0, y1 = session.run({"x0": x0, "x1": x1})
    session.close()
    np.testing.assert_allclose(y0, x0 @ chained.T, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y1, x1 @ merged["seconds_weight"].T, rtol=1e-4, atol=1e-4)
    with pytest.raises(lora.LoraError):
        lora.merge(path, [(lora.Adapter("c", "lora", 1.0, {"model.other": first}), 1.0)])


def test_safetensors_reader_widens_bfloat16(tmp_path):
    values = np.array([[1.0, -2.5], [0.15625, 32768.0]], np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
    f16 = values.astype("<f2").tobytes()
    header = {"__metadata__": {"lora_config": "{}"},
              "a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(bf16)]},
              "b": {"dtype": "F16", "shape": [2, 2], "data_offsets": [len(bf16), len(bf16) + len(f16)]}}
    blob = json.dumps(header).encode()
    path = tmp_path / "x.safetensors"
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + bf16 + f16)
    tensors, metadata = lora.read_safetensors(path)
    assert metadata == {"lora_config": "{}"}
    np.testing.assert_array_equal(tensors["a"], values)
    np.testing.assert_allclose(tensors["b"], values, rtol=1e-3)
