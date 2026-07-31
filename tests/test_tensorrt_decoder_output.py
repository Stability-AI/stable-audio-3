import os
from unittest.mock import patch

import numpy as np
import pytest

from optimized.tensorRT.build.decoder_output import (
    UNBOUNDED_AUDIO_OUTPUT,
    force_unbounded_audio_output_fp32,
    remove_output_hard_clip,
    rewrite_decoder_onnx,
)

onnx = pytest.importorskip("onnx")
TensorProto = onnx.TensorProto
helper = onnx.helper
numpy_helper = onnx.numpy_helper


def _decoder_tail_model(*, large_initializer_value=None):
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 2, 4])
    pcm = helper.make_tensor_value_info("pcm", TensorProto.INT32, [1, 4, 2])
    minimum = numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "minimum")
    maximum = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "maximum")
    scale = numpy_helper.from_array(np.array(32767.0, dtype=np.float32), "scale")
    nodes = [
        helper.make_node("Clip", ["audio", "minimum", "maximum"], ["clipped"]),
        helper.make_node("Mul", ["clipped", "scale"], ["scaled"]),
        helper.make_node(
            "Cast", ["scaled"], ["pcm_channels_first"], to=TensorProto.INT32
        ),
        helper.make_node("Transpose", ["pcm_channels_first"], ["pcm"], perm=[0, 2, 1]),
    ]
    initializers = [minimum, maximum, scale]
    if large_initializer_value is not None:
        initializers.append(
            numpy_helper.from_array(
                np.full(300_000, large_initializer_value, dtype=np.float32),
                "large_external_weight",
            )
        )
    graph = helper.make_graph(
        nodes,
        "decoder_tail",
        [audio],
        [pcm],
        initializer=initializers,
    )
    return helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[helper.make_opsetid("", 17)],
    )


def test_decoder_rewrite_removes_clip_and_marks_unbounded_output():
    model = _decoder_tail_model()

    assert remove_output_hard_clip(model) == 1

    assert all(node.op_type != "Clip" for node in model.graph.node)
    assert model.graph.output[0].name == UNBOUNDED_AUDIO_OUTPUT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(node.op_type != "Mul" for node in model.graph.node)
    assert all(
        node.op_type != "Cast"
        or next(attr.i for attr in node.attribute if attr.name == "to")
        != TensorProto.INT32
        for node in model.graph.node
    )
    onnx.checker.check_model(model)


def test_decoder_rewrite_is_idempotent():
    model = _decoder_tail_model()
    remove_output_hard_clip(model)

    assert remove_output_hard_clip(model) == 0


def test_decoder_rewrite_writes_loadable_onnx(tmp_path):
    source = tmp_path / "decoder.onnx"
    output = tmp_path / "decoder_unbounded.onnx"
    onnx.save(_decoder_tail_model(), source)

    assert rewrite_decoder_onnx(str(source), str(output)) == str(output)

    rewritten = onnx.load(output)
    assert rewritten.graph.output[0].name == UNBOUNDED_AUDIO_OUTPUT
    transpose = next(
        node for node in rewritten.graph.node if node.op_type == "Transpose"
    )
    signal_producer = next(
        node for node in rewritten.graph.node if transpose.input[0] in node.output
    )
    assert signal_producer.op_type == "Cast"
    assert (
        next(attr.i for attr in signal_producer.attribute if attr.name == "to")
        == TensorProto.FLOAT
    )
    assert all(node.op_type != "Mul" for node in rewritten.graph.node)
    onnx.checker.check_model(rewritten)


def test_decoder_rewrite_preserves_out_of_range_sample_ratios():
    onnxruntime = pytest.importorskip("onnxruntime")
    model = _decoder_tail_model()
    remove_output_hard_clip(model)
    force_unbounded_audio_output_fp32(model)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    audio = np.array(
        [[[0.0, 1.0, 1.25, 1.75], [0.0, -1.0, -1.25, -1.75]]],
        dtype=np.float32,
    )

    unbounded = session.run([UNBOUNDED_AUDIO_OUTPUT], {"audio": audio})[0]

    assert unbounded[0, -1, 0] == 1.75
    assert unbounded[0, -2, 0] < unbounded[0, -1, 0]
    ratio = unbounded[0, -2, 0] / unbounded[0, -1, 0]
    assert ratio == pytest.approx(1.25 / 1.75, abs=1e-4)


def test_decoder_rewrite_preserves_nonfinite_and_extreme_float_values():
    onnxruntime = pytest.importorskip("onnxruntime")
    model = _decoder_tail_model()
    remove_output_hard_clip(model)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    audio = np.array(
        [[[np.nan, np.inf, -np.inf, 70_000.0], [0.0, 1.0, -1.0, -70_000.0]]],
        dtype=np.float32,
    )

    unbounded = session.run([UNBOUNDED_AUDIO_OUTPUT], {"audio": audio})[0]

    assert np.isnan(unbounded[0, 0, 0])
    assert np.isposinf(unbounded[0, 1, 0])
    assert np.isneginf(unbounded[0, 2, 0])
    assert unbounded[0, 3, 0] == 70_000.0
    assert unbounded[0, 3, 1] == -70_000.0


def test_decoder_rewrite_refuses_an_unrelated_upstream_clip():
    model = _decoder_tail_model()
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
    model.graph.initializer.append(zero)
    multiply = next(node for node in model.graph.node if node.op_type == "Mul")
    multiply.input[0] = "processed"
    clip_index = next(
        index for index, node in enumerate(model.graph.node) if node.op_type == "Clip"
    )
    model.graph.node.insert(
        clip_index + 1,
        helper.make_node("Add", ["clipped", "zero"], ["processed"]),
    )

    with pytest.raises(RuntimeError, match="not fed directly"):
        remove_output_hard_clip(model)


def test_fp16_mixed_output_is_promoted_to_fp32():
    onnxruntime = pytest.importorskip("onnxruntime")
    model = _decoder_tail_model()
    remove_output_hard_clip(model)
    model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    output_cast = next(node for node in model.graph.node if node.op_type == "Cast")
    next(
        attr for attr in output_cast.attribute if attr.name == "to"
    ).i = TensorProto.FLOAT16

    assert force_unbounded_audio_output_fp32(model) == 1
    assert force_unbounded_audio_output_fp32(model) == 0
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    audio = np.array(
        [[[0.0, 2.0, 2.5, 3.0], [0.0, -2.0, -2.5, -3.0]]],
        dtype=np.float16,
    )

    unbounded = session.run([UNBOUNDED_AUDIO_OUTPUT], {"audio": audio})[0]

    assert unbounded.dtype == np.float32
    assert unbounded[0, -1, 0] == 3.0
    assert unbounded[0, -1, 1] == -3.0


def test_rewrite_replaces_external_sidecar_instead_of_appending(tmp_path):
    source = tmp_path / "decoder.onnx"
    output = tmp_path / "decoder_unbounded.onnx"
    onnx.save(_decoder_tail_model(large_initializer_value=1.0), source)

    rewrite_decoder_onnx(str(source), str(output))
    first_sidecar = _external_sidecar(output)
    first_size = first_sidecar.stat().st_size
    rewrite_decoder_onnx(str(source), str(output))
    second_sidecar = _external_sidecar(output)

    assert second_sidecar != first_sidecar
    assert second_sidecar.stat().st_size == first_size
    assert not first_sidecar.exists()
    onnx.checker.check_model(onnx.load(output, load_external_data=True))


def _external_sidecar(model_path):
    model = onnx.load(model_path, load_external_data=False)
    locations = {
        entry.value
        for initializer in model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    assert len(locations) == 1
    return model_path.parent / locations.pop()


def test_failed_model_swap_keeps_previous_model_and_sidecar_consistent(tmp_path):
    first_source = tmp_path / "decoder_v1.onnx"
    second_source = tmp_path / "decoder_v2.onnx"
    output = tmp_path / "decoder_unbounded.onnx"
    onnx.save(_decoder_tail_model(large_initializer_value=1.0), first_source)
    onnx.save(_decoder_tail_model(large_initializer_value=2.0), second_source)
    rewrite_decoder_onnx(str(first_source), str(output))
    first_sidecar = _external_sidecar(output)

    real_replace = os.replace
    replace_calls = 0

    def fail_second_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected model-swap failure")
        return real_replace(source, destination)

    with (
        patch(
            "optimized.tensorRT.build.decoder_output.os.replace",
            side_effect=fail_second_replace,
        ),
        pytest.raises(OSError, match="injected model-swap failure"),
    ):
        rewrite_decoder_onnx(str(second_source), str(output))

    assert _external_sidecar(output) == first_sidecar
    assert first_sidecar.exists()
    loaded = onnx.load(output, load_external_data=True)
    weight = next(
        item
        for item in loaded.graph.initializer
        if item.name == "large_external_weight"
    )
    assert np.all(numpy_helper.to_array(weight) == 1.0)
