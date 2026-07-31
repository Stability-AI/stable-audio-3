import numpy as np
import pytest

from optimized.tensorRT.build.decoder_output import (
    UNBOUNDED_PCM_OUTPUT,
    remove_output_hard_clip,
    rewrite_decoder_onnx,
)

onnx = pytest.importorskip("onnx")
TensorProto = onnx.TensorProto
helper = onnx.helper
numpy_helper = onnx.numpy_helper


def _decoder_tail_model():
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
    graph = helper.make_graph(
        nodes,
        "decoder_tail",
        [audio],
        [pcm],
        initializer=[minimum, maximum, scale],
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
    assert model.graph.output[0].name == UNBOUNDED_PCM_OUTPUT
    mul = next(node for node in model.graph.node if node.op_type == "Mul")
    assert mul.input[0] == "audio"
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
    assert rewritten.graph.output[0].name == UNBOUNDED_PCM_OUTPUT
    onnx.checker.check_model(rewritten)


def test_decoder_rewrite_preserves_out_of_range_sample_ratios():
    onnxruntime = pytest.importorskip("onnxruntime")
    model = _decoder_tail_model()
    remove_output_hard_clip(model)
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    audio = np.array(
        [[[0.0, 1.0, 1.25, 1.75], [0.0, -1.0, -1.25, -1.75]]],
        dtype=np.float32,
    )

    pcm = session.run([UNBOUNDED_PCM_OUTPUT], {"audio": audio})[0]

    assert pcm[0, -1, 0] > 32767
    assert pcm[0, -2, 0] < pcm[0, -1, 0]
    ratio = pcm[0, -2, 0] / pcm[0, -1, 0]
    assert ratio == pytest.approx(1.25 / 1.75, abs=1e-4)
