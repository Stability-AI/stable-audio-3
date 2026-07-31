"""ONNX rewrite for decoder outputs that preserves out-of-range amplitudes."""

from __future__ import annotations

import os
import tempfile
import uuid
import warnings
from pathlib import Path


UNBOUNDED_AUDIO_OUTPUT = "audio_unbounded"
_FP32_CAST_NAME = "PeakProtectAudioOutputFP32"
_FP32_CAST_OUTPUT = "audio_unbounded_channels_first_fp32"


def _producer_map(model):
    return {output: node for node in model.graph.node for output in node.output}


def _constant_scalar(model, tensor_name: str):
    from onnx import numpy_helper

    initializer_by_name = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    initializer = initializer_by_name.get(tensor_name)
    if initializer is not None:
        value = numpy_helper.to_array(initializer)
        return float(value.reshape(-1)[0]) if value.size == 1 else None

    producer = _producer_map(model).get(tensor_name)
    if producer is None:
        return None
    if producer.op_type == "Cast":
        return _constant_scalar(model, producer.input[0])
    if producer.op_type != "Constant":
        return None
    value_attr = next(
        (attribute for attribute in producer.attribute if attribute.name == "value"),
        None,
    )
    if value_attr is None:
        return None
    value = numpy_helper.to_array(value_attr.t)
    return float(value.reshape(-1)[0]) if value.size == 1 else None


def _attribute_ints(node, name: str):
    attribute = next((item for item in node.attribute if item.name == name), None)
    return tuple(attribute.ints) if attribute is not None else None


def _attribute_int(node, name: str):
    attribute = next((item for item in node.attribute if item.name == name), None)
    return int(attribute.i) if attribute is not None else None


def _find_pcm_tail(model, output_name: str):
    """Return the verified Transpose <- Cast(INT32) <- Mul PCM tail."""
    import numpy as np
    from onnx import TensorProto

    graph_output = next(
        (output for output in model.graph.output if output.name == output_name), None
    )
    if graph_output is None:
        raise RuntimeError(f"decoder ONNX has no {output_name!r} graph output")

    producer_by_output = _producer_map(model)
    output_tensor = graph_output.name
    output_producer = producer_by_output.get(output_tensor)
    output_identity = None
    if output_producer is not None and output_producer.op_type == "Identity":
        output_identity = output_producer
        output_tensor = output_producer.input[0]

    transpose = producer_by_output.get(output_tensor)
    if transpose is None or transpose.op_type != "Transpose":
        raise RuntimeError("decoder PCM output is not produced by Transpose")
    if _attribute_ints(transpose, "perm") != (0, 2, 1):
        raise RuntimeError(
            "decoder PCM Transpose does not use the expected [0, 2, 1] perm"
        )

    cast = producer_by_output.get(transpose.input[0])
    if (
        cast is None
        or cast.op_type != "Cast"
        or _attribute_int(cast, "to") != TensorProto.INT32
    ):
        raise RuntimeError("decoder PCM Transpose is not fed by Cast(to=INT32)")

    multiply = producer_by_output.get(cast.input[0])
    if multiply is None or multiply.op_type != "Mul" or len(multiply.input) != 2:
        raise RuntimeError("decoder PCM Cast is not fed by the expected scale Mul")

    scalar_inputs = [
        (index, _constant_scalar(model, name))
        for index, name in enumerate(multiply.input)
    ]
    scale_inputs = [
        (index, value)
        for index, value in scalar_inputs
        if value is not None and np.isclose(value, 32767.0, rtol=0.0, atol=1.0)
    ]
    if len(scale_inputs) != 1:
        raise RuntimeError(
            "decoder PCM Mul does not have exactly one 32767 scale input"
        )
    scale_index, scale = scale_inputs[0]
    signal_index = 1 - scale_index
    return {
        "output": graph_output,
        "output_identity": output_identity,
        "transpose": transpose,
        "cast": cast,
        "multiply": multiply,
        "signal_index": signal_index,
        "scale_index": scale_index,
        "scale": scale,
        "producer_by_output": producer_by_output,
    }


def _find_unbounded_audio_tail(model):
    """Return the verified sample-major floating-point output Transpose."""
    graph_output = next(
        (
            output
            for output in model.graph.output
            if output.name == UNBOUNDED_AUDIO_OUTPUT
        ),
        None,
    )
    if graph_output is None:
        raise RuntimeError(
            f"decoder ONNX has no {UNBOUNDED_AUDIO_OUTPUT!r} graph output"
        )

    producer_by_output = _producer_map(model)
    output_tensor = graph_output.name
    output_producer = producer_by_output.get(output_tensor)
    if output_producer is not None and output_producer.op_type == "Identity":
        output_tensor = output_producer.input[0]
    transpose = producer_by_output.get(output_tensor)
    if transpose is None or transpose.op_type != "Transpose":
        raise RuntimeError("unbounded decoder audio is not produced by Transpose")
    if _attribute_ints(transpose, "perm") != (0, 2, 1):
        raise RuntimeError(
            "unbounded decoder audio Transpose does not use the expected [0, 2, 1] perm"
        )
    return {
        "output": graph_output,
        "transpose": transpose,
        "producer_by_output": producer_by_output,
    }


def _clip_bounds(model, clip) -> tuple[float | None, float | None]:
    minimum = _constant_scalar(model, clip.input[1]) if len(clip.input) > 1 else None
    maximum = _constant_scalar(model, clip.input[2]) if len(clip.input) > 2 else None
    for attribute in clip.attribute:
        if attribute.name == "min":
            minimum = float(attribute.f)
        elif attribute.name == "max":
            maximum = float(attribute.f)
    return minimum, maximum


def remove_output_hard_clip(model) -> int:
    """Remove only the verified final Clip and expose float sample-major audio.

    The destructive scale and integer cast are removed with the Clip. Runtime
    code applies the shared no-boost attenuation policy to floating-point audio,
    then scales and narrows to INT16. Returns the number of removed Clip nodes.
    """
    import numpy as np
    from onnx import TensorProto, helper

    if any(output.name == UNBOUNDED_AUDIO_OUTPUT for output in model.graph.output):
        _find_unbounded_audio_tail(model)
        return 0

    tail = _find_pcm_tail(model, "pcm")
    multiply = tail["multiply"]
    signal_index = tail["signal_index"]
    producer_by_output = tail["producer_by_output"]
    clip = producer_by_output.get(multiply.input[signal_index])
    if clip is None or clip.op_type != "Clip":
        raise RuntimeError(
            "decoder PCM scale Mul is not fed directly by the expected output Clip"
        )

    minimum, maximum = _clip_bounds(model, clip)
    if minimum is None or maximum is None:
        raise RuntimeError("could not resolve decoder output Clip bounds")
    if not np.isclose(minimum, -1.0) or not np.isclose(maximum, 1.0):
        raise RuntimeError(
            f"refusing to remove unexpected decoder Clip bounds [{minimum}, {maximum}]"
        )

    clipped_output = clip.output[0]
    consumers = [node for node in model.graph.node if clipped_output in node.input]
    if consumers != [multiply]:
        raise RuntimeError(
            "decoder output Clip is shared; refusing to remove a semantic graph node"
        )

    multiply_consumers = [
        node for node in model.graph.node if multiply.output[0] in node.input
    ]
    if multiply_consumers != [tail["cast"]]:
        raise RuntimeError(
            "decoder PCM scale is shared; refusing to remove a semantic graph node"
        )
    cast_consumers = [
        node for node in model.graph.node if tail["cast"].output[0] in node.input
    ]
    if cast_consumers != [tail["transpose"]]:
        raise RuntimeError(
            "decoder PCM Cast is shared; refusing to remove a semantic graph node"
        )

    existing_node_names = {node.name for node in model.graph.node}
    existing_tensor_names = {tensor.name for tensor in model.graph.initializer} | {
        output for node in model.graph.node for output in node.output
    }
    if (
        _FP32_CAST_NAME in existing_node_names
        or _FP32_CAST_OUTPUT in existing_tensor_names
    ):
        raise RuntimeError("decoder graph already uses reserved unbounded-audio names")

    for node in (clip, multiply, tail["cast"]):
        model.graph.node.remove(node)
    transpose = tail["transpose"]
    transpose_index = list(model.graph.node).index(transpose)
    model.graph.node.insert(
        transpose_index,
        helper.make_node(
            "Cast",
            inputs=[clip.input[0]],
            outputs=[_FP32_CAST_OUTPUT],
            name=_FP32_CAST_NAME,
            to=TensorProto.FLOAT,
        ),
    )
    transpose.input[0] = _FP32_CAST_OUTPUT

    terminal = (
        tail["output_identity"] if tail["output_identity"] is not None else transpose
    )
    terminal.output[0] = UNBOUNDED_AUDIO_OUTPUT
    tail["output"].name = UNBOUNDED_AUDIO_OUTPUT
    tail["output"].type.tensor_type.elem_type = TensorProto.FLOAT
    return 1


def force_unbounded_audio_output_fp32(model) -> int:
    """Restore the explicit FP32 output boundary after mixed-precision conversion."""
    from onnx import TensorProto, helper

    tail = _find_unbounded_audio_tail(model)
    transpose = tail["transpose"]
    signal_input = transpose.input[0]
    producer = tail["producer_by_output"].get(signal_input)
    if producer is not None and producer.name == _FP32_CAST_NAME:
        if producer.op_type != "Cast":
            raise RuntimeError("reserved unbounded-audio node is not a Cast")
        to_attribute = next(
            (item for item in producer.attribute if item.name == "to"), None
        )
        if to_attribute is None:
            raise RuntimeError("unbounded-audio Cast has no target dtype")
        if to_attribute.i == TensorProto.FLOAT:
            return 0
        to_attribute.i = TensorProto.FLOAT
        tail["output"].type.tensor_type.elem_type = TensorProto.FLOAT
        return 1

    existing_node_names = {node.name for node in model.graph.node}
    existing_tensor_names = {tensor.name for tensor in model.graph.initializer} | {
        output for node in model.graph.node for output in node.output
    }
    if (
        _FP32_CAST_NAME in existing_node_names
        or _FP32_CAST_OUTPUT in existing_tensor_names
    ):
        raise RuntimeError("decoder graph already uses reserved unbounded-audio names")

    transpose_index = list(model.graph.node).index(transpose)
    model.graph.node.insert(
        transpose_index,
        helper.make_node(
            "Cast",
            inputs=[signal_input],
            outputs=[_FP32_CAST_OUTPUT],
            name=_FP32_CAST_NAME,
            to=TensorProto.FLOAT,
        ),
    )
    transpose.input[0] = _FP32_CAST_OUTPUT
    tail["output"].type.tensor_type.elem_type = TensorProto.FLOAT
    return 1


def rewrite_decoder_onnx(input_path: str, output_path: str) -> str:
    """Write a decoder ONNX exposing sample-major FP32 audio without clipping."""
    import onnx

    model = onnx.load(input_path, load_external_data=True)
    removed = remove_output_hard_clip(model)
    force_unbounded_audio_output_fp32(model)
    onnx.checker.check_model(model)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_prefix = output.name + ".data"
    old_sidecars = set()
    if output.exists():
        old_model = onnx.load(str(output), load_external_data=False)
        for initializer in old_model.graph.initializer:
            for entry in initializer.external_data:
                if entry.key != "location":
                    continue
                location = entry.value
                if location == sidecar_prefix or location.startswith(
                    sidecar_prefix + "."
                ):
                    old_sidecars.add(output.parent / location)

    sidecar_name = f"{sidecar_prefix}.{uuid.uuid4().hex}"
    sidecar = output.parent / sidecar_name
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as staging_dir:
        staged_output = Path(staging_dir) / output.name
        staged_sidecar = Path(staging_dir) / sidecar_name
        onnx.save_model(
            model,
            str(staged_output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar_name,
            size_threshold=1024 * 1024,
        )
        sidecar_published = False
        if staged_sidecar.exists():
            os.replace(staged_sidecar, sidecar)
            sidecar_published = True
        try:
            os.replace(staged_output, output)
        except BaseException:
            if sidecar_published:
                sidecar.unlink(missing_ok=True)
            raise

    for old_sidecar in old_sidecars - {sidecar}:
        try:
            old_sidecar.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"could not remove superseded ONNX sidecar {old_sidecar}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    print(
        f"  decoder peak policy: removed {removed} hard Clip; "
        f"output binding -> {UNBOUNDED_AUDIO_OUTPUT}",
        flush=True,
    )
    return str(output)
