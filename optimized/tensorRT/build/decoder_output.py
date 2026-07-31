"""ONNX rewrite for decoder outputs that preserves out-of-range amplitudes."""

from __future__ import annotations

from pathlib import Path


UNBOUNDED_PCM_OUTPUT = "pcm_unbounded"


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
    if output_producer is not None and output_producer.op_type == "Identity":
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
        "transpose": transpose,
        "cast": cast,
        "multiply": multiply,
        "signal_index": signal_index,
        "scale_index": scale_index,
        "scale": scale,
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
    """Remove only the verified final audio Clip and mark PCM unbounded.

    The decoder's existing scale, INT32 cast, and channel transpose stay in
    the graph. Runtime code can then apply the shared no-boost attenuation
    policy before narrowing to INT16. Returns the number of removed Clip nodes.
    """
    import numpy as np
    from onnx import helper

    if any(output.name == UNBOUNDED_PCM_OUTPUT for output in model.graph.output):
        _find_pcm_tail(model, UNBOUNDED_PCM_OUTPUT)
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

    multiply.input[signal_index] = clip.input[0]
    model.graph.node.remove(clip)

    # Give rebuilt engines an explicit binding name. Runtime can distinguish
    # them from legacy `pcm` engines whose destructive Clip is already baked in.
    model.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[tail["output"].name],
            outputs=[UNBOUNDED_PCM_OUTPUT],
            name="ExposeUnboundedPCM",
        )
    )
    tail["output"].name = UNBOUNDED_PCM_OUTPUT
    return 1


def force_unbounded_pcm_tail_fp32(model) -> int:
    """Force unbounded audio scaling to FP32 before the INT32 cast.

    FP16 can only represent finite values through 65504, so an unbounded
    ``audio * 32767`` tail would overflow for peaks just above 2. This inserts
    a stable FP32 boundary and restores the exact 32767 scale after any mixed-
    precision graph conversion. Returns 1 when the graph changed, else 0.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    tail = _find_pcm_tail(model, UNBOUNDED_PCM_OUTPUT)
    multiply = tail["multiply"]
    signal_index = tail["signal_index"]
    scale_index = tail["scale_index"]
    producer_by_output = tail["producer_by_output"]

    scale_name = "peak_protect_pcm16_scale_fp32"
    cast_name = "PeakProtectPCMInputFP32"
    cast_output = "pcm_unbounded_input_fp32"
    signal_input = multiply.input[signal_index]
    signal_producer = producer_by_output.get(signal_input)
    already_cast = (
        signal_producer is not None
        and signal_producer.op_type == "Cast"
        and signal_producer.name == cast_name
        and _attribute_int(signal_producer, "to") == TensorProto.FLOAT
    )
    if already_cast and multiply.input[scale_index] == scale_name:
        return 0

    existing_node_names = {node.name for node in model.graph.node}
    existing_tensor_names = {tensor.name for tensor in model.graph.initializer} | {
        output for node in model.graph.node for output in node.output
    }
    if cast_name in existing_node_names or cast_output in existing_tensor_names:
        raise RuntimeError("decoder graph already uses reserved FP32 PCM tail names")
    if scale_name in existing_tensor_names:
        raise RuntimeError(
            "decoder graph already uses the reserved FP32 PCM scale name"
        )

    cast = helper.make_node(
        "Cast",
        inputs=[signal_input],
        outputs=[cast_output],
        name=cast_name,
        to=TensorProto.FLOAT,
    )
    multiply_index = list(model.graph.node).index(multiply)
    model.graph.node.insert(multiply_index, cast)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(32767.0, dtype=np.float32), scale_name)
    )
    multiply.input[signal_index] = cast_output
    multiply.input[scale_index] = scale_name
    return 1


def rewrite_decoder_onnx(input_path: str, output_path: str) -> str:
    """Write a decoder ONNX whose INT32 PCM output has no baked hard clip."""
    import onnx

    model = onnx.load(input_path, load_external_data=True)
    removed = remove_output_hard_clip(model)
    force_unbounded_pcm_tail_fp32(model)
    onnx.checker.check_model(model)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=1024 * 1024,
    )
    print(
        f"  decoder peak policy: removed {removed} hard Clip; "
        f"output binding -> {UNBOUNDED_PCM_OUTPUT}",
        flush=True,
    )
    return str(output)
