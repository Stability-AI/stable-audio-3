"""ONNX rewrite for decoder outputs that preserves out-of-range amplitudes."""

from __future__ import annotations

from pathlib import Path


UNBOUNDED_PCM_OUTPUT = "pcm_unbounded"


def remove_output_hard_clip(model) -> int:
    """Remove the final audio Clip and mark the PCM output as unbounded.

    The decoder's existing scale, INT32 cast, and channel transpose stay in
    the graph. Runtime code can then apply the shared no-boost attenuation
    policy before narrowing to INT16. Returns the number of removed Clip nodes.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    if any(output.name == UNBOUNDED_PCM_OUTPUT for output in model.graph.output):
        return 0

    pcm_output = next(
        (output for output in model.graph.output if output.name == "pcm"), None
    )
    if pcm_output is None:
        raise RuntimeError("decoder ONNX has no 'pcm' graph output")

    producer_by_output = {
        output: node for node in model.graph.node for output in node.output
    }
    queue = [(pcm_output.name, 0)]
    seen = set()
    clips = []
    while queue:
        tensor_name, distance = queue.pop(0)
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        producer = producer_by_output.get(tensor_name)
        if producer is None:
            continue
        if producer.op_type == "Clip":
            clips.append((distance, producer))
            continue
        queue.extend((input_name, distance + 1) for input_name in producer.input)

    if not clips:
        raise RuntimeError(
            "decoder ONNX output has no upstream Clip; cannot verify peak-policy rewrite"
        )

    _, clip = min(clips, key=lambda item: item[0])

    initializer_by_name = {
        initializer.name: initializer for initializer in model.graph.initializer
    }

    def constant_scalar(tensor_name):
        initializer = initializer_by_name.get(tensor_name)
        if initializer is not None:
            value = numpy_helper.to_array(initializer)
            return float(value.reshape(-1)[0]) if value.size == 1 else None
        producer = producer_by_output.get(tensor_name)
        if producer is None:
            return None
        if producer.op_type == "Cast":
            return constant_scalar(producer.input[0])
        if producer.op_type == "Constant":
            value_attr = next(
                (
                    attribute
                    for attribute in producer.attribute
                    if attribute.name == "value"
                ),
                None,
            )
            if value_attr is not None:
                value = numpy_helper.to_array(value_attr.t)
                return float(value.reshape(-1)[0]) if value.size == 1 else None
        return None

    minimum = constant_scalar(clip.input[1]) if len(clip.input) > 1 else None
    maximum = constant_scalar(clip.input[2]) if len(clip.input) > 2 else None
    for attribute in clip.attribute:
        if attribute.name == "min":
            minimum = float(attribute.f)
        elif attribute.name == "max":
            maximum = float(attribute.f)
    if minimum is None or maximum is None:
        raise RuntimeError("could not resolve decoder output Clip bounds")
    if not np.isclose(minimum, -1.0) or not np.isclose(maximum, 1.0):
        raise RuntimeError(
            f"refusing to remove unexpected decoder Clip bounds [{minimum}, {maximum}]"
        )

    unclipped_input = clip.input[0]
    clipped_output = clip.output[0]
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == clipped_output:
                node.input[index] = unclipped_input
    model.graph.node.remove(clip)

    # Give rebuilt engines an explicit binding name. Runtime can distinguish
    # them from legacy `pcm` engines whose destructive Clip is already baked in.
    model.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[pcm_output.name],
            outputs=[UNBOUNDED_PCM_OUTPUT],
            name="ExposeUnboundedPCM",
        )
    )
    pcm_output.name = UNBOUNDED_PCM_OUTPUT
    return 1


def rewrite_decoder_onnx(input_path: str, output_path: str) -> str:
    """Write a decoder ONNX whose INT32 PCM output has no baked hard clip."""
    import onnx

    model = onnx.load(input_path, load_external_data=True)
    removed = remove_output_hard_clip(model)
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
