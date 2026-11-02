"""ONNX graph transforms that make the official Stable Audio 3 graphs runnable on DirectML."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SWA_DOMAIN = "samel"
SWA_OP = "diff_attn_swa"
SWA_WINDOW = 17
SWA_HEAD_DIM = 64
CROSS_ATTN_TOKENS = 257
CROSS_ATTN_SCALE_INPUT = "cross_attn_scale"
TOKEN_NOISE_INPUT = "token_noise"
TOKEN_ROWS = "R"
TOKENS_PER_FRAME = 16
AUDIO_OUTPUT = "audio"
_MASK_VALUE = -1.0e9


class _Builder:
    """Appends nodes and constants with unique names derived from a prefix."""

    def __init__(self, graph: onnx.GraphProto, prefix: str):
        self.graph = graph
        self.prefix = prefix
        self.nodes: list[onnx.NodeProto] = []
        self._count = 0

    def _name(self, tag: str) -> str:
        self._count += 1
        return f"{self.prefix}/{tag}_{self._count}"

    def const(self, value, dtype=np.int64) -> str:
        name = self._name("const")
        self.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, op_type: str, inputs: list[str], outputs: int | list[str] = 1, **attrs) -> list[str]:
        name = self._name(op_type)
        if isinstance(outputs, int):
            outputs = [f"{name}_out{i}" for i in range(outputs)]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=name, **attrs))
        return outputs

    def one(self, op_type: str, inputs: list[str], **attrs) -> str:
        return self.op(op_type, inputs, 1, **attrs)[0]


def _producers(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output}


def _consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    table: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            table.setdefault(name, []).append(node)
    return table


def _scalar_value(graph: onnx.GraphProto, name: str, producers) -> float | None:
    """Value of a scalar tensor defined by an initializer, a Constant, or a Cast of either."""
    for init in graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    node = producers.get(name)
    if node is None:
        return None
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                arr = numpy_helper.to_array(attr.t)
                return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    if node.op_type == "Cast":
        return _scalar_value(graph, node.input[0], producers)
    return None


def _set_nodes(graph: onnx.GraphProto, nodes: list[onnx.NodeProto]) -> None:
    del graph.node[:]
    graph.node.extend(nodes)


def _index_of(nodes: list[onnx.NodeProto], output: str) -> int:
    return next(i for i, node in enumerate(nodes) if output in node.output)


def prune(model: onnx.ModelProto) -> None:
    """Drop nodes and initializers that no graph output depends on."""
    graph = model.graph
    nodes = list(graph.node)
    producer = {out: i for i, node in enumerate(nodes) for out in node.output}
    live: set[int] = set()
    needed = [out.name for out in graph.output]
    seen: set[str] = set()
    while needed:
        name = needed.pop()
        if name in seen:
            continue
        seen.add(name)
        index = producer.get(name)
        if index is not None and index not in live:
            live.add(index)
            needed.extend(i for i in nodes[index].input if i)
    kept = [node for i, node in enumerate(nodes) if i in live]
    _set_nodes(graph, kept)
    used = {i for node in kept for i in node.input} | {out.name for out in graph.output}
    initializers = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)


def add_cross_attn_scale(model: onnx.ModelProto) -> None:
    """Add a `cross_attn_scale` [1, 257, 1] input that multiplies the cross-attention conditioning.

    The baked DiT conditioner always emits learned padding for masked prompt tokens plus the
    seconds token. The reference CFG null branch instead feeds an all-zero cross-attention
    sequence, and a negative prompt zeroes its padded tokens. Scaling the conditioning per token
    right before `to_cond_embed` reproduces both cases exactly.
    """
    graph = model.graph
    if any(inp.name == CROSS_ATTN_SCALE_INPUT for inp in graph.input):
        return
    weight_name = "dit.to_cond_embed.0.weight"
    weight = next((init for init in graph.initializer if init.name == weight_name), None)
    if weight is None:
        raise ValueError(f"{weight_name} not found; not a Stable Audio 3 DiT graph")
    consumers = _consumers(graph)
    transpose = next(n for n in consumers[weight_name] if n.op_type == "Transpose")
    matmul = next(n for n in consumers[transpose.output[0]] if n.op_type == "MatMul")
    data = next(i for i in matmul.input if i != transpose.output[0])

    b = _Builder(graph, "/cross_attn_scale")
    scale = b.one("Cast", [CROSS_ATTN_SCALE_INPUT], to=weight.data_type)
    scaled = b.one("Mul", [data, scale])
    matmul.input[list(matmul.input).index(data)] = scaled
    nodes = list(graph.node)
    index = _index_of(nodes, matmul.output[0])
    nodes[index:index] = b.nodes
    _set_nodes(graph, nodes)
    graph.input.append(helper.make_tensor_value_info(
        CROSS_ATTN_SCALE_INPUT, TensorProto.FLOAT, [1, CROSS_ATTN_TOKENS, 1]))


def remove_random_normal(model: onnx.ModelProto) -> int:
    """Remove the decoder bottleneck dither branch: signal + RandomNormalLike(signal) * k -> signal."""
    graph = model.graph
    removed = 0
    while True:
        nodes = list(graph.node)
        start = next((i for i, n in enumerate(nodes) if n.op_type == "RandomNormalLike"), None)
        if start is None:
            return removed
        signal = nodes[start].input[0]
        chain, current = [start], nodes[start].output[0]
        while True:
            users = [i for i, n in enumerate(nodes) if current in n.input]
            if len(users) != 1:
                raise ValueError(f"unexpected fan-out in the dither branch at {current}")
            chain.append(users[0])
            if nodes[users[0]].op_type == "Add":
                break
            current = nodes[users[0]].output[0]
        add = nodes[chain[-1]]
        if signal not in add.input:
            raise ValueError("dither branch is not added back onto its own signal")
        for node in nodes:
            for i, name in enumerate(node.input):
                if name == add.output[0]:
                    node.input[i] = signal
        dead = set(chain)
        _set_nodes(graph, [n for i, n in enumerate(nodes) if i not in dead])
        removed += 1


def add_token_noise(model: onnx.ModelProto, tokens_per_frame: int = TOKENS_PER_FRAME) -> bool:
    """Add a `token_noise` [R, tokens_per_frame, C] input that is added to the learned SAME tokens.

    The reference SAME blocks perturb these tokens with Gaussian noise (`mask_noise`) at
    inference too; the exported graphs dropped it. The decoder appends 16 tokens per latent
    frame, the encoder one; R is the latent length after the block's own chunk padding.
    The caller passes noise already scaled.
    """
    graph = model.graph
    if any(inp.name == TOKEN_NOISE_INPUT for inp in graph.input):
        return True
    tokens = {init.name: init for init in graph.initializer if init.name.endswith("new_tokens")}
    nodes = list(graph.node)
    position = next((i for i, n in enumerate(nodes) if n.op_type == "Expand" and n.input[0] in tokens), None)
    if position is None:
        return False
    expand = nodes[position]
    init = tokens[expand.input[0]]
    b = _Builder(graph, "/token_noise")
    noise = b.one("Reshape", [TOKEN_NOISE_INPUT, b.one("Shape", [expand.output[0]])])
    noise = b.one("Cast", [noise], to=init.data_type)
    noisy = b.one("Add", [expand.output[0], noise])
    for node in nodes:
        for i, name in enumerate(node.input):
            if name == expand.output[0] and node is not expand:
                node.input[i] = noisy
    nodes[position + 1:position + 1] = b.nodes
    _set_nodes(graph, nodes)
    graph.input.append(helper.make_tensor_value_info(
        TOKEN_NOISE_INPUT, TensorProto.FLOAT, [TOKEN_ROWS, tokens_per_frame, int(init.dims[-1])]))
    return True


def zero_padded_frames(model: onnx.ModelProto, latent_input: str = "latent") -> bool:
    """Zero the projected features of the frames a SAME decoder adds to fill its attention chunks.

    The exported graph edge-pads the latent input before the latent projection; the reference
    decoder pads the projected sequence with zeros. A frame mask after the projection restores
    that. Returns False when the graph does not pad its input.
    """
    graph = model.graph
    pad = next((n for n in graph.node if n.op_type == "Pad" and n.input[0] == latent_input), None)
    if pad is None:
        return False
    initializers = {i.name for i in graph.initializer}
    producers = _producers(graph)
    consumers = _consumers(graph)
    frontier, seen, matmul = [pad.output[0]], set(), None
    while frontier and matmul is None:
        name = frontier.pop(0)
        for node in consumers.get(name, []):
            if id(node) in seen:
                continue
            seen.add(id(node))
            weight = [i for i in node.input if i != name and i in producers
                      and producers[i].op_type == "Transpose" and producers[i].input[0] in initializers]
            if node.op_type == "MatMul" and weight:
                matmul = node
                break
            frontier.extend(node.output)
    if matmul is None:
        raise ValueError("latent projection after the input padding not found")
    add = next(n for n in consumers[matmul.output[0]] if n.op_type == "Add")
    features = add.output[0]
    users = consumers.get(features, [])

    b = _Builder(graph, "/zero_padded_frames")
    length = b.one("Gather", [b.one("Shape", [latent_input]), b.const([2])], axis=0)
    padded = b.one("Squeeze", [b.one("Gather", [b.one("Shape", [features]), b.const([1])], axis=0)])
    frames = b.one("Range", [b.const(0), padded, b.const(1)])
    keep = b.one("CastLike", [b.one("Less", [frames, length]), features])
    keep = b.one("Reshape", [keep, b.const([1, -1, 1])])
    masked = b.one("Mul", [features, keep])
    for node in users:
        for i, name in enumerate(node.input):
            if name == features:
                node.input[i] = masked
    nodes = list(graph.node)
    index = _index_of(nodes, features) + 1
    nodes[index:index] = b.nodes
    _set_nodes(graph, nodes)
    return True


def name_input_dims(model: onnx.ModelProto, name: str, dims: list[int | str]) -> None:
    """Give a graph input's dimensions fixed sizes or the given symbolic names."""
    value = next(i for i in model.graph.input if i.name == name)
    shape = value.type.tensor_type.shape
    if len(shape.dim) != len(dims):
        raise ValueError(f"{name} has rank {len(shape.dim)}, expected {len(dims)}")
    for dim, size in zip(shape.dim, dims):
        if isinstance(size, str):
            dim.dim_param = size
        else:
            dim.dim_value = size


def expose_float_audio(model: onnx.ModelProto) -> None:
    """Replace the int32 PCM output with the clamped float32 waveform [1, 2, T]."""
    graph = model.graph
    if any(out.name == AUDIO_OUTPUT for out in graph.output):
        return
    producers = _producers(graph)
    consumers = _consumers(graph)
    clip = None
    for node in graph.node:
        if node.op_type != "Clip" or len(node.input) < 3:
            continue
        lo = _scalar_value(graph, node.input[1], producers)
        hi = _scalar_value(graph, node.input[2], producers)
        if lo != -1.0 or hi != 1.0:
            continue
        for user in consumers.get(node.output[0], []):
            if user.op_type == "Mul" and any(
                    _scalar_value(graph, i, producers) == 32767.0 for i in user.input):
                clip = node
    if clip is None:
        raise ValueError("PCM conversion tail not found in decoder graph")
    graph.node.append(helper.make_node("Identity", [clip.output[0]], [AUDIO_OUTPUT], name="/audio_out"))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(AUDIO_OUTPUT, TensorProto.FLOAT, [1, 2, "T"]))
    prune(model)


def _swa_subgraph(b: _Builder, q: str, k: str, v: str, out: str, num_heads: int) -> None:
    """Differential sliding-window attention with standard ops.

    q, k, v: [B, N, 2H, D] float32 with N a multiple of 17. Each query attends to keys within
    +/-17 positions, softmax(q.k / sqrt(D)) v per head, and the output is the first H heads
    minus the last H heads: [B, N, H, D]. Queries are tiled in blocks of 17, so every block
    only needs its own block and its two neighbours as keys (51 positions). Every MatMul and
    Softmax stays 4-D (heads and blocks folded together) because DirectML GEMM is 4-D at most.
    """
    blk, d, heads2 = SWA_WINDOW, SWA_HEAD_DIM, 2 * num_heads
    perm_bhnd = [0, 2, 1, 3]
    block_shape = b.const([0, -1, blk, d])
    end = np.iinfo(np.int64).max

    def windows(x: str, axis: int, shape: str, concat_axis: int) -> str:
        axes = b.const([axis])
        prev = b.one("Slice", [x, b.const([0]), b.const([-2 * blk]), axes])
        cur = b.one("Slice", [x, b.const([blk]), b.const([-blk]), axes])
        nxt = b.one("Slice", [x, b.const([2 * blk]), b.const([end]), axes])
        return b.one("Concat", [b.one("Reshape", [t, shape]) for t in (prev, cur, nxt)],
                     axis=concat_axis)

    def keys(x: str) -> str:
        x = b.one("Transpose", [x], perm=perm_bhnd)
        padded = b.one("Pad", [x, b.const([0, 0, blk, 0, 0, 0, blk, 0])], mode="constant")
        return windows(padded, 2, block_shape, 2)

    q4 = b.one("Reshape", [b.one("Transpose", [q], perm=perm_bhnd), block_shape])
    k4 = keys(k)
    v4 = keys(v)

    scores = b.one("MatMul", [q4, b.one("Transpose", [k4], perm=[0, 1, 3, 2])])
    scores = b.one("Mul", [scores, b.const(np.float32(d ** -0.5), np.float32)])

    i = np.arange(blk)[:, None]
    j = np.arange(3 * blk)[None, :]
    band = np.where((j >= i) & (j <= i + 2 * blk), 0.0, _MASK_VALUE).astype(np.float32)
    scores = b.one("Add", [scores, b.const(band, np.float32)])

    length = b.one("Gather", [b.one("Shape", [q]), b.const([1])], axis=0)
    edge = b.one("ConstantOfShape", [length], value=numpy_helper.from_array(np.zeros(1, np.float32)))
    edge = b.one("Pad", [edge, b.const([blk, blk]), b.const(np.float32(_MASK_VALUE), np.float32)],
                 mode="constant")
    edge = windows(edge, 0, b.const([-1, blk]), 1)
    edge = b.one("Tile", [edge, b.const([heads2, 1])])
    scores = b.one("Add", [scores, b.one("Unsqueeze", [edge, b.const([1])])])

    probs = b.one("Softmax", [scores], axis=-1)
    attn = b.one("MatMul", [probs, v4])
    attn = b.one("Reshape", [attn, b.const([0, heads2, -1, d])])

    axis1 = b.const([1])
    first = b.one("Slice", [attn, b.const([0]), b.const([num_heads]), axis1])
    second = b.one("Slice", [attn, b.const([num_heads]), b.const([heads2]), axis1])
    diff = b.one("Sub", [first, second])
    b.op("Transpose", [diff], [out], perm=perm_bhnd)


def replace_swa_attention(model: onnx.ModelProto) -> int:
    """Replace every `samel::diff_attn_swa` TensorRT plugin node with standard ONNX ops."""
    graph = model.graph
    count = 0
    while True:
        nodes = list(graph.node)
        position = next((i for i, n in enumerate(nodes)
                         if n.domain == SWA_DOMAIN and n.op_type == SWA_OP), None)
        if position is None:
            break
        node = nodes[position]
        heads = next(helper.get_attribute_value(a) for a in node.attribute if a.name == "num_heads")
        b = _Builder(graph, f"/swa_{count}")
        _swa_subgraph(b, node.input[0], node.input[1], node.input[2], node.output[0], int(heads))
        nodes[position:position + 1] = b.nodes
        _set_nodes(graph, nodes)
        count += 1
    if count:
        imports = [op for op in model.opset_import if op.domain != SWA_DOMAIN]
        del model.opset_import[:]
        model.opset_import.extend(imports)
    return count


def set_initializer(init: onnx.TensorProto, arr: np.ndarray) -> None:
    """Replace an initializer's data with `arr`, cast to the initializer's own dtype."""
    dtype = helper.tensor_dtype_to_np_dtype(init.data_type)
    arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
    if list(arr.shape) != list(init.dims):
        raise ValueError(f"shape mismatch for {init.name}: {list(arr.shape)} vs {list(init.dims)}")
    init.CopyFrom(numpy_helper.from_array(arr, init.name))
