import math
import struct
import wave

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from sa3_dml import surgery
from sa3_dml.audio import write_wav
from sa3_dml.codec import chunk_starts
from sa3_dml.sampling import guided_velocity, latent_length, schedule, valid_length

PROVIDERS = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]


def _model(nodes, inputs, outputs, initializers=(), extra_opsets=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    opsets = [helper.make_opsetid("", 17), *extra_opsets]
    return helper.make_model(graph, opset_imports=opsets, ir_version=8)


def _run(model, feeds, provider="CPUExecutionProvider", dims=None):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_mem_pattern = False
    for name, value in (dims or {}).items():
        options.add_free_dimension_override_by_name(name, value)
    session = ort.InferenceSession(model.SerializeToString(), options, providers=[provider])
    return session.run(None, feeds)


def _swa_reference(q, k, v, heads, window=17):
    n = q.shape[1]
    out = np.zeros(q.shape, np.float64)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        s = np.einsum("bhd,bjhd->bhj", q[:, i], k[:, lo:hi]) / math.sqrt(q.shape[-1])
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[:, i] = np.einsum("bhj,bjhd->bhd", p, v[:, lo:hi])
    return out[:, :, :heads] - out[:, :, heads:]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("blocks", [1, 3, 8])
def test_swa_replacement_matches_reference(provider, blocks):
    heads, dim = 4, surgery.SWA_HEAD_DIM
    node = helper.make_node(surgery.SWA_OP, ["q", "k", "v"], ["o"], domain=surgery.SWA_DOMAIN, num_heads=heads)
    io = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, "N", 2 * heads, dim]) for n in "qkv"]
    out = helper.make_tensor_value_info("o", TensorProto.FLOAT, [1, "N", heads, dim])
    model = _model([node], io, [out], extra_opsets=[helper.make_opsetid(surgery.SWA_DOMAIN, 1)])
    assert surgery.replace_swa_attention(model) == 1
    assert all(n.domain != surgery.SWA_DOMAIN for n in model.graph.node)
    rng = np.random.default_rng(0)
    n = 17 * blocks
    q, k, v = (rng.standard_normal((1, n, 2 * heads, dim)).astype(np.float32) for _ in range(3))
    got = _run(model, {"q": q, "k": k, "v": v}, provider, {"N": n})[0]
    np.testing.assert_allclose(got, _swa_reference(q, k, v, heads), atol=1e-4)


def test_cross_attn_scale_zeroes_conditioning():
    weight = numpy_helper.from_array(np.ones((4, 3), np.float32), "dit.to_cond_embed.0.weight")
    nodes = [helper.make_node("Transpose", [weight.name], ["wt"]), helper.make_node("MatMul", ["c", "wt"], ["y"])]
    model = _model(nodes, [helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 257, 3])],
                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 257, 4])], [weight])
    surgery.add_cross_attn_scale(model)
    cond = np.ones((1, 257, 3), np.float32)
    scale = np.ones((1, 257, 1), np.float32)
    scale[0, 5:] = 0.0
    y = _run(model, {"c": cond, surgery.CROSS_ATTN_SCALE_INPUT: scale})[0]
    assert np.all(y[0, :5] == 3.0) and np.all(y[0, 5:] == 0.0)


def _decoder_tail(signal_ops):
    nodes = signal_ops + [
        helper.make_node("Clip", ["s", "lo", "hi"], ["clipped"]),
        helper.make_node("Mul", ["clipped", "k"], ["scaled"]),
        helper.make_node("Cast", ["scaled"], ["pcm_t"], to=TensorProto.INT32),
        helper.make_node("Transpose", ["pcm_t"], ["pcm"], perm=[0, 2, 1]),
    ]
    consts = [numpy_helper.from_array(np.array(v, np.float32), n)
              for n, v in (("lo", -1.0), ("hi", 1.0), ("k", 32767.0))]
    return nodes, consts


def test_decoder_surgery_noise_and_float_output():
    frames, channels = 3, 2
    tokens = numpy_helper.from_array(np.full((1, 1, channels), 0.5, np.float32), "dec.new_tokens")
    shapes = [numpy_helper.from_array(np.array(v, np.int64), n)
              for n, v in (("expand", [frames, surgery.TOKENS_PER_FRAME, channels]), ("flat", [1, 2, -1]))]
    ops = [
        helper.make_node("RandomNormalLike", ["x"], ["rnd"]),
        helper.make_node("Mul", ["rnd", "k"], ["noise"]),
        helper.make_node("Add", ["x", "noise"], ["dithered"]),
        helper.make_node("Expand", ["dec.new_tokens", "expand"], ["expanded"]),
        helper.make_node("Reshape", ["expanded", "flat"], ["tok"]),
        helper.make_node("Concat", ["dithered", "tok"], ["s"], axis=2),
    ]
    nodes, consts = _decoder_tail(ops)
    model = _model(nodes, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "L"])],
                   [helper.make_tensor_value_info("pcm", TensorProto.INT32, None)], [tokens, *shapes, *consts])
    assert surgery.remove_random_normal(model) == 1
    assert surgery.add_token_noise(model)
    surgery.expose_float_audio(model)
    assert [o.name for o in model.graph.output] == [surgery.AUDIO_OUTPUT]
    assert not any(n.op_type == "RandomNormalLike" for n in model.graph.node)
    assert not any(a.name == "to" and a.i == TensorProto.INT32 for n in model.graph.node for a in n.attribute)
    x = np.full((1, 2, frames), 2.0, np.float32)
    noise = np.full((frames, surgery.TOKENS_PER_FRAME, channels), 0.25, np.float32)
    audio = _run(model, {"x": x, surgery.TOKEN_NOISE_INPUT: noise}, dims={"L": frames})[0]
    np.testing.assert_allclose(audio[..., :frames], 1.0)
    np.testing.assert_allclose(audio[..., frames:], 0.75)


def test_schedule_matches_reference_logsnr_shift():
    for sigma_max in (1.0, 0.7):
        t = np.linspace(sigma_max, 0, 9)
        expected = 1 / (1 + np.exp(2.0 - t * 8.2))
        expected[t <= 0] = 0.0
        expected[t >= 1] = 1.0
        expected[0] = sigma_max
        np.testing.assert_allclose(schedule(8, sigma_max), expected, rtol=1e-6)


def test_latent_length_and_valid_region():
    assert latent_length(10, 5292032) == 174
    assert latent_length(30, 5292032) == 388
    assert latent_length(120, 5292032) == 1292
    assert latent_length(380, 16777216) == 4096
    assert valid_length(30, 388) == 387
    assert valid_length(10, 174) == 172


def test_chunk_starts_cover_signal():
    assert chunk_starts(388, 128, 32) == [0, 96, 192, 260]
    assert chunk_starts(128, 128, 32) == [0]


def test_guidance_reduces_to_cfg_and_identity():
    rng = np.random.default_rng(1)
    x, vc, vu = (rng.standard_normal((1, 4, 6)).astype(np.float32) for _ in range(3))
    sigma = 0.6
    plain = guided_velocity(x, sigma, vc, vu, 3.0, 0.0)
    cond, uncond = x - vc * sigma, x - vu * sigma
    np.testing.assert_allclose(plain, (x - (cond + 2.0 * (cond - uncond))) / sigma, rtol=1e-5)
    np.testing.assert_allclose(guided_velocity(x, sigma, vc, vu, 1.0, 1.0), vc, rtol=1e-5)


def test_wav_writer(tmp_path):
    audio = np.stack([np.linspace(-1, 1, 64), np.zeros(64)]).astype(np.float32)
    with wave.open(str(write_wav(tmp_path / "a.wav", audio, 44100, "pcm16"))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, 44100, 64)
    data = write_wav(tmp_path / "b.wav", audio, 44100, "float32").read_bytes()
    tag, channels, rate = struct.unpack("<HHI", data[20:28])
    assert (tag, channels, rate) == (3, 2, 44100)
    samples = np.frombuffer(data[-64 * 8:], "<f4").reshape(64, 2).T
    np.testing.assert_array_equal(samples, audio)


def test_set_initializer_casts_and_checks_shape():
    init = numpy_helper.from_array(np.zeros((2, 2), np.float16), "w")
    surgery.set_initializer(init, np.ones((2, 2), np.float32))
    assert init.data_type == TensorProto.FLOAT16 and numpy_helper.to_array(init).sum() == 4
    with pytest.raises(ValueError):
        surgery.set_initializer(init, np.ones((3, 2), np.float32))


def test_encoder_token_noise_and_input_dims():
    frames, channels = 4, 3
    tokens = numpy_helper.from_array(np.zeros((1, 1, channels), np.float32), "new_tokens")
    shape = numpy_helper.from_array(np.array([frames, 1, channels], np.int64), "shape")
    nodes = [helper.make_node("Expand", ["new_tokens", "shape"], ["tok"]),
             helper.make_node("ReduceSum", ["audio"], ["s"], keepdims=0),
             helper.make_node("Add", ["tok", "s"], ["latent"])]
    model = _model(nodes, [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 2, "T_samples"])],
                   [helper.make_tensor_value_info("latent", TensorProto.FLOAT, None)], [tokens, shape])
    assert surgery.add_token_noise(model, tokens_per_frame=1)
    surgery.name_input_dims(model, "audio", [1, 2, "N"])
    inputs = {i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    assert inputs == {"audio": [1, 2, "N"], surgery.TOKEN_NOISE_INPUT: [surgery.TOKEN_ROWS, 1, channels]}
    noise = np.arange(frames * channels, dtype=np.float32).reshape(frames, 1, channels)
    out = _run(model, {"audio": np.zeros((1, 2, 8), np.float32), surgery.TOKEN_NOISE_INPUT: noise},
               dims={"N": 8, surgery.TOKEN_ROWS: frames})[0]
    np.testing.assert_array_equal(out, noise)


def test_zero_padded_frames_masks_projected_padding():
    weight = numpy_helper.from_array(np.ones((3, 2), np.float32), "proj.weight")
    bias = numpy_helper.from_array(np.full(3, 0.5, np.float32), "proj.bias")
    pads = numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 1], np.int64), "pads")
    nodes = [helper.make_node("Pad", ["latent", "pads"], ["padded"], mode="edge"),
             helper.make_node("Transpose", ["padded"], ["seq"], perm=[0, 2, 1]),
             helper.make_node("Transpose", ["proj.weight"], ["wt"]),
             helper.make_node("MatMul", ["seq", "wt"], ["mm"]),
             helper.make_node("Add", ["proj.bias", "mm"], ["features"]),
             helper.make_node("Identity", ["features"], ["out"])]
    model = _model(nodes, [helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 2, "L"])],
                   [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)], [weight, bias, pads])
    assert surgery.zero_padded_frames(model)
    out = _run(model, {"latent": np.ones((1, 2, 3), np.float32)}, dims={"L": 3})[0]
    np.testing.assert_allclose(out[0, :3], 2.5)
    np.testing.assert_array_equal(out[0, 3], 0.0)
    assert not surgery.zero_padded_frames(_model([helper.make_node("Identity", ["latent"], ["out"])],
                                                 [helper.make_tensor_value_info("latent", TensorProto.FLOAT, None)],
                                                 [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)]))
