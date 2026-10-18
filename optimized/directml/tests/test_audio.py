import struct

import numpy as np
import pytest
from sa3_dml.audio import fit_length, read_wav, resample, to_channels, write_wav
from sa3_dml.codec import stitch
from sa3_dml.sampling import keep_mask
from sa3_dml.specs import DOWNSAMPLING_RATIO, SAMPLE_RATE


def _wav(path, tag, channels, rate, bits, payload, extensible=False):
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * block, block, bits)
    if extensible:
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + bytes(14)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def test_wav_roundtrip(tmp_path):
    audio = np.stack([np.linspace(-1, 1, 101), np.linspace(0.5, -0.5, 101)]).astype(np.float32)
    got, rate = read_wav(write_wav(tmp_path / "f.wav", audio, 32000, "float32"))
    assert rate == 32000
    np.testing.assert_array_equal(got, audio)
    got, _ = read_wav(write_wav(tmp_path / "p.wav", audio, 32000, "pcm16"))
    np.testing.assert_allclose(got, np.rint(audio * 32767) / 32768, atol=1e-7)


def test_wav_integer_formats(tmp_path):
    ints = np.array([[-8388608, 0, 4194304], [8388607, -1, 1]], np.int32)
    raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints.T.ravel())
    got, _ = read_wav(_wav(tmp_path / "24.wav", 1, 2, 48000, 24, raw, extensible=True))
    np.testing.assert_array_equal(got, ints / 2 ** 23)
    got, _ = read_wav(_wav(tmp_path / "8.wav", 1, 1, 8000, 8, bytes([0, 128, 255])))
    np.testing.assert_array_equal(got, [[-1.0, 0.0, 127 / 128]])
    with pytest.raises(ValueError):
        read_wav(_wav(tmp_path / "alaw.wav", 6, 1, 8000, 8, bytes(4)))


def test_resample_matches_torchaudio():
    n = np.arange(400)
    x = (np.sin(2 * np.pi * 1000 * n / 48000) * np.hanning(400)).astype(np.float32)[None]
    down = resample(x, 48000, 44100)
    assert down.shape == (1, 368)
    np.testing.assert_allclose(down[0, [0, 50, 123, 200, 367]], [6e-07, 0.128647, -0.7335293, -0.2147302, -1.9e-06],
                               atol=2e-6)
    up = resample(x, 22050, 44100)
    assert up.shape == (1, 800)
    np.testing.assert_allclose(up[0, [1, 99, 401, 777]], [2e-06, 0.0282022, 0.8973701, 0.0037877], atol=2e-6)
    np.testing.assert_array_equal(resample(x, 44100, 44100), x)


def test_channels_and_length():
    mono = np.arange(4, dtype=np.float32)[None]
    assert to_channels(mono, 2).shape == (2, 4)
    surround = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(to_channels(surround, 2), surround[:2])
    np.testing.assert_array_equal(fit_length(surround, 6)[:, 4:], 0)
    np.testing.assert_array_equal(fit_length(surround, 2), surround[:, :2])


@pytest.mark.parametrize("regions,duration", [([(1.0, 2.0)], 3.0), ([(0.5, 1.0), (2.0, 2.5)], 2.9),
                                              ([(0.0, 0.2), (3.5, 9.0)], 3.0)])
def test_keep_mask_matches_sample_domain_reference(regions, duration):
    length = 40
    samples = np.ones(length * DOWNSAMPLING_RATIO, np.float32)
    for start, end in regions:
        samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)] = 0
    samples[int(duration * SAMPLE_RATE):] = 0
    np.testing.assert_array_equal(keep_mask(length, regions, duration)[0, 0], samples[::DOWNSAMPLING_RATIO])


@pytest.mark.parametrize("total", [128, 129, 300, 388])
def test_stitch_reassembles_windows(total):
    signal = np.arange(total * 4, dtype=np.float32).reshape(1, 1, -1)
    calls = []

    def run(start):
        calls.append(start)
        return signal[..., start * 4:(start + 128) * 4]

    np.testing.assert_array_equal(stitch(total, 128, 32, 4, run), signal)
    assert calls[-1] == total - 128
