import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from optimized.mlx.models.defs.audio_encoding import (
    SAMPLES_PER_LATENT,
    encode_audio,
    patch_audio,
)


class GroupingEncoder:
    def __init__(self, stride: int = 16):
        self.stride = stride

    def __call__(self, patches):
        batch, channels, patch_count = patches.shape
        return mx.mean(
            patches.reshape(
                batch,
                channels,
                patch_count // self.stride,
                self.stride,
            ),
            axis=-1,
        )


def test_patch_audio_matches_patched_pretransform_layout():
    audio = mx.array(np.arange(2 * 8, dtype=np.float32).reshape(1, 2, 8))

    patched = patch_audio(audio, patch_size=4)

    np.testing.assert_array_equal(
        np.asarray(patched),
        np.array(
            [
                [
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                    [8, 12],
                    [9, 13],
                    [10, 14],
                    [11, 15],
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_encode_audio_pads_to_codec_alignment_and_builds_mask():
    audio = mx.zeros((2, 2, SAMPLES_PER_LATENT + 1))

    encoded = encode_audio(
        GroupingEncoder(),
        audio,
        valid_sample_lengths=[SAMPLES_PER_LATENT, SAMPLES_PER_LATENT + 1],
        pad_modulo=32,
    )

    assert encoded.source_samples == SAMPLES_PER_LATENT + 1
    assert encoded.padded_samples == SAMPLES_PER_LATENT * 2
    assert encoded.latents.shape == (2, 512, 2)
    assert encoded.valid_latent_lengths == (1, 2)
    np.testing.assert_array_equal(
        np.asarray(encoded.padding_mask),
        np.array([[True, False], [True, True]]),
    )


def test_chunked_encoding_matches_unchunked_for_even_and_odd_overlaps():
    rng = np.random.default_rng(21)
    audio = mx.array(
        rng.standard_normal((2, 2, SAMPLES_PER_LATENT * 11)).astype(np.float32)
    )
    encoder = GroupingEncoder()

    unchunked = encode_audio(encoder, audio, pad_modulo=16)
    for overlap in (2, 1):
        chunked = encode_audio(
            encoder,
            audio,
            pad_modulo=16,
            chunked=True,
            chunk_size=4,
            overlap=overlap,
            chunk_batch_size=2,
        )

        np.testing.assert_allclose(
            np.asarray(chunked.latents),
            np.asarray(unchunked.latents),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(chunked.padding_mask),
            np.asarray(unchunked.padding_mask),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"valid_sample_lengths": [1, 2]}, "one value per audio batch item"),
        ({"pad_modulo": 17}, "positive multiple of encoder_stride"),
        (
            {"chunked": True, "chunk_size": 3, "overlap": 1, "pad_modulo": 32},
            "incompatible with pad_modulo",
        ),
    ],
)
def test_encode_audio_rejects_invalid_contracts(kwargs, message):
    audio = mx.zeros((1, 2, SAMPLES_PER_LATENT * 4))

    with pytest.raises(ValueError, match=message):
        encode_audio(GroupingEncoder(), audio, **kwargs)
