import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from optimized.mlx.models.defs.training import (
    rectified_flow_loss,
    sample_training_timesteps,
    shift_training_timesteps,
)


def test_truncated_logit_normal_sampler_matches_sa3_distribution():
    values = sample_training_timesteps(
        "trunc_logit_normal",
        20_000,
        rng=np.random.default_rng(17),
    )

    assert values.dtype == np.float32
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert 0.52 < float(values.mean()) < 0.55
    assert 0.52 < float(np.median(values)) < 0.56


def test_full_training_shift_matches_sa3_defaults():
    shifted = shift_training_timesteps(
        [0.5],
        507,
        shift_type="full",
    )

    assert shifted[0] == pytest.approx(0.6323907626)


def test_rectified_flow_loss_uses_velocity_target_and_mask():
    clean = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    noise = mx.array([[[4.0, 5.0, 6.0]]], dtype=mx.float32)
    timesteps = mx.array([0.25], dtype=mx.float32)
    mask = mx.array([[True, False, True]])

    def perfect_model(noised, timestep):
        del noised, timestep
        return noise - clean

    def zero_model(noised, timestep):
        del timestep
        return mx.zeros_like(noised)

    assert float(
        rectified_flow_loss(
            perfect_model,
            clean,
            timesteps,
            noise=noise,
            loss_mask=mask,
        )
    ) == pytest.approx(0.0)
    assert float(
        rectified_flow_loss(
            zero_model,
            clean,
            timesteps,
            noise=noise,
            loss_mask=mask,
        )
    ) == pytest.approx(9.0)
