import torch

from stable_audio_3.inference.distribution_shift import LogSNRShift
from stable_audio_3.inference.sampling import build_schedule


class SquareShift:
    """Simple endpoint-preserving warp that exposes scaled-before-warp bugs."""

    def shift(self, t, _seq_len):
        return t.square()


def test_audio_to_audio_shift_is_normalized_then_scaled():
    schedule = build_schedule(
        steps=4,
        sigma_max=0.5,
        dist_shift=SquareShift(),
        fallback_seq_len=1,
    )

    expected = torch.linspace(1.0, 0.0, 5).square() * 0.5
    assert torch.equal(schedule, expected)
    assert torch.all(schedule[:-1] >= schedule[1:])
    assert torch.all(schedule <= 0.5)


def test_shifted_schedule_matches_init_mix_at_both_endpoints():
    schedule = build_schedule(
        steps=8,
        sigma_max=0.37,
        dist_shift=LogSNRShift(),
        fallback_seq_len=646,
    )

    assert schedule[0] == schedule.new_tensor(0.37)
    assert schedule[-1].item() == 0.0
    assert torch.all(schedule[:-1] >= schedule[1:])
    assert torch.all(schedule >= 0.0)
    assert torch.all(schedule <= schedule[0])


def test_full_noise_shifted_schedule_is_unchanged():
    grid = torch.linspace(1.0, 0.0, 9)
    shift = LogSNRShift()
    previous_behavior = shift.shift(grid, 646)
    previous_behavior[0] = 1.0

    schedule = build_schedule(
        steps=8,
        sigma_max=1.0,
        dist_shift=shift,
        fallback_seq_len=646,
    )

    assert torch.equal(schedule, previous_behavior)


def test_per_element_schedules_are_bounded_by_sigma_max():
    schedule = build_schedule(
        steps=4,
        sigma_max=0.6,
        dist_shift=LogSNRShift(),
        effective_seq_len=torch.tensor([324, 646]),
    )

    assert schedule.shape == (2, 5)
    assert torch.equal(schedule[:, 0], torch.tensor([0.6, 0.6]))
    assert torch.equal(schedule[:, -1], torch.zeros(2))
    assert torch.all(schedule[:, :-1] >= schedule[:, 1:])
    assert torch.all(schedule <= 0.6)


def test_unshifted_schedule_preserves_endpoint_option():
    with_endpoint = build_schedule(steps=4, sigma_max=0.5)
    without_endpoint = build_schedule(
        steps=4,
        sigma_max=0.5,
        include_endpoint=False,
    )

    assert torch.equal(with_endpoint, torch.tensor([0.5, 0.375, 0.25, 0.125, 0.0]))
    assert torch.equal(without_endpoint, torch.tensor([0.5, 0.375, 0.25, 0.125]))
