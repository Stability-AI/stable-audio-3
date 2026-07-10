import pytest

from stable_audio_3.interface.diffusion_cond import parse_inpaint_regions


def test_parse_single_inpaint_region():
    assert parse_inpaint_regions("4", "8") == (4.0, 8.0)


def test_parse_multiple_inpaint_regions():
    assert parse_inpaint_regions("4, 16", "8, 20") == (
        [4.0, 16.0],
        [8.0, 20.0],
    )


def test_parse_empty_inpaint_regions():
    assert parse_inpaint_regions("", "") == (None, None)


@pytest.mark.parametrize(
    ("starts", "ends"),
    [("4", ""), ("4, 16", "8"), ("8", "4"), ("-1", "4")],
)
def test_parse_invalid_inpaint_regions(starts, ends):
    with pytest.raises(Exception):
        parse_inpaint_regions(starts, ends)
