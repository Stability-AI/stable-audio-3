import numpy as np
import pytest
from sa3_dml.audio import write_wav
from sa3_dml.cli import _lora_specs, _requests, build_parser
from sa3_dml.specs import MODELS


def _args(*argv):
    return build_parser().parse_args(["generate", *argv])


@pytest.fixture
def source(tmp_path):
    return str(write_wav(tmp_path / "src.wav", np.zeros((2, 44100 * 3), np.float32), 44100))


def test_batch_prompts_and_seeds():
    requests = _requests(_args("-p", "a", "b", "--duration", "5", "7", "--seed", "10"), MODELS["small-sfx"])
    assert [(r.prompt, r.duration, r.seed) for r in requests] == [("a", 5.0, 10), ("b", 7.0, 11)]
    requests = _requests(_args("-p", "a", "--batch-size", "3", "--seed", "1"), MODELS["small-sfx"])
    assert [r.seed for r in requests] == [1, 2, 3] and {r.duration for r in requests} == {30.0}
    with pytest.raises(ValueError):
        _requests(_args("-p", "a", "b", "--batch-size", "3"), MODELS["small-sfx"])


def test_audio_modes(source):
    model = MODELS["small-sfx"]
    a2a = _requests(_args("-p", "x", "--init-audio", source, "--init-noise-level", "0.5"), model)[0]
    assert a2a.mode == "audio-to-audio" and a2a.duration == pytest.approx(3.0) and a2a.init_noise_level == 0.5
    inpaint = _requests(_args("-p", "x", "--inpaint-audio", source, "--inpaint-start", "0.5", "--inpaint-end", "1",
                              "--inpaint-start", "2", "--inpaint-end", "2.5"), model)[0]
    assert inpaint.mode == "inpainting" and list(inpaint.inpaint_regions) == [(0.5, 1.0), (2.0, 2.5)]
    cont = _requests(_args("-p", "x", "--continue-audio", source, "--duration", "8"), model)[0]
    assert list(cont.inpaint_regions) == [(pytest.approx(3.0), 8.0)] and cont.paste_back
    for bad in (["--inpaint-audio", source], ["--inpaint-start", "1", "--inpaint-end", "2"],
                ["--inpaint-audio", source, "--inpaint-start", "1"], ["--continue-audio", source],
                ["--continue-audio", source, "--duration", "2"]):
        with pytest.raises(ValueError):
            _requests(_args("-p", "x", *bad), model)


def test_lora_strengths():
    assert [s.strength for s in _lora_specs(_args("-p", "x", "--lora", "a", "--lora", "b"))] == [1.0, 1.0]
    assert [s.strength for s in _lora_specs(_args("-p", "x", "--lora", "a", "--lora", "b", "--lora-strength",
                                                  "0.5"))] == [0.5, 0.5]
    assert [s.strength for s in _lora_specs(_args("-p", "x", "--lora", "a", "--lora", "b", "--lora-strength", "0.5",
                                                  "--lora-strength", "2"))] == [0.5, 2.0]
    assert [s.strength for s in _lora_specs(_args("-p", "x", "--lora-ckpt-path", "a", "--lora", "b",
                                                  "--lora-strength", "0.3", "--lora-index", "1"))] == [1.0, 0.3]
    with pytest.raises(ValueError):
        _lora_specs(_args("-p", "x", "--lora-strength", "0.5"))


def test_checkpoint_folder_fallback_and_errors(tmp_path, monkeypatch):
    import huggingface_hub
    from sa3_dml.assets import AssetConfig, AssetError, Assets

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.delenv("SA3_CHECKPOINT_DIR", raising=False)
    (tmp_path / "medium").mkdir()
    (tmp_path / "medium" / "model.safetensors").write_bytes(b"")
    logs = []

    def assets(root):
        return Assets(AssetConfig(cache_dir=tmp_path / "cache", checkpoint=root), log=logs.append)

    shared = assets(tmp_path)
    assert shared.checkpoint(MODELS["medium"]) == tmp_path / "medium" / "model.safetensors"
    assert shared.checkpoint(MODELS["small-music"]) is None
    assert shared.checkpoint(MODELS["small-music"]) is None
    assert logs == [f"no small-music checkpoint under {tmp_path}: using the weights of the official ONNX graph"]

    (tmp_path / "small-sfx").mkdir()
    (tmp_path / "medium" / "model_config.json").write_text(
        '{"model": {"conditioning": {"configs": [{"config": {"repo_id": "stabilityai/stable-audio-3-medium"}}]}}}')
    for root, model in ((tmp_path, "small-sfx"), (tmp_path / "missing", "medium"),
                        (tmp_path / "small-sfx", "small-sfx"), (tmp_path / "medium", "small-music")):
        with pytest.raises(AssetError):
            assets(root).checkpoint(MODELS[model])
