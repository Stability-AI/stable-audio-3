import torch
import torch.nn.functional as F

from tests.utils.audio import assert_audio_valid, sine_wave

DURATION_SEC = 2


def test_encode_decode(model_pipe):
    pipe = model_pipe
    sr = pipe.model_config["sample_rate"]
    channels = pipe.model_config.get("io_channels", 2)
    device = str(pipe.device)

    audio = sine_wave(DURATION_SEC, sr, channels=channels, device=device)

    with torch.inference_mode():
        latents = pipe.same.encode(audio.unsqueeze(0))
        recon = pipe.same.decode(latents)

    assert_audio_valid(recon, DURATION_SEC, sr)

    # Loose reconstruction check: cosine similarity between flattened signals
    min_len = min(audio.shape[-1], recon.shape[-1])
    orig_flat = audio[..., :min_len].flatten().float()
    recon_flat = recon[0, ..., :min_len].flatten().float()
    similarity = F.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()
    assert similarity > 0.5, f"Reconstruction cosine similarity too low: {similarity:.3f}"
