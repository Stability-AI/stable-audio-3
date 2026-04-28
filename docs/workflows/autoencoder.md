# Autoencoder

Stable Audio 3 uses a audio autoencoder to compress waveforms into a compact continuous latent representation that the diffusion model operates on. This page covers how to use the autoencoder directly, for encoding individual audio files, decoding latents back to audio, and pre-encoding a dataset for training.

## Loading the autoencoder

```python
import json
import torch
from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import all_models

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = all_models["small"]  # "small", "small-rf", "medium", "medium-rf", "same-s", "same-l"
local_config, local_ckpt = cfg.resolve()
autoencoder = load_autoencoder(local_config, local_ckpt, device=device)
autoencoder.eval().requires_grad_(False)

with open(local_config) as f:
    sample_rate = json.load(f)["sample_rate"]  # 44100
```

`load_autoencoder` reads only the autoencoder weights from the checkpoint (not the DiT), so it works with full combined checkpoints and is memory-efficient.

## Encoding audio to latents

Audio must be a float32 tensor of shape `(channels, samples)`, in the range `[-1, 1]`. Pass the input sample rate to `preprocess_audio_for_encoder` and it will resample, convert channels, and pad to a multiple of the model's downsampling ratio automatically.

```python
import torchaudio

waveform, sr = torchaudio.load("audio.wav")  # (channels, samples)

# Preprocess: resample, channel conversion, pad
audio_batch = autoencoder.preprocess_audio_for_encoder(waveform, in_sr=sr)
# → (1, 2, padded_samples)

with torch.no_grad():
    latents = autoencoder.encode(audio_batch.to(device))
    # → (1, latent_dim, latent_time)
```

The latent time dimension is `padded_samples // downsampling_ratio` (4096 for all current models). At 44.1 kHz, 10 seconds of stereo audio produces 216 latent frames.

To preprocess a batch of clips with different lengths in one call:

```python
audio_batch = autoencoder.preprocess_audio_list_for_encoder(
    [waveform_a, waveform_b],
    in_sr_list=[44100, 22050],  # or a single int if all share the same rate
)
# → (2, 2, padded_samples)
```

## Decoding latents to audio

```python
with torch.no_grad():
    audio_out = autoencoder.decode(latents)
    # → (1, 2, samples)

torchaudio.save("reconstructed.wav", audio_out[0].cpu(), sample_rate)
```

## Chunked processing for long audio

For audio that is too long to encode or decode in a single forward pass, use the chunked variants. `chunk_size` and `overlap` are both measured in latent frames (not audio samples).

```python
latents = autoencoder.encode_audio(
    audio_batch.to(device),
    chunked=True,
    chunk_size=128,  # latent frames per chunk
    overlap=32,      # overlap between chunks to avoid boundary artifacts
)

audio_out = autoencoder.decode_audio(
    latents,
    chunked=True,
    chunk_size=128,
    overlap=32,
)
```

The overlap should be at least as large as the model's receptive field. A value of 32 is a reasonable default.

## Saving and loading latents

```python
import numpy as np

# Save
np.save("latents.npy", latents[0].cpu().numpy())  # (latent_dim, latent_time)

# Load and decode
latent_np = np.load("latents.npy")
latent_tensor = torch.from_numpy(latent_np).unsqueeze(0).to(device)
# → (1, latent_dim, latent_time)

with torch.no_grad():
    audio_out = autoencoder.decode(latent_tensor)
```

## Pre-encoding a dataset

For LoRA training, it is much faster to pre-encode your dataset once and train from the saved latents. Use the provided script:

```bash
uv run python scripts/pre_encode_dataset.py \
  --model small \
  --data_dir ./my_data \
  --output_path ./latents_out \
  --batch_size 1
```

The script expects audio files paired with `.txt` caption files:

```
my_data/
  clip1.wav
  clip1.txt
  clip2.wav
  clip2.txt
```

Each encoded clip is written as a `.npy` latent and a `.json` metadata file (which includes a padding mask tracking the valid audio region before padding):

```
latents_out/
  000000000000.npy
  000000000000.json
  000000000001.npy
  000000000001.json
```

Pass the output directory to `train_lora.py` via `--encoded_dir`. See [LoRA training](lora_training.md) for the full training workflow.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `same-l` | Model variant: `small`, `small-rf`, `medium`, `medium-rf`, `same-s`, `same-l` |
| `--data_dir` | — | Folder containing audio + `.txt` pairs |
| `--output_path` | — | Where to write `.npy`/`.json` latent pairs |
| `--batch_size` | `1` | Audio clips to encode per forward pass |
| `--sample_size` | `12582912` | Samples to pad/crop to (default ~380s at 44.1kHz)|
| `--model_half` | off | Run the autoencoder in fp16 to reduce memory |
