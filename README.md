# Stable Audio 3

**A state-of-the-art open platform for fast, high-quality generated audio and music.** 

TBD Paper/blog links

<a href="https://discord.gg/cKpvjey8b"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>


Stable Audio 3 is the next generation of Stable Audio: a focused, streamlined platform for inference and fine-tuning, built on lessons from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). If you're doing foundational research or working with previous Stable Audio models, that repo is still the place to go.


---

## Models

| Model | Autoencoder | Hardware | Params | Max length | Use case |
|---|---|---|---|---|---|
| **Stable Audio 3 Small-Music** | SAME-Small | CPU | 433M | 120s | Lightweight music-only inference, no GPU required |
| **Stable Audio 3 Small-SFX** | SAME-Small | CPU | 433M | 120s | Lightweight sound effects-only inference, no GPU required |
| **Stable Audio 3 Medium** | SAME-Large | GPU (CUDA) | 1.4B | 380s | High Quality, Fast Inference |
| **Stable Audio 3 Large** | SAME-Large | API only | 2.7B | 380s | Highest quality, API only. Not supported by this repo, see the [API docs](#) |
---

## Features
- ⚡ **Fast, state-of-the-art generation** - Generate minutes of audio in milliseconds
- 🎛️ **Three inference modes** — text-to-audio, audio-to-audio editing, and inpainting/continuation
- ↔️ **Variable-length generation** — handles generation of a variety of sequences without wasting inference on unused latents
- 🎯 **Personalization through LoRA fine-tuning** — adapt any model to a target style; stackable, adjustable at runtime
- 💻 **Broad hardware support** — CPU (Small), CUDA/TensorRT (Medium), Apple Silicon via MLX/CoreML, Intel via OpenVINO
- 🎵 **SAME autoencoder** — new Semantic-Acoustic Music Encoder; stereo, 44.1 kHz, 256-dimensional latents optimized for both generative tractability and high-quality reconstruction


## Installation

Stable Audio 3 uses [uv](https://github.com/astral-sh/uv) for fast, lightweight installs. Install only what you need.

```bash
# Base install (Python API only)
uv sync

# With Gradio UI
uv sync --extra ui

# With LoRA training support
uv sync --extra lora

# Everything
uv sync --extra ui --extra lora
```

### CUDA Version

By default, `uv sync` installs PyTorch built against CUDA 12.6. If you need a different CUDA version, install torch and torchaudio manually first (pinning the same version as `pyproject.toml`), then sync without reinstalling them, for example:

```bash
uv pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
uv sync --no-install-package torch --no-install-package torchaudio
```

Replace `cu118` with your target version. For torch 2.7.1, available CUDA variants are `cu118`, `cu126`, and `cu128`. Not all versions are published for every CUDA channel — check the [PyTorch install page](https://pytorch.org/get-started/locally/) to confirm your target is available.

### Flash Attention

Stable Audio 3 Medium requires [Flash Attention 2](https://github.com/Dao-AILab/flash-attention).

**Install from a pre-built wheel** (fast, no compilation). The easiest source is the [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) community repo — browse the releases for a wheel matching your CUDA, PyTorch, and Python versions, then install it directly:

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
```

The filename encodes the requirements — `cu126` is CUDA 12.6, `torch2.7` is PyTorch 2.7, `cp310` is Python 3.10. Pick the URL that matches your environment.

If no pre-built wheel matches your setup, build from source. Install `ninja` first to speed up the C++ compile, then set the environment variables for your machine:

```bash
uv pip install ninja
.venv/bin/python -m ensurepip
FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="9.0" \
MAX_JOBS=8 \
.venv/bin/pip3 install flash-attn --no-build-isolation --no-binary flash-attn \
    --force-reinstall --no-cache-dir --no-deps
```

- `TORCH_CUDA_ARCH_LIST` — set to your GPU's compute capability: `8.0` (A100), `8.6` (A10/RTX 3090), `8.9` (L4/RTX 4090), `9.0` (H100/H200)
- `MAX_JOBS` — number of parallel compile jobs; 4–8 is typical, reduce if you run out of RAM during compilation

## Quick Start

Launch the Gradio UI:

```bash
uv run python run_gradio.py --model medium
```

This starts a local web interface with a shareable link. To load a LoRA checkpoint:

```bash
uv run python run_gradio.py --model medium --lora-ckpt-path path/to/lora.ckpt
```

## Usage

Stable Audio 3 supports several inference modes. For full details, see [Inference Methods](docs/workflows/inference.md).

**Text-to-Audio** — Generate audio from a text prompt:

```python
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
audio = model.generate(
    prompt="Lo-fi boom bap meets orchestral strings 84 BPM",
    duration=180,
)
```

**Audio-to-Audio** — Edit an existing recording using a prompt to steer style and mood:

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
init_audio = torchaudio.load("/path/to/audio.wav")
audio = model.generate(
    init_audio=init_audio,
    init_noise_level=0.9,
    prompt="bossa nova bassline",
    duration=30,
)
```

**Inpainting / Continuation** — Regenerate a specific region of an audio file while keeping the rest intact:

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")

inpaint_audio = torchaudio.load("/path/to/audio.wav")
audio = model.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy kick drum fill",
    duration=30,
)
```

To extend an audio clip (continuation), set `inpaint_mask_start_seconds` to the length of the source file and choose a longer `duration`. See [Inference Methods](docs/workflows/inference.md) for the full controls reference.


**Encoding / Decoding** — Use the autoencoder directly to encode audio to latents or decode latents back to audio:

```python
import torchaudio
from stable_audio_3 import AutoencoderModel

ae = AutoencoderModel.from_pretrained("same-l")
waveform, sr = torchaudio.load("audio.wav")
latents = ae.encode(waveform, sr)
audio_out = ae.decode(latents)
```

See [Autoencoder Workflows](docs/workflows/autoencoder.md) for encoding batches, chunked processing, and pre-encoding datasets for LoRA training.

## Hardware Support

*COMING SOON*

Stable Audio 3 scales from a laptop to a multi-GPU server. Specify your backend at load time:

```python
model = StableAudioModel.from_pretrained(
    "medium",
    backend="tensorrt"  # or "mlx", "coreml", "openvino"
)
```


### Inference Times

TBD

---

## Docs

| Guide | Description |
|-------|-------------|
| [Inference Methods](docs/workflows/inference.md) | Overview of inference modes (text-to-audio, inpainting, etc.) |
| [LoRA Training](docs/workflows/lora.md) | Fine-tune with LoRA: setup, training loop, and checkpointing |
| [Autoencoder Workflows](docs/workflows/autoencoder.md) | Encode and decode audio with the VAE directly |
| [Prompting Guide](docs/guides/prompting.md) | Prompt and control signal reference |
| [Model Overview](docs/guides/model-overview.md) | Architecture and design overview |

---

## Community

Join our [Discord](https://discord.gg/cKpvjey8b) for updates, help, and discussions. We host weekly office hours talking all things AI audio.

---

## Troubleshooting

#### Output audio is a static glitch sound (affects Stable Audio 3 Medium-only)

Likely an issue with flash-attention. Verify it is importable:

```bash
uv run python -c "import flash_attn; from flash_attn import flash_attn_func; print('Version:', flash_attn.__version__, '| flash_attn_func:', flash_attn_func)"
```

If this errors, flash-attn is not installed correctly — see the [Flash Attention install instructions](#flash-attention) above.

---

## License

To use these models commercially, please refer to the 
[Stability AI Community License](https://stability.ai/license)


## Testing

Install dev dependencies:

```bash
uv sync --group dev
```

Run the test suite:

```bash
uv run pytest
```

Save generated audio outputs to `test_audio_outputs/` for manual inspection:

```bash
uv run pytest --save-audio
```
