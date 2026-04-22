# Stable Audio 3

**A state-of-the-art open platform for fast, high-quality generated audio and music.** 

TBD Paper/blog links



Stable Audio 3 is the next generation of Stable Audio: a focused, streamlined platform for inference and fine-tuning, built on lessons from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). If you're doing foundational research or working with previous Stable Audio models, that repo is still the place to go.

---

## Models

| RF Model | Autoencoder | Hardware | Params | Use case |
|---|---|---|---|---|
| **Stable Audio 3 Small** | SAME-Small | CPU | 433M | Lightweight inference, no GPU required |
| **Stable Audio 3 Medium** | SAME-Large | GPU (CUDA) | 1.4B | High Quality, Fast Inference |
| **Stable Audio 3 Large** | SAME-Large | API only | 1.7B | Highest quality, API only. Not supported by this repo, see the [API docs](#) |
---

## Features
- ⚡ **Fast, state-of-the-art generation** - Generate minutes of audio in milliseconds
- 🎛️ **Three inference modes** — text-to-audio, audio-to-audio editing, and inpainting/outpainting (new in Stable Audio 3)
- ↔️ **Variable-length generation** — handles generation of a variety of sequences without wasting inference on unused latents
- 🎯 **LoRA fine-tuning** — adapt any model to a target style; stackable, adjustable at runtime
- 💻 **Broad hardware support** — CPU (Small), CUDA/TensorRT (Medium), Apple Silicon via MLX/CoreML, Intel via OpenVINO
- 📈 **Scales from laptop to server** — 433M param CPU model up to 2.7B param API model
- 🎵 **SAME autoencoder** — new Semantic-Acoustic Music Encoder; stereo, 44.1 kHz, 256-dimensional latents optimized for both generative tractability and high-quality reconstruction


## Installation

Stable Audio 3 uses [uv](https://github.com/astral-sh/uv) for fast, lightweight installs. Install only what you need.

```bash
# Base install
uv sync

# With CUDA support
uv sync --extra cuda

# With Gradio UI
uv sync --extra ui

# Multiple extras
uv sync --extra cuda --extra ui
```

### Flash Attention
Stable Audio 3 Medium requires [Flash Attention](https://github.com/Dao-AILab/flash-attention), follow the instructions from there to install.

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
audio = pipe.generate(
    prompt="Lo-fi boom bap meets orchestral strings 84 BPM",
    duration=180,
)
```

**Audio-to-Audio** — Edit an existing recording using a prompt to steer style and mood:

```python
import torchaudio

init_audio = torchaudio.load("/path/to/audio.wav")
audio = pipe.generate(
    init_audio=init_audio,
    init_noise_level=0.9,
    prompt="bossa nova bassline",
    duration=30,
)
```

**Inpainting / Outpainting** — Regenerate a specific region of an audio file while keeping the rest intact:

```python
import torchaudio

inpaint_audio = torchaudio.load("/path/to/audio.wav")
audio = pipe.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy kick drum fill",
    duration=30,
)
```

To extend an audio clip (outpainting), set `inpaint_mask_start_seconds` to the length of the source file and choose a longer `duration`. See [Inference Methods](docs/workflows/inference.md) for the full controls reference.

## Hardware Support

Stable Audio 3 scales from a laptop to a multi-GPU server. Specify your backend at load time:

```python
model = StableAudioPipeline.from_pretrained(
    "medium",
    backend="tensorrt"  # or "mlx", "coreml", "openvino"
)
```

| Backend | Hardware |
|---|---|
| `cuda` + TensorRT | NVIDIA GPU |
| `mlx` | Apple Silicon (Metal) |
| `coreml` | Apple Neural Engine |
| `openvino` | Intel CPU / GPU |
| `cpu` | Any (via LiteRT / XNNPACK) |

### Inference Times

TBD

---

## Docs

| Guide | Description |
|-------|-------------|
| [Inference Methods](docs/workflows/inference.md) | Overview of inference modes (text-to-audio, inpainting, etc.) |
| [LoRA Training](docs/workflows/lora_training.md) | Fine-tune with LoRA: setup, training loop, and checkpointing |
| [Autoencoder Workflows](docs/workflows/autoencoder.md) | Encode and decode audio with the VAE directly |
| [Prompting Guide](docs/guides/prompting.md) | Prompt and control signal reference |
| [Model Overview](docs/guides/model-overview.md) | Architecture and design overview |

---

## Troublshooting

#### Output audio is a static glitch sound (affects Stable Audio 3 Medium-only)

Likely an issue with flash-attention. Please make sure flash attention is installed correctly.
You can check with

```
python -c "import flash_attn; from flash_attn import flash_attn_func; print('Version:', flash_attn.__version__, '| flash_attn_func:', flash_attn_func)"
```

if there are errors in any of this, `flash_attn` is not installed correctly.

---

## License

[Stability AI Community License](#)


To use this model commercially, please refer to https://stability.ai/license


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
