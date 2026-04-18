# Stable Audio 3

**A state-of-the-art open platform for fast, high-quality generated audio and music.** (WIP tagline)

TBD Paper/blog links

Image(s) TBD

Stable Audio 3 is the next generation of Stable Audio: a focused, streamlined platform for inference and fine-tuning, built on lessons from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). If you're doing foundational research or working with previous Stable Audio models, that repo is still the place to go.

---

## Models

| RF Model | Autoencoder | Hardware | Params | Latency† | Use case |
|---|---|---|---|---|---|
| **Stable Audio 3 Small** | SAME-Small | CPU | 433M | 20ms | Lightweight inference, no GPU required |
| **Stable Audio 3 Medium** | SAME-Large | GPU (CUDA) | 1.4B | 20ms | High Quality, Fast Inference |
| **Stable Audio 3 Large** | SAME-Large | API only | 1.7B | 20ms | Highest quality, API only. Not supported by this repo, see the [API docs](#) |
---

<sub>† Measured generating a 380s clip at 8 steps. Small: Apple M4 CPU. Medium: Intel x86 + NVIDIA H100.</sub>

## Features
- SAME
- Hardware



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

## Usage


## Hardware Support

Stable Audio 3 scales from a laptop to a multi-GPU server. Specify your backend at load time:

```python
model = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-3.0-small",
    backend="tensorrt"  # or "mlx", "coreml", "openvino", "cpu"
)
```

| Backend | Hardware |
|---|---|
| `cuda` + TensorRT | NVIDIA GPU |
| `mlx` | Apple Silicon (Metal) |
| `coreml` | Apple Neural Engine |
| `openvino` | Intel CPU / GPU |
| `cpu` | Any (via LiteRT / XNNPACK) |

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