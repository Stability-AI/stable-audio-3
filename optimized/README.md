# Optimized runtimes

Platform-specific runtimes for **Stable Audio 3** — each a self-contained,
minimal-dependency way to run SA3 on a given accelerator. Pick by hardware:

| Runtime | Platforms | Accelerator | Inference | Web UI | Training |
|---|---|---|---|---|---|
| **[mlx](mlx)** | Apple Silicon Macs | Metal GPU (MLX) | ✅ | ✅ `./sa3-gradio` | ✅ LoRA (pure-MLX) |
| **[tflite](tflite)** | macOS / Linux / Windows, x86 & ARM | CPU (LiteRT / XNNPACK) | ✅ | ✅ `./sa3-gradio` | LoRA *inference* only (weight-patching) |
| **[tensorRT](tensorRT)** | Linux + NVIDIA GPU | CUDA / TensorRT | ✅ | — | — |

All three read the same weights from
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)
(auto-downloaded on first use) and cover text-to-audio, audio-to-audio,
inpainting, and CFG / negative-prompt guidance.

## One-liners

```bash
# Apple Silicon (Metal GPU)
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/mlx/bootstrap.sh | bash

# Any CPU (macOS / Linux / Windows — needs Git Bash or WSL on Windows for curl|bash)
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.sh | bash

# Linux + NVIDIA
curl -LsSf https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tensorRT/bootstrap.sh | bash
```

## Highlights

- **Web UI** — the **mlx** and **tflite** runtimes each ship a gradio app
  (`./sa3-gradio`) with every generation mode wired (text-to-audio, CFG /
  negative prompt, audio-to-audio, inpainting), tinted mel-spectrogram
  previews, model / precision hot-swap, and LoRA support.
- **LoRA training** — the **mlx** runtime has a complete pure-MLX LoRA trainer
  (pre-encode → train → generate) that matches
  [underfit](https://github.com/dada-bots/underfit)'s conventions and
  checkpoint format, so it doubles as underfit's Apple-Silicon backend. See
  [mlx → LoRA training](mlx/README.md#lora-training). For a full training
  **dashboard** on a Mac, use underfit.
- **LoRA inference** — every runtime loads `.safetensors` adapters (mlx/tflite
  merge or patch them into the graph; per-adapter strength + sampling-step
  gating on mlx).

See each runtime's `README.md` for install, usage, flags, and benchmarks.
