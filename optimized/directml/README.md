# sa3_dml — Stable Audio 3 on DirectML

Inference for **Stable Audio 3** on Windows GPUs through ONNX Runtime and the DirectML execution
provider. It runs the official ONNX graphs from
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized),
adapted so that every network kernel executes on the GPU.

## Features

- Models: `medium` (DiT 1.4B + SAME-L), `small-sfx` and `small-music` (DiT 433M + SAME-S).
- Text-to-audio with the reference pipeline: T5Gemma conditioning, rectified-flow pingpong sampler
  (8 steps), classifier-free guidance with negative prompt and APG.
- Audio-to-audio: the source is encoded with SAME and sampling starts from its latents noised to
  `--init-noise-level`, with the schedule starting at that level.
- Inpainting of one or more regions, and continuation of a file beyond its end. Latents outside the
  regions are restored from the source after sampling (`--no-paste-back` keeps the generated ones).
- LoRA adapters in the Stable Audio 3 format (`lora`, `dora-rows`, `dora-cols`, `bora` and their
  `-xs` variants) and PEFT LoRA/DoRA folders; several adapters stack with individual strengths.
- SAME-S / SAME-L encode (WAV to `.npy` latents) and decode as separate commands.
- WAV input (8/16/24/32-bit PCM or float, any sample rate and channel count; resampled to 44.1 kHz
  stereo), WAV output (float32 or 16-bit PCM).
- Several prompts or `--batch-size` outputs per run: each network is loaded once for all of them
  (the DiT once per distinct duration).
- Local checkpoints are reused: DiT weights are read from `model.safetensors` and bound into the
  official graph. With a local Medium checkpoint only the 4 MB graph is downloaded instead of
  2.9 GB of weights.
- One static shape per session, so DirectML compiles each graph into a single fused node and no
  kernel falls back to the CPU (`--verify-placement` reports it).

## Requirements

- Windows 10/11 with a DirectX 12 GPU. Tested on an AMD Radeon RX 7600 (8 GB).
- Python 3.11–3.14.
- Access to the gated Hugging Face repositories: accept the license on the
  [model pages](https://huggingface.co/collections/stabilityai/stable-audio-3), then log in once with
  `.venv\Scripts\hf auth login`.

## Installation

```powershell
cd optimized\directml
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Configuration

| Option | Environment | Purpose |
|---|---|---|
| `--checkpoint-dir` | `SA3_CHECKPOINT_DIR` | Local checkpoint: a model folder, a `model.safetensors`, or a folder with one subfolder per model (`medium`, `small-sfx`, `small-music`). A model without a subfolder in such a folder uses, with a console notice, a checkpoint from the Hugging Face cache or else the weights of the official ONNX graph; any other path without a matching checkpoint is an error. The tokenizer is read from `t5gemma-b-b-ul2/` beside the checkpoint. |
| `--no-checkpoint` | | Use the weights embedded in the official ONNX instead. |
| `--onnx-dir` | | Local copy of the `onnx/` folder of the optimized repository. |
| `--cache-dir` | `SA3_DML_CACHE` | DirectML-ready graphs (default: `models/` in this folder). |
| `--offline` | | Never access the network. |
| | `HF_HOME` | Hugging Face cache location. |

Without `--checkpoint-dir` or `SA3_CHECKPOINT_DIR` the Hugging Face cache is searched for the model
checkpoint, and the weights of the official ONNX graph are used when there is none. Files fetched on
first use:

| Artifact | Download | Needed for |
|---|---|---|
| T5Gemma encoder | 620 MB | generation |
| Medium DiT | 4 MB with a local checkpoint, 2.9 GB without | medium |
| Small DiT | 921 MB | small models |
| SAME-L decoder / encoder | 1.2 GB / 1.2 GB | medium; encoder only for audio inputs |
| SAME-S decoder / encoder | 219 MB / 216 MB | small models; encoder only for audio inputs |

They are converted once into the cache directory, by `prepare` or on first use (about 2.9 GB for
Medium, 0.9 GB per small model, 1.2 GB per SAME-L graph, 0.2 GB per SAME-S graph).

## Usage

```powershell
$env:SA3_CHECKPOINT_DIR = "path\to\StableAudio3"
.venv\Scripts\python -m sa3_dml devices
.venv\Scripts\python -m sa3_dml prepare --model medium

# text-to-audio
.venv\Scripts\python -m sa3_dml generate --model medium -p "A beautiful piano arpeggio grows into a cinematic climax" --duration 30 -o piano.wav
.venv\Scripts\python -m sa3_dml generate --model small-sfx -p "Thunderstorm with heavy rain" --cfg-scale 2 --negative-prompt "music, voices"
.venv\Scripts\python -m sa3_dml generate --model small-sfx -p "Dog barking" "Door creaking open" --duration 10 -o sfx.wav

# audio-to-audio
.venv\Scripts\python -m sa3_dml generate --model medium -p "Bossa nova guitar and brushed drums" --init-audio input.wav --init-noise-level 0.8

# inpainting, two regions
.venv\Scripts\python -m sa3_dml generate --model medium -p "Punchy kick drum fill" --inpaint-audio input.wav --inpaint-start 4 --inpaint-end 8 --inpaint-start 16 --inpaint-end 20 --duration 30

# continuation of a clip shorter than 30 s up to 30 s
.venv\Scripts\python -m sa3_dml generate --model small-music -p "Dreamy synth outro" --continue-audio clip.wav --duration 30

# LoRA: one adapter, or two stacked with their own strengths
.venv\Scripts\python -m sa3_dml generate --model medium -p "Oldschool goa trance" --lora goa.safetensors --lora-strength 0.8
.venv\Scripts\python -m sa3_dml generate --model medium -p "Maqam Bayati on oud" --lora path\to\peft_adapter --lora goa.safetensors --lora-strength 1 --lora-strength 0.4

# SAME autoencoder
.venv\Scripts\python -m sa3_dml encode -i input.wav -o latents.npy --codec same-l
.venv\Scripts\python -m sa3_dml decode -i latents.npy -o decoded.wav --codec same-l

.venv\Scripts\python -m sa3_dml generate --model medium -p "Lo-fi hip hop beat, 90 BPM" --verify-placement
```

| `generate` flag | Default | |
|---|---|---|
| `-p`, `--prompt` | required | One or more prompts; an empty string is unconditional. |
| `--negative-prompt` | none | One, or one per prompt. Used when `--cfg-scale` is not 1. |
| `--duration` | 30, or the source length | Seconds, one or one per output; up to 380 (medium) or 120 (small). |
| `--batch-size` | one per prompt | Outputs; output *i* uses seed + *i* and gets an `_i` file suffix. |
| `--steps` | 8 | Sampling steps. |
| `--cfg-scale`, `--apg-scale` | 1.0, 1.0 | Guidance; 1.0 disables CFG. The released models are tuned for 1.0 and saturate above about 2. |
| `--seed` | random | Seeds the latent noise, sampler noise and autoencoder noise. |
| `-o`, `--output`, `--wav-format` | `output/<prompt>_<seed>.wav`, `float32` | `pcm16` for 16-bit WAV. |
| `--init-audio`, `--init-noise-level` | none, 0.9 | Audio-to-audio source and starting noise level (0.01–1). |
| `--inpaint-audio`, `--inpaint-start`, `--inpaint-end` | none | Source and regions to regenerate, in seconds; repeat start/end for several regions. Past `--duration` everything is generated. |
| `--continue-audio` | none | Source to extend up to `--duration`. |
| `--no-paste-back` | off | Keep the regenerated latents outside the inpaint regions. |
| `--lora` (`--lora-ckpt-path`) | none | `.safetensors` adapter or PEFT folder; repeat to stack. |
| `--lora-strength`, `--lora-index` | 1.0 | One strength for all adapters, one per adapter, or one for the adapter at `--lora-index`. |
| `--codec` (`--decoder`) | model default | `same-l` or `same-s`, used to encode the sources and decode. |
| `--decode-chunk`, `--decode-overlap`, `--encode-chunk`, `--encode-overlap` | 128, 32 | Autoencoder windows in latent frames; chunk 0 processes the whole sequence at once. |
| `--device` | 0 | DirectX adapter index from `devices`. |
| `--verify-placement` | off | Profiles every session and lists kernels per execution provider. |
| `--provider cpu` | | Runs the same graphs on the CPU, as a numerical reference. |

`encode` and `decode` take `-i`, `-o`, `--codec` (default `same-l`), `--seed`, `--chunk`, `--overlap`,
`--device`, `--provider` and `--verify-placement`; `decode` also takes `--wav-format`. Latents are
`.npy` arrays of shape (256, T), one frame per 4096 samples.

From Python, a `Pipeline` keeps prepared graphs and merged LoRA weights between calls, and
`keep_loaded` also keeps the sessions of all networks (`True`) or of the given roles (`"text"`,
`"encoder"`, `"dit"`, `"decoder"`). Kept networks share GPU memory: a repeated small-sfx inpainting
call drops from 8 s to 2 s, while Medium on an 8 GB GPU only has room for a resident `{"dit"}` when
the GPU is otherwise idle.

```python
from sa3_dml import Audio, LoraSpec, Pipeline, Request
from sa3_dml.assets import AssetConfig, Assets, default_cache_dir
from sa3_dml.session import RuntimeConfig

assets = Assets(AssetConfig(cache_dir=default_cache_dir(), checkpoint="path/to/StableAudio3"))
pipe = Pipeline("small-music", assets, RuntimeConfig(), loras=[LoraSpec("style.safetensors", 0.8)],
                keep_loaded=True)
song = Audio.load("input.wav")
first = pipe.generate(Request("Punchy kick drum fill", 30, inpaint_audio=song, inpaint_regions=[(4, 8)]))
second = pipe.generate(Request("Bossa nova guitar", 30, init_audio=song, init_noise_level=0.8))
pipe.close()
```

## Performance

AMD Radeon RX 7600 (8 GiB), 8 steps, process wall time including graph loading.

| Model | Task | Audio | Time | VRAM (largest stage) |
|---|---|---|---|---|
| small-sfx | text-to-audio | 10 s | 8–9 s | 0.9 GiB |
| small-sfx | text-to-audio | 120 s | 11 s | 1.4 GiB |
| small-music | audio-to-audio | 20 s | 9–10 s | 1.0 GiB |
| small-music | continuation | 20 s → 40 s | 10 s | 1.0 GiB |
| medium | text-to-audio | 30 s | 18–19 s | 2.9 GiB |
| medium | audio-to-audio | 30 s | 23 s | 2.9 GiB |
| medium | inpainting | 30 s | 23 s | 2.9 GiB |
| medium | text-to-audio + LoRA (169 layers) | 30 s | 21–22 s | 2.9 GiB |
| medium | text-to-audio | 120 s | 34–35 s | 3.8 GiB |
| medium | text-to-audio | 380 s | 71 s | 6.8 GiB |
| SAME-S | encode / decode | 60 s | 2 s / 2 s | 0.7 GiB |
| SAME-L | encode / decode | 60 s | 6 s / 6 s | 1.7 GiB |

## Graph adaptations

- SAME-L encoder and decoder: the 12 `samel::diff_attn_swa` TensorRT plugin nodes are replaced by
  blocked ±17-frame sliding-window differential attention made of standard ops, with 4-D GEMMs only.
- DiT: a `cross_attn_scale` input scales the cross-attention conditioning, so the CFG
  unconditional branch is an all-zero sequence and a negative prompt zeroes its padding, as in the
  reference sampler. The inpainting mask and masked source latents enter through `local_add_cond`.
- SAME graphs: the in-graph `RandomNormalLike` bottleneck dither is removed and a `token_noise`
  input restores the noise the reference SAME blocks add to their learned tokens (decoders; SAME-L
  encoder); both are drawn on the host from the run seed. The decoders' int32 PCM output is
  replaced by the clamped waveform. The SAME-S decoder edge-pads odd latent lengths; a frame mask
  after its latent projection restores the zero padding of the reference.
- The DiT samples the unmasked frames of the reference sample (duration + 6 s headroom), which is
  what the reference's padding-masked attention computes.
- LoRA: merged DiT weights replace the graph's externally stored weights when the session is
  created; the prepared graph on disk is not modified. Stacked adapters are applied in order, each
  on the weight produced by the previous one, with the strength applied inside the adapter as in
  the reference.

Work done on the CPU: tokenization, WAV reading and resampling, random number generation, the
sampler update and guidance arithmetic (element-wise NumPy on `[256, L]` arrays), inpainting masks,
autoencoder window stitching, LoRA weight merging, WAV encoding and the one-time graph preparation.

## Tests

```powershell
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests
```

## Limitations

- Batch items run one after another (each network is loaded once per run); CFG runs two DiT
  forwards per step.
- Each CLI run loads the graphs again (about 10 s for the Medium DiT); see `keep_loaded` above.
- The DiT runs the official fp16 graph: per-step velocity matches the fp32 reference at 35–48 dB SNR,
  and the 8-step sampler amplifies that into a slightly different render for the same seed.
  DirectML reductions are not deterministic, so two runs with the same seed differ slightly too.
- LoRA adapters apply to every sampling step; the per-adapter step interval and layer filter of the
  reference Gradio interface are not available. `-xs` adapters compute SVD bases of every adapted
  weight on the CPU at load (0.2–1 s per layer).
- Audio inputs must be WAV files.
- At 380 s the Medium DiT needs about 6.8 GiB of VRAM. When other applications hold GPU memory,
  Windows moves allocations to system memory and generation slows down.

## License

Code: MIT, see the repository [LICENSE](../../LICENSE). Model weights: [Stability AI Community
License](https://stability.ai/license); the T5Gemma encoder is subject to the Gemma Terms of Use.
No weights or ONNX graphs are distributed in this repository.
