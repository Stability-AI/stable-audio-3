# Stable Audio 3 Model
> For a more in-depth breakdown of Stable Audio 3, please see our [tech report](https://arxiv.org/abs/2605.17991).

Stable Audio 3 is a family of text-conditioned audio generation models.

## Using the Model

All configurations (`small-music`, `small-sfx`, and `medium`) share the same interface — see the [model table](../../README.md#models) for hardware requirements and generation speed.

| Input | Description |
|---|---|
| `prompt` | Text description of the audio to generate |
| `duration` | Length of audio to generate, in seconds |

| Output | Value |
|---|---|
| Format | 44.1 kHz stereo audio |
| Bit depth | 32-bit float |

**Limitations**
- Not designed for speech or voice generation
- Trained on English descriptions; other languages will underperform


## System Overview
There are two main pieces of the system: the SAME autoencoder and the diffusion transformer.

**SAME Autoencoder**

SAME (Semantic-Acoustic Music Encoder) is a stereo autoencoder that compresses audio into 256-dimensional continuous latents (encoder) and reconstructs them back into audio (decoder). SAME is trained separately from the diffusion transformer.

**Diffusion transformer**

The diffusion transformer learns to generate SAME latents conditioned on inputs like text prompt and duration. These inputs are turned into embeddings that guide the model toward latents matching those conditions, which are then decoded by the SAME decoder into audio.

![Model Architecture](stable-audio-3.png)

## SAME 
[![arXiv](https://img.shields.io/badge/arXiv-2605.18613-b31b1b.svg)](https://arxiv.org/abs/2605.18613)

SAME compresses 44.1 kHz stereo audio into a continuous latent space with total downsampling 4096x and a latent dimension of 256. For a 10-second clip, 2 channels × 441k samples compresses down to 216×256 (216 latents of 256 dimensions each).

SAME is designed to be useful in two mutually reinforcing ways. First, it is a high-fidelity autoencoder that preserves both low-level acoustic detail and high-level semantic content. Second, it is trained to produce a latent space that is structured and generatively tractable. Unlike autoencoders focused purely on reconstruction, SAME latents are easier for a generative model to learn from.

It is trained using a combination of five losses:

- **Spectral Reconstruction** — a phase-aware spectral loss that enforces perceptual fidelity to the original signal
- **Adversarial** — a GAN that pushes the model to reduce audible artifacts
- **Diffusion Alignment** — a small diffusion model trained alongside SAME to ensure its latents are well-suited for generation
- **Semantic Regression** — small regression models for pitch and stereo image
- **Constrastive Latent** - a text/audio contrastive critic that encourages the latents to encode rich, cross-modal meaning

There are two autoencoder variants:

| Model | Params | Attention | Latency (no optimizations) †
|---|---|---|---|
| SAME-S | 266M | Chunked w/ midpoint shift | 58ms
| SAME-L | 1.7B | Sliding window | 214ms

<sub>† Measured encoding/decoding of a 2min song with an Intel x86 + NVIDIA H100.</sub>


- **SAME-L** is the higher quality model and requires a GPU with sliding window attention support.
- **SAME-S** is a [distilled](https://labelbox.com/guides/model-distillation/) version of SAME-L designed for CPU and edge use. Besides being smaller, it uses something we call *modified chunked attention with midpoint shift* as a workaround for sliding window attention on CPU.

## Diffusion Transformer

The generative model in Stable Audio 3 is a conditional latent diffusion model that operates on SAME latents.

It accepts three conditions:

- **Text** — encoded using a [T5Gemma](https://deepmind.google/models/gemma/t5gemma/) model
- **Duration** — total audio length, encoded via sinusoidal embeddings
- **Inpainting** — a SAME-encoded audio clip with a start/end time, allowing a section to be filled in or extended



Training happens in three phases:

![Model Architecture](training-stages.png)

**1. Flow-Matching Pre-Training (base)**

We use flow-matching as our main training objective. The math can get a little complicated here, but put simply, we train a model to learn a trajectory from noise (randomness) to data (latents). One particularly cool feature is that we train with **variable-length diffusion**. Previously, if you just wanted to generate a short output, you will still have to generate a long sequence that would then be trimmed after generation, which sometimes could result in bad outputs. Now, if generating short sequences, it will understand that much better and also generate faster!

**2. Distillation Warmup**

In this stage, we learn to perform the full trajectory in one step. This effectively straightens it, but produces outputs that lack fine-grained detail. 

**3. Adversarial Post-Training**

The model then undergoes a final refinement stage to improve quality and reduce latency, producing the post-trained checkpoint used for inference. During this stage, a discriminator model with the same architecture as the pre-trained model is fine-tuned with three complementary losses:

- **Adversarial relativistic loss** — Trains the discriminator to tell if a latent is real or fake. Helps with perceptual quality.
- **Contrastive loss** - Regularizes latent space so that paired prompts and audios are close together. This helps the discriminator be semantically aligned.
- **[CLAP](https://github.com/LAION-AI/CLAP) loss** - Gives the generator an explicit text-alignment signal such that the generator improves both audio fidelity and prompt alignment.

There are four Diffusion Transformer variants:

| Model | Max Duration | Params | Autoencoder | Available |
|---|---|---|---|---|
| `small-music` | ~2min | 433M | SAME-S | This repo |
| `small-sfx` | ~2min | 433M | SAME-S | This repo |
| `medium` | ~4.75min | 1.4B | SAME-L | This repo |
| `large` | ~6.3min | 2.7B | SAME-L | [API only](https://stableaudio.com/) |

## Provided Checkpoints
Checkpoints aka weights are the saved model artifacts that you use for inference.

Three families of checkpoints are provided, each with Small and Medium variants:

| Key | Family | Purpose |
|---|---|---|
| `small-music`, `small-sfx`, `medium` | Post-trained | Primary inference checkpoints. Use these for generation. |
| `small-music-base`, `small-sfx-base`, `medium-base` | Base | Base checkpoints. Used as the starting point for LoRA training. |
| `same-s`, `same-l` | SAME | Standalone autoencoder checkpoints. Use these if you only need encoding/decoding without the Diffusion Transformer. |

Post-trained checkpoints have no suffix because they are the default choice for inference — the `-base` suffix distinguishes the pre-trained base checkpoints. SAME checkpoints will reuse a locally cached post-trained or base checkpoint automatically if one is already present, avoiding a redundant download.

## How inference works

At inference time, the Diffusion Transformer iteratively denoises noise into SAME latents conditioned on your text prompt and duration. The SAME decoder then reconstructs the latents into a full-resolution 44.1 kHz stereo waveform.


## LoRA

Stable Audio supports LoRA fine-tuning as an easy way to adapt models toward specific styles. See the [LoRA guide](../workflows/lora.md).

Note: LoRAs are trained on the base checkpoint. Once trained, they can be applied to the post-trained model and will work as expected.

## Training Data
All models were trained on a combination of licensed ([AudioSparx](https://www.audiosparx.com/)) and CC0 ([Freesound](https://freesound.org/) data)
