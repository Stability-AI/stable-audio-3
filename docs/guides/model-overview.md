# Stable Audio 3 Model
> For a more in-depth breakdown of Stable Audio 3, please see our tech report.

Stable Audio 3 is a family of text-conditioned audio generation models.

## Using the Model

Both configurations (Small and Medium) share the same interface — see the [model table](#) for hardware requirements and generation speed.

| Input | Description |
|---|---|
| `prompt` | Text description of the audio to generate |
| `duration` | Length of audio to generate, in seconds (max X seconds) |

| Output | Value |
|---|---|
| Format | 44.1 kHz stereo audio |
| Bit depth | 32-bit float |

**Limitations**
- Not designed for speech or voice generation
- Trained on English descriptions; other languages will underperform


## System Overview
There are two main pieces of the system: the SAME autoencoder and the diffusion-transformer (DiT).

**SAME Autoencoder**

SAME (Semantic-Acoustic Music Encoder) is a stereo autoencoder that compresses audio into 256-dimensional continuous latents (encoder) and reconstructs them back into audio (decoder). SAME is trained separately from the DiT.

**DiT**

The diffusion-transformer learns to generate SAME latents conditioned on inputs like text prompt and duration. These inputs are turned into embeddings that guide the model toward latents matching those conditions, which are then decoded by the SAME decoder into audio.

![alt text](image.png)

## SAME

SAME compresses 44.1 kHz stereo audio into a continuous latent space with a downsampling rate of 4096 and a latent dimension of 256. For a 10-second clip, 2 channels × 441k samples compresses down to 216×256 (216 latents of 256 dimensions each).

SAME is designed to be useful in two mutually reinforcing ways. First, it is a high-fidelity autoencoder that preserves both low-level acoustic detail and high-level semantic content. Second, it is trained to produce a latent space that is structured and generatively tractable. Unlike autoencoders focused purely on reconstruction, SAME latents are easier for a generative model to learn from.

It is trained using a combination of four losses:

- **Reconstruction** — a phase-aware spectral loss that enforces perceptual fidelity to the original signal
- **Adversarial** — a GAN that pushes the model to reduce audible artifacts
- **Diffusion alignment** — a small diffusion model trained alongside SAME to ensure its latents are well-suited for generation
- **Semantic** — small regression models for pitch and stereo image, plus a text/audio contrastive critic that encourages the latents to encode rich, cross-modal meaning

There are two variants:

| Model | Params | Attention | Latency (no optimizations) †
|---|---|---|---|
| SAME-S | 266M | Chunked w/ midpoint shift | 58ms
| SAME-L | 1.7B | Sliding window | 214ms

<sub>† Measured encoding/decoding of a 2min song with an Intel x86 + NVIDIA H100.</sub>


- **SAME-L** is the higher quality model and requires a GPU with sliding window attention support.
- **SAME-S** is a [distilled](https://labelbox.com/guides/model-distillation/) version of SAME-L designed for CPU and edge use. Besides being smaller, it uses something we call *modified chunked attention with midpoint shift* as a workaround for sliding window attention on CPU.

## DiT

The generative model in Stable Audio 3 is a conditional latent diffusion model that operates on SAME latents.

It accepts three conditions:

- **Text** — encoded using a [T5Gemma](https://deepmind.google/models/gemma/t5gemma/) model
- **Duration** — total audio length, encoded via sinusoidal embeddings
- **Inpainting** — a SAME-encoded audio clip with a start/end time, allowing a section to be filled in or extended

Training happens in two phases:

**1. Rectified flow (RF) training**

We use rectified flow aka flow-matching as our main training objective. The math can get a little complicated here, but put simply, we train a model to learn a trajectory from noise (randomness) to data (latents). One particularly cool feature is that we train with **variable-length diffusion**. Previously, if you just wanted to generate a short output, you will still have to generate a long sequence that would then be trimmed after generation, which sometimes could result in bad outputs. Now, if generating short sequences, it will understand that much better and also generate faster!

**2. ARC post-training**

After RF training, the model undergoes a final refinement stage to improve quality and reduce latency, producing the final checkpoint used for inference. 

There are three DiT variants:

| Model | Quality | Max Duration | Params | Autoencoder | Available |
|---|---|---|---|---|---|
| Small | Good | ~2min | 433M | SAME-S | This repo |
| Medium | High | ~4.75min | 1.4B | SAME-L | This repo |
| Large | Highest | ~6.3min | 2.7B | SAME-L | [API only](#) |

## Provided Checkpoints
Checkpoints aka weights are the saved model artifacts that you use for inference.

Three families of checkpoints are provided, each with Small and Medium variants:

| Key | Family | Purpose |
|---|---|---|
| `small`, `medium` | ARC | Primary inference checkpoints. Use these for generation. |
| `small-rf`, `medium-rf` | RF | Pre-ARC base checkpoints. Used as the starting point for LoRA training. |
| `same-s`, `same-l` | SAME | Standalone autoencoder checkpoints. Use these if you only need encoding/decoding without the DiT. |

ARC checkpoints have no suffix because they are the default choice for inference — the `-rf` suffix on the RF checkpoints distinguishes them as the earlier-stage base. SAME checkpoints will reuse a locally cached ARC or RF checkpoint automatically if one is already present, avoiding a redundant download.

## How inference works

At inference time, Stable Audio 3 turns your inputs (text prompt and duration) into audio through a two-stage process: latent generation with the DiT, followed by waveform reconstruction with SAME.

**1. Conditioning setup**
 
Your prompt is encoded into a dense embedding using T5Gemma, while the duration is converted into a sinusoidal embedding that represents the desired output length. If inpainting is used, the provided audio is first encoded into SAME latents and combined with a temporal mask indicating where generation should occur.

**2. Latent initialization**

A sequence of Gaussian noise is initialized in the SAME latent space. The length of this sequence is determined directly from the requested duration, thanks to the model’s variable-length training.

**3. Iterative denoising (DiT sampling)**

The DiT progressively transforms this noise into structured latents over a series of steps. At each step, the model predicts how to move the current latent state closer to a valid audio representation while staying consistent with the conditioning (text, duration, and optional inpainting signal). Because the model is trained with rectified flow and ARC, we can reduce the number of steps needed down to 8 while maintaining high quality. 

**4. Latent → audio decoding (SAME)**

Once the final latent sequence is produced, it is passed through the SAME decoder. This reconstructs a full-resolution 44.1 kHz stereo waveform, restoring both fine acoustic details and higher-level musical structure.


## LoRA

Stable Audio supports LoRA fine-tuning as an easy way to adapt models toward specific styles. See the [LoRA guide](docs/workflows/lora.md).

Note: LoRAs are trained on the RF base checkpoint. Once trained, they can be imported onto the ARC model and will work as expected.

## Training Data
All models were trained on a combination of licensed ([AudioSparx](https://www.audiosparx.com/)) and CC0 ([Freesound](https://freesound.org/) data)
