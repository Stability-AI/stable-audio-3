# Stable Audio 3 Inference Methods
An overview of the different inference modes. The python interface is shown, but these controls are the same as for the gradio interface

> New to diffusion/RF models? See [Model Overview](../guides/how-inference-works.md)
> for a conceptual overview before diving in.

## Loading the Model

```python
from stable_audio_3 import StableAudioModel
model = StableAudioModel.from_pretrained("medium", device="cuda")  # device is optional, defaults to cuda → mps → cpu
```

The first argument selects the model to load. Available models:

| Model | Type |
|---|---|
| `medium` | Post-trained |
| `small-music` | Post-trained |
| `small-sfx` | Post-trained |
| `medium-base` | Base |
| `small-music-base` | Base |
| `small-sfx-base` | Base |

> **Note:** `medium` and `medium-base` require a CUDA GPU with Flash Attention support.

## Text-to-Audio
The most common usage is generating audio from text
```python
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
audio = model.generate(
    prompt="120 BPM house loop", 
    negative_prompt="poor quality",
    duration=30,
    steps=8, # default
    cfg_scale=1, # default
    seed=-1, # default
    batch_size=1 # default
)
```

## Controls
Overview of the main controls

- **`prompt`** — Text description of the audio to generate (e.g. `"120 BPM house loop"`). For help crafting good prompts, see [Prompt Guide](../guides/prompting.md)
- **`duration`** — Duration of the generated audio in seconds (default: `120`).
- **`steps`** — Number of sampling steps (default: `8`). For faster inference, reduce this number at some cost to quality. Higher values (e.g. `50`) can improve quality at the cost of speed.
- **`seed`** - Random seed for reproducible outputs if needed. Use -1 to select a random seed (default) or select your favorite number for deterministic results.
- **`batch_size`** - Generate multiple at once, useful is you have a GPU and want to get a lot of variations. The max is limited by your GPU's VRAM.
- **`cfg_scale`** — Classifier-free guidance scale (default: `1.0`; try `7.0` for stronger prompt adherence). Higher values make the output adhere more closely to the prompt; lower values give the model more creative freedom.
- **`negative_prompt`** — Text description of qualities to avoid in the output. Steers generation away from unwanted characteristics.

## Audio-To-Audio
Using init audio, you can edit an existing recording to change the style, genres and mood to create variations. Use the prompt to control the variation.

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
init_audio = torchaudio.load("/path/to/some/audio.wav")
audio = model.generate(
    init_audio=init_audio,
    init_noise_level=0.9,
    prompt="bossa nova bassline",
    duration=30,
)
```


## Controls
- **`init_audio`** - The source audio as a `(sample_rate, tensor)` tuple (e.g. from `torchaudio.load()`). The audio will be noised and then denoised.
- **`init_noise_level`** — Controls how much the init audio influences the output (range: `0.0`–`1.0`, default: `1.0`). At `1.0` the init audio is fully replaced by noise and has no effect (pure generation). Lower values preserve more of the original — for example `0.1` produces a close variation, while `0.5` is a halfway blend between the original and pure generation.

The other controls for text to audio are the same, however the `prompt` is now used to control how the audio will be edited. The [Prompt Guide](../guides/prompting.md) has some examples for this

## Inpainting/Continuation
Inpainting lets you regenerate a specific region of an existing audio file while keeping the rest intact, useful for fixing a section, swapping out a sound, or extending a loop.

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
inpaint_audio = torchaudio.load("/path/to/some/audio.wav")
audio = model.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy kick drum fill",
    duration=30,
)
```

You can also *extend* an audio by performing continuation. Simply choose a duration that is longer than your `inpaint_audio` and set `mask_start_seconds` to be the length of your audio file.

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
inpaint_audio = torchaudio.load("/path/to/some/audio.wav") # Assume this is 10s long
audio = model.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=10.0,
    inpaint_mask_end_seconds=18.0,
    prompt="punchy kick drum fill",
)
```

## Controls

- **`inpaint_audio`** — The source audio as a `(sample_rate, tensor)` tuple (e.g. from `torchaudio.load()`). The region outside the mask is preserved; only the masked region is regenerated.
- **`inpaint_mask_start_seconds`** — Start of the region to regenerate, in seconds.
- **`inpaint_mask_end_seconds`** — End of the region to regenerate, in seconds.

The other controls for text to audio are the same, however the `prompt` is now used to control how the audio will be inpainted. The [Prompt Guide](../guides/prompting.md) has some examples for this


# Per-batch customization
When using batch size > 1, certain controls can be customized per-batch.
For example, with batch_size=4:

```python
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
inpaint_audio = torchaudio.load("/path/to/some/audio1.wav")

audio = model.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=3
    inpaint_mask_end_seconds=10
    prompt=["prompt1", "prompt2", "prompt3", "prompt4"]
    duration=[30, 25, 20, 20],
    steps=8,
    cfg_scale=1,
    batch_size=4
)

```

This currently works for the following parameters:
- `prompt`
- `negative prompt`
- `duration`

# LoRA
## Inference with LoRA

Load one or more LoRA checkpoints onto the model before generating:

```python
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
model.load_lora(["path/to/lora.safetensors"])

audio = model.generate(
    prompt="Lo-fi boom bap meets orchestral strings 84 BPM",
    duration=30,
)
```

Multiple LoRAs can be stacked by passing additional paths:

```python
model.load_lora(["style_a.safetensors", "style_b.safetensors"])
```

### Adjusting LoRA strength

Control how strongly the LoRA influences the output at runtime:

```python
model.set_lora_strength(0.5)              # Half-strength on all LoRAs
model.set_lora_strength(1.5)              # Amplify the effect
model.set_lora_strength(0.0)              # Disable without unloading

# With multiple LoRAs, target by index:
model.set_lora_strength(1.0, lora_index=0)
model.set_lora_strength(0.3, lora_index=1)

# Target only the DiT backbone or conditioner independently:
model.set_lora_strength(1.0, target="dit")
model.set_lora_strength(0.0, target="conditioner")
```

For full details on LoRA training see [LoRA Training](lora.md).
