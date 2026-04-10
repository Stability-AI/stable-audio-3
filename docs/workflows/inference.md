# Inference Methods
An overview of the different inference modes. 
> New to diffusion/RF models? See [Model Overview](../guides/how-inference-works.md)
> for a conceptual overview before diving in.

The python interface is shown, but these controls are the same as for the gradio interface


## Text-to-Audio
The most common usage is generating audio from text
```python
from stable_audio_3 import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(".")
audio = pipe.generate(
    prompt="120 BPM house loop", 
    negative_prompt="poor quality",
    duration=30,
    steps=8, # default
    cfg_scale=1, # default
    batch_size=1 # default
)
```

## Controls
Overview of the main controls

- **`prompt`** — Text description of the audio to generate (e.g. `"120 BPM house loop"`). For help crafting good prompts, see [Prompt Guide](../guides/prompting.md)
- **`negative_prompt`** — Text description of qualities to avoid in the output. Steers generation away from unwanted characteristics.
- **`duration`** — Duration of the generated audio in seconds (default: `120`).
- **`steps`** — Number of sampling steps (default: `8`). With ARC trained-models (default), you generally don't want to go higher than 8. For even faster inference, reduce this number, which may reduce audio quality.
- **`cfg_scale`** — Classifier-free guidance scale (default: `1.0`). Higher values make the output adhere more closely to the prompt; lower values give the model more creative freedom.
- **`batch_size`** - Generate multiple at once, useful is you have a GPU and want to get a lot of variations. The max is limited by your GPU's VRAM.

## Audio-To-Audio
Using init audio, you can edit an existing recording to change the style, genres and mood to create variations. Use the prompt to control the variation.

```python
import torchaudio
from stable_audio_3 import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(".")
init_audio = torchaudio.load("/path/to/some/audio.wav")
audio = pipe.generate(
    init_audio=init_audio,
    init_noise_level=0.9,
    prompt="bossa nova bassline",
    duration=30,
)
```


## Controls
- **`init_noise_level`** — Controls how much the init audio influences the output (range: `0.0`–`1.0`, default: `1.0`). At `1.0` the init audio is fully replaced by noise and has no effect (pure generation). Lower values preserve more of the original — for example `0.1` produces a close variation, while `0.5` is a halfway blend between the original and pure generation.

The other controls for text to audio are the same, however the `prompt` is now used to control how the audio will be edited. The [Prompt Guide](../guides/prompting.md) has some examples for this

## Inpainting/Outpainting (New to Stable Audio 3!)
Inpainting lets you regenerate a specific region of an existing audio file while keeping the rest intact — useful for fixing a section, swapping out a sound, or extending a loop.

```python
import torchaudio
from stable_audio_3 import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(".")
inpaint_audio = torchaudio.load("/path/to/some/audio.wav")
audio = pipe.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy kick drum fill",
    duration=30,
)
```

You can also *extend* an audio by performing outpainting. Simply choose a duration that is longer than your `inpaint_audio` and set `mask_start_seconds` to be the length of your audio file.

```python
import torchaudio
from stable_audio_3 import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(".")
inpaint_audio = torchaudio.load("/path/to/some/audio.wav") # Assume this is 10s long
audio = pipe.generate(
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
from stable_audio_3 import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained(".")
inpaint_audio = torchaudio.load("/path/to/some/audio1.wav")

audio = pipe.generate(
    inpaint_audio=inpaint_audio,
    inpaint_mask_start_seconds=3
    inpaint_mask_end_seconds=10
    prompt=["punchy kick drum fill", "guitar", "trumpet", "piano"]
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