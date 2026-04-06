# Inference Methods
An overview of the different inference modes. 
> New to diffusion/RF models? See [Model Overview](../guides/how-inference-works.md)
> for a conceptual overview before diving in.

The python interface is shown, but these controls are the same as for the gradio interface


## Text-to-Audio
The most common usage is generating audio from text
```
from stable-audio-3 import StableAudio
audio = StableAudio.generate(
    prompt="120 BPM house loop", 
    negative_prompt="poor quality",
    duration=30,
    steps=8,
    cfg_scale=1
)

```

## Controls
Overview of the main controls

- **`prompt`** — Text description of the audio to generate (e.g. `"120 BPM house loop"`). For help crafting good prompts, see [Prompt Guide](../guides/prompting.md)
- **`negative_prompt`** — Text description of qualities to avoid in the output. Steers generation away from unwanted characteristics.
- **`duration`** — Duration of the generated audio in seconds (default: `120`).
- **`steps`** — Number of sampling steps (default: `8`). With ARC trained-models (default), you generally don't want to go higher than 8. For even faster inference, reduce this number, which may reduce audio quality.
- **`cfg_scale`** — Classifier-free guidance scale (default: `1.0`). Higher values make the output adhere more closely to the prompt; lower values give the model more creative freedom.

## Audio-To-Audio
Using init audio, you can edit an existing recording to
change the style, genres and mood to create variations.

```
from stable-audio-3 import StableAudio
audio = StableAudio.generate(
    init_audio=/path/to/some/audio.wav
    init_noise_level=0.9,
    prompt="", 
    negative_prompt="poor quality",
    duration=30,
    steps=8,
    cfg_scale=1
)
```

## Controls
- **`init_noise_level`** — Controls how much the init audio influences the output (range: `0.0`–`1.0`, default: `0.9`). At `1.0` the init audio is fully replaced by noise and has no effect. Lower values preserve more of the original, for example `0.9` produces a close variation, while `0.5` is a halfway blend between the original and pure generation.

The other controls for text to audio are the same, however the `prompt` is now used to control how the audio will be edited. The [Prompt Guide](../guides/prompting.md) has some examples for this

## Inpainting (New to Stable Audio 3!)
Inpainting lets you regenerate a specific region of an existing audio file while keeping the rest intact — useful for fixing a section, swapping out a sound, or extending a loop.

```
from stable-audio-3 import StableAudio
audio = StableAudio.generate(
    inpaint_audio=/path/to/some/audio.wav,
    inpaint_mask_start_seconds=4.0,
    inpaint_mask_end_seconds=8.0,
    prompt="punchy kick drum fill",
    duration=30,
    steps=8,
    cfg_scale=1
)
```

## Controls

- **`inpaint_audio`** — The source audio file to inpaint into. The region outside the mask is preserved; only the masked region is regenerated.
- **`inpaint_mask_start_seconds`** — Start of the region to regenerate, in seconds.
- **`inpaint_mask_end_seconds`** — End of the region to regenerate, in seconds.

The other controls for text to audio are the same, however the `prompt` is now used to control how the audio will be inpainted. The [Prompt Guide](../guides/prompting.md) has some examples for this

