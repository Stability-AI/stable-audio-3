# Prompt Guide

This guide provides tips and tricks for getting the most out of your prompts, and helps you break out of prompt "writer's block." Prompting is more of an art than a science, but there are some best practices we've discovered. This is just a starting point, be creative and go crazy (perhaps share in our [Harmonai Discord](https://discord.gg/cKpvjey8b))!

---

## Generation Modes

This guide covers the following generation modes:

- **Music** — full instrumental tracks, ambient audio
- **Stems & Solo Instruments** — isolated parts or single-instrument pieces
- **Audio Samples & SFX** — sound effects, samples, and one-shots
- **Init Audio, Inpainting & Continuation** — modify or extend existing audio using a seed file

### Model Compatibility

| Model | Music | Stems/Solo | Samples/SFX |
|-------|-------|------------|-------------|
| `medium` | ✓ | ✓ | ✓ |
| `small-music` | ✓ | ✓ | — |
| `small-sfx` | — | — | ✓ |

For additional tips, visit the [Stable Audio User Guide](https://stability.ai/stable-audio).

---

## General Tips

- **Think about the training datasets.** Prompt adherence and audio quality are closely linked with the dataset the model was trained on. Stable Audio 3 uses audio from Freesound and AudioSparx along with their metadata. Prompts that align with that kind of text and audio tend to produce the best results, though out-of-distribution prompts can be interesting to explore.
- **Set a realistic duration.** While our models can generate up to 6m 20s (`medium`) or 2m (`small`), results tend to be better when you choose a duration that fits what you're describing.
- **Embrace the randomness.** The model won't output the same thing twice. If you want a reproducible result, set a seed (in Gradio, this is under `Sampler Params`).

### Not Sure Where to Start?

- Try one of the example prompts below as a jumping-off point
- Run an unconditional (no prompt) generation — the results might surprise or inspire you
- Use the **Gradio Prompt Assistant** (even with an empty text box) — it will generate or refine a prompt for you. Short prompts especially benefit from this, since more detail generally means better results

![alt text](gradio.png)

---

## Music Prompting

*For instrumental or ambient music.*

### Key Elements to Consider

- **Genre** — What style of music is it?
- **Instruments** — What's playing, and how does each instrument sound or feel?
- **Mood & energy** — What emotions or atmosphere are you going for?
- **BPM** — Specify tempo, e.g. *"120 BPM"*

### Example Prompts

> A triumphant and stylish UK bass-flavoured tech-house tune that evokes feelings of the last tune played in a DJ set. The pumping four-to-the-floor kick is supported by an 808 bass that is syncopated. There are gliding emotional synth leads that build sections to their climax. Playful stabs and chops support the rhythm of the drums in sections. There is a beautiful gospel house piano that plays in the drop, giving the track a euphoric feeling.

> A quirky alternative pop ballad instrumental with the vibe of a long car journey with friends. The catchy synth bass is infectious, the tightly tuned acoustic drums add a classy flair, and the dreamy colourful synths bring a fuzzy old VHS tape atmosphere. At the same time, subtle guitars play indie rock-style motifs. The track is deeply nostalgic with a 90s internet aesthetic and a sense of something lost to time — brooding, bold, youthful, and melancholic.

> A funky hip hop instrumental with live recorded instrumentation that has the vibe of a 70's TV show theme. The rhythm guitar strums lightly in the background, the lead guitar occasionally delivers jagged flanger-effected chords and phrases, and abstract sounds punctuate the beat with character. A close-mic'd electric piano plays hard with nostalgic supporting chords, flute glides over the beat adding an exciting texture, and the drums are full of swing and old school flavour. The production has a warm, textured sound associated with analogue gear and tape.

### Helpful AudioSparx Tags

- `TrackType: Music, VocalType: Instrumental` — tends to produce higher quality, more semantically coherent outputs
- `Genre: Funk` — you can specify multiple genres like `Genre: Funk, Genre: Jazz`, or even combine unrelated genres
- `Instruments: Guitar, Saxophone, Bass, Piano`

> ***On vocals**: while our models don't output intelligible vocals, they sometimes generate unintelligible vocal textures, which can be an interesting artifact to play with.*

---

## Stem & Solo Instrument Prompting

*For an individual instrument, stem, or duet.*

### Key Elements to Consider

- Start with `TrackType: Instrument` to maximize your chances of getting an isolated instrument
- **Instrument/Stem** — specify what's playing
- **Genre** — what style of music is the instrument playing in?
- **Mood & energy** — what emotions or atmosphere are you going for?
- **BPM** — specify tempo, e.g. *"120 BPM"*

### Example Prompts

> TrackType: Instrument, a sombre solo acoustic guitar track with cavernous reverb and delicate finger picking.

> TrackType: Instrument, Format: Duo, a dynamic Latin drums and percussion track.

> TrackType: Instrument, an epic solo electric guitar track, live-recorded in a vintage studio.

### Helpful AudioSparx Tags

- `Genre: Funk` — you can specify multiple genres like `Genre: Funk, Genre: Jazz`
- `Format: Duo` — for when you want multiple instruments

> *Tip: Experiment with specifying technique, recording environment, and effects, as many of these descriptors are part of the dataset!*

---

## Audio Samples & SFX

*For sound effects, one-shots, and samples.*

### Key Elements to Consider

- **The core source** — exactly what object, instrument, or synthesizer is making the sound?
- **The action** — how is the sound being triggered, and how long does it last? *(e.g., slamming shut with fast decay, or a massive suck-back followed by a supersonic crack)*
- **The production/characteristics** — where is the mic placed, and what's the room character? How is the sound processed? *(e.g., recorded in a dead room, captured with a vintage ribbon mic, or processed with a dark wooden reverb)*

### Example Prompts

> A blunt, powerful "thud" made by slamming a wooden desk drawer shut. It has a pronounced low-mid body, making it feel heavy, and is given a touch of analog distortion for aggressive character.

> A massive sub-bass note that mimics a vinyl record or tape machine being turned off. The pitch and speed drop simultaneously, causing the high-end harmonics to "smear" and thicken as the sound grinds to a halt at a sub-sonic frequency.

> A classic, natural recording of 14-inch hi-hats played tightly closed. Zero ring and a very fast decay.

### Helpful AudioSparx Tags

- `TrackType: SFX` — tends to produce more semantically reasonable sound effects and samples

> *Tip: Set a short duration. Most sound effects are brief, so set the length accordingly before generating.*

---

## Init Audio, Inpainting & Continuation

If you have existing audio, use these tools to modify it. You can also use a previously generated audio as input. Tips from the previous sections still apply. For example, `TrackType: SFX` can still help when crafting sound effects via these methods.

### Init Audio (Audio-to-Audio)

*For when you have an existing audio you want to use as a seed.*

Instead of starting with noise, audio-to-audio uses your audio as the starting point for generation. Paired with a text prompt, you can remix, style transfer, and edit your audio.

Use `init_noise_level` to control how much the original audio is modified. Think of it like a filter on audio features — as you increase the amount, more lower-level features like melody and rhythm get filtered out, leaving higher-level features like timbre and tonality.

#### Example Applications

*Note: a reasonable `init_noise_level` is highly dependent on your audio and prompt, so expect some tinkering.*

**Timbre transfer** — change the instrument while preserving the notes:
- Input: `violin_solo.wav` | Prompt: *Heavy metal guitar* | Noise level: `0.4`
- Input: `beatbox.wav` | Prompt: *Lofi hip hop beat, chillhop* | Noise level: `0.5`

**Style transfer** — shift the overall style of a track:
- Input: `trance.wav` | Noise level: `0.6`
- Prompt: *A cinematic outlaw country instrumental featuring blues pedal steel guitar, rustic mandolin, fiddle playing call and response, a tape-driven rattly drum-kit, an autoharp, and a soaring accordion solo.*

### Inpainting & Continuation

*For re-creating a section or extending your audio.*

Like audio-to-audio, inpainting uses a seed audio, but instead of generating a completely new sound, it only generates audio within a specified region, preserving everything outside it. The prompt and surrounding context guide generation while maintaining continuity at the transitions.

Set your region using `inpaint_mask_start_seconds` and `inpaint_mask_end_seconds` (or **mask start/end** in Gradio). For continuations, set the mask start to the current end of your audio and mask end to where you want the extension to finish.

#### Examples

- Input: `classical.wav` (120s) | Prompt: *An electric guitar solo live-recorded in a vintage studio* | Mask: `30s → 120s`
  → The classical piece gradually morphs into a guitar solo with the original as background
- Input: `bird_call.wav` (20s) | No prompt | Mask: `20s → 100s`
  → A continuation of the bird call, extending to 100s total

#### Tips

- **The audio context matters more than the prompt.** Masking only a few seconds of a song may produce output very close to the original since the model has too much context to deviate. Try initially generating with a large amount of the audio masked and reduce the mask in further generations.
- **Prompts work best when plausible given the surrounding context.** A progressive house track with a prompt of *"Genre: dubstep buildup and drop"* is far more likely to succeed than *"cat meow."*
