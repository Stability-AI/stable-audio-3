import os
import tempfile
from typing import Optional

import gradio as gr
import torch
import torchaudio

from stable_audio_3 import StableAudioModel


MODEL_CHOICES = [
    "medium",
    "small-music",
    "small-sfx",
    "medium-base",
    "small-music-base",
    "small-sfx-base",
]


def _to_audio_tuple(audio_path: Optional[str]):
    if not audio_path:
        return None
    waveform, sr = torchaudio.load(audio_path)
    return (sr, waveform)


def _normalize_optional_text(s: str):
    s = (s or "").strip()
    return s if s else None


def _parse_float_list(csv_text: str):
    text = (csv_text or "").strip()
    if not text:
        return None
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        return None
    parsed = [float(v) for v in values]
    return parsed[0] if len(parsed) == 1 else parsed


def _build_ui(model_name: str, device: Optional[str], no_half: bool):
    state = {"model": None, "loaded": None}

    def ensure_model(current_model_name: str, current_device: str, current_no_half: bool):
        target = (current_model_name, current_device or None, current_no_half)
        if state["loaded"] != target:
            state["model"] = StableAudioModel.from_pretrained(
                current_model_name,
                device=(current_device or None),
                model_half=not current_no_half,
            )
            state["loaded"] = target
        return state["model"]

    def generate(
        model_name,
        device,
        no_half,
        prompt,
        negative_prompt,
        duration,
        steps,
        cfg_scale,
        seed,
        batch_size,
        init_audio,
        init_noise_level,
        inpaint_audio,
        inpaint_starts_csv,
        inpaint_ends_csv,
        chunked_decode_mode,
        lora_ckpt_paths_csv,
        lora_strength,
        lora_index,
        output_basename,
    ):
        prompt = _normalize_optional_text(prompt)
        if not prompt:
            raise gr.Error("Prompt is required")

        model = ensure_model(model_name, device, no_half)

        lora_paths = [p.strip() for p in (lora_ckpt_paths_csv or "").split(",") if p.strip()]
        if lora_paths:
            model.load_lora(lora_paths)
        if lora_strength is not None:
            model.set_lora_strength(lora_strength, lora_index=int(lora_index) if lora_index >= 0 else None)

        inpaint_starts = _parse_float_list(inpaint_starts_csv)
        inpaint_ends = _parse_float_list(inpaint_ends_csv)
        if (inpaint_starts is None) != (inpaint_ends is None):
            raise gr.Error("inpaint-start and inpaint-end must both be set")

        chunked_decode = None
        if chunked_decode_mode == "on":
            chunked_decode = True
        elif chunked_decode_mode == "off":
            chunked_decode = False

        init_audio_tuple = _to_audio_tuple(init_audio)
        inpaint_audio_tuple = _to_audio_tuple(inpaint_audio)

        audio = model.generate(
            prompt=prompt,
            negative_prompt=_normalize_optional_text(negative_prompt),
            duration=float(duration),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            batch_size=int(batch_size),
            init_audio=init_audio_tuple,
            init_noise_level=float(init_noise_level),
            inpaint_audio=inpaint_audio_tuple,
            inpaint_mask_start_seconds=inpaint_starts,
            inpaint_mask_end_seconds=inpaint_ends,
            chunked_decode=chunked_decode,
        )

        sr = model.model.sample_rate
        first = audio[0].detach().cpu()

        safe_name = (output_basename or "output").strip() or "output"
        fd, out_path = tempfile.mkstemp(prefix=f"{safe_name}_", suffix=".wav")
        os.close(fd)
        torchaudio.save(out_path, first, sr)

        return (sr, first.numpy().T), out_path

    with gr.Blocks(title="Stable Audio 3 Basic CLI WebUI") as demo:
        gr.Markdown("# Stable Audio 3 — Basic CLI WebUI\nMatches the main stable-audio CLI options in a minimal form.")

        with gr.Row():
            model_dd = gr.Dropdown(MODEL_CHOICES, value=model_name, label="--model")
            device_tb = gr.Textbox(value=device or "", label="--device (optional: cuda/mps/cpu)")
            no_half_cb = gr.Checkbox(value=no_half, label="--no-half")

        prompt_tb = gr.Textbox(label="--prompt", lines=3, placeholder="Describe the audio...")
        negative_prompt_tb = gr.Textbox(label="--negative-prompt", lines=2)

        with gr.Row():
            duration_num = gr.Number(value=30, label="--duration")
            steps_num = gr.Number(value=8, label="--steps", precision=0)
            cfg_scale_num = gr.Number(value=1.0, label="--cfg-scale")
            seed_num = gr.Number(value=-1, label="--seed", precision=0)
            batch_size_num = gr.Number(value=1, label="--batch-size", precision=0)

        with gr.Accordion("Audio-to-audio", open=False):
            init_audio_in = gr.Audio(type="filepath", label="--init-audio")
            init_noise_num = gr.Number(value=0.9, label="--init-noise-level")

        with gr.Accordion("Inpainting / continuation", open=False):
            inpaint_audio_in = gr.Audio(type="filepath", label="--inpaint-audio")
            inpaint_starts_tb = gr.Textbox(label="--inpaint-start (comma-separated)", placeholder="4,16")
            inpaint_ends_tb = gr.Textbox(label="--inpaint-end (comma-separated)", placeholder="8,20")

        with gr.Accordion("Decode + LoRA", open=False):
            chunked_decode_mode = gr.Radio(
                ["auto", "on", "off"],
                value="auto",
                label="chunked decode (--chunked-decode / --no-chunked-decode)",
            )
            lora_paths_tb = gr.Textbox(
                label="--lora-ckpt-path (comma-separated)",
                placeholder="/path/a.safetensors,/path/b.safetensors",
            )
            with gr.Row():
                lora_strength_num = gr.Number(value=None, label="--lora-strength")
                lora_index_num = gr.Number(value=-1, label="--lora-index (-1 = all)", precision=0)

        output_basename_tb = gr.Textbox(value="output", label="--output basename")
        run_btn = gr.Button("Generate", variant="primary")

        audio_out = gr.Audio(label="Generated audio")
        file_out = gr.File(label="Download WAV")

        run_btn.click(
            generate,
            inputs=[
                model_dd,
                device_tb,
                no_half_cb,
                prompt_tb,
                negative_prompt_tb,
                duration_num,
                steps_num,
                cfg_scale_num,
                seed_num,
                batch_size_num,
                init_audio_in,
                init_noise_num,
                inpaint_audio_in,
                inpaint_starts_tb,
                inpaint_ends_tb,
                chunked_decode_mode,
                lora_paths_tb,
                lora_strength_num,
                lora_index_num,
                output_basename_tb,
            ],
            outputs=[audio_out, file_out],
        )

    return demo


def launch_basic_cli_webui(model_name: str = "medium", device: Optional[str] = None, no_half: bool = False):
    demo = _build_ui(model_name=model_name, device=device, no_half=no_half)
    demo.queue()
    demo.launch(share=True)
