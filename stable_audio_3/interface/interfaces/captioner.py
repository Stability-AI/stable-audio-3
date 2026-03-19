import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ...interface.aeiou import audio_spectrogram_image
from ...inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint, generate_diffusion_uncond
from ...models.factory import create_model_from_config
# from ...models.pretrained import get_pretrained_model
from ...models.utils import copy_state_dict, load_ckpt_state_dict
from ...inference.utils import prepare_audio

model = None
model_type = None
model_half = False
sample_rate = 32000
sample_size = 1920000

def generate_captioner(
    audio,
    prompt,
    temperature=1.0,
    top_p=0.95,
    top_k=0,    
    batch_size=1,
    ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    print(f"Input audio shape: {audio.shape}")

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    if model_half:
        audio = audio.to(torch.float16)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.to(device)

    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio.device).to(audio.dtype)
        audio = resample_tf(audio)

    audio_length = audio.shape[-1]

    if audio_length > sample_size:
        audio = audio[:, :sample_size]

    caption = model.generate_caption(
        audio.unsqueeze(0), # Add batch dimension
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        max_new_tokens=128
    )[0]

    return (caption)

def create_captioner_ui(model_config, in_model, in_model_half=False):
    global model, sample_size, sample_rate, model_half

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]

    model_half = in_model_half

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
        prompt_textbox = gr.Textbox(label="Prompt", placeholder="Prompt")
        output_caption = gr.Textbox(label="Output caption", interactive=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.7, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_captioner, 
            inputs=[
                input_audio,
                prompt_textbox,
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_caption],
            api_name="generate"
        )

    return ui

