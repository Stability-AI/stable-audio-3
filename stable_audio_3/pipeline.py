import json
import torch
import gc
import numpy as np
from torchaudio import transforms as T
from stable_audio_3.interface.gradio import load_model
from stable_audio_3.models.minlora import set_lora_strength 
from stable_audio_3.inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint #, generate_diffusion_uncond


MODEL_CONFIG_PATH = "44k_taae_4096_256_2_ct_dit_cond_t5gemma_380s_asx_large2_inpaint_ot_varlen_muon.json"
CKPT_PATH = "4096_256_taaev2_rc2_base_cond_t5gemma_380s_inpaint_asxmi_varlen_muon_82k_unwrap.ckpt"
model_half = True
class StableAudioPipeline:
    def __init__(self, model, model_config, device):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.same = self.model.pretransform
        self.dit = self.model.model
    
    @torch.inference_mode()
    def generate(self, prompt, duration=120, **kwargs):
        result = self.generate_cond(prompt, duration)
        print(f"Generated audio with shape {result.shape} and dtype {result.dtype}")
        return result[0].float()
        

    @staticmethod
    def from_pretrained(model_name_or_path):
        # Load the model and any necessary components here
        ## TODO: Work with HuggingFace Hub to load models
        with open(MODEL_CONFIG_PATH) as f:
            model_config = json.load(f)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_config = load_model(model_config, CKPT_PATH, model_half=model_half, device=device)
        return StableAudioPipeline(model, model_config, device)

    def generate_cond(self, prompt, duration=120, negative_prompt=None, **kwargs):
        seed = 124
        batch_size = 1
        inversion_steps=100
        inversion_gamma=0.3
        inversion_unconditional=False        
        sampler_type="euler"
        sigma_min=0.03
        sigma_max=1000
        rho=1.0
        cfg_scale=6.0
        cfg_interval_min=0.0
        cfg_interval_max=1.0
        cfg_rescale=0.0
        cfg_norm_threshold=0.0
        apg_scale=1.0
        init_audio=None
        init_noise_level=1.0
        mask_maskstart=None
        mask_maskend=None
        inpaint_audio=None
        init_audio_type="Init audio"
        inversion_steps=100
        inversion_gamma=0.3
        inversion_unconditional=False
        adapt_duration_to_conditioning=False
        duration_padding_sec=6.0
        use_effective_length_for_schedule=False
        batch_size=1
        dist_shift=None
        steps=100
        mask_padding_attention = False
        n_loras = 0
        lora_args = []


        sample_rate = self.model_config["sample_rate"]
        sample_size = self.model_config["sample_size"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


        # Return fake stereo audio
        conditioning_dict = {"prompt": prompt, "seconds_start": 0, "seconds_total": duration}

        conditioning = [conditioning_dict] * batch_size

        if negative_prompt:
            negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": 0, "seconds_total": duration}

            negative_conditioning = [negative_conditioning_dict] * batch_size
        else:
            negative_conditioning = None
            
        #Get the device from the model
        device = next(self.model.parameters()).device

        seed = int(seed)
        # if seed is -1, define the seed value now, randomly, so we can save it in the filename
        if(seed==-1):
            seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)
        
        # Parse per-LoRA controls from trailing args
        # Each LoRA has 5 controls: dit_strength, cond_strength, interval_min, interval_max, layer_filter
        lora_configs = None
        if n_loras > 0 and len(lora_args) >= n_loras * 5:
            lora_configs = []
            for i in range(n_loras):
                off = i * 5
                dit_strength = lora_args[off]
                cond_strength = lora_args[off + 1]
                interval_min = lora_args[off + 2]
                interval_max = lora_args[off + 3]
                layer_filter = lora_args[off + 4]
                set_lora_strength(self.model.model, dit_strength, lora_index=i)
                set_lora_strength(self.model.conditioner, cond_strength, lora_index=i)
                lora_configs.append({
                    "lora_index": i,
                    "interval": (interval_min, interval_max),
                    "layer_filter": layer_filter,
                })

        input_sample_size = sample_size

        if init_audio is not None:
            in_sr, init_audio = init_audio

            if init_audio.dtype == np.float32:
                init_audio = torch.from_numpy(init_audio)
            elif init_audio.dtype == np.int16:
                init_audio = torch.from_numpy(init_audio).float().div(32767)
            elif init_audio.dtype == np.int32:
                init_audio = torch.from_numpy(init_audio).float().div(2147483647)
            else:
                raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")

            if model_half:
                init_audio = init_audio.to(torch.float16)
            
            if init_audio.dim() == 1:
                init_audio = init_audio.unsqueeze(0) # [1, n]
            elif init_audio.dim() == 2:
                init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

            if in_sr != sample_rate:
                resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device).to(init_audio.dtype)
                init_audio = resample_tf(init_audio)

            audio_length = init_audio.shape[-1]

            if audio_length > sample_size:

                #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
                init_audio = init_audio[:, :sample_size]

            init_audio = (sample_rate, init_audio)

        if inpaint_audio is not None:
            in_sr, inpaint_audio = inpaint_audio
            
            if inpaint_audio.dtype == np.float32:
                inpaint_audio = torch.from_numpy(inpaint_audio)
            elif inpaint_audio.dtype == np.int16:
                inpaint_audio = torch.from_numpy(inpaint_audio).float().div(32767)
            elif inpaint_audio.dtype == np.int32:
                inpaint_audio = torch.from_numpy(inpaint_audio).float().div(2147483647)
            else:
                raise ValueError(f"Unsupported audio data type: {inpaint_audio.dtype}")

            if model_half:
                inpaint_audio = inpaint_audio.to(torch.float16)
            
            if inpaint_audio.dim() == 1:
                inpaint_audio = inpaint_audio.unsqueeze(0) # [1, n]
            elif inpaint_audio.dim() == 2:
                inpaint_audio = inpaint_audio.transpose(0, 1) # [n, 2] -> [2, n]

            if in_sr != sample_rate:
                resample_tf = T.Resample(in_sr, sample_rate).to(inpaint_audio.device).to(inpaint_audio.dtype)
                inpaint_audio = resample_tf(inpaint_audio)

            audio_length = inpaint_audio.shape[-1]

            if audio_length > sample_size:

                #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
                inpaint_audio = inpaint_audio[:, :sample_size]

            inpaint_audio = (sample_rate, inpaint_audio)


        if init_audio_type == "RF-Inversion":
            inversion_params = {
                "inversion_steps": inversion_steps,
                "inversion_gamma": inversion_gamma,
                "inversion_unconditional": inversion_unconditional,
                "inversion_cfg_scale": 1.0,
                "inversion_sigma_max": 1.0
            }
        else:
            inversion_params = None

        generate_args = {
            "model": self.model,
            "conditioning": conditioning,
            "negative_conditioning": negative_conditioning,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "cfg_interval": (cfg_interval_min, cfg_interval_max),
            "lora_configs": lora_configs,
            "batch_size": batch_size,
            "sample_size": input_sample_size,
            "seed": seed,
            "device": device,
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "init_audio": init_audio,
            "init_noise_level": init_noise_level,
            "callback": None,
            "scale_phi": cfg_rescale,
            "cfg_norm_threshold": cfg_norm_threshold,
            "apg_scale": apg_scale,
            "rho": rho,
            "adapt_duration_to_conditioning": adapt_duration_to_conditioning,
            "duration_padding_sec": duration_padding_sec,
            "use_effective_length_for_schedule": use_effective_length_for_schedule,
            "mask_padding_attention": mask_padding_attention,
            "dist_shift": dist_shift,
        }

        # If inpainting, send mask args
        # This will definitely change in the future
        if inpaint_audio is not None:
            generate_args.update({
                "inpaint_audio": inpaint_audio,
                "inpaint_mask_start_seconds": mask_maskstart,
                "inpaint_mask_end_seconds": mask_maskend,
            })

        audio = generate_diffusion_cond_inpaint(**generate_args)

        return audio