import json
import torch
import os
from functools import partial
from stable_audio_3.inference.generation import generate
from stable_audio_3.model import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_3.core.minlora import add_lora, LoRAParametrization, set_lora_strength, infer_global_rank, get_lora_layers, load_lora_checkpoint, remap_lora_state_dict

# MODEL_CONFIG_PATH = "44k_taae_4096_256_2_ct_dit_cond_t5gemma_380s_asx_large2_inpaint_ot_varlen_muon.json"
# CKPT_PATH = "4096_256_taaev2_rc2_base_cond_t5gemma_380s_inpaint_asxmi_varlen_muon_82k_unwrap.ckpt"
MODEL_CONFIG_PATH = "SA3-S-ARC-CLAP.json"
CKPT_PATH = "SA3-S-ARC-CLAP-5k.ckpt"
model_half = False
SEED = 124

class StableAudioPipeline:
    def __init__(self, model, model_config, device):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.same = self.model.pretransform
        self.dit = self.model.model
    
    @torch.inference_mode()
    def generate(self, prompt, duration=120, **kwargs):
        result = generate(model=self.model, prompt=prompt, duration=duration, device=str(self.device), **kwargs)
        print(f"Generated audio with shape {result.shape} and dtype {result.dtype}")
        return result[0].float()
        

    @staticmethod
    def from_pretrained(model_name_or_path):
        # Load the model and any necessary components here
        ## TODO: Work with HuggingFace Hub to load models
        with open(MODEL_CONFIG_PATH) as f:
            model_config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_config = StableAudioPipeline._load_model(model_config, CKPT_PATH, model_half=model_half, device=device)
        return StableAudioPipeline(model, model_config, device)

    @staticmethod
    def _load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False, lora_ckpt_paths=None):
        if pretrained_name is not None:
            pass
            # print(f"Loading pretrained model {pretrained_name}")
            # model, model_config = get_pretrained_model(pretrained_name)

        elif model_config is not None and model_ckpt_path is not None:
            print(f"Creating model from config")
            model = create_diffusion_cond_from_config(model_config)

            print(f"Loading model checkpoint from {model_ckpt_path}")
            # Load checkpoint and unwrap if it's a wrapped training checkpoint
            state_dict = load_ckpt_state_dict(model_ckpt_path)
            # state_dict = unwrap_state_dict(state_dict, model_config.get("model_type"))
            copy_state_dict(model, state_dict)

        model_type = model_config["model_type"]

        if pretransform_ckpt_path is not None:
            print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
            model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
            print(f"Done loading pretransform")

        if lora_ckpt_paths:
            # Move to device before add_lora so SVD (for LoRA-XS) runs on GPU
            model.to(device)
            lora_names = []
            for i, lora_path in enumerate(lora_ckpt_paths):
                print(f"Loading LoRA {i} from {lora_path}")
                lora_state_dict, lora_config_dict = load_lora_checkpoint(lora_path)
                lora_rank = lora_config_dict.get("rank", infer_global_rank(lora_state_dict))
                lora_alpha = lora_config_dict.get("alpha", lora_rank)
                lora_adapter_type = lora_config_dict.get("adapter_type", "lora")
                lora_include = lora_config_dict.get("include", None)
                lora_exclude = lora_config_dict.get("exclude", None)
                lora_config = {
                    torch.nn.Linear: {
                        "weight": partial(LoRAParametrization.from_linear, rank=lora_rank, lora_alpha=lora_alpha, adapter_type=lora_adapter_type, lora_index=i),
                    },
                    torch.nn.Conv1d: {
                        "weight": partial(LoRAParametrization.from_conv1d, rank=lora_rank, lora_alpha=lora_alpha, adapter_type=lora_adapter_type, lora_index=i),
                    }
                }
                if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
                    add_lora(model.model, lora_config, include=lora_include, exclude=lora_exclude)
                    add_lora(model.conditioner, lora_config, include=lora_include, exclude=lora_exclude)
                # Remap state dict keys to target the correct parametrization index
                remapped_sd = remap_lora_state_dict(lora_state_dict, i)
                model.model.load_state_dict(remapped_sd, strict=False)
                model.conditioner.load_state_dict(remapped_sd, strict=False)
                # Store display name from filename
                lora_names.append(os.path.splitext(os.path.basename(lora_path))[0])

            print("lora layers:", len(get_lora_layers(model)))
            model.use_lora = True
            model.lora_names = lora_names
        else:
            model.use_lora = False
            model.lora_names = []

        model.to(device).eval().requires_grad_(False)

        if model_half:
            model.to(torch.float16)

        print(f"Done loading model")

        return model, model_config