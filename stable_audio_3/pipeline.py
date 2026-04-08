import json
import numpy as np
import torch
import typing as tp
import os
from functools import partial
from torch.nn.functional import interpolate

from stable_audio_3.inference.audio_utils import prepare_audio, numpy_audio_to_tensor
from stable_audio_3.inference.sampling import sample_diffusion
from stable_audio_3.model import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_3.models.minlora import (
    add_lora,
    LoRAParametrization,
    set_lora_strength,
    infer_global_rank,
    get_lora_layers,
    load_lora_checkpoint,
    remap_lora_state_dict,
)


# MODEL_CONFIG_PATH = "44k_taae_4096_256_2_ct_dit_cond_t5gemma_380s_asx_large2_inpaint_ot_varlen_muon.json"
# CKPT_PATH = "4096_256_taaev2_rc2_base_cond_t5gemma_380s_inpaint_asxmi_varlen_muon_82k_unwrap.ckpt"
MODEL_CONFIG_PATH = "SA3-S-ARC-CLAP.json"
CKPT_PATH = "SA3-S-ARC-CLAP-5k.ckpt"
SEED = 124


class StableAudioPipeline:
    def __init__(self, model, model_config, device):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.same = self.model.pretransform
        self.dit = self.model.model

    @torch.inference_mode()
    def generate(
        self,
        # Simple path: pass a prompt string and duration
        prompt: str = None,
        duration: float = 120,
        seconds_start: float = 0,
        lyrics: tp.Optional[str] = None,
        negative_prompt: str = None,
        # Low-level path: pass pre-built conditioning dicts
        conditioning: tp.Optional[tp.List[dict]] = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: tp.Optional[tp.List[dict]] = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        # Generation parameters
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        sample_size: int = 5292032,
        seed: int = -1,
        # Audio inputs
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask=None,
        inpaint_mask_start_seconds: tp.Optional[float] = None,
        inpaint_mask_end_seconds: tp.Optional[float] = None,
        # Duration / schedule options
        adapt_duration_to_conditioning: bool = True,
        duration_padding_sec: float = 6.0,
        use_effective_length_for_schedule: bool = False,
        mask_padding_attention: bool = False,
        apg_scale: float = 1.0,
        dist_shift=None,
        return_latents: bool = False,
        **sampler_kwargs,
    ) -> torch.Tensor:
        """
        Generate audio.

        Simple path:
            pipeline.generate(prompt="...", duration=30, steps=100)

        Low-level path (pre-built conditioning):
            pipeline.generate(conditioning=[{"prompt": "...", "seconds_total": 30}], steps=100, ...)

        Args:
            steps: The number of diffusion steps to use.
            cfg_scale: Classifier-free guidance scale
            conditioning: A dictionary of conditioning parameters to use for generation.
            conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
            batch_size: The batch size to use for generation.
            sample_size: The length of the audio to generate, in samples.
            sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
            seed: The random seed to use for generation, or -1 to use a random seed.
            device: The device to use for generation.
            init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
            init_noise_level: The noise level to use when generating from an initial audio sample.
            return_latents: Whether to return the latents used for generation instead of the decoded audio.
            adapt_duration_to_conditioning: If True, adapt the sample size based on seconds_total in conditioning plus duration_padding_sec.
                This enables variable-length generation for models trained with padding masking.
            duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
            use_effective_length_for_schedule: If True, use effective_seq_len for distribution shift.
                Enable this for models trained with use_effective_length_for_schedule=True.
            mask_padding_attention: If True, create padding_mask for attention based on effective_seq_len.
                Only needed when generating at full sequence length but wanting to mask padding.
            apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
            dist_shift: Optional distribution shift override for sampling. If None, uses model.sampling_dist_shift.
            **sampler_kwargs: Additional keyword arguments to pass to the sampler.
        """

        model = self.model
        device = str(self.device)

        # Build conditioning from prompt string if not provided directly
        if conditioning is None and conditioning_tensors is None:
            assert prompt is not None, "Must provide either prompt or conditioning"
            cond_dict = {
                "prompt": prompt,
                "seconds_start": seconds_start,
                "seconds_total": duration,
            }
            if lyrics:
                cond_dict["lyrics"] = lyrics
            conditioning = [cond_dict] * batch_size
            if negative_prompt:
                neg_dict = {
                    "prompt": negative_prompt,
                    "seconds_start": seconds_start,
                    "seconds_total": duration,
                }
                if lyrics:
                    neg_dict["lyrics"] = lyrics
                negative_conditioning = [neg_dict] * batch_size
        # The length of the output in audio samples
        audio_sample_size = sample_size

        # Read from model wrapper (with fallback to training config for backward compat)
        trained_with_effective_length = getattr(
            model, "use_effective_length_for_schedule", False
        )
        trained_with_masking = getattr(model, "mask_padding_attention", False)
        adapt_duration_to_conditioning = (
            adapt_duration_to_conditioning or trained_with_masking
        )
        if not trained_with_effective_length or not trained_with_masking:
            training_config = self.model_config.get("training", {})
            if not trained_with_effective_length:
                trained_with_effective_length = training_config.get(
                    "use_effective_length_for_schedule", False
                )
            if not trained_with_masking:
                trained_with_masking = training_config.get(
                    "mask_padding_attention", False
                )
        use_effective_length_for_schedule = (
            use_effective_length_for_schedule or trained_with_effective_length
        )
        mask_padding_attention = mask_padding_attention or trained_with_masking
        # Optionally adapt sample size based on seconds_total conditioning
        if adapt_duration_to_conditioning and conditioning is not None:
            max_seconds = 0.0
            for cond_dict in conditioning:
                if "seconds_total" in cond_dict:
                    max_seconds = max(max_seconds, cond_dict["seconds_total"])

            if max_seconds > 0:
                target_audio_samples = int(
                    (max_seconds + duration_padding_sec) * model.sample_rate
                )
                latent_align = 1
                if model.pretransform is not None:
                    ds_ratio = model.pretransform.downsampling_ratio
                    # Round up to nearest multiple of downsampling ratio
                    target_audio_samples = (
                        (target_audio_samples + ds_ratio - 1) // ds_ratio
                    ) * ds_ratio
                    encoder_config = self.model_config["model"]["pretransform"][
                        "config"
                    ]["encoder"]["config"]
                    chunk_size = encoder_config.get("chunk_size", 32)
                    stride = encoder_config["strides"][0]  # or min(strides) if multiple
                    latent_align = (
                        chunk_size // stride
                    )  # For chunked attention with latent space, we need to align to the chunk size after downsampling
                    align = ds_ratio * latent_align
                    target_audio_samples = (
                        (target_audio_samples + align - 1) // align
                    ) * align

                audio_sample_size = min(target_audio_samples, sample_size)

        # Convert audio sample size to latent size
        latent_sample_size = audio_sample_size
        if model.pretransform is not None:
            latent_sample_size = (
                audio_sample_size // model.pretransform.downsampling_ratio
            )

        # Build inpaint mask from seconds if provided
        if (
            inpaint_mask_start_seconds is not None
            and inpaint_mask_end_seconds is not None
        ):
            mask_start_samples = min(
                int(inpaint_mask_start_seconds * model.sample_rate), audio_sample_size
            )
            mask_end_samples = min(
                int(inpaint_mask_end_seconds * model.sample_rate), audio_sample_size
            )
            inpaint_mask = torch.ones(1, audio_sample_size, device=device)
            inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

        if inpaint_mask is not None:
            inpaint_mask = inpaint_mask.float()

        # Seed
        seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        noise = torch.randn(
            [batch_size, model.io_channels, latent_sample_size], device=device
        )

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

        # Encode conditioning
        if conditioning_tensors is None:
            conditioning_tensors = model.conditioner(conditioning, device)
        if (
            negative_conditioning is not None
            or negative_conditioning_tensors is not None
        ):
            if negative_conditioning_tensors is None:
                negative_conditioning_tensors = model.conditioner(
                    negative_conditioning, device
                )
        else:
            negative_conditioning_tensors = {}

        # Process init audio
        if init_audio is not None:
            in_sr, audio_data = init_audio
            if isinstance(audio_data, np.ndarray):
                audio_data = numpy_audio_to_tensor(audio_data)
            in_sr, init_audio = (in_sr, audio_data)
            io_channels = (
                model.pretransform.io_channels
                if model.pretransform is not None
                else model.io_channels
            )
            init_audio = prepare_audio(
                init_audio,
                in_sr=in_sr,
                target_sr=model.sample_rate,
                target_length=audio_sample_size,
                target_channels=io_channels,
                device=device,
            )
            if model.pretransform is not None:
                init_audio = model.pretransform.encode(init_audio)
                if inpaint_mask is not None:
                    inpaint_mask = interpolate(
                        inpaint_mask.unsqueeze(1),
                        size=init_audio.shape[-1],
                        mode="nearest",
                    ).squeeze(1)
            init_audio = init_audio.repeat(batch_size, 1, 1)

        # Process inpaint audio
        if inpaint_audio is not None:
            inpaint_sr, inpaint_data = inpaint_audio
            if isinstance(inpaint_data, np.ndarray):
                inpaint_data = numpy_audio_to_tensor(inpaint_data)
            inpaint_sr, inpaint_audio = (inpaint_sr, inpaint_data)
            io_channels = (
                model.pretransform.io_channels
                if model.pretransform is not None
                else model.io_channels
            )
            inpaint_audio = prepare_audio(
                inpaint_audio,
                in_sr=inpaint_sr,
                target_sr=model.sample_rate,
                target_length=audio_sample_size,
                target_channels=io_channels,
                device=device,
            )
            if model.pretransform is not None:
                inpaint_audio = model.pretransform.encode(inpaint_audio)
                # inpaint_audio = inpaint_audio[..., :latent_sample_size]
                if inpaint_mask is not None:
                    inpaint_mask = interpolate(
                        inpaint_mask.unsqueeze(1),
                        size=inpaint_audio.shape[-1],
                        mode="nearest",
                    ).squeeze(1)
            inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
        else:
            if inpaint_mask is not None:
                inpaint_mask = interpolate(
                    inpaint_mask.unsqueeze(1), size=latent_sample_size, mode="nearest"
                ).squeeze(1)

        # Build inpaint mask tensor
        if inpaint_mask is None:
            mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
        else:
            mask = inpaint_mask.unsqueeze(1)
        mask = mask.to(device)

        inpaint_input = (
            inpaint_audio * mask.expand_as(inpaint_audio)
            if inpaint_audio is not None
            else torch.zeros(
                (batch_size, model.io_channels, latent_sample_size), device=device
            )
        )

        conditioning_tensors["inpaint_mask"] = [mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

        if negative_conditioning_tensors:
            negative_conditioning_tensors["inpaint_mask"] = [mask]
            negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
            negative_conditioning_tensors = model.get_conditioning_inputs(
                negative_conditioning_tensors, negative=True
            )

        model_dtype = next(model.model.parameters()).dtype
        noise = noise.type(model_dtype)
        conditioning_inputs = {
            k: v.type(model_dtype) if v is not None else v
            for k, v in conditioning_inputs.items()
        }

        cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

        sampler_type = sampler_kwargs.pop("sampler_type", None)

        result = sample_diffusion(
            model=model.model,
            noise=noise,
            cond_inputs=cond_inputs,
            diffusion_objective=model.diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_rate=model.sample_rate,
            pretransform=model.pretransform,
            mask_padding_attention=mask_padding_attention,
            use_effective_length_for_schedule=use_effective_length_for_schedule,
            headroom_seconds=duration_padding_sec,
            dist_shift=dist_shift
            if dist_shift is not None
            else model.sampling_dist_shift,
            sampler_type=sampler_type,
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=apg_scale,
            init_data=init_audio,
            init_noise_level=init_noise_level,
            decode=not return_latents,
            **sampler_kwargs,
        )
        print(f"Generated audio with shape {result.shape} and dtype {result.dtype}")
        return result

    @staticmethod
    def from_pretrained(model_name_or_path, model_half=False):
        # Load the model and any necessary components here
        ## TODO: Work with HuggingFace Hub to load models
        with open(MODEL_CONFIG_PATH) as f:
            model_config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_config = StableAudioPipeline._load_model(
            model_config, CKPT_PATH, model_half=model_half, device=device
        )
        return StableAudioPipeline(model, model_config, device)

    @staticmethod
    def _load_model(
        model_config=None,
        model_ckpt_path=None,
        pretrained_name=None,
        pretransform_ckpt_path=None,
        device="cuda",
        model_half=False,
        lora_ckpt_paths=None,
    ):
        if pretrained_name is not None:
            pass
            # print(f"Loading pretrained model {pretrained_name}")
            # model, model_config = get_pretrained_model(pretrained_name)

        elif model_config is not None and model_ckpt_path is not None:
            print("Creating model from config")
            model = create_diffusion_cond_from_config(model_config)

            print(f"Loading model checkpoint from {model_ckpt_path}")
            # Load checkpoint and unwrap if it's a wrapped training checkpoint
            state_dict = load_ckpt_state_dict(model_ckpt_path)
            # state_dict = unwrap_state_dict(state_dict, model_config.get("model_type"))
            copy_state_dict(model, state_dict)

        model_type = model_config["model_type"]

        if pretransform_ckpt_path is not None:
            print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
            model.pretransform.load_state_dict(
                load_ckpt_state_dict(pretransform_ckpt_path), strict=False
            )
            print("Done loading pretransform")

        if lora_ckpt_paths:
            # Move to device before add_lora so SVD (for LoRA-XS) runs on GPU
            model.to(device)
            lora_names = []
            for i, lora_path in enumerate(lora_ckpt_paths):
                print(f"Loading LoRA {i} from {lora_path}")
                lora_state_dict, lora_config_dict = load_lora_checkpoint(lora_path)
                lora_rank = lora_config_dict.get(
                    "rank", infer_global_rank(lora_state_dict)
                )
                lora_alpha = lora_config_dict.get("alpha", lora_rank)
                lora_adapter_type = lora_config_dict.get("adapter_type", "lora")
                lora_include = lora_config_dict.get("include", None)
                lora_exclude = lora_config_dict.get("exclude", None)
                lora_config = {
                    torch.nn.Linear: {
                        "weight": partial(
                            LoRAParametrization.from_linear,
                            rank=lora_rank,
                            lora_alpha=lora_alpha,
                            adapter_type=lora_adapter_type,
                            lora_index=i,
                        ),
                    },
                    torch.nn.Conv1d: {
                        "weight": partial(
                            LoRAParametrization.from_conv1d,
                            rank=lora_rank,
                            lora_alpha=lora_alpha,
                            adapter_type=lora_adapter_type,
                            lora_index=i,
                        ),
                    },
                }
                if (
                    model_type == "diffusion_cond"
                    or model_type == "diffusion_cond_inpaint"
                ):
                    add_lora(
                        model.model,
                        lora_config,
                        include=lora_include,
                        exclude=lora_exclude,
                    )
                    add_lora(
                        model.conditioner,
                        lora_config,
                        include=lora_include,
                        exclude=lora_exclude,
                    )
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

        print("Done loading model")

        return model, model_config
