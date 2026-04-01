import numpy as np
import torch
import typing as tp
from torch.nn.functional import interpolate

from .audio_utils import prepare_audio, numpy_audio_to_tensor
from .sampling import sample_diffusion



def generate(
        model,
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
        steps: int = 250,
        cfg_scale: float = 6.0,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        # Audio inputs
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask=None,
        inpaint_mask_start_seconds: tp.Optional[float] = None,
        inpaint_mask_end_seconds: tp.Optional[float] = None,
        # Duration / schedule options
        adapt_duration_to_conditioning: bool = False,
        duration_padding_sec: float = 6.0,
        use_effective_length_for_schedule: bool = False,
        mask_padding_attention: bool = False,
        apg_scale: float = 1.0,
        dist_shift=None,
        return_latents: bool = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    """
    Generate audio from a model.

    Can be called two ways:

    Simple:
        generate(model, prompt="...", duration=30, steps=100)

    Low-level (pre-built conditioning):
        generate(model, conditioning=[{"prompt": "...", "seconds_total": 30}], steps=100, ...)

    Args:
        model: The model to use for generation.
        prompt: Text prompt (simple path). Builds conditioning internally.
        duration: Duration in seconds (simple path, default 120).
        seconds_start: Start time in seconds for conditioning (simple path, default 0).
        lyrics: Optional lyrics string added to conditioning (simple path).
        negative_prompt: Negative text prompt (simple path).
        conditioning: Pre-built conditioning dicts (low-level path).
        conditioning_tensors: Pre-computed conditioning tensors.
        negative_conditioning: Pre-built negative conditioning dicts.
        negative_conditioning_tensors: Pre-computed negative conditioning tensors.
        steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale.
        batch_size: Batch size.
        sample_size: Output length in audio samples.
        seed: Random seed (-1 for random).
        device: Device to run on.
        init_audio: (sample_rate, audio tensor) for variation/inpainting.
        init_noise_level: Noise level for init_audio mixing.
        inpaint_audio: (sample_rate, audio tensor) to inpaint into.
        inpaint_mask: Pre-built inpaint mask tensor [batch, sample_size].
        inpaint_mask_start_seconds: Start of inpaint region in seconds.
        inpaint_mask_end_seconds: End of inpaint region in seconds.
        adapt_duration_to_conditioning: Shrink sample_size to match seconds_total + padding.
        duration_padding_sec: Extra padding when adapting duration.
        use_effective_length_for_schedule: Use effective_seq_len for distribution shift.
        mask_padding_attention: Mask padding tokens in attention.
        apg_scale: APG scale (1.0 = full APG, 0.0 = vanilla CFG).
        dist_shift: Distribution shift override.
        return_latents: Return raw latents instead of decoded audio.
        **sampler_kwargs: Extra args forwarded to the sampler (sampler_type, sigma_min, sigma_max, etc.).
    """

    # Build conditioning from prompt string if not provided directly
    if conditioning is None and conditioning_tensors is None:
        assert prompt is not None, "Must provide either prompt or conditioning"
        cond_dict = {"prompt": prompt, "seconds_start": seconds_start, "seconds_total": duration}
        if lyrics:
            cond_dict["lyrics"] = lyrics
        conditioning = [cond_dict] * batch_size
        if negative_prompt:
            neg_dict = {"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": duration}
            if lyrics:
                neg_dict["lyrics"] = lyrics
            negative_conditioning = [neg_dict] * batch_size

    # The length of the output in audio samples
    audio_sample_size = sample_size

    # Optionally adapt sample size based on seconds_total conditioning
    if adapt_duration_to_conditioning and conditioning is not None:
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])

        if max_seconds > 0:
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)

            if model.pretransform is not None:
                ds_ratio = model.pretransform.downsampling_ratio
                target_audio_samples = ((target_audio_samples + ds_ratio - 1) // ds_ratio) * ds_ratio

            audio_sample_size = min(target_audio_samples, sample_size)

    # Convert audio sample size to latent size
    latent_sample_size = audio_sample_size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio
    sample_size = latent_sample_size

    # Build inpaint mask from seconds if provided
    if inpaint_mask_start_seconds is not None and inpaint_mask_end_seconds is not None:
        mask_start_samples = min(int(inpaint_mask_start_seconds * model.sample_rate), audio_sample_size)
        mask_end_samples = min(int(inpaint_mask_end_seconds * model.sample_rate), audio_sample_size)
        inpaint_mask = torch.ones(1, audio_sample_size, device=device)
        inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Encode conditioning
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    # Process init audio
    if init_audio is not None:
        in_sr, audio_data = init_audio
        if isinstance(audio_data, np.ndarray):
            audio_data = numpy_audio_to_tensor(audio_data)
        init_audio = (in_sr, audio_data)
        in_sr, init_audio = init_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)
        init_audio = init_audio.repeat(batch_size, 1, 1)

    # Process inpaint audio
    if inpaint_audio is not None:
        inpaint_sr, inpaint_data = inpaint_audio
        if isinstance(inpaint_data, np.ndarray):
            inpaint_data = numpy_audio_to_tensor(inpaint_data)
        inpaint_audio = (inpaint_sr, inpaint_data)
        inpaint_sr, inpaint_audio = inpaint_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)
        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
        if inpaint_mask is not None:
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    # Build inpaint mask tensor
    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)
    else:
        mask = inpaint_mask.unsqueeze(1)
    mask = mask.to(device)

    inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio) if inpaint_audio is not None else \
        torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    sampler_type = sampler_kwargs.pop("sampler_type", None)

    return sample_diffusion(
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
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        init_data=init_audio,
        init_noise_level=init_noise_level,
        decode=not return_latents,
        **sampler_kwargs
    )
