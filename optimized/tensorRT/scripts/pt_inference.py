"""Vanilla PyTorch FP32 inference for SA3 medium DiT + SAME-L decoder.

Drop-in alternative to SA3Inference for the "GT PyTorch eager" comparison
backend in the gradio app. Uses stable_audio_tools directly for everything
except T5 encoding (we still reuse the TRT T5 engine — it's fast, validated
equivalent to PT BF16, and avoids re-loading the 538MB T5Gemma weights).

Public API matches SA3Inference's relevant subset:
    inf = PTInference()           # lazy-loads PT model at first .generate()
    pcm, timing = inf.generate(prompt, seconds=120.0, steps=8, seed=1)
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "stable_audio_3"))
from audio_output import (  # noqa: E402
    PCM16_CEILING,
    dbfs_to_amplitude,
    protect_audio_peak,
)

TRT_REPO = Path("/weka2/cj/clod/sa3s/stable-audio-3/optimized/tensorRT")
SCRIPTS_DIR = TRT_REPO / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# stable_audio_tools is expected to be importable (the venv this gradio runs in
# must include it). The producer venv at /admin/home-cj/sa/venvs/sa3_torch29
# has both gradio + tensorrt + stable_audio_tools.
SAT_PATH = "/admin/home-cj/sa/stable-audio-tools-dev-latest"
if SAT_PATH not in sys.path:
    sys.path.insert(0, SAT_PATH)


# Defaults — matches the user's "medium / SAME-L / fp32" scope.
PT_MODEL_DIR = Path("/weka2/cj/clod/sa3s/models/SA3-M-hf")
T5_TRT_PATH = TRT_REPO / "models" / "sm_90" / "t5gemma" / "t5gemma_fp16mixed.trt"
SAMEL_CKPT_DIR = Path("/weka2/cj/clod/sa3s/models/SAME-L")  # local SAME-L = TRT decoder's weight source
SAMES_CKPT_DIR = Path("/weka2/cj/clod/sa3s/models/SAME-S")  # local SAME-S = TRT SAME-S decoder's weight source

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
T5_MAX_LEN = 256
SIGMA_MAX = 1.0


def _patch_pt():
    """No-op for now. The transformer SDPA fallback in stable_audio_tools is NOT
    numerically equivalent to flash_attn for this model: encode→decode round-trip
    cosine drops from 0.78 (flash_attn) → 0.005 (SDPA) on real music. Keep
    flash_attn enabled."""
    return


def _resolve_pt_model_dir() -> Path:
    """Find the SA3-M HF safetensors dir.

    The producer build references either SA3-M-hf or SA3-medium-ARC. Try both;
    return the first match.
    """
    for name in ("SA3-M-hf", "SA3-M", "SA3-medium-ARC", "SA3-medium"):
        d = Path("/weka2/cj/clod/sa3s/models") / name
        if (d / "model.safetensors").exists() and (d / "model_config.json").exists():
            return d
    raise FileNotFoundError(
        f"Couldn't locate the medium DiT PyTorch checkpoint under "
        f"/weka2/cj/clod/sa3s/models/. Looked for SA3-M-hf / SA3-M / SA3-medium-ARC / SA3-medium."
    )


def _load_pt_model(model_dir: Path, device="cuda"):
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict
    from safetensors.torch import load_file

    with open(model_dir / "model_config.json") as f:
        cfg = json.load(f)
    # SAT's T5GemmaConditioner only takes model_name; HF cfg adds repo_id/subfolder
    for c in cfg.get("model", {}).get("conditioning", {}).get("configs", []):
        if c.get("type") == "t5gemma":
            cc = c.get("config", {})
            cc.pop("repo_id", None); cc.pop("subfolder", None)
            cc.setdefault("model_name", "google/t5gemma-b-b-ul2")
    model = create_model_from_config(cfg)
    sd = load_file(model_dir / "model.safetensors")
    copy_state_dict(model, sd)
    model = model.float().to(device).eval()
    return model, sd


def _load_decoder(ckpt_dir: Path, name: str, device="cuda"):
    """Load a LOCAL SAME-{S,L} AudioAutoencoder — same weights as the TRT decoder.

    Noise sources are disabled to match TRT's deterministic rewrite (the producer's
    ONNX validator disables `bottleneck.noise_regularize` and
    `decoder.layers[3].mask_noise` before comparing — those random kernels would
    inject buzz at inference if left on).
    """
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
    with open(ckpt_dir / f"{name}.json") as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg).to(device).eval()
    sd = load_ckpt_state_dict(str(ckpt_dir / f"{name}.ckpt"))
    copy_state_dict(model, sd)
    model.bottleneck.noise_regularize = False
    model.decoder.layers[3].mask_noise = 0
    return model


def _load_samel_decoder(device="cuda"):
    return _load_decoder(SAMEL_CKPT_DIR, "SAME-L", device=device)


def _load_sames_decoder(device="cuda"):
    return _load_decoder(SAMES_CKPT_DIR, "SAME-S", device=device)


def _build_dist_shift():
    """Import runtime.DistributionShift for the sigma schedule shift."""
    import runtime as rt
    return rt.DistributionShift()


class PTInference:
    """Vanilla PyTorch FP32 inference: PT DiT + PT decoder(s), TRT T5.

    Scope: medium DiT + SAME-L / SAME-S decoders + FP32 only.
    The DiT is loaded once; both decoders are loaded so generate(decoder=...)
    can pick between them without reloading anything.
    Other configs raise NotImplementedError.

    Thread-safe via an internal Lock (the underlying PyTorch model state is
    shared mutable; serialize generate() calls).
    """

    def __init__(self, device: str = "cuda", *,
                  load_samel: bool = True, load_sames: bool = True):
        import threading
        self._lock = threading.Lock()
        self.device = device

        print("  PT FP32 GT: patching stable_audio_tools + loading checkpoint...", flush=True)
        _patch_pt()
        t0 = time.time()
        model_dir = _resolve_pt_model_dir()
        self.pt_model, self.sd = _load_pt_model(model_dir, device=device)
        print(f"  PT model loaded from {model_dir.name} in {time.time()-t0:.1f}s "
              f"({sum(p.numel() for p in self.pt_model.parameters())/1e6:.0f}M params)",
              flush=True)
        self.pt_dit = self.pt_model.model.model

        # Use the LOCAL SAME-L / SAME-S AudioAutoencoders — same weights as the
        # TRT decoder engines (the engines are an ONNX rewrite of these PT models).
        self.pt_samel = None
        self.pt_sames = None
        if load_samel:
            t0 = time.time()
            self.pt_samel = _load_samel_decoder(device=device)
            print(f"  SAME-L decoder loaded from {SAMEL_CKPT_DIR.name} in {time.time()-t0:.1f}s "
                  f"(noise sources disabled)", flush=True)
        if load_sames:
            t0 = time.time()
            self.pt_sames = _load_sames_decoder(device=device)
            print(f"  SAME-S decoder loaded from {SAMES_CKPT_DIR.name} in {time.time()-t0:.1f}s "
                  f"(noise sources disabled)", flush=True)

        # Reuse TRT T5 (fast + validated). Loaded lazily via canon to share
        # the same heavy-imports path the SA3Inference uses.
        import sa3_trt_core as canon
        canon._import_heavy()
        import runtime as rt
        rt.MODELS_DIR = str(TRT_REPO / "models")
        rt.ARCH_DIR = str(TRT_REPO / "models" / "sm_90")
        self.tokenizer = rt.load()["tokenizer"]
        self.t5_runner = canon.TRTRunner(T5_TRT_PATH)
        self.dist_shift = _build_dist_shift()
        self._canon = canon

    # ── conditioning helper (FP32 PT path with the 2π factor baked in) ────
    def _build_cond(self, t5_hidden, t5_mask, seconds_total: float):
        sd = self.sd
        padding_emb = sd["conditioner.conditioners.prompt.padding_embedding"].float().to(self.device)
        sec_w = sd["conditioner.conditioners.seconds_total.embedder.embedding.1.weight"].float().to(self.device)
        sec_b = sd["conditioner.conditioners.seconds_total.embedder.embedding.1.bias"].float().to(self.device)
        half = 128
        ramp = torch.linspace(0, 1, half, device=self.device)
        freqs = torch.exp(ramp * (math.log(10000.0) - math.log(0.5)) + math.log(0.5))
        s = torch.tensor([seconds_total], device=self.device).clamp(0, 384) / 384
        args = s.unsqueeze(-1) * freqs * 2 * math.pi
        ff = torch.cat([args.cos(), args.sin()], dim=-1)
        sec_emb = (ff @ sec_w.T + sec_b).unsqueeze(1)
        m = t5_mask.unsqueeze(-1).bool()
        pe = padding_emb.view(1, 1, -1).expand_as(t5_hidden)
        t5_padded = torch.where(m, t5_hidden, pe)
        return torch.cat([t5_padded, sec_emb], dim=1), sec_emb.squeeze(1)

    @staticmethod
    def resolve_T_lat(seconds: float, decoder: str) -> int:
        """seconds → T_lat (no even-bump for SAME-L, since SAME-L doesn't care)."""
        T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
        return T_lat

    def generate(self, prompt: str, *,
                  seconds: float = 120.0, steps: int = 8,
                  seed: Optional[int] = None,
                  decoder: str = "same-l",
                  # Future kwargs (NotImplementedError until wired):
                  init_noise_level: float = 1.0,
                  negative_prompt: Optional[str] = None,
                  cfg: float = 1.0,
                  init_audio_path: Optional[str] = None,
                  inpaint_range: Optional[tuple] = None,
                  peak_ceiling_dbfs: float = 0.0,
                  ) -> tuple[np.ndarray, dict]:
        if cfg != 1.0:
            raise NotImplementedError("CFG not yet wired through PTInference")
        if init_audio_path is not None or inpaint_range is not None:
            raise NotImplementedError("audio-to-audio / inpaint not yet wired")
        if init_noise_level != 1.0:
            raise NotImplementedError("non-unity init_noise_level not yet wired")
        if decoder == "same-l":
            pt_decoder = self.pt_samel
        elif decoder == "same-s":
            pt_decoder = self.pt_sames
        else:
            raise ValueError(f"unknown decoder={decoder!r}; valid: same-l, same-s")
        if pt_decoder is None:
            raise RuntimeError(f"decoder {decoder!r} was not loaded — pass load_{decoder.replace('-','_')}=True to PTInference()")

        canon = self._canon
        device = self.device
        T_lat = self.resolve_T_lat(seconds, decoder)
        if not (1 <= T_lat <= 4096):
            raise ValueError(f"T_lat={T_lat} out of [1, 4096]")
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)

        with self._lock:
            t_total = time.time()

            # Match official generate_diffusion_cond: disable TF32 + benchmark for
            # FP32-faithful math.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cudnn.benchmark = False

            # 1. T5 encode via TRT (fast).
            t0 = time.time()
            t5_hidden, t5_mask = canon.t5gemma_encode(self.t5_runner, self.tokenizer, prompt)
            t5_hidden = t5_hidden.to(device).float(); t5_mask = t5_mask.to(device)
            t5_ms = (time.time() - t0) * 1000

            # 2. Conditioning + sigmas.
            cond_ca, cond_g = self._build_cond(t5_hidden, t5_mask, seconds)
            sigmas = torch.linspace(SIGMA_MAX, 0.0, steps + 1, device=device)
            sigmas = self.dist_shift.shift(sigmas, T_lat); sigmas[0] = SIGMA_MAX

            # 3. Sampling.
            g_init = torch.Generator(device=device); g_init.manual_seed(int(seed))
            g_loop = torch.Generator(device=device); g_loop.manual_seed(int(seed) + 1)
            x = torch.randn(1, 256, T_lat, device=device, dtype=torch.float32, generator=g_init)
            local_add_cond = torch.zeros(1, 257, T_lat, device=device)

            t0 = time.time()
            with torch.no_grad():
                for i in range(steps):
                    t_curr = sigmas[i]; t_next = sigmas[i + 1]
                    v = self.pt_dit._forward(
                        x, t_curr.unsqueeze(0).contiguous(),
                        cross_attn_cond=cond_ca, global_embed=cond_g,
                        local_add_cond=local_add_cond,
                    )
                    denoised = x - t_curr * v
                    if i < steps - 1:
                        noise = torch.randn(*x.shape, device=device, dtype=x.dtype, generator=g_loop)
                        x = (1.0 - t_next) * denoised + t_next * noise
                    else:
                        x = denoised
            torch.cuda.synchronize()
            sampling_ms = (time.time() - t0) * 1000

            # 4. Decode FP32 via selected decoder (matches TRT decoder weights).
            t0 = time.time()
            with torch.no_grad():
                audio_fp32 = pt_decoder.decode(x.float())
            torch.cuda.synchronize()
            decode_ms = (time.time() - t0) * 1000

            # 5. Trim before peak protection so discarded decoder padding
            # cannot attenuate the requested clip.
            actual_samples = int(round(seconds * SAMPLE_RATE))
            audio_fp32 = audio_fp32[..., :actual_samples]
            audio_fp32 = protect_audio_peak(
                audio_fp32,
                ceiling=dbfs_to_amplitude(peak_ceiling_dbfs),
            )
            pcm_torch = (audio_fp32 * PCM16_CEILING).to(torch.int16)
            pcm = pcm_torch.squeeze(0).T.contiguous().cpu().numpy()

        pcm = pcm.copy()

        inference_ms = (time.time() - t_total) * 1000
        return pcm, {
            "inference_ms":   inference_ms,
            "graph_build_ms": 0.0,  # n/a for PT
            "realtime":       (seconds * 1000.0) / inference_ms if inference_ms > 0 else 0.0,
            "seed":           int(seed),
            "T_lat":          T_lat,
            "samples":        actual_samples,
            "t5_ms":          t5_ms,
            "sampling_ms":    sampling_ms,
            "decode_ms":      decode_ms,
        }


# ── module-level singleton (lazy) ─────────────────────────────────────────
_pt_inference: Optional[PTInference] = None


def get_pt_inference() -> PTInference:
    """Return the singleton PTInference, loading on first call (~30 sec)."""
    global _pt_inference
    if _pt_inference is None:
        _pt_inference = PTInference()
    return _pt_inference
