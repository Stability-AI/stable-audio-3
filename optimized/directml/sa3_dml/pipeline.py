"""End-to-end Stable Audio 3 inference on ONNX Runtime + DirectML."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dxgi, lora
from .assets import Assets
from .audio import Audio, prepare
from .codec import Decoder, Encoder
from .dit import DiT
from .sampling import keep_mask, latent_length, pingpong, schedule, valid_length
from .session import DirectMLUnavailable, Placement, RuntimeConfig, SessionPool
from .specs import AUDIO_CHANNELS, CODECS, DOWNSAMPLING_RATIO, LATENT_DIM, MODELS, SAMPLE_RATE, CodecSpec, ModelSpec
from .text import PromptEmbedding, TextEncoder

MIN_NOISE_LEVEL = 0.01
ROLES = ("text", "encoder", "dit", "decoder")


@dataclass
class Request:
    """One generation. `init_audio` starts sampling from its noised latents at `init_noise_level`;
    `inpaint_audio` is kept outside `inpaint_regions` (seconds) and regenerated inside them."""

    prompt: str
    duration: float
    negative_prompt: str | None = None
    steps: int = 8
    cfg_scale: float = 1.0
    apg_scale: float = 1.0
    seed: int | None = None
    init_audio: Audio | None = None
    init_noise_level: float = 1.0
    inpaint_audio: Audio | None = None
    inpaint_regions: Sequence[tuple[float, float]] = ()
    paste_back: bool = True

    @property
    def mode(self) -> str:
        if self.inpaint_audio is not None:
            return "inpainting"
        return "audio-to-audio" if self.init_audio is not None else "text-to-audio"


@dataclass
class Stage:
    seconds: float = 0.0
    vram_bytes: int = 0
    placement: Placement = field(default_factory=Placement)


@dataclass
class Result:
    audio: np.ndarray
    sample_rate: int
    seed: int
    latent_frames: int
    forwards: int
    stages: dict[str, Stage]
    latents: np.ndarray | None = None


@dataclass
class Graphs:
    text_encoder: Path
    tokenizer: Path
    dit: Path
    decoder: Path


@dataclass
class _Streams:
    noise: np.random.Generator
    sampler: np.random.Generator
    decoder: np.random.Generator
    encoder: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> _Streams:
        return cls(*(np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(4)))


@dataclass
class _Job:
    request: Request
    seed: int
    rng: _Streams
    length: int
    frames: int
    prompt: PromptEmbedding | None = None
    negative: PromptEmbedding | None = None
    init_latents: np.ndarray | None = None
    inpaint_latents: np.ndarray | None = None
    latents: np.ndarray | None = None
    forwards: int = 0


def _adapter_for(runtime: RuntimeConfig) -> dxgi.Adapter | None:
    found = dxgi.adapters() if runtime.provider == "dml" else []
    adapter = found[runtime.device_id] if 0 <= runtime.device_id < len(found) else None
    if adapter is not None and adapter.software:
        raise DirectMLUnavailable(f"adapter {runtime.device_id} ({adapter.name}) is a software "
                                  "renderer; pick a GPU from `python -m sa3_dml devices`")
    return adapter


class _Runner:
    """Session pool plus per-stage timing, VRAM and kernel placement bookkeeping."""

    def __init__(self, runtime: RuntimeConfig, keep_loaded: bool | Collection[str]):
        self.runtime = runtime
        self.keep = set(ROLES) if keep_loaded is True else set(keep_loaded or ())
        unknown = self.keep - set(ROLES)
        if unknown:
            raise ValueError(f"unknown session roles {sorted(unknown)}; expected some of {ROLES}")
        self.adapter = _adapter_for(runtime)
        self.pool = SessionPool(runtime)

    def vram(self) -> int:
        if self.adapter is None:
            return 0
        try:
            usage = dxgi.memory_usage(self.adapter.luid)
        except OSError:
            return 0
        return usage.local_usage if usage else 0

    @contextmanager
    def stage(self, stages: dict[str, Stage], name: str, role: str | None = None) -> Iterator[Stage]:
        stage = stages.setdefault(name, Stage())
        start = time.perf_counter()
        try:
            yield stage
            stage.vram_bytes = max(stage.vram_bytes, self.vram())
        finally:
            role = role or name
            if role not in self.keep:
                self.pool.release(role)
            stage.placement.merge(self.pool.placements.pop(role, Placement()))
            stage.seconds += time.perf_counter() - start

    def close(self) -> None:
        self.pool.release()


class Pipeline:
    """Text-to-audio, audio-to-audio, inpainting and continuation with optional LoRA adapters.

    A list of requests is processed stage by stage, so each network is loaded once for the whole
    list and only one network occupies GPU memory at a time. `keep_loaded` keeps the sessions of
    all networks (True) or of the given roles ("text", "encoder", "dit", "decoder") loaded between
    calls; they then share GPU memory, so on 8 GB GPUs keep only {"dit"} for the medium model.
    """

    def __init__(self, model: str | ModelSpec, assets: Assets, runtime: RuntimeConfig,
                 codec: str | None = None, decode_chunk: int | None = None, decode_overlap: int | None = None,
                 encode_chunk: int | None = None, encode_overlap: int | None = None,
                 loras: Sequence[lora.LoraSpec] = (), keep_loaded: bool | Collection[str] = False,
                 log: Callable[[str], None] = print):
        self.model = MODELS[model] if isinstance(model, str) else model
        self.codec: CodecSpec = CODECS[codec or self.model.codec]
        self.decode_chunk, self.decode_overlap = decode_chunk, decode_overlap
        self.encode_chunk, self.encode_overlap = encode_chunk, encode_overlap
        self.assets = assets
        self.runtime = runtime
        self.log = log
        self._runner = _Runner(runtime, keep_loaded)
        self.adapter = self._runner.adapter
        self._graphs: Graphs | None = None
        self._encoder_graph: Path | None = None
        self._adapters: dict[tuple, lora.Adapter] = {}
        self._merged: tuple[tuple, dict[str, np.ndarray]] | None = None
        self.loras: list[lora.LoraSpec] = []
        self.set_loras(loras)

    def set_loras(self, specs: Sequence[lora.LoraSpec]) -> None:
        self.loras = [lora.LoraSpec(Path(s.path), float(s.strength)) for s in specs]
        for spec in self.loras:
            lora.identity([spec])

    def set_lora_strength(self, strength: float, index: int | None = None) -> None:
        if index is not None and not 0 <= index < len(self.loras):
            raise ValueError(f"LoRA index {index} out of range ({len(self.loras)} loaded)")
        self.loras = [lora.LoraSpec(s.path, float(strength)) if index is None or i == index else s
                      for i, s in enumerate(self.loras)]

    def graphs(self) -> Graphs:
        if self._graphs is None:
            self._graphs = Graphs(
                text_encoder=self.assets.text_encoder(),
                tokenizer=self.assets.tokenizer(self.model),
                dit=self.assets.dit(self.model),
                decoder=self.assets.decoder(self.codec),
            )
        return self._graphs

    def encoder_graph(self) -> Path:
        if self._encoder_graph is None:
            self._encoder_graph = self.assets.encoder(self.codec)
        return self._encoder_graph

    def close(self) -> None:
        self._runner.close()

    def _lora_weights(self) -> tuple[dict[str, np.ndarray] | None, tuple]:
        if not self.loras:
            return None, ()
        key = lora.identity(self.loras)
        if self._merged is None or self._merged[0] != key:
            self._merged = None
            adapters = []
            for spec in self.loras:
                ident = lora.identity([lora.LoraSpec(spec.path)])
                if ident not in self._adapters:
                    self._adapters[ident] = lora.Adapter.load(spec.path)
                adapters.append((self._adapters[ident], spec.strength))
            start = time.perf_counter()
            merged, _ = lora.merge(self.graphs().dit, adapters, self.log)
            self.log(f"lora: merged {len(merged)} DiT weights in {time.perf_counter() - start:.1f} s")
            self._merged = (key, merged)
        return self._merged[1], key

    def _job(self, request: Request) -> _Job:
        if not 0 < request.duration <= self.model.max_seconds:
            raise ValueError(f"duration must be in (0, {self.model.max_seconds:.1f}] s for {self.model.name}")
        if request.steps < 1:
            raise ValueError("steps must be >= 1")
        if request.init_audio is not None and not MIN_NOISE_LEVEL <= request.init_noise_level <= 1.0:
            raise ValueError(f"init noise level must be in [{MIN_NOISE_LEVEL}, 1]")
        if request.inpaint_audio is not None and not request.inpaint_regions:
            raise ValueError("inpainting needs at least one region")
        for start, end in request.inpaint_regions:
            if not 0 <= start < end:
                raise ValueError(f"invalid inpaint region {start}-{end} s: needs 0 <= start < end")
        seed = request.seed if request.seed is not None else random.randrange(2 ** 31)
        length = latent_length(request.duration, self.model.sample_size)
        return _Job(request, seed, _Streams.from_seed(seed), length, valid_length(request.duration, length))

    def generate(self, requests: Request | Sequence[Request],
                 on_step: Callable[[int, int, int], None] | None = None) -> Result | list[Result]:
        """Run one request or a list of them; `on_step(index, step, steps)` reports sampling progress."""
        single = isinstance(requests, Request)
        jobs = [self._job(r) for r in ([requests] if single else requests)]
        graphs = self.graphs()
        run = self._runner
        stages: dict[str, Stage] = {}
        try:
            self._encode_prompts(jobs, graphs, stages)
            if any(j.request.init_audio is not None or j.request.inpaint_audio is not None for j in jobs):
                self._encode_sources(jobs, stages)
            self._sample(jobs, graphs, stages, on_step)
            audio = self._decode(jobs, graphs, stages)
        finally:
            for role in set(ROLES) - run.keep:
                run.pool.release(role)
        results = [Result(a, SAMPLE_RATE, j.seed, j.frames, j.forwards, stages, j.latents)
                   for j, a in zip(jobs, audio)]
        return results[0] if single else results

    def _encode_prompts(self, jobs: list[_Job], graphs: Graphs, stages: dict[str, Stage]) -> None:
        with self._runner.stage(stages, "text", TextEncoder.role):
            encoder = TextEncoder(graphs.text_encoder, graphs.tokenizer, self._runner.pool)
            cache: dict[str, PromptEmbedding] = {}

            def embed(text: str) -> PromptEmbedding:
                if text not in cache:
                    cache[text] = encoder.encode(text)
                return cache[text]

            for job in jobs:
                job.prompt = embed(job.request.prompt)
                if job.request.cfg_scale != 1.0 and job.request.negative_prompt is not None:
                    job.negative = embed(job.request.negative_prompt)

    def _encode_sources(self, jobs: list[_Job], stages: dict[str, Stage]) -> None:
        with self._runner.stage(stages, "encoder", Encoder.role):
            encoder = Encoder(self.encoder_graph(), self._runner.pool, self.codec,
                              self.encode_chunk, self.encode_overlap)
            shared: dict[tuple, np.ndarray] = {}
            for job in jobs:
                done = shared if not self.codec.encoder_token_noise else {}
                for attr in ("init_audio", "inpaint_audio"):
                    source = getattr(job.request, attr)
                    if source is None:
                        continue
                    key = (id(source), job.length)
                    if key not in done:
                        samples = prepare(source, SAMPLE_RATE, AUDIO_CHANNELS, job.length * DOWNSAMPLING_RATIO)
                        done[key] = encoder(samples, job.rng.encoder)
                    setattr(job, attr.replace("_audio", "_latents"), done[key])

    def _sample(self, jobs: list[_Job], graphs: Graphs, stages: dict[str, Stage],
                on_step: Callable[[int, int, int], None] | None) -> None:
        with self._runner.stage(stages, "dit"):
            weights, key = self._lora_weights()
            for index, job in enumerate(jobs):
                request, frames = job.request, job.frames
                session = self._runner.pool.get("dit", graphs.dit, {"L": frames}, weights, key)
                keep = None
                local_add_cond = None
                if job.inpaint_latents is not None:
                    keep = keep_mask(job.length, request.inpaint_regions, request.duration)[..., :frames]
                    source = job.inpaint_latents[..., :frames]
                    local_add_cond = np.concatenate([keep, source * keep], axis=1)
                dit = DiT(session, frames, job.prompt, request.duration, negative=job.negative,
                          cfg_scale=request.cfg_scale, apg_scale=request.apg_scale, local_add_cond=local_add_cond)
                x = job.rng.noise.standard_normal((1, LATENT_DIM, frames), dtype=np.float32)
                sigma_max = 1.0
                if job.init_latents is not None:
                    sigma_max = float(request.init_noise_level)
                    x = job.init_latents[..., :frames] * (1.0 - sigma_max) + x * sigma_max
                progress = None if on_step is None else (lambda s, n, i=index: on_step(i, s, n))
                latents = pingpong(dit, x, schedule(request.steps, sigma_max), job.rng.sampler, progress)
                if not np.isfinite(latents).all():
                    raise FloatingPointError("the diffusion model produced non-finite latents")
                if keep is not None and request.paste_back:
                    latents = job.inpaint_latents[..., :frames] * keep + latents * (1.0 - keep)
                job.latents, job.forwards = latents.astype(np.float32), dit.forwards

    def _decode(self, jobs: list[_Job], graphs: Graphs, stages: dict[str, Stage]) -> list[np.ndarray]:
        out = []
        with self._runner.stage(stages, "decoder", Decoder.role):
            decoder = Decoder(graphs.decoder, self._runner.pool, self.codec, self.decode_chunk, self.decode_overlap)
            for job in jobs:
                padded = np.pad(job.latents, ((0, 0), (0, 0), (0, job.length - job.frames)), mode="edge")
                audio = decoder(padded, job.rng.decoder)
                out.append(np.clip(audio[0, :, :int(job.request.duration * SAMPLE_RATE)], -1.0, 1.0))
        return out


class Autoencoder:
    """Standalone SAME encode (audio -> latents) and decode (latents -> audio)."""

    def __init__(self, codec: str | CodecSpec, assets: Assets, runtime: RuntimeConfig,
                 chunk: int | None = None, overlap: int | None = None,
                 keep_loaded: bool | Collection[str] = False):
        self.codec = CODECS[codec] if isinstance(codec, str) else codec
        self.assets = assets
        self.chunk, self.overlap = chunk, overlap
        self._runner = _Runner(runtime, keep_loaded)
        self.adapter = self._runner.adapter
        self.stages: dict[str, Stage] = {}

    def encode(self, audio: Audio, seed: int | None = None) -> np.ndarray:
        """[256, ceil(N / 4096)] latents of audio resampled to 44.1 kHz stereo."""
        graph = self.assets.encoder(self.codec)
        samples = prepare(audio, SAMPLE_RATE, AUDIO_CHANNELS)
        rng = _Streams.from_seed(seed if seed is not None else random.randrange(2 ** 31)).encoder
        with self._runner.stage(self.stages, "encoder", Encoder.role):
            latents = Encoder(graph, self._runner.pool, self.codec, self.chunk, self.overlap)(samples, rng)
        return latents[0]

    def decode(self, latents: np.ndarray, seed: int | None = None) -> np.ndarray:
        """[2, L * 4096] audio in [-1, 1] from [256, L] latents."""
        graph = self.assets.decoder(self.codec)
        rng = _Streams.from_seed(seed if seed is not None else random.randrange(2 ** 31)).decoder
        with self._runner.stage(self.stages, "decoder", Decoder.role):
            audio = Decoder(graph, self._runner.pool, self.codec, self.chunk, self.overlap)(latents, rng)
        return np.clip(audio[0], -1.0, 1.0)

    def close(self) -> None:
        self._runner.close()
