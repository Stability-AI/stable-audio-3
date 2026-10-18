"""Command-line interface: `python -m sa3_dml {generate,encode,decode,prepare,devices}`."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import dxgi
from .assets import AssetConfig, AssetError, Assets, default_cache_dir
from .audio import WAV_FORMATS, Audio, write_wav
from .lora import LoraError, LoraSpec
from .pipeline import Autoencoder, Pipeline, Request, Stage
from .session import DirectMLUnavailable, RuntimeConfig
from .specs import CODECS, LATENT_DIM, MODELS, SAMPLE_RATE

GIB = 2 ** 30


def _add_assets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--onnx-dir", type=Path, default=None,
                        help="local copy of the onnx/ folder of stabilityai/stable-audio-3-optimized")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where DirectML-ready graphs are stored (env SA3_DML_CACHE, default: ./models)")
    parser.add_argument("--offline", action="store_true", help="never access the network")
    parser.add_argument("--revision", default=None, help="Hugging Face revision of the ONNX repo")


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=sorted(MODELS), default="medium",
                        help="model to run (default: medium)")
    parser.add_argument("--codec", "--decoder", dest="codec", choices=sorted(CODECS), default=None,
                        help="SAME autoencoder for audio inputs and decoding (default: same-l for medium, "
                             "same-s for the small models)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="local checkpoint: a model folder, a model.safetensors file, or a folder "
                             "holding one subfolder per model (env SA3_CHECKPOINT_DIR)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="ignore local checkpoints and use the weights embedded in the official ONNX")
    _add_assets(parser)


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=int, default=0, help="DirectML adapter index (default: 0)")
    parser.add_argument("--provider", choices=("dml", "cpu"), default="dml",
                        help="execution provider; cpu only serves as a numerical reference")
    parser.add_argument("--verify-placement", action="store_true",
                        help="profile every session and report which execution provider ran each kernel")


def _assets(args) -> Assets:
    return Assets(AssetConfig(cache_dir=args.cache_dir or default_cache_dir(), onnx_dir=args.onnx_dir,
                              checkpoint=getattr(args, "checkpoint_dir", None), offline=args.offline,
                              revision=args.revision, use_checkpoint=not getattr(args, "no_checkpoint", False)))


def _runtime(args) -> RuntimeConfig:
    return RuntimeConfig(provider=args.provider, device_id=args.device, profile=args.verify_placement)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "audio"


def _outputs(base: Path | None, prompts: list[str], seeds: list[int]) -> list[Path]:
    if base is None:
        return [Path("output") / f"{_slug(p)}_{s}.wav" for p, s in zip(prompts, seeds)]
    base = base.with_suffix(base.suffix or ".wav")
    if len(prompts) == 1:
        return [base]
    return [base.with_name(f"{base.stem}_{i}{base.suffix}") for i in range(len(prompts))]


def _print_stages(stages: dict[str, Stage], args) -> None:
    for name, stage in stages.items():
        line = f"  {name:8s} {stage.seconds:7.2f} s"
        if stage.vram_bytes:
            line += f"   VRAM in use {stage.vram_bytes / GIB:.2f} GiB"
        if args.verify_placement:
            line += f"   kernels: {stage.placement.describe()}"
        print(line)
    if args.verify_placement:
        cpu = [n for n, s in stages.items() if s.placement.cpu_ops]
        print("  placement: " + ("all kernels ran on DirectML" if not cpu and args.provider == "dml"
                                 else f"CPU kernels in {', '.join(cpu) or 'all stages'}"))


def _device_label(adapter, args) -> str:
    return adapter.name if adapter else ("CPU" if args.provider == "cpu" else f"#{args.device}")


def _broadcast(values: list, count: int, name: str) -> list:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"got {len(values)} {name} values for {count} outputs")
    return values


def _lora_specs(args) -> list[LoraSpec]:
    paths = args.lora or []
    strengths = args.lora_strength or []
    if not paths:
        if strengths or args.lora_index is not None:
            raise ValueError("--lora-strength and --lora-index need --lora")
        return []
    if args.lora_index is not None:
        if len(strengths) != 1 or not 0 <= args.lora_index < len(paths):
            raise ValueError("--lora-index needs one --lora-strength and an index of a loaded --lora")
        values = [1.0] * len(paths)
        values[args.lora_index] = strengths[0]
    else:
        values = _broadcast(strengths or [1.0], len(paths), "--lora-strength")
    return [LoraSpec(Path(p), s) for p, s in zip(paths, values)]


def _requests(args, model) -> list[Request]:
    count = args.batch_size or len(args.prompt)
    if len(args.prompt) > 1 and count != len(args.prompt):
        raise ValueError(f"--batch-size {count} does not match the {len(args.prompt)} prompts")
    prompts = _broadcast(args.prompt, count, "--prompt")
    negatives = _broadcast(args.negative_prompt or [None], count, "--negative-prompt")

    if (args.inpaint_start is None) != (args.inpaint_end is None):
        raise ValueError("--inpaint-start and --inpaint-end must be given together")
    regions = list(zip(args.inpaint_start or [], args.inpaint_end or []))
    if len(args.inpaint_start or []) != len(args.inpaint_end or []):
        raise ValueError("--inpaint-start and --inpaint-end must be repeated the same number of times")
    if regions and args.inpaint_audio is None:
        raise ValueError("--inpaint-start/--inpaint-end need --inpaint-audio")
    if args.inpaint_audio is not None and not regions:
        raise ValueError("--inpaint-audio needs --inpaint-start and --inpaint-end")
    if args.continue_audio is not None and args.inpaint_audio is not None:
        raise ValueError("--continue-audio and --inpaint-audio cannot be combined")

    init = Audio.load(args.init_audio) if args.init_audio else None
    inpaint = Audio.load(args.inpaint_audio) if args.inpaint_audio else None
    if args.continue_audio:
        inpaint = Audio.load(args.continue_audio)
    source = inpaint or init
    if args.duration:
        durations = args.duration
    elif args.continue_audio:
        raise ValueError("--continue-audio needs --duration longer than the source")
    elif source is not None:
        durations = [min(source.seconds, model.max_seconds)]
    else:
        durations = [30.0]
    durations = _broadcast(durations, count, "--duration")
    if args.continue_audio:
        if any(d <= inpaint.seconds for d in durations):
            raise ValueError(f"--duration must exceed the {inpaint.seconds:.2f} s of the source to continue it")

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2 ** 31 - count))
    out = []
    for i in range(count):
        spans = [(inpaint.seconds, durations[i])] if args.continue_audio else regions
        out.append(Request(prompt=prompts[i], duration=durations[i], negative_prompt=negatives[i],
                           steps=args.steps, cfg_scale=args.cfg_scale, apg_scale=args.apg_scale, seed=seed + i,
                           init_audio=init, init_noise_level=args.init_noise_level, inpaint_audio=inpaint,
                           inpaint_regions=spans, paste_back=not args.no_paste_back))
    return out


def cmd_generate(args) -> int:
    model = MODELS[args.model]
    requests = _requests(args, model)
    pipeline = Pipeline(args.model, _assets(args), _runtime(args), codec=args.codec,
                        decode_chunk=args.decode_chunk, decode_overlap=args.decode_overlap,
                        encode_chunk=args.encode_chunk, encode_overlap=args.encode_overlap,
                        loras=_lora_specs(args))
    print(f"model {args.model} | codec {pipeline.codec.name} | {requests[0].mode} | "
          f"provider {args.provider} ({_device_label(pipeline.adapter, args)})")
    pipeline.graphs()
    if any(r.init_audio is not None or r.inpaint_audio is not None for r in requests):
        pipeline.encoder_graph()

    def progress(index: int, step: int, total: int) -> None:
        tag = f"[{index + 1}/{len(requests)}] " if len(requests) > 1 else ""
        print(f"\r  {tag}sampling step {step}/{total}", end="" if step < total else "\n", flush=True)

    started = time.perf_counter()
    results = pipeline.generate(requests, on_step=progress)
    total = time.perf_counter() - started
    _print_stages(results[0].stages, args)
    paths = _outputs(args.output, [r.prompt for r in requests], [r.seed for r in results])
    for result, path in zip(results, paths):
        write_wav(path, result.audio, result.sample_rate, args.wav_format)
        seconds = result.audio.shape[-1] / result.sample_rate
        print(f"  {result.latent_frames} latent frames, {result.forwards} DiT forwards, "
              f"{seconds:.1f} s of audio, seed {result.seed} -> {Path(path).resolve()}")
    print(f"done in {total:.1f} s")
    return 0


def _autoencoder(args) -> Autoencoder:
    return Autoencoder(args.codec, _assets(args), _runtime(args), chunk=args.chunk, overlap=args.overlap)


def cmd_encode(args) -> int:
    audio = Audio.load(args.input)
    ae = _autoencoder(args)
    print(f"encode {args.input} ({audio.seconds:.2f} s, {audio.sample_rate} Hz, {audio.samples.shape[0]} ch) | "
          f"codec {ae.codec.name} | provider {args.provider} ({_device_label(ae.adapter, args)})")
    try:
        latents = ae.encode(audio, args.seed)
    finally:
        ae.close()
    output = args.output or Path("output") / f"{Path(args.input).stem}_{ae.codec.name}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, latents)
    _print_stages(ae.stages, args)
    print(f"  latents {tuple(latents.shape)} -> {Path(output).resolve()}")
    return 0


def cmd_decode(args) -> int:
    latents = np.load(args.input).astype(np.float32)
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents[0]
    if latents.ndim != 2 or latents.shape[0] != LATENT_DIM:
        raise ValueError(f"expected latents of shape ({LATENT_DIM}, T), got {latents.shape}")
    ae = _autoencoder(args)
    print(f"decode {args.input} ({latents.shape[-1]} frames) | codec {ae.codec.name} | "
          f"provider {args.provider} ({_device_label(ae.adapter, args)})")
    try:
        audio = ae.decode(latents, args.seed)
    finally:
        ae.close()
    output = args.output or Path("output") / f"{Path(args.input).stem}.wav"
    write_wav(output, audio, SAMPLE_RATE, args.wav_format)
    _print_stages(ae.stages, args)
    print(f"  {audio.shape[-1] / SAMPLE_RATE:.2f} s of audio -> {Path(output).resolve()}")
    return 0


def cmd_prepare(args) -> int:
    assets = _assets(args)
    model = MODELS[args.model]
    codec = CODECS[args.codec or model.codec]
    for label, fn in (("text encoder", assets.text_encoder), ("tokenizer", lambda: assets.tokenizer(model)),
                      ("dit", lambda: assets.dit(model)), ("decoder", lambda: assets.decoder(codec)),
                      ("encoder", lambda: assets.encoder(codec))):
        print(f"  {label:12s} {fn()}")
    return 0


def cmd_devices(args) -> int:
    found = dxgi.adapters()
    if not any(not a.software for a in found):
        print("no DirectX 12 hardware adapter found")
        return 1
    for adapter in found:
        kind = "software" if adapter.software else f"{adapter.dedicated_bytes / GIB:.1f} GiB dedicated"
        print(f"  --device {adapter.index}: {adapter.name} ({kind})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sa3_dml", description="Stable Audio 3 on DirectX 12 GPUs through DirectML")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="text-to-audio, audio-to-audio, inpainting and continuation")
    gen.add_argument("-p", "--prompt", required=True, nargs="+",
                     help="text prompt (empty string = unconditional); several prompts generate one output each")
    gen.add_argument("--negative-prompt", nargs="+", default=None,
                     help="negative prompt, used when --cfg-scale != 1; one, or one per prompt")
    gen.add_argument("--duration", type=float, nargs="+", default=None,
                     help="seconds of audio, one or one per output (default: 30, or the source audio length)")
    gen.add_argument("--batch-size", type=int, default=None,
                     help="outputs to generate (default: one per prompt); output i uses seed + i")
    gen.add_argument("--steps", type=int, default=8, help="sampling steps (default: 8)")
    gen.add_argument("--cfg-scale", type=float, default=1.0, help="classifier-free guidance (default: 1.0 = off)")
    gen.add_argument("--apg-scale", type=float, default=1.0,
                     help="adaptive projected guidance blend when CFG is on (1 = APG, 0 = plain CFG)")
    gen.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    gen.add_argument("-o", "--output", type=Path, default=None,
                     help="output WAV; several outputs get an _<index> suffix (default: output/<prompt>_<seed>.wav)")
    gen.add_argument("--wav-format", choices=WAV_FORMATS, default="float32", help="WAV sample format")
    src = gen.add_argument_group("audio inputs")
    src.add_argument("--init-audio", type=Path, default=None, help="source WAV for audio-to-audio generation")
    src.add_argument("--init-noise-level", type=float, default=0.9,
                     help="noise level the init audio is sampled from (0.01-1, default: 0.9)")
    src.add_argument("--inpaint-audio", type=Path, default=None,
                     help="source WAV for inpainting; kept outside the --inpaint-start/--inpaint-end regions")
    src.add_argument("--inpaint-start", type=float, action="append", default=None, metavar="SECONDS",
                     help="start of a region to regenerate; repeat for several regions")
    src.add_argument("--inpaint-end", type=float, action="append", default=None, metavar="SECONDS",
                     help="end of a region to regenerate; repeat for several regions")
    src.add_argument("--continue-audio", type=Path, default=None,
                     help="source WAV to extend up to --duration (inpainting from the end of the source)")
    src.add_argument("--no-paste-back", action="store_true",
                     help="keep the regenerated latents outside the inpaint regions instead of the source ones")
    lr = gen.add_argument_group("LoRA")
    lr.add_argument("--lora", "--lora-ckpt-path", dest="lora", type=Path, action="append", default=None,
                    help="LoRA adapter (.safetensors, or a PEFT folder); repeat to stack adapters")
    lr.add_argument("--lora-strength", type=float, action="append", default=None,
                    help="strength for all adapters, or repeated once per adapter (default: 1)")
    lr.add_argument("--lora-index", type=int, default=None,
                    help="apply the single --lora-strength to this adapter only")
    ch = gen.add_argument_group("autoencoder windows")
    ch.add_argument("--decode-chunk", type=int, default=None,
                    help="latent frames per decoder window, 0 = whole sequence (default: 128)")
    ch.add_argument("--decode-overlap", type=int, default=None,
                    help="latent frames shared by neighbouring decoder windows (default: 32)")
    ch.add_argument("--encode-chunk", type=int, default=None,
                    help="latent frames per encoder window, 0 = whole sequence (default: 128)")
    ch.add_argument("--encode-overlap", type=int, default=None,
                    help="latent frames shared by neighbouring encoder windows (default: 32)")
    _add_model(gen)
    _add_runtime(gen)
    gen.set_defaults(func=cmd_generate)

    for name, help_text, func in (("encode", "encode a WAV file into SAME latents (.npy)", cmd_encode),
                                  ("decode", "decode SAME latents (.npy) into a WAV file", cmd_decode)):
        ae = sub.add_parser(name, help=help_text)
        ae.add_argument("-i", "--input", type=Path, required=True,
                        help="input WAV" if name == "encode" else "input .npy latents of shape (256, T)")
        ae.add_argument("-o", "--output", type=Path, default=None,
                        help="output .npy" if name == "encode" else "output WAV")
        ae.add_argument("--codec", choices=sorted(CODECS), default="same-l", help="SAME autoencoder (default: same-l)")
        ae.add_argument("--seed", type=int, default=None, help="seed of the reference autoencoder noise")
        ae.add_argument("--chunk", type=int, default=None,
                        help="latent frames per window, 0 = whole sequence (default: 128)")
        ae.add_argument("--overlap", type=int, default=None, help="latent frames shared by windows (default: 32)")
        if name == "decode":
            ae.add_argument("--wav-format", choices=WAV_FORMATS, default="float32", help="WAV sample format")
        _add_assets(ae)
        _add_runtime(ae)
        ae.set_defaults(func=func)

    prep = sub.add_parser("prepare", help="download and convert the graphs for a model")
    _add_model(prep)
    prep.set_defaults(func=cmd_prepare)

    dev = sub.add_parser("devices", help="list DirectX adapters and their --device index")
    dev.set_defaults(func=cmd_devices)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (AssetError, DirectMLUnavailable, LoraError, ValueError, FloatingPointError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
