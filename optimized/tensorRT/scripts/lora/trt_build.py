"""STRONGLY_TYPED TensorRT build for the SA3-medium DiT, and its optimization profile.

The fp16 island surgery (RMSNorm/RoPE fp32 islands, fp16 trunk, bound_attention_core so
TRT's FMHA fuser fires) is already baked into the published ONNX; this is only the builder.
`on_network(network)` runs after parse and before build -- the branch path uses it to splice
in the low-rank inputs. STRONGLY_TYPED is mandatory here and is the opposite of the fp8
recipe, which needs a weakly-typed network.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T5_TOKENS, T5_HIDDEN_DIM = 256, 768

def profile_for(batch):
    """The shipped medium profile, with the batch dim opened up.

    L range and opt point are unchanged (1 / 1292 / 4096) — only the batch dimension is
    parameterised, and each batch gets its OWN optimization profile so TRT tunes tactics
    for it (a single min=1,max=2 profile tunes for opt only and leaves the other shape on
    fallback kernels).
    """
    b = batch
    return {
        "x":              [(b, 256, 1),   (b, 256, 1292),   (b, 256, 4096)],
        "t":              [(b,), (b,), (b,)],
        "t5_hidden":      [(b, T5_TOKENS, T5_HIDDEN_DIM)] * 3,
        "t5_mask":        [(b, T5_TOKENS)] * 3,
        "seconds_total":  [(b,), (b,), (b,)],
        "local_add_cond": [(b, 257, 1),   (b, 257, 1292),   (b, 257, 4096)],
    }


def build_engine(onnx_path, engine_path, workspace_gb=48, detailed=True, batches=(1,),
                 on_network=None, extra_flags=(), profile_hook=None):
    """on_network(network) runs after parse, before build -- used by the LoRA path to mark
    weights refittable. extra_flags are BuilderFlag names to set."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    print(f"[build] parsing {onnx_path} ({os.path.getsize(onnx_path)/1e6:.0f} MB proto)", flush=True)
    t0 = time.time()
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("  parse error:", parser.get_error(i))
        sys.exit(2)
    print(f"[build] parsed in {time.time()-t0:.0f}s: {network.num_layers} layers", flush=True)

    if on_network is not None:
        on_network(network)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    for f in extra_flags:
        cfg.set_flag(getattr(trt.BuilderFlag, f))
    if detailed:
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if profile_hook is not None:
        profile_hook(builder, cfg)          # caller owns every profile (LoRA branch shapes)
    else:
        for b in batches:
            profile = builder.create_optimization_profile()
            for name, (lo, opt, hi) in profile_for(b).items():
                profile.set_shape(name, lo, opt, hi)
            cfg.add_optimization_profile(profile)

    print(f"[build] STRONGLY_TYPED, workspace {workspace_gb} GB, "
          f"{len(batches)} profile(s) batch={list(batches)} × L=1..4096 ...", flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise SystemExit("[build] FAILED")
    dt = time.time() - t0
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(ser)
    print(f"[build] built in {dt/60:.1f} min -> {engine_path} ({ser.nbytes/1e9:.2f} GB)", flush=True)


# The standalone CLI that lived here drove a different engine and is not part of this
# package: lora/build_branch.py is the entry point, and it calls build_engine()
# with its own on_network hook to splice in the low-rank inputs.
