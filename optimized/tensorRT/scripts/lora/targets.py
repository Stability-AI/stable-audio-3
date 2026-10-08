"""Which linears of the SA3-medium DiT get a LoRA branch, and how to find them in the ONNX.

Lifted out of the refit-engine builder so the branch build does not depend on it: these
tables are a property of the model graph, not of either engine. `classify(mm_name)` maps an
ONNX MatMul node name onto the checkpoint suffix an adapter uses for the same linear.
"""
import tensorrt as trt

TARGETS = {
    "self_attn/to_qkv": "self_attn.to_qkv",
    "self_attn/to_out": "self_attn.to_out",
    "cross_attn/to_q": "cross_attn.to_q",
    "cross_attn/to_kv": "cross_attn.to_kv",
    "cross_attn/to_out": "cross_attn.to_out",
    "ff/ff.0/proj": "ff.ff.0.proj",
    "ff/ff.2": "ff.ff.2",
}


# to_local_embed is per-layer too, but only ~10% of trained adapters touch it (0 of the 18
# most recent), so it is opt-in -- every branch is graph weight that most adapters never use.
LOCAL_TARGETS = {
    "to_local_embed/to_local_embed.0": "to_local_embed.0",
    "to_local_embed/to_local_embed.2": "to_local_embed.2",
}

# Adapted linears that sit OUTSIDE the transformer blocks, keyed by exact layer name because
# they carry no /layers.N/ to match on. Each carries (map slug, bases suffix, adapter key):
# the adapter key is absolute because these do not share the block linears' prefix -- the
# seconds embedder lives under conditioners., the to_*_embed stack under model. .
#
# /Gemm is the seconds_total embedder ((1,256)->(1,768), the tail of the sinusoidal chain).
# EVERY sa3-medium adapter trained so far targets it -- 181 of 181 across both run trees --
# and it is the only layer standing between the branch engine and canonical parity for the
# whole recent corpus, so it is always branched, never opt-in.
SECONDS_TARGET = {
    "/Gemm": ("seconds_embed", "conditioners.seconds_total.embedder.embedding.1",
              "conditioners.seconds_total.embedder.embedding.1"),
}
EXTRA_TARGETS = {
    "/to_timestep_embed/to_timestep_embed.0/Gemm":
        ("to_timestep_embed.0", "to_timestep_embed.0", "model.to_timestep_embed.0"),
    "/to_timestep_embed/to_timestep_embed.2/Gemm":
        ("to_timestep_embed.2", "to_timestep_embed.2", "model.to_timestep_embed.2"),
    "/to_cond_embed/to_cond_embed.0/MatMul":
        ("to_cond_embed.0", "to_cond_embed.0", "model.to_cond_embed.0"),
    "/to_cond_embed/to_cond_embed.2/MatMul":
        ("to_cond_embed.2", "to_cond_embed.2", "model.to_cond_embed.2"),
    "/to_global_embed/to_global_embed.0/MatMul":
        ("to_global_embed.0", "to_global_embed.0", "model.to_global_embed.0"),
    "/to_global_embed/to_global_embed.2/MatMul":
        ("to_global_embed.2", "to_global_embed.2", "model.to_global_embed.2"),
    "/transformer/global_cond_embedder/global_cond_embedder.0/Gemm":
        ("global_cond_embedder.0", "global_cond_embedder.0",
         "model.transformer.global_cond_embedder.0"),
    "/transformer/global_cond_embedder/global_cond_embedder.2/Gemm":
        ("global_cond_embedder.2", "global_cond_embedder.2",
         "model.transformer.global_cond_embedder.2"),
    "/transformer/project_in/MatMul":
        ("project_in", "project_in", "model.transformer.project_in"),
    "/transformer/project_out/MatMul":
        ("project_out", "project_out", "model.transformer.project_out"),
}
# The last 2 of the 228 adapted DiT tensors. They parse as CONVOLUTION, not MATRIX_MULTIPLY,
# and their activation is NCHW [1,C,L,1] rather than [1,S,D]. Both are kernel 1x1 and carry NO
# bias, which is what makes them safe to branch: a TRT convolution fuses its bias, and a DoRA
# pout rescale on a biased conv output would scale the bias too -- which canonical DoRA, which
# parametrises only .weight, does not do.
CONV_TARGETS = {
    "/preprocess_conv/Conv":  ("preprocess_conv", "preprocess_conv", "model.preprocess_conv"),
    "/postprocess_conv/Conv": ("postprocess_conv", "postprocess_conv", "model.postprocess_conv"),
}


def as_constant(layer):
    """TRT Python hands back ILayer; reassigning __class__ is the documented way to reach the
    IConstantLayer members (there is no cast method)."""
    layer.__class__ = trt.IConstantLayer
    return layer


def classify(mm_name, targets=None):
    """-> (weight_name, ckpt_suffix) or (None, None). Longest pattern first so cross_attn/to_q
    cannot shadow a longer match."""
    import re
    targets = TARGETS if targets is None else targets
    m = re.search(r"/layers\.(\d+)/", mm_name)
    if not m:
        return None, None
    for frag in sorted(targets, key=len, reverse=True):
        if "/" + frag + "/" in mm_name:
            suf = targets[frag]
            return f"lora::layers.{m.group(1)}.{suf}", f"layers.{m.group(1)}.{suf}"
    return None, None

