#!/usr/bin/env python3
"""Build a SA3 DiT TRT engine with an FP8 GEMM trunk: ModelOpt PTQ on top of
the FP16-mixed recipe (FP32 islands around RMSNorm/Softmax/RoPE) for ~1.8x
faster steps than ``dit_fp16mixed`` on Ada/Blackwell.

This is a PRODUCER recipe (like ``build_dit_fp16mixed.py``): it transforms the
canonical ``dit_fp16mixed.onnx`` into a quantized ``dit_fp8.onnx`` and compiles
it. Consumers who just want the engine compile the published ``dit_fp8.onnx``
with ``build_from_onnx.py`` (plain STRONGLY_TYPED, no graphsurgeon / ModelOpt).

Background — why this is more than ``mtq.quantize(...)``:
- The DiT's per-step velocity error compounds over the 8 pingpong steps. BF16
  fails (final-latent cos ~0.81); naive FP8 PTQ lands ~0.91. The decisive
  fixes, in order, take it to 0.9982 worst single-step latent cosine (n=376)
  and a compounded euler final-latent cosine of mean 0.953 / worst 0.873 over
  the 47 reprompt prompts vs the FP16-mixed engine; under the stochastic
  pingpong sampler the result is a different but comparable sample, judged by
  ear:
    1. ModelOpt rejects the canonical ONNX because upstream's island surgery
       leaves it un-toposorted, and its opset bump leaves ReduceMean's pre-18
       attribute-form axes. We Kahn-sort + version_convert to opset 19 first.
    2. ModelOpt corrupts single elements of a few initializers during
       preprocessing (e.g. ``dit.to_timestep_embed.2.bias`` absmax 0.12 ->
       6060). One exploded element inflates all adaLN conditioning -> fp16
       overflow -> a NaN engine and invalid calibration scales. We restore
       every corrupted initializer from the source graph and recalibrate all
       activation scales on a Q/DQ-bypassed copy (ORT, real conditioning).
    3. ModelOpt flattens the FP32 islands the fp16mixed recipe established
       (RMSNorm variance overflows fp16 -> NaN). We re-apply them with the
       FP16-mixed recipe's ``find_fp32_islands``, plus the conditioning
       front-end that computes in fp32 upstream (timestep expo features / cond
       embeds): in fp16 those flush and produce the entire t>=0.984 base
       divergence. Initializers we upcast take their true fp32 values from the
       source graph, not ModelOpt's fp16-rounded copies.
    4. Per-channel weight scales (GEMM N axis): weights are stored fp16 and
       quantized at build time, so one outlier row otherwise crushes the
       whole tensor's resolution. Per-channel constant-folds at build time and
       costs nothing at runtime. Activations stay per-tensor (TRT requires it
       for fp8 activation quant), calibrated with ``max`` (SA3 activation
       outliers are signal; percentile clipping noticeably regresses parity).

Calibration data is an INPUT (``--calib sa3-m.calib.npz``): a capture of real
(x_t, t, t5_hidden, t5_mask, seconds_total, local_add_cond) DiT inputs across
the pingpong sampling schedule. Produce it from the model checkpoint with the
companion ``make_calib.py`` (``python make_calib.py --model-config ... --checkpoint
... --out sa3-m.calib.npz``); the npz keys are the six ONNX input names, each a
leading-axis batch of samples.

Inputs/outputs stay FP32 so the runtime can swap engines transparently.

Validated (sa3-m, vs the FP16-mixed engine, over the 47 reprompt Music prompts
x 8 sigmas at L=646; the recipe is shape-independent: activation scales are
per-tensor and the default profile below is dynamic, so the numbers carry over):
  - Worst single-step latent cosine (x + dt*v, n=376): 0.9982 (mean 0.9997)
  - 8-step compounded euler final-latent cosine, distribution over the 47
    prompts: mean 0.953, median 0.957, p5 0.915, worst 0.873. The compounded
    rollout is chaotic at the early sigmas (an eps=1e-3 input perturbation
    alone compounds to ~0.967) and the FP16-mixed engine itself scores only
    ~0.998 vs PT eager, so this is a guide, not a gate; the shipped gate is
    decoded audio under the production pingpong sampler, judged by ear
    (RMS-curve correlation ~0.90 vs the FP16-mixed engine's generation).
  - Step time (B=1, L=646): ~10.6-11.0 ms (FP16-mixed: ~18.7-19.4 ms) -> ~1.8x.
    TRT tactic selection is nondeterministic per build; if a fresh engine
    benches noticeably slower, rebuild it.
  - A true batched forward amortizes (~1.4x at B=4) once compute drops, unlike
    FP16-mixed (<=1.09x): fp8 frees the SM throughput the FP16 engine saturated.

Usage:
    python build_dit_fp8.py
        --input  onnx/sa3-m/dit_fp16mixed.onnx
        --calib  sa3-m.calib.npz
        --onnx   /tmp/dit_fp8.onnx                  # intermediate (publishable)
        --engine ../models/<arch>/sa3-m/dit_fp8.trt
        [--islands-mode {minimal,rope,hybrid}]      # default: hybrid
        [--calib-samples 16] [--workspace-gb 16]
        [--work-dir DIR] [--keep-intermediates]     # ~15 GB scratch
        [--skip-convert] [--skip-build]

Requires (producer): tensorrt, torch, onnx, numpy, nvidia-modelopt,
onnxruntime-gpu (the repair pass calibrates activation scales on CUDA EP).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

BUILD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD_DIR))
# Reuse the FP16-mixed recipe's structural FP32-island finder verbatim (the fp8
# island re-apply is the same RMSNorm/Softmax/RoPE detection it uses).
from build_dit_fp16mixed import find_fp32_islands  # noqa: E402

T5_TOKENS = 256
T5_HIDDEN_DIM = 768
IO_CHANNELS = 256

SCALE_FLOOR = 6.2e-5   # fp16 min-normal floor for fp8-stored scales (TRT > 0)
FP8_MAX = 448.0        # e4m3 max

# Default sa3-m latent profile (matches the published FP16-mixed sa3-m engine
# so the fp8 engine is a drop-in). min=1 keeps short windows on-engine.
_DEFAULT_PROFILE_LATENTS = (1, 1292, 4096)  # (min, opt, max)


def _dit_profile(min_l: int, opt_l: int, max_l: int) -> dict:
    return {
        "x":              [(1, IO_CHANNELS, min_l), (1, IO_CHANNELS, opt_l), (1, IO_CHANNELS, max_l)],
        "t":              [(1,), (1,), (1,)],
        "t5_hidden":      [(1, T5_TOKENS, T5_HIDDEN_DIM)] * 3,
        "t5_mask":        [(1, T5_TOKENS)] * 3,
        "seconds_total":  [(1,), (1,), (1,)],
        "local_add_cond": [(1, 257, min_l), (1, 257, opt_l), (1, 257, max_l)],
    }

# Initializers ModelOpt is known to corrupt (single exploded elements). Every
# save downstream of the repair re-verifies these against the source graph,
# because an externalized modified tensor can silently revert on save.
KNOWN_RESTORES = (
    "dit.to_timestep_embed.2.bias",
    "dit.transformer.layers.22.to_local_embed.0.weight",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _kahn_sort(g) -> None:
    """Topologically sort ``g.node`` in place (Kahn). Values, initializers and
    external-data references are untouched."""
    available = {i.name for i in g.initializer}
    available.update(inp.name for inp in g.input)
    available.add("")
    pending, ordered = list(g.node), []
    while pending:
        progressed, remaining = False, []
        for n in pending:
            if all(i in available for i in n.input):
                ordered.append(n)
                available.update(n.output)
                progressed = True
            else:
                remaining.append(n)
        if not progressed:
            missing = {i for n in remaining for i in n.input} - available
            raise RuntimeError(f"toposort stuck: {list(missing)[:5]}")
        pending = remaining
    del g.node[:]
    g.node.extend(ordered)


def _detach_external(model) -> None:
    """``onnx.load`` fills ``raw_data`` but keeps stale external-data metadata;
    a later ``save_as_external_data`` then skips re-pointing those tensors and
    silently references old offsets. Detach everything so saves re-serialize."""
    import onnx

    for i in model.graph.initializer:
        if i.data_location == onnx.TensorProto.EXTERNAL:
            del i.external_data[:]
            i.data_location = onnx.TensorProto.DEFAULT


def _save_external(model, path: Path) -> None:
    """Save with external data, keeping repaired/scale tensors inline
    (size_threshold) and never appending into a pre-existing sidecar.

    The sidecar is ``<name>.onnx.data`` — the repo-wide convention
    (build_from_onnx.py / build_dit_fp16mixed.py) so the published
    ``dit_fp8.onnx`` + ``dit_fp8.onnx.data`` pair round-trips through HF and
    the consumer recipe resolves the external reference."""
    import onnx

    sidecar = Path(str(path) + ".data")
    for p in (path, sidecar):
        if p.exists():
            os.remove(p)
    onnx.save(model, str(path), save_as_external_data=True,
              location=sidecar.name, size_threshold=262144)


def _reverted_names(path: Path, src_inits: dict, names) -> list:
    """Reload ``path`` and return the subset of ``names`` whose values drifted
    >1% from ``src_inits`` (i.e. reverted to a corrupted copy on save)."""
    import onnx
    import onnx.numpy_helper as nh

    chk = onnx.load(str(path))
    ci = {i.name: i for i in chk.graph.initializer}
    out = []
    for name in names:
        if name not in ci or name not in src_inits:
            continue
        va = nh.to_array(ci[name]).astype(np.float32)
        vb = nh.to_array(src_inits[name]).astype(np.float32)
        if np.abs(va - vb).max() > np.abs(vb).max() * 1e-2 + 1e-12:
            out.append(name)
    return out


def _patch_reverts(path: Path, src_inits: dict, names, *, verify=True) -> list:
    """Inline-patch (proto only) any initializer in ``names`` that reverted to a
    corrupted value when ``path`` was saved, restoring it from ``src_inits``.
    Handles the external-data revert bug where a >256 KB tensor silently keeps
    stale bytes even after detaching. Returns the names patched."""
    import onnx
    import onnx.numpy_helper as nh

    reverted = _reverted_names(path, src_inits, names)
    if not reverted:
        return []
    mp = onnx.load(str(path), load_external_data=False)
    mi = {i.name: i for i in mp.graph.initializer}
    for name in reverted:
        t, s = mi[name], src_inits[name]
        del t.external_data[:]
        t.data_location = onnx.TensorProto.DEFAULT
        t.raw_data = s.raw_data if s.raw_data else nh.from_array(
            nh.to_array(s), name).raw_data
    onnx.save(mp, str(path))  # proto only
    if verify:
        still = _reverted_names(path, src_inits, reverted)
        assert not still, f"inline patch did not persist for {still}"
    return reverted


def _guard_known_restores(path: Path, source_onnx: Path) -> None:
    """Reload ``path``, compare KNOWN_RESTORES against ``source_onnx``, and
    inline-patch (proto only) any tensor that reverted on save."""
    import onnx

    src = onnx.load(str(source_onnx))
    src_inits = {i.name: i for i in src.graph.initializer}
    patched = _patch_reverts(path, src_inits, KNOWN_RESTORES)
    if patched:
        print(f"  [guard] {len(patched)} tensors reverted on save; "
              f"patched inline: {patched}")


# ---------------------------------------------------------------------------
# step 1: toposort + opset 19  (also a latent fix for the fp16mixed ONNX)
# ---------------------------------------------------------------------------


def toposort_opset19(input_onnx: Path, sorted_onnx: Path) -> None:
    """Kahn-sort the FP16-mixed graph and convert to a native opset 19.

    Upstream's island surgery leaves the node list un-toposorted (TRT's parser
    tolerates it; onnx.checker and ModelOpt do not). ModelOpt's own opset bump
    only rewrites the import and leaves ReduceMean's pre-18 ``axes`` attribute,
    which ORT then rejects. Do both correctly here so ModelOpt sees a clean
    opset-19 model. Copies the external-data sidecar next to the sorted proto.
    """
    import shutil

    import onnx

    m = onnx.load(str(input_onnx), load_external_data=False)
    _kahn_sort(m.graph)
    cur = max((imp.version for imp in m.opset_import
               if imp.domain in ("", "ai.onnx")), default=0)
    if cur < 19:
        from onnx import version_converter

        print(f"  converting opset {cur} -> 19 ...")
        m = version_converter.convert_version(m, 19)
    onnx.save(m, str(sorted_onnx))
    src_data = Path(str(input_onnx) + ".data")
    dst_data = Path(str(sorted_onnx).rsplit(".onnx", 1)[0] + ".onnx.data")
    # sorted proto references the sidecar by basename; mirror it alongside.
    want = None
    for i in m.graph.initializer:
        if i.data_location == onnx.TensorProto.EXTERNAL and i.external_data:
            want = i.external_data[0].value
            break
    if want is not None:
        dst_data = sorted_onnx.parent / want
    if src_data.exists() and not dst_data.exists():
        print(f"  copying external-data sidecar -> {dst_data.name}")
        shutil.copyfile(src_data, dst_data)
    onnx.checker.check_model(str(sorted_onnx))
    print(f"  toposorted opset-19 proto OK: {sorted_onnx}")


# ---------------------------------------------------------------------------
# step 2: fp8 PTQ
# ---------------------------------------------------------------------------


def quantize_fp8(sorted_onnx: Path, calib_npz: Path, out_onnx: Path) -> None:
    """ModelOpt FP8 PTQ of the weighted GEMMs only. ``disable_mha_qdq`` keeps
    attention BMMs and the softmax path on the FP16/FP32 recipe; the trunk
    around each Q/DQ stays fp16."""
    from modelopt.onnx.quantization import quantize

    data = dict(np.load(calib_npz))
    n = data[next(iter(data))].shape[0]
    print(f"  {n} calibration samples, fp8 PTQ on {sorted_onnx.name}")
    t0 = time.time()
    quantize(
        str(sorted_onnx),
        quantize_mode="fp8",
        calibration_data=data,
        calibration_method="max",   # amax scales; histogram calibrators fail on
                                    # the zero-range all-zeros local_add_cond
        calibration_eps=["cuda:0", "cpu"],
        op_types_to_quantize=["MatMul", "Gemm"],
        disable_mha_qdq=True,
        high_precision_dtype="fp16",
        use_external_data_format=True,
        output_path=str(out_onnx),
    )
    print(f"  wrote {out_onnx} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# step 3: repair corrupted initializers + recalibrate activation scales
# ---------------------------------------------------------------------------


def repair(fp8_onnx: Path, sorted_onnx: Path, calib_npz: Path,
           n_samples: int, out_onnx: Path) -> None:
    import onnx
    import onnx.numpy_helper as nh
    import onnxruntime as ort

    def _load_detached(path):
        mm = onnx.load(str(path))
        _detach_external(mm)
        return mm

    m = _load_detached(fp8_onnx)
    g = m.graph
    inits = {i.name: i for i in g.initializer}

    # --- 1. restore every corrupted initializer vs the source graph --------
    src = _load_detached(sorted_onnx)
    src_inits = {i.name: i for i in src.graph.initializer}
    restored_names = []
    for name in sorted(set(inits) & set(src_inits)):
        init = inits[name]
        if init.data_type == onnx.TensorProto.FLOAT8E4M3FN:
            continue  # quantized weights legitimately differ
        va = nh.to_array(init).astype(np.float32)
        vb = nh.to_array(src_inits[name]).astype(np.float32)
        if va.shape != vb.shape:
            continue
        ref = np.abs(vb).max() + 1e-12
        if np.abs(va - vb).max() / ref > 1e-2:
            n_bad = int((np.abs(va - vb) > ref * 1e-2).sum())
            orig_dtype = nh.to_array(init).dtype
            init.CopyFrom(nh.from_array(
                nh.to_array(src_inits[name]).astype(orig_dtype), name))
            print(f"  restored {name}: {n_bad} corrupted elems, "
                  f"absmax {np.abs(va).max():.4g} -> {np.abs(vb).max():.4g}")
            restored_names.append(name)
    print(f"  {len(restored_names)} corrupted initializers restored")

    # --- 2. build a Q/DQ-bypassed probe model ------------------------------
    LAYOUT = ("Transpose", "Reshape", "Squeeze", "Unsqueeze")
    bypass = onnx.ModelProto()
    bypass.CopyFrom(m)
    bg = bypass.graph
    bcons: dict[str, list] = {}
    for n in bg.node:
        for i in n.input:
            bcons.setdefault(i, []).append(n)
    pairs, drop = [], set()
    for q in list(bg.node):
        if q.op_type != "QuantizeLinear":
            continue
        cur, chain, dq = q.output[0], [], None
        for _ in range(4):
            cs = bcons.get(cur, [])
            if len(cs) != 1:
                break
            if cs[0].op_type == "DequantizeLinear":
                dq = cs[0]
                break
            if cs[0].op_type in LAYOUT:
                chain.append(cs[0])
                cur = cs[0].output[0]
                continue
            break
        if dq is None:
            continue
        x_in = q.input[0]
        if chain:
            chain[0].input[0] = x_in
            tail = chain[-1].output[0]
        else:
            tail = x_in
        for c in bg.node:
            for k, ci in enumerate(c.input):
                if ci == dq.output[0]:
                    c.input[k] = tail
        drop.add(q.name)
        drop.add(dq.name)
        pairs.append((q.name, dq.name, x_in))
    keep = [n for n in bg.node if n.name not in drop]
    del bg.node[:]
    bg.node.extend(keep)
    probe_tensors = sorted({p[2] for p in pairs
                            if p[2] not in {i.name for i in bg.initializer}})
    del bg.output[:]
    bg.output.extend(
        onnx.helper.make_empty_tensor_value_info(t) for t in probe_tensors)
    print(f"  bypassed {len(pairs)} q/dq pairs, probing {len(probe_tensors)} "
          f"activation tensors")
    bp_path = out_onnx.parent / "_fp8_repair_bypass.onnx"
    _save_external(bypass, bp_path)

    # --- 3. recalibrate activation amax on real conditioning ----------------
    d = np.load(calib_npz)
    n_total = d[next(iter(d.files))].shape[0]
    idx = np.linspace(0, n_total - 1, n_samples).round().astype(int)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(bp_path), so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    active_ep = sess.get_providers()[0]
    if "CUDA" not in active_ep:
        print(f"  WARNING: CUDA EP unavailable; calibrating on {active_ep}. "
              f"amax is EP-robust so scales are still valid, but install "
              f"onnxruntime-gpu for a faster repair pass.")
    print(f"  calibrating on {len(idx)} samples ({active_ep})")
    amax = {t: 0.0 for t in probe_tensors}
    nonfinite: set = set()
    feed_keys = ("x", "t", "t5_hidden", "t5_mask", "seconds_total",
                 "local_add_cond")
    for j, i in enumerate(idx):
        feed = {k: d[k][i:i + 1] for k in feed_keys}
        outs = sess.run(None, feed)
        for t, o in zip(probe_tensors, outs):
            o = np.asarray(o).astype(np.float32)
            fin = o[np.isfinite(o)]
            if fin.size != o.size:
                # additive-mask path (-inf fill): meaningless to quantize;
                # mark its q/dq pair for removal
                if t not in nonfinite:
                    print(f"  non-finite tensor {t}: q/dq pair removed")
                nonfinite.add(t)
            amax[t] = max(amax[t],
                          float(np.abs(fin).max()) if fin.size else 0.0)
        print(f"  sample {j + 1}/{len(idx)} done", flush=True)

    # --- 4. write recalibrated scales / neutralize mask-path pairs ----------
    node_by_name = {n.name: n for n in g.node}
    written = removed = 0
    for q_name, dq_name, x_in in pairs:
        if x_in in nonfinite:
            for nn in (q_name, dq_name):
                nd = node_by_name[nn]
                nd.op_type = "Identity"
                del nd.input[1:]
                del nd.attribute[:]
            removed += 1
            continue
        scale = max(amax[x_in] / FP8_MAX, SCALE_FLOOR)
        for nn in (q_name, dq_name):
            nd = node_by_name[nn]
            sinit = inits[nd.input[1]]
            orig_dtype = sinit.data_type
            arr = nh.to_array(sinit)
            sinit.CopyFrom(nh.from_array(np.array(scale, dtype=arr.dtype),
                                         nd.input[1]))
            assert sinit.data_type == orig_dtype
            written += 1
    del g.value_info[:]  # stale fp8 dtype claims on neutralized paths
    print(f"  wrote {written} recalibrated scales, neutralized {removed} "
          f"mask-path q/dq pairs")

    _save_external(m, out_onnx)
    # reload-verify EVERY restored tensor (large ones hit the external-data
    # revert bug even after detaching); inline-patch any that reverted. This
    # set is a superset of KNOWN_RESTORES, so no separate guard call is needed.
    patched = _patch_reverts(out_onnx, src_inits, restored_names)
    if patched:
        print(f"  inline-patched {len(patched)} reverted restores")
    print(f"  wrote+verified {out_onnx} ({len(restored_names)} restores)")


# ---------------------------------------------------------------------------
# step 4: re-apply FP32 islands (ModelOpt flattened them)
# ---------------------------------------------------------------------------


def reapply_fp32_islands(repaired_onnx: Path, sorted_onnx: Path,
                         out_onnx: Path, mode: str = "hybrid") -> None:
    """Re-protect the FP32 islands the fp16mixed recipe established, which
    ModelOpt strips. ``hybrid`` = structural islands (RMSNorm/Softmax) + RoPE
    region + the conditioning front-end that computes fp32 upstream. Upcast
    island initializers to their true fp32 values from the source graph."""
    import re

    import onnx
    import onnx.numpy_helper as nh

    m = onnx.load(str(repaired_onnx))
    _detach_external(m)
    g = m.graph
    by_name = {n.name: n for n in g.node}
    blocked = set(find_fp32_islands(m, mode="minimal"))

    FP16, FP32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(str(repaired_onnx), load_external_data=False))
    dtype = {vi.name: vi.type.tensor_type.elem_type
             for vi in (*inferred.graph.value_info, *inferred.graph.output,
                        *inferred.graph.input)}
    for i in g.initializer:
        dtype[i.name] = i.data_type

    upstream_fp32_inits: dict = {}
    if mode in ("rope", "hybrid"):
        srt_proto = onnx.load(str(sorted_onnx), load_external_data=False)
        rope_named = (set(find_fp32_islands(srt_proto, mode="rope"))
                      - set(find_fp32_islands(srt_proto, mode="minimal")))
        kept = 0
        for name in rope_named:
            n = by_name.get(name)
            if n is not None and any(dtype.get(o) == FP16 for o in n.output):
                blocked.add(name)
                kept += 1
        print(f"  rope extras: {kept} fp16-carrying nodes kept")
        srt_full = onnx.load(str(sorted_onnx))
        for i in srt_full.graph.initializer:
            if i.data_type == FP32:
                upstream_fp32_inits[i.name] = nh.to_array(i)
        del srt_full
        print(f"  loaded {len(upstream_fp32_inits)} upstream fp32 "
              f"initializers for value restore")

    if mode == "hybrid":
        # the conditioning front-end (timestep expo features / cond embeds /
        # memory-token plumbing — everything before the transformer layers)
        # computes fp32 upstream; fp16 here flushes the expo features and is
        # the entire t>=0.984 base divergence.
        inferred_s = onnx.shape_inference.infer_shapes(
            onnx.load(str(sorted_onnx), load_external_data=False))
        sdtype = {vi.name: vi.type.tensor_type.elem_type
                  for vi in (*inferred_s.graph.value_info,
                             *inferred_s.graph.output, *inferred_s.graph.input)}
        srt_p = onnx.load(str(sorted_onnx), load_external_data=False)
        for i in srt_p.graph.initializer:
            sdtype[i.name] = i.data_type
        layer_pat = re.compile(r"^/transformer/layers\.")
        ours_by_out = {o: n for n in g.node for o in n.output}
        fe = 0
        for n in srt_p.graph.node:
            if n.op_type == "Cast":
                continue
            for o in n.output:
                if sdtype.get(o) == FP32 and not layer_pat.match(o):
                    ours = ours_by_out.get(o)
                    if ours is not None and ours.name:
                        blocked.add(ours.name)
                        fe += 1
        print(f"  hybrid front-end fp32 ops added: {fe}")

    blocked_nodes = [by_name[b] for b in blocked if b in by_name]
    print(f"  {len(blocked_nodes)} island nodes "
          f"({sum(1 for n in blocked_nodes if n.op_type == 'Softmax')} Softmax)")

    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    blocked_set = {n.name for n in blocked_nodes}

    def _const_dtype(n):
        for a in n.attribute:
            if a.name == "value":
                return a.t.data_type
        return None

    new_nodes, upcast_inits, casts_in, casts_out = [], 0, 0, 0
    for n in blocked_nodes:
        for k, inp in enumerate(n.input):
            if n.op_type.startswith("Reduce") and k >= 1:
                continue  # int64 axes initializer — never wrap in a Cast
            if inp in inits and inits[inp].data_type == FP16:
                if inp in upstream_fp32_inits:
                    arr = np.asarray(upstream_fp32_inits[inp], dtype=np.float32)
                else:
                    arr = nh.to_array(inits[inp]).astype(np.float32)
                inits[inp].CopyFrom(nh.from_array(arr, inp))
                upcast_inits += 1
            elif inp in prod and prod[inp].op_type == "Constant" \
                    and _const_dtype(prod[inp]) == FP16:
                for a in prod[inp].attribute:
                    if a.name == "value":
                        a.t.CopyFrom(nh.from_array(
                            nh.to_array(a.t).astype(np.float32), a.t.name))
                upcast_inits += 1
            elif inp in prod and prod[inp].name not in blocked_set \
                    and dtype.get(inp, FP16) == FP16:
                cast_out = f"{inp}__refp32_{casts_in}"
                new_nodes.append(onnx.helper.make_node(
                    "Cast", [inp], [cast_out],
                    name=f"refp32_in_{casts_in}", to=FP32))
                n.input[k] = cast_out
                casts_in += 1
        for o in list(n.output):
            if dtype.get(o, FP16) != FP16:
                continue
            consumers = [c for c in g.node
                         if o in c.input and c.name not in blocked_set
                         and not c.name.startswith("refp32_")]
            if not consumers:
                continue
            cast_out = f"{o}__refp16_{casts_out}"
            new_nodes.append(onnx.helper.make_node(
                "Cast", [o], [cast_out],
                name=f"refp16_out_{casts_out}", to=FP16))
            for c in consumers:
                for k, ci in enumerate(c.input):
                    if ci == o:
                        c.input[k] = cast_out
            casts_out += 1

    g.node.extend(new_nodes)
    _kahn_sort(g)
    del g.value_info[:]
    print(f"  upcast {upcast_inits} consts, {casts_in} in-casts, "
          f"{casts_out} out-casts")
    _save_external(m, out_onnx)
    onnx.checker.check_model(str(out_onnx))
    _guard_known_restores(out_onnx, sorted_onnx)
    print(f"  saved {out_onnx}")


# ---------------------------------------------------------------------------
# step 5: per-channel weight scales
# ---------------------------------------------------------------------------


def perchannel_weights(islands_onnx: Path, out_onnx: Path,
                       sorted_onnx: Path) -> None:
    """Upgrade weight-side Q/DQ pairs (initializer -> Transpose -> Q -> DQ ->
    MatMul) to per-channel scales along the GEMM N (output-feature) axis.
    Constant-folds at build time, free at runtime. Activation pairs stay
    per-tensor (TRT requires it for fp8 activation quant).

    This is the FINAL save of the published artifact, so it re-externalizes
    every initializer (incl. the >256 KB ``layers.22`` weight that hits the
    external-data revert bug); re-guard the known restores afterwards."""
    import onnx
    import onnx.numpy_helper as nh

    m = onnx.load(str(islands_onnx))
    _detach_external(m)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    upgraded = skipped = 0
    upgraded_scales: set = set()
    for q in g.node:
        if q.op_type != "QuantizeLinear":
            continue
        p = prod.get(q.input[0])
        if p is None or p.op_type != "Transpose" or p.input[0] not in inits:
            continue
        w_init = inits[p.input[0]]
        if w_init.data_type == onnx.TensorProto.FLOAT8E4M3FN:
            continue
        w = nh.to_array(w_init).astype(np.float32)
        if w.ndim != 2:
            skipped += 1
            continue
        perm = [a.ints for a in p.attribute if a.name == "perm"]
        if perm and list(perm[0]) != [1, 0]:
            skipped += 1
            continue
        cs = cons.get(q.output[0], [])
        if len(cs) != 1 or cs[0].op_type != "DequantizeLinear":
            skipped += 1
            continue
        dq = cs[0]
        # stored weight is [out, in]; per-row amax = per-output-channel scale
        # (axis 1 of the [in, out] transposed weight the GEMM consumes).
        amax = np.abs(w).max(axis=1)
        scales = np.maximum(amax / FP8_MAX, SCALE_FLOOR)
        for node in (q, dq):
            sinit = inits[node.input[1]]
            orig_dtype = nh.to_array(sinit).dtype
            sinit.CopyFrom(nh.from_array(scales.astype(orig_dtype), node.input[1]))
            if len(node.input) > 2 and node.input[2] in inits:
                zp = inits[node.input[2]]
                znew = np.zeros(scales.shape, dtype=nh.to_array(zp).dtype)
                if zp.data_type == onnx.TensorProto.FLOAT8E4M3FN:
                    t = onnx.helper.make_tensor(
                        node.input[2], zp.data_type, list(znew.shape),
                        bytes(len(znew.ravel())), raw=True)
                else:
                    t = nh.from_array(znew, node.input[2])
                    t.data_type = zp.data_type
                zp.CopyFrom(t)
            for a in list(node.attribute):
                if a.name == "axis":
                    node.attribute.remove(a)
            node.attribute.append(onnx.helper.make_attribute("axis", 1))
            upgraded_scales.add(node.input[1])
        upgraded += 1
    print(f"  upgraded {upgraded} weight pairs to per-channel, skipped {skipped}")
    del g.value_info[:]
    _save_external(m, out_onnx)
    onnx.checker.check_model(str(out_onnx))
    # the final save re-guards the corrupted-then-restored initializers, then
    # confirms the per-channel scales survived the round-trip (they grew from
    # scalars to vectors but stay inline, well under size_threshold).
    _guard_known_restores(out_onnx, sorted_onnx)
    chk = onnx.load(str(out_onnx), load_external_data=False)
    nvec = sum(1 for i in chk.graph.initializer
               if i.name in upgraded_scales and list(i.dims) not in ([], [1]))
    assert nvec >= upgraded, \
        f"per-channel scales did not persist ({nvec} vectors < {upgraded})"
    print(f"  saved {out_onnx} ({nvec} per-channel scale vectors verified)")


# ---------------------------------------------------------------------------
# step 6: build the TRT engine
# ---------------------------------------------------------------------------


def build_trt_engine(onnx_path: Path, engine_path: Path,
                     profile_latents: tuple = _DEFAULT_PROFILE_LATENTS,
                     workspace_gb: int = 16) -> None:
    import tensorrt as trt

    print(f"\n  building TRT engine -> {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  parse error: {parser.get_error(i)}")
        sys.exit(2)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    for name, (lo, opt, hi) in _dit_profile(*profile_latents).items():
        profile.set_shape(name, lo, opt, hi)
    cfg.add_optimization_profile(profile)
    print(f"  latent profile (min/opt/max): {profile_latents}")
    print(f"  building (workspace {workspace_gb} GB, STRONGLY_TYPED, fp8 trunk)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print("  BUILD FAILED")
        sys.exit(3)
    print(f"  built in {time.time() - t0:.0f}s ({serialized.nbytes / 1e6:.0f} MB)")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  wrote {engine_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def convert_to_fp8(input_onnx: Path, calib_npz: Path, out_onnx: Path,
                   *, mode: str, calib_samples: int, work_dir: Path) -> None:
    # Intermediates (five ~3 GB ONNX + sidecars) live in a dedicated work
    # dir, NOT next to the published artifact — the documented usage writes
    # --onnx straight into the HF repo, which must hold only dit_fp8.onnx(.data).
    work = work_dir
    work.mkdir(parents=True, exist_ok=True)
    sorted_onnx = work / "_fp8_sorted.onnx"
    quant_onnx = work / "_fp8_quant.onnx"
    repaired_onnx = work / "_fp8_repaired.onnx"
    islands_onnx = work / "_fp8_islands.onnx"

    print("=== [1/5] toposort + opset 19 ===")
    toposort_opset19(input_onnx, sorted_onnx)
    print("=== [2/5] fp8 PTQ ===")
    quantize_fp8(sorted_onnx, calib_npz, quant_onnx)
    print("=== [3/5] repair + recalibrate ===")
    repair(quant_onnx, sorted_onnx, calib_npz, calib_samples, repaired_onnx)
    print(f"=== [4/5] re-apply FP32 islands (mode={mode}) ===")
    reapply_fp32_islands(repaired_onnx, sorted_onnx, islands_onnx, mode=mode)
    print("=== [5/5] per-channel weight scales ===")
    perchannel_weights(islands_onnx, out_onnx, sorted_onnx)
    print(f"\nfp8 ONNX ready: {out_onnx}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Canonical FP16-mixed ONNX (onnx/sa3-m/dit_fp16mixed.onnx)")
    ap.add_argument("--calib", required=True,
                    help="Calibration .npz (six DiT input tensors, batched)")
    ap.add_argument("--onnx", default="/tmp/dit_fp8.onnx",
                    help="Output fp8 ONNX (intermediate; publishable to HF)")
    ap.add_argument("--engine", default=None,
                    help="Output TRT engine path (default: alongside --onnx)")
    ap.add_argument("--islands-mode", choices=("minimal", "rope", "hybrid"),
                    default="hybrid",
                    help="FP32 island coverage (hybrid = +RoPE +front-end; "
                         "the validated recipe)")
    ap.add_argument("--calib-samples", type=int, default=16,
                    help="Samples used to recalibrate activation scales")
    ap.add_argument("--min-latents", type=int, default=_DEFAULT_PROFILE_LATENTS[0])
    ap.add_argument("--opt-latents", type=int, default=_DEFAULT_PROFILE_LATENTS[1])
    ap.add_argument("--max-latents", type=int, default=_DEFAULT_PROFILE_LATENTS[2],
                    help="TRT latent profile (min/opt/max); default = sa3-m's "
                         "published profile")
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--work-dir", default=None,
                    help="Scratch dir for the ~15 GB of _fp8_* intermediates "
                         "(default: <onnx-dir>/_fp8_work, auto-removed on "
                         "success; a user-supplied dir is never auto-removed)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="Leave the default work dir in place after conversion")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Reuse an existing --onnx (just build)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Only produce the fp8 ONNX")
    args = ap.parse_args()

    out_onnx = Path(args.onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    work_is_default = args.work_dir is None
    work_dir = out_onnx.parent / "_fp8_work" if work_is_default else Path(args.work_dir)
    if not args.skip_convert:
        convert_to_fp8(Path(args.input), Path(args.calib), out_onnx,
                       mode=args.islands_mode, calib_samples=args.calib_samples,
                       work_dir=work_dir)
    if not args.skip_build:
        engine = Path(args.engine) if args.engine else out_onnx.with_suffix(".trt")
        print("\n=== build TRT engine ===")
        build_trt_engine(
            out_onnx, engine,
            profile_latents=(args.min_latents, args.opt_latents, args.max_latents),
            workspace_gb=args.workspace_gb)
    # Only auto-remove the dedicated dir we created; never rmtree a
    # user-supplied --work-dir (it may be shared or hold other files).
    if (work_is_default and not args.skip_convert
            and not args.keep_intermediates and work_dir.exists()):
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"cleaned intermediates: {work_dir}")


if __name__ == "__main__":
    main()
