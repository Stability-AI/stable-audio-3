"""Merge LoRA adapters into a baked SA3 DiT ``.tflite`` by patching its weight
buffers — the TFLite counterpart of the MLX weight-dict merge.

A ``.tflite`` is a frozen FlatBuffer: there is no weight dict to edit at load, so
LoRA inference means rewriting the FULLY_CONNECTED weight buffers in place. The
mechanism (validated on the 5.8 GB medium DiT):

  1. Parse the FlatBuffer header (mmap, no full read) and map every
     checkpoint layer name → the file offset/size/dtype of its FC weight buffer.
  2. Clone the base file (APFS ``cp -c`` = a few ms and no extra disk until
     written; plain copy elsewhere).
  3. For each adapter layer: read W0 from the clone's mmap, compute W' with the
     shared numpy merge math (``lora_core.merged_weight``), and overwrite the
     buffer's bytes at its exact offset. Deltas from multiple adapters accumulate
     against the ORIGINAL W0 (order-independent), matching the MLX path.

Patching happens BEFORE ``Interpreter(...)`` so XNNPACK packs the merged weights
at ``allocate_tensors`` exactly as for an unpatched model — no runtime cost, no
cache interaction. Patched files are cached by a content hash of (base, adapters,
strengths), so repeat runs reuse them instantly.

Precision: fp32 (weights are direct fp32 consts) and w16a32 (weights are fp16
consts behind a DEQUANTIZE) are supported. Quantized-int8 variants (w8a32,
w8a8-dyn, w4a32) are refused — merging requires a dequant→merge→requant that
sacrifices the GPTQ error-feedback grid (a phase-2 item).

Name mapping tiers (all hard-fail on ambiguity — never silently mis-patch):
  1. Block linears (self_attn/cross_attn/ff): name carries ``TransformerBlock_N``
     → (block, module, leaf) + shape → unique.
  2. ``to_local_embed.seq.{0,2}``: per-block but the tflite names carry NO block
     index and their ``;NN`` counter runs BACKWARDS — block identity comes from
     FC topological order within the group, never the counter.
  3. Prefix embedders / project_in / project_out: singleton names → unique.
  4. Seconds conditioner: the unique ``(768, 256)`` FC (in-graph, patchable).
"""

from __future__ import annotations

import hashlib
import mmap
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_core as lc  # noqa: E402

# Bump when the merge math or mapping changes so stale cache entries are missed.
_MERGE_VERSION = "1"

_QUANTIZED_PRECISIONS = ("w8a32", "w8a8-dyn", "w4a32")


# ── FlatBuffer FC-weight discovery ─────────────────────────────────────────────

def _fc_weight_buffers(buf):
    """Yield one record per FULLY_CONNECTED weight, in graph topological order:
    ``dict(name, shape, off, size, np_dtype, op_index)``.

    Resolves the weight buffer through a DEQUANTIZE (w16a32: fp16 const) or takes
    the FC's weight input directly (fp32: fp32 const). Refuses int8 weights.
    """
    from ai_edge_litert import schema_py_generated as schema
    BO = schema.BuiltinOperator
    TT = schema.TensorType
    _NP = {TT.FLOAT32: np.float32, TT.FLOAT16: np.float16}

    m = schema.Model.GetRootAs(buf, 0)
    sg = m.Subgraphs(0)
    producer = {}
    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        bc = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        for j in range(op.OutputsLength()):
            producer[op.Outputs(j)] = (bc, op)

    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        if m.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != BO.FULLY_CONNECTED:
            continue
        w_idx = op.Inputs(1)
        wt = sg.Tensors(w_idx)
        shape = tuple(wt.Shape(i) for i in range(wt.ShapeLength()))
        name = (wt.Name() or b"?").decode()
        prod = producer.get(w_idx)
        if prod is not None and prod[0] == BO.DEQUANTIZE:
            # w16a32: FC weight ← DEQUANTIZE ← fp16 const (the patch target).
            src = sg.Tensors(prod[1].Inputs(0))
            ttype = src.Type()
            b = m.Buffers(src.Buffer())
            src_name = (src.Name() or wt.Name() or b"?").decode()
        else:
            ttype, b, src_name = wt.Type(), m.Buffers(wt.Buffer()), name
        if ttype not in _NP:
            raise lc.LoraError(
                f"FC weight {name!r} is {TT.__dict__ and _tt_name(ttype)} — LoRA merge "
                f"on the TFLite backend supports fp32 and w16a32 (fp16) weights only. "
                f"Quantized-int8 models (w8a32 / w8a8-dyn / w4a32) are not supported "
                f"(merging would destroy the GPTQ grid). Use --precision fp32 or w16a32."
            )
        off, size = b.Offset(), b.Size()
        if off == 0 or size == 0:
            raise lc.LoraError(
                f"FC weight {name!r} is not stored in an out-of-band buffer "
                f"(off={off}, size={size}) — cannot patch in place."
            )
        yield dict(name=name, dbg=src_name, shape=shape, off=off, size=size,
                   np_dtype=_NP[ttype], op_index=oi)


def _tt_name(code):
    from ai_edge_litert import schema_py_generated as schema
    return next((k for k, v in schema.TensorType.__dict__.items() if v == code), str(code))


# ── name classification → checkpoint-layer resolution ──────────────────────────

_LEAF = re.compile(r"torch\.nn\.modules\.linear\.Linear_([A-Za-z0-9_]+)")
_BLOCK = re.compile(r"TransformerBlock_(\d+)/")

# checkpoint (post model.-strip) block-module suffix → (module tag, tflite leaf, glu)
_BLOCK_CKPT = {
    "self_attn.to_qkv": ("self_attn", "to_qkv", False),
    "self_attn.to_out": ("self_attn", "to_out", False),
    "cross_attn.to_q": ("cross_attn", "to_q", False),
    "cross_attn.to_kv": ("cross_attn", "to_kv", False),
    "cross_attn.to_out": ("cross_attn", "to_out", False),
    "ff.ff.0.proj": ("ff", "proj", True),
    "ff.ff.2": ("ff", "2", False),
    # to_local_embed handled separately (topological order); accept both the
    # torch (.0/.2) and MLX (.seq.0/.seq.2) spellings.
    "to_local_embed.0": ("local_embed", "0", False),
    "to_local_embed.2": ("local_embed", "2", False),
    "to_local_embed.seq.0": ("local_embed", "0", False),
    "to_local_embed.seq.2": ("local_embed", "2", False),
}

# non-block singleton checkpoint names → a substring that uniquely identifies the
# tflite FC name (verified against the exported graph).
_SINGLETON_CKPT = {
    "transformer.project_in": "Linear_project_in",
    "transformer.project_out": "Linear_project_out",
    "to_timestep_embed.0": "Sequential_to_timestep_embed/torch.nn.modules.linear.Linear_0",
    "to_timestep_embed.2": "Sequential_to_timestep_embed/torch.nn.modules.linear.Linear_2",
    "to_cond_embed.0": "Sequential_to_cond_embed/torch.nn.modules.linear.Linear_0",
    "to_cond_embed.2": "Sequential_to_cond_embed/torch.nn.modules.linear.Linear_2",
    "to_global_embed.0": "Sequential_to_global_embed/torch.nn.modules.linear.Linear_0",
    "to_global_embed.2": "Sequential_to_global_embed/torch.nn.modules.linear.Linear_2",
    "transformer.global_cond_embedder.0": "Sequential_global_cond_embedder/torch.nn.modules.linear.Linear_0",
    "transformer.global_cond_embedder.2": "Sequential_global_cond_embedder/torch.nn.modules.linear.Linear_2",
}


def _module_tag(name: str) -> str | None:
    if "SelfAttention_self_attn" in name:
        return "self_attn"
    if "CrossAttention_cross_attn" in name:
        return "cross_attn"
    if "Seq_to_local_embed" in name or "Sequential_seq" in name:
        return "local_embed"
    if "FeedForward_ff" in name:
        return "ff"
    return None


class _Mapper:
    """Resolves checkpoint layer names → FC weight records for one base model."""

    def __init__(self, fcs):
        self._fcs = fcs
        # tier-1 block linears: (block, module, leaf, glu) → record
        self._block = {}
        # tier-2 local_embed: leaf ('0'/'2') → [records in topological order]
        self._local = {"0": [], "2": []}
        self._seconds = []  # tier-4
        for rec in fcs:  # fcs already in topological (op) order
            name = rec["name"]
            mod = _module_tag(name)
            mleaf = _LEAF.search(name)
            leaf = mleaf.group(1) if mleaf else None
            mblk = _BLOCK.search(name)
            if mod == "local_embed" and leaf in ("0", "2"):
                self._local[leaf].append(rec)
            elif mod in ("self_attn", "cross_attn", "ff") and mblk and leaf:
                glu = "_GLUWrap_0" in name
                self._block[(int(mblk.group(1)), mod, leaf, glu)] = rec
            if rec["shape"] == (768, 256):
                self._seconds.append(rec)

    def resolve(self, layer: str, want_shape):
        """Return the FC record for a checkpoint layer, or raise LoraError."""
        if layer == lc.COND_SECONDS_LAYER:
            cands = [r for r in self._seconds if r["shape"] == want_shape]
            return self._one(layer, want_shape, cands, "seconds conditioner")

        stripped = layer[len("model."):] if layer.startswith("model.") else layer
        mm = re.match(r"transformer\.layers\.(\d+)\.(.+)$", stripped)
        if mm:
            block, suffix = int(mm.group(1)), mm.group(2)
            spec = _BLOCK_CKPT.get(suffix)
            if spec is None:
                raise lc.LoraError(f"{layer}: unrecognised block sub-module {suffix!r}")
            module, leaf, glu = spec
            if module == "local_embed":
                # tier 2 — topological rank within the group is the block index.
                group = self._local[leaf]
                if block >= len(group):
                    raise lc.LoraError(
                        f"{layer}: to_local_embed block {block} out of range "
                        f"(only {len(group)} in the graph)")
                rec = group[block]
                if rec["shape"] != tuple(want_shape):
                    raise lc.LoraError(
                        f"{layer}: shape {rec['shape']} != adapter {tuple(want_shape)} "
                        f"— wrong base model?")
                return rec
            rec = self._block.get((block, module, leaf, glu))
            cands = [rec] if rec is not None and rec["shape"] == tuple(want_shape) else []
            return self._one(layer, want_shape, cands, "block linear")

        sub = _SINGLETON_CKPT.get(stripped)
        if sub is None:
            raise lc.LoraError(f"{layer}: no TFLite tensor mapping for this layer")
        cands = [r for r in self._fcs if sub in r["name"] and r["shape"] == tuple(want_shape)]
        return self._one(layer, want_shape, cands, "singleton")

    @staticmethod
    def _one(layer, want_shape, cands, tier):
        if len(cands) == 1:
            return cands[0]
        raise lc.LoraError(
            f"{layer}: expected exactly 1 matching FC ({tier}, shape {tuple(want_shape)}), "
            f"found {len(cands)} — wrong base model or unsupported target module?"
        )


# ── clone + patch + cache ──────────────────────────────────────────────────────

def _clone(src: str, dst: str) -> None:
    """COW-clone src→dst (APFS ``cp -c``, instant) or fall back to a full copy."""
    try:
        r = subprocess.run(["cp", "-c", src, dst], capture_output=True)
        if r.returncode == 0 and os.path.exists(dst):
            return
    except (OSError, FileNotFoundError):
        pass
    shutil.copyfile(src, dst)


def _cache_key(base_path: str, specs) -> str:
    h = hashlib.sha256()
    h.update(_MERGE_VERSION.encode())
    st = os.stat(base_path)
    h.update(f"{os.path.basename(base_path)}:{st.st_size}".encode())
    for path, strength in sorted((lc._resolve_path(s["path"]), s["strength"]) for s in specs):
        fh = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        h.update(f"{fh.hexdigest()}:{strength}".encode())
    return h.hexdigest()[:12]


def get_patched_dit(base_path, specs, *, family: str, precision: str,
                    cache_dir=None, log=print) -> Path:
    """Return a path to a DiT ``.tflite`` with ``specs`` (from
    :func:`lora_core.parse_lora_spec`) merged into its weights. Result is cached
    by content hash; a hit returns instantly. Raises LoraError on any mapping,
    shape, precision, or trust-boundary failure — never mis-patches silently.
    """
    base_path = str(base_path)
    if precision in _QUANTIZED_PRECISIONS:
        raise lc.LoraError(
            f"--lora is not supported with --precision {precision} (quantized-int8) — "
            f"merging int8 weights would destroy the GPTQ grid. Use fp32 or w16a32."
        )

    cache_dir = Path(cache_dir or (Path(base_path).resolve().parents[2] / "lora_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(base_path, specs)
    out = cache_dir / f"dit_{family}_{precision}_{key}.tflite"
    if out.exists():
        log(f"lora: reusing cached patched model {out.name}")
        return out

    # Parse all adapters up front (fail fast before the clone).
    parsed = []
    for s in specs:
        path = lc._resolve_path(s["path"])
        adapter_type, scaling, layers = lc.parse_adapter(path)
        parsed.append((os.path.basename(path), adapter_type, scaling,
                       float(s["strength"]), layers))
        log(f"lora: {os.path.basename(path)} — {adapter_type}, "
            f"scaling {scaling:g}, strength {s['strength']:g}, {len(layers)} layers")

    # Map every adapter layer against the base model BEFORE cloning (so a
    # wrong-base adapter errors without leaving a stale clone behind).
    with open(base_path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        fcs = list(_fc_weight_buffers(buf))
        mapper = _Mapper(fcs)
        # accum[off] = [record, W0(fp32), summed_delta(fp32)]
        accum = {}
        matched = 0
        for name, atype, scaling, strength, layers in parsed:
            need = lc._PARAMS_FOR.get(atype, ())
            for layer, params in layers.items():
                want = (params.get("lora_B", params.get("M_xs")).shape[0],
                        params["lora_A"].shape[1] if "lora_A" in params
                        else params["M_xs"].shape[1])
                # For -xs, fan_in comes from the base weight; resolve then re-check.
                rec = mapper.resolve(layer, want if atype not in
                                     ("lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs")
                                     else _shape_from_leaf(mapper, layer, params))
                W0 = np.frombuffer(buf, rec["np_dtype"],
                                   count=rec["size"] // np.dtype(rec["np_dtype"]).itemsize,
                                   offset=rec["off"]).reshape(rec["shape"]).astype(np.float32)
                missing = [n for n in need if n not in params]
                if missing:
                    raise lc.LoraError(f"{layer}: adapter is {atype} but missing {missing}")
                lc.check_shapes(layer, W0, params, atype)
                merged = lc.merged_weight(W0, params, atype, scaling)
                delta = strength * (merged - W0)
                if rec["off"] in accum:
                    accum[rec["off"]][2] += delta
                else:
                    accum[rec["off"]] = [rec, W0, delta]
                matched += 1
        buf.close()
    if not matched:
        raise lc.LoraError("no adapter layers matched this DiT (wrong base model?)")
    log(f"lora: mapped {len(accum)} weight tensor(s) across {matched} adapter layer(s)")

    # Clone, then patch each buffer once (W0 + summed delta).
    tmp = out.with_suffix(".tflite.tmp")
    if tmp.exists():
        tmp.unlink()
    _clone(base_path, str(tmp))
    n_bytes = 0
    with open(tmp, "r+b") as fw:
        for off, (rec, W0, delta) in accum.items():
            Wp = (W0 + delta).astype(rec["np_dtype"])
            b = Wp.tobytes()
            assert len(b) == rec["size"], (rec["name"], len(b), rec["size"])
            fw.seek(off)
            fw.write(b)
            n_bytes += rec["size"]
        fw.flush()
        os.fsync(fw.fileno())
    os.replace(tmp, out)
    log(f"lora: patched {n_bytes / 1e6:.0f} MB → {out.name}")
    return out


def _shape_from_leaf(mapper: _Mapper, layer: str, params: dict):
    """For -xs adapters the fan_in isn't in the params (only M_xs r×r), so map by
    a shape probe: try each FC whose descriptor matches and return its shape.
    Resolves by re-using the mapper's block/singleton/seconds index with a
    wildcard shape, then reads the found record's shape."""
    # Cheap approach: the -xs magnitude (rows/cols) or M_xs can't give fan_in, so
    # locate the record ignoring shape via a temporary relaxation.
    stripped = layer[len("model."):] if layer.startswith("model.") else layer
    if layer == lc.COND_SECONDS_LAYER and mapper._seconds:
        return mapper._seconds[0]["shape"]
    mm = re.match(r"transformer\.layers\.(\d+)\.(.+)$", stripped)
    if mm:
        block, suffix = int(mm.group(1)), mm.group(2)
        spec = _BLOCK_CKPT.get(suffix)
        if spec:
            module, leaf, glu = spec
            if module == "local_embed":
                grp = mapper._local[leaf]
                if block < len(grp):
                    return grp[block]["shape"]
            else:
                rec = mapper._block.get((block, module, leaf, glu))
                if rec:
                    return rec["shape"]
    sub = _SINGLETON_CKPT.get(stripped)
    if sub:
        for r in mapper._fcs:
            if sub in r["name"]:
                return r["shape"]
    raise lc.LoraError(f"{layer}: cannot locate a base weight for this -xs layer")
