"""Merge fixed-size rung tflites into ONE .tflite with N signatures + SHARED weight buffers (dedup by
content hash). argv: PREFIX SUFFIX OUT [RUNGS_csv].  Files read from / OUT written under $SA3_BUILD_WORK.
  PREFIX = e.g. 'same-l_enc_windowed_'  SUFFIX = '' (fp32) or '_w8a8'  OUT = e.g. 'same-l/enc_fp32.tflite'
  RUNGS_csv (optional) = explicit ladder, else auto-discovered from the files present.

Handles BOTH in-band models (SAME rungs, the w8a8 DiT — all < 2 GB) and OUT-OF-BAND models (the fp32 DiT,
~5.8 GB: ai_edge_torch stores weights past the 2 GB flatbuffer limit as Buffer.offset/size into a trailing
data region). OOB buffers come back from load_modelT with data=None, so we hash their ACTUAL bytes (read
from the raw file at the offset) and, when any OOB buffer is present, serialize with the OOB-aware
write_model (plain flatbuffers.Builder.Pack would silently drop the >2 GB data → a 16 MB stub)."""
import sys, os, re, glob, copy, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # build/ (tfl_surgery + build_paths)
from build_paths import WORK
import tfl_surgery as T
from ai_edge_litert import schema_py_generated as schema
import flatbuffers

PREFIX = sys.argv[1]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = sys.argv[3] if len(sys.argv) > 3 else "merged.tflite"
if len(sys.argv) > 4 and sys.argv[4].strip():
    RUNGS = sorted({int(x) for x in sys.argv[4].split(",")}, reverse=True)
else:
    pat = re.compile(re.escape(PREFIX) + r"(\d+)" + re.escape(SUFFIX) + r"\.tflite$")
    RUNGS = sorted({int(m.group(1)) for f in glob.glob(str(WORK / f"{PREFIX}*{SUFFIX}.tflite"))
                    for m in [pat.search(f)] if m}, reverse=True)
print(f"merging rungs {RUNGS}")


def bbytes(mt, raw, orig, bi):
    """The actual bytes of buffer bi: in-band Buffer.data, else the out-of-band data region at its original
    offset (orig[bi]=(offset,size) snapshotted by load_modelT). None = empty (an activation buffer)."""
    b = mt.buffers[bi]
    if b.data is not None and len(b.data) > 0:
        return bytes(bytearray(b.data))
    o, s = orig[bi] if bi < len(orig) else (0, 0)
    return bytes(raw[o:o + s]) if o else None


def bhash(mt, raw, orig, bi):
    d = bbytes(mt, raw, orig, bi)
    return hashlib.sha1(d).hexdigest() if d is not None else None


raw0, base, orig0 = T.load_modelT(str(WORK / f"{PREFIX}{RUNGS[0]}{SUFFIX}.tflite"))
# keep the source serving_default signature (canonical alias names + input ORDER) so each merged s<R>
# signature is a faithful rename of it; falls back to building from sg.inputs if a source has none.
base_sig0 = base.signatureDefs[0] if getattr(base, "signatureDefs", None) else None
oob_present = any(o for (o, _) in orig0 if o)           # does the base store weights out-of-band?

buf_by_hash = {}
for bi in range(len(base.buffers)):
    h = bhash(base, raw0, orig0, bi)
    if h and h not in buf_by_hash:
        buf_by_hash[h] = bi
opc_by_key = {(oc.builtinCode, oc.deprecatedBuiltinCode): i for i, oc in enumerate(base.operatorCodes)}


def opc_index(oc):
    key = (oc.builtinCode, oc.deprecatedBuiltinCode)
    if key not in opc_by_key:
        base.operatorCodes.append(oc); opc_by_key[key] = len(base.operatorCodes) - 1
    return opc_by_key[key]


def add_subgraph(add_path):
    raw_i, add, orig_i = T.load_modelT(add_path); sg = add.subgraphs[0]
    bmap = {}
    for bi in range(len(add.buffers)):
        h = bhash(add, raw_i, orig_i, bi)
        if h is None:
            bmap[bi] = 0                                 # empty (activation) -> the shared empty buffer 0
        elif h in buf_by_hash:
            bmap[bi] = buf_by_hash[h]                    # identical weight/const -> dedup to the base's copy
        else:
            bmap[bi] = T.add_buffer(base, bbytes(add, raw_i, orig_i, bi))   # length-specific -> new in-band
            buf_by_hash[h] = bmap[bi]
    for t in sg.tensors:
        t.buffer = bmap[t.buffer]
    ocmap = {i: opc_index(oc) for i, oc in enumerate(add.operatorCodes)}
    for op in sg.operators:
        op.opcodeIndex = ocmap[op.opcodeIndex]
    base.subgraphs.append(sg)
    return len(base.subgraphs) - 1, sg, (add.signatureDefs[0] if getattr(add, "signatureDefs", None) else None)


def mk_sig(key, sg, sidx, src_sig):
    """Clone the source serving_default signature (canonical aliases + input order; tensorIndex values stay
    valid because the subgraph is copied whole). Fallback (no source sig): map ALL subgraph IO positionally
    — correct for 1-IO SAME and for the 7-input DiT (the original bug mapped only inputs[0])."""
    sd = schema.SignatureDefT(); sd.signatureKey = key.encode(); sd.subgraphIndex = sidx
    if src_sig is not None:
        sd.inputs = [copy.deepcopy(tm) for tm in src_sig.inputs]
        sd.outputs = [copy.deepcopy(tm) for tm in src_sig.outputs]
    else:
        def tm(ti):
            m = schema.TensorMapT(); m.name = base.subgraphs[sidx].tensors[ti].name; m.tensorIndex = ti; return m
        sd.inputs = [tm(i) for i in sg.inputs]; sd.outputs = [tm(o) for o in sg.outputs]
    return sd


sig_defs = [mk_sig(f"s{RUNGS[0]}", base.subgraphs[0], 0, base_sig0)]
for r in RUNGS[1:]:
    idx, sg, add_sig = add_subgraph(str(WORK / f"{PREFIX}{r}{SUFFIX}.tflite"))
    sig_defs.append(mk_sig(f"s{r}", sg, idx, add_sig))
# Fold in extra NAMED subgraphs (SA3_MERGE_EXTRA="key:path[,key2:path2]"); path relative to WORK unless
# absolute. Used to bundle the DiT's global-cond preamble as a `gcond` signature in the same .tflite (one
# shipped file). Its buffers dedup like any other — they're unique (fp32, not in the int8/fp32 rungs) so they
# land once; a single non-shared subgraph is cache-safe (the cache bug needs a weight shared across >=2 rungs).
for spec in os.environ.get("SA3_MERGE_EXTRA", "").split(","):
    spec = spec.strip()
    if not spec:
        continue
    key, pth = spec.split(":", 1)
    pth = pth if os.path.isabs(pth) else str(WORK / pth)
    idx, sg, add_sig = add_subgraph(pth)
    sig_defs.append(mk_sig(key, sg, idx, add_sig))
    print(f"  + folded extra subgraph '{key}' <- {os.path.basename(pth)}")
base.signatureDefs = sig_defs

out = WORK / OUT
out.parent.mkdir(parents=True, exist_ok=True)
if oob_present:                                          # >2 GB weights live in the OOB data region
    T.write_model(str(out), raw0, base, orig0)
else:                                                    # fully in-band (SAME, w8a8 DiT) — Pack is fine
    bld = flatbuffers.Builder(1 << 20); bld.Finish(base.Pack(bld), file_identifier=b"TFL3")
    out.write_bytes(bytes(bld.Output()))
print(f"wrote {out} ({os.path.getsize(out)/1e6:.0f} MB) — {len(base.subgraphs)} subgraphs, "
      f"oob={oob_present}, sigs={[s.signatureKey.decode() for s in sig_defs]}")
