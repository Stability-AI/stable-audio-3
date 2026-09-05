"""Merge fixed-size rung tflites into ONE .tflite with N signatures + SHARED weight buffers (dedup by
md5 content). argv: PREFIX SUFFIX OUT [RUNGS_csv].  Files read from / OUT written under $SA3_BUILD_WORK.
  PREFIX = e.g. 'same-l_enc_windowed_'   SUFFIX = '' (fp32) or '_w8a8'   OUT = e.g. 'same-l/enc_fp32.tflite'
  RUNGS_csv (optional) = explicit ladder, else auto-discovered from the files present."""
import sys, hashlib, os, re, glob
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

_, base, _ = T.load_modelT(str(WORK / f"{PREFIX}{RUNGS[0]}{SUFFIX}.tflite"))


def bhash(mt, bi):
    d = mt.buffers[bi].data
    return hashlib.md5(bytes(bytearray(d))).hexdigest() if d is not None and len(d) else f"empty{bi}"


buf_by_hash = {}
for bi in range(len(base.buffers)):
    buf_by_hash.setdefault(bhash(base, bi), bi)
opc_by_key = {(oc.builtinCode, oc.deprecatedBuiltinCode): i for i, oc in enumerate(base.operatorCodes)}


def opc_index(oc):
    key = (oc.builtinCode, oc.deprecatedBuiltinCode)
    if key not in opc_by_key:
        base.operatorCodes.append(oc); opc_by_key[key] = len(base.operatorCodes) - 1
    return opc_by_key[key]


def add_subgraph(add_path):
    _, add, _ = T.load_modelT(add_path); sg = add.subgraphs[0]
    bmap = {}
    for bi in range(len(add.buffers)):
        h = bhash(add, bi)
        if h in buf_by_hash and not h.startswith("empty"):
            bmap[bi] = buf_by_hash[h]
        else:
            base.buffers.append(add.buffers[bi]); bmap[bi] = len(base.buffers) - 1
            if not h.startswith("empty"):
                buf_by_hash[h] = bmap[bi]
    for t in sg.tensors:
        t.buffer = bmap[t.buffer]
    ocmap = {i: opc_index(oc) for i, oc in enumerate(add.operatorCodes)}
    for op in sg.operators:
        op.opcodeIndex = ocmap[op.opcodeIndex]
    base.subgraphs.append(sg)
    return len(base.subgraphs) - 1, sg


def mk_sig(key, sg, sidx):
    sd = schema.SignatureDefT(); sd.signatureKey = key.encode()
    sd.subgraphIndex = sidx
    def tm(ti):
        m = schema.TensorMapT(); m.name = base.subgraphs[sidx].tensors[ti].name; m.tensorIndex = ti; return m
    sd.inputs = [tm(sg.inputs[0])]; sd.outputs = [tm(sg.outputs[0])]
    return sd


sig_defs = [mk_sig(f"s{RUNGS[0]}", base.subgraphs[0], 0)]
for r in RUNGS[1:]:
    idx, sg = add_subgraph(str(WORK / f"{PREFIX}{r}{SUFFIX}.tflite"))
    sig_defs.append(mk_sig(f"s{r}", sg, idx))
base.signatureDefs = sig_defs

out = WORK / OUT
out.parent.mkdir(parents=True, exist_ok=True)
bld = flatbuffers.Builder(1 << 20); bld.Finish(base.Pack(bld), file_identifier=b"TFL3")
out.write_bytes(bytes(bld.Output()))
print(f"wrote {out} ({os.path.getsize(out)/1e6:.0f} MB) — {len(base.subgraphs)} subgraphs, "
      f"sigs={[s.signatureKey.decode() for s in sig_defs]}")
