"""FlatBuffer surgery toolkit for LARGE (>2GB) TFLite models that use the
out-of-band buffer extension (Buffer.offset/size point past the flatbuffer header).

Key constraint: the file exceeds the 2GB FlatBuffer limit, so we CANNOT pack
everything in-band. We must preserve the giant data region verbatim and only
rewrite the (small) flatbuffer header that precedes it.

Layout of these files:
    [ flatbuffer header (tables + small in-band buffers) ][ data region (big tensors) ]
Each big Buffer has offset (absolute into file) + size; small buffers use Data.

Approach (write_model):
  1. Parse to ModelT (object API). OOB buffers come back with data=None, offset/size set.
  2. data_region_start = min(nonzero offsets in ORIGINAL file).
  3. Caller mutates ModelT freely (add tensors/ops/inputs, edit small Data buffers).
     OOB buffers keep their ORIGINAL absolute offset in a parallel dict (we snapshot
     before mutation). New big buffers can be appended (rare; not needed here).
  4. Pack header once with placeholder offsets to measure header length H.
  5. Align data start to A=16. new_data_start = round_up(H, A).
  6. For each OOB buffer: new_offset = new_data_start + (orig_offset - data_region_start).
  7. Re-pack header with final offsets (size is stable -> H unchanged; assert it).
  8. Write header + pad + original_data_region bytes.

This keeps every big tensor's BYTES identical and just relocates the header.
"""
from __future__ import annotations
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema


def load_modelT(path):
    raw = bytearray(open(path, "rb").read())
    model = schema.Model.GetRootAs(raw, 0)
    # snapshot original OOB offsets/sizes BEFORE building ModelT (ModelT also carries them,
    # but we keep an explicit list for clarity)
    n_buf = model.BuffersLength()
    orig = []
    for bi in range(n_buf):
        b = model.Buffers(bi)
        orig.append((b.Offset(), b.Size()))
    mt = schema.ModelT.InitFromObj(model)
    return raw, mt, orig


def _data_region_start(orig):
    offs = [o for (o, s) in orig if o]
    return min(offs) if offs else None


def write_model(out_path, raw, mt, orig, align=16):
    """Serialize mt preserving OOB data from `raw` (the original file bytes)."""
    drs = _data_region_start(orig)
    n_buf = len(mt.buffers)
    # which buffers are OOB: had an offset in the ORIGINAL file AND have NOT been replaced
    # with in-band data by the caller (i.e. mt.buffers[i].data is still empty). A caller that
    # edited a const sets .data -> that buffer must serialize in-band, not be re-pointed to OOB.
    def _has_inband(i):
        d = mt.buffers[i].data
        return d is not None and len(d) > 0
    oob_idx = [i for i in range(n_buf)
               if (i < len(orig) and orig[i][0] and not _has_inband(i))]

    def pack_header():
        b = flatbuffers.Builder(1 << 20)
        b.Finish(mt.Pack(b), file_identifier=b"TFL3")
        return b.Output()

    # Normalize OOB buffers FIRST so the two packs differ only in fixed-width offset values
    # (a uint64 field -> identical encoded length regardless of value). data=None, size=orig.
    for i in oob_idx:
        mt.buffers[i].data = None
        mt.buffers[i].size = orig[i][1]
        mt.buffers[i].offset = orig[i][0]  # placeholder (original abs); fixed-width

    # Iterate to a fixpoint: pick new_data_start, set offsets, re-pack; if header length
    # changed (it shouldn't for fixed-width offsets, but be safe), repeat until stable.
    new_data_start = None
    hdr2 = None
    for _ in range(6):
        hdr = pack_header()
        H = len(hdr)
        cand = ((H + align - 1) // align) * align
        if new_data_start == cand and hdr2 is not None:
            break
        new_data_start = cand
        for i in oob_idx:
            orig_off, sz = orig[i]
            mt.buffers[i].offset = new_data_start + (orig_off - drs)
            mt.buffers[i].size = sz
            mt.buffers[i].data = None
        hdr2 = pack_header()
    # Final: ensure offsets are consistent with the header length we will actually write.
    H = len(hdr2)
    new_data_start = ((H + align - 1) // align) * align
    for i in oob_idx:
        orig_off, sz = orig[i]
        mt.buffers[i].offset = new_data_start + (orig_off - drs)
        mt.buffers[i].size = sz
        mt.buffers[i].data = None
    hdr2 = pack_header()
    assert len(hdr2) <= new_data_start, (len(hdr2), new_data_start)

    data_region = raw[drs:]  # everything after original data start (verbatim)
    pad = new_data_start - H
    with open(out_path, "wb") as f:
        f.write(hdr2)
        if pad:
            f.write(b"\x00" * pad)
        f.write(data_region)
    return out_path, new_data_start, len(data_region)


# ---- small helpers for graph editing on ModelT ----

def add_buffer(mt, data_bytes=None):
    b = schema.BufferT()
    if data_bytes is not None:
        b.data = list(data_bytes) if not isinstance(data_bytes, (bytes, bytearray)) else data_bytes
    mt.buffers.append(b)
    return len(mt.buffers) - 1


def int32_const_buffer(mt, values):
    arr = np.asarray(values, dtype=np.int32)
    return add_buffer(mt, arr.tobytes())


def add_tensor(mt, sg, name, shape, ttype, buffer_idx=0, shape_signature=None):
    t = schema.TensorT()
    t.name = name
    t.shape = list(shape)
    t.type = ttype
    t.buffer = buffer_idx
    if shape_signature is not None:
        t.shapeSignature = list(shape_signature)
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def opcode_index(mt, builtin_code):
    for i, oc in enumerate(mt.operatorCodes):
        if oc.builtinCode == builtin_code:
            return i
    oc = schema.OperatorCodeT()
    oc.builtinCode = builtin_code
    oc.deprecatedBuiltinCode = builtin_code if builtin_code < 127 else 127
    oc.version = 1
    mt.operatorCodes.append(oc)
    return len(mt.operatorCodes) - 1


def make_op(mt, builtin_code, inputs, outputs, builtin_options=None, builtin_options_type=0):
    op = schema.OperatorT()
    op.opcodeIndex = opcode_index(mt, builtin_code)
    op.inputs = list(inputs)
    op.outputs = list(outputs)
    if builtin_options is not None:
        op.builtinOptions = builtin_options
        op.builtinOptionsType = builtin_options_type
    return op
