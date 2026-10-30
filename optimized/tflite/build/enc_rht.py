"""RHT on block-0 glu_proj of an exported SAME-S encoder.

A randomized-Hadamard rotation (reused from the DiT export) is applied to the input of the first block's
GLU feed-forward projection. That layer's input is a DyT-norm output whose per-token range stays large even
in quiet passages, with a few outlier channels inflating the dynamic-int8 activation scale -- so the w8a8
encoder carries a broadband quantization floor there. Spreading those channels with the rotation lets int8
keep its resolution: ~11 dB lower quiet-section floor, and cos-to-fp32 improves, at the same int8 size and
speed (the only op added is one BATCH_MATMUL).

The rotation R is folded into the weight (W@R), so the result is numerically identical to the un-rotated
model (latent cos 1.0); the online x@R is emitted as a fp32 op named HadRot_rot, which quant_one's existing
`.*HadRot_rot.*` carve-out keeps in fp32. Applied between export and quantization, so the fp32 encoder is
untouched. Works on a single fixed rung or a merged multi-rung file (every subgraph is rotated; the shared
weight buffer is folded once).

Usage: enc_rht.py <src.tflite> <dst.tflite>
"""
import os
import sys
from pathlib import Path

import flatbuffers
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "export"))
from ai_edge_litert import schema_py_generated as S
from export_dit import _hadamard_parts

GLU_IN = 768                 # SAME-S model dim == block-0 glu_proj input width
SEED = 1234 + GLU_IN         # same seed convention as the DiT's HadRotLinear(seed=1234+in_dim)


def _shape(t):
    return tuple(int(x) for x in (t.shape if t.shape is not None else []))


def _name(t):
    n = t.name
    return n.decode() if isinstance(n, (bytes, bytearray)) else (n or "")


def apply_rht(src, dst):
    R = _hadamard_parts(GLU_IN, SEED)[0].double().numpy()            # dense orthonormal rotation
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(bytearray(open(src, "rb").read()), 0))
    BOP, F32 = S.BuiltinOperator, S.TensorType.FLOAT32
    name_of = {v: k for k, v in BOP.__dict__.items() if isinstance(v, int)}

    def add_buffer(arr):
        b = S.BufferT()
        b.data = np.asarray(arr).tobytes()
        m.buffers.append(b)
        return len(m.buffers) - 1

    def opcode(code):
        for i, oc in enumerate(m.operatorCodes):
            if oc.builtinCode == code:
                return i
        oc = S.OperatorCodeT()
        oc.builtinCode = code
        oc.deprecatedBuiltinCode = min(code, 127)
        m.operatorCodes.append(oc)
        return len(m.operatorCodes) - 1

    r_buffer = add_buffer(R.astype(np.float32))
    folded = set()
    rotated = 0
    for si, sg in enumerate(m.subgraphs):
        gp = next((o for o in sg.operators
                   if name_of.get(m.operatorCodes[o.opcodeIndex].builtinCode) == "FULLY_CONNECTED"
                   and "TransformerBlock_0/" in _name(sg.tensors[o.outputs[0]])
                   and "glu_proj" in _name(sg.tensors[o.outputs[0]])), None)
        if gp is None:
            continue
        at = sg.operators.index(gp)
        act, wi = gp.inputs[0], gp.inputs[1]
        wt = sg.tensors[wi]
        if wt.buffer not in folded:                                 # fold W@R once (weight buffer is shared across rungs)
            W = np.frombuffer(bytes(bytearray(m.buffers[wt.buffer].data)), np.float32).reshape(_shape(wt))
            m.buffers[wt.buffer].data = (W.astype(np.float64) @ R).astype(np.float32).tobytes()
            folded.add(wt.buffer)
        rt = S.TensorT()
        rt.shape, rt.type, rt.buffer = [GLU_IN, GLU_IN], F32, r_buffer
        rt.name = f"HadRot_rot/R__u{si}".encode()
        sg.tensors.append(rt)
        rot_in = len(sg.tensors) - 1
        xo = S.TensorT()
        xo.shape, xo.type, xo.buffer = list(_shape(sg.tensors[act])), F32, 0
        xo.name = f"HadRot_rot/xrot__u{si}".encode()
        sg.tensors.append(xo)
        rot_out = len(sg.tensors) - 1
        op = S.OperatorT()
        op.opcodeIndex = opcode(BOP.BATCH_MATMUL)
        op.inputs, op.outputs = [act, rot_in], [rot_out]
        opts = S.BatchMatMulOptionsT()
        opts.adjX = opts.adjY = False
        op.builtinOptions, op.builtinOptionsType = opts, S.BuiltinOptions.BatchMatMulOptions
        gp.inputs[0] = rot_out
        sg.operators[at:at] = [op]
        rotated += 1

    b = flatbuffers.Builder(1 << 28)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    open(dst, "wb").write(bytes(b.Output()))
    print(f"enc_rht: rotated block-0 glu_proj in {rotated} subgraph(s) -> "
          f"{os.path.basename(dst)} ({os.path.getsize(dst) / 1e6:.0f} MB)")


if __name__ == "__main__":
    apply_rht(sys.argv[1], sys.argv[2])
