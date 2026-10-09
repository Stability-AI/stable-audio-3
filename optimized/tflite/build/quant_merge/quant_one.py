import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # build/ — for tfl_surgery
import numpy as np, flatbuffers
from ai_edge_quantizer import quantizer, recipe, qtyping
from ai_edge_quantizer.algorithm_manager import AlgorithmName
import tfl_surgery as T
from ai_edge_litert import schema_py_generated as schema

src, dst = sys.argv[1], sys.argv[2]
q = quantizer.Quantizer(src)
q.load_quantization_recipe(recipe.dynamic_wi8_afp32())
# Keep the baked output limiter in fp32: int8 on its 9-tap Hann filter makes the smoothed gain no longer
# sum to 1, breaking the brickwall ceiling (w8a8 peaked 1.015 > 1.0). Tiny, so fp32 there costs ~nothing.
# No-op on encoders / the DiT (no OutputLimiter ops).
q.update_quantization_recipe(regex=".*OutputLimiter.*",
                             operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                             algorithm_key=AlgorithmName.NO_QUANTIZE)
q.quantize().export_model(dst)

# Post-quant fix (robust, in-band): dynamic int8 mis-forms any FULLY_CONNECTED whose ACTIVATION input
# (input[0]) is a CONSTANT — it int8's that constant, so the runtime picks the fully-integer FC kernel
# (which asserts an int8 output) while the output stays fp32 → "output->type == int8/.." prepare failure.
# Dequantize input[0] back to fp32 so the op is a standard hybrid FC (fp32 act × int8 weight → fp32 out).
# Detected by signature (int8 input[0] + non-int8 output), so it's robust across rungs; the medium DiT has
# exactly one such op (a const@const projection the converter didn't fold), the SAME models have none.
_, m, _ = T.load_modelT(dst)
sg = m.subgraphs[0]
I8 = schema.TensorType.INT8
fixed = 0
for op in sg.operators:
    if m.operatorCodes[op.opcodeIndex].builtinCode != schema.BuiltinOperator.FULLY_CONNECTED:
        continue
    t0 = sg.tensors[op.inputs[0]]
    if t0.type != I8 or sg.tensors[op.outputs[0]].type == I8:
        continue
    buf = m.buffers[t0.buffer].data
    if buf is None or len(buf) == 0:                     # only an in-place-dequantizable constant
        continue
    qp = t0.quantization
    scale = np.asarray(qp.scale, np.float32); zp = np.asarray(qp.zeroPoint, np.float64)
    arr = np.frombuffer(bytes(bytearray(buf)), np.int8).astype(np.float32).reshape([int(s) for s in t0.shape])
    if scale.size == 1:
        deq = (arr - float(zp[0])) * float(scale[0])
    else:                                                # per-axis
        ax = int(qp.quantizedDimension); bc = [1] * arr.ndim; bc[ax] = scale.size
        deq = (arr - zp.reshape(bc)) * scale.reshape(bc)
    m.buffers[t0.buffer].data = deq.astype(np.float32).tobytes()
    t0.type = schema.TensorType.FLOAT32
    t0.quantization = None
    fixed += 1
if fixed:                                                # re-serialize only if we changed something (w8a8 < 2 GB → in-band)
    b = flatbuffers.Builder(1 << 20); b.Finish(m.Pack(b), file_identifier=b"TFL3")
    open(dst, "wb").write(bytes(b.Output()))
print(f"{os.path.basename(dst)} {os.path.getsize(dst)/1e6:.0f}MB (+{fixed} const-input FC fixed)")
