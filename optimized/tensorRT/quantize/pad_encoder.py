#!/usr/bin/env python3
"""Add a dynamic silence-pad node to an encoder onnx so it accepts ANY audio length: pads the sample
axis up to the next multiple of 4096 with zeros (matches what eager does internally). argv: <in> <out>."""
import sys, os
import onnx
from onnx import helper, TensorProto
IN, OUT = sys.argv[1], sys.argv[2]
m = onnx.load(IN, load_external_data=True); g = m.graph
audio = g.input[0].name; adt = g.input[0].type.tensor_type.elem_type
opset = {o.domain: o.version for o in m.opset_import}.get("", 0)
print(f"input '{audio}' dtype={adt} opset={opset}", flush=True)
g.initializer.extend([
    helper.make_tensor("_pc4096", TensorProto.INT64, [1], [4096]),
    helper.make_tensor("_pzeros5", TensorProto.INT64, [5], [0, 0, 0, 0, 0]),
    helper.make_tensor("_pidx2", TensorProto.INT64, [1], [2]),
    helper.make_tensor("_pc0", adt, [], [0]),
])
for n in g.node:                                                        # rewire consumers to the padded tensor
    for i, inp in enumerate(n.input):
        if inp == audio: n.input[i] = "_audio_pad"
g.node.extend([                                                         # pad = (4096 - N%4096) % 4096, pad end of axis 2
    helper.make_node("Shape", [audio], ["_pshape"], name="_pad_shape"),
    helper.make_node("Gather", ["_pshape", "_pidx2"], ["_pN"], name="_pad_gather", axis=0),
    helper.make_node("Mod", ["_pN", "_pc4096"], ["_pr"], name="_pad_mod1"),
    helper.make_node("Sub", ["_pc4096", "_pr"], ["_pt"], name="_pad_sub"),
    helper.make_node("Mod", ["_pt", "_pc4096"], ["_ppad"], name="_pad_mod2"),
    helper.make_node("Concat", ["_pzeros5", "_ppad"], ["_ppads"], name="_pad_concat", axis=0),
    helper.make_node("Pad", [audio, "_ppads", "_pc0"], ["_audio_pad"], name="_pad_pad", mode="constant"),
])
avail = {i.name for i in g.initializer} | {i.name for i in g.input} | {""}
order, rem = [], list(g.node)
while rem:
    nx, prog = [], False
    for nd in rem:
        if all(i in avail for i in nd.input): order.append(nd); [avail.add(o) for o in nd.output]; prog = True
        else: nx.append(nd)
    rem = nx
    if not prog: raise RuntimeError("topo stuck")
del g.node[:]; g.node.extend(order)
for f in (OUT, OUT + ".data"):
    if os.path.exists(f): os.remove(f)
onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(OUT) + ".data", size_threshold=1024)
print(f"padded encoder -> {OUT}", flush=True)
