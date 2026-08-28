"""Promote the limiter ceiling from a baked constant to a runtime graph INPUT.

The grafted limiter computes gain = min(ceiling / env, 1) via
    Reciprocal(env) -> Mul(1/env, ceiling) -> Clip(max=1.0)
with `ceiling` a scalar Constant (0.977 = -0.2021 dBFS). Making it an input gives a runtime knob
with no branching, no second engine and no IIfConditional (which is not graph-capturable):

    ceiling = 0.977   the shipped setting
    ceiling = 1.0     a 0 dBFS ceiling
    ceiling = 0.891   -1 dBFS
    ceiling >> 1      gain is identically 1.0 -> limiter bypassed, and the graph's +-1.0 clamp
                      catches the overshoot, reproducing the OLD hard-clip behaviour

The clamp stays in place, so the int32/int16 delivery tail is unchanged and the engine still emits
already-bounded PCM. Only the gain curve becomes tunable.

Cost: the ceiling can no longer be constant-folded into the reciprocal. That is one scalar Mul.
In the mega-graph it is a persistent buffer written once per render -- the same mechanism as
`seconds_total`, and free under graph replay.
"""
import argparse, sys
from pathlib import Path
import onnx
from onnx import helper, TensorProto

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="limiter_ceiling")
a = ap.parse_args()
m = onnx.load(a.src, load_external_data=True)
g = m.graph
c = [n for n in g.node if n.name == "lim//Constant_4"]
assert len(c) == 1, f"expected exactly one ceiling constant, found {len(c)}"
c = c[0]
old = c.output[0]
users = [n for n in g.node if old in n.input]
assert len(users) == 1 and users[0].op_type == "Mul", \
    f"ceiling should feed exactly one Mul, got {[(u.op_type, u.name) for u in users]}"
# scalar float input, matching the constant's rank so the Mul broadcast is unchanged
g.input.append(helper.make_tensor_value_info(a.name, TensorProto.FLOAT, []))
for i, t in enumerate(users[0].input):
    if t == old:
        users[0].input[i] = a.name
g.node.remove(c)
onnx.checker.check_model(m, full_check=False)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
big = sum(t.raw_data.__sizeof__() for t in g.initializer) > 1_900_000_000
if big:
    onnx.save(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
              location=out.name + ".data")
else:
    onnx.save(m, str(out))
chk = onnx.load(str(out), load_external_data=False)
names = [i.name for i in chk.graph.input]
print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB)")
print(f"  graph inputs now: {names}")
assert a.name in names and not [n for n in chk.graph.node if n.name == "lim//Constant_4"]
print(f"  ceiling is a runtime input; the +-1.0 clamp and int32 tail are untouched")
