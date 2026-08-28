"""Remove the SAME-S decoder's bottleneck dither so decodes are bit-reproducible.

The shipped graph is:
    signal = Slice_1 * running_std                      (running_std = 0.0657)
    noise  = RandomNormalLike(signal) * running_std * 0.001
    out    = signal + noise
i.e. a -60 dB Gaussian dither injected at the bottleneck. It makes every decode differ from the
last, which is why SAME-S has a "noise floor" (~0.9989 between two decodes of the SAME latent)
while SAME-L is bit-exact -- and why any SAME-S A/B has to be run against a matched call history
or it just measures this dither.

Surgery: rewire consumers of the Add onto the signal tensor directly and delete the noise branch
(RandomNormalLike, both Muls, the Add). Exact -- not a reliance on the builder folding a
zero-valued constant away.
"""
import argparse, shutil, sys
from pathlib import Path
import onnx

def _default_src():
    """Fetch the base decoder from the Hub rather than a pinned cache path.

    This used to hardcode a snapshot hash, which pins the file to one revision and breaks
    the moment anything in the repo changes -- as the dec_dynamic_bf16 -> dec_bf16 rename
    just did.
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download("stabilityai/stable-audio-3-optimized", "onnx/same-s/dec_bf16.onnx")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=None, help="base ONNX (default: fetch from the Hub)")
ap.add_argument("--out", default="onnx/same-s/dec_bf16_det.onnx")
a = ap.parse_args()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

m = onnx.load(a.src, load_external_data=True)
g = m.graph
prod = {o: n for n in g.node for o in n.output}

rnd = [n for n in g.node if n.op_type == "RandomNormalLike"]
assert len(rnd) == 1, f"expected exactly one RandomNormalLike, found {len(rnd)}"
rnd = rnd[0]
signal = rnd.input[0]                      # RandomNormalLike takes the signal only for its SHAPE

# walk the noise branch: RandomNormalLike -> Mul(running_std) -> Mul(0.001) -> Add(signal, noise)
chain, cur = [rnd], rnd.output[0]
while True:
    nxt = [n for n in g.node if cur in n.input]
    assert len(nxt) == 1, f"noise branch forks at {cur}: {[n.name for n in nxt]}"
    n = nxt[0]; chain.append(n)
    if n.op_type == "Add":
        add = n; break
    cur = n.output[0]
assert set(add.input) == {signal, chain[-2].output[0]}, f"unexpected Add inputs {list(add.input)}"
print("noise branch:", " -> ".join(f"{n.op_type}" for n in chain))
print(f"  signal tensor kept: {signal}")
print(f"  Add output rewired: {add.output[0]} -> {signal}")

# rewire every consumer of the Add's output onto the signal, then drop the branch
n_rewired = 0
for n in g.node:
    for i, inp in enumerate(n.input):
        if inp == add.output[0]:
            n.input[i] = signal; n_rewired += 1
for o in g.output:
    assert o.name != add.output[0], "Add output is a graph output; rewire needed there too"
dead = {id(n) for n in chain}
keep = [n for n in g.node if id(n) not in dead]
print(f"  rewired {n_rewired} consumer input(s); removed {len(g.node)-len(keep)} node(s)")
del g.node[:]
g.node.extend(keep)

onnx.checker.check_model(m, full_check=False)
size = sum(t.raw_data.__sizeof__() for t in g.initializer)
if size > 1_900_000_000:
    onnx.save(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
              location=out.name + ".data")
else:
    onnx.save(m, str(out))
print(f"\nwrote {out} ({out.stat().st_size/1e6:.0f} MB)")
assert not [n for n in onnx.load(str(out), load_external_data=False).graph.node
            if "Random" in n.op_type], "a random op survived"
print("verified: no random ops remain")
