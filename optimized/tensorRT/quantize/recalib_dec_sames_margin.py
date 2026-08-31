"""Retighten the SAME-S fp8 DECODER activation scales.

The shipped scales are amax*1.5 over 6 calibration latents, and on real music that leaves
1.19-1.80x of headroom (median 1.50) with nothing clipping. Headroom above 1.0 is wasted fp8
range, so this recalibrates from a WIDER real-music set with a smaller margin.

⚠ Calibrated on tracks DISJOINT from the evaluation set -- calibrating and evaluating on the same
audio would report an overfit number.
"""
import argparse, json, os, random, sys
sys.path.append("/path/to/sa3s/stable-audio-3/optimized/tensorRT/scripts")
sys.path.insert(0, "gradio"); sys.path.insert(0, "lora")
os.environ.setdefault("SA3_SWA_PLUGIN", "aot")
import numpy as np, onnx, onnxruntime as ort, torch, soundfile as sf
import sa3_trt_core as canon; canon._import_heavy()
from onnx import numpy_helper as nh
E4M3 = 448.0; SR, SPL, L = 44100, 4096, 256; N = L * SPL
ap = argparse.ArgumentParser()
ap.add_argument("--fp8", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.1)
ap.add_argument("--ntracks", type=int, default=16)
ap.add_argument("--seed", type=int, default=99)          # != the eval seed (5)
a = ap.parse_args()
def load_excerpt(path, start_s=35.0):
    from math import gcd
    from scipy.signal import resample_poly
    try:
        i = sf.info(path); need = int(N * i.samplerate / SR) + 256
        if i.frames < int(start_s * i.samplerate) + need: return None
        w, sr = sf.read(path, start=int(start_s*i.samplerate), frames=need,
                        dtype="float32", always_2d=True)
    except Exception: return None
    if w.shape[1] == 1: w = np.repeat(w, 2, 1)
    w = w[:, :2]
    if sr != SR:
        g = gcd(SR, sr); w = resample_poly(w, SR//g, sr//g, axis=0).astype(np.float32)
    if w.shape[0] < N: return None
    w = w[:N]
    if not np.isfinite(w).all() or np.abs(w).max() < 1e-3: return None
    return np.ascontiguousarray(w.T)
from concurrent.futures import ThreadPoolExecutor
cat = json.load(open("/path/to/promptlists/asx_comma_audio.json"))
mus = [e for e in cat if "tracktype: sfx" not in e.get("p", "").lower()]
random.Random(a.seed).shuffle(mus)
picked = []
with ThreadPoolExecutor(32) as ex:
    for e, w in zip(mus[:900], ex.map(lambda x: load_excerpt(x["audio"]), mus[:900])):
        if w is not None:
            picked.append(w)
            if len(picked) >= a.ntracks: break
print(f"  calibrating on {len(picked)} real tracks (seed {a.seed}, disjoint from eval)", flush=True)
E = canon.TRTRunner("engines/ship/same-s_enc_2prof_v3.trt", None, False, 0)
lats = [canon.encode_chunked(E, torch.from_numpy(w).unsqueeze(0).cuda(),
                             warmup_passes=0, balance=True).float().cpu().numpy() for w in picked]
E.free(); torch.cuda.empty_cache()
from huggingface_hub import hf_hub_download
base = hf_hub_download("stabilityai/stable-audio-3-optimized", "onnx/same-s/dec_dynamic_bf16.onnx")
m = onnx.load(base, load_external_data=True); g = m.graph
inits = {i.name for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}
targets = {}
for n in g.node:
    if n.op_type == "MatMul":
        w = n.input[1]
        if w in inits or (prod.get(w) and prod[w].op_type == "Transpose" and prod[w].input[0] in inits):
            targets[n.name] = n.input[0]
have = {o.name for o in g.output}
for t in set(targets.values()):
    if t not in have:
        g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
WK = "/tmp/_dec_recal.onnx"
onnx.save(m, WK, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(WK)+".data", size_threshold=1024)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(WK, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
iname = sess.get_inputs()[0].name; ks = list(targets); ts = [targets[k] for k in ks]
amax = {k: 0.0 for k in ks}
for i, z in enumerate(lats):
    for k, o in zip(ks, sess.run(ts, {iname: z.astype(np.float32)})):
        amax[k] = max(amax[k], float(np.abs(o).max()))
del sess
new = {k.strip("/").replace("/", "_") + "_asc": amax[k] * a.margin / E4M3 for k in ks}
mq = onnx.load(a.fp8, load_external_data=True)
old_v, new_v, hits = [], [], 0
for init in mq.graph.initializer:
    if init.name in new:
        arr = nh.to_array(init); o = float(np.asarray(arr).ravel()[0])
        old_v.append(o); new_v.append(new[init.name]); hits += 1
        init.CopyFrom(nh.from_array(np.full(arr.shape, new[init.name], dtype=arr.dtype), init.name))
r = np.array(new_v) / np.array(old_v)
print(f"  {hits} activation scales rewritten (margin {a.margin})")
print(f"  new/old ratio: min {r.min():.2f}x  median {np.median(r):.2f}x  max {r.max():.2f}x")
onnx.save(mq, a.out, save_as_external_data=False)
print(f"  wrote {a.out}")
