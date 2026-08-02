#!/usr/bin/env python3
"""Quality + speed gate for the torch-free C++ AMX-BF16 T5Gemma encoder.

(1) 3 groundtruth prompts: per-prompt cosine + PSNR vs the fp32 groundtruth
    last_hidden_state AND vs the TFLite fp16 oracle, measured over REAL token
    positions (mask==1).
(2) decode time (best-of-N ms/prompt) + x-realtime for a 29.72 s conditioning clip.
(3) >=3 extra prompts (tokenized via the npz SentencePiece) cross-checked vs TFLite.

GATE: per-token cosine >= 0.999 and PSNR >= 45 dB vs groundtruth on all 3 prompts.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np

DIR = "/weka2/cj/clod/t5gemma_cpu_amx"
sys.path.insert(0, DIR)
from t5gemma_cpu_backend import T5GemmaCPU

REF = "/weka2/cj/clod/q4/sa3-w4-cluster/groundtruth/t5gemma/ref_b3_seq256.npz"
TFL = "/weka2/cj/clod/q4/sa3-w4-cluster/models/tflite/t5gemma_seq256_fp16.tflite"
NPZ = "/weka2/cj/clod/t5gemma_cpu_amx/hf/MLX/t5gemma_f16.npz"
THREADS = int(os.environ.get("OMP_NUM_THREADS", "16"))


def psnr(ref, test, m):
    ref = ref[m].astype(np.float64).ravel(); test = test[m].astype(np.float64).ravel()
    mse = np.mean((ref - test) ** 2)
    return float("inf") if mse <= 0 else 20.0 * np.log10(np.max(np.abs(ref)) / np.sqrt(mse))


def cos_stats(ref, test, m):
    r = ref[m].astype(np.float64); t = test[m].astype(np.float64)
    num = (r * t).sum(-1); den = np.linalg.norm(r, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-30
    c = num / den
    return float(c.min()), float(c.mean())


class TFLiteOracle:
    def __init__(self, path, threads=8):
        from ai_edge_litert import interpreter as tfl
        self.it = tfl.Interpreter(model_path=path, num_threads=threads); self.it.allocate_tensors()
        det = sorted(self.it.get_input_details(), key=lambda d: d["name"])
        self.i_ids, self.i_mask = det[0]["index"], det[1]["index"]
        self.o = self.it.get_output_details()[0]["index"]

    def __call__(self, ids, mask):
        self.it.set_tensor(self.i_ids, ids.reshape(1, 256).astype(np.int32))
        self.it.set_tensor(self.i_mask, mask.reshape(1, 256).astype(np.int32))
        self.it.invoke()
        return self.it.get_tensor(self.o).copy()[0]


def main():
    m = T5GemmaCPU(threads=THREADS)
    z = np.load(REF, allow_pickle=True)
    ids_b, mask_b, gt = z["input_ids"], z["attention_mask"], z["last_hidden_state"]
    prompts = [str(p) for p in z["prompts"]]

    try:
        tfl = TFLiteOracle(TFL, threads=8)
    except Exception as e:
        tfl = None; print(f"(TFLite oracle unavailable: {e})")

    print(f"\n=== (1) 3 groundtruth prompts  (gate: cos>=0.999, PSNR>=45 dB vs GT) ===")
    print(f"{'p':>2} {'ntok':>4} | {'PSNR-GT':>8} {'mincos-GT':>10} {'meancos-GT':>10} | "
          f"{'PSNR-TFL':>8} {'mincos-TFL':>10} | prompt")
    ok = True
    for b in range(ids_b.shape[0]):
        mrow = mask_b[b].astype(bool)
        out = m(ids_b[b], mask_b[b])[0]
        dg = psnr(gt[b], out, mrow); cg_min, cg_mean = cos_stats(gt[b], out, mrow)
        if tfl is not None:
            to = tfl(ids_b[b], mask_b[b]); dt = psnr(to, out, mrow); ct_min, _ = cos_stats(to, out, mrow)
            s_tfl = f"{dt:8.2f} {ct_min:10.6f}"
        else:
            s_tfl = f"{'-':>8} {'-':>10}"
        flag = "" if (cg_min >= 0.999 and dg >= 45.0) else "  <-- FAIL"
        ok &= (cg_min >= 0.999 and dg >= 45.0)
        print(f"{b:>2} {int(mask_b[b].sum()):>4} | {dg:8.2f} {cg_min:10.6f} {cg_mean:10.6f} | "
              f"{s_tfl} | {prompts[b][:40]}{flag}")
    print(f"GATE: {'PASS' if ok else 'FAIL'}")

    # ---- (2) timing ----
    print(f"\n=== (2) decode time (threads={THREADS}) ===")
    N = 50
    for _ in range(5):  # warmup
        m(ids_b[0], mask_b[0])
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); m(ids_b[0], mask_b[0]); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    best, med = ts.min(), np.median(ts)
    print(f"  best {best:.2f} ms | median {med:.2f} ms | mean {ts.mean():.2f} ms  (per 256-token prompt)")
    print(f"  x-realtime vs 29.72 s conditioning clip: {29720.0/best:.0f}x (best) / {29720.0/med:.0f}x (median)")
    if tfl is not None:
        for _ in range(3): tfl(ids_b[0], mask_b[0])
        tt = []
        for _ in range(20):
            t0 = time.perf_counter(); tfl(ids_b[0], mask_b[0]); tt.append((time.perf_counter() - t0) * 1e3)
        print(f"  (TFLite fp16 reference, 8 threads: best {min(tt):.2f} ms | median {np.median(tt):.2f} ms)")

    # ---- (3) extra prompts vs TFLite ----
    if tfl is not None:
        print(f"\n=== (3) extra prompts vs TFLite fp16 oracle ===")
        import sentencepiece as spm
        arrs = np.load(NPZ, allow_pickle=True)
        sp = spm.SentencePieceProcessor(); sp.LoadFromSerializedProto(arrs["TOKENIZER_MODEL"].tobytes())
        extra = ["A beautiful piano arpeggio grows into a grand cinematic climax",
                 "lofi house loop", "Amen break 174 BPM",
                 "deep dub techno with a rolling bassline and tape hiss",
                 "solo cello playing a melancholic adagio in a cathedral",
                 "8-bit chiptune boss battle theme, fast and frantic"]
        print(f"{'ntok':>4} | {'PSNR-TFL':>8} {'mincos':>9} {'meancos':>9} | prompt")
        for p in extra:
            toks = sp.Encode(p)[:256]
            ids = np.zeros((256,), np.int32); mask = np.zeros((256,), np.int32)
            ids[:len(toks)] = toks; mask[:len(toks)] = 1
            mrow = mask.astype(bool)
            out = m(ids, mask)[0]; to = tfl(ids, mask)
            d = psnr(to, out, mrow); cmin, cmean = cos_stats(to, out, mrow)
            print(f"{len(toks):>4} | {d:8.2f} {cmin:9.6f} {cmean:9.6f} | {p[:44]}")


if __name__ == "__main__":
    main()
