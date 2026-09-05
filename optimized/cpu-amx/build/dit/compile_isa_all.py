#!/usr/bin/env python
"""Per-ISA AOT compile of the full fused-runtime Triton kernel set for the SA3-medium int8 DiT.

Run once per ISA in its OWN subprocess with (set by the launcher):
  TRITON_CPU_TARGET_FEATURES=<feat,...>   (forces backend.cpu_features -> MLIR dot lowering)
  TRITON_CPU_AOT_FORCE_ASM_FEATURES=1     (forces emitted ISA via translate_to_asm; REQUIRED)
  TRITON_CACHE_DIR=<per-ISA dir>          (cpu_features is NOT in the cache key -> must isolate)
Argv: <isa_tag {amx,vnni,avx2}> <out_dir>

Produces  <out_dir>/{so/, cpp_kernels.txt, kernels_abi.json, so_flash/_flash_bm{64,128}.so}
matching the dispatch keys the C++ driver expects (dump_bin.py mapping), so
  dit_isa_forward --aotdir <out_dir>   loads the per-ISA kernels.

AVX2 note: gemm_i8 (BK=256) triggers a multi-minute / multi-GB LLVM blow-up on AVX2 (Stage-0
Risk 2). It is NOT on the fused (oneDNN) runtime path, so for AVX2 we compile it at BK=64
(int8->int32 is tiling-invariant-exact, Stage-0 D2) purely to keep the set complete + the
--gemm triton isolation runnable.  Set NOGEMM=1 to skip gemm_i8 entirely.
"""
import os, sys, re, glob, json, shutil, time, subprocess
os.environ.setdefault("TRITON_CPU_BACKEND", "1")
import numpy as np, torch
sys.path.insert(0, os.environ.get("SA3_TRITON_CPU",
                os.path.join(os.environ.get("SA3_CPUAMX_HOME", "engines"), "dit_triton")))
import kernels as K, kernels_fused as KF
from model_p3 import DiTTritonP3
from gen_reference import read_subset, CKPT

TAG = sys.argv[1]
OUT = sys.argv[2]
SO_DIR = os.path.join(OUT, "so")
FLASH_DIR = os.path.join(OUT, "so_flash")
os.makedirs(SO_DIR, exist_ok=True)
os.makedirs(FLASH_DIR, exist_ok=True)
L = int(os.getenv("L", "128"))
NOGEMM = os.getenv("NOGEMM", "0") == "1"
torch.set_num_threads(8)

# AVX2: shrink gemm_i8 contraction tile to dodge the BK=256 legalization blow-up (unused on the
# fused path; int8 GEMM is exact regardless of BK).  AMX/VNNI keep the shipped BK=256.
if TAG == "avx2":
    K.GEMM_I8 = {"BM": 32, "BN": 64, "BK": 64, "GROUP_M": 8}
    print(f"[{TAG}] GEMM_I8 -> BK=64 (avoid AVX2 blow-up)", flush=True)

print(f"[{TAG}] TRITON_CPU_TARGET_FEATURES={os.getenv('TRITON_CPU_TARGET_FEATURES')}")
print(f"[{TAG}] AOT_FORCE_ASM_FEATURES={os.getenv('TRITON_CPU_AOT_FORCE_ASM_FEATURES')}")
print(f"[{TAG}] TRITON_CACHE_DIR={os.getenv('TRITON_CACHE_DIR')}  NOGEMM={NOGEMM}", flush=True)

KERNELS = {
    "_gemm_kernel": K._gemm_kernel,
    "_quant_rows_kernel": K._quant_rows_kernel,
    "_rmsnorm_kernel": K._rmsnorm_kernel,
    "_rope_kernel": K._rope_kernel,
    "_flash_diff_i8_kernel": K._flash_diff_i8_kernel,
    "_rmsnorm_mod_q_kernel": KF._rmsnorm_mod_q_kernel,
    "_gemm_i8_raw_kernel": KF._gemm_i8_raw_kernel,
    "_deq_kernel": KF._deq_kernel,
    "_deq_glu_q_kernel": KF._deq_glu_q_kernel,
    "_deq_gate_res_kernel": KF._deq_gate_res_kernel,
    "_deq_add_kernel": KF._deq_add_kernel,
}


def all_compiled(fn):
    out = []
    def scan(o, d=0):
        if d > 6 or o is None: return
        if isinstance(getattr(o, "asm", None), dict) and o.asm:
            out.append(o); return
        if isinstance(o, dict): [scan(v, d + 1) for v in o.values()]
        elif isinstance(o, (list, tuple, set)): [scan(v, d + 1) for v in o]
    for a in ("cache", "device_caches"):
        scan(getattr(fn, a, None))
    return out


def ctype_code(ty):
    if ty.startswith("*"): return "P"
    return {"i1": "i1", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
            "u32": "u32", "u64": "u64", "fp16": "f16", "bf16": "bf16",
            "fp32": "f32", "fp64": "f64"}.get(ty, ty)


def define_line(ck):
    for ln in ck.asm.get("llir", "").splitlines():
        if ln.startswith("define") and "@" in ln:
            return ln[:800]
    return ""


def locate_so(ck, name):
    so = ck.asm.get("so", None)
    if isinstance(so, (bytes, bytearray)):
        p = os.path.join(SO_DIR, f"_tmp_{ck.hash}.so")
        with open(p, "wb") as f: f.write(so)
        return p
    if isinstance(so, str) and os.path.exists(so):
        return so
    cache = os.getenv("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    hits = glob.glob(os.path.join(cache, ck.hash, f"{name}.so")) or \
        glob.glob(os.path.join(cache, "**", f"{name}.so"), recursive=True)
    return hits[0] if hits else None


def mnem(sopath):
    txt = subprocess.run(["objdump", "-d", sopath], capture_output=True, text=True).stdout
    c = {}
    for m in ("tdpbssd", "vpdpbusd", "vpmaddubsw", "vpmaddwd", "vpmulld", "ldtilecfg", "tileloadd"):
        c[m] = len(re.findall(r"\b" + m + r"\b", txt))
    c["zmm"] = len(re.findall(r"%zmm\d+", txt)); c["ymm"] = len(re.findall(r"%ymm\d+", txt))
    return c


if __name__ == "__main__":
    ref = np.load(os.path.join(os.environ.get("SA3_TRITON_CPU",
              os.path.join(os.environ.get("SA3_CPUAMX_HOME", "engines"), "dit_triton")),
              "ref", f"ref_L{L}.npz"))
    inp = [torch.from_numpy(ref[k]) for k in ("x", "t", "cross", "gcond")]
    sd = read_subset(CKPT)
    m = DiTTritonP3(sd, T_lat=L, prec="int8", backend="tri")
    t0 = time.time()
    with torch.no_grad():
        m.forward(*inp, num_cpu_threads=8)
    print(f"[{TAG}] warm int8/tri forward {time.time()-t0:.1f}s", flush=True)

    manifest = []
    seen = set()
    for name, fn in KERNELS.items():
        for ck in all_compiled(fn):
            if ck.hash in seen: continue
            seen.add(ck.hash)
            sig = ck.src.signature
            consts = {k[0] if isinstance(k, tuple) else k: v for k, v in ck.src.constants.items()}
            surv, cxpr = [], {}
            for i, (an, ty) in enumerate(sig.items()):
                if ty == "constexpr": cxpr[an] = consts.get(i)
                else: surv.append((an, ctype_code(ty)))
            dl = define_line(ck)
            ndef = dl.count("ptr ") + len(re.findall(r"\bi\d+ %", dl)) + len(re.findall(r"\bfloat %", dl)) + len(re.findall(r"\bdouble %", dl))
            sp = locate_so(ck, name)
            so_dst = None
            if sp and os.path.exists(sp):
                so_dst = os.path.join(SO_DIR, f"{name}__{ck.hash[:10]}.so")
                shutil.copy(sp, so_dst)
            manifest.append(dict(kernel=name, hash=ck.hash,
                so=os.path.basename(so_dst) if so_dst else None,
                surviving_args=surv, n_surviving=len(surv), n_define_args=ndef,
                abi_ok=(ndef == len(surv) + 6), constexprs=cxpr))
    with open(os.path.join(OUT, "kernels_abi.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{TAG}] {len(manifest)} specializations; abi_ok all: {all(r['abi_ok'] for r in manifest)}", flush=True)

    # ---- dispatch table (dump_bin.py mapping) ----
    def pick(kernel, **cx):
        c = [r for r in manifest if r["kernel"] == kernel and all(r["constexprs"].get(k) == v for k, v in cx.items())]
        return c[0] if c else None
    rows = []
    def add(key, kernel, **cx):
        r = pick(kernel, **cx)
        if r is None or r["so"] is None:
            print(f"[{TAG}] WARN no .so for {key} ({kernel} {cx}); skipping"); return
        rows.append((key, r["so"], kernel))
    gf = [r for r in manifest if r["kernel"] == "_gemm_kernel" and r["constexprs"].get("HAS_BIAS") is False and "M" not in r["constexprs"]]
    if gf: rows.append(("gemm_fp", gf[0]["so"], "_gemm_kernel"))
    if not NOGEMM: add("gemm_i8", "_gemm_i8_raw_kernel")
    add("quant", "_quant_rows_kernel")
    add("rmsnorm", "_rmsnorm_kernel", HAS_G=True, BK=64)
    add("rope", "_rope_kernel")
    add("flash", "_flash_diff_i8_kernel")
    add("rmsmodq_mod1", "_rmsnorm_mod_q_kernel", HAS_MOD=True)
    add("rmsmodq_mod0", "_rmsnorm_mod_q_kernel", HAS_MOD=False)
    for bn in (256, 4096, 8192):
        add(f"deq_bn{bn}_bias0", "_deq_kernel", BN=bn, HAS_BIAS=False)
    add("deqglu", "_deq_glu_q_kernel")
    add("deqgate_bn2048_bias0", "_deq_gate_res_kernel", BN=2048, HAS_BIAS=False)
    add("deqgate_bn2048_bias1", "_deq_gate_res_kernel", BN=2048, HAS_BIAS=True)
    add("deqadd_bn2048", "_deq_add_kernel", BN=2048, HAS_LOCAL=True)
    with open(os.path.join(OUT, "cpp_kernels.txt"), "w") as f:
        for key, so, sym in rows:
            f.write(f"{key} {so} {sym}\n")
    print(f"[{TAG}] cpp_kernels.txt {len(rows)} dispatch slots", flush=True)

    # ---- retiled flash BM=64,128 (fast path; same ABI, only BM differs) ----
    H, D = 24, 64
    g = lambda s: (torch.randn(H, s, D) * 4)
    for BM in (64, 128):
        K.flash_diff_i8(g(1356), g(1356), g(1356), g(1356), g(1356), 0.125, num_cpu_threads=8, BM=BM, BN=64)
    for ck in all_compiled(K._flash_diff_i8_kernel):
        names = list(ck.src.signature.keys()); idx = {n: i for i, n in enumerate(names)}
        c = dict(ck.src.constants)
        def lk(nm): i = idx[nm]; return c.get((i,), c.get(i))
        bm, bn = lk('BM'), lk('BN')
        if bn == 64 and bm in (64, 128):
            sp = locate_so(ck, "_flash_diff_i8_kernel")
            if sp:
                dst = os.path.join(FLASH_DIR, f"_flash_bm{bm}.so"); shutil.copy(sp, dst)
                print(f"[{TAG}] flash BM={bm} -> {dst}", flush=True)

    # ---- mnemonic proof for the two int8-dot kernels ----
    for key in ("gemm_i8", "flash"):
        r = [x for x in rows if x[0] == key]
        if r:
            print(f"[{TAG}] mnem {key}: {mnem(os.path.join(SO_DIR, r[0][1]))}", flush=True)
    print(f"[{TAG}] DONE", flush=True)
