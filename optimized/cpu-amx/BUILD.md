# Building the cpu-amx engines

The `cpu-amx` runtime loads six torch-free C++/AMX engines as `.so`s via ctypes. This doc explains
how to (re)build each `.so` and dump its weight blob. The runtime itself (CLI/gradio) needs only the
built `.so`s + weight blobs; you only need this if you're rebuilding from source.

All sources live under `build/`. Prebuilt `.so`s + weight blobs currently live in the per-engine dirs
under `/weka2/cj/clod/` (see "Where the runtime looks" below) — `backends.py` points there.

## Components

| engine | source (`build/…`) | GEMM path | weight blob | notes |
|---|---|---|---|---|
| **T5Gemma encoder** | `t5gemma/` | oneDNN AMX-BF16 | `weights.bin` (563 MB bf16) | 12-layer Gemma2 encoder |
| **medium DiT** (int8) | `dit/` | AOT Triton (int8) + oneDNN | `core_L{N}.bin` | **the complex one** — AOT-compiled Triton kernels |
| **SAME-S decoder** (bf16) | `same_s_bf16/` | oneDNN AMX-BF16 | `weights.bin` | fastest / distilled 50M |
| **SAME-L decoder** (bf16) | `same_l_bf16/` | oneDNN AMX-BF16 | `weights.bin` (852 MB) | native 426M |
| **SAME-S decoder** (int8) | `same_s_int8/` | oneDNN AMX-INT8 (fused) | `weights.bin` (55 MB) | ships SQ+GPTQ ("improved") grid |
| **SAME-L decoder** (int8) | `same_l_int8/` | oneDNN AMX-INT8 (fused) | `weights.bin` (427 MB) | ships SQ+GPTQ ("improved") grid |

## Prerequisites

1. **CPU with AMX** (Sapphire Rapids / Emerald Rapids Xeon; reports `cpu_isa_avx10_1_512_amx`).
   The engines fall back to VNNI/AVX2 but the headline speed needs AMX (see `LESSONS.md`).
2. **Static oneDNN with the OpenMP runtime** at `/weka2/cj/tmp/onednn-omp` (`include/`, `lib/libdnnl.a`).
   Rebuild from oneDNN source with `-DDNNL_CPU_RUNTIME=OMP -DDNNL_LIBRARY_TYPE=STATIC` if it's missing.
3. **g++** with C++17 + `-march=native` (must resolve AMX intrinsics).
4. **triton-cpu fork** at `/weka2/cj/clod/tritoncpu_sa3/` — ONLY for rebuilding the DiT's AOT kernels.
5. Python venv `/weka/cj/venvs/sad310/bin/python3` (numpy; for weight dumping).

## Weight sourcing

All weights derive from the HF release **`stabilityai/stable-audio-3-optimized`** → the `MLX/*.npz`
files (fp16 named arrays). Download with `huggingface_hub.hf_hub_download`:
- `MLX/t5gemma_f16.npz`  → T5Gemma
- `MLX/same_s_decoder_f32.npz`, `MLX/same_l_decoder_f32.npz` → decoders
- the medium DiT int8 weights come from the campaign's `aot_stage2/weights_int8_L{N}.npz` (int8-quantized
  offline from `MLX/dit_medium_f16.npz`).

## Build recipes

### T5Gemma, and both bf16 / int8 decoders (the easy five)

All five are a single `g++` + static-oneDNN link. The pattern (see each dir's `build.sh`, or the build
comment at the top of the `.cpp`):

```bash
ONE=/weka2/cj/tmp/onednn-omp
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I"$ONE/include" \
    <engine>.cpp -o <engine>.so "$ONE/lib/libdnnl.a" -ldl -lpthread -lm
```

Then dump the weight blob from the HF npz (each dir has its dumper):

```bash
# T5Gemma
python build/t5gemma/dump_weights.py   <t5gemma_f16.npz>        # -> weights.bin + weights_manifest.txt
# bf16 decoders
python build/same_s_bf16/dump_weights.py <same_s_decoder_f32.npz>
python build/same_l_bf16/dump_weights.py <same_l_decoder_f32.npz>
# int8 decoders (naive grid)
python build/same_s_int8/dump_weights_int8.py <same_s_decoder_f32.npz>
python build/same_l_int8/dump_weights_int8.py <same_l_decoder_f32.npz>
```

The dumpers write `weights.bin` (bf16 or per-out-channel int8) + `weights_manifest.txt` (the mmap layout
the C++ loader reads). Linear weights are pre-transposed to `(in,out)` so the row-major oneDNN matmul
consumes them directly. **The int8 "improved" (SQ+GPTQ) blobs that actually ship** are produced by the
separate calibration tooling in `/weka2/cj/clod/gptqsq_test/` (SmoothQuant α0.9 → GPTQ → re-quant); they
are byte-compatible drop-ins for the int8 `.so` and are what the shipped `weights.bin` symlinks point to.

### medium DiT (int8) — the AOT-Triton path

The DiT does its int8 GEMMs + fused int8 attention through **AOT-compiled Triton-CPU kernels** (not plain
oneDNN), because the fully-fused all-integer path is the CPU win (see `LESSONS.md`). Two stages:

**1. Compile the per-ISA kernel set** (`build/dit/compile_isa_all.py`). Run once per ISA in its OWN
subprocess with the env the launcher sets — the ISA is NOT in Triton's cache key, so each ISA needs an
isolated cache dir:

```bash
TRITON_CPU_BACKEND=1 \
TRITON_CPU_TARGET_FEATURES=+amx-tile,+amx-int8,+amx-bf16,+avx512f,... \
TRITON_CPU_AOT_FORCE_ASM_FEATURES=1 \
TRITON_CACHE_DIR=/tmp/triton_cache_amx \
python build/dit/compile_isa_all.py amx <out_dir>
```

Produces `<out_dir>/{so/, cpp_kernels.txt, kernels_abi.json, so_flash/}` matching the dispatch keys the
C++ driver expects. Repeat for `vnni` / `avx2` if you want the fallbacks (AMX is the shipped path).
(Note: `gemm_i8` at BK=256 blows LLVM up on AVX2 — it's compiled at BK=64 there, int8→int32 is
tiling-invariant-exact; `NOGEMM=1` skips it.)

**2. Dump the weight blob** for the sequence length(s) you need (`build/dit/gen_core.py`):

```bash
python build/dit/gen_core.py 320    # -> core_L320.bin + core_L320_manifest.txt (L-independent block weights)
```

`core_L{N}.bin` flattens the int8 block weights + a golden preamble. The block weights are
sequence-length-independent, so one `core_L*.bin` serves any length (the `L` in the name is just the
golden shape).

**3. Build the C++ driver** (`build/dit/dit_cpu_amx.cpp`) — links static oneDNN like the others and
`dlopen`s the AOT kernels from `<out_dir>`. See the build comment at the top of the `.cpp`.

⚠ A full clean DiT rebuild needs the triton-cpu fork + the int8-quantized weight npzs
(`aot_stage2/weights_int8_L{N}.npz`). If you only need to run (not rebuild), use the prebuilt
`dit_cpu_amx.so` + `core_L320.bin`.

## Where the runtime looks

`scripts/backends.py` loads each engine's `.so` + weight blob from an engine dir under `/weka2/cj/clod/`
(`t5gemma_cpu_amx/`, `same_{s,l}_cpu_amx/`, `same_{s,l}_int8fused_cpu_amx/`, `same_{s,l}_encoder_cpu_amx/`,
`tritoncpu_sa3/aot_speedprove/`). **These binaries are NOT in git — `scripts/weights.py` `ensure()`
downloads them from HF** (`stabilityai/stable-audio-3-optimized/cpu-amx/`, flat) into those dirs on first
use; a local build (this BUILD.md) satisfies them too and skips the download. Override the base dir with
`SA3_CPUAMX_HOME`, or `SA3_CPUAMX_NO_HF=1` to force local-only. The DiT `.so` bakes in absolute kernel/core
paths (`tritoncpu_sa3/aot_stage2`, `aot_speedprove`), so those are fixed for now — making the DiT `.so`
env-configurable (like the other seven engines, which load relative to their dir) is the remaining
portability follow-up.

See `LESSONS.md` for the numerics gotchas each of these engines encodes (fp32 islands, the bf16-RoPE
long-sequence bug, the global-static-weights trap, the DiT teardown quirk, …). Read it before touching
the `.cpp`s.
