# Testing the cpu-amx pipeline

Three layers, from unit to end-to-end:

| layer | what it checks | where |
|---|---|---|
| **unit** | each C++/AMX engine's output vs a reference oracle (cosine / PSNR gate) | `tests/validate_*.py` |
| **e2e** | the full C++ path (T5Gemma → DiT → decoder) produces valid audio + matches the TFLite-T5 baseline | `tests/e2e_pipeline.py` |
| **integration** | every CLI mode (t2a / a2a / inpaint / CFG / neg-prompt / APG / step counts / …) yields a valid WAV | `scripts/test_all_configs.py` |

Run everything: `bash tests/run_all.sh`  (`unit` or `cli` to run one layer).

## Unit validations — engine vs oracle

Each script loads the C++ `.so`, runs it on real inputs, and compares to an independent reference. It
prints a per-input cosine + PSNR table and a **GATE PASS/FAIL**.

| test | engine | oracle | gate |
|---|---|---|---|
| `validate_t5gemma.py` | T5Gemma encoder | groundtruth `ref_b3_seq256.npz` + TFLite fp16 | cos ≥ 0.999, PSNR ≥ 45 dB (got 61–67) |
| `validate_same_s_decoder_bf16.py` | SAME-S decoder (bf16) | torch decoder port | ~62 dB vs port |
| `validate_same_l_decoder_bf16.py` | SAME-L decoder (bf16) | torch decoder port + groundtruth | ~62 dB vs port |
| `validate_same_s_decoder_int8.py` | SAME-S decoder (int8, shipped SQ+GPTQ) | torch decoder port | ~40 dB (audio) vs fp32 |
| `validate_same_l_decoder_int8.py` | SAME-L decoder (int8, shipped SQ+GPTQ) | torch port + groundtruth | ~45 dB vs fp32 |
| `validate_same_s_encoder.py` | SAME-S encoder | TFLite `enc_fp32` + numpy-fp32 ref | cos ≥ 0.999 (got 49–51 dB) |
| `validate_same_l_encoder.py` | SAME-L encoder | TFLite `enc_fp32` + numpy-fp32 ref | mean-cos ≥ 0.9999 (bf16; fp32 mode → 107–116 dB) |

The **medium DiT** has no standalone `validate_*.py`: it is checked against a **golden preamble+output**
baked into its weight blob (`build/dit/gen_core.py` writes `x_init/ctx/gc/out` golden tensors), and
end-to-end by `e2e_pipeline.py`.

Note the two accuracy conventions (from `LESSONS.md`): decoder numbers are **vs the torch port**, which
itself sits ~38 dB (corr 0.998) from the true stable-audio-tools AE — so "62 dB vs port" is not "62 dB
vs the real autoencoder." Encoder numbers are **vs the TFLite fp32 encoder** (the clean matched oracle).

## Oracle data these tests need

The `.so`s + weight blobs are built per `BUILD.md`. The oracles are extra data:

- **Groundtruth** `ref_b3_seq256.npz` (T5Gemma) — at `sa3-w4-cluster/groundtruth/t5gemma/`.
- **TFLite oracles** — the decoders' torch ports live in `sa3-w4-cluster/models/defs/`; the encoders'
  `enc_fp32.tflite` and T5Gemma `t5gemma_seq256_fp16.tflite` come from HF
  `stabilityai/stable-audio-3-optimized` (`tflite/…`), auto-downloaded into each engine's `hf/` dir.
- **Real audio** for encoder/round-trip tests — `gguf/ground-truth/asx/285s/audio/*.flac`.
- The unit scripts are self-contained (absolute `sys.path` to the campaign engine dirs under
  `/weka2/cj/clod/`); if you relocate the engines, update those paths (same coupling as `backends.py`,
  see `BUILD.md` "Where the runtime looks").

## Integration matrix — `scripts/test_all_configs.py`

Phase 1 asserts every engine `.so` + vendored asset is present; phase 2 runs the CLI in ~14 configs
(medium DiT, 3 s clips, steps=4) and gates each output WAV on **correct duration ± 0.1 s, finite, and
not silent** (peak ≥ 0.005, rms ≥ 0.0005). Prints a `✓/✗` row per config and `✓ ALL PASS` / a failure
list. This is the authoritative "does every mode still work" check — **currently 14/14 PASS**, including
a2a + inpaint on the C++ AMX encoders (~6 s each; they were ~42 s on the old torch AE encoder).
