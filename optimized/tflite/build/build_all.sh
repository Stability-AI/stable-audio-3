#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Rebuild ALL 8 SAME-AE rung tflites from the original checkpoints, from scratch.
# Nothing here needs the pre-built artifacts — this is the full reproducible path.
#
# Two Python envs (kept separate on purpose):
#   $PY_EXPORT  — torch + ai_edge_torch + ai_edge_quantizer (litert-export 1.2.0)  → extract/export/quant/merge
#   $PY_RUNTIME — ai_edge_litert >= 2.2.0 (CompiledModel + weight cache)            → verify (RungEncoder/Decoder)
#
# Checkpoints (public on HF; standalone AE repos also work — see stable_audio_3.model_configs):
#   sa3-medium ARC.safetensors  → SAME-L enc+dec        sa3-sm-music ckpt → SAME-S enc+dec
#
# ⚠ PREREQ before this runs clean from a fresh checkout: the export/quant/merge scripts currently
#   hardcode a scratch dir (SC=...). Parameterize them to read $WORK (output dir) — tracked cleanup.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/_work}"; mkdir -p "$WORK"
PY_EXPORT="${PY_EXPORT:-python}"
PY_RUNTIME="${PY_RUNTIME:-python}"
LADDER_L="1 2 4 8 12 16 32 64 128 256"
LADDER_S="2 4 8 12 16 32 64 128 256"
csv() { local IFS=,; echo "$*"; }

echo "== 1. extract weights (ckpt -> npz)  [4 extractors, all recreated + verified bit-exact] =="
$PY_EXPORT "$HERE/extract/extract_same_l_encoder_weights.py"
$PY_EXPORT "$HERE/extract/extract_same_l_decoder_weights.py"
$PY_EXPORT "$HERE/extract/extract_same_s_encoder_weights.py"
$PY_EXPORT "$HERE/extract/extract_same_s_decoder_weights.py"

echo "== 2. export fixed windowed rungs (torch -> tflite) =="
for S in $LADDER_L; do
  $PY_EXPORT "$HERE/export/export_windowed_param.py" "$S"   # SAME-L decoder  -> same-l_windowed_$S.tflite
  $PY_EXPORT "$HERE/export/export_enc_windowed.py"   "$S"   # SAME-L encoder  -> same-l_enc_windowed_$S.tflite
done
for S in $LADDER_S; do
  $PY_EXPORT "$HERE/export/export_same_s_fixed.py" enc "$S" # SAME-S encoder  -> same-s_enc_fixed_$S.tflite
  $PY_EXPORT "$HERE/export/export_same_s_fixed.py" dec "$S" # SAME-S decoder  -> same-s_dec_fixed_$S.tflite
done

echo "== 3. quantize every fixed rung -> w8a8 (recipe.dynamic_wi8_afp32) =="
for S in $LADDER_L; do
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/same-l_windowed_$S.tflite"     "$WORK/same-l_windowed_${S}_w8a8.tflite"
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/same-l_enc_windowed_$S.tflite" "$WORK/same-l_enc_windowed_${S}_w8a8.tflite"
done
for S in $LADDER_S; do
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/same-s_enc_fixed_$S.tflite" "$WORK/same-s_enc_fixed_${S}_w8a8.tflite"
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/same-s_dec_fixed_$S.tflite" "$WORK/same-s_dec_fixed_${S}_w8a8.tflite"
done

echo "== 4. merge fixed rungs -> 8 canonical files (weight-dedup, explicit ladder) =="
LL="$(csv $LADDER_L)"; SS="$(csv $LADDER_S)"
M="$HERE/quant_merge/merge_rungs_generic.py"
$PY_EXPORT "$M" "same-l_windowed_"     ""      "same-l/dec_fp32.tflite"  "$LL"
$PY_EXPORT "$M" "same-l_windowed_"     "_w8a8" "same-l/dec_w8a8.tflite"  "$LL"
$PY_EXPORT "$M" "same-l_enc_windowed_" ""      "same-l/enc_fp32.tflite"  "$LL"
$PY_EXPORT "$M" "same-l_enc_windowed_" "_w8a8" "same-l/enc_w8a8.tflite"  "$LL"
$PY_EXPORT "$M" "same-s_dec_fixed_"    ""      "same-s/dec_fp32.tflite"  "$SS"
$PY_EXPORT "$M" "same-s_dec_fixed_"    "_w8a8" "same-s/dec_w8a8.tflite"  "$SS"
$PY_EXPORT "$M" "same-s_enc_fixed_"    ""      "same-s/enc_fp32.tflite"  "$SS"
$PY_EXPORT "$M" "same-s_enc_fixed_"    "_w8a8" "same-s/enc_w8a8.tflite"  "$SS"

echo "== 5. verify (runtime env): discovered ladders + tiny-L exact + round-trip vs GT =="
$PY_RUNTIME "$HERE/verify_final.py"
echo "== DONE: 8 canonical rung tflites under $WORK/same-{l,s}/ =="
