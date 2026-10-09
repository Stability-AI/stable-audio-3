#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Rebuild the medium DiT rung tflites from scratch: the repo's own PyTorch model
# (stable_audio_3.models.dit.DiffusionTransformer) + the ARC checkpoint -> two
# weight-shared N=10 rung ladders:   sa3-m/dit_fp32.tflite   sa3-m/dit_w8a8.tflite
#
# Env (same convention as build_all.sh; build_paths.py reads SA3_BUILD_WORK):
#   SA3_BUILD_WORK   output/scratch dir (default build/_work)
#   DIT_LADDER       rung lengths (default: the shipped N=10 set)
#   DIT_JOBS         parallel exports (default 1; each big rung peaks tens of GB RAM)
#   PY_EXPORT        python with torch + ai_edge_torch + ai_edge_quantizer
# The DiT model is loaded from the HF cache (stabilityai/stable-audio-3-medium
# ARC .safetensors); export_dit.py locates it. No AE / T5 needed (T5 hidden is an input).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"                 # build -> tflite -> optimized -> repo root
export PYTHONPATH="$REPO:${PYTHONPATH:-}"            # make stable_audio_3 importable
export SA3_BUILD_WORK="${SA3_BUILD_WORK:-$HERE/_work}"
WORK="$SA3_BUILD_WORK"; mkdir -p "$WORK/sa3-m"
PY_EXPORT="${PY_EXPORT:-python}"
DIT_LADDER="${DIT_LADDER:-8 48 96 192 416 704 1056 1496 2040 2824 3536 4096}"
DIT_JOBS="${DIT_JOBS:-1}"
M="$HERE/quant_merge/merge_rungs_generic.py"
csv() { local IFS=,; echo "$*"; }

echo "== DiT 1. export rungs (torch -> tflite, each self-verifies tflite==torch) [$DIT_JOBS-way] =="
pids=()
for R in $DIT_LADDER; do
  $PY_EXPORT "$HERE/export/export_dit.py" "$R" > "$WORK/dit_export_$R.log" 2>&1 &
  pids+=($!)
  # throttle to DIT_JOBS concurrent exports
  while [ "$(jobs -rp | wc -l)" -ge "$DIT_JOBS" ]; do wait -n; done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for R in $DIT_LADDER; do
  grep -qE "EXPORTED R=$R .* (OK|CHECK)" "$WORK/dit_export_$R.log" || { echo "!! export R=$R did not finish"; fail=1; }
  grep -qE "EXPORTED R=$R .* CHECK"      "$WORK/dit_export_$R.log" && { echo "!! export R=$R FAILED tflite==torch"; fail=1; }
done
[ "$fail" = 0 ] || { echo "DiT export stage FAILED — see $WORK/dit_export_*.log"; exit 1; }
echo "   all $(echo $DIT_LADDER | wc -w) rungs exported + tflite==torch OK"

echo "== DiT 1b. export gcond.tflite (global-cond preamble: (seconds,t)->gc[1,9216]) =="
# gc is computed OUTSIDE the rung and fed per step — keeps the 24x-inlined global_cond_embedder FC out of the
# int8 rung, where under the XNNPACK weight cache it corrupts multi-rung int8 (memory sa3-rung-weightcache-int8-bug).
$PY_EXPORT "$HERE/export/export_dit.py" --gcond > "$WORK/dit_export_gcond.log" 2>&1
grep -qE "EXPORTED gcond .* OK"    "$WORK/dit_export_gcond.log" || { echo "!! gcond export failed"; cat "$WORK/dit_export_gcond.log"; exit 1; }
grep -qE "EXPORTED gcond .* CHECK" "$WORK/dit_export_gcond.log" && { echo "!! gcond FAILED tflite==torch"; exit 1; }
echo "   gcond.tflite exported + tflite==torch OK (folded into the merged ladders below as the 'gcond' signature)"

echo "== DiT 2. quantize rungs -> w8a8 (dynamic_wi8_afp32) =="
for R in $DIT_LADDER; do
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/dit_fp32_$R.tflite" "$WORK/dit_fp32_${R}_w8a8.tflite"
done

echo "== DiT 3. merge rungs -> sa3-m/dit_{fp32,w8a8}.tflite (weight-dedup, N signatures) =="
DL="$(csv $DIT_LADDER)"
# SA3_MERGE_EXTRA folds gcond.tflite in as a 'gcond' signature so each ladder is ONE self-contained file.
SA3_MERGE_EXTRA="gcond:gcond.tflite" $PY_EXPORT "$M" "dit_fp32_" ""      "sa3-m/dit_fp32.tflite" "$DL"
SA3_MERGE_EXTRA="gcond:gcond.tflite" $PY_EXPORT "$M" "dit_fp32_" "_w8a8" "sa3-m/dit_w8a8.tflite" "$DL"
echo "== DiT DONE: $WORK/sa3-m/dit_fp32.tflite + dit_w8a8.tflite (each with s<R> rungs + a 'gcond' signature) =="
