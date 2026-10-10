#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a SMALL DiT (sm-music or sm-sfx) rung ladder from the repo's own PyTorch
# model (stable_audio_3.models.dit) + the small checkpoint -> two weight-shared
# ladders:   sa3-sm-<fam>/dit_fp32.tflite   sa3-sm-<fam>/dit_w8a8.tflite
#
# Same export/quant/merge pipeline as build_dit.sh (the medium builder), driven by
# SA3_DIT_FAMILY so export_dit.py loads the right checkpoint and derives GCE_OUT
# (6*embed_dim = 6144 for the small DiTs vs 9216 medium). The 2-minute ceiling is
# valid_T_lat(120) = 1292 latents, so the ladder tops out there (a full 120 s render
# fits the 1292 rung exactly; medium's ladder went to 4096).
#
# Usage:   build_dit_small.sh <sm-music|sm-sfx>
# Env:     SA3_BUILD_WORK (scratch; default build/_work/<fam>), DIT_LADDER, DIT_JOBS,
#          PY_EXPORT, SA3_CKPT_SMMUSIC / SA3_CKPT_SMSFX (optional explicit ckpt paths).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"                 # build -> tflite -> optimized -> repo root
export PYTHONPATH="$REPO:${PYTHONPATH:-}"            # make stable_audio_3 importable

FAMILY="${1:?usage: build_dit_small.sh <sm-music|sm-sfx>}"
case "$FAMILY" in
  sm-music) OUTDIR="sa3-sm-music" ;;
  sm-sfx)   OUTDIR="sa3-sm-sfx" ;;
  *) echo "family must be sm-music or sm-sfx (got '$FAMILY')"; exit 1 ;;
esac
export SA3_DIT_FAMILY="$FAMILY"

export SA3_BUILD_WORK="${SA3_BUILD_WORK:-$HERE/_work/$FAMILY}"
WORK="$SA3_BUILD_WORK"; mkdir -p "$WORK"
PY_EXPORT="${PY_EXPORT:-python}"
# 2-minute ladder: reuse medium's low/mid rungs, cap at valid_T_lat(120)=1292 (full 120 s fits exactly).
DIT_LADDER="${DIT_LADDER:-8 48 96 192 416 704 1056 1292}"
DIT_JOBS="${DIT_JOBS:-1}"
M="$HERE/quant_merge/merge_rungs_generic.py"
csv() { local IFS=,; echo "$*"; }

echo "== $FAMILY DiT 1. export rungs (torch -> tflite, each self-verifies tflite==torch) [$DIT_JOBS-way] =="
pids=()
for R in $DIT_LADDER; do
  $PY_EXPORT "$HERE/export/export_dit.py" "$R" > "$WORK/dit_export_$R.log" 2>&1 &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$DIT_JOBS" ]; do wait -n; done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for R in $DIT_LADDER; do
  grep -qE "EXPORTED R=$R .* (OK|CHECK)" "$WORK/dit_export_$R.log" || { echo "!! export R=$R did not finish"; fail=1; }
  grep -qE "EXPORTED R=$R .* CHECK"      "$WORK/dit_export_$R.log" && { echo "!! export R=$R FAILED tflite==torch"; fail=1; }
done
[ "$fail" = 0 ] || { echo "$FAMILY DiT export stage FAILED — see $WORK/dit_export_*.log"; exit 1; }
echo "   all $(echo $DIT_LADDER | wc -w) rungs exported + tflite==torch OK"

echo "== $FAMILY DiT 1b. export gcond.tflite (global-cond preamble: (seconds,t)->gc[1,$((6*1024))]) =="
$PY_EXPORT "$HERE/export/export_dit.py" --gcond > "$WORK/dit_export_gcond.log" 2>&1
grep -qE "EXPORTED gcond .* OK"    "$WORK/dit_export_gcond.log" || { echo "!! gcond export failed"; cat "$WORK/dit_export_gcond.log"; exit 1; }
grep -qE "EXPORTED gcond .* CHECK" "$WORK/dit_export_gcond.log" && { echo "!! gcond FAILED tflite==torch"; exit 1; }
echo "   gcond.tflite exported + tflite==torch OK (folded into the merged ladders below as 'gcond')"

echo "== $FAMILY DiT 2. quantize rungs -> w8a8 (dynamic_wi8_afp32) =="
for R in $DIT_LADDER; do
  $PY_EXPORT "$HERE/quant_merge/quant_one.py" "$WORK/dit_fp32_$R.tflite" "$WORK/dit_fp32_${R}_w8a8.tflite"
done

echo "== $FAMILY DiT 3. merge rungs -> $OUTDIR/dit_{fp32,w8a8}.tflite (weight-dedup, N signatures) =="
DL="$(csv $DIT_LADDER)"
mkdir -p "$WORK/$OUTDIR"
SA3_MERGE_EXTRA="gcond:gcond.tflite" $PY_EXPORT "$M" "dit_fp32_" ""      "$OUTDIR/dit_fp32.tflite" "$DL"
SA3_MERGE_EXTRA="gcond:gcond.tflite" $PY_EXPORT "$M" "dit_fp32_" "_w8a8" "$OUTDIR/dit_w8a8.tflite" "$DL"
echo "== $FAMILY DiT DONE: $WORK/$OUTDIR/dit_fp32.tflite + dit_w8a8.tflite (s<R> rungs + 'gcond') =="
