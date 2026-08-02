#!/bin/bash
# Full cpu-amx test suite: per-engine unit validations (C++ .so vs oracle) + full-pipeline e2e
# + the CLI integration matrix. Needs the oracle data (see ../TESTING.md).
#   Usage:  bash tests/run_all.sh          (everything)
#           bash tests/run_all.sh unit     (just the per-engine validations)
#           bash tests/run_all.sh cli      (just the CLI integration matrix)
set -o pipefail
PY="${PY:-/weka/cj/venvs/sad310/bin/python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CPU="$(dirname "$HERE")"
MODE="${1:-all}"

UNIT=(validate_t5gemma
      validate_same_s_decoder_bf16 validate_same_l_decoder_bf16
      validate_same_s_decoder_int8 validate_same_l_decoder_int8
      validate_same_s_encoder      validate_same_l_encoder)

if [ "$MODE" = "all" ] || [ "$MODE" = "unit" ]; then
  echo "══════ per-engine unit validations (C++ engine vs reference oracle) ══════"
  for t in "${UNIT[@]}"; do
    echo "── $t ──"
    env OMP_NUM_THREADS=16 "$PY" "$HERE/$t.py" 2>&1 \
      | grep -iE "GATE|PASS|FAIL|PSNR|min-?cos|dB|mismatch" | tail -5
  done
  echo "── e2e_pipeline (full C++ path: T5Gemma→DiT→decoder) ──"
  env OMP_NUM_THREADS=16 "$PY" "$HERE/e2e_pipeline.py" 2>&1 \
    | grep -iE "VERDICT|FULL-PIPELINE|encoder C\+\+" | tail -3
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "cli" ]; then
  echo
  echo "══════ CLI integration matrix (every mode: t2a/a2a/inpaint/CFG/neg/APG/…) ══════"
  cd "$CPU" && env OMP_NUM_THREADS=16 "$PY" scripts/test_all_configs.py 2>&1 \
    | grep -E "✓|✗|ALL PASS|failures|phase"
fi
