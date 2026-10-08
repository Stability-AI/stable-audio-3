#!/bin/bash
# Refuse to publish anything that names SA3-large, a private path, or a private adapter.
#
# SA3-large is LOCAL ONLY and always will be. That rule has no other mechanical enforcement:
# one missed docstring in a 536-line file is the whole failure, and it is not something to
# catch by eye on every commit. Run this on the staged tree before any push.
#
#   tools/grep_gate.sh [dir]      exits non-zero on any hit
set -uo pipefail
DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Patterns that must never appear in published source, docs or filenames.
PATTERNS=(
  'sa3-large'            # the model, by name
  'sa3_large'
  'SA3-large'
  'SA3_LARGE'
  'branch_map_large'
  'dit_fp16_branch'      # the large branch engine
  'dit_fp16_refit'       # the large refit engine
  'build_fp16_large'
  'large_common'
  'RefitLora'            # large-only runtime
  'refit_runtime'
  '/admin/home-cj'       # private absolute paths
  '/weka2/cj'
  '/weka/cj'
  'cjmathcore'           # private adapters
  'nosoul'
  'plini'
  'kid.wav'
)
# Lines that legitimately contain a pattern (reviewed, kept deliberately). One regex per line.
ALLOW='^$'

fail=0
for p in "${PATTERNS[@]}"; do
  hits=$(grep -rIn --exclude-dir=.git --exclude='grep_gate.sh' -F -- "$p" "$DIR" 2>/dev/null \
         | grep -Ev "$ALLOW" || true)
  if [ -n "$hits" ]; then
    echo "LEAK  '$p'"
    echo "$hits" | sed 's/^/        /' | head -12
    n=$(echo "$hits" | wc -l)
    [ "$n" -gt 12 ] && echo "        ... +$((n-12)) more"
    fail=1
  fi
done
# filenames too
fn=$(find "$DIR" -name '*large*' -o -name '*refit*' -o -name '*cjmathcore*' 2>/dev/null || true)
if [ -n "$fn" ]; then echo "LEAK  filename"; echo "$fn" | sed 's/^/        /'; fail=1; fi

if [ "$fail" -eq 0 ]; then
  echo "grep gate: clean ($(find "$DIR" -type f | wc -l) files checked)"
else
  echo; echo "grep gate: FAILED — do not publish"
fi
exit $fail
