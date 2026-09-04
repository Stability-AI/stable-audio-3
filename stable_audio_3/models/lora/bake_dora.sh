#!/usr/bin/env bash
# Convenience wrapper for the offline DoRA norm baker.
#
# bake_dora.py imports from its own package, so it runs as
# `python -m stable_audio_3.models.lora.bake_dora` rather than as a script path.
# This does that for you, from any working directory:
#
#   ./stable_audio_3/models/lora/bake_dora.sh adapter.safetensors \
#       --base-weights /path/to/model.safetensors
#   ./stable_audio_3/models/lora/bake_dora.sh adapter.safetensors --check
#
# Every argument is passed through, and the exit status is the baker's own
# (--check returns 1 when an adapter is unbaked), so this is usable in a script.
# Set PYTHON to choose an interpreter; defaults to python3.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"

# Repo first on the path: a wrapper shipped inside the checkout should bake with
# the checkout's code, not with whatever copy happens to be installed.
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
  exec "${PYTHON:-python3}" -m stable_audio_3.models.lora.bake_dora "$@"
