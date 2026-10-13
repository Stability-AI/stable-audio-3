#!/usr/bin/env bash
# Build the torch-free C++ AMX-BF16 T5Gemma encoder engine -> t5gemma_cpu_amx.so
set -euo pipefail
ONE=${ONEDNN_HOME:?set ONEDNN_HOME to a static oneDNN+OpenMP build}
cd "${SA3_CPUAMX_HOME:?set SA3_CPUAMX_HOME}/t5gemma_cpu_amx"
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I"$ONE/include" \
    t5gemma_cpu_amx.cpp -o t5gemma_cpu_amx.so "$ONE/lib/libdnnl.a" -ldl -lpthread -lm
echo "built t5gemma_cpu_amx.so ($(stat -c%s t5gemma_cpu_amx.so) bytes)"
