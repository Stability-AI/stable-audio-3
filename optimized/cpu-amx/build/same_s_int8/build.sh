#!/bin/bash
# Build the FUSED int8 SAME-S engine, STATIC-linked against the threaded OMP oneDNN (clean ldd,
# same threaded GEMM as the naive/bf16 engines). ONE=... points at the OMP static oneDNN build.
set -e
ONE=/weka2/cj/tmp/onednn-omp
cd "$(dirname "$0")"
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I"$ONE/include" \
    same_s_int8fused_cpu_amx.cpp -o same_s_int8fused_cpu_amx.so \
    "$ONE/lib/libdnnl.a" -ldl -lpthread -lm
echo "built $(ls -la same_s_int8fused_cpu_amx.so | awk '{print $5}') bytes"
