#!/bin/bash
set -e
ONE=/weka2/cj/tmp/onednn-omp
cd "$(dirname "$0")"
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -I"$ONE/include" \
    same_l_int8fused_cpu_amx.cpp -o same_l_int8fused_cpu_amx.so \
    "$ONE/lib/libdnnl.a" -ldl -lpthread -lm
echo "built $(ls -la same_l_int8fused_cpu_amx.so | awk '{print $5}') bytes"
