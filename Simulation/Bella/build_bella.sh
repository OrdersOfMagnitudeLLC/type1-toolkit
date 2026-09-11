#!/bin/bash
set -e
echo "=== Bella DFT-FE Build ==="
mkdir -p /bella/engine/build
cd /bella/engine/build
cmake /bella/engine \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_MPI=ON \
  -DWITH_ELPA=OFF \
  -DWITH_GPU=OFF \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  2>&1 | tee /bella/findings/cmake_output.txt
echo "=== CMake done, building ==="
make -j$(nproc) dft 2>&1 | tee /bella/findings/build_output.txt
echo "=== BUILD COMPLETE ==="
