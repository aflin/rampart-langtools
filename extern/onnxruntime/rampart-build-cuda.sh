#!/bin/sh
# Build the vendored ONNX Runtime with the CUDA execution provider into BUILD_DIR.
#
# Unlike the CPU build (rampart-build-cpu.sh, fully static), ORT ships the CUDA EP
# as a SEPARATE shared provider library that it dlopen()s at runtime -- it cannot
# be whole-archived into rampart-onnx.so. So this is a SHARED build and emits, into
# BUILD_DIR/Release:
#     libonnxruntime.so                      (core, dynamically linked by the module)
#     libonnxruntime_providers_shared.so     (provider bridge)
#     libonnxruntime_providers_cuda.so       (the CUDA EP itself)
# rampart-onnx_cu*.so links libonnxruntime.so and these three ride alongside it in
# the install dir (rpath $ORIGIN). At runtime the target still supplies CUDA 12
# (libcudart.so.12, libcublas*.so.12) and, unless minimal, cuDNN 9 (libcudnn.so.9).
#
#   ./rampart-build-cuda.sh [BUILD_DIR]
#
# Env knobs:
#   ONNX_CUDA_ARCH     CMAKE_CUDA_ARCHITECTURES (default "89-real;89-virtual" -> Ada/RTX-40xx;
#                      add e.g. "80-real;86-real;89-real;90-virtual" for a wider fleet)
#   ONNX_CUDA_MINIMAL  =1 builds WITHOUT cuDNN via ORT's --enable_cuda_minimal_build.
#                      WARNING: that build has NO compute kernels AT ALL -- every op
#                      falls back to CPU (verified on Jetson sm_87: GPU-EP timings
#                      identical to CPU). Plumbing tests only. Real GPU execution
#                      needs the FULL EP: cuDNN 9 at build time AND libcudnn at run
#                      time. Default 0 (full EP).
#   CUDA_HOME          CUDA toolkit root (default /usr/local/cuda)
#   CUDNN_HOME         cuDNN root (required unless ONNX_CUDA_MINIMAL=1)
set -eu
cd "$(dirname "$0")"

BUILD_DIR="${1:-$(cd ../.. && pwd)/build/extern/onnxruntime-cuda}"

# ORT's build.sh drives everything through python3; the manylinux ovens may not
# expose one on PATH (see rampart-build-cpu.sh). Discover a stable one.
if ! command -v python3 >/dev/null 2>&1; then
  for cand in /opt/python/cp312-cp312/bin /opt/python/cp311-cp311/bin \
              /opt/python/cp310-cp310/bin /opt/python/cp313-cp313/bin; do
    if [ -x "$cand/python3" ]; then PATH="$cand:$PATH"; export PATH; break; fi
  done
fi
command -v python3 >/dev/null 2>&1 || \
  { echo "rampart-build-cuda.sh: no python3 found (ORT build requires it)" >&2; exit 1; }

# ORT's CUDA build is very memory-hungry: each parallel nvcc compiling flash-attn /
# quantized kernels can use 2-4 GB, MULTIPLIED across every -real arch. On a RAM-
# constrained builder a high --parallel can OOM/lock the host, so make it tunable.
# 15 GB box building the full fleet (6 arches): use 2. Default stays 8 for big boxes.
PARALLEL="${ONNX_CUDA_PARALLEL:-8}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Accept the arch list comma-separated (a ';' can't survive transport through CMake's
# `-E env`, which splits on it). Convert to the ';' that CMAKE_CUDA_ARCHITECTURES wants.
CUDA_ARCH="${ONNX_CUDA_ARCH:-89-real,89-virtual}"
CUDA_ARCH=$(printf '%s' "$CUDA_ARCH" | tr ',' ';')

# -Wno-psabi silences GCC's harmless aarch64 "parameter passing ... changed to
# match C++14 in GCC 10.1" note (float4.h ToFloat2), which otherwise prints for
# hundreds of ORT .cc translation units.  It's a no-op on x86.
set -- \
  --build_dir "$BUILD_DIR" \
  --config Release --parallel "$PARALLEL" --compile_no_warning_as_error \
  --skip_tests --skip_submodule_sync \
  --allow_running_as_root \
  --build_shared_lib \
  --use_cuda --cuda_home "$CUDA_HOME" \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines onnxruntime_USE_XNNPACK=OFF \
  --cmake_extra_defines CMAKE_POSITION_INDEPENDENT_CODE=ON \
  --cmake_extra_defines "CMAKE_CXX_FLAGS=-Wno-psabi" \
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"

if [ "${ONNX_CUDA_MINIMAL:-0}" = "1" ]; then
  # A minimal build links NO cuDNN, but build.py still validates that --cudnn_home
  # exists (it only os.path.exists() checks it). Point it at CUDA_HOME (a real dir)
  # to satisfy that gate; the CUDA_MINIMAL cmake branch never looks inside it.
  set -- "$@" --enable_cuda_minimal_build --cudnn_home "${CUDNN_HOME:-$CUDA_HOME}"
else
  # Full EP: cuDNN at build time. Default to /usr (where the cu12/cu13 ovens' dnf
  # cuDNN lands: /usr/include/cudnn.h + /usr/lib64/libcudnn.so.9).
  CUDNN_HOME="${CUDNN_HOME:-/usr}"
  [ -e "$CUDNN_HOME/include/cudnn.h" ] || \
    echo "rampart-build-cuda.sh: warning: $CUDNN_HOME/include/cudnn.h not found (full CUDA EP needs cuDNN; set ONNX_CUDA_MINIMAL=1 or install cuDNN)" >&2
  set -- "$@" --cudnn_home "$CUDNN_HOME"
fi

# Per-nvcc ARCH-compile concurrency (distinct from --parallel, the nvcc COUNT).
# One nvcc compiles a .cu for EVERY -real arch, and by default runs several of
# those arch-compiles at once (nvcc's own threads).  The cutlass flash-attn /
# quantized-GEMM cicc passes use 2-4 GB EACH, so on a RAM-tight builder even
# --parallel 1 OOMs (default 4 threads x ~4 GB = ~16 GB for one flash-attn file).
# ONNX_FLASH_NVCC_THREADS caps the biggest (flash-attn) files ONLY -- changing it
# recompiles just those; ONNX_NVCC_THREADS caps ALL CUDA -- recompiles everything.
# Unset => ORT defaults (fast; fine on big boxes).  A 7 GB box needs =1.
[ -n "${ONNX_NVCC_THREADS:-}" ]       && set -- "$@" --nvcc_threads "$ONNX_NVCC_THREADS"
[ -n "${ONNX_FLASH_NVCC_THREADS:-}" ] && set -- "$@" --flash_nvcc_threads "$ONNX_FLASH_NVCC_THREADS"

./build.sh "$@" --update --build

REL="$BUILD_DIR/Release"
echo "rampart-build-cuda.sh: CUDA ORT shared libs ->"
ls -1 "$REL"/libonnxruntime.so "$REL"/libonnxruntime_providers_shared.so \
       "$REL"/libonnxruntime_providers_cuda.so 2>/dev/null || true
