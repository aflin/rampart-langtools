#!/bin/sh
# Build the vendored ONNX Runtime (CPU, shared lib) into BUILD_DIR.
#
# ONNX Runtime can't be add_subdirectory'd into the rampart-langtools CMake
# build (it's a self-contained top-level build.py project), so its own build is
# driven here. This is invoked automatically by the CMake build via
# ExternalProject_Add (extern/extern.cmake); it can also be run standalone:
#
#     ./rampart-build-cpu.sh [BUILD_DIR]
#
# Emits libonnxruntime.so into BUILD_DIR/Release. Needs network (ORT pulls deps
# via FetchContent) and CMake >= 3.28 (the onnx dep floors at 3.26). build.sh
# hardcodes its own --build_dir first; ours is appended and wins (argparse: last).
set -eu
cd "$(dirname "$0")"

BUILD_DIR="${1:-$(cd ../.. && pwd)/build/extern/onnxruntime}"

# macOS: pin the deployment target to rampart-langtools' floor (CMakeLists uses
# -mmacosx-version-min=11.0). Without this, ORT's objects build at the HOST's
# macOS version and the final rampart-onnx.so would not run on older systems.
# CMake reads this env as the default CMAKE_OSX_DEPLOYMENT_TARGET.
if [ "$(uname)" = "Darwin" ]; then
  MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
  export MACOSX_DEPLOYMENT_TARGET
fi

# ORT's build.sh drives everything through `python3`. The manylinux ovens ship
# several pythons under /opt/python/cpXX-*/bin but leave none on PATH as python3,
# so find a stable one (prefer 3.10-3.12 over the beta 3.14/3.15 builds) and
# prepend it. No-op on hosts that already have python3 on PATH.
if ! command -v python3 >/dev/null 2>&1; then
  for cand in /opt/python/cp312-cp312/bin /opt/python/cp311-cp311/bin \
              /opt/python/cp310-cp310/bin /opt/python/cp313-cp313/bin; do
    if [ -x "$cand/python3" ]; then PATH="$cand:$PATH"; export PATH; break; fi
  done
fi
command -v python3 >/dev/null 2>&1 || \
  { echo "rampart-build-cpu.sh: no python3 found (ORT build requires it)" >&2; exit 1; }

# STATIC build: omit --build_shared_lib so ORT emits static archives (no
# libonnxruntime.so), and force PIC since those .a's get linked into the
# rampart-onnx.so shared object. CMAKE_POSITION_INDEPENDENT_CODE=ON propagates
# to the FetchContent deps (abseil/protobuf/onnx/re2/...) too.
# FreeBSD: ORT's FetchContent patch steps use GNU-patch flags; /usr/bin/patch
# is BSD patch and rejects them ("patch: usage" -> abseil populate fails).
# Point cmake's find at gpatch (pkg install patch).
EXTRA_DEFINES=""
if [ "$(uname)" = "FreeBSD" ]; then
  GP="$(command -v gpatch || true)"
  [ -n "$GP" ] && EXTRA_DEFINES="--cmake_extra_defines Patch_EXECUTABLE=$GP"
fi

# macOS: the CoreML execution provider is deliberately NOT built. onnx gets no
# worthwhile acceleration from it on macOS for our models -- the MPSGraph (GPU)
# path aborts on this transformer class, the ANE path re-specializes per input
# shape and loses on the ragged batches the embed path feeds, and Apple Silicon's
# CPU EP (AMX via Accelerate) is already fast; it was opt-in + off by default for
# exactly that reason. Dropping it also keeps the module buildable against older
# macOS SDKs -- ORT 1.27's CoreML EP uses macOS-13 APIs (MLComputeUnitsCPUAnd-
# NeuralEngine, getBytesWithHandler:) absent from the macOS 11/12 SDK -- so the
# low-floor x86 build works. (llama.cpp still uses Metal where it helps; onnx on
# macOS is CPU-only either way.)

# ONNX_CPU_PARALLEL: build parallelism (default 8).  Lower it on small-RAM
# hosts (e.g. 6 on a 7 GB aarch64 VM) -- some ORT translation units are
# memory-hungry and 8 concurrent gcc's can OOM.
./build.sh \
  --build_dir "$BUILD_DIR" \
  --config Release --parallel "${ONNX_CPU_PARALLEL:-8}" --compile_no_warning_as_error \
  --skip_tests --skip_submodule_sync \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines onnxruntime_USE_XNNPACK=OFF \
  --cmake_extra_defines onnxruntime_USE_COREML=OFF \
  --cmake_extra_defines CMAKE_POSITION_INDEPENDENT_CODE=ON \
  --cmake_extra_defines "CMAKE_CXX_FLAGS=-Wno-psabi" \
  $EXTRA_DEFINES \
  --update --build

# ORT produces ~80 static archives (10 onnxruntime_*.a + deps). Merge them into
# two linkable libs at fixed paths so CMake can reference them without globbing:
#   libonnxruntime_core.a  -- the 10 ORT internal libs; the rampart-onnx link
#                             --whole-archive's this so CPU provider/kernel
#                             static-init registrations are not dropped.
#   libonnxruntime_deps.a  -- abseil/onnx/protobuf-lite/re2/cpuinfo/...; normal-linked.
# Within each single merged archive ld resolves intra-archive circular refs, so
# no --start-group is needed.
REL="$BUILD_DIR/Release"
cd "$REL"
ORT_LIBS=$(find . -maxdepth 1 -name 'libonnxruntime_*.a' \
            ! -name 'libonnxruntime_core.a' ! -name 'libonnxruntime_deps.a' | sort)
DEP_LIBS=$(find . -name '*.a' | grep -v 'libonnxruntime_' | sort)
rm -f libonnxruntime_core.a libonnxruntime_deps.a
if [ "$(uname)" = "Darwin" ]; then
  # BSD ar has no -M (MRI scripts); Apple libtool -static flattens archive
  # members into one archive, which is exactly what the MRI ADDLIBs do.
  libtool -static -o libonnxruntime_core.a $ORT_LIBS
  libtool -static -o libonnxruntime_deps.a $DEP_LIBS
else
  { echo "CREATE libonnxruntime_core.a"; for a in $ORT_LIBS; do echo "ADDLIB $a"; done; echo SAVE; echo END; } | ar -M
  { echo "CREATE libonnxruntime_deps.a"; for a in $DEP_LIBS; do echo "ADDLIB $a"; done; echo SAVE; echo END; } | ar -M
fi
ranlib libonnxruntime_core.a libonnxruntime_deps.a
echo "rampart-build-cpu.sh: merged static archives -> $REL/libonnxruntime_{core,deps}.a"
