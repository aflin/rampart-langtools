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

# Vendored FetchContent sources: extern/onnxruntime-deps/ort/<name> holds
# the unpacked, PRE-PATCHED source tree for every dependency this build
# would otherwise download (see extern/onnxruntime-deps/MANIFEST.txt).
# FETCHCONTENT_SOURCE_DIR_<NAME> makes CMake use each tree directly (no
# download, no extract; patch steps are skipped, hence pre-patched trees).
# Build artifacts still land in the build dir as usual.
DEPS_SRC="$(cd .. && pwd)/onnxruntime-deps/ort"
DEPS_DEFINES=""
for d in ABSEIL_CPP:abseil_cpp DATE:date EIGEN3:eigen3 FLATBUFFERS:flatbuffers \
         GSL:gsl KLEIDIAI:kleidiai MP11:mp11 NLOHMANN_JSON:nlohmann_json \
         ONNX:onnx PROTOBUF:protobuf PYTORCH_CPUINFO:pytorch_cpuinfo \
         RE2:re2 SAFEINT:safeint; do
  DEPS_DEFINES="$DEPS_DEFINES --cmake_extra_defines FETCHCONTENT_SOURCE_DIR_${d%%:*}=$DEPS_SRC/${d#*:}"
done

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

# macOS: ORT downloads a prebuilt universal protoc for ALL Apple hosts
# (deps.txt protoc_mac_universal) via a deprecated FetchContent_Populate
# (CMP0169 dev warning) — and it was the one network fetch left after
# vendoring.  Build protoc from the vendored protobuf tree instead (same
# v21.12, so generated code matches exactly) and pass it via
# ONNX_CUSTOM_PROTOC_EXECUTABLE, which skips ORT's download block
# entirely.  Host-arch build, one-time per build dir (~1 min).
PROTOC_DEFINES=""
if [ "$(uname)" = "Darwin" ]; then
  PROTOC_BUILD="$BUILD_DIR/protoc-host"
  if [ ! -x "$PROTOC_BUILD/protoc" ]; then
    cmake -S "$DEPS_SRC/protobuf" -B "$PROTOC_BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_SHARED_LIBS=OFF \
      -Dprotobuf_WITH_ZLIB=OFF \
      -DCMAKE_C_FLAGS=-w -DCMAKE_CXX_FLAGS=-w
    cmake --build "$PROTOC_BUILD" -j 8 --target protoc
  fi
  [ -x "$PROTOC_BUILD/protoc" ] || \
    { echo "rampart-build-cpu.sh: vendored protoc build failed" >&2; exit 1; }
  PROTOC_DEFINES="--cmake_extra_defines ONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC_BUILD/protoc"
fi

# ELF targets: assemble the MLAS .S files with an explicit (non-exec)
# .note.GNU-stack section; without it ld warns "missing .note.GNU-stack
# section implies executable stack" when the static archives are linked
# into rampart-onnx.so. Mach-O has no such note, so skip on macOS.
ASM_DEFINES=""
if [ "$(uname)" != "Darwin" ]; then
  ASM_DEFINES="--cmake_extra_defines CMAKE_ASM_FLAGS=-Wa,--noexecstack"
fi

# ONNX_CPU_PARALLEL: build parallelism (default 8).  Lower it on small-RAM
# hosts (e.g. 6 on a 7 GB aarch64 VM) -- some ORT translation units are
# memory-hungry and 8 concurrent gcc's can OOM.
# -w: vendored code -- silence its warnings (noise at our -Wall, not ours
# to fix); -Wno-psabi kept for compilers where -w doesn't cover the note.
# CMP0169=OLD: ORT's own cmake still calls the deprecated
# FetchContent_Populate() for mp11/safeint, which cmake >= 3.30 flags with
# a dev warning on every configure; the policy default quiets exactly that
# (no behavior change -- the vendored FETCHCONTENT_SOURCE_DIR trees are
# used either way).
# --allow_running_as_root: inert for normal builds; needed when uid==0
# (containers, unshare -r offline verification) — build.py refuses
# otherwise.  Linux-only: build.py registers the flag in
# add_linux_specific_args(), macOS argparse rejects it.
ROOT_FLAG=""
if [ "$(uname)" = "Linux" ]; then
  ROOT_FLAG="--allow_running_as_root"
fi
# Apple Silicon has no SVE; build.py defaults it ON for arm64 and ORT's
# configure then warns "USE_SVE ... not supported ... will be disabled".
# Pass --no_sve to state the truth up front (same result, no warning).
SVE_FLAG=""
if [ "$(uname)" = "Darwin" ]; then
  SVE_FLAG="--no_sve"
fi
./build.sh \
  --build_dir "$BUILD_DIR" \
  $ROOT_FLAG \
  $SVE_FLAG \
  --config Release --parallel "${ONNX_CPU_PARALLEL:-8}" --compile_no_warning_as_error \
  --skip_tests --skip_submodule_sync \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines onnxruntime_USE_XNNPACK=OFF \
  --cmake_extra_defines onnxruntime_USE_COREML=OFF \
  --cmake_extra_defines CMAKE_POSITION_INDEPENDENT_CODE=ON \
  --cmake_extra_defines "CMAKE_CXX_FLAGS=-Wno-psabi -w" \
  --cmake_extra_defines "CMAKE_C_FLAGS=-w" \
  --cmake_extra_defines CMAKE_POLICY_DEFAULT_CMP0169=OLD \
  $DEPS_DEFINES \
  $PROTOC_DEFINES \
  $ASM_DEFINES \
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
  # The "duplicate member name" warnings are benign — different TUs that
  # share a basename (e.g. cpu/ and contrib_ops/ both have activations.cc);
  # both members are kept and link fine — so filter exactly that message
  # (via a temp file so libtool's own exit status is preserved under set -e).
  libtool -static -o libonnxruntime_core.a $ORT_LIBS 2> libtool.err || { cat libtool.err >&2; exit 1; }
  grep -v "warning duplicate member name" libtool.err >&2 || true
  libtool -static -o libonnxruntime_deps.a $DEP_LIBS 2> libtool.err || { cat libtool.err >&2; exit 1; }
  grep -v "warning duplicate member name" libtool.err >&2 || true
  rm -f libtool.err
else
  { echo "CREATE libonnxruntime_core.a"; for a in $ORT_LIBS; do echo "ADDLIB $a"; done; echo SAVE; echo END; } | ar -M
  { echo "CREATE libonnxruntime_deps.a"; for a in $DEP_LIBS; do echo "ADDLIB $a"; done; echo SAVE; echo END; } | ar -M
fi
ranlib libonnxruntime_core.a libonnxruntime_deps.a
echo "rampart-build-cpu.sh: merged static archives -> $REL/libonnxruntime_{core,deps}.a"
