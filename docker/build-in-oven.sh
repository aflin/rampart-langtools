#!/bin/bash
# rampart-langtools/docker/build-in-oven.sh <stage> [variant] -- runs INSIDE the
# langtools oven.  Invoke via docker/build.sh, not directly.
#
# stage   = build | install
# variant = (none) -> rampart-langtools.so       (cpu, unsuffixed, no CUDA)
#           cpu     -> rampart-langtools_cpu.so   (cpu, no CUDA)
#           cu11    -> rampart-langtools_cu11.so  (CUDA 11.8 image)
#           cu12    -> rampart-langtools_cu12.so  (CUDA 12.8 image)
#           cu13    -> rampart-langtools_cu13.so  (CUDA 13.0 image)
#
# Builds against the bind-mounted /usr/local/rampart-ml (headers + the rampart
# binary used to resolve $RP_PATH at configure time).
set -euo pipefail

STAGE="${1:-build}"
VARIANT="${2:-}"
# cpu_2_28 = the cpu flavor on the 2_28 base: distinct build dir + tier, but the
# SAME _cpu module name (tier is implied by the install prefix, like cu12/cu13).
case "$VARIANT" in cpu_2_28) SUFFIX=cpu ;; *) SUFFIX=$VARIANT ;; esac
LT=/lt
BUILD="$LT/build/oven${VARIANT:+-$VARIANT}"
PREFIX="${RAMPART_PREFIX:-/usr/local/rampart-ml}"

export PATH="$PREFIX/bin:$PATH"   # so the CMake's `rampart -c process.installPath` resolves

is_cuda() { case "$VARIANT" in cu11|cu12|cu13) return 0 ;; *) return 1 ;; esac; }

# CMAKE_CUDA_ARCHITECTURES per variant + host arch (real SASS + a top -virtual PTX
# for forward-compat).  Edit freely -- these are sensible defaults, not gospel.
cuda_arches() {
    m=$(uname -m)
    case "$VARIANT" in
      cu11) [ "$m" = aarch64 ] && echo "72-real;87-real;87-virtual" \
                                || echo "70-real;75-real;80-real;86-real;89-real;89-virtual" ;;
      cu12) [ "$m" = aarch64 ] && echo "87-real;90-real;100-real;120-real;120-virtual" \
                                || echo "80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ;;
      cu13) [ "$m" = aarch64 ] && echo "87-real;90-real;100-real;110-real;120-real;120-virtual" \
                                || echo "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ;;
    esac
}

enable_toolchain() {  # optional $1 = preferred gcc major (e.g. 11 for CUDA 11.8)
    set +u
    sc=""
    # devtoolset (manylinux2014) OR gcc-toolset (manylinux_2_28).
    if [ -n "${1:-}" ]; then
        sc=$(ls /opt/rh/gcc-toolset-$1/enable /opt/rh/devtoolset-$1/enable 2>/dev/null | head -1) || true
    fi
    [ -z "$sc" ] && { sc=$(ls /opt/rh/gcc-toolset-*/enable /opt/rh/devtoolset-*/enable 2>/dev/null | sort -V | tail -1) || true; }
    [ -n "$sc" ] && source "$sc"
    set -u
}
cmake_bin() { command -v cmake || command -v cmake3; }

case "$STAGE" in
  build)
    # Pin host gcc: cu11->11 (CUDA 11.8 ceiling); cu12/cu13/cpu_2_28->13 (proven on
    # the 2_28 oven; its default gcc-14 is unneeded here); else newest available.
    case "$VARIANT" in
      cu11)               enable_toolchain 11 ;;
      cu12|cu13|cpu_2_28) enable_toolchain 13 ;;
      *)                  enable_toolchain ;;
    esac
    echo "==> toolchain: $(gcc --version | head -1)"
    command -v rampart >/dev/null || { echo "rampart not on PATH (mount /usr/local/rampart-ml)" >&2; exit 1; }
    git config --global --add safe.directory '*' 2>/dev/null || true
    CMAKE=$(cmake_bin)

    GPU_FLAGS=""
    if is_cuda; then
        export PATH="/usr/local/cuda/bin:$PATH"
        command -v nvcc >/dev/null 2>&1 || {
            echo "$VARIANT requested but nvcc not found -- is the CUDA layer in this image?" >&2; exit 1; }
        export CUDACXX="$(command -v nvcc)"
        export CUDAHOSTCXX="$(command -v g++)"
        arches="$(cuda_arches)"
        echo "==> nvcc: $(nvcc --version | grep -i release)  host-cxx: $CUDAHOSTCXX  arch: $arches"
        # -allow-unsupported-compiler: Debian/AlmaLinux host gcc can be newer than
        # the CUDA version officially allows; this lets the build proceed (drop it
        # if you want nvcc to enforce its gcc ceiling).
        GPU_FLAGS="-DLT_ENABLE_GPU=1 -DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX \
                   -DCMAKE_CUDA_ARCHITECTURES=$arches \
                   -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    fi

    mkdir -p "$BUILD"
    # -DSUFFIX="" (empty variant) -> unsuffixed; cpu/cuNN -> _cpu/_cuNN.
    # Pin RAMPART_EXECUTABLE to the mounted path; otherwise find_program's cached
    # value in an existing build dir can point at a stale (unmounted) rampart path
    # and the installPath query fails.  -D overrides the cache on reconfigure.
    "$CMAKE" -S "$LT" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRAMPART_EXECUTABLE="$PREFIX/bin/rampart" \
        -DSUFFIX="$SUFFIX" $GPU_FLAGS
    "$CMAKE" --build "$BUILD" -j"$(nproc)"
    # record the install dir baked into this build (= rampart's installPath at
    # configure); the install stage verifies it matches before installing.
    printf '%s\n' "$PREFIX" > "$BUILD/.rampart-prefix"

    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
        chown -R "${HOST_UID}:${HOST_GID}" "$BUILD"
    fi
    echo
    ls -l "$BUILD"/rampart-*"${SUFFIX:+_$SUFFIX}".so 2>/dev/null || true
    echo "==> langtools ${VARIANT:-default} build OK"
    ;;

  install)
    enable_toolchain
    CMAKE=$(cmake_bin)
    [ -d "$BUILD" ] || { echo "no ${VARIANT:-default} build at $BUILD -- run 'docker/build.sh build $VARIANT' first" >&2; exit 1; }
    cfg=$(cat "$BUILD/.rampart-prefix" 2>/dev/null || true)
    if [ "$cfg" = "$PREFIX" ]; then
        # home tier: cmake's baked DESTINATION is $PREFIX/modules -> normal install.
        "$CMAKE" --install "$BUILD"
    else
        # graft: this build was configured for a different tree ($cfg), e.g. installing
        # the 2_17-built cu11 into the 2_28 tree.  The modules are relocatable (they
        # resolve rampart's libs via $ORIGIN/../lib at load time), so COPY them into
        # $PREFIX instead of rebuilding/re-baking.  Mirrors the normal install's files:
        # the module .so (stripped) + llamacpp-test.js.
        echo "==> grafting ${VARIANT:-default} build (configured for '${cfg:-?}') into $PREFIX"
        mkdir -p "$PREFIX/modules"
        sos=$(ls "$BUILD"/rampart-*"${SUFFIX:+_$SUFFIX}".so "$BUILD"/rampart-sentencepiece.so 2>/dev/null | sort -u) || true
        for so in $sos; do
            install -m 755 "$so" "$PREFIX/modules/"
            strip -S "$PREFIX/modules/$(basename "$so")" 2>/dev/null || true
        done
        [ -f "$LT/llamacpp-test.js" ] && { mkdir -p "$PREFIX/test"; install -m 644 "$LT/llamacpp-test.js" "$PREFIX/test/"; }
    fi
    echo
    ls -l "$PREFIX"/modules/rampart-*"${SUFFIX:+_$SUFFIX}".so 2>/dev/null || true
    { [ -n "$VARIANT" ] && ls -l "$PREFIX"/modules/rampart-sentencepiece.so 2>/dev/null; } || true
    echo "==> langtools ${VARIANT:-default} install OK"
    ;;

  *)
    echo "unknown stage: $STAGE  (expected: build | install)" >&2
    exit 1
    ;;
esac
