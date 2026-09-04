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
# cpu_2_28 / cu11_2_28 = the cpu / cu11 flavor on the 2_28 base: distinct build dir +
# tier, but the SAME _cpu / _cu11 module name (tier is implied by the install prefix,
# like cu12/cu13).
case "$VARIANT" in cpu_2_28) SUFFIX=cpu ;; cu11_2_28) SUFFIX=cu11 ;; *) SUFFIX=$VARIANT ;; esac
LT=/lt
BUILD="$LT/build/oven${VARIANT:+-$VARIANT}"
PREFIX="${RAMPART_PREFIX:-/usr/local/rampart-ml}"

export PATH="$PREFIX/bin:$PATH"   # so the CMake's `rampart -c process.installPath` resolves

is_cuda() { case "$VARIANT" in cu11|cu11_2_28|cu12|cu13) return 0 ;; *) return 1 ;; esac; }

# CMAKE_CUDA_ARCHITECTURES per variant + host arch (real SASS + a top -virtual PTX
# for forward-compat).  Edit freely -- these are sensible defaults, not gospel.
cuda_arches() {
    m=$(uname -m)
    case "$VARIANT" in
      cu11|cu11_2_28) [ "$m" = aarch64 ] && echo "72-real;87-real;87-virtual" \
                                || echo "70-real;75-real;80-real;86-real;89-real;89-virtual" ;;
      cu12) [ "$m" = aarch64 ] && echo "87-real;90-real;100-real;120-real;120-virtual" \
                                || echo "80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ;;
      # NB: cu12 stops at sm_120 -- GB10/sm_121 needs the family-specific sm_121a
      # kernel ggml forces, and CUDA 12.8's nvcc has compute_121 but NOT compute_121a
      # (only CUDA 13 does).  So GB10/Spark uses the cu13 module, not cu12.
      cu13) [ "$m" = aarch64 ] && echo "87-real;90-real;100-real;110-real;120-real;121-real;121-virtual" \
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
      cu11|cu11_2_28)     enable_toolchain 11 ;;
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
        # -DGGML_CUDA_NO_VMM=ON: ggml's VMM pool reserves 32GB of virtual address
        # space per llama_context (cuMemAddressReserve).  Tegra's VA aperture fits
        # only two, so the third model load aborts with "CUDA error: out of memory"
        # -- address space, not physical memory.  extern/extern.cmake sets this too
        # (see the long note there); passed here as well so the oven's own flags
        # record it.
        GPU_FLAGS="-DLT_ENABLE_GPU=1 -DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX \
                   -DCMAKE_CUDA_ARCHITECTURES=$arches \
                   -DGGML_CUDA_NO_VMM=ON \
                   -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    fi

    # rampart-onnx tiering: ORT 1.27 needs glibc >= 2.28 + a newer gcc than the
    # 2_17 (manylinux2014) ovens have, and cu11 tiers don't ship an onnx runtime
    # (ORT's CUDA EP needs CUDA >= 12).  So: 2_17-tier + cu11 variants build all
    # modules EXCEPT onnx; cpu_2_28 builds the unified rampart-onnx.so; cu12/cu13
    # build only their modules/onnx-cuNN/ runtime dirs.
    ONNX_GATE=""
    case "$VARIANT" in
      ""|cpu|cu11|cu11_2_28) ONNX_GATE="-DLT_ONNX=0" ;;
    esac

    mkdir -p "$BUILD"
    # Optional onnx knobs forwarded from the host env (build.sh passes them with -e):
    #   ONNX_CUDA_PARALLEL -- ORT CUDA build --parallel (memory ~ parallel x arches;
    #     raise on a big-RAM builder, e.g. 44 on the 512GB box; default 1 in extern.cmake).
    #   ONNX_CUDA_ARCH     -- override the ORT CUDA arch list.
    ONNX_FLAGS=""
    [ -n "${ONNX_CUDA_PARALLEL:-}" ] && ONNX_FLAGS="$ONNX_FLAGS -DONNX_CUDA_PARALLEL=$ONNX_CUDA_PARALLEL"
    [ -n "${ONNX_CUDA_ARCH:-}" ]     && ONNX_FLAGS="$ONNX_FLAGS -DONNX_CUDA_ARCH=$ONNX_CUDA_ARCH"
    # ONNX_CUDA_MINIMAL=1: cuBLAS-only CUDA EP, no cuDNN at build or run time
    # (embed/rerank matmuls still on GPU; conv-family ops fall back). For ovens
    # without cuDNN and cuDNN-less deploy targets (e.g. Jetson without JetPack's
    # cuDNN installed).
    [ -n "${ONNX_CUDA_MINIMAL:-}" ]  && ONNX_FLAGS="$ONNX_FLAGS -DONNX_CUDA_MINIMAL=$ONNX_CUDA_MINIMAL"
    # -DSUFFIX="" (empty variant) -> unsuffixed; cpu/cuNN -> _cpu/_cuNN.
    # Pass RP_PATH (rampart's install prefix) directly instead of letting CMake run the
    # mounted rampart to query it: this oven's glibc can be OLDER than the host's, and a
    # host-built rampart then fails to execute inside the container ("libc.so.6: version
    # GLIBC_2.xx not found").  $PREFIX is exactly rampart's installPath, so hand it over.
    # (RAMPART_EXECUTABLE is still pinned for any tooling that wants it, but is no longer
    # executed for the install-path query.)
    "$CMAKE" -S "$LT" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRP_PATH="$PREFIX" \
        -DRAMPART_EXECUTABLE="$PREFIX/bin/rampart" \
        -DSUFFIX="$SUFFIX" $GPU_FLAGS $ONNX_FLAGS $ONNX_GATE
    # LT_TARGET: optionally build ONE cmake target (e.g. onnxruntime_ep to get
    # just a flavor's ORT runtime dir without rebuilding llamacpp/faiss).
    # Parallelism for the MAIN build = all cores by default; override with
    # LT_BUILD_PARALLEL (forwarded by build.sh) to raise/cap it -- e.g. when nproc
    # under-reports, or to leave headroom.  (The ORT sub-build has its OWN knobs:
    # ONNX_CPU_PARALLEL, default 8; ONNX_CUDA_PARALLEL, default 1 -- see extern.cmake.)
    "$CMAKE" --build "$BUILD" -j"${LT_BUILD_PARALLEL:-$(nproc)}" ${LT_TARGET:+--target "$LT_TARGET"}
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
        # the module .so's (stripped) + the langtools test scripts + rampart-models.js.
        echo "==> grafting ${VARIANT:-default} build (configured for '${cfg:-?}') into $PREFIX"
        mkdir -p "$PREFIX/modules"
        # exclude rampart-langtools*.so -- it's built but intentionally not shipped
        # (covered by llamacpp/faiss/sentencepiece), matching the CMake install rules.
        sos=$(ls "$BUILD"/rampart-*"${SUFFIX:+_$SUFFIX}".so "$BUILD"/rampart-sentencepiece.so 2>/dev/null | grep -v '/rampart-langtools' | sort -u) || true
        for so in $sos; do
            install -m 755 "$so" "$PREFIX/modules/"
            strip -S "$PREFIX/modules/$(basename "$so")" 2>/dev/null || true
        done
        # test scripts (mirror the CMake install(FILES ... test) list, so a graft
        # install matches a home-tier `cmake --install`)
        mkdir -p "$PREFIX/test"
        for t in llamacpp-test.js faiss-test.js clip-test.js; do
            [ -f "$LT/$t" ] && install -m 644 "$LT/$t" "$PREFIX/test/"
        done
        # clip-test.js's photos (mirror the CMake install(DIRECTORY test_images ...))
        [ -d "$LT/test_images" ] && { rm -rf "$PREFIX/test/test_images"; cp -R "$LT/test_images" "$PREFIX/test/"; }
        # the script only -- rampart-models-catalog.json is fetched+cached at
        # runtime and must not be shipped (see the CMake install rule)
        [ -f "$LT/rampart-models.js" ] && install -m 644 "$LT/rampart-models.js" "$PREFIX/modules/"
        # unified rampart-onnx: the module is UNSUFFIXED (one .so for cpu+gpu)...
        if [ -f "$BUILD/rampart-onnx.so" ]; then
            install -m 755 "$BUILD/rampart-onnx.so" "$PREFIX/modules/"
            strip -S "$PREFIX/modules/rampart-onnx.so" 2>/dev/null || true
        fi
        # ...and a cuNN build produces a runtime DIR instead of a module
        if [ -n "$SUFFIX" ] && [ -d "$BUILD/extern/onnxruntime/Release" ] && \
           [ -f "$BUILD/extern/onnxruntime/Release/libonnxruntime.so.1.27.0" ]; then
            RTD="$PREFIX/modules/onnx-$SUFFIX"
            mkdir -p "$RTD"
            cp -P "$BUILD/extern/onnxruntime/Release/libonnxruntime.so"* "$RTD/" 2>/dev/null || true
            cp "$BUILD/extern/onnxruntime/Release/libonnxruntime_providers_shared.so" \
               "$BUILD/extern/onnxruntime/Release/libonnxruntime_providers_cuda.so" "$RTD/" 2>/dev/null || true
        fi
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
