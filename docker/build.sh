#!/bin/sh
# build.sh <stage> [variant] -- build the rampart-langtools modules (FAISS,
#                  llama.cpp, sentencepiece) in a manylinux "oven", CPU-only
#                  or with CUDA for GPU.
#
#   build.sh build [variant]       # grab libraries + compile -> build/oven[-variant]/
#   build.sh install [variant]     # install the module(s) into <prefix>/modules
#   build.sh shell [variant]       # interactive shell in the matching oven
#   build.sh save-image [variant]  # persist the oven image to a .tar.gz
#
#   variant:   (none)/cpu  CPU only    glibc 2.17  -> rampart-2_17
#              cpu_2_28     CPU only    glibc 2.28  -> rampart-2_28  (newer SIMD)
#              cu11         CUDA 11.8   glibc 2.17 (x86) / 2.28 (ARM)
#              cu12         CUDA 12.8   glibc 2.28  -> rampart-2_28
#              cu13         CUDA 13.0   glibc 2.28  -> rampart-2_28
#   CUDA 12+ needs glibc >= 2.28, so cu11 is the only CUDA build for the 2.17 tier.
#
#   Flags:
#      --rebuild-image [build [variant]]   # force a fresh oven image first
#      -d <dir>                            # install into <dir>; if <dir> isn't the
#                                          #   build's home tier, copy it in (graft)
#
# What it touches:
#   build      -> build/oven[-variant]/
#   install    -> adds the module(s) to <prefix>/modules (+ test/llamacpp-test.js).
#                 Default <prefix> matches the build's glibc tier: /usr/local/rampart-2_17,
#                 or (cu12/cu13, and cu11 on ARM) /usr/local/rampart-2_28.
#                 To put ONE build in BOTH tiers (e.g. cu11), install it, then
#                 `install -d <other-tier>` -- the modules are copied in, no rebuild.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PREFIX_DIR=""; [ "${1:-}" = "-d" ] && { PREFIX_DIR="$2"; shift 2; }

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  CUDA_ARCHDIR=x86_64 ;;
    aarch64) CUDA_ARCHDIR=sbsa ;;
    *)       CUDA_ARCHDIR=$ARCH ;;
esac

BASE2014=rampart-langtools-oven           # manylinux2014, glibc 2.17 (cpu + cu11 base)
BASE2_28=rampart-langtools-oven-2_28      # manylinux_2_28, glibc 2.28 (cu12/cu13 base)

# cuda_cfg <variant> -> sets CU_IMG CU_BASE CU_BASEDF CU_DISTRO CU_PKGS (returns 1 if not a cuda variant)
cuda_cfg() {
    case "$1" in
        cu11) if [ "$ARCH" = x86_64 ]; then
                  CU_BASE=$BASE2014; CU_BASEDF=Dockerfile;      CU_DISTRO=rhel7   # x86: keep glibc 2.17
              else
                  CU_BASE=$BASE2_28; CU_BASEDF=Dockerfile.2_28; CU_DISTRO=rhel8   # arm: no rhel7/sbsa repo -> rhel8
              fi
              CU_PKGS="cuda-toolkit-11-8 cuda-driver-devel-11-8" ;;
        cu12) CU_BASE=$BASE2_28; CU_BASEDF=Dockerfile.2_28; CU_DISTRO=rhel8
              CU_PKGS="cuda-toolkit-12-8 cuda-driver-devel-12-8" ;;
        cu13) CU_BASE=$BASE2_28; CU_BASEDF=Dockerfile.2_28; CU_DISTRO=rhel8
              CU_PKGS="cuda-toolkit-13-0 cuda-driver-devel-13-0" ;;
        *)    return 1 ;;
    esac
    CU_IMG="rampart-langtools-oven-$1"
}

# docker image used to build/install a variant
image_for() {
    if cuda_cfg "$1" >/dev/null 2>&1; then echo "$CU_IMG"
    elif [ "$1" = cpu_2_28 ]; then echo "$BASE2_28"
    else echo "$BASE2014"; fi
}

save_image() {  # $1 image, $2 tar
    mkdir -p "$(dirname "$2")"
    echo "==> persisting image $1 to $2"
    docker save "$1" | gzip > "$2"
}

# build a base oven ($1 image, $2 dockerfile) if missing (or always, when forced)
build_base() {
    echo "==> building base oven '$1' ($2)…"
    docker build --build-arg ARCH="$ARCH" -t "$1" -f "$HERE/$2" "$HERE"
}
ensure_base() {  # $1 image, $2 dockerfile
    docker image inspect "$1" >/dev/null 2>&1 && { echo "==> using existing base '$1'"; return; }
    tar="$REPO/build/$1.image.tar.gz"
    [ -f "$tar" ] && { echo "==> restoring '$1' from $tar"; docker load -i "$tar" && return; }
    build_base "$1" "$2"
}

# build a cuda image for a variant (ensures its base first)
build_cuda() {  # $1 variant
    cuda_cfg "$1"
    ensure_base "$CU_BASE" "$CU_BASEDF"
    echo "==> building cuda oven '$CU_IMG' ($1: $CU_PKGS on $CU_BASE)…"
    docker build \
        --build-arg BASE_IMAGE="$CU_BASE" \
        --build-arg CUDA_DISTRO="$CU_DISTRO" \
        --build-arg CUDA_ARCHDIR="$CUDA_ARCHDIR" \
        --build-arg CUDA_PKGS="$CU_PKGS" \
        -t "$CU_IMG" -f "$HERE/Dockerfile.cuda" "$HERE"
}
ensure_cuda() {  # $1 variant
    cuda_cfg "$1"
    docker image inspect "$CU_IMG" >/dev/null 2>&1 && { echo "==> using existing cuda oven '$CU_IMG'"; return; }
    tar="$REPO/build/$CU_IMG.image.tar.gz"
    [ -f "$tar" ] && { echo "==> restoring '$CU_IMG' from $tar"; docker load -i "$tar" && return; }
    build_cuda "$1"
}

# ensure whatever image a variant needs
ensure_image() {
    if cuda_cfg "$1" >/dev/null 2>&1; then ensure_cuda "$1"
    elif [ "$1" = cpu_2_28 ]; then ensure_base "$BASE2_28" Dockerfile.2_28
    else ensure_base "$BASE2014" Dockerfile; fi
}

if [ "$1" = "--rebuild-image" ]; then
    shift
    v="${2:-}"   # after shift: $1=stage, $2=variant
    if cuda_cfg "$v" >/dev/null 2>&1; then build_cuda "$v"
    elif [ "$v" = cpu_2_28 ]; then build_base "$BASE2_28" Dockerfile.2_28
    else build_base "$BASE2014" Dockerfile; fi
fi

require_rampart() {
    [ -x "$PREFIX_DIR/bin/rampart" ] || {
        echo "missing $PREFIX_DIR/bin/rampart -- install the rampart-ml base first" >&2
        exit 1; }
}

do_build() {
    variant="$1"
    ensure_image "$variant"
    require_rampart
    echo "==> [langtools build:${variant:-default}] compiling into build/oven${variant:+-$variant}/…"
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e RAMPART_PREFIX="$PREFIX_DIR" \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
        -v "$REPO:/lt" -w /lt \
        -v "$PREFIX_DIR":"$PREFIX_DIR":ro \
        "$(image_for "$variant")" /lt/docker/build-in-oven.sh build "$variant"
}

do_install() {
    variant="$1"
    ensure_image "$variant"
    require_rampart
    [ -d "$REPO/build/oven${variant:+-$variant}" ] || {
        echo "no ${variant:-default} build -- run 'docker/build.sh build $variant' first" >&2; exit 1; }
    echo "==> [langtools install:${variant:-default}] installing modules into $PREFIX_DIR/modules…"
    docker run --rm \
        -e HOME=/tmp -e RAMPART_PREFIX="$PREFIX_DIR" \
        -v "$REPO:/lt" -w /lt \
        -v "$PREFIX_DIR":"$PREFIX_DIR" \
        "$(image_for "$variant")" /lt/docker/build-in-oven.sh install "$variant"
}

STAGE="${1:-}"
VARIANT="${2:-}"
if [ -z "$PREFIX_DIR" ]; then
    if { cuda_cfg "$VARIANT" >/dev/null 2>&1 && [ "$CU_BASE" = "$BASE2_28" ]; } || [ "$VARIANT" = cpu_2_28 ]; then
        PREFIX_DIR=/usr/local/rampart-2_28
    else
        PREFIX_DIR=/usr/local/rampart-2_17
    fi
fi
case "$STAGE" in
    build)   do_build "$VARIANT" ;;
    install) do_install "$VARIANT" ;;
    save-image)
        img=$(image_for "$VARIANT")
        docker image inspect "$img" >/dev/null 2>&1 || {
            echo "image '$img' not built yet -- build it first" >&2; exit 1; }
        save_image "$img" "$REPO/build/$img.image.tar.gz" ;;
    shell)
        ensure_image "$VARIANT"
        exec docker run --rm -it -e HOME=/tmp -e RAMPART_PREFIX="$PREFIX_DIR" \
            -v "$REPO:/lt" -w /lt -v "$PREFIX_DIR":"$PREFIX_DIR":ro \
            "$(image_for "$VARIANT")" /bin/bash ;;
    ""|-h|--help)
        sed -n '2,/^set -e/{/^set -e/!p}' "$0" | sed 's/^# \{0,1\}//' ;;
    *)
        echo "unknown stage: $STAGE  (build | install | save-image | shell) [cpu|cpu_2_28|cu11|cu12|cu13]" >&2
        exit 1 ;;
esac
