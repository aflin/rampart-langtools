# rampart-langtools "oven" -- builds the langtools modules (full FAISS,
# llama.cpp, sentencepiece) for the portable centos7-x86_64 target (glibc 2.17).
# This is the cpu/default image.  The cuda variant uses a SEPARATE image
# (docker/Dockerfile.cuda, FROM this one + the CUDA toolkit), so a plain
# `build.sh build` / `build.sh build cpu` never downloads the ~5GB CUDA toolkit.
#
# Base = manylinux2014 (glibc 2.17) + devtoolset-11 (gcc 11: AVX-512, C++17) +
# OpenBLAS from source (static libopenblas.a for full FAISS; glibc <= 2.17).
# sentencepiece bundles its own protobuf + absl (SPM_*_PROVIDER=internal), so
# no system protobuf is needed.
#
# Self-contained on purpose (not FROM rampart-manylinux2014): this builds its
# own OpenBLAS ONCE at image-build time, then the image is persisted to disk
# (docker save) and reused -- so an ordinary langtools change never rebuilds it.
#
# rampart headers + the rampart binary are NOT baked in: build.sh bind-mounts
# /usr/local/rampart (the installed centos7 rampart), and the build reads
# $RP_PATH from `rampart -c "console.log(process.installPath)"`.
#
# Built automatically by docker/build.sh; or: docker build -t <tag> docker/
# Target arch: x86_64 (default) or aarch64.  build.sh passes --build-arg ARCH=$(uname -m).
ARG ARCH=x86_64
FROM quay.io/pypa/manylinux2014_${ARCH}

# devtoolset's enable script is a bash-ism; make every RUN a bash shell.
SHELL ["/bin/bash", "-c"]

# Newer GCC (devtoolset-11) -- gcc 11 for AVX-512 SimSIMD/ggml kernels + C++17;
# glibc stays 2.17 (devtoolset links the base glibc, statically links newer
# libstdc++ symbols).  Same toolchain as the core rampart oven.
ARG ARCH=x86_64
RUN if [ "$ARCH" = "x86_64" ]; then \
        yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++ \
            devtoolset-11-gcc-gfortran devtoolset-11-binutils && yum clean all; \
    else \
        echo "==> $ARCH: CentOS 7 has no devtoolset-11 (and no AVX-512 on arm); using the image's bundled devtoolset-10 (gcc/g++/gfortran 10.2)"; \
    fi

# OpenBLAS from source (static + shared, glibc <= 2.17) -- full FAISS links the
# static libopenblas.a; DYNAMIC_ARCH=1 gives runtime CPU dispatch.
ARG OPENBLAS_VERSION=0.3.21
RUN source "$(ls /opt/rh/devtoolset-*/enable | sort -V | tail -1)" && \
    curl -fsSL -o /tmp/openblas.tar.gz \
      "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" && \
    tar -xzf /tmp/openblas.tar.gz -C /tmp && \
    make -C "/tmp/OpenBLAS-${OPENBLAS_VERSION}" -j"$(nproc)" \
        DYNAMIC_ARCH=1 NUM_THREADS=64 USE_OPENMP=1 && \
    make -C "/tmp/OpenBLAS-${OPENBLAS_VERSION}" PREFIX=/usr/local install && \
    rm -rf /tmp/OpenBLAS-* /tmp/openblas.tar.gz && \
    echo /usr/local/lib > /etc/ld.so.conf.d/openblas.conf && ldconfig

# CUDA is intentionally NOT here -- it lives in docker/Dockerfile.cuda (FROM this
# image), so cpu/default builds never pull it.  `build.sh build cuda` builds that
# image on first use.

# No ENTRYPOINT: docker/build.sh invokes docker/build-in-oven.sh explicitly.
