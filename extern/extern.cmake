set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)

# LLAMA.CPP
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

# Build ggml WITHOUT OpenMP so it uses its own pthread threadpool. We run llama
# inference on a dedicated worker thread (the generation engine); macOS libomp
# misbehaves ("thread identifier invalid" / kmp assertion / Metal libdispatch
# crash) when a non-initial thread runs OpenMP parallel regions while the JS
# event-loop thread is concurrently active. ggml's native threadpool is what
# llama-server uses and is safe here. (libomp stays linked for faiss.)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

if(NOT APPLE)
  set(GGML_CUDA ${LT_ENABLE_GPU} CACHE BOOL "" FORCE)
endif()

# Portable ARM build.  ggml defaults to GGML_NATIVE=ON -> -mcpu=native, which bakes
# the BUILD host's CPU features into the .so.  Run that on a different/older ARM
# (e.g. a Raspberry Pi) and it SIGILLs ("Illegal instruction") the instant a kernel
# uses a missing feature.  So on ARM, pin an explicit -march instead of native.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  if(NOT APPLE)
    # Linux ARM (the manylinux ovens): ALWAYS turn off native and pin a portable -march.
    # docker/build.sh sets LT_ARM_ARCH=armv8-a (baseline) for BOTH glibc tiers -- the
    # tier (2_17/2_28) is a glibc floor, NOT an ISA floor, so ggml must run on any armv8
    # (Pi 3/4 .. Orin) like the rest of the build.  Default to armv8-a if unset.  (Modern
    # ARM SIMD = ggml runtime dispatch, a separate axis -- not a raised -march here.)
    if(DEFINED ENV{LT_ARM_ARCH} AND NOT "$ENV{LT_ARM_ARCH}" STREQUAL "")
      set(LT_ARM_ARCH "$ENV{LT_ARM_ARCH}")
    else()
      set(LT_ARM_ARCH "armv8-a")
    endif()
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "${LT_ARM_ARCH}" CACHE STRING "" FORCE)
    message(STATUS "rampart-langtools: portable ARM build -- GGML_NATIVE off, -march=${LT_ARM_ARCH}")
  else()
    # macOS: bare-metal -mcpu=native is optimal (gets i8mm), but a virtualized guest
    # (e.g. macOS in UTM) advertises i8mm it can't execute; ggml then emits +noi8mm
    # while clang still defines __ARM_FEATURE_MATMUL_INT8 -> "vmmlaq_s32 requires
    # target feature 'i8mm'".  So only override (off + explicit -march) when i8mm is
    # actually missing; otherwise leave GGML_NATIVE on.
    execute_process(
      COMMAND sysctl -n hw.optional.arm.FEAT_I8MM
      OUTPUT_VARIABLE LT_FEAT_I8MM OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT LT_FEAT_I8MM STREQUAL "1")
      set(LT_ARM_ARCH "armv8.2-a")
      execute_process(COMMAND sysctl -n hw.optional.arm.FEAT_DotProd
        OUTPUT_VARIABLE LT_FEAT_DOTPROD OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      execute_process(COMMAND sysctl -n hw.optional.arm.FEAT_FP16
        OUTPUT_VARIABLE LT_FEAT_FP16 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(LT_FEAT_DOTPROD STREQUAL "1")
        string(APPEND LT_ARM_ARCH "+dotprod")
      endif()
      if(LT_FEAT_FP16 STREQUAL "1")
        string(APPEND LT_ARM_ARCH "+fp16")
      endif()
      set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
      set(GGML_CPU_ARM_ARCH "${LT_ARM_ARCH}" CACHE STRING "" FORCE)
      message(STATUS "rampart-langtools: i8mm not available (virtualized CPU?); "
        "disabling GGML_NATIVE, using -march=${LT_ARM_ARCH}")
    endif()
  endif()
endif()

# When llama.cpp is built as a subproject, its `common` subdirectory is OFF by
# default (it defaults to LLAMA_STANDALONE, which is OFF here). But we link
# `llama-common`/`llama-common-base`, so force it on. TOOLS/SERVER/EXAMPLES/TESTS
# stay off (subproject defaults) — we don't build them (we used to need tools/mtmd
# for an old vision experiment; that's gone, image vectors live in rampart-clip).
# Without this a fresh configure fails: "target llama-common does not exist".
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

# Pin CUDA architectures BEFORE adding llama.cpp (and faiss below). ggml and faiss
# each fall back to their own default arch list when CMAKE_CUDA_ARCHITECTURES is
# unset, and ggml's default leaves common GPUs (V100/T4/A100) as PTX-only. Setting
# it here forces native SASS for the GPUs we deploy to, with a PTX fallback for
# newer parts. Must precede the add_subdirectory calls so both subprojects honor it.
#
# Built with the CUDA 11.8 toolkit (/usr/local/cuda-11.8) on purpose: a CUDA 11.x
# binary runs on any driver R450+ (incl. Debian 11's stock apt 470 driver) via
# minor-version compatibility, and needs only the .so.11 runtime libs that stock apt
# already ships (Debian 12 nvidia-cuda-toolkit = 11.8). That keeps deployment to a
# plain `apt` -- which is what rampart-sql's embed() needs -- with NVIDIA's CUDA
# download as the fallback. Do NOT add sm_90: CUDA 11.8 cannot compile the Hopper
# PDL device intrinsics (see ggml-cuda/common.cuh), which breaks the build.
#   70-89      : native SASS -> V100, T4, A100, RTX30/A10/A40, RTX40/L4/L40 (incl. 3070 Ti)
#   89-virtual : PTX JIT     -> Hopper / Blackwell / future GPUs (still run, on newer drivers)
# Floor is Volta (7.0); add 60;61 for Pascal (GTX 10-series) reach, or drop 70;75 to shrink.
if(LT_ENABLE_GPU AND NOT APPLE)
  set(CMAKE_CUDA_ARCHITECTURES "70-real;75-real;80-real;86-real;89-real;89-virtual" CACHE STRING "")
  # Use ggml's plain-cudaMalloc pool (ggml_cuda_pool_leg) instead of its VMM pool.
  # ggml_cuda_pool_vmm reserves CUDA_POOL_VMM_MAX_SIZE == 32GB of *virtual address
  # space* per llama_context via cuMemAddressReserve (ggml-cuda.cu).  A discrete GPU
  # never notices; Tegra's VA aperture for those reservations is far smaller -- on an
  # Orin Nano exactly two fit (verified: a single 64GB PROT_NONE range in
  # /proc/<pid>/maps), so loading a third model -- e.g. reranker + two per-language
  # embed models -- aborted the process at ggml-cuda.cu:532 with
  # "CUDA error: out of memory" from cuMemAddressReserve.  That is address space, not
  # physical memory: the box had GBs free.
  #
  # The leg pool reserves no VA at all, and unlike the VMM pool it flushes its cached
  # buffers and retries on a real OOM instead of aborting on first failure.  Cost is a
  # slightly larger steady-state pool footprint (best-fit over up to 256 cached
  # buffers, +5% round-up per new allocation, returned only on flush) -- noise on any
  # GPU we deploy to.  FORCEd so an existing build cache can't silently keep the VMM
  # pool; to experiment with VMM again, change it here and reconfigure.
  set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${EXTERN_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/extern/llama.cpp EXCLUDE_FROM_ALL)

# SENTENCEPIECE
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERN_DIR}/sentencepiece EXCLUDE_FROM_ALL)


# FAISS
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "FAISS_ENABLE_PYTHON" FORCE)

if(APPLE)

  # faiss doesn't use metal
  set(FAISS_ENABLE_GPU OFF CACHE BOOL "FAISS_ENABLE_GPU" FORCE)

  set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate" CACHE STRING "CMAKE_EXE_LINKER_FLAGS" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS "-framework Accelerate -framework Foundation" CACHE STRING "CMAKE_SHARED_LINKER_FLAGS" FORCE)
  set(BLA_VENDOR "Apple" CACHE STRING "BLA_VENDOR" FORCE)
  set(BLAS_LIBRARIES "/System/Library/Frameworks/Accelerate.framework" CACHE STRING "BLAS_LIBRARIES" FORCE)
  set(LAPACK_LIBRARIES "/System/Library/Frameworks/Accelerate.framework" CACHE STRING "LAPACK_LIBRARIES" FORCE)


  set(OpenMP_omp_LIBRARY "${OMP_PREFIX}/lib/libomp.a" CACHE STRING "OpenMP_omp_LIBRARY" FORCE)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" CACHE STRING "OpenMP_C_FLAGS" FORCE)
  set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "OpenMP_C_LIB_NAMES" FORCE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include" CACHE STRING "OpenMP_CXX_FLAGS" FORCE)
  set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP_CXX_LIB_NAMES" FORCE)

else()

  #linux faiss
  set(FAISS_ENABLE_GPU ${LT_ENABLE_GPU} CACHE BOOL "FAISS_ENABLE_GPU" FORCE)
  set(env{FAISS_ENABLE_GPU} ${LT_ENABLE_GPU})

  # Pin faiss's OpenMP runtime to the one the compiler actually emits calls for.
  # faiss is the only subproject that uses OpenMP (ggml has GGML_OPENMP OFF), and
  # it runs its own find_package(OpenMP REQUIRED), exporting OpenMP::OpenMP_CXX as
  # an INTERFACE dependency we inherit when libfaiss.a is linked into
  # rampart-faiss.so.  gcc emits GOMP_* calls that MUST resolve against GNU
  # libgomp; if FindOpenMP instead latches onto a system LLVM libomp (its dev
  # symlink can sort ahead of libgomp in the default search), the module ends up
  # with BOTH runtimes -- and the mismatched libomp corrupts libstdc++ locale/TLS
  # state so std::regex inside faiss::index_factory() segfaults at the very first
  # openFactory() (only with ASLR on, which is why it hides under gdb).
  #
  # Force the GNU runtime under gcc by pre-seeding the variables FindOpenMP keys
  # off (already set => it skips its own probing); leave clang alone, since its
  # __kmpc_* calls correctly want libomp.  Mirrors LT_OMP_LIB in the top-level
  # CMakeLists, and the OpenMP pin the APPLE branch above does for libomp.a.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_library(LT_LIBGOMP NAMES gomp libgomp.so.1)
    if(LT_LIBGOMP)
      set(OpenMP_C_FLAGS       "-fopenmp"      CACHE STRING "" FORCE)
      set(OpenMP_CXX_FLAGS     "-fopenmp"      CACHE STRING "" FORCE)
      set(OpenMP_C_LIB_NAMES   "gomp;pthread"  CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "gomp;pthread"  CACHE STRING "" FORCE)
      set(OpenMP_gomp_LIBRARY  "${LT_LIBGOMP}" CACHE STRING "" FORCE)
      find_library(LT_LIBPTHREAD NAMES pthread)
      if(LT_LIBPTHREAD)
        set(OpenMP_pthread_LIBRARY "${LT_LIBPTHREAD}" CACHE STRING "" FORCE)
      endif()
      message(STATUS "rampart-langtools: pinned faiss OpenMP to GNU libgomp (${LT_LIBGOMP})")
    endif()
  endif()

  # CMAKE_CUDA_ARCHITECTURES is pinned once, earlier (before the llama.cpp/faiss
  # subdirectories), so faiss inherits the same arch list. Don't set it again here.

endif()

set(FAISS_ENABLE_C_API ON CACHE BOOL "FAISS_ENABLE_C_API" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE" FORCE)
set(THREADS_PREFER_PTHREAD_FLAG ON CACHE BOOL "THREADS_PREFER_PTHREAD_FLAG" FORCE)
set(CMAKE_HAVE_LIBC_PTHREAD ON CACHE BOOL "CMAKE_HAVE_LIBC_PTHREAD" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "BUILD_TESTING" FORCE)
set(CMAKE_THREAD_LIBS_INIT "" CACHE STRING "CMAKE_THREAD_LIBS_INIT" FORCE)

#not sure these are necessary
set(env{FAISS_ENABLE_C_API} ON)
set(env{CMAKE_BUILD_TYPE} "Release")
set(env{THREADS_PREFER_PTHREAD_FLAG} ON)
set(env{CMAKE_HAVE_LIBC_PTHREAD} ON)
set(env{BUILD_TESTING} OFF)
set(env{CMAKE_THREAD_LIBS_INIT} "")

add_subdirectory(${EXTERN_DIR}/faiss EXCLUDE_FROM_ALL)

# the tiny wrapper for sentencepiece
add_library(spm_c_wrapper_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper/spm_c_wrapper.cc
)

# C++ shim: multi-session slot-based generation engine over llama.cpp + libcommon.
# Compiled as its own object and linked into rampart-llamacpp.so (mirrors the spm
# wrapper above). C ABI in llama_gen_shim.h keeps rampart-llamacpp.c pure C.
set(LLAMA_GEN_SHIM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/llamacpp/wrapper/llama_gen_shim.cc
)
if(APPLE)
    # Cocoa multithreaded-mode primer (Objective-C++) so Metal/Foundation are
    # thread-safe when inference runs on the dedicated thread.
    list(APPEND LLAMA_GEN_SHIM_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/llamacpp/wrapper/llama_gen_macos.mm)
endif()
add_library(llama_gen_shim_obj OBJECT ${LLAMA_GEN_SHIM_SRCS})
set_target_properties(llama_gen_shim_obj PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(llama_gen_shim_obj PRIVATE
    ${EXTERN_DIR}/llama.cpp
    ${EXTERN_DIR}/llama.cpp/include
    ${EXTERN_DIR}/llama.cpp/common
    ${EXTERN_DIR}/llama.cpp/vendor        # nlohmann/json (pulled in by common/chat.h)
    ${EXTERN_DIR}/llama.cpp/ggml/include
)
add_dependencies(llama_gen_shim_obj llama-common llama ggml)

# ----------------------------------------------------------------------------
# ONNX RUNTIME
#
# LT_ONNX gates the whole section (module OR GPU runtime dir).  Default ON;
# forced OFF automatically when this CMake is too old for ORT's deps.  The
# 2_17-tier ovens pass -DLT_ONNX=0 explicitly: ORT 1.27 requires glibc >= 2.28
# and a newer gcc than manylinux2014 carries, so those tiers ship every module
# EXCEPT rampart-onnx (the 2_28 tier is the onnx-capable Linux tier).
#
# Unlike llama.cpp/sentencepiece/faiss, ONNX Runtime is NOT consumed via
# add_subdirectory(): its CMake is a self-contained top-level project driven by
# build.py (~80 cache vars + FetchContent deps) and is not meant to be embedded
# as a subproject. So we drive its own build out-of-tree via ExternalProject --
# this keeps the normal `cmake .. && make` workflow building ORT automatically
# (no separate prebuild step). It emits libonnxruntime.so into
# ${CMAKE_BINARY_DIR}/extern/onnxruntime/Release (same build/extern/<dep>
# location the add_subdirectory deps use), which we consume as an IMPORTED .so.
# Public headers live in the vendored source tree (flat C API dir + CPU factory).
#
# NB: the ORT build downloads its deps via FetchContent -> needs network on the
# first build, and needs CMake >= 3.28 (onnx dep floor is 3.26). It is slow
# (~30 min) the first time; ExternalProject stamps it so later builds skip it.
# ONNXRUNTIME_LIB_DIR can be overridden (-D) so alternate build trees can point
# at one already-built ORT instead of rebuilding it.
option(LT_ONNX "Build rampart-onnx (or, on GPU flavors, its ORT runtime dir)" ON)
if(LT_ONNX AND CMAKE_VERSION VERSION_LESS 3.28)
  message(STATUS "rampart-onnx: CMake ${CMAKE_VERSION} < 3.28 (ORT dep floor) -> disabled")
  set(LT_ONNX OFF)
endif()
if(NOT LT_ONNX)
  set(ONNX_LIBS "")
  set(ONNX_GPU OFF)
  message(STATUS "rampart-onnx: disabled (LT_ONNX=0)")
else()

include(ExternalProject)
set(ONNX_DIR ${EXTERN_DIR}/onnxruntime)
set(ONNX_INCLUDE_DIRS
    ${ONNX_DIR}/include/onnxruntime/core/session
    ${ONNX_DIR}/include/onnxruntime/core/providers/cpu
)
set(ONNXRUNTIME_LIB_DIR "${CMAKE_BINARY_DIR}/extern/onnxruntime/Release"
    CACHE PATH "Directory of the (ExternalProject-built) ONNX Runtime static archives")
# CPU flavor: two merged static archives produced by rampart-build-cpu.sh.
set(ONNX_CORE_A "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_core.a")  # 10 ORT internal libs
set(ONNX_DEPS_A "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_deps.a")  # abseil/onnx/protobuf/re2/...
# GPU flavor: ORT can't static-link the CUDA EP (it's a dlopen'd shared provider),
# so rampart-build-cuda.sh does a SHARED build emitting these three. libonnxruntime.so
# carries soname libonnxruntime.so.1 (-> .so.1.27.0); the module DT_NEEDs that and
# finds it + the providers beside itself via $ORIGIN rpath (set in CMakeLists).
set(ONNX_SHARED_LIB       "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
set(ONNX_PROVIDERS_SHARED "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so")
set(ONNX_PROVIDERS_CUDA   "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so")

# rampart-onnx GPU flavor gate. ORT 1.27's CUDA EP requires CUDA >= 12.0, so a cu11
# (CUDA 11.8) build can't use it -- onnx there falls back to the static CPU EP. Only
# engage the GPU EP on a GPU build with a CUDA-12+ toolkit (or when the version is
# not yet known, which is the cu12/cu13 ovens' case at this point).
set(ONNX_GPU OFF)
if(LT_ENABLE_GPU AND NOT APPLE)
  # Resolve the CUDA toolkit version robustly. CMAKE_CUDA_COMPILER_VERSION is set by
  # enable_language(CUDA) -- but that ran inside the llama.cpp add_subdirectory() (a
  # child scope), so it does NOT propagate back up here. Relying on it left the var
  # empty on a native CUDA-11 build (e.g. firefly, CUDA 11.8 nvcc at /usr/bin), which
  # fell through to ONNX_GPU=ON and made ORT try to build its CUDA EP with a dangling
  # /usr/local/cuda/bin/nvcc. CMAKE_CUDA_COMPILER (a *cache* var) IS visible here, so
  # query it directly. The cu12/cu13 ovens land on 12.x/13.x this way and stay GPU.
  set(_onnx_cuda_ver "${CMAKE_CUDA_COMPILER_VERSION}")
  if(NOT _onnx_cuda_ver AND CMAKE_CUDA_COMPILER)
    execute_process(
      COMMAND "${CMAKE_CUDA_COMPILER}" --version
      OUTPUT_VARIABLE _onnx_nvcc_out ERROR_QUIET)
    if(_onnx_nvcc_out MATCHES "release ([0-9]+\\.[0-9]+)")
      set(_onnx_cuda_ver "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(_onnx_cuda_ver AND _onnx_cuda_ver VERSION_LESS 12.0)
    message(WARNING
      "rampart-onnx: CUDA ${_onnx_cuda_ver} is < 12 -- ORT 1.27's CUDA execution "
      "provider requires CUDA >= 12, so the GPU onnx runtime will NOT be built. "
      "rampart-onnx will still be built as a CPU-only module (embed/rerank run on "
      "the CPU). llama.cpp/faiss keep their CUDA ${_onnx_cuda_ver} GPU support; only "
      "onnx GPU acceleration is unavailable at this toolkit version. Build with a "
      "CUDA >= 12 toolkit to get the GPU onnx runtime.")
  elseif(NOT _onnx_cuda_ver)
    message(WARNING
      "rampart-onnx: no CUDA toolkit version could be resolved -- the GPU onnx "
      "runtime will NOT be built. rampart-onnx will still be built as a CPU-only "
      "module.")
  else()
    message(STATUS "rampart-onnx: CUDA ${_onnx_cuda_ver} >= 12 -> onnx builds the GPU EP")
    set(ONNX_GPU ON)
  endif()
endif()

if(ONNX_GPU)
  # Match the arch coverage the ggml/faiss GPU build uses: build-in-oven.sh pins
  # CMAKE_CUDA_ARCHITECTURES per variant (cu12 x86 = 80;86;89;90;100;120;120-virtual),
  # and we inherit it so onnx stays in sync with llama.cpp/faiss. ORT normalizes
  # Blackwell to the arch-specific form itself (adds 100a/120a in
  # cuda_configuration.cmake), so no pre-rewrite is needed here. Fallback = Ada-only.
  # NB: ORT recompiles its kernel set per -real arch, so the full fleet is a MUCH
  # longer build (multi-hour) than a single arch -- narrow with -DONNX_CUDA_ARCH=... to
  # iterate (e.g. just "89-real;89-virtual" for an Ada-only test box).
  if(CMAKE_CUDA_ARCHITECTURES)
    set(_onnx_arch_default "${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set(_onnx_arch_default "89-real;89-virtual")
  endif()
  set(ONNX_CUDA_ARCH "${_onnx_arch_default}" CACHE STRING "CUDA archs for the ORT CUDA EP")
  # FULL CUDA EP by default (complete kernel set -> transformers actually run on GPU).
  # Needs cuDNN at build time; the cu12/cu13 ovens install it (Dockerfile.cuda
  # CUDNN_PKGS) at CUDNN_HOME=/usr. Set ONNX_CUDA_MINIMAL=1 for a cuBLAS-only EP with
  # no cuDNN (smaller; but most transformer ops fall back to CPU -> little GPU gain).
  set(ONNX_CUDA_MINIMAL "0"   CACHE STRING "Build the ORT CUDA EP without cuDNN when 1")
  set(ONNX_CUDNN_HOME   "/usr" CACHE PATH   "cuDNN root for the full ORT CUDA EP")
  # ORT CUDA build parallelism. Its cutlass kernels (flash-attn, quantized GEMM) are
  # memory-monsters, and each nvcc compiles one .cu for EVERY -real arch at once -- so
  # memory ~ parallel x arches. sm_89 (1 arch) was fine at 8; the 6-arch fleet at 8
  # OOM-locked a 15 GB host. Default 2 is safe for the fleet on ~16 GB; raise via
  # -DONNX_CUDA_PARALLEL=N on a big-RAM builder. (Outer langtools -j is unaffected.)
  set(ONNX_CUDA_PARALLEL "1" CACHE STRING "ORT CUDA build --parallel (memory ~ parallel x arches; measured: 15GB box needs 1 for a fleet, 2 spikes into swap, 8 locks. Raise via -D on a big-RAM builder.)")
  # Transport the arch list comma-separated: a ';' would be split by CMake list
  # semantics inside `-E env`, corrupting the var. The script converts ',' -> ';'.
  string(REPLACE ";" "," ONNX_CUDA_ARCH_CSV "${ONNX_CUDA_ARCH}")
  set(ONNX_BUILD_CMD ${CMAKE_COMMAND} -E env
        "ONNX_CUDA_ARCH=${ONNX_CUDA_ARCH_CSV}"
        "ONNX_CUDA_MINIMAL=${ONNX_CUDA_MINIMAL}"
        "CUDNN_HOME=${ONNX_CUDNN_HOME}"
        "ONNX_CUDA_PARALLEL=${ONNX_CUDA_PARALLEL}"
        sh ${ONNX_DIR}/rampart-build-cuda.sh ${CMAKE_BINARY_DIR}/extern/onnxruntime)
  set(ONNX_BYPRODUCTS ${ONNX_SHARED_LIB} ${ONNX_PROVIDERS_SHARED} ${ONNX_PROVIDERS_CUDA})
else()
  set(ONNX_BUILD_CMD sh ${ONNX_DIR}/rampart-build-cpu.sh ${CMAKE_BINARY_DIR}/extern/onnxruntime)
  set(ONNX_BYPRODUCTS ${ONNX_CORE_A} ${ONNX_DEPS_A})
endif()

# Drive ORT's own build.sh (via the cpu/cuda wrapper) as an external build.
# CONFIGURE/UPDATE/PATCH/INSTALL are no-ops. BUILD_BYPRODUCTS lets Ninja/Make track them.
ExternalProject_Add(onnxruntime_ep
    SOURCE_DIR          ${ONNX_DIR}
    DOWNLOAD_COMMAND    ""
    UPDATE_COMMAND      ""
    PATCH_COMMAND       ""
    CONFIGURE_COMMAND   ""
    BINARY_DIR          ${CMAKE_BINARY_DIR}/extern/onnxruntime
    BUILD_COMMAND       ${ONNX_BUILD_CMD}
    INSTALL_COMMAND     ""
    BUILD_BYPRODUCTS    ${ONNX_BYPRODUCTS}
    USES_TERMINAL_BUILD ON
)

# --------------------------------------------------------------------------
# ONNX RUNTIME EXTENSIONS (tokenizers)
# --------------------------------------------------------------------------
# Robust C++ tokenizers replacing the JS wordpiece + the rampart-sentencepiece
# callout: WordPiece via extensions' BertTokenizer C++ class (used directly),
# SentencePiece/BPE via the Ortx C API (OrtxCreateTokenizer). Built static +
# CPU-only (tokenization never touches the GPU, so identical in every flavor)
# and folded into ONE relocatable object with its bundled protobuf/re2 symbols
# LOCALIZED, so they don't collide with ORT's copies at link time (ORT's
# libonnxruntime_deps.a bundles the same libs -- ~1655 overlapping symbols).
# See rampart-build-ext.sh for the ld -r + objcopy --localize-symbols step.
set(ONNXEXT_DIR ${EXTERN_DIR}/onnxruntime-extensions)
set(ONNXEXT_OBJ "${CMAKE_BINARY_DIR}/extern/onnxruntime-extensions/onnxext_all.o"
    CACHE FILEPATH "The (ExternalProject-built) combined+localized extensions object")
set(ONNXEXT_INCLUDE_DIRS
    ${ONNXEXT_DIR}/include
    ${ONNXEXT_DIR}/operators/tokenizer
    ${ONNXEXT_DIR}/base
)
# protobuf per flavor: CPU statically links ORT's FULL protobuf into the module
# (ext must NOT add a second copy -> double static-init SIGSEGV; sentencepiece
# binds to ORT's); GPU links the SHARED libonnxruntime.so.1 whose protobuf is
# hidden, so ext must carry its own protobuf-lite.
if(ONNX_GPU)
  set(ONNXEXT_BUNDLE_PB "1")
else()
  set(ONNXEXT_BUNDLE_PB "0")
endif()
if(NOT ONNX_GPU)
ExternalProject_Add(onnxext_ep
    SOURCE_DIR          ${ONNXEXT_DIR}
    DOWNLOAD_COMMAND    ""
    UPDATE_COMMAND      ""
    PATCH_COMMAND       ""
    CONFIGURE_COMMAND   ""
    BINARY_DIR          ${CMAKE_BINARY_DIR}/extern/onnxruntime-extensions
    BUILD_COMMAND       ${CMAKE_COMMAND} -E env "ONNXEXT_BUNDLE_PROTOBUF=${ONNXEXT_BUNDLE_PB}"
                        sh ${ONNXEXT_DIR}/rampart-build-ext.sh ${CMAKE_BINARY_DIR}/extern/onnxruntime-extensions
    INSTALL_COMMAND     ""
    BUILD_BYPRODUCTS    ${ONNXEXT_OBJ}
    USES_TERMINAL_BUILD ON
)
endif()  # NOT ONNX_GPU: extensions build (module-only dependency)

if(ONNX_GPU)
  # UNIFIED-MODULE SCHEME: a GPU (cu12/cu13) build no longer produces a
  # rampart-onnx module at all.  The ONE rampart-onnx.so (built by the cpu
  # flavors, static CPU ORT inside) dlopens this flavor's runtime at first use
  # if it finds modules/onnx-${SUFFIX}/ beside itself (see onnx_shim.cc's
  # selection ladder: env override > driver version > sm.list > built-in CPU).
  # So here we only build the shared ORT core + CUDA providers and install
  # them, with upstream filenames intact (ORT dlopens the providers by
  # hardcoded name from the core's own dir), into the flavor subdir --
  # which is what lets cu12 and cu13 COEXIST in one modules/ directory.
  set(ONNX_LIBS "")
  set(ONNX_RUNTIME_DIR_NAME "onnx-${SUFFIX}")
  # sm.list: the -real arches baked into this runtime's CUDA kernels; the
  # selection ladder prefers a runtime whose list contains the device's exact
  # compute capability (miss = demoted, not rejected: ORT's PTX may JIT).
  # (Install rules for the runtime dir live in CMakeLists.txt -- RP_PATH is
  # not known yet when this file is included.)
  set(_sm_entries "")
  foreach(_a IN LISTS ONNX_CUDA_ARCH)
    string(REGEX REPLACE "-real$" "" _a2 "${_a}")
    if(NOT _a2 MATCHES "-virtual$")
      string(REGEX REPLACE "[^0-9]" "" _a3 "${_a2}")   # 120a -> 120
      if(_a3)
        list(APPEND _sm_entries "${_a3}")
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _sm_entries)
  string(REPLACE ";" " " ONNX_SM_LIST "${_sm_entries}")
  message(STATUS "rampart-onnx: GPU flavor -> runtime dir ${ONNX_RUNTIME_DIR_NAME} (sm: ${ONNX_SM_LIST}); the module itself comes from the cpu flavor")
else()
  # CPU: ORT is statically linked into rampart-onnx.so -- no runtime libonnxruntime.so,
  # nothing extra to install/rpath. --whole-archive the core so its CPU provider/kernel
  # static initializers survive; deps archive is normal-linked. -Bsymbolic-functions
  # (top-level) keeps ORT's bundled protobuf/abseil from clashing with other modules'.
  if(APPLE)
    # Apple ld: -force_load,<archive> == --whole-archive for one archive; no librt.
    # CoreML is NOT built (rampart-build-cpu.sh no longer passes --use_coreml -- it
    # buys nothing on macOS and blocks the older-SDK x86 build; see the note there).
    # The framework links are kept but inert: linking an unused framework is just a
    # harmless load command, and it avoids a mac-only relink surprise if any Foundation
    # symbol is pulled in. Drop them once a mac build confirms nothing references them.
    set(ONNX_LIBS
        "-Wl,-force_load,${ONNX_CORE_A}"
        ${ONNX_DEPS_A}
        ${ONNXEXT_OBJ}
        "-framework CoreML"
        "-framework Foundation"
        -ldl -lpthread
    )
  else()
    set(ONNX_LIBS
        -Wl,--whole-archive ${ONNX_CORE_A} -Wl,--no-whole-archive
        ${ONNX_DEPS_A}
        ${ONNXEXT_OBJ}
        -ldl -lrt -lpthread
    )
  endif()
endif()

# C++ shim fronting the ORT C API (keeps rampart-onnx.c pure C). The vendored ORT
# headers exist at configure time, so this compiles without waiting on the ORT
# build; only the final rampart-onnx link needs libonnxruntime.so. Mirrors
# llama_gen_shim_obj.
if(NOT ONNX_GPU)
add_library(onnx_shim_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/onnxruntime/wrapper/onnx_shim.cc)
set_target_properties(onnx_shim_obj PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(onnx_shim_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/onnxruntime/wrapper
    ${ONNX_INCLUDE_DIRS}
    ${ONNXEXT_INCLUDE_DIRS}
)
# Extensions' bert_tokenizer.hpp transitively includes onnxruntime_c_api.h; it
# resolves to OUR vendored 1.27 header (ONNX_INCLUDE_DIRS, listed first) -- the
# C API is backward-compatible and the BertTokenizer class layout is ORT-version
# independent (no OrtApi types in its members).
target_compile_definitions(onnx_shim_obj PRIVATE RAMPART_ONNX_EXT=1)
endif()  # NOT ONNX_GPU: extensions + shim + module link libs (cpu flavors only)
endif()  # LT_ONNX
