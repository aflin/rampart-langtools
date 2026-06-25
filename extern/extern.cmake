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
