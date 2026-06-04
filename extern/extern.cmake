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

# Work around a clang/ggml interaction that breaks the build under
# virtualization (e.g. macOS guests in UTM).  ggml's -mcpu=native build probes
# CPU features by *running* a test binary; when the host advertises i8mm but the
# guest cannot execute it (sysctl hw.optional.arm.FEAT_I8MM == 0), ggml appends
# +noi8mm to -mcpu=native.  clang still defines __ARM_FEATURE_MATMUL_INT8 for
# that flag combination, so the i8mm code path compiles in while codegen has the
# feature disabled, failing with "always_inline 'vmmlaq_s32' requires target
# feature 'i8mm'".  On bare metal (i8mm present) -mcpu=native is fine, so only
# disable GGML_NATIVE and pin an explicit -march when i8mm is actually missing.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  execute_process(
    COMMAND sysctl -n hw.optional.arm.FEAT_I8MM
    OUTPUT_VARIABLE LT_FEAT_I8MM
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT LT_FEAT_I8MM STREQUAL "1")
    # Build an arch string from the features this CPU really exposes.
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

# When llama.cpp is built as a subproject, its `common` subdirectory is OFF by
# default (it defaults to LLAMA_STANDALONE, which is OFF here). But we link
# `llama-common`/`llama-common-base`, so force it on. TOOLS/SERVER/EXAMPLES/TESTS
# stay off (subproject defaults) — we don't build them (we used to need tools/mtmd
# for an old vision experiment; that's gone, image vectors live in rampart-clip).
# Without this a fresh configure fails: "target llama-common does not exist".
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

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

  if(LT_ENABLE_GPU)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "")
    set(env{CMAKE_CUDA_ARCHITECTURES} "80;86;89")
  endif()

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
