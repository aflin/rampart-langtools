set(EXTERN_DIR ${CMAKE_SOURCE_DIR}/extern)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)

add_subdirectory(${EXTERN_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/extern/llama.cpp EXCLUDE_FROM_ALL)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)

add_subdirectory(${EXTERN_DIR}/sentencepiece EXCLUDE_FROM_ALL)

  set(ENV{FAISS_ENABLE_GPU} OFF)
  set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "FAISS_ENABLE_PYTHON" FORCE)

if(Apple)

  set(FAISS_ENABLE_GPU OFF CACHE BOOL "FAISS_ENABLE_GPU" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate" CACHE STRING "CMAKE_EXE_LINKER_FLAGS" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS "-framework Accelerate" CACHE STRING "CMAKE_SHARED_LINKER_FLAGS" FORCE)
  set(BLA_VENDOR "Apple" CACHE STRING "BLA_VENDOR" FORCE)
  set(BLAS_LIBRARIES "/System/Library/Frameworks/Accelerate.framework" CACHE STRING "BLAS_LIBRARIES" FORCE)
  set(LAPACK_LIBRARIES "/System/Library/Frameworks/Accelerate.framework" CACHE STRING "LAPACK_LIBRARIES" FORCE)
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.a" CACHE STRING "OpenMP_omp_LIBRARY" FORCE)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" CACHE STRING "OpenMP_C_FLAGS" FORCE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" CACHE STRING "OpenMP_CXX_FLAGS" FORCE)
  set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP_CXX_LIB_NAMES" FORCE)
  set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "OpenMP_C_LIB_NAMES" FORCE)
  set(env{FAISS_ENABLE_GPU} OFF)
  set(env{BLA_VENDOR} "Apple")
  set(env{BLAS_LIBRARIES} "/System/Library/Frameworks/Accelerate.framework")
  set(env{LAPACK_LIBRARIES} "/System/Library/Frameworks/Accelerate.framework")
  set(env{CMAKE_EXE_LINKER_FLAGS} "-framework Accelerate")
  set(env{CMAKE_SHARED_LINKER_FLAGS} "-framework Accelerate")
  set(env{OpenMP_CXX_FLAGS} "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
  set(env{OpenMP_C_FLAGS} "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
  set(env{OpenMP_omp_LIBRARY} "/opt/homebrew/opt/libomp/lib/libomp.a")
  set(env{OpenMP_CXX_LIB_NAMES} "omp")
  set(env{OpenMP_C_LIB_NAMES} "omp")

else()

  set(FAISS_ENABLE_GPU ON CACHE BOOL "FAISS_ENABLE_GPU" FORCE)
  set(env{FAISS_ENABLE_GPU} ON)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "")
  set(env{CMAKE_CUDA_ARCHITECTURES} "80;86;89")

endif()

  set(FAISS_ENABLE_C_API ON CACHE BOOL "FAISS_ENABLE_C_API" FORCE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE" FORCE)
  set(THREADS_PREFER_PTHREAD_FLAG ON CACHE BOOL "THREADS_PREFER_PTHREAD_FLAG" FORCE)
  set(CMAKE_HAVE_LIBC_PTHREAD ON CACHE BOOL "CMAKE_HAVE_LIBC_PTHREAD" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "BUILD_TESTING" FORCE)
  set(CMAKE_THREAD_LIBS_INIT "" CACHE STRING "CMAKE_THREAD_LIBS_INIT" FORCE)


  set(env{FAISS_ENABLE_C_API} ON)
  set(env{CMAKE_BUILD_TYPE} "Release")
  set(env{THREADS_PREFER_PTHREAD_FLAG} ON)
  set(env{CMAKE_HAVE_LIBC_PTHREAD} ON)
  set(env{BUILD_TESTING} OFF)
  set(env{CMAKE_THREAD_LIBS_INIT} "")

add_subdirectory(${EXTERN_DIR}/faiss EXCLUDE_FROM_ALL)

add_library(spm_c_wrapper_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper/spm_c_wrapper.cc
)
