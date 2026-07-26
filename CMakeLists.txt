cmake_minimum_required(VERSION 3.15)
project(rampart_langtools C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional suffix for GPU-related output files (e.g., -DSUFFIX=cuda)
set(SUFFIX "" CACHE STRING "Suffix appended to GPU-related output filenames")
if(SUFFIX)
  set(LT_NAME_SUFFIX "_${SUFFIX}")
else()
  set(LT_NAME_SUFFIX "")
endif()

# Common compile flags
add_compile_options(-Wall -g -O3)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")

if(APPLE)

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    option(LT_ENABLE_GPU "Enable GPU acceleration" ON)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=11.0 -arch arm64")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=11.0 -arch arm64")
  else()
    option(LT_ENABLE_GPU "Enable GPU acceleration" OFF)
  endif()

  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE OMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

else()
  # LINUX

  option(LT_ENABLE_GPU "Enable GPU acceleration" OFF)

endif()




include(${CMAKE_CURRENT_SOURCE_DIR}/extern/extern.cmake)

# Resolve rampart's install prefix (its process.installPath; modules install to
# <prefix>/modules, headers come from <prefix>/include).  Normally we ask the rampart
# binary, but the cross-glibc ovens can't: they mount the host's rampart install into
# an OLDER-glibc manylinux container, and a rampart built against a newer glibc than
# the container provides won't even execute there ("/lib64/libc.so.6: version
# GLIBC_2.xx not found").  The oven already knows the prefix, so it passes
# -DRP_PATH=<prefix> to skip running the binary entirely.
if(NOT RP_PATH)
  find_program(RAMPART_EXECUTABLE rampart)
  if(NOT RAMPART_EXECUTABLE)
    message(FATAL_ERROR "rampart not found in PATH (or pass -DRP_PATH=<rampart install prefix>)")
  endif()
  execute_process(
    COMMAND ${RAMPART_EXECUTABLE} -c "console.log(process.installPath)"
    RESULT_VARIABLE _res
    OUTPUT_VARIABLE RP_PATH
    ERROR_VARIABLE  _err
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _res EQUAL 0)
    message(FATAL_ERROR "rampart get installPath failed: ${_err}")
  endif()
endif()
message(STATUS "rampart install prefix (RP_PATH) = ${RP_PATH}")


include_directories(
  ${RP_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/faiss
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/ggml/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llamacpp/wrapper
)

# Libraries
set(LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libllama-common.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libllama-common-base.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/src/libllama.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-cpu.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-base.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/sentencepiece/src/libsentencepiece.a
    spm_c_wrapper_obj
    llama_gen_shim_obj
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/c_api/libfaiss_c.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/faiss/libfaiss.a
    -lpthread
    -lm
)

set(LLAMA_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libllama-common.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libllama-common-base.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/src/libllama.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-cpu.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-base.a
    -lpthread
    -lm
)

set(FAISS_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/c_api/libfaiss_c.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/faiss/libfaiss.a
    -lpthread
    -lm
)

set(SP_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/sentencepiece/src/libsentencepiece.a
    spm_c_wrapper_obj
    llama_gen_shim_obj
    -lpthread
)

#Simple helper to find files
function(find_library_in_dirs OUTVAR FILENAME)
    set(dirs ${ARGN})
    set(found "")
    foreach(d ${dirs})
        if(EXISTS "${d}/${FILENAME}")
            set(found "${d}/${FILENAME}")
            break()
        endif()
    endforeach()

    # Return the result to the caller
    set(${OUTVAR} "${found}" PARENT_SCOPE)
endfunction()

# find_libraries_in_dirs(
#   OUT_LIST_VAR
#   FILES <file1> [file2 ...]
#   DIRS  <dir1>  [dir2  ...]
#   [PREFIX <var_prefix>]           # default: LIB_
# )

# right now, we just need the list and are ignoring set vars

function(find_libraries_in_dirs OUT_LIST_VAR)
    set(options)
    set(oneValueArgs PREFIX)
    set(multiValueArgs FILES DIRS)
    cmake_parse_arguments(FLID "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT FLID_FILES OR NOT FLID_DIRS)
        message(FATAL_ERROR "find_libraries_in_dirs: require FILES and DIRS.")
    endif()

    if(NOT FLID_PREFIX)
        set(FLID_PREFIX "LIB_")
    endif()

    set(_found_list "")
    set(_missing "")

    foreach(LIBFILE IN LISTS FLID_FILES)
        set(_found "")
        foreach(D IN LISTS FLID_DIRS)
            if(EXISTS "${D}/${LIBFILE}")
                set(_found "${D}/${LIBFILE}")
                break()
            endif()
        endforeach()

        # Derive variable name: libfoo.so -> LIB_FOO
        get_filename_component(_fname "${LIBFILE}" NAME)
        string(REGEX REPLACE "^lib" "" _base "${_fname}")
        string(REGEX REPLACE "\\..*$" "" _base "${_base}")
        string(TOUPPER "${_base}" _upper)
        set(_var "${FLID_PREFIX}${_upper}")

        # Export per-library variable
        set(${_var} "${_found}" PARENT_SCOPE)

        if(_found)
            list(APPEND _found_list "${_found}")
            message(STATUS "Found ${LIBFILE}: ${_found}")
        else()
            list(APPEND _missing "${LIBFILE}")
        endif()
    endforeach()

    if(_missing)
        list(JOIN _missing ", " _miss_str)
        message(FATAL_ERROR "Missing required libraries: ${_miss_str}")
    endif()

    # Export the list of found paths
    set(${OUT_LIST_VAR} "${_found_list}" PARENT_SCOPE)
endfunction()

# The big one
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
add_library(rampart-langtools SHARED
    faiss_detect.cpp
    rampart-langtools.c
    rp-chunker.c
    rp-embed-cache.c
    $<TARGET_OBJECTS:llama_gen_shim_obj>
)

# llama.cpp
add_library(rampart-llamacpp SHARED
    rampart-llamacpp.c
    rp-chunker.c
    rp-embed-cache.c
    $<TARGET_OBJECTS:llama_gen_shim_obj>
)

# faiss
add_library(rampart-faiss SHARED
    faiss_detect.cpp
    rampart-faiss.c
)

# sentencepiece
add_library(rampart-sentencepiece SHARED
    rampart-sentencepiece.c
)

# clip (CLIP image+text embedding in a shared space).  Its ggml-facing shim links
# the SAME ggml rampart-llamacpp builds -- one ggml in the tree, no second copy.
add_library(rampart-clip SHARED
    rampart-clip.c
    extern/clip/clip_shim.cpp
)
target_include_directories(rampart-clip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/clip)
# llama + llama-common included: rampart-clip links LLAMA_LIBS by file
# path (libllama-common.a etc.); without the target-level edges a fresh
# parallel build can reach the link step before those archives exist
# ("No rule to make target ... libllama-common.a").
add_dependencies(rampart-clip ggml ggml-cpu ggml-base llama llama-common)
# clip's per-platform link options (soname/install_name, -Bsymbolic-functions,
# and --exclude-libs to hide the vendored ggml) live in the Apple / Linux
# target_link_options blocks below, alongside the other modules'.

if(LT_ENABLE_GPU)
    target_compile_definitions(rampart-langtools PRIVATE LT_ENABLE_GPU=1)
    target_compile_definitions(rampart-llamacpp PRIVATE LT_ENABLE_GPU=1)
    target_compile_definitions(rampart-faiss PRIVATE LT_ENABLE_GPU=1)
    target_compile_definitions(rampart-clip PRIVATE LT_ENABLE_GPU=1)

    # A manual CUDA toolkit install (e.g. /usr/local/cuda on Debian 11, drivers
    # from NVIDIA rather than apt) doesn't put cublas_v2.h and friends on the
    # default include path the way apt's CUDA does on Debian 12. Add it when
    # present; harmless where the toolkit headers are already on the search path.
    if(EXISTS "/usr/local/cuda/include")
        target_include_directories(rampart-langtools PRIVATE /usr/local/cuda/include)
        target_include_directories(rampart-llamacpp PRIVATE /usr/local/cuda/include)
        target_include_directories(rampart-faiss PRIVATE /usr/local/cuda/include)
        target_include_directories(rampart-clip PRIVATE /usr/local/cuda/include)
    endif()
endif()

# rampart-llamacpp GPU-arch guard: bake the native-SASS sm list (the -real archs from
# CMAKE_CUDA_ARCHITECTURES, e.g. "87 90 100 120") into LT_CUDA_SM_LIST.  At init,
# rampart-llamacpp.c compares the device's compute capability against it and throws a
# clean error on a mismatch instead of letting ggml abort mid-graph (e.g. a cu12 module
# on a Blackwell GB10 / sm_121).  -virtual (PTX) archs are excluded on purpose: ggml's
# family-specific 'a' kernels don't JIT across SM versions.  No CUDA language / kernel
# needed -- it's a host-side cudaGetDeviceProperties check.
if(LT_ENABLE_GPU AND NOT APPLE)
    set(_lt_sm "")
    foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_a MATCHES "^([0-9]+)(-real)?$")          # numeric / -real archs; skip -virtual
            string(APPEND _lt_sm "${CMAKE_MATCH_1} ")
        endif()
    endforeach()
    string(STRIP "${_lt_sm}" _lt_sm)
    message(STATUS "rampart-langtools: GPU-arch guard sm list = ${_lt_sm}")
    target_compile_definitions(rampart-llamacpp PRIVATE "LT_CUDA_SM_LIST=\"${_lt_sm}\"")
    # rampart-clip uses the SAME ggml (real-arch SASS) -> same guard, same list.
    target_compile_definitions(rampart-clip PRIVATE "LT_CUDA_SM_LIST=\"${_lt_sm}\"")
endif()


if(APPLE)

  ##########  APPLE ############
  if(LT_ENABLE_GPU AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") 
    message(FATAL_ERROR "Cannot build for GPU on Apple x86_64")
  endif()

  if (NOT OMP_PREFIX)
    message(FATAL_ERROR "OpenMP not found, try brew install libomp")
  endif()

  execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE BREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if (NOT BREW_PREFIX)
    message(FATAL_ERROR "brew package manager is required")
  endif()

  # Extra includes on macOS
  include_directories(
   "${BREW_PREFIX}/include"
   "${OMP_PREFIX}/include"
  )

  # Extra compile flags
  add_compile_options(-Wno-deprecated-declarations)

  # Where to look on macOS
  set(LIB_DIRS
    "${BREW_PREFIX}/lib"
    /usr/lib
    /usr/local/lib
    /opt/homebrew/lib
    /opt/local/lib
    "${OMP_PREFIX}/lib"
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
  )

  set(LIB_FILES
    libomp.a
    libcurl.tbd
  )
  find_libraries_in_dirs(FOUND_LIBS FILES ${LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  set(LLAMA_LIB_FILES
    libcurl.tbd
    libomp.a
  )
  find_libraries_in_dirs(LLAMA_FOUND_LIBS FILES ${LLAMA_LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  set(FAISS_LIB_FILES
    libomp.a
  )

  find_libraries_in_dirs(FAISS_FOUND_LIBS FILES ${FAISS_LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)
  list(APPEND FAISS_LIBS ${FAISS_FOUND_LIBS})

  list(APPEND LIBS
          ${FOUND_LIBS}
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-blas/libggml-blas.a
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-metal/libggml-metal.a
  )
  list(APPEND LLAMA_LIBS
          ${LLAMA_FOUND_LIBS}
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-blas/libggml-blas.a
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-metal/libggml-metal.a
  )

  target_link_options(rampart-langtools PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-langtools${LT_NAME_SUFFIX}.so"
  )

  target_link_options(rampart-llamacpp PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-llamacpp${LT_NAME_SUFFIX}.so"
  )

  target_link_options(rampart-faiss PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-faiss${LT_NAME_SUFFIX}.so"
  )

  target_link_options(rampart-sentencepiece PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-sentencepiece.so"
  )

  target_link_options(rampart-clip PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-clip${LT_NAME_SUFFIX}.so"
    # macOS symbol hiding (the -Wl,--exclude-libs equivalent -- -hidden-l / an
    # -exported_symbols_list of just _duk_open_module) is deferred with the macOS
    # port; a two-level namespace already binds each module to its own ggml.
  )

  # Frameworks
  target_link_libraries(rampart-langtools PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework Accelerate"
  )
  target_link_libraries(rampart-llamacpp PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework Accelerate"
  )
  target_link_libraries(rampart-faiss PRIVATE
    "-framework Foundation"
    "-framework Accelerate"
  )

else()

  ############ Linux ##############

  set(LIB_DIRS
    /lib
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/i386-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/lib/arm-linux-gnueabihf
    /usr/local/cuda/lib64
  )

  # OpenMP runtime by compiler: gcc emits GOMP_* -> GNU libgomp (already in the
  # bundle's lib/, shared with rampart-sql's FAISS); clang emits __kmpc_* ->
  # LLVM libomp (libomp5.so -- system on FreeBSD).  macOS uses static libomp.a
  # in its own branch above; this only affects the Linux/FreeBSD path.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LT_OMP_LIB libgomp.so.1)
  else()
    set(LT_OMP_LIB libomp5.so)
  endif()

  if(LT_ENABLE_GPU)
    set(LIB_FILES
      ${LT_OMP_LIB}
      libopenblas.a
      libgfortran.so.5
      libcudart.so
      libcublas.so
      libcuda.so
    )
  else()
    set(LIB_FILES
      ${LT_OMP_LIB}
      libopenblas.a
      libgfortran.so.5
    )
  endif()

  find_libraries_in_dirs(FOUND_LIBS FILES ${LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  if(LT_ENABLE_GPU)
    list(APPEND LIBS
      ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-cuda/libggml-cuda.a
      ${FOUND_LIBS}
    )
    list(APPEND LLAMA_LIBS
      ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-cuda/libggml-cuda.a
      ${FOUND_LIBS}
    )
  else()
    list(APPEND LIBS
      ${FOUND_LIBS}
    )
    list(APPEND LLAMA_LIBS
      ${FOUND_LIBS}
    )
  endif()

  list(APPEND FAISS_LIBS
    ${FOUND_LIBS}
  )

  # -Bsymbolic-functions: bind each module's *own* function references to its
  # own definitions at link time, so they are not interposed at load time.
  # rampart dlopen's modules with RTLD_GLOBAL (see duktape/core/module.c), which
  # publishes every module's symbols into the global namespace.  These modules
  # statically link a lot of C++ (faiss/llama.cpp/ggml) and therefore carry their
  # own inline/template copies of libstdc++ internals (std::string, std::regex,
  # locale facets, ...).  Without -Bsymbolic-functions those calls resolve to
  # whatever copy was loaded first -- the rampart executable's, built against an
  # OLDER libstdc++ (max GLIBCXX_3.4.18) than a native gcc build of these modules
  # (GLIBCXX_3.4.30).  The ABI-incompatible mix corrupts std::string state and
  # faiss::index_factory()'s std::regex segfaults at the first openFactory()
  # (with ASLR on; it hides under gdb).  Binding locally keeps each module on one
  # consistent libstdc++.  Linux-only: macOS uses a two-level namespace, so its
  # symbols already bind to the library they were linked against.  (The shipped
  # "oven"/manylinux builds happen to match rampart's old toolchain and so dodge
  # this, but a plain native `cmake .. && make` must not.)
  target_link_options(rampart-langtools PRIVATE
    "-Wl,-soname,@rpath/rampart-langtools${LT_NAME_SUFFIX}.so"
    "-Wl,-Bsymbolic-functions"
  )

  target_link_options(rampart-llamacpp PRIVATE
    "-Wl,-soname,@rpath/rampart-llamacpp${LT_NAME_SUFFIX}.so"
    "-Wl,-Bsymbolic-functions"
  )

  target_link_options(rampart-faiss PRIVATE
    "-Wl,-soname,@rpath/rampart-faiss${LT_NAME_SUFFIX}.so"
    "-Wl,-Bsymbolic-functions"
  )

  # clip: same as the others (own libstdc++/ggml statics need local binding), PLUS
  # --exclude-libs,ALL to fully hide the vendored ggml/gguf/stb symbols.  rampart
  # dlopens modules RTLD_GLOBAL, and rampart-llamacpp.so also statically links ggml;
  # -Bsymbolic-functions binds clip's own ggml *function* calls locally, and
  # --exclude-libs stops clip exporting those symbols at all -- so the two modules'
  # ggml never interpose, even when they differ (a CUDA llamacpp vs a CPU clip).
  # Only duk_open_module (clip's own object) stays exported.
  target_link_options(rampart-clip PRIVATE
    "-Wl,-soname,@rpath/rampart-clip${LT_NAME_SUFFIX}.so"
    "-Wl,-Bsymbolic-functions"
    "-Wl,--exclude-libs,ALL"
  )

  ## OLD LINUX
  include(CheckCXXSourceCompiles)

  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")
  set(FS_TEST_SRC "
  #include <filesystem>
  int main() {
      auto p = std::filesystem::current_path();
      (void)p;
      return 0;
  }")


  # Try with no extra lib (GCC>=9 / modern Clang)
  unset(CMAKE_REQUIRED_LIBRARIES)
  check_cxx_source_compiles("${FS_TEST_SRC}" HAS_FS_NO_LINK)

  if (HAS_FS_NO_LINK)
      # nothing to do
  else()
      # Try libstdc++fs (older GCC / Clang+libstdc++)
      set(CMAKE_REQUIRED_LIBRARIES stdc++fs)
      check_cxx_source_compiles("${FS_TEST_SRC}" HAS_FS_STDCXXFS)
      if (HAS_FS_STDCXXFS)
          list(APPEND LIBS stdc++fs)
          list(APPEND LLAMA_LIBS stdc++fs)
      else()
          # Try libc++fs (older Clang+libc++)
          set(CMAKE_REQUIRED_LIBRARIES c++fs)
          check_cxx_source_compiles("${FS_TEST_SRC}" HAS_FS_CXXFS)
          if (HAS_FS_CXXFS)
              list(APPEND LIBS stdc++fs)
              list(APPEND LLAMA_LIBS stdc++fs)
          else()
              message(FATAL_ERROR "C++17 <filesystem> not available on this toolchain.")
          endif()
      endif()
  endif()
  ## end old linux


endif()
##end linux

# RPATH "$ORIGIN/../lib" so the langtools .so's find the rampart
# install bundle's libopenblas / libgfortran / libgomp / libquadmath
# under <prefix>/lib/.  Same pattern as rampart-sql.so and
# rampart-graphicsmagick.so in the main build.  $ORIGIN is honoured
# by both glibc's rtld and FreeBSD's rtld.
set(LT_RPATH "\$ORIGIN/../lib")

set_target_properties(
    rampart-langtools
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-langtools${LT_NAME_SUFFIX}"
    INSTALL_RPATH "${LT_RPATH}"
    BUILD_RPATH   "${LT_RPATH}"
)

set_target_properties(
    rampart-llamacpp
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-llamacpp${LT_NAME_SUFFIX}"
    INSTALL_RPATH "${LT_RPATH}"
    BUILD_RPATH   "${LT_RPATH}"
)

set_target_properties(
    rampart-clip
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-clip${LT_NAME_SUFFIX}"
    INSTALL_RPATH "${LT_RPATH}"
    BUILD_RPATH   "${LT_RPATH}"
)

set_target_properties(
    rampart-faiss
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-faiss${LT_NAME_SUFFIX}"
    INSTALL_RPATH "${LT_RPATH}"
    BUILD_RPATH   "${LT_RPATH}"
)

set_target_properties(
    rampart-sentencepiece
    PROPERTIES SUFFIX ".so"
    PREFIX ""
    INSTALL_RPATH "${LT_RPATH}"
    BUILD_RPATH   "${LT_RPATH}"
)

if(NOT DEFINED LT_ENABLE_GPU_LAST)
    set(LT_ENABLE_GPU_LAST "unset")
    message(STATUS "GPU set ${LT_ENABLE_GPU}, was not previously set")
else()
    message(STATUS "GPU set ${LT_ENABLE_GPU}, was set ${LT_ENABLE_GPU_LAST}")
endif()

#always rebuild if we change to/from gpu build
if("${LT_ENABLE_GPU}" STREQUAL "${LT_ENABLE_GPU_LAST}")

  #faiss really likes to rebuild unnecessarily
  set(FAISS_C_LIB "${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/c_api/libfaiss_c.a")
  set(FAISS_LIB   "${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/faiss/libfaiss.a")

  set(LLAMA_LIB   "${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/src/libllama.a")
  set(GGML_LIB    "${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml.a")

  set(SP_LIB      "${CMAKE_CURRENT_BINARY_DIR}/extern/sentencepiece/src/libsentencepiece.a")

  if(NOT EXISTS "${FAISS_C_LIB}" OR NOT EXISTS "${FAISS_LIB}")
    add_dependencies(rampart-langtools
      faiss
      faiss_c
    )
    add_dependencies(rampart-faiss
      faiss
      faiss_c
    )
  endif()

  if(NOT EXISTS "${LLAMA_LIB}" OR NOT EXISTS "${GGML_LIB}")
    add_dependencies(rampart-langtools
      llama
      llama-common
      ggml
    )
    add_dependencies(rampart-llamacpp
      llama
      llama-common
      ggml
    )
  endif()

  if(NOT EXISTS "${SP_LIB}")
    add_dependencies(rampart-langtools
      sentencepiece
    )
  endif()
  if(NOT EXISTS "${SP_LIB}")
    add_dependencies(rampart-sentencepiece
      sentencepiece
    )
  endif()
else()
  # build/rebuild from scratch
  if(NOT "${LT_ENABLE_GPU_LAST}" STREQUAL "unset")
    if(LT_ENABLE_GPU)
      message(STATUS "Rebuilding for gpu")
    else()
      message(STATUS "Rebuilding for cpu")
    endif()
  endif()
  add_dependencies(rampart-langtools
    llama
    llama-common
    ggml
    sentencepiece
    faiss
    faiss_c
  )

  add_dependencies(rampart-llamacpp
    llama
    llama-common
    ggml
  )
  add_dependencies(rampart-sentencepiece
    sentencepiece
  )
  add_dependencies(rampart-faiss
    faiss
    faiss_c
  )

endif()




set(LT_ENABLE_GPU_LAST "${LT_ENABLE_GPU}" CACHE INTERNAL "Previous value of LT_ENABLE_GPU")

# Link with all libs
target_link_libraries(rampart-langtools PRIVATE ${LIBS})

# Link with selected libs
target_link_libraries(rampart-llamacpp PRIVATE
    ${LLAMA_LIBS}
    stdc++
)

# Link with selected libs
target_link_libraries(rampart-faiss PRIVATE ${FAISS_LIBS})

target_link_libraries(rampart-sentencepiece PRIVATE
    ${SP_LIBS}
)

# clip needs only ggml (+ its deps).  Linking the full LLAMA_LIBS is safe: static
# archives contribute only referenced objects, so libllama/-common add nothing,
# while clip rides the exact same ggml set (CPU, or GPU cuda/metal) llama resolved.
target_link_libraries(rampart-clip PRIVATE
    ${LLAMA_LIBS}
    stdc++
)

# rampart-langtools.so is the legacy combined module; its functionality is now
# covered by the separate rampart-llamacpp / rampart-faiss / rampart-sentencepiece
# modules, so it is built but intentionally NOT installed (skip its TARGETS + strip).
install(TARGETS rampart-llamacpp
  DESTINATION "${RP_PATH}/modules"
)
install(TARGETS rampart-faiss
  DESTINATION "${RP_PATH}/modules"
)
install(TARGETS rampart-sentencepiece
  DESTINATION "${RP_PATH}/modules"
)
install(TARGETS rampart-clip
  DESTINATION "${RP_PATH}/modules"
)

install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-llamacpp${LT_NAME_SUFFIX}.so\")")
install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-faiss${LT_NAME_SUFFIX}.so\")")
install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-sentencepiece.so\")")
install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-clip${LT_NAME_SUFFIX}.so\")")

# rampart-models: pure-JS model downloader/locator (single file, embedded
# catalog).  Installed into modules/ so `require('rampart-models')` works.
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/rampart-models.js"
  DESTINATION "${RP_PATH}/modules"
)

# test scripts, runnable from the install tree (e.g. rampart ${RP_PATH}/test/llamacpp-test.js)
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/llamacpp-test.js"
  "${CMAKE_CURRENT_SOURCE_DIR}/faiss-test.js"
  "${CMAKE_CURRENT_SOURCE_DIR}/clip-test.js"
  DESTINATION "${RP_PATH}/test"
)
# clip-test.js's cross-modal retrieval check reads these Public-Domain photos
# from <prefix>/test/test_images/{horses,flowers}/.
install(DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/test_images"
  DESTINATION "${RP_PATH}/test"
)

# ===========================================================================
# rampart-onnx : ONNX Runtime bindings. extern.cmake defines the `onnxruntime`
# IMPORTED target (+ ONNX_LIBS / ONNX_INCLUDE_DIRS / ONNXRUNTIME_LIB_DIR), the
# onnx_shim_obj object lib, and the onnxruntime_ep ExternalProject that actually
# builds libonnxruntime.so. We depend on onnxruntime_ep so a plain `make` builds
# ORT first, then links it here. Self-contained block, out of the per-dep
# plumbing above.
# ===========================================================================
if(ONNX_LIBS)
  add_library(rampart-onnx SHARED
    rampart-onnx.c
    rp-chunker.c
    rp-embed-cache.c
    $<TARGET_OBJECTS:onnx_shim_obj>
  )
  # build + merge the ORT static archives (ExternalProject) before linking,
  # plus the extensions tokenizers object (onnxext_ep -> onnxext_all.o)
  add_dependencies(rampart-onnx onnxruntime_ep onnxext_ep)
  # rampart-onnx.c only needs onnx_shim.h + rampart.h; the shim object already
  # carries the ORT headers.  The sentencepiece wrapper header is header-only
  # and used by the rp_onnx_embed_* C API to declare spm_processor_t; the
  # actual SPM symbols are dlsym'd at runtime from rampart-sentencepiece.
  target_include_directories(rampart-onnx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/onnxruntime/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper
  )
  # ONE unsuffixed module for every flavor/tier: it contains a full static CPU
  # ORT and dlopens an OPTIONAL modules/onnx-cuNN/ runtime dir at first use
  # (absolute path via dladdr -- no $ORIGIN rpath needed for that).
  set(ONNX_RPATH "${LT_RPATH}")
  set_target_properties(rampart-onnx PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "rampart-onnx"
    INSTALL_RPATH "${ONNX_RPATH}"
    BUILD_RPATH   "${ONNX_RPATH}"
  )
  if(APPLE)
    # macOS equivalent of the version script: export only the module entry point
    # so the statically-linked ORT (protobuf/abseil/onnx) stays private and can't
    # clash with other modules' bundled copies.
    target_link_options(rampart-onnx PRIVATE
      "-Wl,-headerpad_max_install_names"
      "-Wl,-undefined,dynamic_lookup"
      "-Wl,-install_name,@rpath/rampart-onnx.so"
      "-Wl,-exported_symbol,_duk_open_module"
      # the C embed API (dlsym'd by rampart-sql) -- mirrors rampart-onnx.map
      "-Wl,-exported_symbol,_rp_onnx_embed_*"
    )
  else()
    # Localize all of statically-linked ORT's symbols (bundled protobuf/abseil/
    # onnx/...) -- export only duk_open_module -- so they don't collide via
    # RTLD_GLOBAL with other modules' bundled deps (see rampart-onnx.map).
    target_link_options(rampart-onnx PRIVATE
      "-Wl,-soname,@rpath/rampart-onnx.so"
      "-Wl,-Bsymbolic-functions"
      "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/rampart-onnx.map"
      # ORT's MLAS .S objects lack .note.GNU-stack; force a non-executable stack.
      "-Wl,-z,noexecstack"
      # Statically-linked ORT registers static-dtor exit handlers whose code lives
      # in this .so. Mark it non-unloadable so rampart's dlclose can't unmap it
      # before libc runs those handlers at exit -- otherwise __run_exit_handlers
      # jumps into freed text and segfaults (manifested on the glibc-2.17 runtime).
      "-Wl,-z,nodelete"
    )
  endif()
  target_link_libraries(rampart-onnx PRIVATE
    ${ONNX_LIBS}
    stdc++
  )
  install(TARGETS rampart-onnx DESTINATION "${RP_PATH}/modules")
  install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-onnx.so\")")
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx-test.js"
    DESTINATION "${RP_PATH}/test")
  message(STATUS "rampart-langtools: rampart-onnx target enabled")
elseif(ONNX_GPU)
  # GPU (cu12/cu13) flavor: no module -- install this flavor's ORT runtime dir
  # (modules/onnx-${SUFFIX}/) for the unified rampart-onnx.so to dlopen.
  # FOLLOW_SYMLINK_CHAIN installs exactly the .so -> .so.1 -> .so.1.27.0 chain;
  # providers keep their upstream names (ORT dlopens them by hardcoded name
  # from the core's own dir), which is what lets cu12 + cu13 COEXIST.
  install(CODE "file(INSTALL \"${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so\"
                     DESTINATION \"${RP_PATH}/modules/${ONNX_RUNTIME_DIR_NAME}\" FOLLOW_SYMLINK_CHAIN)")
  install(FILES ${ONNX_PROVIDERS_SHARED} ${ONNX_PROVIDERS_CUDA}
          DESTINATION "${RP_PATH}/modules/${ONNX_RUNTIME_DIR_NAME}")
  install(CODE "file(WRITE \"${RP_PATH}/modules/${ONNX_RUNTIME_DIR_NAME}/sm.list\" \"${ONNX_SM_LIST}\n\")")
  message(STATUS "rampart-langtools: rampart-onnx GPU runtime dir ${ONNX_RUNTIME_DIR_NAME} (sm: ${ONNX_SM_LIST})")
else()
  message(STATUS "rampart-langtools: rampart-onnx skipped (ONNX Runtime not built)")
endif()

