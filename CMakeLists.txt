cmake_minimum_required(VERSION 3.15)
project(rampart_langtools C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common compile flags
add_compile_options(-Wall -g -O3)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")

if(APPLE)

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    option(LT_ENABLE_GPU "Enable GPU acceleration" ON)
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

find_program(RAMPART_EXECUTABLE rampart)
if(NOT RAMPART_EXECUTABLE)
  message(FATAL_ERROR "rampart not found in PATH")
endif()

# Get the install path of rampart
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


include_directories(
  ${RP_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/faiss
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/ggml/include/
)

# Libraries
set(LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libcommon.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/src/libllama.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-cpu.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/libggml-base.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/sentencepiece/src/libsentencepiece.a
    spm_c_wrapper_obj
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/c_api/libfaiss_c.a
    ${CMAKE_CURRENT_BINARY_DIR}/extern/faiss/faiss/libfaiss.a
    -lpthread
    -lm
)

set(LLAMA_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/common/libcommon.a
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
)

# llama.cpp
add_library(rampart-llamacpp SHARED
    rampart-llamacpp.c
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

# sentencepiece
add_library(rampart-vecutils SHARED
    rampart-vecutils.c
)

if(LT_ENABLE_GPU)
    target_compile_definitions(rampart-langtools PRIVATE LT_ENABLE_GPU=1)
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
  )

  set(LIB_FILES
    libomp.a
    libcurl.tbd
  )
  find_libraries_in_dirs(FOUND_LIBS FILES ${LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  set(LLAMA_LIB_FILES
    libcurl.tbd
  )
  find_libraries_in_dirs(LLAMA_FOUND_LIBS FILES ${LLAMA_LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  set(FAISS_LIB_FILES
    libomp.a
  )

  find_libraries_in_dirs(FAISS_FOUND_LIBS FILES ${FAISS_LIB_FILES} DIRS ${LIB_DIRS} PREFIX LIB_)

  if(LT_ENABLE_GPU)
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
  else()
    list(APPEND LIBS
          ${FOUND_LIBS}
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-blas/libggml-blas.a
    )
    list(APPEND LLAMA_LIBS
          ${LLAMA_FOUND_LIBS}
          ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-blas/libggml-blas.a
    )
  endif()
  target_link_options(rampart-langtools PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-langtools.so"
  )

  target_link_options(rampart-llamacpp PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-llamacpp.so"
  )

  target_link_options(rampart-faiss PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-faiss.so"
  )

  target_link_options(rampart-sentencepiece PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-sentencepiece.so"
  )

  target_link_options(rampart-vecutils PRIVATE
    "-Wl,-headerpad_max_install_names"
    "-Wl,-undefined,dynamic_lookup"
    "-Wl,-install_name,@rpath/rampart-vecutils.so"
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
  )

  if(LT_ENABLE_GPU)
    set(LIB_FILES
      libomp5.so
      libopenblas.a
      libgfortran.so.5
      libcudart.so
      libcublas.so
      libcuda.so
    )
  else()
    set(LIB_FILES
      libomp5.so
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
    list(APPEND FAISS_LIBS
      ${FOUND_LIBS}
    )
  endif()

  target_link_options(rampart-langtools PRIVATE
    "-Wl,-soname,@rpath/rampart-langtools.so"
  )

  target_link_options(rampart-llamacpp PRIVATE
    "-Wl,-soname,@rpath/rampart-llamacpp.so"
  )

  target_link_options(rampart-faiss PRIVATE
    "-Wl,-soname,@rpath/rampart-faiss.so"
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

set_target_properties(
    rampart-langtools
    PROPERTIES SUFFIX ".so"
    PREFIX ""
)

set_target_properties(
    rampart-llamacpp
    PROPERTIES SUFFIX ".so"
    PREFIX ""
)

set_target_properties(
    rampart-faiss
    PROPERTIES SUFFIX ".so"
    PREFIX ""
)

set_target_properties(
    rampart-sentencepiece
    PROPERTIES SUFFIX ".so"
    PREFIX ""
)

set_target_properties(
    rampart-vecutils
    PROPERTIES SUFFIX ".so"
    PREFIX ""
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
      common
      ggml
    )
    add_dependencies(rampart-llamacpp
      llama
      common
      ggml
    )
  endif()

  if(NOT EXISTS "${SP_LIB}")
    add_dependencies(rampart-langtools
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
    common
    ggml
    sentencepiece
    faiss
    faiss_c
  )

  add_dependencies(rampart-llamacpp
    llama
    common
    ggml
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

install(TARGETS rampart-langtools
  DESTINATION "${RP_PATH}/modules"
)

install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-langtools.so\")")
install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-llamacpp.so\")")

