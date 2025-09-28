cmake_minimum_required(VERSION 3.15)
project(rampart_langtools C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/extern/extern.cmake)

# Common compile flags
add_compile_options(-Wall -g -O3)

find_program(RAMPART_EXECUTABLE rampart)
if(NOT RAMPART_EXECUTABLE)
  message(FATAL_ERROR "rampart not found in PATH")
endif()

execute_process(
  COMMAND ${RAMPART_EXECUTABLE} -c "console.log(process.installPath)"
  RESULT_VARIABLE _res
  OUTPUT_VARIABLE RP_PATH
  ERROR_VARIABLE  _err
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _res EQUAL 0)
  message(FATAL_ERROR "rampart failed: ${_err}")
endif()

# Include directories (common)
include_directories(
  ${RP_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/faiss
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/wrapper
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


if(APPLE)
  # Extra includes on macOS
  include_directories(
    /opt/homebrew/include
  )

  # Extra compile flags
  add_compile_options(-Wno-deprecated-declarations)

    list(APPEND LIBS
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk/usr/lib/libcurl.tbd
        /opt/homebrew/opt/libomp/lib/libomp.a
        ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-blas/libggml-blas.a
        ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-metal/libggml-metal.a
    )

  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-headerpad_max_install_names -undefined dynamic_lookup -Wl,-rpath,@loader_path -Wl,-rpath,/usr/local/lib -Wl,-rpath,/opt/homebrew/lib -framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
  )

else()
  # Linux
    list(APPEND LIBS
        /usr/lib/x86_64-linux-gnu/libomp5.so
        /usr/lib/x86_64-linux-gnu/libopenblas.a
        /usr/lib/x86_64-linux-gnu/libgfortran.so.5
        ${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp/ggml/src/ggml-cuda/libggml-cuda.a
        /usr/lib/x86_64-linux-gnu/libcudart.so
        /usr/lib/x86_64-linux-gnu/libcublas.so
        /usr/lib/x86_64-linux-gnu/libcuda.so
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-soname,rampart-langtools.so"
    )
endif()


add_library(rampart-langtools SHARED rampart-langtools.c)
set_target_properties(
    rampart-langtools
    PROPERTIES SUFFIX ".so"
    PREFIX ""
)

add_dependencies(rampart-langtools
  llama
  common
  ggml
  llama-gguf
  sentencepiece
  faiss
  faiss_c
)

# Link with all libs
target_link_libraries(rampart-langtools PRIVATE ${LIBS})

install(TARGETS rampart-langtools
  DESTINATION "${RP_PATH}/modules"
)

install(CODE "execute_process(COMMAND strip -S \"${RP_PATH}/modules/rampart-langtools.so\")")

