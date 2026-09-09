# ggml-cpu-tiers.cmake -- ONE module, several ggml-cpu ISA tiers, chosen at runtime.
#
# ggml compiles its whole CPU backend per instruction-set tier and has no
# simsimd-style function multi-versioning; upstream's answer (GGML_CPU_ALL_VARIANTS
# + GGML_BACKEND_DL) ships ~14 dlopen'd ggml-cpu-*.so and needs BUILD_SHARED_LIBS.
# We want one self-contained module per rampart module, so instead:
#
#   * ggml's own ggml-cpu target is pinned to the x86-64 baseline (extern.cmake) and
#     becomes the `x64` tier -- the floor that runs on any x86-64 CPU;
#   * each extra tier recompiles the SAME sources with that tier's flags;
#   * namespace-tier.sh renames every symbol a tier defines, so the tiers coexist;
#   * ggml-cpu-dispatch.c defines the two symbols the rest of the build calls and
#     forwards them to the best tier, chosen by ggml's own CPUID scorer.
#
# Nothing runs before the choice is made: ggml-cpu has no global constructors, and
# the scorers are compiled WITHOUT arch flags (see below), so the AVX-512 tier never
# executes an instruction on a CPU that lacks AVX-512.

if(APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    return()
endif()

option(LT_GGML_CPU_TIERS "build several ggml-cpu ISA tiers into each module" ON)
if(NOT LT_GGML_CPU_TIERS)
    return()
endif()

# tier -> compiler flags / -D defines, mirroring ggml's own x86 variant list
# (ggml/src/CMakeLists.txt).  The defines are what ggml's sources and its CPUID
# scorer test; the -m flags are what the compiler may emit.
set(LT_TIERS sse42 haswell skylakex)

set(LT_TIER_sse42_FLAGS   -msse4.2)
set(LT_TIER_sse42_DEFS    GGML_SSE42)

set(LT_TIER_haswell_FLAGS -msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2)
set(LT_TIER_haswell_DEFS  GGML_SSE42 GGML_F16C GGML_FMA GGML_BMI2 GGML_AVX GGML_AVX2)

set(LT_TIER_skylakex_FLAGS ${LT_TIER_haswell_FLAGS}
                           -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)
set(LT_TIER_skylakex_DEFS  ${LT_TIER_haswell_DEFS} GGML_AVX512)

# take the source list, includes and defines from ggml's own target so a llama.cpp
# upgrade that adds a file needs no change here
get_target_property(_lt_cpu_srcs ggml-cpu SOURCES)
get_target_property(_lt_cpu_dir  ggml-cpu SOURCE_DIR)
get_target_property(_lt_cpu_inc  ggml-cpu INCLUDE_DIRECTORIES)
get_target_property(_lt_cpu_defs ggml-cpu COMPILE_DEFINITIONS)
if(NOT _lt_cpu_srcs OR NOT _lt_cpu_dir)
    message(FATAL_ERROR "ggml-cpu target not found; include this after llama.cpp")
endif()
if(NOT _lt_cpu_defs)
    set(_lt_cpu_defs "")
endif()

# ggml sets _GNU_SOURCE / _XOPEN_SOURCE with add_compile_definitions(), i.e. on its
# own DIRECTORY -- targets declared here would not inherit them, and ggml-cpu.c then
# fails on SCHED_BATCH.  Pull the directory scope in too, and link ggml-base for its
# interface includes, exactly as ggml's own backend targets do.
get_directory_property(_lt_cpu_dirdefs DIRECTORY ${_lt_cpu_dir} COMPILE_DEFINITIONS)
if(NOT _lt_cpu_dirdefs)
    set(_lt_cpu_dirdefs "")
endif()

set(_lt_srcs_abs "")
foreach(_s ${_lt_cpu_srcs})
    if(IS_ABSOLUTE "${_s}")
        list(APPEND _lt_srcs_abs "${_s}")
    else()
        list(APPEND _lt_srcs_abs "${_lt_cpu_dir}/${_s}")
    endif()
endforeach()

set(_lt_feats_src "${_lt_cpu_dir}/ggml-cpu/arch/x86/cpu-feats.cpp")
set(LT_CPU_TIER_OBJECTS "")

foreach(_tier ${LT_TIERS})
    # the tier's kernels, built with its instruction set
    add_library(lt-cpu-${_tier} OBJECT ${_lt_srcs_abs})
    target_include_directories(lt-cpu-${_tier} PRIVATE ${_lt_cpu_inc})
    target_compile_definitions(lt-cpu-${_tier} PRIVATE
        ${_lt_cpu_defs} ${_lt_cpu_dirdefs} ${LT_TIER_${_tier}_DEFS})
    target_link_libraries(lt-cpu-${_tier} PRIVATE ggml-base)
    target_compile_options(lt-cpu-${_tier} PRIVATE ${LT_TIER_${_tier}_FLAGS} -w)
    set_target_properties(lt-cpu-${_tier} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # The CPUID scorer, compiled with the tier's DEFINES BUT NOT ITS FLAGS (and no
    # LTO).  It runs before any dispatch decision, so a scorer built for its own tier
    # would trap on the very CPU it exists to rule out.  Upstream does the same in
    # ggml_add_cpu_backend_features(); getting this wrong SIGILLs at load.
    add_library(lt-cpu-${_tier}-feats OBJECT ${_lt_feats_src})
    target_include_directories(lt-cpu-${_tier}-feats PRIVATE ${_lt_cpu_inc})
    target_compile_definitions(lt-cpu-${_tier}-feats PRIVATE
        ${_lt_cpu_defs} ${_lt_cpu_dirdefs} ${LT_TIER_${_tier}_DEFS} GGML_BACKEND_DL)
    target_link_libraries(lt-cpu-${_tier}-feats PRIVATE ggml-base)
    target_compile_options(lt-cpu-${_tier}-feats PRIVATE -fno-lto -w)
    set_target_properties(lt-cpu-${_tier}-feats PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set(_tier_obj "${CMAKE_CURRENT_BINARY_DIR}/lt-cpu-tier-${_tier}.o")
    add_custom_command(OUTPUT ${_tier_obj}
        COMMAND ${CMAKE_COMMAND} -E env
                LD=${CMAKE_LINKER} NM=${CMAKE_NM} OBJCOPY=${CMAKE_OBJCOPY}
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace-tier.sh
                ${_tier} ${_tier_obj}
                $<TARGET_OBJECTS:lt-cpu-${_tier}> $<TARGET_OBJECTS:lt-cpu-${_tier}-feats>
        DEPENDS lt-cpu-${_tier} lt-cpu-${_tier}-feats
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace-tier.sh
        COMMAND_EXPAND_LISTS
        COMMENT "namespacing ggml-cpu tier ${_tier}")
    list(APPEND LT_CPU_TIER_OBJECTS ${_tier_obj})
endforeach()

# the baseline tier IS ggml's own ggml-cpu (pinned to x86-64 in extern.cmake)
set(_x64_obj "${CMAKE_CURRENT_BINARY_DIR}/lt-cpu-tier-x64.o")
add_custom_command(OUTPUT ${_x64_obj}
    COMMAND ${CMAKE_COMMAND} -E env
            LD=${CMAKE_LINKER} NM=${CMAKE_NM} OBJCOPY=${CMAKE_OBJCOPY}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace-tier.sh
            x64 ${_x64_obj} --whole-archive $<TARGET_FILE:ggml-cpu>
    DEPENDS ggml-cpu ${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace-tier.sh
    COMMENT "namespacing ggml-cpu tier x64 (baseline)")
list(APPEND LT_CPU_TIER_OBJECTS ${_x64_obj})

# ggml's baseline build has no scorer of its own (upstream only compiles cpu-feats.cpp
# under GGML_BACKEND_DL), so give the x64 tier one.  No defines => it accepts any CPU
# with score 1, which is exactly the floor behaviour we want.
add_library(lt-cpu-x64-feats OBJECT ${_lt_feats_src})
target_include_directories(lt-cpu-x64-feats PRIVATE ${_lt_cpu_inc})
target_compile_definitions(lt-cpu-x64-feats PRIVATE
    ${_lt_cpu_defs} ${_lt_cpu_dirdefs} GGML_BACKEND_DL)
target_link_libraries(lt-cpu-x64-feats PRIVATE ggml-base)
target_compile_options(lt-cpu-x64-feats PRIVATE -fno-lto -w)
set_target_properties(lt-cpu-x64-feats PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_x64_feats_obj "${CMAKE_CURRENT_BINARY_DIR}/lt-cpu-tier-x64-feats.o")
add_custom_command(OUTPUT ${_x64_feats_obj}
    COMMAND ${CMAKE_OBJCOPY} --redefine-sym ggml_backend_score=ggml_backend_score_x64
            $<TARGET_OBJECTS:lt-cpu-x64-feats> ${_x64_feats_obj}
    DEPENDS lt-cpu-x64-feats
    COMMAND_EXPAND_LISTS
    COMMENT "namespacing ggml-cpu tier x64 scorer")
list(APPEND LT_CPU_TIER_OBJECTS ${_x64_feats_obj})

# The dispatcher + every tier.  An OBJECT library, NOT a static one: archive members
# are only pulled in to satisfy a reference already seen, and the reference to
# ggml_backend_cpu_reg comes from libggml.a, which is listed AFTER this on the link
# line.  As a static lib the dispatcher would silently not be pulled -- and, with
# libggml-cpu.a still on the line, the un-namespaced baseline would satisfy the
# reference instead, leaving a module that looks fine and never dispatches.  Objects
# are unconditional, so this cannot happen.  (CMakeLists.txt drops libggml-cpu.a from
# LIBS/LLAMA_LIBS when this target exists, since the x64 tier IS that archive.)
add_library(lt-ggml-cpu-tiers OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-cpu-dispatch.c)
set_source_files_properties(${LT_CPU_TIER_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)

# NOTE: the namespaced tier objects must be added to each module DIRECTLY, not to the
# object library above -- $<TARGET_OBJECTS:> expands only to objects a target itself
# COMPILES, so pre-built .o files listed as its sources are silently dropped.  That
# produced a module which linked and loaded and then failed at first use with
# "undefined symbol: ggml_backend_score_skylakex".  LT_GGML_CPU_TIER_SOURCES carries
# them; CMakeLists.txt adds it, plus the dispatcher objects, to every ggml module.
set(LT_GGML_CPU_TIER_SOURCES ${LT_CPU_TIER_OBJECTS}
    CACHE INTERNAL "namespaced ggml-cpu tier objects")
target_include_directories(lt-ggml-cpu-tiers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/ggml/include)
set_target_properties(lt-ggml-cpu-tiers PROPERTIES POSITION_INDEPENDENT_CODE ON)

# one target owns the generation, so the consuming modules just depend on it
add_custom_target(lt-cpu-tiers-gen DEPENDS ${LT_CPU_TIER_OBJECTS})

message(STATUS "rampart-langtools: ggml CPU tiers = x64 ${LT_TIERS} (runtime dispatch)")
