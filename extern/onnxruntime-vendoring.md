# Vendoring notes: extern/onnxruntime + extern/onnxruntime-extensions

## Current versions

| | onnxruntime | onnxruntime-extensions |
|---|---|---|
| **Upstream version** | v1.27.0 (`VERSION_NUMBER`) | v0.15.0 (`version.txt`) |
| **Source** | https://github.com/microsoft/onnxruntime | https://github.com/microsoft/onnxruntime-extensions |
| **License** | MIT | MIT |

## The vendored trees are upstream + a small patch set

Almost all customization lives **outside** the vendored trees:

- `extern/extern.cmake` — the `onnxruntime_ep` / `onnxext_ep` ExternalProjects,
  `ONNX_LIBS` per-platform link flags, the GPU-flavor gate.
- `extern/onnxruntime/rampart-build-cpu.sh` / `rampart-build-cuda.sh` — drive
  ORT's own `build.sh`, then merge its ~80 static archives into
  `libonnxruntime_{core,deps}.a` (GNU `ar -M` MRI on Linux/FreeBSD,
  Apple `libtool -static` on macOS).  Also pins
  `MACOSX_DEPLOYMENT_TARGET=11.0` on Darwin and passes
  `Patch_EXECUTABLE=gpatch` on FreeBSD.  These are OUR scripts, not upstream's
  (upstream `build.sh` sits beside them untouched).
- `extern/onnxruntime-extensions/rampart-build-ext.sh` — OUR script: builds the
  tokenizers-only static config and folds it into one relocatable
  `onnxext_all.o` (`ld -r --whole-archive` on Linux/FreeBSD,
  `ld -r -arch <cpu> -all_load` on macOS; protobuf bundling is per GPU/CPU
  flavor — see the script header).
- `extern/onnxruntime/wrapper/onnx_shim.{h,cc}` — our C++→C shim (separate dir).
- `rampart-onnx.c` / `rampart-onnx.map` — the module itself + its export list.

But we **do** carry source patches inside `extern/onnxruntime` — the FreeBSD
fixes below.  **These MUST be reapplied after re-vendoring.**
`extern/onnxruntime-extensions` is currently patch-free.

### Vendored patches (reapply on every re-vendor)

Both are FreeBSD build fixes (upstream ships real `__FreeBSD__` code paths —
`thr_self()` etc. — but they have bit-rotted; nobody CI-builds them).  Each is
marked in-source with a `rampart:` comment.

1. **`onnxruntime/core/common/logging/logging.cc`** — the `#if __FreeBSD__`
   block contained `#include "logging.h"`, which is unresolvable (the header
   lives in the separated `include/onnxruntime/...` tree, not beside the .cc)
   and redundant (`core/common/logging/logging.h` is already included at the
   top of the file).  The include is removed.
   Symptom if missing: `fatal error: 'logging.h' file not found`.

2. **`onnxruntime/core/platform/posix/env.cc`** — the thread-affinity block in
   `ThreadMain` assumes glibc (`cpu_set_t`, `CPU_SET`, `syscall(SYS_gettid)`);
   FreeBSD spells these `pthread_np.h`/`cpuset_t` and has no `gettid`.
   `!defined(__FreeBSD__)` is added to the block's platform guard — the same
   treatment upstream gives Apple/Android/wasm/AIX.  Affinity is an
   optimization rampart never uses (sessions default to 1 intra-op thread).
   Symptom if missing: `use of undeclared identifier 'pthread_setaffinity_np'`
   / `'SYS_gettid'`.

3. **`cmake/onnxruntime_providers_coreml.cmake`** — macOS static-build fix
   (`--use_coreml` + no `--build_shared_lib`, our combination; upstream only
   CI-builds CoreML as a shared lib or Apple framework): the static-build
   `install(TARGETS coreml_proto ...)` lacks `EXPORT onnxruntimeTargets`,
   while `onnxruntime_providers_coreml` (which links it) IS in that export
   set — CMake's generate step then fails.  `EXPORT onnxruntimeTargets` is
   added, and (required by the export) `coreml_proto`'s bare
   `${CMAKE_CURRENT_BINARY_DIR}` PUBLIC include is wrapped in
   `$<BUILD_INTERFACE:...>`.  Symptoms if missing: `CMake Error:
   install(EXPORT "onnxruntimeTargets" ...) includes target
   "onnxruntime_providers_coreml" which requires target "coreml_proto" that
   is not in any export set.` / `Target "coreml_proto"
   INTERFACE_INCLUDE_DIRECTORIES property contains path ... which is
   prefixed in the build directory.`

## Unified module + drop-in GPU runtimes (2026-07)

There is ONE `rampart-onnx.so` per platform/tier -- no `_cpu`/`_cu12`/`_cu13`
module flavors.  The module statically contains a full CPU ORT (single-file
CPU deployment) and, at first use, looks for an OPTIONAL external runtime
directory beside itself:

```
modules/rampart-onnx.so         one module, CPU-complete alone
modules/onnx-cu12/              drop-in CUDA-12 runtime (may coexist with cu13)
    libonnxruntime.so.1(.27.0), libonnxruntime_providers_{shared,cuda}.so, sm.list
modules/onnx-cu13/              drop-in CUDA-13 runtime
```

Selection ladder (onnx_shim.cc; `onnx.runtimeInfo()` reports the outcome):
1. `RAMPART_ONNX_RUNTIME` = `cpu` | `cu12` | `cu13` | `/abs/dir` overrides all.
2. NVIDIA driver probe (`libcuda.so.1` -> `cuDriverGetVersion`, driver API
   only): newest runtime the driver supports wins (driver >= 13 -> cu13 else
   cu12); among driver-compatible dirs, one whose `sm.list` contains the
   device's exact compute capability is preferred (a miss only demotes --
   ORT's PTX may still JIT).  A dir that fails to load drops to the next.
3. Floor: the built-in static CPU ORT.  Any selection failure ends in CPU
   inference, never a crash.

The external core is dlopen'd `RTLD_LOCAL` (its hidden protobuf/abseil cannot
clash with the internal, version-script-hidden copies); ORT dlopens the
provider libs from the core's own directory (their names are hardcoded
strings inside the core -- do NOT rename them; the subdirs are what make
cu12/cu13 coexist).  A loaded runtime must satisfy `GetApi(ORT_API_VERSION)`
or it is rejected with both version strings printed.

Build-wise: cpu flavors build the module (static ORT + extensions + shim);
cu12/cu13 flavors build ONLY the shared core + providers and install
`modules/onnx-<flavor>/` (plus `sm.list` from the baked arch list).  The
2_28 tier's module comes from the `cpu_2_28` oven variant.

## Platform build requirements

| | Linux | macOS (11.0+) | FreeBSD |
|---|---|---|---|
| CMake | >= 3.28 | >= 3.28 | >= 3.28 |
| python3 | any modern | any modern (build machine only) | any modern |
| extra packages | — | Xcode CLT | **`pkg install patch`** (gpatch — BSD `patch` rejects the GNU flags in ORT's FetchContent dependency patches; `rampart-build-cpu.sh` auto-detects it) |
| archive merge | GNU `ar -M` | Apple `libtool -static` | llvm-ar (`ar -M` works) |
| whole-archive | `-Wl,--whole-archive` | `-Wl,-force_load,<a>` | lld (`--whole-archive` works) |
| symbol hiding | `--version-script=rampart-onnx.map` | `-exported_symbol _duk_open_module` + `_rp_onnx_embed_*` | `--version-script` (lld) |

macOS note: `MACOSX_DEPLOYMENT_TARGET=11.0` is exported by both build scripts
so ORT/extensions objects match the module's `-mmacosx-version-min=11.0`
(verify with `otool -l rampart-onnx.so | grep -A3 LC_BUILD_VERSION` →
`minos 11.0`).  **Changing the target does NOT trigger recompiles** — wipe
`build/extern/onnxruntime{,-extensions}` and the `*_ep-prefix` stamp dirs.

## Re-vendoring checklist

1. Replace `extern/onnxruntime` / `extern/onnxruntime-extensions` with the new
   upstream trees (keep `rampart-build-*.sh` and `wrapper/`).
2. Reapply the vendored patches above (check whether upstream fixed them).
3. Rebuild from a clean `build/extern/` on all three platforms; run
   `onnx-test.js` (35 tests) on each.
4. Update the version table at the top of this file.
