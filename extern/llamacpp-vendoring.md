# Vendoring notes: extern/llama.cpp

## Current version

| | |
|---|---|
| **Upstream tag** | `b9494` |
| **Upstream commit** | `c8d6a0063613ebd359b0030273746e05658dd605` |
| **Tag date** | 2026-06-03 |
| **Source** | https://github.com/ggml-org/llama.cpp |

## `extern/llama.cpp` is upstream + a few small Metal patches

We keep llama.cpp as close to upstream as possible — **almost all** customization lives
*outside* `extern/llama.cpp`:
- `extern/extern.cmake` and the top-level `CMakeLists.txt` (build integration)
- `extern/llamacpp/wrapper/` (our C++↔C generation shim — a separate dir)
- `rampart-llamacpp.c` (the module: API usage + runtime env workarounds)

But we **do** carry a small set of **source patches inside `extern/llama.cpp`** — the
older-macOS Metal fixes below. They are Objective-C changes in ggml-metal, so they
cannot live outside the vendored tree. **These MUST be reapplied after re-vendoring.**

### Vendored patches (reapply on every re-vendor)

All three are the same fix: on macOS 11, `newBufferWithBytesNoCopy` returns nil for
non-page-aligned host memory (macOS 12+ tolerates it), which upstream then
`GGML_ASSERT`s → crash in embed/rerank. The fix: on nil, fall back to a copying/staging
Metal buffer. Each is marked in-source with `// rampart-langtools:`.

| file | function | fallback used on nil |
|------|----------|----------------------|
| `ggml/src/ggml-metal/ggml-metal-context.m` | `ggml_metal_get_tensor_async` (~351) | `newBufferWithLength` staging buf + blit + memcpy |
| `ggml/src/ggml-metal/ggml-metal-device.m`  | `ggml_metal_buffer_set_tensor` (~1695) | copying `newBufferWithBytes` |
| `ggml/src/ggml-metal/ggml-metal-device.m`  | `ggml_metal_buffer_get_tensor` (~1761) | `newBufferWithLength` staging buf + memcpy |

Apple Silicon takes the `is_shared` memcpy path in the two `device.m` functions, so it
only needs the `context.m` one; **Intel Macs (non-unified memory) need all three.**

### Verify the version / find our patches

```sh
git clone --depth 1 -b b9494 https://github.com/ggml-org/llama.cpp /tmp/llamacpp-b9494
diff -rq --strip-trailing-cr extern/llama.cpp /tmp/llamacpp-b9494
#   -> the clone's .git, PLUS exactly the two patched files above
#      (ggml-metal-context.m, ggml-metal-device.m). Anything else differing is unexpected.
grep -rln "rampart-langtools:" extern/llama.cpp    # lists our patched files
```
(Note: the in-tree `build/.../build-info.cpp` reports a bogus `b81` / `LLAMA_COMMIT
9bd2ff0` — that's the *langtools* repo's git leaking in because the vendored copy has no
`.git`. Ignore it.)

## `extern/llamacpp/` — rampart's own additions (NOT upstream)

Mind the naming: **`extern/llama.cpp`** (with a dot) is the vendored upstream;
**`extern/llamacpp`** (no dot) is rampart's own code. The latter is an aid layer we
add *on top of* llama.cpp; it is not part of any upstream tree, so on re-vendoring it
is **kept**, not replaced.

| file | ~lines | purpose |
|------|--------|---------|
| `wrapper/llama_gen_shim.h`   | 136 | C ABI for the multi-session, slot-based, continuous-batching generation engine (llama-server style). Lets the pure-C module `rampart-llamacpp.c` drive a C++ engine without seeing libcommon. |
| `wrapper/llama_gen_shim.cc`  | 683 | The engine: one shared `llama_context` split into `nSeqMax` slots with continuous batching; all llama.cpp + **libcommon** contact (sampling chain, chat templates, partial-stop handling) lives here. Includes a CPU-device-pin fallback when GPU context init fails (no-Metal VM/headless). |
| `wrapper/llama_gen_macos.mm` | 28  | macOS-only: flips the Cocoa/Foundation runtime into multithreaded mode (detaches an `NSThread`) so Metal/Foundation are thread-safe when inference runs on a worker thread. Compiled only on Apple. |

Built as the `llama_gen_shim_obj` OBJECT lib (in `extern/extern.cmake`) and linked
into `rampart-llamacpp.so` and the umbrella via `$<TARGET_OBJECTS:llama_gen_shim_obj>`.
Because the engine rides on the fast-churning **libcommon** API, expect to fix
mechanical breaks in `llama_gen_shim.cc` on each upgrade (see the re-check list below).

## Re-vendoring procedure (next upgrade)

1. Pick an upstream tag `bNNNN`; record the tag + commit here.
2. Replace the contents of `extern/llama.cpp` wholesale with that tag's tree, then
   **reapply the Metal patches** from "Vendored patches" above (find them in the old
   tree with `grep -rln "rampart-langtools:" extern/llama.cpp`). If upstream has fixed
   the `newBufferWithBytesNoCopy`-nil-on-macOS-11 behavior, drop the patches instead and
   update this doc.
3. Clean-build (`rm -rf build && cmake .. && make`) and work through the
   integration points below.
4. Run the gate: `pu_test/pu_gate.sh` + `rampart llamacpp-test.js` (embed/gen) and
   `pu_test/base_rr_bge.js` (rerank). Vectors/scores must match the prior baseline.

## Integration points to RE-CHECK on every upgrade

These are the things that broke (or could break) across the last upgrade. Apart from the
in-tree Metal patches noted above, these are not patches to llama.cpp — they are how *we*
build and call it.

### Build glue — `extern/extern.cmake`
- **`GGML_OPENMP OFF`** (line ~13). ggml uses its own pthread threadpool. Two reasons:
  (a) macOS libomp aborts when a non-initial thread runs OpenMP regions alongside the
  JS event loop; (b) avoids a second static libomp colliding with faiss/rampart-sql
  (`OMP: Error #15`). Keep this OFF. (libomp still enters the build via faiss only.)
- **`LLAMA_BUILD_COMMON ON`** (line ~73). As a subproject, common is OFF by default;
  we link `libllama-common`. SERVER/EXAMPLES/TESTS/TOOLS stay off.
- **Target/lib name churn.** b9494 renamed the common target `common` → `llama-common`
  (now builds `libllama-common.a` + `libllama-common-base.a`). If an upgrade renames
  targets again, update `add_dependencies(...)` and the `lib*.a` paths in
  `CMakeLists.txt` (see below). Symptom of a stale name: dlopen "symbol not found
  common_sampler_init" or "target llama-common does not exist" at configure.
- **`GGML_NATIVE`** handling for portable ARM builds (rpi) — leave as-is.
- **No mtmd.** We do NOT build `tools/mtmd` (vision lives in rampart-clip). `LLAMA_BUILD_TOOLS`
  stays off and there are no `libmtmd.a` links. Don't let an upgrade reintroduce it.

### Lib paths / shim — top-level `CMakeLists.txt`
- Static libs linked by hard path; update if upstream renames/moves them:
  `common/libllama-common.a`, `common/libllama-common-base.a`, `src/libllama.a`,
  `ggml/src/libggml.a` `libggml-cpu.a` `libggml-base.a`, and per-backend
  `ggml/src/ggml-metal/libggml-metal.a`, `ggml-blas/libggml-blas.a`,
  `ggml-cuda/libggml-cuda.a`.
- The generation shim is an OBJECT lib and MUST be consumed via
  `$<TARGET_OBJECTS:llama_gen_shim_obj>` in BOTH `add_library` targets (a plain target
  name does not pull an OBJECT lib's objects on macOS → missing `lgen_*` at dlopen).

### Generation shim — `extern/llamacpp/wrapper/llama_gen_shim.{h,cc}`
- Built against **libcommon** (`common_sampler_*`, `common_chat_templates_*`,
  `string_find_partial_stop`, `common_batch_*`) — the fastest-churning llama.cpp API.
  Expect to fix mechanical breaks here on every upgrade.
- The header `#include`s `llama.h` and stores `llama_model_params`/`llama_context_params`
  by value in `lgen_engine_params`; if upstream changes those structs, recompile covers it.

### Module API + runtime workarounds — `rampart-llamacpp.c`
- **Embed/rerank context fix (b9494):** contexts must set `kv_unified = true`,
  `n_seq_max = llama_max_parallel_sequences()`, `llama_set_embeddings(ctx, true)`, and use
  `llama_decode` (not `llama_encode`). Without it, b9494 gives NaN/0.0/segfault. RE-VERIFY
  embed vectors + rerank scores match baseline after any upgrade.
- **`GGML_METAL_NO_RESIDENCY=1`** set at module load on macOS (`rampart-llamacpp.c` ~2727).
  Avoids a ggml-metal residency-set `GGML_ASSERT` in its static destructor at process exit
  (a buffer outlives `exit()` because the initGen engine tears down async). Opt out:
  `RAMPART_METAL_RESIDENCY=1`. If upstream removes residency sets, this becomes a no-op.
- **initGen VM gate (Apple Silicon).** `initGen` is blocked when `arm64 && macOS < 15 && in a VM`
  (detected via `kern.hv_vmm_present` / `rp_in_vm()`): paravirtual Metal in those VMs can't build
  the generation kernels (nil pipeline → crash). Real hardware at any version, and VMs on
  macOS 15+, are allowed; embed/rerank are unaffected; x86_64 is not gated. Override:
  `RAMPART_FORCE_GEN=1`. RE-CHECK on upgrade — if upstream changes Metal kernel specialization,
  the gate may be loosenable.
- **`GGML_CUDA_DISABLE_GRAPHS=1`** set for the batched embed/rerank paths (~1649). Avoids a
  CUDA-graph-cache VRAM leak under many sequential embeds. Opt out: `RAMPART_LLAMA_CUDA_GRAPHS=1`.
- `nCtx: 0/-1 => model n_ctx_train` (matches llama-server); resolved in the shim/embed builder.

### Known non-issues (don't chase on upgrade)
- `Qwen3-Reranker` returns 0.0 — the GGUF isn't a valid reranking conversion; the official
  `llama-embedding` also returns 0.0. Not a regression. (`pu_test/FUTURE-qwen3-reranker.md`.)
- macOS Metal **embed/rerank/gen now work on macOS 11+** thanks to the vendored Metal patches
  above — verified on real Apple Silicon and Intel macOS 11. (Older macOS previously crashed on
  the `newBufferWithBytesNoCopy`-nil issue; cf. upstream #16266.) The only remaining macOS-version
  limit is **gen in a VM on macOS < 15** (the initGen VM gate). Make sure the README reflects this.
