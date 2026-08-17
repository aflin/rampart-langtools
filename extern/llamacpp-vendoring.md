# Vendoring notes: extern/llama.cpp

## Current version

| | |
|---|---|
| **Upstream tag** | `b10446` |
| **Upstream commit** | `adb55e5148dc93bcdca7212a2d1df3ccc422959a` |
| **Tag date** | 2026-08-15 |
| **ggml version** | 0.20.0 |
| **Source** | https://github.com/ggml-org/llama.cpp |

Previous: `b9494` / `c8d6a0063613ebd359b0030273746e05658dd605` / 2026-06-03 / ggml 0.13.1.
That upgrade spanned 647 tagged releases and cost **two struct fields at three call
sites** — see "What the b9494 -> b10446 upgrade actually cost" below, which is the
best available estimate of what the *next* one will cost.

## `extern/llama.cpp` is upstream + a few small Metal patches

We keep llama.cpp as close to upstream as possible — **almost all** customization lives
*outside* `extern/llama.cpp`:
- `extern/extern.cmake` and the top-level `CMakeLists.txt` (build integration)
- `extern/llamacpp/wrapper/` (our C++<->C generation shim — a separate dir)
- `rampart-llamacpp.c` (the module: API usage + runtime env workarounds)

But we **do** carry a small set of **source patches inside `extern/llama.cpp`** — the
older-macOS Metal fixes below. They are Objective-C changes in ggml-metal, so they
cannot live outside the vendored tree. **These MUST be reapplied after re-vendoring.**

### Vendored patches — now kept as files in `extern/patches/`

```
extern/patches/ggml-metal-context.patch    ggml_metal_get_tensor_async
extern/patches/ggml-metal-device.patch     ggml_metal_buffer_{set,get}_tensor  (2 hunks)
```

All three hunks are the same fix: on macOS 11, `newBufferWithBytesNoCopy` returns nil for
non-page-aligned host memory (macOS 12+ tolerates it), which upstream then `GGML_ASSERT`s
-> crash in embed/rerank. The fix: on nil, fall back to a copying/staging Metal buffer.
Each is marked in-source with `// rampart-langtools:`.

Apple Silicon takes the `is_shared` memcpy path in the two `device.m` functions, so it
only needs the `context.m` one; **Intel Macs (non-unified memory) need all three.**

Apply after replacing the tree:

```sh
cd extern/llama.cpp
for p in ../patches/ggml-metal-*.patch; do patch -p1 --no-backup-if-mismatch < "$p"; done
grep -rn "rampart-langtools:" ggml/src/ggml-metal/    # expect 3 hits
```

**Both the patched tree and the patch files are committed. That is not redundant.** The
vendored tree is committed *already patched* (it always has been — that is what builds).
The patch files exist for the *next* re-vendor: they let you `rm -rf extern/llama.cpp`,
drop in the new tag and reapply, instead of reverse-engineering the edits out of the old
tree with `grep -rln "rampart-langtools:"`. Without them committed, the next person has
nothing to apply.

**Re-anchor them after every re-vendor**, so the committed patches always match the
committed tree with zero offsets:

```sh
P=/path/to/pristine-clone-of-current-tag
for f in ggml/src/ggml-metal/ggml-metal-context.m ggml/src/ggml-metal/ggml-metal-device.m; do
    diff -u "$P/$f" "extern/llama.cpp/$f" \
      | sed -e "1s|^--- .*|--- a/$f|" -e "2s|^+++ .*|+++ b/$f|" > "extern/patches/$(basename $f .m).patch"
done
```

Then round-trip them — this proves the patch set fully reproduces the vendored tree:

```sh
rm -rf /tmp/rt && cp -a "$P" /tmp/rt && rm -rf /tmp/rt/.git
cd /tmp/rt && for p in .../extern/patches/*.patch; do patch -p1 < "$p"; done
diff -rq /tmp/rt/ggml/src/ggml-metal .../extern/llama.cpp/ggml/src/ggml-metal   # must be silent
```

They are deliberately kept as **files, not in-tree edits you have to go find**. Across
b9494 -> b10446 (647 releases, `device.m` itself +177 lines) they reapplied with nothing
but line offsets:

```
patching file ggml/src/ggml-metal/ggml-metal-context.m     (context.m was byte-identical between the two tags)
patching file ggml/src/ggml-metal/ggml-metal-device.m
Hunk #1 succeeded at 1863 (offset 157 lines).
Hunk #2 succeeded at 1932 (offset 157 lines).
```

If upstream ever fixes the `newBufferWithBytesNoCopy`-nil behavior, delete the patch files
and this section. Better still: upstream them (cf. #16266) and the burden goes to zero.

### Verify the version / find our patches

```sh
git clone --depth 1 -b b10446 https://github.com/ggml-org/llama.cpp /tmp/llamacpp-b10446
diff -rq --strip-trailing-cr extern/llama.cpp /tmp/llamacpp-b10446
#   -> the clone's .git, PLUS exactly the two patched files
#      (ggml-metal-context.m, ggml-metal-device.m). Anything else differing is unexpected.
grep -rln "rampart-langtools:" extern/llama.cpp    # lists our patched files
```
(Note: the in-tree `build/.../build-info.cpp` reports a bogus build number — that's the
*langtools* repo's git leaking in because the vendored copy has no `.git`. Ignore it.
Configure also warns "Git repository not found" for the same reason; harmless.)

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

The shim rides **libcommon**, the fastest-churning llama.cpp API, so the standing advice
was to expect mechanical breaks here every upgrade. b9494 -> b10446 says otherwise: it
touches only 8 libcommon entry points and reads exactly one field of `common_chat_params`
(`.prompt`), so the heavy `chat.h` churn of that range (roles became an enum,
`thinking_end_tag` -> `thinking_end_tags`, `message_spans` -> `message_delimiters`, plus
+493/-94 across the in-tree jinja engine, which b9494 already had) missed us entirely.
**Keeping the shim's libcommon
surface small is what makes upgrades cheap — resist widening it.**

## Re-vendoring procedure (next upgrade)

1. **Probe first.** Shallow-fetch the candidate tag and syntax-check our two sources
   against its headers — two minutes, and it tells you the whole source-level cost:
   ```sh
   git clone --depth 1 -b bNNNN https://github.com/ggml-org/llama.cpp /tmp/lc
   gcc -fsyntax-only -std=gnu11 -I/tmp/lc/include -I/tmp/lc/ggml/include \
       -Iextern/llamacpp/wrapper -I. -I/usr/local/src/rampart/src/include rampart-llamacpp.c
   g++ -fsyntax-only -std=c++17 -I/tmp/lc/include -I/tmp/lc/ggml/include \
       -I/tmp/lc/common -I/tmp/lc/vendor -Iextern/llamacpp/wrapper \
       extern/llamacpp/wrapper/llama_gen_shim.cc
   ```
   Also `patch -p1 --dry-run` the Metal patches against `/tmp/lc`, and re-check the
   silent-corruption invariants under "Batched chunk embedding" below — those do NOT
   show up as compile errors.
2. Pick the tag; record tag + commit + ggml version in the table above.
3. Replace the contents of `extern/llama.cpp` wholesale with that tag's tree (drop its
   `.git`), then **reapply `extern/patches/ggml-metal-*.patch`**.
4. Clean-build (`rm -rf build && cmake .. && make`) and work through the
   integration points below.
5. Run the gate: `pu_test/pu_gate.sh` + `rampart llamacpp-test.js` (embed/gen) and
   `pu_test/base_rr_bge.js` (rerank). Vectors/scores must match the prior baseline —
   see "Numerical drift" for what "match" honestly means.

## What the b9494 -> b10446 upgrade actually cost

Done on firefly (32-core x86, RTX 4070 Ti, driver 570.211.01), 2026-08-15, on branch
`llamacpp-b10446`. Recorded because it is the only real datapoint for sizing the next one.

**Total source-level break: one API change, three call sites.**

- `llama_model_params` dropped `use_mmap` / `use_mlock` / `use_direct_io` for a single
  `enum llama_load_mode` (`AUTO/NONE/MMAP/MLOCK/MMAP_MLOCK/DIRECT_IO`).
  `rampart-llamacpp.c` folds the JS `useMmap`/`useMlock` booleans back into it, so the
  **module's JS API is unchanged**. `LLAMA_LOAD_MODE_AUTO` resolves to (mmap on, mlock
  off) — exactly the old defaults — so we only override when a script asks, which also
  preserves llama.cpp's "auto" load-mode log line. `llama_gen_shim.cc`'s model-cache key
  keys on `load_mode` instead of the two booleans.
- Everything else held: all 58 `llama_*`/`ggml_*` symbols we call still exist with the
  same signatures, target names `llama-common`/`llama-common-base` are unchanged, and
  `libcpp-httplib.a` never needed adding to the link line (`ldd` shows only
  libc/libm/libstdc++/libgcc; zero httplib symbols in the module).

Two build-glue items were cleaned up during the upgrade but are **not** b10446 changes —
both were already true at b9494 and simply hadn't been noticed:

- `LLAMA_CURL` has been deprecated (`llama_option_depr(WARNING …)`) since at least b9494;
  our `set(LLAMA_CURL OFF …)` had been a no-op producing a CMake warning for a while.
  Removed.
- `LLAMA_OPENSSL` has existed and defaulted **ON** since at least b9494, so every build
  in that era ran `find_package(OpenSSL)` and linked `OpenSSL::SSL/::Crypto` into
  `libcpp-httplib.a`. Nothing we link references `download.cpp`, so it never reached our
  `.so` — but `extern.cmake` now FORCEs it OFF so the trap can't spring on a builder where
  OpenSSL 3 is present and the deploy tier lacks it.

Build: **1m13s at -j32, zero errors, zero warnings, first try** (native CPU).

### Build matrix — all six oven variants, zero errors

Three targets touch llama.cpp/ggml and all three were rebuilt in every variant.
`rampart-faiss`, `rampart-sentencepiece` and `rampart-onnx` do not link llama.cpp or
ggml, so they are legitimately untouched by a llama.cpp upgrade — don't waste oven time
rebuilding them (`LT_TARGET=<one target> ./build.sh build <variant>`; it takes exactly
one target, so it's three invocations per variant).

| oven variant | toolchain | llamacpp | clip | umbrella | runtime-verified |
|---|---|---|---|---|---|
| `cpu`       | manylinux2014, gcc-11, glibc 2.17 | ok | ok | ok | embed dim 384, norm 1.0 |
| `cpu_2_28`  | manylinux_2_28, glibc 2.28        | ok | ok | ok | embed dim 384, norm 1.0 |
| `cu11`      | CUDA 11.8, glibc 2.17             | ok | ok | ok | embed dim 384, norm 1.0 |
| `cu11_2_28` | CUDA 11.8, glibc 2.28             | ok | ok | ok | embed dim 384, norm 1.0 |
| `cu12`      | CUDA 12.8, glibc 2.28             | ok | ok | ok | embed dim 384, norm 1.0 |
| `cu13`      | CUDA 13.0, glibc 2.28             | ok | ok | ok | **build only** — firefly has no `libcudart.so.13` |

Two results worth keeping:
- **CUDA 11.8 still compiles ggml 0.20.** That was the likeliest failure point in the whole
  upgrade (cu11 is the oldest toolkit we ship, and `extern.cmake` already documents that it
  can't handle Hopper PDL intrinsics). It went through clean.
- **glibc tier does not perturb numerics.** The umbrella's first embedding component came out
  identical within a backend and differed only across backends: cpu/cpu_2_28 both -0.038241,
  cu11/cu11_2_28 both -0.038830, cu12 -0.038287. Backend selects the kernel; the tier doesn't.

Building the `cpu`/`cu11` (2_17-tier) variants requires `/usr/local/rampart-2_17/bin/rampart`
installed on the builder — `build.sh` bails on a prerequisite check before compiling anything
if it's missing. That failure looks nothing like a build break; don't chase it.

### Numerical drift — GPU is bit-identical, CPU is not

Same corpus, batching **off** on both sides, so this is pure upstream-version drift:

| model | weights | CPU drift | CPU cosine | GPU (cu12) |
|---|---|---|---|---|
| all-MiniLM-L6      | F16     | 1.14e-4 | 0.9999998 | **bit-identical** |
| bge-small-en-v1.5  | Q8_0    | 3.24e-3 | 0.999905  | **bit-identical** |
| nomic-embed-text-v1.5 | Q4_K_M | 1.56e-2 | 0.98888 | **bit-identical** |
| bge-m3             | Q8_0    | —       | —         | **bit-identical** |

The GPU dumps compare equal by md5, all four models. So **the drift is a CPU-backend
phenomenon** — ggml's CPU quant/SIMD paths churned across 0.13 -> 0.20 while the CUDA
kernels for these ops did not. On CPU the drift tracks quantization exactly
(f16 << q8_0 << Q4_K_M), the same ordering as batching drift: it is kernel selection,
and low-bit quants are the sensitive ones.

Practical consequence: a **CPU-built Q4_K_M index is not bit-comparable across an
upgrade** (nomic at cos 0.989 — retrieval won't visibly break, but it isn't free).
Rebuild the index with whichever version serves it, or serve it from GPU.

Generation was **token-identical** on both backends (through the entirely rewritten
jinja chat-template stack), and rerank rankings were unchanged — scores identical on
GPU, shifted on CPU in line with the table above.

Batching was unaffected: speedups on the 4070 Ti stayed 2.2x / 1.7x / 1.2x
(bge-small / nomic / bge-m3), and re-sweeping `batchTokens` reproduced the b9494 curve,
confirming **512 is still the largest value that never regresses**.

## Integration points to RE-CHECK on every upgrade

These are the things that broke (or could break). Apart from the in-tree Metal patches
noted above, these are not patches to llama.cpp — they are how *we* build and call it.

### Build glue — `extern/extern.cmake`
- **`GGML_OPENMP OFF`**. ggml uses its own pthread threadpool. Two reasons:
  (a) macOS libomp aborts when a non-initial thread runs OpenMP regions alongside the
  JS event loop; (b) avoids a second static libomp colliding with faiss/rampart-sql
  (`OMP: Error #15`). Keep this OFF. (libomp still enters the build via faiss only.)
- **`LLAMA_BUILD_COMMON ON`**. As a subproject, common is OFF by default;
  we link `libllama-common`. SERVER/EXAMPLES/TESTS/TOOLS stay off.
- **`LLAMA_OPENSSL OFF`**. Upstream defaults it ON (true since at least b9494), which makes
  cpp-httplib do `find_package(OpenSSL)` and link `OpenSSL::SSL/::Crypto`. Keeps a system
  OpenSSL dependency out of a portable build. Re-check it still exists and still defaults ON.
- **Do not set `LLAMA_CURL`** — deprecated via `llama_option_depr` since at least b9494;
  downloads go through the vendored cpp-httplib. Setting it only produces a CMake warning.
- **Target/lib name churn.** b9494 renamed the common target `common` -> `llama-common`
  (builds `libllama-common.a` + `libllama-common-base.a`); unchanged at b10446. If an
  upgrade renames targets again, update `add_dependencies(...)` and the `lib*.a` paths in
  `CMakeLists.txt`. Symptom of a stale name: dlopen "symbol not found
  common_sampler_init" or "target llama-common does not exist" at configure.
- **`GGML_NATIVE`** handling for portable ARM builds (rpi) — leave as-is.
- **No mtmd.** We do NOT build `tools/mtmd` (vision lives in rampart-clip). `LLAMA_BUILD_TOOLS`
  stays off and there are no `libmtmd.a` links. Don't let an upgrade reintroduce it.
  b10446 also adds `LLAMA_BUILD_APP` and a standalone `LLAMA_BUILD_MTMD` hook; both
  default off as a subproject. Keep it that way.

### Lib paths / shim — top-level `CMakeLists.txt`
- Static libs linked by hard path; update if upstream renames/moves them:
  `common/libllama-common.a`, `common/libllama-common-base.a`, `src/libllama.a`,
  `ggml/src/libggml.a` `libggml-cpu.a` `libggml-base.a`, and per-backend
  `ggml/src/ggml-metal/libggml-metal.a`, `ggml-blas/libggml-blas.a`,
  `ggml-cuda/libggml-cuda.a`.
- `vendor/cpp-httplib/libcpp-httplib.a` is linked PRIVATE by `llama-common` (as of b9494
  already). We do NOT link it and do not need to — `common.cpp` doesn't include `http.h`, so the
  objects we pull never reference httplib. If a future version moves download/HTTP code
  into a TU we do pull, the symptom is undefined `httplib::` symbols at link; the fix is
  adding that archive (and then `LLAMA_OPENSSL OFF` really matters).
- The generation shim is an OBJECT lib and MUST be consumed via
  `$<TARGET_OBJECTS:llama_gen_shim_obj>` in BOTH `add_library` targets (a plain target
  name does not pull an OBJECT lib's objects on macOS -> missing `lgen_*` at dlopen).

### Generation shim — `extern/llamacpp/wrapper/llama_gen_shim.{h,cc}`
- Built against **libcommon** (`common_sampler_*`, `common_chat_templates_*`,
  `string_find_partial_stop`, `common_batch_*`). Survived b9494 -> b10446 with one
  mechanical fix (the `load_mode` cache key). Keep the surface small.
- The header `#include`s `llama.h` and stores `llama_model_params`/`llama_context_params`
  by value in `lgen_engine_params`; if upstream changes those structs, recompile covers it.
- Rule of thumb from the last upgrade: **accessor functions were 100% stable; public
  struct fields were not.** Both breaks were direct writes to struct members. Where
  llama.h offers a setter or accessor, prefer it over touching the struct.

### Module API + runtime workarounds — `rampart-llamacpp.c`
- **Embed/rerank context fix:** contexts must set `kv_unified = true`,
  `n_seq_max = llama_max_parallel_sequences()`, `llama_set_embeddings(ctx, true)`, and use
  `llama_decode` (not `llama_encode`). Without it, b9494+ gives NaN/0.0/segfault. RE-VERIFY
  embed vectors + rerank scores match baseline after any upgrade.
- **`GGML_METAL_NO_RESIDENCY=1`** set at module load on macOS (`rampart-llamacpp.c` ~2727).
  Avoids a ggml-metal residency-set `GGML_ASSERT` in its static destructor at process exit
  (a buffer outlives `exit()` because the initGen engine tears down async). Opt out:
  `RAMPART_METAL_RESIDENCY=1`. If upstream removes residency sets, this becomes a no-op.
- **initGen VM gate (Apple Silicon).** `initGen` is blocked when `arm64 && macOS < 15 && in a VM`
  (detected via `kern.hv_vmm_present` / `rp_in_vm()`): paravirtual Metal in those VMs can't build
  the generation kernels (nil pipeline -> crash). Real hardware at any version, and VMs on
  macOS 15+, are allowed; embed/rerank are unaffected; x86_64 is not gated. Override:
  `RAMPART_FORCE_GEN=1`. RE-CHECK on upgrade — if upstream changes Metal kernel specialization,
  the gate may be loosenable.
- **`GGML_CUDA_DISABLE_GRAPHS=1`** set for the batched embed/rerank paths (~1649). Avoids a
  CUDA-graph-cache VRAM leak under many sequential embeds. Opt out: `RAMPART_LLAMA_CUDA_GRAPHS=1`.
- `nCtx: 0/-1 => model n_ctx_train` (matches llama-server); resolved in the shim/embed builder.
- **`useMmap`/`useMlock` are ours, not upstream's** (b10446+). They are folded into
  `llama_model_params.load_mode` in `parse_common_opts`. If a future version reshapes
  `load_mode` again, that mapping is the single place to fix. Upstream also offers
  `LLAMA_LOAD_MODE_DIRECT_IO`, which we do not expose — a `loadMode` string option is the
  obvious way to if it's ever wanted.
- **Batched chunk embedding (EXPERIMENTAL, `ll_decode_pooled_batch`).** A document's chunks are
  packed as independent sequences into one `llama_decode` (chunk *j* -> `seq_id j`, positions
  restarting at 0 per sequence, every token flagged as an output). This leans on four upstream
  behaviors. **None of these produce a compile error if they change — they silently corrupt
  output. RE-VERIFY each on upgrade.** (All four re-verified at b10446.)
  1. **Variable-length packing is legal.** With `kv_unified = true`, `llama_kv_cache::init_batch`
     picks `split_simple`, which has no equal-length requirement. If a future version routes the
     unified path through `split_equal` instead, ragged chunks would be split across ubatches.
     *b10446: `llama-kv-cache.cpp:709` — `n_stream == 1 ? split_simple(n_ubatch) : split_equal(...)`,
     and `unified` sets `n_stream = 1`. Holds.*
  2. **Per-sequence pooled output.** `llama_get_embeddings_seq(ctx, j)` returns sequence *j*'s
     pooled vector, and `embd_seq` is cleared at the top of the NEXT decode — so all K vectors
     must be copied out before returning (we do). *b10446: still a per-seq map, still cleared;
     the fills are now `ggml_backend_tensor_get_async` with an explicit sync before free, which
     does not change our copy-out-before-return contract. Holds.*
  3. **`n_ubatch >= n_tokens` is asserted only for NON-causal models.** For causal embedders
     (qwen3-embedding) the assert is skipped, and a batch split across ubatches silently
     overwrites each sequence's pooled output. `ll_batch_budget()` enforces the invariant
     ourselves rather than relying on the assert — keep it that way.
     *b10446: `llama-context.cpp:1713` — `GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all))`.
     Still skipped for causal. Holds.*
  4. **Positions must restart per sequence.** BERT-style models index a learned position table of
     `n_ctx_train` rows with the raw position and do NOT clamp (`src/models/bert.cpp`), and
     CLS/LAST pooling select by min/max position within a sequence.
     *b10446: `bert.cpp` position handling untouched (only an `n_layer` -> `n_layer()` accessor
     rename). Holds.*
- **`batchTokens` (default 512) is a PERFORMANCE constant, not a correctness one.** No-KV encoders
  compute attention over the whole packed batch as one NxN matrix (cross-sequence pairs are only
  masked), so cost grows with batch tokens x model width while the overhead saved grows linearly.
  Uncapped, bge-m3 measured **0.65x — slower than unbatched** on an RTX 4070 Ti (0.64x at b10446).
  Note flash attention is auto-probed per (model x backend) and changes this cost curve: on one
  4070 Ti / cu12 build, nomic and bge-m3 resolved FA **on** while bge-small (head_dim 32) resolved
  **off**. Re-measure the optimum after an upgrade or on new hardware; `claude-work/gpu-batch-test/`
  (`sweep.js`, `TOKENS=1`) does exactly that. *b10446 sweep reproduced the b9494 curve; 512 stands.*

### Known non-issues (don't chase on upgrade)
- `Qwen3-Reranker` returns 0.0 — the GGUF isn't a valid reranking conversion; the official
  `llama-embedding` also returns 0.0. Not a regression. (`pu_test/FUTURE-qwen3-reranker.md`.)
- macOS Metal **embed/rerank/gen now work on macOS 11+** thanks to the vendored Metal patches
  above — verified on real Apple Silicon and Intel macOS 11. (Older macOS previously crashed on
  the `newBufferWithBytesNoCopy`-nil issue; cf. upstream #16266.) The only remaining macOS-version
  limit is **gen in a VM on macOS < 15** (the initGen VM gate). Make sure the README reflects this.
- ggml's minor version moves fast (0.13 -> 0.20 in ~950 build numbers, roughly one bump per
  10 days). It is a release counter on ggml's own cadence, not a semantic-stability signal —
  our public-ggml surface is 7 symbols (backend enumeration + logging) and none of them moved.
