# Vendoring notes: extern/llama.cpp

## Current version

| | |
|---|---|
| **Upstream tag** | `b11479` |
| **Upstream commit** | `42c787e8c191d49c01c757f4b7029d69ec0bf2c8` (tag b11479; extracted via `git archive` from a local clone) |
| **Tag date** | 2026-10-07 |
| **ggml version** | 0.26.0 |
| **Source** | https://github.com/ggml-org/llama.cpp |

### libmtmd is now linked (2026-10-07): multimodal embedding, Phase 1

`tools/mtmd` is built, using upstream's own `LLAMA_BUILD_MTMD` hook (library only;
`LLAMA_BUILD_TOOLS` stays off). It is linked into **rampart-llamacpp and the umbrella
only**, never through `LLAMA_LIBS`: rampart-clip links `LLAMA_LIBS` and carries its own
(monatis) clip code. Its symbols are hidden with `-Wl,--exclude-libs,libmtmd.a:libvendor-hash.a`
(0 `clip_`/`mtmd_` exports). `MTMD_VIDEO` is OFF because upstream's video path shells out to ffmpeg.
The umbrella defines `LANGTOOLS_MAIN_INCLUDE`, so `rampart-llamacpp.c` skips its own
includes there and `rampart-langtools.c` must `#include "mtmd.h"` itself.

**Phase 2 (same day): `initEmbed(model, {mmproj, imageTokens})` + `embedMediaToNumbers/Fp32Buf/Fp16Buf({text, image, audio})`.**
It builds one pooled vector from one mixed batch: `mtmd_helper_bitmap_init_from_file/_buf`,
then `mtmd_tokenize` (add_special/parse_special true, as llama-server), then
`mtmd_encode_chunk` under a mutex, then `llama_batch_ext_add_token/_add_embd` +
`_set_pos` + `_set_output_embd`, then `llama_process(DECODE)`, then
`llama_get_embeddings_seq`. Media rows are `llama_model_n_embd_inp()` wide. The
projector is a refcounted shared entry (`lt_mproj_*`, keyed by path, model, gpu and
imageTokens), referenced per context like the model (checked addref on a thread
rebuild, released in `emb_free` before the model). The **silent traps on a re-vendor**
are: the media-row width (`n_embd_inp`, not `n_embd`), and `mtmd_get_output_embd()`
being reused by the next encode (copy it out under the lock).
Parity (firefly): equal to upstream llama-server to 4 decimals on every item, CPU and
cu13. Vs Google's reference: text 1.0, audio >=0.9988, images mean 0.9865 (budget
280), interleaved 0.956-0.988; retrieval identical. Speed: about 0.1 s/image on a
4070 Ti; on CPU 3.8 s/image at the default 4 threads, 1.8 s at `threadsBatch: 16`.

**On every re-vendor, `mtmd.h` is now a contact surface too.** Today we use
`mtmd_context_params_default`, `mtmd_init_from_file`, `mtmd_free`,
`mtmd_support_vision/audio`, `mtmd_get_audio_sample_rate` and `mtmd_log_set`. Add the
`-fsyntax-only` probe's `-I.../tools/mtmd` include path.

The first consumer is `llamacpp.mediaInfo(model, mmproj[, opts])`, which is stateless
(load, report, free). Static archive members link only when referenced, so nothing
proves the link until something calls in.

Verified on firefly (cpu_2_28 and cu13 ovens, 6/6 clean): mediaInfo on embeddinggemma-2
reports vision, audio and 16 kHz on CPU and GPU. Error paths (missing mmproj, wrong
text model) throw without crashing. 5 load/free cycles and two threads at once both
pass. rampart-clip still works in the same process. `llamacpp-test.js` passes 37/37 on
both builds. Text vectors for 4 models are **bit-identical** to the pre-mtmd build.
Size cost: about +22 MB CPU (208 -> 231 MB unstripped), +24 MB cu13.
Not yet built: macOS, FreeBSD, ARM ovens, cu11/cu12.

### b11349 -> b11479 (2026-10-07): embeddinggemma-2, one API break, two real bugs

The reason for this upgrade: `gemma-embedding2` (EmbeddingGemma 2) landed upstream in
b11452 (PR #30054). b11349 rejects the GGUF with "unknown model architecture".

**Source break (one, in the gen shim):** `common_chat_parse()` takes a
`common_chat_input` (text + token ids aligned byte-for-byte) instead of a string
(upstream #29876). Fixed by wrapping: `common_chat_parse(common_chat_input(slot.generated), ...)`.
That constructor is upstream's own "plain text, no tokens" path. **Today it parses
identically**: `common_peg_parse_context` stores the tokens, but no parser reads them.
**If a future version starts matching special tokens by id**, accumulate a
`common_chat_input` with `append(piece, token)` the way `tools/server` does. Note that our
pieces use `special=false`, and `append` drops empty pieces.

Metal patches reapplied with offsets only (+21 / +477/+479) and were re-anchored. The round trip is silent.

**Two bugs found while testing, both fixed:**

1. **Embedding width: `llama_model_n_embd()` -> `llama_model_n_embd_out()`**
   (`rampart-llamacpp.c`, both `vec_dim` sites). embeddinggemma-2 has a 512 -> 768
   projection head, and every llama.cpp output buffer is sized by `n_embd_out`. We read
   only the first 512 of 768 values: unit-normalized, retrieval still "worked", and the
   vectors were silently wrong. `modelInfo()` already read `embedding_length_out`, so
   `embedDim` reported 768 while `initEmbed` returned 512. For every model without a
   projection head, `n_embd_out == n_embd`, so vectors are unchanged (verified bit-identical below).
2. **Stale CPU tier objects (`cmake/ggml-cpu-tiers.cmake`).** The namespaced
   `lt-cpu-tier-{haswell,skylakex,sse42}.o` and `x64-feats.o` rules had
   `DEPENDS <OBJECT library target>`. An OBJECT library has no output file, so that is
   ordering only, never a rerun. An incremental oven build after a re-vendor therefore
   linked **b11349 tier code into a b11479 ggml**. upstream #23671 inserted two slots into
   `ggml_backend_buffer_type_i`, so every Q4_K model segfaulted in the loader: the old
   repack buffer type's vtable was read with the new layout. (`x64` was fine because it
   depends on `ggml-cpu`, a real `.a`.) Fixed by adding `$<TARGET_OBJECTS:...>` to
   `DEPENDS`. **This bug predates b11479**: any earlier incremental re-vendor could have
   shipped stale tiers. Clean `build/oven-*` dirs never hit it.

**Verified on firefly (x86_64, RTX 4070 Ti, driver 580) 2026-10-07; 2_28 ovens:**

| check | result |
|---|---|
| oven builds `cpu_2_28`, `cu13` (llamacpp, clip, umbrella) | **6/6 clean, 0 errors** |
| `llamacpp-test.js` CPU / cu13 GPU | **37/37 / 37/37** |
| parity vs installed b11349 modules, 4 models (MiniLM f16, bge-m3 Q8_0, bge-base Q4_K_M, bge-small Q8_0), short + 6.3k-token doc | **bit-identical on CPU AND cu13** (maxdiff 0, same chunk counts) |
| Q4_K models after the tier fix (MiniLM, bge-m3, bge-base) | load + embed (were rc 139) |
| embeddinggemma-2 Q8_0 vs upstream `llama-embedding` b11479 (CPU) | **cos 1.00000000**, short and long doc |
| embeddinggemma-2 BF16 / UD-Q4_K_XL vs Q8_0 reference | cos 0.99989 / 0.995 |
| embeddinggemma-2 on cu13 | 768d, cos 0.9999 vs CPU ref, retrieval 3/3 |

**embeddinggemma-2 notes:**
- GGUF says `context_length = 262144` (the model card says 8K, shared across modalities). The
  existing 8192 auto-cap in both embed paths handles this; no special case needed.
- Uses the same prompts as embeddinggemma-300m: `task: search result | query: ` / `title: none | text: `.
- 768 output dims; Matryoshka 512/256/128 per the card. Truncate, then re-normalize.
- Slow for its size: 6.3k tokens = ~17 s CPU, ~7 s on a 4070 Ti. Upstream CUDA matches
  (6.4 s with flash attention, 4.6 s with `-fa off`). Cause: head dims 512 global / 256 SWA,
  where the CUDA flash-attention kernels are slow. This is upstream, not offload: 25/25 layers on GPU.
- Text only. Image/audio need mtmd + the `mmproj` GGUF, which the embed path doesn't do.
- Not tested: Metal, FreeBSD, ARM ovens, cu11/cu12.

### b10446 -> b11349 (2026-10-02): three breaks, all in the gen shim

`rampart-llamacpp.c` compiled **clean** — the module API and the whole
embed/chunk-batching path were untouched, because the C-level `llama_batch`,
`llama_batch_init/free`, `llama_decode` and `llama_get_embeddings_seq` are all
unchanged. The new `llama_batch_ext` is purely additive. Metal patches reapplied
with offsets only (+21, +479/+481). All four batching invariants and all four
chat-adapter traps held, and every libcommon symbol the adapter needs survived.

The three breaks, all in `llama_gen_shim.cc`:

1. **The `json` alias is gone.** The common headers no longer define `json`, and
   **re-aliasing it to nlohmann does not work** — it errors inside llama.cpp's own
   `common/json.h`, because of (3).
2. **`common_batch_clear` / `common_batch_add` removed**, replaced by methods on a
   `common_batch` class wrapping `llama_batch_ext`. Fixed by filling `llama_batch`
   directly (`lg_batch_clear` / `lg_batch_add`), the same lines the embed path
   already hand-rolls — which drops two more libcommon dependencies.
3. **The oaicompat interface moved from `nlohmann::ordered_json` to `common_json`**,
   a pimpl wrapper that deliberately hides the backing library; `chat.h` no longer
   mentions nlohmann at all. Nearly 1:1 for us: `parse_no_throw` replaces
   `parse(text, nullptr, false)`, and `is_discarded` / `is_array` / `array()` /
   `dump()` all carry the same names. We only move JSON across that boundary as
   text, so the dependency is just parse-from-string / dump-to-string.

**Verified on moe (DGX Spark GB10, aarch64) 2026-10-02, main branch, uncommitted:**

| check | result |
|---|---|
| oven builds (ARM: cpu, cpu_2_28, cu12, cu13) | **4/4 clean, 0 errors** |
| `llamacpp-test.js` on CPU | **37/37** |
| `llamacpp-test.js` on cu13 / GB10 GPU | **37/37** |
| embed vector parity vs b10446, 4 models | **PASS** — identical k/spans, 0 row misassignments |
| batching still correct under b11349 | **PASS** on all 4 models |
| GPU actually exercised | yes, 1206 MiB VRAM during gen |

**Also verified on `en` (Mac Studio, arm64, macOS 15, Metal) 2026-10-02:**

| check | result |
|---|---|
| cmake build with Metal | clean, 0 errors |
| Metal patches | all 3 markers in place after reapply |
| `llamacpp-test.js` | **37/37** |
| embed parity vs b10446 (MiniLM f16, bge-m3 Q8_0) | **BIT-IDENTICAL** — drift 0.0000e+00, cosine 1.000000000 |
| batching correct under b11349 | PASS (8.6e-5 / 2.0e-4) |

The bit-identical Metal result repeats what CUDA did across b9494 -> b10446: a
GPU backend produces the same vectors before and after, while the CPU backend
drifts with quantization. **The drift is a CPU-backend phenomenon, not a general
property of upgrading.** Useful operationally: a GPU-served index survives an
upgrade unchanged; a CPU-built low-bit one does not.

Upstream drift vs b10446 (batching off, CPU) tracks quantization exactly as the
b9494 -> b10446 upgrade did: f16 6.1e-5, bge-small q4_k_m 2.1e-3, bge-m3 Q8_0
2.8e-3, nomic Q4_K_M 1.4e-2 (cos 0.9908 — the usual low-bit outlier). A
CPU-built Q4_K_M index is still not bit-comparable across an upgrade.

**`batchTokens` re-swept on firefly (RTX 4070 Ti, cu12, driver 570.211.01)
2026-10-02: 512 still correct, no retune needed.** Five ggml minors did not move
the cost curve -- every cell is within noise of the b10446 measurement on the
same box:

| batchTokens | bge-small q8_0 | nomic Q4_K_M | bge-m3 Q8_0 |
|---|---|---|---|
| 256 | 1.79 -> 1.82 | 1.45 -> 1.49 | 1.12 -> 1.13 |
| **512 (default)** | 2.21 -> **2.23** | 1.73 -> **1.75** | 1.15 -> **1.17** |
| 1024 | 2.19 -> 2.24 | 1.79 -> **1.86** | 0.99 -> 1.00 |
| 2048 | 2.19 -> 2.23 | 1.26 -> 1.23 | 0.90 -> 0.91 |
| 8192 | 2.21 -> 2.23 | 1.27 -> 1.30 | 0.65 -> **0.65** |

512 remains the largest value that never regresses. nomic still peaks at 1024
(1.86x), but that setting costs bge-m3 (1.00x, and falling to 0.65x by 8192), so
512 is still the right compromise rather than the per-model optimum.

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
**As of 2026-08-17 that is no longer true, deliberately: the tool-calling +
reasoning adapter widened this surface on purpose**
The shim now reads ~8 fields of
`common_chat_params` and calls `common_chat_parse`,
`common_chat_msg_diff::compute_diffs`, the three `*_parse_oaicompat` helpers,
`common_chat_templates_support_enable_thinking` and `common_cpu_get_num_math`.
All of it is confined to ONE adapter block in `llama_gen_shim.cc` so the next
bump has one place to fix — do not let it spread. The probe in step 1 below is
the tripwire; it is no longer optional.

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

   **The syntax check covers the chat adapter too**, which is the widest and
   fastest-churning contact we have; a `chat.h` rename shows up here as a compile
   error in seconds. But four things it CANNOT catch, all of which fail silently —
   re-verify them by running `llamacpp-test.js` (tool + reasoning sections) against
   the candidate build:
   1. **`common_params_sampling.generation_prompt` must be set** whenever a
      tool-calls grammar is set, or the grammar starts misaligned against the
      already-prefilled assistant prompt: `toolChoice` misbehaves and the
      generation prompt leaks into the output. Costs nothing to set; silent if
      omitted.
   2. **The serialized PEG parser must be loaded explicitly.**
      `common_chat_parser_params(chat_params)` copies only `format` and
      `generation_prompt` — NOT `parser`. Omit `parser.load(p.parser)` and every
      PEG-format model degrades to content-only parsing: no error, no tool calls.
   3. **`preserved_tokens` needs tokenizing** (`vector<string>` on the params side,
      `set<llama_token>` on the sampler side; single-token strings only).
   4. **`common_grammar` is a struct**, not a string:
      `{COMMON_GRAMMAR_TYPE_TOOL_CALLS, str}`.
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

### GPU detection: IGPU is a GPU (unified-memory parts)

`ggml_backend_cuda_device_get_type()` returns `GGML_BACKEND_DEVICE_TYPE_IGPU`
rather than `..._GPU` whenever `cudaDeviceProp.integrated` is set -- i.e. on
every unified-memory CUDA part: **DGX Spark (GB10), Jetson**. Verified on moe:

    ggml_backend_dev_count = 2
      dev[0] CUDA0   type=2 (IGPU)
      dev[1] CPU     type=0 (CPU)

`lt_gpu_in_use()` originally matched `..._GPU` only, so on the Spark it returned
0 despite a fully working CUDA backend (1206 MiB in use during gen). That was
wrong in three ways, in descending order of seriousness:

1. **It skipped the post-fork refusal.** `LT_FORK_REFUSAL` is gated on this
   predicate at four call sites. On a Spark/Jetson a pre-fork handle was USED in
   the child rather than refused -- crashing the CUDA runtime instead of raising
   a clean error. This is the reason the fix matters: a correctness hole on
   exactly the hardware people run `rampart-server` daemon mode on.
2. It left embed chunk batching off (auto resolves on this predicate).
3. It misreported `embedDefaults().gpuInUse`.

**The fix splits the predicate, because the callers do not all want the same
question answered.** `lt_gpu_in_use()` now counts GPU **or** IGPU -- used by fork
safety and reporting. A second `lt_dedicated_gpu_in_use()` (GPU only) drives the
chunk-batching auto-default, because that is a performance question whose answer
depends on memory architecture:

| model | discrete (RTX 4070 Ti) | unified (GB10) |
|---|---|---|
| bge-small | 2.2x faster | **0.89x (slower)** |
| nomic | 1.7x faster | 1.02x |
| bge-m3 | 1.2x faster | **0.89x (slower)** |

Batching amortizes per-decode launch/transfer overhead, which a discrete part
pays over PCIe; a unified-memory part pays neither, so the packed batch's
quadratic attention cost dominates and batching becomes a small net LOSS.
Auto-on therefore follows **dedicated VRAM**, not merely "has a GPU". Confirmed
on the GB10: auto 809 ms == explicit false 812 ms, vs explicit true 928 ms.

(llama.cpp's own `llama-bench` tests `GPU || IGPU`, which is precedent for
counting IGPU -- but note it is not making a batching decision.)

### Thread defaults — `lgen_default_n_threads()` in the shim
- Gen resolves its default thread count through libcommon's `common_cpu_get_num_math()`
  rather than a constant. Exposed through the shim's C ABI because it is C++ and
  `rampart-llamacpp.c` is C. Note `GGML_DEFAULT_N_THREADS` (4) is only the raw ggml
  struct fallback — llama.cpp's own tools resolve `-1` through this heuristic.
- **FreeBSD needs our own branch and always will, until upstream adds one.**
  `common_cpu_get_num_physical_cores()` has branches for AIX, Linux, macOS and
  Windows; FreeBSD falls through to the generic tail
  (`hardware_concurrency()`, then `/2 if > 4`), which assumes SMT is always on and
  therefore HALVES a non-SMT box. We read `sysctl kern.smp.cores` first. On upgrade,
  check whether upstream has grown a `__FreeBSD__` branch; if so, drop ours.
- Verified values: 16 on a 16-core Linux box; 16 (not 32) on a 32-thread/16-core box,
  correctly discounting hyperthread siblings; 16 (not 20) on an Apple Silicon Mac
  Studio, excluding its 4 E-cores (macOS uses `hw.perflevel0.physicalcpu`); 4 on the
  FreeBSD VM.

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
  **off**. Re-measure the optimum after an upgrade or on new hardware.

### Known non-issues (don't chase on upgrade)
- **`toolChoice:"required"` is not enforced by b10446.** Upstream builds the correct
  non-lazy tool-calls grammar (2000 bytes, 0 triggers) and hands it to
  `common_sampler_init`, but generation comes out unconstrained. Reproduced with zero
  rampart code, with and without `grammar_first`, on two
  models. We pass the choice through correctly. `llamacpp-test.js` asserts only that
  the request is accepted, so it will start failing usefully if upstream fixes this —
  at which point tighten the test.
- `Qwen3-Reranker` returns 0.0 — the GGUF isn't a valid reranking conversion; the official
  `llama-embedding` also returns 0.0. Not a regression. (`pu_test/FUTURE-qwen3-reranker.md`.)
- macOS Metal **embed/rerank/gen now work on macOS 11+** thanks to the vendored Metal patches
  above — verified on real Apple Silicon and Intel macOS 11. (Older macOS previously crashed on
  the `newBufferWithBytesNoCopy`-nil issue; cf. upstream #16266.) The only remaining macOS-version
  limit is **gen in a VM on macOS < 15** (the initGen VM gate). Make sure the README reflects this.
- ggml's minor version moves fast (0.13 -> 0.20 in ~950 build numbers, roughly one bump per
  10 days). It is a release counter on ggml's own cadence, not a semantic-stability signal —
  our public-ggml surface is 7 symbols (backend enumeration + logging) and none of them moved.
