/* clip_shim.h -- pure C ABI over a modern-ggml CLIP (two-tower: image AND text
 * encoders in a shared space).  Keeps rampart-clip.c free of any C++/ggml
 * headers, exactly as llama_gen_shim.h / onnx_shim.h front their engines.
 *
 * The implementation is derived from clip.cpp (https://github.com/monatis/clip.cpp),
 * Copyright (c) 2023 Yusuf Sarigoz, MIT License -- see ./LICENSE.
 *
 * clip.cpp (monatis/clip.cpp) was ported off its bundled ancient ggml onto the
 * SHARED ggml that rampart-llamacpp already builds (libggml/-base/-cpu, plus the
 * cuda/metal backends in a GPU build) -- so there is exactly ONE ggml in the
 * langtools tree.  The graph math is unchanged; only the scaffolding was
 * modernized (backend + gallocr; inputs set after allocation).
 *
 * Lifecycle mirrors rampart-onnx's embed handle: a process-global, path-keyed,
 * refcounted cache shares ONE set of read-only weights across every rampart
 * thread (thread copies of the JS handle carry the same pointer).  Each thread
 * lazily builds its OWN compute context (backend + graph allocator); the weights
 * are shared.  Handles are process-lifetime (release decrements a refcount but
 * never frees) -- the same trade rampart-onnx/llamacpp make for heavy embedding
 * models, so a re-load of the same path is instant. */
#ifndef CLIP_SHIM_H
#define CLIP_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct clip_handle clip_handle;

/* Load (or share, from the cache) a CLIP GGUF model.  Refcount++ on the shared
 * handle.  Returns NULL and fills err[] on failure. */
clip_handle *clip_load(const char *path, char *err, size_t errlen);

/* Refcount bookkeeping.  clip_acquire is for a JS handle thread-copy; clip_release
 * decrements.  Weights are NOT freed (process-lifetime cache) -- see the header
 * note -- so a release simply records that one holder went away. */
void clip_acquire(clip_handle *h);
void clip_release(clip_handle *h);

int clip_dim(clip_handle *h);         /* shared-space embedding dim (512 / 768 / ...) */
int clip_has_text(clip_handle *h);
int clip_has_vision(clip_handle *h);
/* 1 if this model's weights and compute live on a GPU (CUDA, or macOS Metal), 0 if
 * it runs on the CPU -- either a CPU-only build, no usable device, or a GPU the
 * arch/driver guard or the op probe ruled out (which also warns via clip_warn). */
int clip_on_gpu(clip_handle *h);

/* Embed an image FILE (load + preprocess + encode) into out[clip_dim], on the
 * CALLING thread's own compute context.  normalize!=0 => L2-normalized.
 * Returns 0 on success, -1 and fills err[] otherwise. */
int clip_embed_image_file(clip_handle *h, const char *path, float *out,
                          int normalize, char *err, size_t errlen);

/* Same, from in-memory image BYTES (a decoded-format blob: JPEG/PNG/... -- the
 * file's contents, not raw pixels).  `buf` is borrowed, const, and need not be
 * NUL-terminated; `len` is authoritative and must be > 0 and <= INT_MAX. */
int clip_embed_image_mem(clip_handle *h, const void *buf, size_t len, float *out,
                         int normalize, char *err, size_t errlen);

/* Embed a text string into out[clip_dim], on the calling thread's context. */
int clip_embed_text(clip_handle *h, const char *text, float *out,
                    int normalize, char *err, size_t errlen);

/* Cosine similarity (dot of unit vectors) of two fp32 vectors. */
float clip_similarity(const float *a, const float *b, int dim);

/* WARNINGS: defined in rampart-clip.c (the side with duktape); the shim calls it
 * for non-fatal problems (e.g. a GPU that can't run this build -> CPU fallback).
 * Routes to `this.errMsg`; never reaches stdout/stderr (RAMPART_CLIP_DEBUG aside). */
void clip_warn(const char *fmt, ...);

/* Captured ggml log (the informational firehose -- ORT-style CUDA-init banner,
 * etc. -- routed off stderr into a buffer).  clip_log_dup returns a malloc'd copy
 * ("" if empty); caller frees.  Exposed to JS as clip.getLog() / clip.clearLog(). */
char *clip_log_dup(void);
void  clip_log_clear(void);

#ifdef __cplusplus
}
#endif
#endif /* CLIP_SHIM_H */
