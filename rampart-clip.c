/* rampart-clip.c -- duktape module: CLIP image+text embedding in a shared space.
 *
 *   var clip  = require("rampart-clip");
 *   var model = clip.initEmbed("model.gguf");        // (clip.load = alias)
 *   var iv = model.embedImageToFp16Buf("cat.jpg");   // + Fp32Buf / Numbers
 *   var tv = model.embedTextToFp16Buf("a photo of a cat");
 *   var s  = model.similarity(iv, tv);               // cosine, -1..1
 *   var d  = model.dimension;                        // 512 / 768 / ...
 *   model.destroy();
 *
 * Wraps a modern-ggml two-tower CLIP (extern/clip/clip_shim) that links the SAME
 * ggml rampart-llamacpp builds -- one ggml in the tree.  The native handle is a
 * process-global, path-keyed, REFCOUNTED cache entry: every rampart thread copy
 * of the model object shares ONE read-only weight set and lazily builds its own
 * per-thread compute context (mirrors rampart-onnx's embed handle).  So a model
 * object survives thread copies -- unlike the old single-thread-bound design.
 *
 * Three rules (as elsewhere in langtools): failures throw a JS Error; warnings
 * go to `errMsg`; nothing is written to stdout/stderr. */

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include "rampart.h"
#include "clip_shim.h"

/* get_current_thread() is provided by the rampart binary; keep the reference WEAK
 * so a non-rampart host that dlopens this .so still links (there errMsg is a no-op). */
#pragma weak get_current_thread

/* ---- this.errMsg: warnings + non-fatal errors (mirrors rampart-sql / -onnx) ----
 * Set on the object the call was made on (the model handle); cleared at the top of
 * each call; undefined when clean.  Failures RP_THROW; nothing hits stdout/stderr.
 * (Today the CPU path warns nowhere; the property exists so the future GPU->CPU
 * fallback -- like llamacpp/onnx -- has a home that getLog-style noise can't bury.) */
static duk_context *clip_thr_ctx(void) {
    RPTHR *t = get_current_thread ? get_current_thread() : NULL;
    return t ? t->ctx : NULL;
}
#define CLIP_MODULE_STASH DUK_HIDDEN_SYMBOL("clip_module")
static int clip_push_errmsg_target(duk_context *ctx) {
    duk_push_this(ctx);
    if (duk_is_object(ctx, -1)) return 1;
    duk_pop(ctx);
    duk_push_global_stash(ctx);
    if (!duk_get_prop_string(ctx, -1, CLIP_MODULE_STASH) || !duk_is_object(ctx, -1)) {
        duk_pop_2(ctx); return 0;
    }
    duk_remove(ctx, -2);
    return 1;
}
static void clip_errmsg_append(duk_context *ctx, const char *msg) {
    if (!clip_push_errmsg_target(ctx)) return;
    if (duk_get_prop_string(ctx, -1, "errMsg")) {
        const char *s = duk_get_string(ctx, -1);
        if (s && *s) duk_push_sprintf(ctx, "%s\n%s", s, msg);
        else         duk_push_string(ctx, msg);
        duk_remove(ctx, -2);
    } else { duk_pop(ctx); duk_push_string(ctx, msg); }
    duk_put_prop_string(ctx, -2, "errMsg");
    duk_pop(ctx);
}
static void clip_errmsg_clear(duk_context *ctx) {
    if (!clip_push_errmsg_target(ctx)) return;
    duk_del_prop_string(ctx, -1, "errMsg");
    duk_pop(ctx);
}
/* A warning from anywhere in the module -- including the shim (extern-declared in
 * clip_shim.h, mirroring llamacpp's lt_warn).  RAMPART_CLIP_DEBUG=1 also echoes to
 * stderr; that opt-in hatch is the only thing that may ever write there. */
void clip_warn(const char *fmt, ...) {
    duk_context *ctx = clip_thr_ctx();
    char line[1024]; size_t l; va_list ap;
    va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
    if (getenv("RAMPART_CLIP_DEBUG")) fputs(line, stderr);
    if (!ctx) return;
    l = strlen(line);
    while (l && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = '\0';
    if (l) clip_errmsg_append(ctx, line);
}

#define CLIP_PTR DUK_HIDDEN_SYMBOL("clip_handle")
#define CLIP_DIM DUK_HIDDEN_SYMBOL("clip_dim")

/* fetch the handle on `this`; throws if destroy()'d */
static clip_handle *get_handle(duk_context *ctx) {
    duk_push_this(ctx);
    if (!duk_get_prop_string(ctx, -1, CLIP_PTR))
        RP_THROW(ctx, "rampart-clip: not a model object");
    clip_handle *h = (clip_handle *) duk_get_pointer(ctx, -1);
    duk_pop_2(ctx);
    if (!h) RP_THROW(ctx, "rampart-clip: model has been destroyed");
    return h;
}
static int get_dim(duk_context *ctx) {
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, CLIP_DIM);
    int d = duk_get_int(ctx, -1);
    duk_pop_2(ctx);
    return d;
}

/* pack a float[dim] embedding as fp16 buffer / fp32 buffer / Number array */
enum { PACK16, PACK32, NOPACK };
static void push_vec(duk_context *ctx, const float *vec, int dim, int pack) {
    if (pack == PACK16) {
        void *b = duk_push_fixed_buffer(ctx, (duk_size_t) dim * sizeof(uint16_t));
        rpvec_f32_to_f16((float *) vec, (uint16_t *) b, dim);
    } else if (pack == PACK32) {
        void *b = duk_push_fixed_buffer(ctx, (duk_size_t) dim * sizeof(float));
        memcpy(b, vec, (size_t) dim * sizeof(float));
    } else {
        duk_push_array(ctx);
        for (int i = 0; i < dim; i++) { duk_push_number(ctx, (double) vec[i]); duk_put_prop_index(ctx, -2, (duk_uarridx_t) i); }
    }
}

/* embedImage*() accepts EITHER a String (image file path) OR a Buffer (the image
 * bytes -- a JPEG/PNG/... blob, e.g. from readFile() or a SQL varbyte column).
 * Both mean "an image", so this is unambiguous; a String is never sniffed as
 * bytes and a Buffer is never treated as a path.  duk_get_buffer_data accepts
 * every buffer flavor (plain buffer, Uint8Array, ArrayBuffer, Node Buffer). */
static duk_ret_t embed_image_to_(duk_context *ctx, int pack) {
    clip_errmsg_clear(ctx);
    clip_handle *h = get_handle(ctx);
    int dim = get_dim(ctx);
    float *vec = (float *) duk_alloc(ctx, (duk_size_t) dim * sizeof(float));
    if (!vec) RP_THROW(ctx, "rampart-clip: out of memory");
    char err[256] = {0};
    int rc;
    duk_size_t blen = 0;
    void *buf = duk_is_string(ctx, 0) ? NULL : duk_get_buffer_data(ctx, 0, &blen);
    if (buf) {
        rc = clip_embed_image_mem(h, buf, (size_t) blen, vec, 1, err, sizeof err);
    } else {
        const char *path = REQUIRE_STRING(ctx, 0,
            "rampart-clip: embedImage*() argument must be a String (image file path) or a Buffer (image bytes)");
        rc = clip_embed_image_file(h, path, vec, 1, err, sizeof err);
    }
    if (rc != 0) {
        duk_free(ctx, vec);
        RP_THROW(ctx, "rampart-clip: %s", err[0] ? err : "image embed failed");
    }
    push_vec(ctx, vec, dim, pack);
    duk_free(ctx, vec);
    return 1;
}
static duk_ret_t embed_image_to_buf16(duk_context *ctx)   { return embed_image_to_(ctx, PACK16); }
static duk_ret_t embed_image_to_buf32(duk_context *ctx)   { return embed_image_to_(ctx, PACK32); }
static duk_ret_t embed_image_to_numbers(duk_context *ctx) { return embed_image_to_(ctx, NOPACK); }

static duk_ret_t embed_text_to_(duk_context *ctx, int pack) {
    clip_errmsg_clear(ctx);
    clip_handle *h = get_handle(ctx);
    if (duk_is_buffer_data(ctx, 0)) duk_buffer_to_string(ctx, 0);
    const char *text = REQUIRE_STRING(ctx, 0, "rampart-clip: embedText*() argument must be a String");
    int dim = get_dim(ctx);
    float *vec = (float *) duk_alloc(ctx, (duk_size_t) dim * sizeof(float));
    if (!vec) RP_THROW(ctx, "rampart-clip: out of memory");
    char err[256] = {0};
    if (clip_embed_text(h, text, vec, 1, err, sizeof err) != 0) {
        duk_free(ctx, vec);
        RP_THROW(ctx, "rampart-clip: %s", err[0] ? err : "text embed failed");
    }
    push_vec(ctx, vec, dim, pack);
    duk_free(ctx, vec);
    return 1;
}
static duk_ret_t embed_text_to_buf16(duk_context *ctx)   { return embed_text_to_(ctx, PACK16); }
static duk_ret_t embed_text_to_buf32(duk_context *ctx)   { return embed_text_to_(ctx, PACK32); }
static duk_ret_t embed_text_to_numbers(duk_context *ctx) { return embed_text_to_(ctx, NOPACK); }

/* cosine similarity of two fp16 OR fp32 buffers (both same type/size) */
static duk_ret_t clip_similarity_method(duk_context *ctx) {
    clip_errmsg_clear(ctx);
    int dim = get_dim(ctx);
    duk_size_t s1 = 0, s2 = 0;
    void *b1 = REQUIRE_BUFFER_DATA(ctx, 0, &s1, "rampart-clip: similarity() arguments must be Buffers");
    void *b2 = REQUIRE_BUFFER_DATA(ctx, 1, &s2, "rampart-clip: similarity() arguments must be Buffers");
    if (s1 != s2) RP_THROW(ctx, "rampart-clip: similarity() buffers must be the same size");
    float *v1 = NULL, *v2 = NULL; int need_free = 0; float score;
    if ((int)(s1 / sizeof(float)) == dim) {
        v1 = (float *) b1; v2 = (float *) b2;
    } else if ((int)(s1 / sizeof(uint16_t)) == dim) {
        v1 = (float *) duk_alloc(ctx, (duk_size_t) dim * sizeof(float));
        v2 = (float *) duk_alloc(ctx, (duk_size_t) dim * sizeof(float));
        if (!v1 || !v2) { duk_free(ctx, v1); duk_free(ctx, v2); RP_THROW(ctx, "rampart-clip: out of memory in similarity()"); }
        rpvec_f16_to_f32((uint16_t *) b1, v1, dim);
        rpvec_f16_to_f32((uint16_t *) b2, v2, dim);
        need_free = 1;
    } else {
        RP_THROW(ctx, "rampart-clip: similarity() buffer size does not match model dimension (%d)", dim);
    }
    score = clip_similarity(v1, v2, dim);
    if (need_free) { duk_free(ctx, v1); duk_free(ctx, v2); }
    duk_push_number(ctx, (double) score);
    return 1;
}

/* destroy(): release this copy's refcount and mark it unusable.  Weights stay in
 * the process-global cache (a re-load of the same path is instant), matching how
 * rampart-onnx/llamacpp keep heavy embedding models resident. */
static duk_ret_t clip_destroy(duk_context *ctx) {
    duk_push_this(ctx);
    if (duk_get_prop_string(ctx, -1, CLIP_PTR)) {
        clip_handle *h = (clip_handle *) duk_get_pointer(ctx, -1);
        duk_pop(ctx);
        if (h) {
            clip_release(h);
            duk_push_pointer(ctx, NULL);
            duk_put_prop_string(ctx, -2, CLIP_PTR);
        }
    } else duk_pop(ctx);
    duk_pop(ctx);
    return 0;
}

/* GC finalizer: release unless destroy() already did (CLIP_PTR nulled). */
static duk_ret_t clip_finalizer(duk_context *ctx) {
    if (duk_get_prop_string(ctx, 0, CLIP_PTR)) {
        clip_handle *h = (clip_handle *) duk_get_pointer(ctx, -1);
        if (h) clip_release(h);
    }
    duk_pop(ctx);
    return 0;
}

/* thread-copy hook: a copy of the model object shares the SAME native handle, so
 * take a refcount for it (the harness re-attaches the finalizer that releases it). */
static duk_ret_t clip_copy_callback(duk_context *ctx) {
    if (duk_get_prop_string(ctx, 0, CLIP_PTR)) {
        clip_handle *h = (clip_handle *) duk_get_pointer(ctx, -1);
        if (h) clip_acquire(h);
    }
    duk_pop(ctx);
    return 0;
}

/* getLog()/clearLog(): the ggml informational firehose (CUDA-init banner, etc.),
 * captured off stderr by the shim.  Separate from errMsg (warnings) -- same split
 * as rampart-llamacpp / rampart-onnx. */
static duk_ret_t clip_get_log(duk_context *ctx) {
    char *s = clip_log_dup();
    duk_push_string(ctx, s ? s : "");
    free(s);
    return 1;
}
static duk_ret_t clip_clear_log(duk_context *ctx) {
    clip_log_clear();
    return 0;
}

static duk_ret_t clip_load_method(duk_context *ctx) {
    const char *path = REQUIRE_STRING(ctx, 0, "rampart-clip: load() argument 1 must be a String (model path)");
    /* opts (nThreads/verbosity) accepted for source-compat; CPU backend uses ggml's
     * default thread count.  Validated but currently advisory. */
    if (duk_is_object(ctx, 1)) {
        if (duk_get_prop_string(ctx, 1, "nThreads") && !duk_is_number(ctx, -1))
            RP_THROW(ctx, "rampart-clip: option nThreads must be a Number");
        duk_pop(ctx);
    }
    struct stat st;
    if (stat(path, &st) != 0)
        RP_THROW(ctx, "rampart-clip: cannot open model file '%s': %s", path, strerror(errno));

    char err[256] = {0};
    clip_handle *h = clip_load(path, err, sizeof err);
    if (!h) RP_THROW(ctx, "rampart-clip: %s", err[0] ? err : "failed to load model");
    int dim = clip_dim(h);
    if (dim <= 0) { clip_release(h); RP_THROW(ctx, "rampart-clip: could not determine embedding dimension"); }

    duk_push_object(ctx);
    duk_push_pointer(ctx, h);        duk_put_prop_string(ctx, -2, CLIP_PTR);
    duk_push_int(ctx, dim);          duk_put_prop_string(ctx, -2, CLIP_DIM);
    duk_push_int(ctx, dim);          duk_rp_put_prop_string_ro(ctx, -2, "dimension");

    duk_push_c_function(ctx, embed_image_to_buf16, 1);   duk_put_prop_string(ctx, -2, "embedImageToFp16Buf");
    duk_push_c_function(ctx, embed_image_to_buf32, 1);   duk_put_prop_string(ctx, -2, "embedImageToFp32Buf");
    duk_push_c_function(ctx, embed_image_to_numbers, 1); duk_put_prop_string(ctx, -2, "embedImageToNumbers");
    duk_push_c_function(ctx, embed_text_to_buf16, 1);    duk_put_prop_string(ctx, -2, "embedTextToFp16Buf");
    duk_push_c_function(ctx, embed_text_to_buf32, 1);    duk_put_prop_string(ctx, -2, "embedTextToFp32Buf");
    duk_push_c_function(ctx, embed_text_to_numbers, 1);  duk_put_prop_string(ctx, -2, "embedTextToNumbers");
    duk_push_c_function(ctx, clip_similarity_method, 2); duk_put_prop_string(ctx, -2, "similarity");
    duk_push_c_function(ctx, clip_destroy, 0);           duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, clip_copy_callback, 1);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("objOnCopyCallback"));
    duk_push_c_function(ctx, clip_finalizer, 1);
    duk_set_finalizer(ctx, -2);
    return 1;
}

/* ============================================================================
 * rp_clip_* : the C ABI rampart-sql binds via dlsym for `sql.set({clipEmbed:..})`,
 * exactly as it binds rp_embed_* (llamacpp) and rp_onnx_embed_* (onnx).  Same
 * calling convention as those: each embed function ALLOCATES *out_vec (a malloc'd
 * float[dim] the caller frees) and RETURNS the dimension as size_t (0 on failure),
 * so rampart-sql's dispatch is uniform across all three engines.  Vectors are
 * always L2-normalized (likev is cosine on unit-length vectors).  rampart-sql
 * requires the COMPLETE set -- a missing symbol is treated as a version mismatch;
 * rp_clip_iface_v1 is the explicit version marker (bump it if a signature changes
 * so a stale module fails at sql.set rather than mid-query).
 * ==========================================================================*/
int rp_clip_iface_v1(void) { return 1; }

void *rp_clip_embed_load(const char *path, char *err, size_t errlen) {
    return clip_load(path, err, errlen);        /* refcounted, process-lifetime (like onnx) */
}
void rp_clip_embed_release(void *h) { if (h) clip_release((clip_handle *)h); }
int  rp_clip_embed_dim(void *h)  { return h ? clip_dim((clip_handle *)h) : 0; }
int  rp_clip_has_text(void *h)   { return h ? clip_has_text((clip_handle *)h) : 0; }
int  rp_clip_has_vision(void *h) { return h ? clip_has_vision((clip_handle *)h) : 0; }

/* text query (length-delimited: SQL varchar bytes are NOT NUL-terminated) */
size_t rp_clip_embed_text(void *handle, const char *text, size_t tlen, float **out_vec) {
    if (out_vec) *out_vec = NULL;
    if (!handle || !out_vec) return 0;
    clip_handle *h = (clip_handle *) handle;
    int dim = clip_dim(h);
    if (dim <= 0) return 0;
    char *tmp = (char *) malloc(tlen + 1);      /* bounded NUL-terminated copy for the tokenizer */
    if (!tmp) return 0;
    if (tlen) memcpy(tmp, text, tlen);
    tmp[tlen] = '\0';
    float *vec = (float *) malloc((size_t) dim * sizeof(float));
    if (!vec) { free(tmp); return 0; }
    char err[256] = {0};
    int rc = clip_embed_text(h, tmp, vec, 1, err, sizeof err);
    free(tmp);
    if (rc != 0) { free(vec); if (err[0]) clip_warn("rp_clip_embed_text: %s\n", err); return 0; }
    *out_vec = vec;
    return (size_t) dim;
}

/* image from in-memory bytes (a varbyte column blob: JPEG/PNG/...) */
size_t rp_clip_embed_image(void *handle, const void *buf, size_t len, float **out_vec) {
    if (out_vec) *out_vec = NULL;
    if (!handle || !out_vec) return 0;
    clip_handle *h = (clip_handle *) handle;
    int dim = clip_dim(h);
    if (dim <= 0) return 0;
    float *vec = (float *) malloc((size_t) dim * sizeof(float));
    if (!vec) return 0;
    char err[256] = {0};
    int rc = clip_embed_image_mem(h, buf, len, vec, 1, err, sizeof err);
    if (rc != 0) { free(vec); if (err[0]) clip_warn("rp_clip_embed_image: %s\n", err); return 0; }
    *out_vec = vec;
    return (size_t) dim;
}

/* image from a file path (for images that live on disk beside the database) */
size_t rp_clip_embed_image_path(void *handle, const char *path, float **out_vec) {
    if (out_vec) *out_vec = NULL;
    if (!handle || !out_vec) return 0;
    clip_handle *h = (clip_handle *) handle;
    int dim = clip_dim(h);
    if (dim <= 0) return 0;
    float *vec = (float *) malloc((size_t) dim * sizeof(float));
    if (!vec) return 0;
    char err[256] = {0};
    int rc = clip_embed_image_file(h, path, vec, 1, err, sizeof err);
    if (rc != 0) { free(vec); if (err[0]) clip_warn("rp_clip_embed_image_path: %s\n", err); return 0; }
    *out_vec = vec;
    return (size_t) dim;
}

duk_ret_t duk_open_module(duk_context *ctx) {
    duk_push_object(ctx);
    /* initEmbed is the canonical name (langtools convention: initEmbed / initGen /
     * initRerank / ...); a CLIP model is an image+text embedder.  `load` is kept as
     * an alias for source-compat with the standalone rampart-clip project -- both
     * names are the same C function. */
    duk_push_c_function(ctx, clip_load_method, DUK_VARARGS);
    duk_dup_top(ctx);
    duk_put_prop_string(ctx, -3, "initEmbed");
    duk_put_prop_string(ctx, -2, "load");

    duk_push_c_function(ctx, clip_get_log, 0);
    duk_put_prop_string(ctx, -2, "getLog");
    duk_push_c_function(ctx, clip_clear_log, 0);
    duk_put_prop_string(ctx, -2, "clearLog");

    /* stash the module object as the errMsg fallback target (see clip_push_errmsg_target) */
    duk_push_global_stash(ctx);
    duk_dup(ctx, -2);
    duk_put_prop_string(ctx, -2, CLIP_MODULE_STASH);
    duk_pop(ctx);
    return 1;
}
