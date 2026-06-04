#ifndef LANGTOOLS_MAIN_INCLUDE

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <llama.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include "llama.h"
#include "llama_gen_shim.h"   /* C ABI for the multi-session generation engine */
#include "rampart.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#endif

// --- CUDA availability check ---
#if ( defined(LT_ENABLE_GPU) && !defined(__APPLE__) )
    #define HAVE_CUDA 1

    #include <cuda_runtime.h>
    #include "ggml-backend.h"

    static int has_gpu_backend()
    {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);

            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU)
            {
                return 1;
            }
        }
        return 0;
    }

#else

    #undef HAVE_CUDA

#endif


typedef struct rp_llama_info
{
    RPTHR *thr;
    struct llama_context *lctx;
    struct llama_model *lmodel;
    const struct llama_vocab *vocab;
    uint32_t n_ctx;
    int32_t n_vocab;
    uint32_t ga_n;
    uint32_t ga_w;
    uint32_t cur_pos;
    uint8_t store_last;
    duk_idx_t func_idx;
    char *out;
    size_t out_len;
    size_t out_cap;
    uint8_t stop;
    llama_seq_id seq_id;
    llama_memory_t mem;
    int max_tokens;
    struct llama_sampler *smpl;
    int n_generated;
    int n_keep;
    struct llama_context_params cp;
    int init_thr;
    int init_pid;

    // for async:
    duk_context *ctx;
    void *func_ptr;
    const char *errmsg;

    // multi-session generation engine (new path; replaces lctx-based gen)
    lgen_engine  *eng;        // per-thread slot engine (context pinned to this thread)
    char         *last_out;   // last generation text (for getLast)
    size_t        last_out_len;
    uint8_t       armed;      // 1 while the predictAsync step-pump timeout is scheduled
    uint8_t       destroyed;  // set by destroy when a pump is still armed (deferred free)
} rp_llama_info;

/* ==================================================================
 * New multi-session generation engine (P1): C glue over llama_gen_shim.
 * The slot scheduler lives in extern/llamacpp/wrapper/llama_gen_shim.cc.
 * ================================================================== */

// Build a malloc'd const char** of stop strings from options.stop (JS array).
static const char **lg_build_stops(duk_context *ctx, duk_idx_t obj_idx, size_t *n)
{
    *n = 0;
    if (!duk_get_prop_string(ctx, obj_idx, "stop")) { duk_pop(ctx); return NULL; }
    if (!duk_is_array(ctx, -1)) { duk_pop(ctx); return NULL; }
    duk_size_t len = duk_get_length(ctx, -1);
    if (!len) { duk_pop(ctx); return NULL; }
    const char **stops = NULL;
    REMALLOC(stops, sizeof(char *) * len);
    size_t k = 0;
    for (duk_size_t i = 0; i < len; i++) {
        duk_get_prop_index(ctx, -1, (duk_uarridx_t)i);
        if (duk_is_string(ctx, -1)) stops[k++] = strdup(duk_get_string(ctx, -1));
        duk_pop(ctx);
    }
    duk_pop(ctx);
    *n = k;
    return stops;
}
static void lg_free_stops(const char **stops, size_t n)
{
    if (!stops) return;
    for (size_t i = 0; i < n; i++) free((void *)stops[i]);
    free((void *)stops);
}

// Fill an lgen_request from the options object at obj_idx. The prompt /
// messages_json strings are left on the duk stack (valid until the caller
// restores the stack); they are only needed during lgen_session_submit().
static void lg_build_request(duk_context *ctx, duk_idx_t obj_idx, lgen_request *req,
                             const char ***stops_out, size_t *n_stops)
{
    memset(req, 0, sizeof(*req));
    req->max_tokens = 512;
    req->temp = -1; req->top_p = -1; req->min_p = -1; req->typ_p = -1; req->top_k = 0;
    req->penalty_repeat = -1; req->penalty_last_n = -1;
    req->dry_multiplier = -1; req->dry_base = -1; req->dry_allowed_length = -1;
    req->seed = (uint32_t)time(NULL);
    req->add_assistant = 1;

    if (duk_get_prop_string(ctx, obj_idx, "prompt")) {
        req->prompt = REQUIRE_STRING(ctx, -1, "prompt must be a string"); // leave on stack
    } else {
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, obj_idx, "messages")) {
            if (!duk_is_array(ctx, -1)) RP_THROW(ctx, "messages must be an Array");
            duk_dup(ctx, -1);
            req->messages_json = duk_json_encode(ctx, -1); // leave encoded string on stack
        } else {
            duk_pop(ctx);
            RP_THROW(ctx, "predict requires 'prompt' or 'messages'");
        }
    }

    if (duk_get_prop_string(ctx, obj_idx, "maxTokens")) req->max_tokens = (int)REQUIRE_UINT(ctx, -1, "maxTokens must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "temp")) req->temp = (float)REQUIRE_NUMBER(ctx, -1, "temp must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "topP")) req->top_p = (float)REQUIRE_NUMBER(ctx, -1, "topP must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "topK")) req->top_k = (int)REQUIRE_UINT(ctx, -1, "topK must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "minP")) req->min_p = (float)REQUIRE_NUMBER(ctx, -1, "minP must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "repeatPenalty")) req->penalty_repeat = (float)REQUIRE_NUMBER(ctx, -1, "repeatPenalty must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "repeatLastN")) req->penalty_last_n = (int)REQUIRE_UINT(ctx, -1, "repeatLastN must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "seed")) req->seed = (uint32_t)REQUIRE_UINT(ctx, -1, "seed must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "addAssistant")) req->add_assistant = REQUIRE_BOOL(ctx, -1, "addAssistant must be a Boolean");
    duk_pop(ctx);

    *stops_out = lg_build_stops(ctx, obj_idx, n_stops);
    req->stop = *stops_out;
    req->n_stop = *n_stops;
}

/* libevent cross-thread delivery. Declared locally (minimal prototypes) rather
 * than including <event2/event.h>, whose event-config.h lives in rampart's build
 * tree; the symbols resolve from the host rampart binary (RTLD_GLOBAL). struct
 * event / struct event_base are already forward-declared via rampart.h. */
#include <sys/time.h>
typedef int rp_evutil_socket_t;
extern struct event *event_new(struct event_base *base, rp_evutil_socket_t fd, short events,
                               void (*cb)(rp_evutil_socket_t, short, void *), void *arg);
extern int  event_add(struct event *ev, const struct timeval *timeout);
extern void event_free(struct event *ev);
#define evutil_socket_t rp_evutil_socket_t

// Build a fresh per-thread engine + info from creation params. The model is
// shared via the shim's refcounted cache, so this only allocates a new context
// + slots on the current thread (cheap next to loading weights).
static rp_llama_info *lg_new_info(duk_context *ctx, const lgen_engine_params *p)
{
    char err[256];
    lgen_engine *eng = lgen_engine_create(p, err, sizeof err);
    if (!eng) RP_THROW(ctx, "initGen: %s", err);
    rp_llama_info *info = NULL;
    CALLOC(info, sizeof(rp_llama_info));
    info->thr = get_current_thread();
    info->eng = eng;
    info->init_thr = get_thread_num();
    info->init_pid = (int)getpid();
    return info;
}

// Resolve the engine for the gen handle `this` refers to. Each thread-copy of
// the handle keeps its OWN engine, built lazily on first use from the stored
// path+params (the model is shared via the cache) — the same "rebuild per
// thread" trick the embedding path uses, applied to the whole engine. It is
// additive: a thread never touches another thread's engine, so there is no
// shared mutable state and no cross-thread guard needed. Ownership is tracked
// in per-copy hidden props (lg_thr/lg_pid), never by reading another thread's
// struct.
static rp_llama_info *lg_get_info(duk_context *ctx)
{
    duk_push_this(ctx); // [this]

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("destroyed")) &&
        duk_get_boolean_default(ctx, -1, 0))
        RP_THROW(ctx, "generation object was destroyed");
    duk_pop(ctx);

    int cur_thr = get_thread_num();
    int cur_pid = (int)getpid();

    int owner_thr = -1, owner_pid = -1;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_thr"))) owner_thr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_pid"))) owner_pid = duk_get_int(ctx, -1);
    duk_pop(ctx);

    rp_llama_info *info = NULL;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rp_llama_info")))
        info = (rp_llama_info *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    // already have an engine built for THIS thread+process? use it.
    if (info && owner_thr == cur_thr && owner_pid == cur_pid) {
        duk_pop(ctx); // this
        return info;
    }

#ifdef HAVE_CUDA
    if (owner_pid != -1 && owner_pid != cur_pid && has_gpu_backend())
        RP_THROW(ctx, "llama.cpp - cannot fork llama.cpp with CUDA initialized");
#endif

    // used on a new thread (or after fork): build a new per-thread engine from
    // the stored creation params and stash it in THIS copy's own hidden slots.
    lgen_engine_params p;
    if (!duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_params")))
        RP_THROW(ctx, "generation engine not initialized");
    memcpy(&p, duk_get_buffer_data(ctx, -1, NULL), sizeof p);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_model_path")); // [this, path]
    p.model_path  = duk_get_string(ctx, -1);
    /* re-point chat_template too (the params buffer's copy is stale on this thread) */
    if (duk_get_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_chat_template")))   // [this, path, tmpl]
        p.chat_template = duk_get_string(ctx, -1);
    else
        p.chat_template = NULL;
    rp_llama_info *ninfo = lg_new_info(ctx, &p); // copies path+template; may throw
    duk_pop_2(ctx); // tmpl, path -> [this]

    duk_push_pointer(ctx, ninfo); duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rp_llama_info"));
    duk_push_int(ctx, cur_thr);   duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_thr"));
    duk_push_int(ctx, cur_pid);   duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_pid"));
    duk_pop(ctx); // this
    return ninfo;
}

/* ---- step pump: one recurring 0-delay timeout per engine drives all of this
 *      thread's slots. The decode runs HERE, on the owning rampart thread (like
 *      embedding), and on_piece/on_done fire here too — duktape-safe, no
 *      cross-thread marshaling. Armed lazily on the first predictAsync. ---- */

static void lg_info_free(rp_llama_info *info)
{
    if (info->eng) { lgen_engine_free(info->eng); info->eng = NULL; }
    if (info->last_out) free(info->last_out);
    free(info);
}

static int lg_pump(void *arg, int stage)
{
    rp_llama_info *info = (rp_llama_info *)arg;
    if (stage) { // stage 1 ("after"): return value becomes the new repeat flag
        if (info->eng && lgen_engine_has_active(info->eng)) return 1; // re-arm
        info->armed = 0;
        if (info->destroyed) lg_info_free(info); // deferred free (destroyed mid-stream)
        return 0; // done: one-shot, event is freed
    }
    // stage 0: advance every active slot one batched decode (fires callbacks here).
    // Return 1 (not 0) so the loop proceeds to stage 1, which decides re-arming;
    // returning 0 here would skip stage 1 and make this a single-shot timeout.
    if (info->eng) (void)lgen_engine_step(info->eng);
    return 1;
}

static void lg_arm_pump(duk_context *ctx, rp_llama_info *info)
{
    if (info->armed) return;
    info->armed = 1;
    (void)duk_rp_insert_timeout(ctx, 0, "predictAsync", lg_pump, (void *)info,
                                DUK_INVALID_INDEX, DUK_INVALID_INDEX, 0.0);
}

/* ---- SYNC predict: drive the engine on THIS thread until the request ends ---- */

typedef struct {
    int    done;
    int    status;
    char  *err;
    char  *full;
    size_t full_len;
} lg_sync_ud;

static void lg_sync_on_done(void *ud, int status, const char *err, int reason,
                            const char *full, size_t full_len)
{
    (void)reason;
    lg_sync_ud *s = (lg_sync_ud *)ud;
    s->status = status;
    if (status != 0 && err) s->err = strdup(err);
    if (full && full_len) {
        s->full = malloc(full_len + 1);
        memcpy(s->full, full, full_len);
        s->full[full_len] = '\0';
        s->full_len = full_len;
    }
    s->done = 1;
}

// gen.predict(opts) -> blocks this thread, returns the full string. On a server
// this blocks the worker thread's event loop; use predictAsync in hot handlers.
static duk_ret_t lg_predict(duk_context *ctx)
{
    rp_llama_info *info = lg_get_info(ctx);
    REQUIRE_OBJECT(ctx, 0, "first argument must be an options Object");

    duk_idx_t base = duk_get_top(ctx);
    lgen_request req;
    const char **stops = NULL; size_t n_stops = 0;
    lg_build_request(ctx, 0, &req, &stops, &n_stops);

    lg_sync_ud s;
    memset(&s, 0, sizeof s);

    char err[256];
    uint64_t rid = lgen_engine_submit(info->eng, &req, NULL, lg_sync_on_done, &s, err, sizeof err);
    lg_free_stops(stops, n_stops);
    duk_set_top(ctx, base);
    if (!rid) RP_THROW(ctx, "predict failed: %s", err);

    // Drive the engine to completion on this thread. on_done sets s.done; the
    // active guard prevents a hang if the engine goes idle unexpectedly.
    while (!s.done) {
        int active = lgen_engine_step(info->eng);
        if (!active && !s.done) break;
    }

    if (s.status != 0) {
        char *e = s.err ? s.err : strdup("generation error");
        if (s.full) free(s.full);
        duk_push_error_object(ctx, DUK_ERR_ERROR, "%s", e);
        free(e);
        (void)duk_throw(ctx);
    }

    if (info->last_out) free(info->last_out);
    info->last_out = s.full; info->last_out_len = s.full_len;
    duk_push_lstring(ctx, info->last_out ? info->last_out : "", info->last_out_len);
    return 1;
}

/* ---- ASYNC predictAsync: submit, then let the per-engine pump stream tokens.
 *      Callbacks fire on this thread inside lg_pump's step — no cross-thread
 *      delivery, no refcount. The per-request state lives until on_done. ---- */

typedef struct {
    duk_context   *ctx;       // owning JS thread's context (callbacks + stash live here)
    rp_llama_info *info;
    lgen_engine   *eng;
    uint64_t       req_id;
    int            canceled;
} lg_areq;

static int lg_areq_get_cb(duk_context *ctx, lg_areq *r, const char *prefix)
{
    duk_push_global_stash(ctx);
    duk_push_sprintf(ctx, "%s%p", prefix, (void *)r);
    if (!duk_get_prop(ctx, -2)) { duk_pop_2(ctx); return 0; }
    duk_remove(ctx, -2);
    return 1;
}
static void lg_areq_clear_stash(lg_areq *r)
{
    duk_context *ctx = r->ctx;
    duk_push_global_stash(ctx);
    duk_push_sprintf(ctx, "lgtok_%p", (void *)r); duk_del_prop(ctx, -2);
    duk_push_sprintf(ctx, "lgfin_%p", (void *)r); duk_del_prop(ctx, -2);
    duk_pop(ctx);
}

// on_piece — fired on THIS thread inside lg_pump->lgen_engine_step. Returning a
// truthy value from the perToken callback cancels the request.
static void lg_infer_on_piece(void *ud, const char *piece, size_t len)
{
    lg_areq *r = (lg_areq *)ud;
    duk_context *ctx = r->ctx;
    if (r->canceled) return;
    if (lg_areq_get_cb(ctx, r, "lgtok_")) {
        duk_push_object(ctx);
        duk_push_lstring(ctx, piece ? piece : "", (duk_size_t)len);
        duk_put_prop_string(ctx, -2, "token");
        duk_push_false(ctx); duk_put_prop_string(ctx, -2, "done");
        if (duk_pcall(ctx, 1) == 0) {
            if (duk_get_boolean_default(ctx, -1, 0) && !r->canceled) {
                r->canceled = 1;
                lgen_engine_cancel(r->eng, r->req_id);
            }
        }
        duk_pop(ctx);
    }
}
static void lg_infer_on_done(void *ud, int status, const char *err, int reason,
                             const char *full, size_t full_len)
{
    (void)reason;
    lg_areq *r = (lg_areq *)ud;
    duk_context *ctx = r->ctx;

    if (lg_areq_get_cb(ctx, r, "lgtok_")) {
        duk_push_object(ctx);
        duk_push_true(ctx); duk_put_prop_string(ctx, -2, "done");
        if (status != 0 && err) { duk_push_string(ctx, err); duk_put_prop_string(ctx, -2, "error"); }
        if (duk_pcall(ctx, 1) != 0) { /* swallow */ }
        duk_pop(ctx);
    }
    if (lg_areq_get_cb(ctx, r, "lgfin_")) {
        duk_push_object(ctx);
        if (full) { duk_push_lstring(ctx, full, (duk_size_t)full_len); duk_put_prop_string(ctx, -2, "fullText"); }
        if (status != 0 && err) { duk_push_string(ctx, err); duk_put_prop_string(ctx, -2, "error"); }
        if (duk_pcall(ctx, 1) != 0) { /* swallow */ }
        duk_pop(ctx);
    }
    if (full && full_len && r->info) {
        if (r->info->last_out) free(r->info->last_out);
        r->info->last_out = malloc(full_len + 1);
        memcpy(r->info->last_out, full, full_len);
        r->info->last_out[full_len] = '\0';
        r->info->last_out_len = full_len;
    }

    lg_areq_clear_stash(r);
    free(r);
}

static duk_ret_t lg_predict_async(duk_context *ctx)
{
    rp_llama_info *info = lg_get_info(ctx);
    REQUIRE_OBJECT(ctx, 0, "first argument must be an options Object");
    int has_tok = duk_is_function(ctx, 1);
    int has_fin = duk_is_function(ctx, 2);
    if (!has_tok && !has_fin)
        RP_THROW(ctx, "predictAsync requires a perToken and/or final callback Function");

    duk_idx_t base = duk_get_top(ctx);
    lgen_request req;
    const char **stops = NULL; size_t n_stops = 0;
    lg_build_request(ctx, 0, &req, &stops, &n_stops); // may throw before we allocate

    lg_areq *r = NULL;
    CALLOC(r, sizeof *r);
    r->ctx = ctx; r->info = info; r->eng = info->eng;

    duk_push_global_stash(ctx);
    if (has_tok) { duk_push_sprintf(ctx, "lgtok_%p", (void *)r); duk_dup(ctx, 1); duk_put_prop(ctx, -3); }
    if (has_fin) { duk_push_sprintf(ctx, "lgfin_%p", (void *)r); duk_dup(ctx, 2); duk_put_prop(ctx, -3); }
    duk_pop(ctx);

    char err[256];
    uint64_t rid = lgen_engine_submit(info->eng, &req, lg_infer_on_piece, lg_infer_on_done, r, err, sizeof err);
    lg_free_stops(stops, n_stops);
    duk_set_top(ctx, base);

    if (!rid) {
        lg_areq_clear_stash(r);
        free(r);
        RP_THROW(ctx, "predictAsync submit failed: %s", err);
    }
    r->req_id = rid;
    lg_arm_pump(ctx, info);
    return 0;
}

static duk_ret_t lg_get_last(duk_context *ctx)
{
    // best-effort: return this thread's own engine's last output (each thread has
    // its own engine, so getLast is inherently per-thread). Empty if this copy
    // hasn't generated on this thread yet.
    duk_push_this(ctx);
    int owner_thr = -1, owner_pid = -1;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_thr"))) owner_thr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_pid"))) owner_pid = duk_get_int(ctx, -1);
    duk_pop(ctx);
    rp_llama_info *info = NULL;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rp_llama_info")))
        info = (rp_llama_info *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (info && owner_thr == get_thread_num() && owner_pid == (int)getpid())
        duk_push_lstring(ctx, info->last_out ? info->last_out : "", info->last_out_len);
    else
        duk_push_string(ctx, "");
    return 1;
}

static duk_ret_t lg_destroy(duk_context *ctx)
{
    duk_push_this(ctx);
    if (!duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rp_llama_info"))) return 0;
    rp_llama_info *info = (rp_llama_info *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    int owner_thr = -1, owner_pid = -1;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_thr"))) owner_thr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("lg_pid"))) owner_pid = duk_get_int(ctx, -1);
    duk_pop(ctx);

    // Only tear down the engine if THIS copy built it on THIS thread+process.
    // A copy whose info still points at another thread's engine (never used
    // here) must not free it — that thread's own copy owns and frees it.
    if (info && owner_thr == get_thread_num() && owner_pid == (int)getpid()) {
    if (info->armed) {
        // A step-pump timeout is mid-flight — and we may be INSIDE it right now
        // (destroy() called from a predictAsync callback runs nested under
        // lg_pump -> lgen_engine_step). Freeing the engine here would delete it
        // under the running step (use-after-free). Defer the WHOLE teardown
        // (engine + info) to the pump's stage 1, which runs after the step and
        // after the engine drains to idle. See lg_pump / lg_info_free.
        info->destroyed = 1;
    } else {
        // Not armed: no pump running, safe to tear down now. lgen_engine_free
        // fails any queued requests (firing their on_done) before freeing.
        lg_info_free(info);
    }
    }
    duk_push_pointer(ctx, NULL);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rp_llama_info"));
    duk_push_true(ctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("destroyed"));
    return 0;
}

/* ====================================================================
 * Shared option parsing: maps llama-server's CLI options 1:1 onto the
 * camelCase keys of init{Gen,Embed,Rerank}({opt:val}). This fills the
 * COMMON model-loading + context options into the real llama structs;
 * each init function then layers on its mode-specific options.
 *   --gpu-layers      -> gpuLayers          --rope-freq-base  -> ropeFreqBase
 *   --split-mode      -> splitMode          --rope-freq-scale -> ropeFreqScale
 *   --main-gpu        -> mainGpu            --rope-scaling    -> ropeScaling
 *   --no-mmap         -> useMmap (bool)     --yarn-*          -> yarn*
 *   --mlock           -> useMlock           --flash-attn      -> flashAttn
 *   --check-tensors   -> checkTensors       --cache-type-k/v  -> cacheTypeK/V
 *   --ctx-size        -> nCtx               --no-kv-offload   -> offloadKqv
 *   --batch-size      -> nBatch             --op-offload      -> opOffload
 *   --ubatch-size     -> nUBatch            --kv-unified      -> kvUnified
 *   --parallel        -> nSeqMax            --threads(-batch) -> threads/threadsBatch
 * Pointer-valued opts (tensorSplit, overrideKv, devices) and process-global
 * ones (numa) are intentionally not handled here. Mode opts (pooling/attention/
 * embdNormalize, jinja/chatTemplate) are handled by the individual init funcs.
 * ==================================================================== */
static enum ggml_type lt_kv_type_from_str(duk_context *ctx, const char *s) {
    if (!strcmp(s, "f32") || !strcmp(s, "fp32")) return GGML_TYPE_F32;
    if (!strcmp(s, "f16") || !strcmp(s, "fp16")) return GGML_TYPE_F16;
    if (!strcmp(s, "bf16"))                      return GGML_TYPE_BF16;
    if (!strcmp(s, "q8_0"))                      return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_0"))                      return GGML_TYPE_Q4_0;
    if (!strcmp(s, "q4_1"))                      return GGML_TYPE_Q4_1;
    if (!strcmp(s, "q5_0"))                      return GGML_TYPE_Q5_0;
    if (!strcmp(s, "q5_1"))                      return GGML_TYPE_Q5_1;
    if (!strcmp(s, "iq4_nl"))                    return GGML_TYPE_IQ4_NL;
    RP_THROW(ctx, "cacheTypeK/cacheTypeV: unknown type '%s' "
                  "(f32,f16,bf16,q8_0,q4_0,q4_1,q5_0,q5_1,iq4_nl)", s);
    return GGML_TYPE_F16; /* unreachable */
}

/* read an optional bool prop; supports an alias key. returns 1 if present. */
static int lt_opt_bool2(duk_context *ctx, duk_idx_t o, const char *k, const char *alias,
                        const char *errmsg, int *out) {
    int have = duk_get_prop_string(ctx, o, k);
    if (!have && alias) { duk_pop(ctx); have = duk_get_prop_string(ctx, o, alias); }
    if (have) *out = REQUIRE_BOOL(ctx, -1, errmsg);
    duk_pop(ctx);
    return have;
}

static void parse_common_opts(duk_context *ctx, duk_idx_t o,
                              struct llama_model_params *mp,
                              struct llama_context_params *cp)
{
    if (o < 0) return;
    int b;

    /* ---- model loading (llama_model_params) ---- */
    if (duk_get_prop_string(ctx, o, "gpuLayers")) mp->n_gpu_layers = REQUIRE_INT(ctx, -1, "gpuLayers must be an integer (-1 = all)");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "mainGpu"))   mp->main_gpu     = REQUIRE_INT(ctx, -1, "mainGpu must be an integer");
    duk_pop(ctx);
    if (lt_opt_bool2(ctx, o, "useMmap",      NULL, "useMmap must be boolean",      &b)) mp->use_mmap      = b;
    if (lt_opt_bool2(ctx, o, "useMlock",     NULL, "useMlock must be boolean",     &b)) mp->use_mlock     = b;
    if (lt_opt_bool2(ctx, o, "checkTensors", NULL, "checkTensors must be boolean", &b)) mp->check_tensors = b;
    if (duk_get_prop_string(ctx, o, "splitMode")) {
        const char *s = REQUIRE_STRING(ctx, -1, "splitMode must be a string (none|layer|row)");
        if      (!strcmp(s, "none"))  mp->split_mode = LLAMA_SPLIT_MODE_NONE;
        else if (!strcmp(s, "layer")) mp->split_mode = LLAMA_SPLIT_MODE_LAYER;
        else if (!strcmp(s, "row"))   mp->split_mode = LLAMA_SPLIT_MODE_ROW;
        else RP_THROW(ctx, "splitMode: unknown '%s' (none|layer|row)", s);
    }
    duk_pop(ctx);

    /* ---- context (llama_context_params) ---- */
    if (duk_get_prop_string(ctx, o, "nCtx")) {
        int n = REQUIRE_INT(ctx, -1, "nCtx must be an integer (0 or -1 = model's max context)");
        cp->n_ctx = (n <= 0) ? 0 : (uint32_t)n;   /* 0 resolved to n_ctx_train at create */
    }
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "nBatch"))       cp->n_batch         = (uint32_t)REQUIRE_UINT(ctx, -1, "nBatch must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "nUBatch"))      cp->n_ubatch        = (uint32_t)REQUIRE_UINT(ctx, -1, "nUBatch must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "nSeqMax"))      cp->n_seq_max       = (uint32_t)REQUIRE_UINT(ctx, -1, "nSeqMax must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "threads"))      cp->n_threads       = REQUIRE_INT(ctx, -1, "threads must be an integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "threadsBatch")) cp->n_threads_batch = REQUIRE_INT(ctx, -1, "threadsBatch must be an integer");
    duk_pop(ctx);
    if (lt_opt_bool2(ctx, o, "kvUnified", NULL, "kvUnified must be boolean", &b)) cp->kv_unified = b;
    if (lt_opt_bool2(ctx, o, "offloadKqv", "offloadKQV", "offloadKqv must be boolean", &b)) cp->offload_kqv = b;
    if (lt_opt_bool2(ctx, o, "opOffload", NULL, "opOffload must be boolean", &b)) cp->op_offload = b;
    if (duk_get_prop_string(ctx, o, "ropeFreqBase"))   cp->rope_freq_base   = (float)REQUIRE_NUMBER(ctx, -1, "ropeFreqBase must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "ropeFreqScale"))  cp->rope_freq_scale  = (float)REQUIRE_NUMBER(ctx, -1, "ropeFreqScale must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "yarnExtFactor"))  cp->yarn_ext_factor  = (float)REQUIRE_NUMBER(ctx, -1, "yarnExtFactor must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "yarnAttnFactor")) cp->yarn_attn_factor = (float)REQUIRE_NUMBER(ctx, -1, "yarnAttnFactor must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "yarnBetaFast"))   cp->yarn_beta_fast   = (float)REQUIRE_NUMBER(ctx, -1, "yarnBetaFast must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "yarnBetaSlow"))   cp->yarn_beta_slow   = (float)REQUIRE_NUMBER(ctx, -1, "yarnBetaSlow must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "yarnOrigCtx"))    cp->yarn_orig_ctx    = (uint32_t)REQUIRE_UINT(ctx, -1, "yarnOrigCtx must be a positive integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "ropeScaling")) {
        const char *s = REQUIRE_STRING(ctx, -1, "ropeScaling must be a string (none|linear|yarn|longrope)");
        if      (!strcmp(s, "none"))     cp->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        else if (!strcmp(s, "linear"))   cp->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        else if (!strcmp(s, "yarn"))     cp->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        else if (!strcmp(s, "longrope")) cp->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
        else RP_THROW(ctx, "ropeScaling: unknown '%s' (none|linear|yarn|longrope)", s);
    }
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "cacheTypeK")) cp->type_k = lt_kv_type_from_str(ctx, REQUIRE_STRING(ctx, -1, "cacheTypeK must be a type string"));
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "cacheTypeV")) cp->type_v = lt_kv_type_from_str(ctx, REQUIRE_STRING(ctx, -1, "cacheTypeV must be a type string"));
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "flashAttn")) {
        if (duk_is_boolean(ctx, -1))
            cp->flash_attn_type = duk_get_boolean(ctx, -1) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        else {
            const char *s = REQUIRE_STRING(ctx, -1, "flashAttn must be boolean or 'on'|'off'|'auto'");
            if      (!strcmp(s, "on")  || !strcmp(s, "enabled"))  cp->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (!strcmp(s, "off") || !strcmp(s, "disabled")) cp->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (!strcmp(s, "auto"))                          cp->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else RP_THROW(ctx, "flashAttn: unknown '%s' (on|off|auto)", s);
        }
    }
    duk_pop(ctx);
}

/* read a whole file into a malloc'd NUL-terminated string (for chatTemplateFile) */
static char *lt_read_file_alloc(duk_context *ctx, const char *fn) {
    FILE *f = fopen(fn, "rb");
    if (!f) RP_THROW(ctx, "chatTemplateFile: cannot open '%s'", fn);
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); RP_THROW(ctx, "chatTemplateFile: cannot size '%s'", fn); }
    char *buf = malloc((size_t)n + 1);
    size_t rd = buf ? fread(buf, 1, (size_t)n, f) : 0;
    fclose(f);
    if (!buf) RP_THROW(ctx, "chatTemplateFile: out of memory");
    buf[rd] = '\0';
    return buf;
}

static duk_ret_t lg_init_gen(duk_context *ctx)
{
    const char *model_path = REQUIRE_STRING(ctx, 0, "initGen: first argument must be a String (path to .gguf)");
    duk_idx_t o = duk_is_object(ctx, 1) ? 1 : -1;

    /* defaults: one slot, one CPU thread (the GPU does the math), n_ctx 0 => the
       model's trained max (resolved at context build). flashAttn defaults to auto
       from llama_context_default_params(). */
    struct llama_model_params   mp = llama_model_default_params();
    struct llama_context_params cp = llama_context_default_params();
    cp.n_seq_max       = 1;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.n_ctx           = 0;

    char *chat_template = NULL;  /* malloc'd; freed after lg_new_info copies it */
    int   use_jinja     = 1;

    if (o > -1) {
        parse_common_opts(ctx, o, &mp, &cp);   /* shared model + context options */

        /* gen-specific options */
        lt_opt_bool2(ctx, o, "jinja", NULL, "jinja must be boolean", &use_jinja);
        if (duk_get_prop_string(ctx, o, "chatTemplate"))
            chat_template = strdup(REQUIRE_STRING(ctx, -1, "chatTemplate must be a string"));
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, o, "chatTemplateFile")) {
            const char *fn = REQUIRE_STRING(ctx, -1, "chatTemplateFile must be a string");
            free(chat_template);
            chat_template = lt_read_file_alloc(ctx, fn);
        }
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, o, "mmproj")) { duk_pop(ctx); free(chat_template); RP_THROW(ctx, "initGen: vision (mmproj) generation is not supported"); }
        duk_pop(ctx);
    }

    lgen_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path    = model_path;
    p.chat_template = chat_template;   /* shim copies into the engine */
    p.use_jinja     = use_jinja;
    p.mparams       = mp;
    p.cparams       = cp;

    rp_llama_info *info = lg_new_info(ctx, &p); // builds engine (shared model) + info
    lgen_engine *eng = info->eng;

    duk_push_object(ctx);
    duk_push_int(ctx, (int)lgen_engine_n_ctx(eng));
    duk_rp_put_prop_string_ro(ctx, -2, "nCtx");
    duk_push_int(ctx, (int)lgen_engine_n_vocab(eng));
    duk_rp_put_prop_string_ro(ctx, -2, "nVocab");

    duk_push_pointer(ctx, info);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rp_llama_info"));

    // Per-copy state so each thread that receives a copy of this handle can build
    // its OWN engine (sharing the cached model) on first use — see lg_get_info.
    duk_push_int(ctx, get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_thr"));
    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_pid"));
    duk_push_string(ctx, model_path);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_model_path"));
    /* chat_template (a malloc'd string) is owned by the rebuilt engine; the params
       buffer below keeps a copy of the now-stale pointer, but lg_get_info re-points
       it from this symbol on each thread before rebuilding (like lg_model_path). */
    if (chat_template) {
        duk_push_string(ctx, chat_template);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_chat_template"));
    }
    void *pbuf = duk_push_fixed_buffer(ctx, sizeof p);
    memcpy(pbuf, &p, sizeof p);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("lg_params"));
    free(chat_template);   /* the shim engine already owns its own copy */

    duk_push_c_function(ctx, lg_destroy, 0);
    duk_put_prop_string(ctx, -2, "destroy");
    duk_push_c_function(ctx, lg_destroy, 1);
    duk_set_finalizer(ctx, -2);
    duk_push_c_function(ctx, lg_get_last, 0);
    duk_put_prop_string(ctx, -2, "getLast");
    duk_push_c_function(ctx, lg_predict, 2);
    duk_put_prop_string(ctx, -2, "predict");
    duk_push_c_function(ctx, lg_predict_async, 3);
    duk_put_prop_string(ctx, -2, "predictAsync");

    return 1;
}

/* ------------------------------------------------------------------------------
 * initGen (transparent cross-thread batching).
 *
 * The exposed initGen() runs this embedded JS coordinator: it spawns ONE dedicated
 * owner rampart.thread holding a single shared slot-engine (built via the module's
 * __rawInitGen, which rides across to the owner thread as a C-function on the
 * module object), and returns a wrapper that LOOKS like a normal gen object. Each
 * predict() triggers the request to the owner (rampart.event) and PARKS on
 * thread.get for the result — so during the owner's GPU decode every caller is
 * parked (no concurrent event loop = the Metal/CUDA-safe condition), and N normal
 * threads' requests transparently batch through the one shared context.
 *
 * State that must survive the gen object being deep-copied to other threads is
 * kept as object PROPERTIES (read via `this`), not closures.
 * v1: predict() is fully batched; predictAsync() emulates via a blocking predict
 * then fires the callbacks (true token streaming is a separate, gated step). ---- */
static const char *BATCHGEN_SCRIPT =
"(function(mod, model, opts, uid){\n"
"  var thread = rampart.thread;\n"
"  var owner = new thread();\n"
"  owner.exec(function(a){\n"
"    var raw = a.rawInit(a.model, a.opts);\n"
"    var canc = {};\n"   /* request ids signalled to cancel (set by the cancel event) */
"    rampart.event.on('can_'+a.uid, 'c', function(uv, c){ canc[c.id] = 1; });\n"
"    rampart.event.on('sub_'+a.uid, 'h', function(uv, r){\n"
"      if (r.stream) {\n"
"        raw.predictAsync(r.req,\n"
"          function(t){ if (canc[r.id]) { delete canc[r.id]; return true; }\n"   /* hard cancel -> frees the slot */
"                       if (!t.done && !t.error && t.token) rampart.event.trigger('tok_'+a.uid+'_'+r.id, { tok: t.token }); },\n"
"          function(res){ delete canc[r.id]; rampart.event.trigger('fin_'+a.uid+'_'+r.id, { full: res.fullText || '', err: res.error }); });\n"
"      } else {\n"
"        raw.predictAsync(r.req, function(){}, function(res){\n"
"          rampart.thread.put('res_'+a.uid+'_'+r.id,\n"
"            res.error ? ('[gen err:'+res.error+']') : (res.fullText || ''));\n"
"        });\n"
"      }\n"
"    });\n"
"    rampart.thread.put('ready_'+a.uid, { nCtx: raw.nCtx, nVocab: raw.nVocab });\n"
"  }, { rawInit: mod.__rawInitGen, model: model, opts: opts || {}, uid: uid });\n"
"  var meta = thread.get('ready_'+uid, 120000) || {};\n"
"  return {\n"
"    __uid: uid, nCtx: meta.nCtx, nVocab: meta.nVocab, _last: '', _ctr: 0,\n"
"    predict: function(o){\n"
"      var id = rampart.thread.getCurrentId() + '_' + (this._ctr = (this._ctr||0)+1);\n"
"      rampart.event.trigger('sub_'+this.__uid, { id: id, req: o });\n"
"      var text = rampart.thread.get('res_'+this.__uid+'_'+id, 120000);\n"
"      this._last = (text === undefined) ? '' : text;\n"
"      return this._last;\n"
"    },\n"
"    predictAsync: function(o, perTok, fin){\n"
"      var uid = this.__uid;\n"
"      var id = rampart.thread.getCurrentId() + '_' + (this._ctr = (this._ctr||0)+1);\n"
"      var tn = 'tok_'+uid+'_'+id, fn = 'fin_'+uid+'_'+id, full = '';\n"
"      rampart.event.on(tn, 'h', function(uv, t){ full += t.tok;\n"
"        if (typeof perTok === 'function') perTok({ token: t.tok, done: false }); });\n"
"      rampart.event.on(fn, 'h', function(uv, f){\n"
"        rampart.event.remove(tn); rampart.event.remove(fn);\n"
"        if (typeof fin === 'function') fin({ fullText: (f.full !== undefined ? f.full : full), error: f.err }); });\n"
"      rampart.event.trigger('sub_'+uid, { id: id, req: o, stream: true });\n"
"      return { cancel: function(){ rampart.event.trigger('can_'+uid, { id: id }); } };\n"   /* hard cancel handle */
"    },\n"
"    getLast: function(){ return this._last; },\n"
"    destroy: function(){ try { owner.terminate(); } catch(e) {} }\n"
"  };\n"
"})\n";

static duk_ret_t lg_init_gen_batched(duk_context *ctx)
{
    REQUIRE_STRING(ctx, 0, "initGen: first argument must be a String (path to .gguf)");
    /* opts (optional object) at index 1 */

    static int bg_ctr = 0;
    int n = __atomic_add_fetch(&bg_ctr, 1, __ATOMIC_SEQ_CST);
    char uid[64];
    snprintf(uid, sizeof uid, "bg%d_%d", (int)getpid(), n);

    duk_eval_string(ctx, BATCHGEN_SCRIPT);   /* -> wrapper function on the stack */
    duk_push_this(ctx);                       /* mod (this module object, has __rawInitGen) */
    duk_dup(ctx, 0);                          /* model */
    duk_dup(ctx, 1);                          /* opts (may be undefined) */
    duk_push_string(ctx, uid);
    duk_call(ctx, 4);                         /* wrapper(mod, model, opts, uid) -> gen object */
    return 1;
}

// LLAMA.CPP EMBEDDING MODELS

// Tear down an embed/rerank handle. A handle may be COPIED across threads (each
// copy rebuilds its own per-thread llama_ctx from the shared model). So free each
// resource by its correct owner, or copies will double-free:
//   - llama_ctx + a model refcount: per CONTEXT — only the copy that built it on
//     THIS thread (ctx_thread/ctx_pid match) frees its context and releases one
//     model refcount (the model is freed by the cache when the last is released).
//   - rerank_toks: per HANDLE (allocated once at init) — only the ORIGIN copy
//     (emb_origin_thr/pid, never changed by a rebuild) frees it.
static duk_ret_t emb_free(duk_context *ctx)
{
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

    duk_push_this(ctx);

    // guard against a second teardown of the same copy (explicit destroy + finalizer)
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("destroyed")) && duk_get_boolean_default(ctx, -1, 0)) {
        duk_pop_2(ctx);
        return 0;
    }
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("model"));
    lmodel = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("llama_ctx"));
    lctx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    int ctx_thr = -1, ctx_pid = -1, org_thr = -1, org_pid = -1;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_thread")))
        ctx_thr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_pid")))
            ctx_pid = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("emb_origin_thr")))
        org_thr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("emb_origin_pid")))
        org_pid = duk_get_int(ctx, -1);
    duk_pop(ctx);

    int cur_thr = get_thread_num();
    int cur_pid = (int)getpid();
    int own_context = (lctx && ctx_thr == cur_thr && ctx_pid == cur_pid);
    int is_origin   = (org_thr == cur_thr && org_pid == cur_pid);

    // per-handle resource: only the origin copy frees it
    if (is_origin)
    {
        if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rerank_toks")))
        {
            void *toks = duk_get_pointer(ctx, -1);
            if (toks) free(toks);
        }
        duk_pop(ctx);
    }

    // per-context resources: only the copy that built this context frees them
    if (own_context)
    {
        llama_free(lctx);
        lgen_model_release(lmodel); // refcount--, model freed by cache at zero
        duk_del_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("model"));
        duk_del_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("llama_ctx"));
    }

    duk_push_true(ctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("destroyed"));

    return 0;
}

/* Mode-specific options for embed/rerank, parsed into cp AFTER parse_common_opts.
 * Includes the legacy names (nctx/ubatch/nthreads/nthreads_batch) kept as aliases
 * so existing initEmbed/initRerank callers keep working, plus pooling/attention. */
static void parse_embed_opts(duk_context *ctx, duk_idx_t o, struct llama_context_params *cp)
{
    if (o < 0) return;
    int b;

    /* legacy aliases (camelCase equivalents are handled by parse_common_opts) */
    if (duk_get_prop_string(ctx, o, "nctx"))           cp->n_ctx           = (uint32_t)REQUIRE_INT(ctx, -1, "nctx must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "ubatch"))         cp->n_ubatch        = (uint32_t)REQUIRE_INT(ctx, -1, "ubatch must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "nthreads"))       cp->n_threads       = REQUIRE_INT(ctx, -1, "nthreads must be a number");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "nthreads_batch")) cp->n_threads_batch = REQUIRE_INT(ctx, -1, "nthreads_batch must be a number");
    duk_pop(ctx);

    /* mode options */
    if (lt_opt_bool2(ctx, o, "embeddings", NULL, "embeddings must be boolean", &b)) cp->embeddings = b;
    if (duk_get_prop_string(ctx, o, "pooling")) {
        if (duk_is_number(ctx, -1)) {
            cp->pooling_type = (enum llama_pooling_type) duk_get_int(ctx, -1);
        } else {
            const char *s = REQUIRE_STRING(ctx, -1, "pooling must be 'none'|'mean'|'cls'|'last'|'rank'");
            if      (!strcmp(s, "none")) cp->pooling_type = LLAMA_POOLING_TYPE_NONE;
            else if (!strcmp(s, "mean")) cp->pooling_type = LLAMA_POOLING_TYPE_MEAN;
            else if (!strcmp(s, "cls"))  cp->pooling_type = LLAMA_POOLING_TYPE_CLS;
            else if (!strcmp(s, "last")) cp->pooling_type = LLAMA_POOLING_TYPE_LAST;
            else if (!strcmp(s, "rank")) cp->pooling_type = LLAMA_POOLING_TYPE_RANK;
            else RP_THROW(ctx, "pooling: unknown '%s' (none|mean|cls|last|rank)", s);
        }
    }
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, o, "attention")) {
        const char *s = REQUIRE_STRING(ctx, -1, "attention must be 'causal'|'non-causal'");
        if      (!strcmp(s, "causal"))                              cp->attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        else if (!strcmp(s, "non-causal") || !strcmp(s, "noncausal")) cp->attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        else RP_THROW(ctx, "attention: unknown '%s' (causal|non-causal)", s);
    }
    duk_pop(ctx);
}

/* Build the embedding context from an ALREADY-PARSED cp (common + embed opts).
 * Finalizes embed-specific derived values (n_ctx cap, ubatch, batch, unified KV)
 * and stashes cp_buf for the per-thread rebuild, then creates the context. */
static struct llama_context *new_embed_context(duk_context *ctx, struct llama_model *lmodel, struct llama_context_params *cpin)
{
    struct llama_context_params cp = *cpin;

    // If user didn't specify a context size, default to the model's max (capped
    // so an auto setting doesn't blow up memory; an explicit nCtx is honored).
    if (cp.n_ctx <= 0)
    {
        int n_train = llama_model_n_ctx_train(lmodel);
        if (n_train > 8192) n_train = 8192;
        if (n_train > 0) cp.n_ctx = n_train;
    }

    if (cp.n_ubatch <= 0)
        cp.n_ubatch = cp.n_ctx > 0 ? cp.n_ctx : 0;  // full window in one micro-batch

    cp.n_batch = cp.n_ubatch;  // batch >= ubatch, to prevent clamping

    // b9494: match llama.cpp's embedding tool (examples/embedding/embedding.cpp).
    // Enabling a unified KV cache makes the context actually allocate a memory;
    // without it llama_decode produces no pooled output (NULL/NaN) and LLM-arch
    // embedders (qwen2vl/qwen3) null-deref in the KV-cache code path.
    cp.n_seq_max  = (uint32_t)llama_max_parallel_sequences();
    cp.kv_unified = true;

    void *cp_buf = duk_push_fixed_buffer(ctx, sizeof(struct llama_context_params));
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("cp_buf"));
    memcpy(cp_buf, &cp, sizeof(struct llama_context_params));

    return llama_init_from_model(lmodel, cp);
}

#define NOPACK 0
#define PACK16 1
#define PACK32 2

/* pack != 0 - return fp16 */
static duk_ret_t embed_text_to_(duk_context *ctx, int pack)
{
    if (duk_is_buffer_data(ctx, 0))
        duk_buffer_to_string(ctx, 0);

    const char *text = REQUIRE_STRING(ctx, 0, "rampart-llama-cpp:embedTextToBuf - argument must be a String");

    int vec_dim = 0;
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("model"));
    lmodel = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("llama_ctx"));
    lctx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("vec_dim"));
    vec_dim = duk_get_int(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_thread"));
    int thrno = duk_get_int(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_pid"));
    int pidno = duk_get_int(ctx, -1);
    duk_pop(ctx);

    int curthr = get_thread_num();
    int curpid = (int)getpid();

    // get a new context if in a new thread.  Model stays the same.
    if (curthr != thrno || pidno != curpid )
    {

#ifdef HAVE_CUDA
        // forking after is bad, mkay
        if(pidno != curpid && has_gpu_backend() )
        {
            RP_THROW(ctx, "llama.cpp - cannot fork llama.cpp with CUDA initialized");
        }
#endif
        duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("cp_buf"));
        struct llama_context_params *cp_buf = duk_get_buffer_data(ctx, -1, NULL);
        duk_pop(ctx);
        lctx = llama_init_from_model(lmodel, *cp_buf);

        //lctx = new_embed_context(ctx, lmodel, -1);

        // this copy now holds its own context using the shared model: take a model
        // refcount for it (released in emb_free when this context is freed).
        lgen_model_addref(lmodel);

        duk_push_pointer(ctx, lctx);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

        duk_push_int(ctx, curthr);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));
        duk_push_int(ctx, curpid);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_pid"));
    }

    if (!lctx)
    {
        RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - NULL llama_context");
        return 0;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);

    // ---- tokenize full input (probe length)
    int need = llama_tokenize(vocab, text, (int)strlen(text),
                              /*tokens*/ NULL, /*n_tokens_max*/ 0,
                              /*add_special*/ true, /*parse_special*/ true);
    if (need <= 0)
        need = -need; // some builds return negative "needed"

    if (need <= 0)
    {
        // return empty array for empty/whitespace-only input
        duk_push_array(ctx);
        return 1;
    }

    // materialize tokens
    llama_token *toks = NULL;
    CALLOC(toks, (size_t)need * sizeof *toks);

    int nw = llama_tokenize(vocab, text, (int)strlen(text), toks, need, /*add_special*/ true, /*parse_special*/ true);

    if (nw <= 0)
        nw = -nw;
    if (nw > need)
        nw = need;

    // runtime limits
    const int n_ctx = llama_n_ctx(lctx);
    int n_ubatch = llama_n_ubatch(lctx); // recent llama.cpp API
    if (n_ubatch <= 0)
        n_ubatch = n_ctx; // permissive fallback

    // chunking params
    int chunk_tokens = (n_ctx < n_ubatch ? n_ctx : n_ubatch);
    int overlap = chunk_tokens / 8;

    if (chunk_tokens <= 0)
    {
        free(toks);
        RP_THROW(ctx, "invalid runtime limits (ctx=%d, ubatch=%d)", n_ctx, n_ubatch);
        return 0;
    }

    if (overlap < 0)
        overlap = 0;
    if (overlap >= chunk_tokens)
        overlap = chunk_tokens - 1;
    int stride = chunk_tokens - overlap;

    // for avg vector
    float *avgvec = NULL;
    CALLOC(avgvec, sizeof(float) * vec_dim);

    // the return object
    duk_push_object(ctx);

    // result array (Array of ArrayBuffer)
    duk_idx_t arr_idx = duk_push_array(ctx);

    int k = 0;

    const enum llama_pooling_type p = llama_pooling_type(lctx);

    llama_set_embeddings(lctx, true); // b9494: ensure embeddings output mode is on

    for (int start = 0; start < nw; start += stride, ++k)
    {
        llama_memory_clear(llama_get_memory(lctx), /*clear_kv=*/true);
        int n = nw - start;
        if (n > chunk_tokens)
            n = chunk_tokens;

        // encoder path requires: n_tokens <= n_ubatch
        if (n > n_ubatch)
        {
            free(toks);
            RP_THROW(ctx, "chunk too large for micro-batch (n=%d > n_ubatch=%d). Increase ubatch and/or batch.", n, n_ubatch);
            return 0;
        }

        // build batch for this chunk
        struct llama_batch batch = llama_batch_init(/*capacity*/ n, /*embd*/ 0, /*n_seq_max*/ 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits)
        {
            llama_batch_free(batch);
            free(toks);
            RP_THROW(ctx, "llama_batch_init failed");
            return 0;
        }
        for (int i = 0; i < n; ++i)
        {
            batch.token[i] = toks[start + i];
            batch.pos[i] = i; // 0..n-1 within this window
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0; // single sequence id 0
            batch.logits[i] = 1;    // contribute to pooled embedding
        }
        batch.n_tokens = n;
        // NOTE (llama.cpp b9494): embedding/pooling now uses llama_decode with the
        // per-token output flags (batch.logits[i]=1) set above; the old workaround
        // of nulling batch.logits and calling llama_encode yields no pooled output.

        if (llama_decode(lctx, batch) != 0)
        {
            llama_batch_free(batch);
            free(toks);
            RP_THROW(ctx, "llama_decode failed on chunk %d (tokens %d..%d)", k, start, start + n - 1);
            return 0;
        }

        // read pooled embedding
        const float *emb =
            (p == LLAMA_POOLING_TYPE_NONE) ? llama_get_embeddings_ith(lctx, n - 1) : llama_get_embeddings_seq(lctx, 0);

        llama_batch_free(batch);

        if (!emb)
        {
            free(toks);
            RP_THROW(ctx, "no embedding returned (chunk %d)", k);
            return 0;
        }

        // L2-normalize and pack to fp16 (little-endian) or else make an array of Numbers
        double norm2 = 0.0;
        for (int i = 0; i < vec_dim; ++i)
            norm2 += (double)emb[i] * (double)emb[i];

        float inv = norm2 > 0.0 ? (float)(1.0 / (sqrt(norm2))) : 1;

        if (pack == PACK16)
        {
            uint16_t *out = (uint16_t *)duk_push_fixed_buffer(ctx, (duk_size_t)(2 * vec_dim));
            float v[vec_dim];
            for (int i = 0; i < vec_dim; ++i)
            {
                v[i] = emb[i] * inv;
                avgvec[i] += v[i];
            }
            rpvec_f32_to_f16(v, out, vec_dim);
        }
        else if (pack == PACK32)
        {
            float *out = (float *)duk_push_fixed_buffer(ctx, (duk_size_t)(4 * vec_dim));
            for (int i = 0; i < vec_dim; ++i)
            {
                out[i] = emb[i] * inv;
                avgvec[i] += out[i];
            }
        }
        else
        {
            duk_push_array(ctx);
            for (int i = 0; i < vec_dim; ++i)
            {
                double v = (double)emb[i] * (double)inv;
                duk_push_number(ctx, v);
                duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
                avgvec[i] += v;
            }
        }

        // arr[k] = buffer/array of Numbers(doubles)
        duk_put_prop_index(ctx, arr_idx, (duk_uarridx_t)k);
    }
    // we have only one, just copy it to avgvec
    if (k == 1)
    {
        free(toks);
        free(avgvec); // not needed - same as existing sole vec
        // [ ..., object, array ]
        duk_dup(ctx, -1);
        // [ ..., object, array, arraydup ]
        duk_put_prop_string(ctx, -3, "vecs");
        // [ ..., object, arraydup ]
        duk_get_prop_index(ctx, -1, 0);
        // [ ..., object, arraydup, vec ]
        duk_put_prop_string(ctx, -3, "avgVec");
        // [ ..., object, arraydup]
        duk_pop(ctx);
        // [ ..., object]
        return 1;
    }

    duk_put_prop_string(ctx, -2, "vecs");

    if (k > 1)
    {
        double norm2 = 0.0;

        for (int i = 0; i < vec_dim; ++i)
        {
            avgvec[i] /= (float)k;
            norm2 += (double)avgvec[i] * (double)avgvec[i];
        }

        float inv = norm2 > 0.0 ? (float)(1.0 / (sqrt(norm2))) : 1;

        if (pack == PACK16)
        {
            uint16_t *out = (uint16_t *)duk_push_fixed_buffer(ctx, (duk_size_t)(2 * vec_dim));

            for (int i = 0; i < vec_dim; ++i)
            {
                avgvec[i] *= inv;
            }
            rpvec_f32_to_f16(avgvec, out, vec_dim);
        }
        else if (pack == PACK32)
        {
            float *out = (float *)duk_push_fixed_buffer(ctx, (duk_size_t)(4 * vec_dim));
            for (int i = 0; i < vec_dim; ++i)
            {
                out[i] = avgvec[i] * inv;
            }
        }
        else
        {
            duk_push_array(ctx);
            for (int i = 0; i < vec_dim; ++i)
            {
                double v = (double)avgvec[i] * (double)inv;
                duk_push_number(ctx, v);
                duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
            }
        }
        duk_put_prop_string(ctx, -2, "avgVec");
    }

    free(toks);
    free(avgvec);

    return 1; // -> [ ArrayBuffer(fp16), ArrayBuffer(fp16), ... ]
}

static duk_ret_t embed_text_to_buf32(duk_context *ctx)
{
    return embed_text_to_(ctx, PACK32);
}

static duk_ret_t embed_text_to_buf16(duk_context *ctx)
{
    return embed_text_to_(ctx, PACK16);
}

static duk_ret_t embed_text_to_numbers(duk_context *ctx)
{
    return embed_text_to_(ctx, NOPACK);
}

/* ============================================================
 * C API for non-JS callers (rampart-sql.c et al.)
 *
 * Symbols (rp_embed_*) are externally visible.  Rampart's module
 * loader uses RTLD_GLOBAL|RTLD_NOW (see core/module.c:724), so
 * after `require("rampart-llamacpp")` succeeds in the JS layer,
 * dlsym(RTLD_DEFAULT, "rp_embed_load") finds these from anywhere.
 *
 * Lifecycle:
 *   rp_embed_load(path)        → opaque rp_embed_handle*, refcounted
 *   rp_embed_text(h, ...)      → avgVec (L2-normalized mean of
 *                                 L2-normalized per-chunk vecs)
 *   rp_embed_dim(h)            → model embedding dim
 *   rp_embed_release(h)        → decrement; free if last ref
 *
 * Per-handle pthread_mutex serializes llama_context use; the
 * model is shared read-only (mmap'd weights).  v1 uses mutex
 * Two modes (selected at runtime via rp_embed_set_per_thread()):
 *
 *   per_thread = 1 (default):
 *     One llama_model is shared across all calling threads, but each
 *     thread gets its own llama_context lazily on first use.  No mutex
 *     is held during llama_encode → multiple threads embed concurrently
 *     (real parallelism on GPU; multiple threads on CPU).
 *     Memory: per-thread KV/scratch (~10-50 MB CPU; tiny VRAM since
 *     weights are deduped).
 *
 *   per_thread = 0:
 *     One model AND one context shared by all callers, protected by a
 *     mutex around the entire embed call.  Serialized but minimal
 *     memory.  Useful in tightly memory-constrained environments.
 * ============================================================ */

/* Shared-model variant: one llama_model owned by the handle, each
 * thread gets its own llama_context (no per-thread model load). */
typedef struct {
    int                    thread_num;
    struct llama_context  *lctx;
} rp_thread_ctx_t;

typedef struct rp_embed_handle_s {
    char                       *path;       /* strdup'd absolute path */
    struct llama_model         *lmodel;
    struct llama_context       *lctx;       /* shared single ctx (per_thread=0 mode) */
    struct llama_context_params cp;
    int                         vec_dim;
    int                         refcount;
    pthread_mutex_t             mtx;        /* guards lctx + thread_ctxs[] */
    rp_thread_ctx_t            *thread_ctxs;
    int                         n_thread_ctxs;
    int                         cap_thread_ctxs;
    struct rp_embed_handle_s   *next;       /* cache linked list */
} rp_embed_handle_t;

/* Process-global mode flag.  Default = per-thread on. */
static int g_embed_per_thread = 1;

void rp_embed_set_per_thread(int on)
{
    g_embed_per_thread = on ? 1 : 0;
}

int rp_embed_get_per_thread(void)
{
    return g_embed_per_thread;
}

static rp_embed_handle_t *rp_embed_cache_head = NULL;
static pthread_mutex_t    rp_embed_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static rp_embed_handle_t *rp_embed_cache_get(const char *path)
{
    pthread_mutex_lock(&rp_embed_cache_lock);
    rp_embed_handle_t *h = NULL;
    for (rp_embed_handle_t *c = rp_embed_cache_head; c; c = c->next) {
        if (strcmp(c->path, path) == 0) { h = c; h->refcount++; break; }
    }
    pthread_mutex_unlock(&rp_embed_cache_lock);
    return h;
}

static void rp_embed_cache_put(rp_embed_handle_t *h)
{
    pthread_mutex_lock(&rp_embed_cache_lock);
    h->next = rp_embed_cache_head;
    rp_embed_cache_head = h;
    pthread_mutex_unlock(&rp_embed_cache_lock);
}

void *rp_embed_load(const char *path, char *err, size_t errlen)
{
    if (!path) {
        if (err && errlen) snprintf(err, errlen, "rp_embed_load: null path");
        return NULL;
    }

    rp_embed_handle_t *h = rp_embed_cache_get(path);
    if (h) return h;

    h = (rp_embed_handle_t *)calloc(1, sizeof(*h));
    if (!h) {
        if (err && errlen) snprintf(err, errlen, "rp_embed_load: oom");
        return NULL;
    }
    h->path = strdup(path);
    pthread_mutex_init(&h->mtx, NULL);

    struct llama_model_params mp = llama_model_default_params();
    h->lmodel = llama_model_load_from_file(path, mp);
    if (!h->lmodel) {
        if (err && errlen)
            snprintf(err, errlen, "rp_embed_load: could not load '%s': %s",
                     path, strerror(errno));
        free(h->path);
        pthread_mutex_destroy(&h->mtx);
        free(h);
        return NULL;
    }

    h->vec_dim = llama_model_n_embd(h->lmodel);
    if (h->vec_dim <= 0) {
        if (err && errlen)
            snprintf(err, errlen, "rp_embed_load: bad vec dim %d", h->vec_dim);
        llama_model_free(h->lmodel);
        free(h->path);
        pthread_mutex_destroy(&h->mtx);
        free(h);
        return NULL;
    }

    /* Build context params mirroring new_embed_context's defaults
     * (without duktape opts — we don't expose tuning here). */
    h->cp = llama_context_default_params();
    h->cp.embeddings     = true;
    h->cp.pooling_type   = LLAMA_POOLING_TYPE_MEAN;
    h->cp.n_threads      = 1;
    h->cp.n_threads_batch = -1;
    int n_train = llama_model_n_ctx_train(h->lmodel);
    if (n_train > 8192) n_train = 8192;
    h->cp.n_ctx     = n_train > 0 ? n_train : 0;
    h->cp.n_ubatch  = h->cp.n_ctx > 0 ? h->cp.n_ctx : 0;
    h->cp.n_batch   = h->cp.n_ubatch;
    h->cp.n_seq_max  = (uint32_t)llama_max_parallel_sequences(); // b9494: unified KV
    h->cp.kv_unified = true;

    h->lctx = llama_init_from_model(h->lmodel, h->cp);
    if (!h->lctx) {
        if (err && errlen)
            snprintf(err, errlen, "rp_embed_load: llama_init_from_model failed");
        llama_model_free(h->lmodel);
        free(h->path);
        pthread_mutex_destroy(&h->mtx);
        free(h);
        return NULL;
    }

    h->refcount = 1;
    rp_embed_cache_put(h);
    return h;
}

int rp_embed_dim(void *handle)
{
    if (!handle) return 0;
    return ((rp_embed_handle_t *)handle)->vec_dim;
}

/* avgVec compute path, mirrors lines 1985-2207 of embed_text_to_
 * minus the duktape array building / NOPACK / PACK16 paths.
 * Always returns a freshly malloc'd L2-normalized f32 vec of dim.
 * Takes lctx explicitly so per-thread variants can pass their own;
 * lmodel comes from h (shared). */
static size_t rp_embed_compute_avgvec(rp_embed_handle_t *h,
                                      struct llama_context *lctx,
                                      const char *text, size_t tlen,
                                      float **out_vec,
                                      char *err, size_t errlen)
{
    *out_vec = NULL;
    if (!h || !lctx || !text || tlen == 0) return 0;

    const struct llama_vocab *vocab = llama_model_get_vocab(h->lmodel);
    int need = llama_tokenize(vocab, text, (int)tlen,
                              NULL, 0, /*add_special*/true, /*parse_special*/true);
    if (need <= 0) need = -need;
    if (need <= 0) return 0;   /* empty/whitespace text */

    llama_token *toks = (llama_token *)calloc((size_t)need, sizeof(*toks));
    if (!toks) { if (err) snprintf(err, errlen, "oom tokens"); return 0; }

    int nw = llama_tokenize(vocab, text, (int)tlen, toks, need,
                            /*add_special*/true, /*parse_special*/true);
    if (nw <= 0) nw = -nw;
    if (nw > need) nw = need;

    const int n_ctx     = llama_n_ctx(lctx);
    int       n_ubatch  = llama_n_ubatch(lctx);
    if (n_ubatch <= 0) n_ubatch = n_ctx;

    int chunk_tokens = (n_ctx < n_ubatch ? n_ctx : n_ubatch);
    int overlap      = chunk_tokens / 8;
    if (chunk_tokens <= 0) {
        free(toks);
        if (err) snprintf(err, errlen, "invalid limits (n_ctx=%d ubatch=%d)",
                          n_ctx, n_ubatch);
        return 0;
    }
    if (overlap < 0) overlap = 0;
    if (overlap >= chunk_tokens) overlap = chunk_tokens - 1;
    int stride = chunk_tokens - overlap;

    float *avgvec = (float *)calloc((size_t)h->vec_dim, sizeof(float));
    if (!avgvec) { free(toks); if (err) snprintf(err, errlen, "oom avgvec"); return 0; }

    int k = 0;
    const enum llama_pooling_type p = llama_pooling_type(lctx);

    llama_set_embeddings(lctx, true); // b9494: ensure embeddings output mode is on

    for (int start = 0; start < nw; start += stride, ++k) {
        llama_memory_clear(llama_get_memory(lctx), /*clear_kv=*/true);
        int n = nw - start;
        if (n > chunk_tokens) n = chunk_tokens;
        if (n > n_ubatch) {
            free(toks); free(avgvec);
            if (err) snprintf(err, errlen, "chunk too large (n=%d > ubatch=%d)",
                              n, n_ubatch);
            return 0;
        }

        struct llama_batch batch =
            llama_batch_init(/*capacity*/n, /*embd*/0, /*n_seq_max*/1);
        if (!batch.token || !batch.pos || !batch.n_seq_id ||
            !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            free(toks); free(avgvec);
            if (err) snprintf(err, errlen, "llama_batch_init failed");
            return 0;
        }
        for (int i = 0; i < n; ++i) {
            batch.token[i]    = toks[start + i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = 1;
        }
        batch.n_tokens = n;

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            free(toks); free(avgvec);
            if (err) snprintf(err, errlen, "llama_decode failed on chunk %d", k);
            return 0;
        }

        const float *emb = (p == LLAMA_POOLING_TYPE_NONE)
                         ? llama_get_embeddings_ith(lctx, n - 1)
                         : llama_get_embeddings_seq(lctx, 0);
        llama_batch_free(batch);

        if (!emb) {
            free(toks); free(avgvec);
            if (err) snprintf(err, errlen, "no embedding on chunk %d", k);
            return 0;
        }

        /* L2-normalize this chunk and accumulate */
        double norm2 = 0.0;
        for (int i = 0; i < h->vec_dim; ++i)
            norm2 += (double)emb[i] * (double)emb[i];
        float inv = norm2 > 0.0 ? (float)(1.0 / sqrt(norm2)) : 1.0f;
        for (int i = 0; i < h->vec_dim; ++i)
            avgvec[i] += emb[i] * inv;
    }
    free(toks);

    if (k == 0) {
        free(avgvec);
        return 0;
    }

    /* Mean across chunks, then L2-normalize.  For k==1 this is a
     * no-op (avgvec is already a unit-norm vector) but the math
     * comes out identical so no special case needed. */
    double norm2 = 0.0;
    for (int i = 0; i < h->vec_dim; ++i) {
        avgvec[i] /= (float)k;
        norm2 += (double)avgvec[i] * (double)avgvec[i];
    }
    float inv = norm2 > 0.0 ? (float)(1.0 / sqrt(norm2)) : 1.0f;
    for (int i = 0; i < h->vec_dim; ++i) avgvec[i] *= inv;

    *out_vec = avgvec;
    return (size_t)h->vec_dim;
}

/* Find or create the calling thread's llama_context (shared lmodel
 * comes from h).  Returns NULL on failure.  Mutex briefly during
 * map lookup and lazy llama_init_from_model. */
static struct llama_context *
get_per_thread_ctx(rp_embed_handle_t *h)
{
    int thrno = (int)get_thread_num();
    struct llama_context *lctx = NULL;

    pthread_mutex_lock(&h->mtx);
    for (int i = 0; i < h->n_thread_ctxs; i++) {
        if (h->thread_ctxs[i].thread_num == thrno) {
            lctx = h->thread_ctxs[i].lctx;
            goto out;
        }
    }
    if (h->n_thread_ctxs >= h->cap_thread_ctxs) {
        int new_cap = h->cap_thread_ctxs ? h->cap_thread_ctxs * 2 : 4;
        rp_thread_ctx_t *g = (rp_thread_ctx_t *)realloc(h->thread_ctxs,
            (size_t)new_cap * sizeof *g);
        if (!g) goto out;
        h->thread_ctxs = g;
        h->cap_thread_ctxs = new_cap;
    }
    lctx = llama_init_from_model(h->lmodel, h->cp);
    if (lctx) {
        h->thread_ctxs[h->n_thread_ctxs].thread_num = thrno;
        h->thread_ctxs[h->n_thread_ctxs].lctx       = lctx;
        h->n_thread_ctxs++;
    }
out:
    pthread_mutex_unlock(&h->mtx);
    return lctx;
}

size_t rp_embed_text(void *handle, const char *text, size_t tlen, float **out_vec)
{
    if (!handle || !out_vec) return 0;
    rp_embed_handle_t *h = (rp_embed_handle_t *)handle;
    char err[256] = {0};
    size_t dim = 0;

    if (g_embed_per_thread) {
        struct llama_context *lctx = get_per_thread_ctx(h);
        if (!lctx) {
            fprintf(stderr, "rp_embed_text: failed to allocate per-thread context\n");
            return 0;
        }
        /* No mutex around the embed — each thread owns its lctx.
         * Shares h->lmodel with all other threads. */
        dim = rp_embed_compute_avgvec(h, lctx, text, tlen,
                                      out_vec, err, sizeof err);
    } else {
        /* Serialized path: one shared (model, ctx), full embed under mutex. */
        pthread_mutex_lock(&h->mtx);
        dim = rp_embed_compute_avgvec(h, h->lctx, text, tlen,
                                      out_vec, err, sizeof err);
        pthread_mutex_unlock(&h->mtx);
    }

    if (dim == 0 && err[0])
        fprintf(stderr, "rp_embed_text: %s\n", err);
    return dim;
}

void rp_embed_release(void *handle)
{
    if (!handle) return;
    rp_embed_handle_t *h = (rp_embed_handle_t *)handle;

    pthread_mutex_lock(&rp_embed_cache_lock);
    h->refcount--;
    /* v1: never actually free; embedding models are heavy and the
     * common pattern is process-lifetime ownership.  Unlink/free
     * path can be added when there's a real use case. */
    pthread_mutex_unlock(&rp_embed_cache_lock);
}

/* ============================================================ */

static duk_ret_t llamacpp_init_embed(duk_context *ctx)
{
    const char *model = REQUIRE_STRING(ctx, 0, "init: argument 1 must be a string");
    duk_idx_t obj_idx = -1;

    if (duk_is_object(ctx, 1))
        obj_idx = 1;

    duk_push_object(ctx); // return object

    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

    /* model + context options (llama-server parity), shared parser + embed mode */
    struct llama_model_params   mp = llama_model_default_params();
    struct llama_context_params cp = llama_context_default_params();
    cp.embeddings      = true;
    cp.pooling_type    = LLAMA_POOLING_TYPE_MEAN;
    cp.n_threads       = 1;
    cp.n_ctx           = 0;
    cp.n_ubatch        = 0;
    cp.n_threads_batch = -1;
    if (obj_idx > -1) { parse_common_opts(ctx, obj_idx, &mp, &cp); parse_embed_opts(ctx, obj_idx, &cp); }

    // Shared, refcounted load (one llama_model per path even across thread-copies;
    // one refcount per context, released in emb_free). Fixes the cross-copy
    // double-free and shares weights across threads.
    char lerr[256] = {0};
    lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:init - Could not load ggml file '%s': %s", model, lerr[0] ? lerr : strerror(errno));

    int vec_dim = llama_model_n_embd(lmodel);

    if (vec_dim <= 0)
        RP_THROW(ctx, "rampart-llama-cpp:init - Internal error getting vector dimensions");

    lctx = new_embed_context(ctx, lmodel, &cp);

    if (!lctx)
        RP_THROW(ctx, "rampart-llama-cpp:init - Failed to init llama from model");

    duk_push_pointer(ctx, lmodel);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("model"));

    duk_dup(ctx, 0);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("model_path"));

    duk_push_pointer(ctx, lctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

    duk_push_int(ctx, vec_dim);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("vec_dim"));

    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));

    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_pid"));

    // origin thread/pid: NEVER updated on rebuild — identifies the one copy that
    // owns per-handle resources (e.g. rerank_toks). See emb_free.
    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("emb_origin_thr"));
    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("emb_origin_pid"));

    duk_push_c_function(ctx, embed_text_to_buf32, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp32Buf");

    duk_push_c_function(ctx, embed_text_to_buf16, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp16Buf");

    duk_push_c_function(ctx, embed_text_to_numbers, 1);
    duk_put_prop_string(ctx, -2, "embedTextToNumbers");

    duk_push_c_function(ctx, emb_free, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, emb_free, 1);
    duk_set_finalizer(ctx, -2);

    return 1;
}


typedef struct rp_rerank_toks {
    const char *bos;
    const char *sep;
    const char *eos;
    size_t len;
} rp_rerank_toks;

static void get_rr_toks(const struct llama_vocab *vocab, rp_rerank_toks *toks)
{
    if(!toks)
        return;

    // Get SEP and EOS tokens from vocabulary
    llama_token bos_token = llama_vocab_bos(vocab);
    llama_token sep_token = llama_vocab_sep(vocab);
    llama_token eos_token = llama_vocab_eos(vocab);

    // Check if these tokens should be added
    bool add_bos = llama_vocab_get_add_bos(vocab);
    bool add_sep = llama_vocab_get_add_sep(vocab);
    bool add_eos = llama_vocab_get_add_eos(vocab);

    // If llama.cpp auto-adds a special token, DO NOT include its text in the prompt.
    toks->bos = (!add_bos && bos_token != LLAMA_TOKEN_NULL) ? llama_vocab_get_text(vocab, bos_token) : "";
    toks->sep = (!add_sep && sep_token != LLAMA_TOKEN_NULL) ? llama_vocab_get_text(vocab, sep_token) : "";
    toks->eos = (!add_eos && eos_token != LLAMA_TOKEN_NULL) ? llama_vocab_get_text(vocab, eos_token) : "";

    toks->len = strlen(toks->bos) +
                strlen(toks->sep) +
                (strlen(toks->eos)*2);
}

// Build rerank input string using vocabulary's SEP and EOS tokens
// Returns allocated string that caller must free()
static char *build_rerank_input(rp_rerank_toks *toks, const char *query, const char *document)
{
    // Calculate total length needed
    size_t total_len = strlen(query)    +
                       strlen(document) +
                       toks->len        + 1;
    // Allocate buffer
    char *input = (char *)malloc(total_len);
    if (!input)
        return NULL;

    snprintf(input, total_len, "%s%s%s%s%s%s", toks->bos, query, toks->eos, toks->sep, document, toks->eos);

    return input;
}

// Helper to tokenize text for reranking
static llama_token *tokenize_for_rerank(duk_context *ctx, struct llama_context *lctx, const struct llama_vocab *vocab,
                                        const char *text, int *n_tokens, bool add_special, bool parse_special)
{
    // Get rough estimate of tokens needed
    int max_tokens = strlen(text) + 32;
    llama_token *tokens = NULL;
    REMALLOC(tokens, max_tokens * sizeof(llama_token));

    int n = llama_tokenize(vocab, text, strlen(text), tokens, max_tokens, add_special, parse_special);

    if (n < 0)
    {
        // Need more space
        max_tokens = -n;
        REMALLOC(tokens, max_tokens * sizeof(llama_token));
        n = llama_tokenize(vocab, text, strlen(text), tokens, max_tokens, add_special, parse_special);
    }

    if (n < 0)
    {
        free(tokens);
        RP_THROW(ctx, "Failed to tokenize text for reranking");
        return NULL;
    }

    *n_tokens = n;
    return tokens;
}

float rerank_one(duk_context *ctx, struct llama_context *lctx, struct llama_model *lmodel,
    const struct llama_vocab *vocab, rp_rerank_toks *toks, const char *query, const char *text)
{
    // Build input string
    char *input = build_rerank_input(toks, query, text);

    // Tokenize the input
    int n_tokens = 0;
    int n_ubatch = llama_n_ubatch(lctx);

    llama_token *tokens = tokenize_for_rerank(ctx, lctx, vocab, input, &n_tokens, true, true);
    free(input);

    if (!tokens)
        RP_THROW(ctx, "Failed to tokenize input for reranking");

    // Clear the KV cache using llama_memory_clear
    llama_memory_clear(llama_get_memory(lctx), true);

    // sanity: must be a rerank model
    if (llama_pooling_type(lctx) != LLAMA_POOLING_TYPE_RANK) {
        // not a reranker; this would return a full embedding vector
        return 0.0f;
    }

    // Create batch using llama_batch_init
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits)
    {
        llama_batch_free(batch);
        free(tokens);
        RP_THROW(ctx, "llama_batch_init failed for reranking");
        return 0;
    }

    //clamp to max batch size
    if(n_tokens > n_ubatch)
        n_tokens = n_ubatch;

    // Fill the batch
    for (int i = 0; i < n_tokens; i++)
    {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;           // position 0..n-1
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;     // single sequence id 0
        batch.logits[i] = 1;        // contribute to pooled embedding
    }
    batch.n_tokens = n_tokens;

    // llama.cpp b9494: must explicitly enable embeddings output mode; setting
    // cp.embeddings at context creation is no longer sufficient. Encoder models
    // (BERT/RANK reranker) have no KV memory, so use llama_encode (not decode).
    llama_set_embeddings(lctx, true);
    int ret = llama_decode(lctx, batch);

    llama_batch_free(batch);
    free(tokens);

    if (ret != 0)
        RP_THROW(ctx, "llama_decode failed for reranking");

    // Get embeddings using llama_get_embeddings_seq
    const float *emb = llama_get_embeddings_seq(lctx, 0);

    if (!emb)
        RP_THROW(ctx, "Failed to get embeddings for reranking");

    // For reranking models with RANK pooling, the first value is the relevance score
    return emb[0];
}

// Rerank function: takes query and text, returns a score
static duk_ret_t rerank_text(duk_context *ctx)
{
    // Get the reranker context
    duk_push_this(ctx);

    // Check if destroyed
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("destroyed")))
    {
        if (duk_get_boolean_default(ctx, -1, 0))
            RP_THROW(ctx, "reranker object was destroyed");
    }
    duk_pop(ctx);

    // Get model and context pointers
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;
    const struct llama_vocab *vocab = NULL;
    rp_rerank_toks *toks=NULL;

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("model")))
        lmodel = (struct llama_model *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("llama_ctx")))
        lctx = (struct llama_context *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rerank_toks")))
        toks = (rp_rerank_toks *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_thread"));
    int thrno = duk_get_int(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("ctx_pid"));
    int pidno = duk_get_int(ctx, -1);
    duk_pop(ctx);

    int curthr = get_thread_num();
    int curpid = (int)getpid();

    // get a new context if in a new thread.  Model stays the same.
    if (curthr != thrno || pidno != curpid )
    {
#ifdef HAVE_CUDA
        // forking after is bad, mkay
        if(pidno != curpid && has_gpu_backend() )
        {
            RP_THROW(ctx, "llama.cpp - cannot fork llama.cpp with CUDA initialized");
        }
#endif
        duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("cp_buf"));
        struct llama_context_params *cp_buf = duk_get_buffer_data(ctx, -1, NULL);
        duk_pop(ctx);
        lctx = llama_init_from_model(lmodel, *cp_buf);

        // model refcount for this copy's own context (released in emb_free)
        lgen_model_addref(lmodel);

        duk_push_pointer(ctx, lctx);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

        duk_push_int(ctx, curthr);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));
        duk_push_int(ctx, curpid);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_pid"));
    }


    if (!lmodel || !lctx || !toks)
        RP_THROW(ctx, "rerank: Invalid model or context");

    vocab = llama_model_get_vocab(lmodel);
    if (!vocab)
        RP_THROW(ctx, "rerank: Failed to get vocab from model");

    duk_pop(ctx); // pop 'this'

    // Get query and text arguments
    const char *query = REQUIRE_STRING(ctx, 0, "rerank: argument 1 (query) must be a String");
    if( duk_is_string(ctx, 1) )
    {
        double score = (double)rerank_one(ctx, lctx, lmodel, vocab, toks, query, duk_get_string(ctx, 1));
        duk_push_number(ctx, score);
        return 1;
    }

    REQUIRE_ARRAY(ctx, 1, "rerank: argument 2 (documents) must be a String or Array of Strings");
    const char *text = NULL;

    int scores_only = 0;
    if(!duk_is_undefined(ctx, 2))
    {
        scores_only = REQUIRE_BOOL(ctx, 2, "rerank: argument 3 (scoresOnly), if present, must be a Boolean");
    }
    duk_uarridx_t i=0, len = (duk_uarridx_t) duk_get_length(ctx, 1);
    double score;

    duk_push_array(ctx); //return scores
    if(scores_only)
    {
        for(;i<len;i++)
        {
            duk_get_prop_index(ctx, 1, i);
            text = REQUIRE_STRING(ctx, -1, "rerank: argument 2 (documents) must be a String or Array of Strings");
            duk_pop(ctx);
            score = (double) rerank_one(ctx, lctx, lmodel, vocab, toks, query, text);
            duk_push_number(ctx, score);
            duk_put_prop_index(ctx, -2, i); //score into return scores array
        }
    }
    else
    {
        for(;i<len;i++)
        {
            duk_push_object(ctx); //entry of {document:text, score: score}
            duk_get_prop_index(ctx, 1, i);
            text = REQUIRE_STRING(ctx, -1, "rerank: argument 2 (documents) must be a String or Array of Strings");
            duk_put_prop_string(ctx, -2, "document");
            score = (double) rerank_one(ctx, lctx, lmodel, vocab, toks, query, text);
            duk_push_number(ctx, score);
            duk_put_prop_string(ctx, -2, "score");
            duk_put_prop_index(ctx, -2, i); //entry into return scores array
        }
    }
    return 1;
}

// Initialize a reranking model
static duk_ret_t llamacpp_init_rerank(duk_context *ctx)
{
    const char *model = REQUIRE_STRING(ctx, 0, "init: argument 1 must be a string");
    duk_idx_t obj_idx = -1;

    if (duk_is_object(ctx, 1))
        obj_idx = 1;

    duk_push_object(ctx); // return object

    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

    // model + context options (llama-server parity), shared parser + embed mode.
    // RANK pooling is the rerank default (overridable via the pooling option).
    struct llama_model_params   mp = llama_model_default_params();
    struct llama_context_params cp = llama_context_default_params();
    cp.embeddings      = true;
    cp.pooling_type    = LLAMA_POOLING_TYPE_RANK;
    cp.n_threads       = -1;
    cp.n_ctx           = 0;
    cp.n_ubatch        = 0;
    cp.n_threads_batch = -1;
    if (obj_idx >= 0) { parse_common_opts(ctx, obj_idx, &mp, &cp); parse_embed_opts(ctx, obj_idx, &cp); }

    // shared, refcounted load (one refcount per context; released in emb_free)
    char lerr[256] = {0};
    lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:initRerank - Could not load ggml file '%s': %s", model, strerror(errno));

    // If user didn't specify nctx or ubatch, set both to model's max
    if (cp.n_ctx <= 0)
    {
        int n_train = llama_model_n_ctx_train(lmodel);

        // keep it tighter for rerank
        if (n_train > 1024)
            n_train = 1024;

        if (n_train > 0)
            cp.n_ctx = n_train;
    }

    if (cp.n_ubatch <= 0)
    {
        // default ubatch to 512, to keep it tight.  Unlike embeddings, extra is truncated.
        cp.n_ubatch = 512;
    }

    //prevent clamping
    cp.n_batch = cp.n_ubatch;

    // b9494: unified KV cache so the RANK context allocates memory (see
    // examples/embedding/embedding.cpp); without it the rerank score reads as 0.
    cp.n_seq_max  = (uint32_t)llama_max_parallel_sequences();
    cp.kv_unified = true;

    lctx = llama_init_from_model(lmodel, cp);

    if (!lctx)
    {
        llama_model_free(lmodel);
        RP_THROW(ctx, "rampart-llama-cpp:initRerank - Failed to init llama context from model");
    }

    // Store pointers
    duk_push_pointer(ctx, lmodel);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("model"));

    duk_push_pointer(ctx, lctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));

    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_pid"));

    // origin (never updated on rebuild): the one copy that frees rerank_toks
    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("emb_origin_thr"));
    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("emb_origin_pid"));

    void *cp_buf = duk_push_fixed_buffer(ctx, sizeof(struct llama_context_params));
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("cp_buf"));
    memcpy(cp_buf, &cp, sizeof(struct llama_context_params));

    //get bos, sep, eos tokens
    rp_rerank_toks *toks = NULL;
    REMALLOC(toks, sizeof(rp_rerank_toks));

    get_rr_toks(llama_model_get_vocab(lmodel), toks);

    duk_push_pointer(ctx, toks);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rerank_toks"));

    // Add the rerank function
    duk_push_c_function(ctx, rerank_text, 3);
    duk_put_prop_string(ctx, -2, "rerank");

    duk_push_c_function(ctx, emb_free, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, emb_free, 1);
    duk_set_finalizer(ctx, -2);

    return 1;
}

#define MAX_LOG_BUFFER 40960

#include <pthread.h>

struct llog_cap
{
    char *buf;
    char *pos;  // Current write position
    size_t len;
    size_t alloc;
    pthread_mutex_t mutex;
};

static void llamacpp_logger(enum ggml_log_level level, const char *text, void *ud)
{
    (void)level; // or filter by level
    struct llog_cap *cap = (struct llog_cap *)ud;

    pthread_mutex_lock(&cap->mutex);

    size_t text_len = strlen(text);

    // Check if adding new text would exceed maximum
    if (cap->len + text_len > MAX_LOG_BUFFER)
    {
        // Cut buffer in half, keep second half, prepend overflow warning
        static const char *warn = "WARN: log overflow\n";
        size_t wlen = strlen(warn);
        size_t half = cap->len / 2;
        size_t keep = cap->len - half;
        memmove(cap->buf + wlen, cap->buf + half, keep);
        memcpy(cap->buf, warn, wlen);
        cap->len = wlen + keep;
        cap->buf[cap->len] = '\0';
        cap->pos = cap->buf + cap->len;
    }

    // Ensure we have enough allocation
    if (cap->len + text_len + 1 > cap->alloc)
    {
        if (cap->alloc == 0)
        {
            cap->alloc = (cap->len + text_len) > 1023 ? (cap->len + text_len) * 2 : 1024;
            REMALLOC(cap->buf, cap->alloc);
            cap->buf[0] = '\0';
            cap->pos = cap->buf;  // Initialize position
        }
        else
        {
            size_t pos_offset = cap->pos - cap->buf;  // Save offset before realloc
            cap->alloc = (3 * cap->alloc) / 2;
            if (cap->len + text_len + 1 > cap->alloc)
                cap->alloc = (cap->len + text_len + 1) * 2;
            REMALLOC(cap->buf, cap->alloc);
            cap->pos = cap->buf + pos_offset;  // Restore position after realloc
        }
    }
    strcpy(cap->pos, text);
    cap->pos += text_len;
    cap->len += text_len;

    pthread_mutex_unlock(&cap->mutex);
}

static duk_ret_t getlog(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("caplog"));
    struct llog_cap *caplog = duk_get_pointer(ctx, -1);
    if (caplog)
    {
        pthread_mutex_lock(&caplog->mutex);
        if (caplog->buf)
            duk_push_string(ctx, caplog->buf);
        else
            duk_push_string(ctx, "");
        pthread_mutex_unlock(&caplog->mutex);
    }
    else
        RP_THROW(ctx, "Error getting log");
    return 1;
}

static duk_ret_t resetlog(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("caplog"));
    struct llog_cap *caplog = duk_get_pointer(ctx, -1);

    pthread_mutex_lock(&caplog->mutex);
    free(caplog->buf);
    caplog->buf = NULL;
    caplog->pos = NULL;
    caplog->alloc = 0;
    caplog->len = 0;
    pthread_mutex_unlock(&caplog->mutex);

    return 0;
}

void close_llama_on_exit(void *arg)
{
    llama_backend_free();
}
#ifdef LANGTOOLS_MAIN_INCLUDE
static duk_ret_t open_llama(duk_context *ctx)
#else
duk_ret_t duk_open_module(duk_context *ctx)
#endif
{
    struct llog_cap *cap = NULL;
    static int isloaded = 0;

    /* the return object */
    duk_push_object(ctx);

    if (!isloaded)
    {
#ifdef __APPLE__
        /* macOS 15+ ggml-metal "residency sets" GGML_ASSERT in their static
         * destructor at process exit if any Metal buffer outlives exit() — which
         * happens routinely here because an initGen engine is torn down
         * asynchronously on its owner thread (and the shared model cache may still
         * hold weights). Disable the feature by default (it is a marginal perf
         * optimization); set RAMPART_METAL_RESIDENCY=1 to keep it. This must run
         * before the first model load creates the Metal device. */
        if (!getenv("RAMPART_METAL_RESIDENCY"))
            setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
#endif
        CALLOC(cap, sizeof(struct llog_cap));
        pthread_mutex_init(&cap->mutex, NULL);
        llama_log_set(llamacpp_logger, cap);

        duk_push_pointer(ctx, cap);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("caplog"));

        isloaded = 1;
    }

    duk_push_c_function(ctx, llamacpp_init_embed, 2);
    duk_put_prop_string(ctx, -2, "initEmbed");

    duk_push_c_function(ctx, lg_init_gen, 2);
    duk_put_prop_string(ctx, -2, "__rawInitGen");   // raw per-thread slot engine (used by initGen's owner thread)

    duk_push_c_function(ctx, lg_init_gen_batched, 2);
    duk_put_prop_string(ctx, -2, "initGen");        // transparent cross-thread batching wrapper

    duk_push_c_function(ctx, llamacpp_init_rerank, 2);
    duk_put_prop_string(ctx, -2, "initRerank");

    duk_push_c_function(ctx, getlog, 0);
    duk_put_prop_string(ctx, -2, "getLog");

    duk_push_c_function(ctx, resetlog, 0);
    duk_put_prop_string(ctx, -2, "resetLog");

    add_exit_func(close_llama_on_exit, NULL);

    return 1;
}

