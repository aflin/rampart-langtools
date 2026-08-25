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
#include "ggml-backend.h"     /* ggml_backend_dev_by_type: CPU-only context fallback */
#include "llama_gen_shim.h"   /* C ABI for the multi-session generation engine */
#include "rp-chunker.h"        /* structure-aware embed chunking (shared with rampart-onnx) */
#include "rp-embed-cache.h"    /* content-keyed doc-result LRU (shared with rampart-onnx) */
#include "rampart.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#endif

#include <stdarg.h>

/* get_current_thread() lives in the rampart binary (like the duk_* symbols).
 * Keep the reference WEAK so a non-rampart host that dlopens this module just
 * for the rp_embed_* C API still loads and runs -- there lt_thr_ctx() below
 * returns NULL instead of faulting on the call. */
#pragma weak get_current_thread

/* ---- this.errMsg: warnings + non-fatal errors ------------------------------
 * Three rules: a failure throws a JS error; a warning goes to this.errMsg; NOTHING
 * is written to stdout/stderr (RAMPART_LT_GPU_DEBUG is the one opt-in hatch).
 *
 * errMsg mirrors rampart-sql (rp_log_copy_to_errMsg there): warnings accumulate on
 * `this` -- the module object for llamacpp.initGen(), a handle for handle methods,
 * exactly as Sql/sql share the property -- and are cleared at the top of each call.
 * It is deliberately SEPARATE from getLog(), which is ggml/llama's informational
 * firehose and would bury a warning like "GPU context init failed". */
/* The duk context of the CALLING rampart thread: reachable from ANY C code here --
 * the CUDA probe below, the gen shim, and the rp_embed_* exports that rampart-sql
 * calls -- not just from a duk_ret_t that was handed a ctx.  NULL only in a bare
 * non-rampart host (no JS to carry the warning). */
static duk_context *lt_thr_ctx(void)
{
    RPTHR *t = get_current_thread ? get_current_thread() : NULL;
    return t ? t->ctx : NULL;
}

static void lt_errmsg_append(duk_context *ctx, const char *msg);

/* A warning, from anywhere in the module (including llama_gen_shim.cc).  Goes
 * straight onto this.errMsg -- no buffer, no drain.  RAMPART_LT_GPU_DEBUG=1 also
 * echoes to stderr; that opt-in hatch is the only thing that may write there. */
void lt_warn(const char *fmt, ...)
{
    duk_context *ctx = lt_thr_ctx();
    char line[1024];
    size_t l;
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (getenv("RAMPART_LT_GPU_DEBUG")) fputs(line, stderr);   /* opt-in hatch */
    if (!ctx) return;                  /* non-rampart host: no JS to carry it */
    l = strlen(line);
    while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = '\0';
    if (l) lt_errmsg_append(ctx, line);
}

#define LT_MODULE_STASH DUK_HIDDEN_SYMBOL("lt_module")

/* Push the object a warning belongs on: `this` when there is one (the module for
 * llamacpp.initEmbed(), a handle for handle methods -- as Sql/sql share errMsg),
 * else the module object itself.  The fallback matters because some entry points
 * are reached with no `this`: initGen dispatches through a JS wrapper, and the
 * rp_embed_* exports are called from rampart-sql.  Returns 0 if neither exists
 * (nothing is stashed until duk_open_module runs). */
static int lt_push_errmsg_target(duk_context *ctx)
{
    duk_push_this(ctx);
    if (duk_is_object(ctx, -1)) return 1;
    duk_pop(ctx);
    duk_push_global_stash(ctx);
    if (!duk_get_prop_string(ctx, -1, LT_MODULE_STASH) || !duk_is_object(ctx, -1)) {
        duk_pop_2(ctx);
        return 0;
    }
    duk_remove(ctx, -2);                 /* drop the stash, leave the module object */
    return 1;
}

static void lt_errmsg_append(duk_context *ctx, const char *msg)
{
    if (!lt_push_errmsg_target(ctx)) return;
    if (duk_get_prop_string(ctx, -1, "errMsg")) {
        const char *s = duk_get_string(ctx, -1);
        if (s && *s) duk_push_sprintf(ctx, "%s\n%s", s, msg);
        else         duk_push_string(ctx, msg);
        duk_remove(ctx, -2);
    } else {
        duk_pop(ctx);
        duk_push_string(ctx, msg);
    }
    duk_put_prop_string(ctx, -2, "errMsg");
    duk_pop(ctx);
}

/* clear this.errMsg -- at the top of every JS entry point */
static void lt_errmsg_clear(duk_context *ctx)
{
    if (!lt_push_errmsg_target(ctx)) return;
    duk_del_prop_string(ctx, -1, "errMsg");
    duk_pop(ctx);
}


// --- CUDA availability check ---
#if ( defined(LT_ENABLE_GPU) && !defined(__APPLE__) )
    #define HAVE_CUDA 1

    #include <cuda_runtime.h>
    #include "ggml-backend.h"

    /* sm numbers this build has NATIVE SASS for, baked from CMAKE_CUDA_ARCHITECTURES
     * at configure (e.g. "87 90 100 120").  ggml rewrites Blackwell/Hopper archs to
     * family-specific 'a' kernels that do NOT JIT across SM versions, so a native-SASS
     * match for the device's EXACT compute capability is the reliable test.  PTX
     * (-virtual) archs are intentionally excluded: ggml's 'a' PTX won't JIT onto a
     * newer SM (a cu12 module on a GB10/sm_121 proved it). */
#ifndef LT_CUDA_SM_LIST
#define LT_CUDA_SM_LIST ""
#endif
    /* 1 if this build can run on `device` (native SASS for its cc), or if there's no
     * CUDA GPU / the list wasn't baked (don't block).  Else fills `eb` and returns 0
     * so init fails cleanly instead of ggml's CUDA_CHECK aborting mid-graph. */
    static int lt_gpu_kernel_supported(int device, char *eb, size_t n)
    {
        int dbg = getenv("RAMPART_LT_GPU_DEBUG") != NULL;
        if (LT_CUDA_SM_LIST[0] == '\0') {            /* arch list not baked -> don't block */
            if (dbg) fprintf(stderr, "[lt-gpu] sm list empty -> allow\n");
            return 1;
        }
        /* Query CUDA DIRECTLY, not has_gpu_backend(): ggml registers its GPU backend
         * lazily (at model load / first compute), so at init time has_gpu_backend() is
         * still 0 even on a working GPU -- which is exactly why this check never fired.
         * cudaGetDeviceCount initializes and queries the runtime right now. */
        int ndev = 0;
        cudaError_t ce = cudaGetDeviceCount(&ndev);
        if (dbg) fprintf(stderr, "[lt-gpu] cudaGetDeviceCount=%d (err=%d) sm='%s'\n",
                         ndev, (int)ce, LT_CUDA_SM_LIST);
        if (ce != cudaSuccess || ndev <= 0) {
            /* No CUDA GPU: ggml registers no CUDA backend and the context is built
             * on CPU -- it works, but silently.  Warn once (this is a GPU build, so
             * the user expected the GPU) so the drop to CPU is visible, mirroring
             * the Metal fallback notice. */
            static int warned = 0;
            if (!warned) {
                warned = 1;
                lt_warn("rampart-llamacpp: no usable CUDA GPU (%s) -- running on CPU\n",
                        ce != cudaSuccess ? cudaGetErrorString(ce) : "0 devices");
            }
            return 1;  /* no CUDA GPU -> CPU build runs, allow */
        }
        if (device < 0 || device >= ndev) device = 0;
        struct cudaDeviceProp p;
        cudaError_t ge = cudaGetDeviceProperties(&p, device);
        if (ge != cudaSuccess) {                     /* can't tell -> allow */
            if (dbg) fprintf(stderr, "[lt-gpu] cudaGetDeviceProperties(%d) err=%d -> allow\n", device, (int)ge);
            return 1;
        }
        int cc = p.major * 10 + p.minor;             /* 12.1 -> 121, 8.7 -> 87 */
        if (dbg) fprintf(stderr, "[lt-gpu] device %d '%s' cc %d.%d (%d) vs sm '%s'\n",
                         device, p.name, p.major, p.minor, cc, LT_CUDA_SM_LIST);
        /* Driver-version floor.  This module's kernels are native-arch SASS built by the
         * oven's nvcc (CUDART_VERSION).  A cubin built with toolkit X will NOT load on a
         * driver older than X (NVIDIA guarantees cubin forward-compat only, not backward)
         * -- so on a too-old driver ggml aborts mid-graph in ggml_cuda_error (seen: a
         * CUDA-12.8 cu12 module on a 535 / CUDA-12.2 driver -> abort in ggml_cuda_op_add).
         * cudaDriverGetVersion() reports the max CUDA the installed driver supports; refuse
         * cleanly when it's below this module's build version instead of letting the launch
         * abort with a cryptic backtrace.  Checked BEFORE the sm-list match on purpose: the
         * cc can be present yet the SASS still fail to load on an old driver.  (onnx/ORT is
         * immune -- it JITs kernels from PTX; ggml ships real-arch SASS only.) */
        int drv = 0;
        if (cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0 && drv < CUDART_VERSION) {
            if (dbg) fprintf(stderr, "[lt-gpu] driver CUDA %d.%d < build CUDA %d.%d -> REFUSE\n",
                             drv/1000, (drv%1000)/10, CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
            if (eb && n)
                snprintf(eb, n,
                    "GPU '%s' (sm_%d): the NVIDIA driver supports CUDA %d.%d, but this "
                    "rampart-llamacpp was built for CUDA %d.%d -- the driver is too old to "
                    "load its GPU kernels (they would abort mid-graph).  Upgrade the NVIDIA "
                    "driver to one shipping CUDA %d.%d or newer, or use the cuNN module that "
                    "matches your driver (e.g. cu11 for a CUDA 11.x driver).",
                    p.name, cc, drv/1000, (drv%1000)/10,
                    CUDART_VERSION/1000, (CUDART_VERSION%1000)/10,
                    CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
            return 0;
        }
        for (const char *s = LT_CUDA_SM_LIST; *s; ) {
            char *end;
            long v = strtol(s, &end, 10);
            if (end == s) { s++; continue; }         /* skip spaces/separators */
            if ((int)v == cc) {                      /* native SASS for this device */
                if (dbg) fprintf(stderr, "[lt-gpu] cc %d supported -> allow\n", cc);
                return 1;
            }
            s = end;
        }
        if (dbg) fprintf(stderr, "[lt-gpu] cc %d NOT in sm list -> REFUSE\n", cc);
        if (eb && n)
            snprintf(eb, n,
                "GPU '%s' (compute %d.%d / sm_%d) has no compatible kernels in this "
                "rampart-llamacpp build (CUDA %d; built for sm: %s). Use the cuNN module "
                "matching your GPU (e.g. cu13 for Blackwell GB10 / sm_121), or rebuild "
                "adding sm_%d.",
                p.name, p.major, p.minor, cc, CUDART_VERSION / 1000, LT_CUDA_SM_LIST, cc);
        return 0;
    }

#else

    #undef HAVE_CUDA

#endif

/* Fork policy (Aaron, 2026-07): rampart forks only at STARTUP (rampart-server
 * daemon mode, which has an explicit postForkFunc); forking after models are
 * live is unsupported.  GPU runtimes (CUDA, macOS Metal) usually CRASH when a
 * forked child touches inherited driver state, so any post-fork use of a
 * GPU-backed handle is REFUSED with a clear error.  CPU-only operation is
 * allowed to continue after a fork (contexts are rebuilt per pid).
 *
 * lt_gpu_in_use(): true iff ggml has a GPU-class backend device registered
 * (CUDA on Linux, Metal on macOS).  Registration happens at model load, so by
 * the time a post-fork check runs (a model existed before the fork), the
 * inherited registry answers correctly.  Pure-CPU builds compile no GPU
 * backends and always return 0. */
static int lt_gpu_in_use(void)
{
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU)
            return 1;
    return 0;
}

/* ---- embed defaults, settable from JS via llamacpp.embedDefaults() -------
 * These are the defaults for initEmbed()'s options AND the only way to
 * configure the rp_embed_* C entry points that rampart-sql drives, since
 * those take no options object.  Process-global and deliberately simple:
 * set them once at startup, before models are loaded.
 *
 *  batchChunks  -1 = auto: pack chunks into one decode on a GPU backend, one
 *                  chunk per decode on CPU.  Measured on an RTX 4070 Ti:
 *                  2.1x faster batched; on a 16-core CPU: 1.03x, i.e. nothing.
 *                  Batching also perturbs vectors slightly (different matmul
 *                  kernels for a larger batch), so it is not worth enabling
 *                  where it does not pay.  1 = never batch, >1 = cap the
 *                  sequences per decode.
 *  threads      per-token thread count (n_threads).
 *  threadsBatch multi-token/prompt thread count (n_threads_batch) -- the one
 *               that matters for embedding, where every decode is multi-token.
 *               -1 hands the choice to ggml, which uses GGML_DEFAULT_N_THREADS
 *               (4) REGARDLESS of core count: llama.cpp passes the value
 *               through unclamped (llama-context.cpp) and ggml-cpu.c falls
 *               back to the constant.  Set it explicitly to use the machine. */
static int g_embed_batch_chunks  = -1;
static int g_embed_batch_tokens  = 512;
static int g_embed_threads       = 1;
static int g_embed_threads_batch = -1;

/* Resolve a batchChunks setting against a live context: returns the max
 * sequences per decode, where 1 means "one chunk per decode" (identical to
 * the pre-batching code path). */
static int ll_resolve_batch(struct llama_context *lctx, int setting)
{
    int cap = lctx ? (int)llama_n_seq_max(lctx) : 1;
    if (cap < 1) cap = 1;
    if (setting < 0) setting = lt_gpu_in_use() ? cap : 1;   /* auto */
    if (setting <= 1) return 1;
    return setting < cap ? setting : cap;
}

#define LT_FORK_REFUSAL "llama.cpp: this handle was created before a fork() and " \
    "a GPU backend (CUDA/Metal) is initialized -- using it in the child would " \
    "crash the GPU runtime. Fork before loading models (rampart-server daemon " \
    "mode + postForkFunc), or run this model on CPU."


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
        if (duk_is_string(ctx, -1)) {
            char *d = strdup(duk_get_string(ctx, -1));
            if (d) stops[k++] = d;
        }
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
    /* mix in a counter so same-second requests don't share a default seed */
    static uint32_t seed_ctr = 0;
    req->seed = (uint32_t)time(NULL) + __atomic_add_fetch(&seed_ctr, 1, __ATOMIC_RELAXED);
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

    /* ---- tool calling.  The array is JSON-encoded here and parsed in the shim
       by common_chat_tools_parse_oaicompat (OpenAI shape), so the wire format
       between the two layers is just JSON -- no struct marshalling. */
    req->tool_choice = LGEN_TOOL_CHOICE_AUTO;
    if (duk_get_prop_string(ctx, obj_idx, "tools")) {
        if (!duk_is_array(ctx, -1)) RP_THROW(ctx, "tools must be an Array");
        if (duk_get_length(ctx, -1) > 0) {
            duk_dup(ctx, -1);
            /* The encoded string MUST stay on the stack: req->tools_json points
               into it and duktape frees the value as soon as it is popped (the
               same reason `messages` is left there).  Drop the array underneath
               instead, so the string survives until duk_set_top after submit. */
            req->tools_json = duk_json_encode(ctx, -1);
            duk_remove(ctx, -2);
        } else {
            duk_pop(ctx);   /* empty tools array == no tools */
        }
    } else {
        duk_pop(ctx);
    }
    if (duk_get_prop_string(ctx, obj_idx, "toolChoice")) {
        const char *tc = REQUIRE_STRING(ctx, -1, "toolChoice must be a String (auto|none|required)");
        if      (!strcmp(tc, "auto"))     req->tool_choice = LGEN_TOOL_CHOICE_AUTO;
        else if (!strcmp(tc, "none"))     req->tool_choice = LGEN_TOOL_CHOICE_NONE;
        else if (!strcmp(tc, "required")) req->tool_choice = LGEN_TOOL_CHOICE_REQUIRED;
        else RP_THROW(ctx, "toolChoice: unknown '%s' (auto|none|required)", tc);
    }
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, obj_idx, "parallelToolCalls"))
        req->parallel_tool_calls = REQUIRE_BOOL(ctx, -1, "parallelToolCalls must be a Boolean");
    duk_pop(ctx);

    /* Run the chat parser without tools, so reasoning is separated from the
       answer.  Needed wherever the caller must machine-read the reply: a format
       that carries reasoning as a channel (rather than a <think> span) cannot be
       stripped by the caller -- only llama.cpp's parser knows where it ends. */
    if (duk_get_prop_string(ctx, obj_idx, "reasoning"))
        req->reasoning_separate = REQUIRE_BOOL(ctx, -1, "reasoning must be a Boolean");
    duk_pop(ctx);

    /* Ask the model not to deliberate.  Tri-state: absent leaves the template's
       own default alone.  Silently ignored by templates that don't support it --
       unlike tools, there is no wrong output to guard against. */
    req->thinking = -1;
    if (duk_get_prop_string(ctx, obj_idx, "thinking"))
        req->thinking = REQUIRE_BOOL(ctx, -1, "thinking must be a Boolean") ? 1 : 0;
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
/* non-throwing core (lg_init_gen must free its chat_template before a throw) */
/* defined with the log capture, far below */
static size_t lg_log_mark(void);
static void   lg_load_reason(size_t from, char *out, size_t outlen);

/* Model-load progress, for initGenAsync().  llama.cpp calls the callback on
 * the LOADING thread, inside the load, so it may only take the lock and
 * store a float -- never re-enter duktape.  A waiting thread reads it by
 * id.  Returning false would cancel the load; nothing does yet. */
#define LG_PROG_SLOTS 8
static struct lg_prog { char id[64]; float pct; int busy; int cancel; } lg_prog[LG_PROG_SLOTS];
static pthread_mutex_t lg_prog_mtx = PTHREAD_MUTEX_INITIALIZER;

static struct lg_prog *lg_prog_claim(const char *id)
{
    struct lg_prog *s = NULL;
    if (!id || !*id) return NULL;
    pthread_mutex_lock(&lg_prog_mtx);
    for (int i = 0; i < LG_PROG_SLOTS; i++)
        if (!lg_prog[i].busy) { s = &lg_prog[i]; break; }
    if (s) {
        snprintf(s->id, sizeof s->id, "%s", id);
        s->pct = 0.0f;
        s->cancel = 0;
        s->busy = 1;
    }
    pthread_mutex_unlock(&lg_prog_mtx);
    return s;
}

static void lg_prog_release(struct lg_prog *s)
{
    if (!s) return;
    pthread_mutex_lock(&lg_prog_mtx);
    s->busy = 0; s->id[0] = 0; s->pct = 0.0f; s->cancel = 0;
    pthread_mutex_unlock(&lg_prog_mtx);
}

static bool lg_on_load_progress(float pct, void *ud)
{
    struct lg_prog *s = (struct lg_prog *)ud;
    int go = 1;
    if (s) {
        pthread_mutex_lock(&lg_prog_mtx);
        s->pct = pct;
        go = !s->cancel;        /* false here tells llama.cpp to stop loading */
        if (!go) s->cancel = 2; /* 2 = the load actually saw it and stopped */
        pthread_mutex_unlock(&lg_prog_mtx);
    }
    return go ? true : false;
}

/* __cancelLoad(id) -> true if a load with that id was in flight.  The load
 * stops at its next progress report, so a cancel is not instant. */
static duk_ret_t lg_cancel_load(duk_context *ctx)
{
    const char *id = duk_get_string(ctx, 0);
    int found = 0;
    if (id && *id) {
        pthread_mutex_lock(&lg_prog_mtx);
        for (int i = 0; i < LG_PROG_SLOTS; i++)
            if (lg_prog[i].busy && !strcmp(lg_prog[i].id, id)) {
                lg_prog[i].cancel = 1; found = 1; break;
            }
        pthread_mutex_unlock(&lg_prog_mtx);
    }
    duk_push_boolean(ctx, found);
    return 1;
}

/* __loadProgress(id) -> 0..1 while loading, -1 when not in flight */
static duk_ret_t lg_load_progress(duk_context *ctx)
{
    const char *id = duk_get_string(ctx, 0);
    float pct = -1.0f;
    if (id && *id) {
        pthread_mutex_lock(&lg_prog_mtx);
        for (int i = 0; i < LG_PROG_SLOTS; i++)
            if (lg_prog[i].busy && !strcmp(lg_prog[i].id, id)) { pct = lg_prog[i].pct; break; }
        pthread_mutex_unlock(&lg_prog_mtx);
    }
    duk_push_number(ctx, (double)pct);
    return 1;
}

static rp_llama_info *lg_new_info_e(const lgen_engine_params *p, char *err, size_t errlen)
{
    lgen_engine *eng = lgen_engine_create(p, err, errlen);
    if (!eng) return NULL;
    rp_llama_info *info = NULL;
    CALLOC(info, sizeof(rp_llama_info));
    info->thr = get_current_thread();
    info->eng = eng;
    info->init_thr = get_thread_num();
    info->init_pid = (int)getpid();
    return info;
}

static rp_llama_info *lg_new_info(duk_context *ctx, const lgen_engine_params *p)
{
    char err[256];
    rp_llama_info *info = lg_new_info_e(p, err, sizeof err);
    if (!info) RP_THROW(ctx, "initGen: %s", err);
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

    if (owner_pid != -1 && owner_pid != cur_pid && lt_gpu_in_use())
        RP_THROW(ctx, LT_FORK_REFUSAL);

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

/* LGEN_FINISH_* -> the JS `finishReason` string */
static const char *lg_finish_name(int reason)
{
    switch (reason) {
        case LGEN_FINISH_TOOL_CALLS: return "tool_calls";
        case LGEN_FINISH_LENGTH:     return "length";
        case LGEN_FINISH_CANCEL:     return "cancel";
        case LGEN_FINISH_ERROR:      return "error";
        default:                     return "stop";   /* STOP and EOG both */
    }
}

/* Attach finishReason / toolCalls / reasoning to the object on top of the stack.
   `toolCalls` is left ABSENT (not an empty array) when the model called nothing,
   so `if (res.toolCalls)` is the caller's test. */
static void lg_push_result_extra(duk_context *ctx, int reason, const lgen_result_extra *extra)
{
    duk_push_string(ctx, lg_finish_name(reason));
    duk_put_prop_string(ctx, -2, "finishReason");
    if (extra && extra->tool_calls_json) {
        duk_push_string(ctx, extra->tool_calls_json);
        duk_json_decode(ctx, -1);   /* shim-generated, always valid JSON */
        duk_put_prop_string(ctx, -2, "toolCalls");
    }
    if (extra && extra->reasoning) {
        duk_push_lstring(ctx, extra->reasoning, (duk_size_t)extra->reasoning_len);
        duk_put_prop_string(ctx, -2, "reasoning");
    }
}

typedef struct {
    int    done;
    int    status;
    char  *err;
    char  *full;
    size_t full_len;
    int    reason;
    char  *tool_calls_json;   /* strdup'd: the shim's copy dies with the callback */
    char  *reasoning;
    size_t reasoning_len;
} lg_sync_ud;

static void lg_sync_on_done(void *ud, int status, const char *err, int reason,
                            const char *full, size_t full_len,
                            const lgen_result_extra *extra)
{
    lg_sync_ud *s = (lg_sync_ud *)ud;
    s->status = status;
    s->reason = reason;
    if (status != 0 && err) s->err = strdup(err);
    if (full && full_len) {
        s->full = NULL;
        REMALLOC(s->full, full_len + 1);
        memcpy(s->full, full, full_len);
        s->full[full_len] = '\0';
        s->full_len = full_len;
    }
    if (extra && extra->tool_calls_json) s->tool_calls_json = strdup(extra->tool_calls_json);
    if (extra && extra->reasoning) {
        s->reasoning = NULL;
        REMALLOC(s->reasoning, extra->reasoning_len + 1);
        memcpy(s->reasoning, extra->reasoning, extra->reasoning_len);
        s->reasoning[extra->reasoning_len] = '\0';
        s->reasoning_len = extra->reasoning_len;
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

    /* predict() has always returned a plain String.  Callers that supply `tools`
       are opting into the structured form, so only they get an Object back --
       every existing caller keeps the String it has today. */
    int want_object = (req.tools_json != NULL) || req.reasoning_separate;

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
        if (s.tool_calls_json) free(s.tool_calls_json);
        if (s.reasoning) free(s.reasoning);
        duk_push_error_object(ctx, DUK_ERR_ERROR, "%s", e);
        free(e);
        (void)duk_throw(ctx);
    }

    if (info->last_out) free(info->last_out);
    info->last_out = s.full; info->last_out_len = s.full_len;

    if (want_object) {
        lgen_result_extra extra;
        memset(&extra, 0, sizeof extra);
        extra.tool_calls_json = s.tool_calls_json;
        extra.reasoning       = s.reasoning;
        extra.reasoning_len   = s.reasoning_len;
        duk_push_object(ctx);
        duk_push_lstring(ctx, info->last_out ? info->last_out : "", info->last_out_len);
        duk_put_prop_string(ctx, -2, "fullText");
        lg_push_result_extra(ctx, s.reason, &extra);
    } else {
        duk_push_lstring(ctx, info->last_out ? info->last_out : "", info->last_out_len);
    }
    if (s.tool_calls_json) free(s.tool_calls_json);
    if (s.reasoning) free(s.reasoning);
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
static void lg_infer_on_piece(void *ud, const char *piece, size_t len, int is_reasoning)
{
    lg_areq *r = (lg_areq *)ud;
    duk_context *ctx = r->ctx;
    if (r->canceled) return;
    if (lg_areq_get_cb(ctx, r, "lgtok_")) {
        duk_push_object(ctx);
        duk_push_lstring(ctx, piece ? piece : "", (duk_size_t)len);
        duk_put_prop_string(ctx, -2, "token");
        /* deliberation rather than the answer; absent for ordinary tokens so
           `if (t.reasoning)` is the test */
        if (is_reasoning) { duk_push_true(ctx); duk_put_prop_string(ctx, -2, "reasoning"); }
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
                             const char *full, size_t full_len,
                             const lgen_result_extra *extra)
{
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
        lg_push_result_extra(ctx, reason, extra);
        if (duk_pcall(ctx, 1) != 0) { /* swallow */ }
        duk_pop(ctx);
    }
    if (full && full_len && r->info) {
        if (r->info->last_out) free(r->info->last_out);
        r->info->last_out = NULL;
        REMALLOC(r->info->last_out, full_len + 1);
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

/* inner teardown: expects the gen object at the TOP of the stack.  The
   destroy() method wrapper pushes `this`; as a finalizer duktape passes the
   object as the argument (duk_push_this would yield undefined there). */
static duk_ret_t lg_destroy_(duk_context *ctx)
{
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

static duk_ret_t lg_destroy(duk_context *ctx)
{
    duk_push_this(ctx);
    return lg_destroy_(ctx);
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
    /* b10446 replaced the independent use_mmap/use_mlock booleans with a single
     * load_mode enum.  Keep the JS option names and fold them back into it.
     * LLAMA_LOAD_MODE_AUTO already resolves to (mmap on, mlock off) -- exactly
     * the old defaults -- so only override it when the script asked for
     * something, which also preserves llama.cpp's "auto" load-mode log line. */
    {
        int mmap_set = 0, mlock_set = 0, want_mmap = 1, want_mlock = 0;
        if (lt_opt_bool2(ctx, o, "useMmap",  NULL, "useMmap must be boolean",  &b)) { mmap_set  = 1; want_mmap  = b; }
        if (lt_opt_bool2(ctx, o, "useMlock", NULL, "useMlock must be boolean", &b)) { mlock_set = 1; want_mlock = b; }
        if (mmap_set || mlock_set) {
            mp->load_mode = want_mmap
                ? (want_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                : (want_mlock ? LLAMA_LOAD_MODE_MLOCK      : LLAMA_LOAD_MODE_NONE);
        }
    }
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
    lt_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    const char *model_path = REQUIRE_STRING(ctx, 0, "initGen: first argument must be a String (path to .gguf)");
    duk_idx_t o = duk_is_object(ctx, 1) ? 1 : -1;

    /* defaults: one slot, one CPU thread (the GPU does the math), n_ctx 0 => the
       model's trained max (resolved at context build). flashAttn defaults to auto
       from llama_context_default_params(). */
    struct llama_model_params   mp = llama_model_default_params();
    struct llama_context_params cp = llama_context_default_params();
    cp.n_seq_max       = 1;
    /* Tune to the machine rather than hard-coding: n_threads_batch decides how
       long a reader waits for the first token, and a retrieval payload is
       thousands of tokens of prompt -- at the old default of 1 that was one core
       at 100% while the rest idled.  lgen_default_n_threads() is libcommon's own
       heuristic (common_cpu_get_num_math): physical/performance cores, ignoring
       hyperthread siblings, which is what llama.cpp's tools resolve to and what
       actually helps matmul.  `threads` / `threadsBatch` still override both.

       Note this assumes ONE engine per process, which is the initGen case (its
       coordinator runs a single shared engine on a dedicated owner thread).
       Several concurrent engines would oversubscribe -- set `threads` explicitly
       if you build more than one. */
    cp.n_threads       = lgen_default_n_threads();
    cp.n_threads_batch = cp.n_threads;
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
            if (!duk_is_string(ctx, -1)) {
                free(chat_template);
                RP_THROW(ctx, "chatTemplateFile must be a string");
            }
            const char *fn = duk_get_string(ctx, -1);
            free(chat_template);
            chat_template = NULL;
            chat_template = lt_read_file_alloc(ctx, fn);
        }
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, o, "mmproj")) { duk_pop(ctx); free(chat_template); RP_THROW(ctx, "initGen: vision (mmproj) generation is not supported"); }
        duk_pop(ctx);
    }

#if HAVE_CUDA
    {  /* GPU build: always check; lt_gpu_kernel_supported self-gates on has_gpu_backend */
        char eb[512];
        int dev = mp.main_gpu >= 0 ? mp.main_gpu : 0;
        if (!lt_gpu_kernel_supported(dev, eb, sizeof eb)) {
            free(chat_template);
            RP_THROW(ctx, "initGen: %s", eb);
        }
    }
#endif

    lgen_engine_params p;
    memset(&p, 0, sizeof p);
    p.model_path    = model_path;
    p.chat_template = chat_template;   /* shim copies into the engine */
    p.use_jinja     = use_jinja;
    /* opts.__loadId, set by the coordinator, opts a load into progress
     * reporting.  A cache hit loads nothing and reports nothing, which is
     * why a caller must treat "no progress" as normal. */
    struct lg_prog *prog = NULL;
    if (o > -1 && duk_get_prop_string(ctx, o, "__loadId")) {
        prog = lg_prog_claim(duk_get_string(ctx, -1));
        if (prog) {
            mp.progress_callback           = lg_on_load_progress;
            mp.progress_callback_user_data = prog;
        }
    }
    if (o > -1) duk_pop(ctx);

    p.mparams       = mp;
    p.cparams       = cp;

    char nerr[256] = {0};
    size_t logmark = lg_log_mark();
    rp_llama_info *info = lg_new_info_e(&p, nerr, sizeof nerr); // engine (shared model) + info
    int cancelled = 0;
    if (prog) {
        pthread_mutex_lock(&lg_prog_mtx);
        cancelled = (prog->cancel == 2);
        pthread_mutex_unlock(&lg_prog_mtx);
    }
    lg_prog_release(prog);
    if (!info) {
        char why[512];
        lg_load_reason(logmark, why, sizeof why);
        free(chat_template);   /* would leak across the longjmp otherwise */
        if (cancelled) RP_THROW(ctx, "initGen: load cancelled");
        if (why[0]) RP_THROW(ctx, "initGen: %s -- %s", nerr, why);
        RP_THROW(ctx, "initGen: %s", nerr);
    }
    lgen_engine *eng = info->eng;

    duk_push_object(ctx);
    duk_push_int(ctx, (int)lgen_engine_n_ctx(eng));
    duk_rp_put_prop_string_ro(ctx, -2, "nCtx");
    duk_push_int(ctx, (int)lgen_engine_n_vocab(eng));
    duk_rp_put_prop_string_ro(ctx, -2, "nVocab");

    /* Tool-calling capability.  There is no upstream query for this, so the shim
       probes the template at create time; a caller checks it before building a
       request (supplying tools to a template that can't render them throws). */
    duk_push_boolean(ctx, lgen_engine_supports_tools(info->eng));
    duk_rp_put_prop_string_ro(ctx, -2, "supportsTools");
    duk_push_string(ctx, lgen_engine_chat_format(info->eng));
    duk_rp_put_prop_string_ro(ctx, -2, "chatFormat");
    duk_push_boolean(ctx, lgen_engine_supports_thinking_toggle(info->eng));
    duk_rp_put_prop_string_ro(ctx, -2, "supportsThinkingToggle");

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
    duk_push_c_function(ctx, lg_destroy_, 1); /* finalizer: object arrives as the argument */
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
"(function(mod, model, opts, uid, onprog, ondone){\n"
"  var thread = rampart.thread;\n"
"  var owner = new thread();\n"
"  owner.exec(function(a){\n"
/* A load that fails here must reach the CALLER.  Without this the owner
   dies before publishing `ready_', and initGen blocks for the full
   timeout and then returns a handle with no engine behind it. */
"    var raw;\n"
"    try { a.opts.__loadId = a.uid; raw = a.rawInit(a.model, a.opts); }\n"
"    catch(e) { rampart.thread.put('ready_'+a.uid, { err: String(e.message||e) }); return; }\n"
"    var canc = {};\n"   /* request ids signalled to cancel (set by the cancel event) */
"    rampart.event.on('can_'+a.uid, 'c', function(uv, c){ canc[c.id] = 1; });\n"
"    rampart.event.on('sub_'+a.uid, 'h', function(uv, r){\n"
/* The request crosses a thread boundary as a JSON string, not as a live object:
   the event transport does not round-trip nested Arrays faithfully (an
   assistant turn's tool_calls arrived as {\"0\":{...}}), which broke tool loops
   and silently dropped scalars like toolChoice.  Stringify/parse is exact. */
"      r.req = (typeof r.req === 'string') ? JSON.parse(r.req) : r.req;\n"
"      if (r.stream) {\n"
"        raw.predictAsync(r.req,\n"
"          function(t){ if (canc[r.id]) { delete canc[r.id]; return true; }\n"   /* hard cancel -> frees the slot */
"                       if (!t.done && !t.error && t.token) rampart.event.trigger('tok_'+a.uid+'_'+r.id, { tok: t.token, r: t.reasoning ? 1 : 0 }); },\n"
"          function(res){ delete canc[r.id]; rampart.event.trigger('fin_'+a.uid+'_'+r.id, { full: res.fullText || '', err: res.error,\n"
"                           toolCalls: res.toolCalls, reasoning: res.reasoning, finishReason: res.finishReason }); });\n"
"      } else {\n"
"        raw.predictAsync(r.req, function(){}, function(res){\n"
"          rampart.thread.put('res_'+a.uid+'_'+r.id,\n"
"            { text: res.error ? ('[gen err:'+res.error+']') : (res.fullText || ''),\n"
"              toolCalls: res.toolCalls, reasoning: res.reasoning, finishReason: res.finishReason });\n"
"        });\n"
"      }\n"
"    });\n"
"    rampart.thread.put('ready_'+a.uid, { nCtx: raw.nCtx, nVocab: raw.nVocab,\n"
"                                         supportsTools: raw.supportsTools, chatFormat: raw.chatFormat,\n"
"                                         supportsThinkingToggle: raw.supportsThinkingToggle });\n"
"  }, { rawInit: mod.__rawInitGen, model: model, opts: opts || {}, uid: uid });\n"
"  function fail(m){ return new Error(String(m).indexOf('initGen') === 0\n"
"                                    ? m : 'initGen: ' + m); }\n"
"  function mkgen(meta){ return {\n"
"    __uid: uid, nCtx: meta.nCtx, nVocab: meta.nVocab,\n"
"    supportsTools: !!meta.supportsTools, chatFormat: meta.chatFormat || '',\n"
"    supportsThinkingToggle: !!meta.supportsThinkingToggle,\n"
"    _last: '', _ctr: 0,\n"
"    predict: function(o){\n"
"      var id = rampart.thread.getCurrentId() + '_' + (this._ctr = (this._ctr||0)+1);\n"
"      rampart.event.trigger('sub_'+this.__uid, { id: id, req: JSON.stringify(o) });\n"
/* A tool request carries a much larger prompt than a bare chat turn, so the
   old 120s deadline was easy to exceed on a slow/CPU-only box -- and expiring
   it returned an empty string that was indistinguishable from a real empty
   reply.  Wait longer, and report an expiry the same way engine errors are
   reported rather than silently yielding ''. */
"      var r = rampart.thread.get('res_'+this.__uid+'_'+id, 600000);\n"
"      if (r === undefined || r === null)\n"
"        r = { text: '[gen err:timed out waiting for the generation engine]' };\n"
"      this._last = (r.text === undefined) ? '' : r.text;\n"
/* structured form only when the caller opted in with tools -- otherwise the
   String that predict() has always returned */
"      if (o && ((o.tools && o.tools.length) || o.reasoning))\n"
"        return { fullText: this._last, toolCalls: r.toolCalls,\n"
"                 reasoning: r.reasoning, finishReason: r.finishReason };\n"
"      return this._last;\n"
"    },\n"
"    predictAsync: function(o, perTok, fin){\n"
"      var uid = this.__uid;\n"
"      var id = rampart.thread.getCurrentId() + '_' + (this._ctr = (this._ctr||0)+1);\n"
"      var tn = 'tok_'+uid+'_'+id, fn = 'fin_'+uid+'_'+id, full = '';\n"
"      rampart.event.on(tn, 'h', function(uv, t){ if (!t.r) full += t.tok;\n"
"        if (typeof perTok === 'function')\n"
"          perTok(t.r ? { token: t.tok, reasoning: true, done: false } : { token: t.tok, done: false }); });\n"
"      rampart.event.on(fn, 'h', function(uv, f){\n"
"        rampart.event.remove(tn); rampart.event.remove(fn);\n"
"        if (typeof fin === 'function') fin({ fullText: (f.full !== undefined ? f.full : full), error: f.err,\n"
"                                             toolCalls: f.toolCalls, reasoning: f.reasoning, finishReason: f.finishReason }); });\n"
"      rampart.event.trigger('sub_'+uid, { id: id, req: JSON.stringify(o), stream: true });\n"
"      return { cancel: function(){ rampart.event.trigger('can_'+uid, { id: id }); } };\n"   /* hard cancel handle */
"    },\n"
"    getLast: function(){ return this._last; },\n"
"    destroy: function(){ try { owner.terminate(); } catch(e) {} }\n"
"  }; }\n"
/* ASYNC: return at once, report percent while llama.cpp loads, hand back
   the engine when the owner publishes.  A cache hit loads nothing and
   reports no percent at all, so a caller must not read silence as a
   stall. */
"  if (typeof onprog === 'function' || typeof ondone === 'function') {\n"
"    var seen = -1;\n"
"    var poll = function(){\n"
"      var m = thread.get('ready_'+uid);\n"
"      if (m === undefined) {\n"
"        var pct = mod.__loadProgress(uid);\n"
"        if (onprog && pct >= 0 && pct !== seen) { seen = pct; onprog(pct); }\n"
"        setTimeout(poll, 250); return;\n"
"      }\n"
/* An async failure with nowhere to report it must not be silent: without
   an onDone the load error had no path out at all. */
"      if (m.err) {\n"
"        var e = fail(m.err);\n"
"        if (ondone) ondone(e, null);\n"
"        else rampart.utils.fprintf(rampart.utils.stderr,\n"
"               'initGenAsync: %s (no onDone callback was given)\\n', e.message);\n"
"        return;\n"
"      }\n"
"      if (onprog && seen !== 1) onprog(1);\n"
"      if (ondone) ondone(null, mkgen(m));\n"
"    };\n"
"    setTimeout(poll, 50);\n"
/* the caller gets a way to stop it.  llama.cpp only checks between progress
   reports, so a cancel lands at the next one, not instantly -- and a load
   already served from the model cache cannot be cancelled at all. */
"    return { cancel: function(){\n"
"      return !!mod.__cancelLoad(uid);\n"
"    } };\n"
"  }\n"
/* SYNC: no deadline.  A cold 35GB model off a slow disk outlasts any
   constant anyone would pick, and the owner always publishes a result,
   so the wait is bounded by the load itself. */
"  var meta;\n"
"  while ((meta = thread.get('ready_'+uid, 30000)) === undefined) ;\n"
"  if (meta.err) throw fail(meta.err);\n"
"  return mkgen(meta);\n"
"})\n";

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
#include <sys/utsname.h>
#include <sys/sysctl.h>
/* macOS major version (e.g. 12, 14, 15) from uname's Darwin release.
   Darwin K = macOS K-9: Darwin 21 = macOS 12, 23 = 14, 24 = 15.  Returns
   0 if it can't be determined; callers should treat 0 as "unknown,
   don't block".  install.sh's min-supported macOS is 11.0 (Darwin 20);
   anything below that is either not macOS or not a supported install
   target -- treat as unknown. */
static int rp_macos_major(void)
{
    struct utsname u;
    int darwin;
    if (uname(&u) != 0) return 0;
    darwin = atoi(u.release);
    if (darwin < 20) return 0;
    return darwin - 9;
}
/* true if running under a hypervisor (a VM).  kern.hv_vmm_present is 1 in a
   guest, 0 on bare metal.  Unknown -> 0 (assume real hardware, don't gate). */
static int rp_in_vm(void)
{
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname("kern.hv_vmm_present", &v, &sz, NULL, 0) != 0)
        return 0;
    return v != 0;
}
#endif

static duk_ret_t lg_init_gen_batched(duk_context *ctx)
{
    lt_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    REQUIRE_STRING(ctx, 0, "initGen: first argument must be a String (path to .gguf)");
    /* opts (optional object) at index 1 */

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    /* On Apple Silicon, initGen's Metal generation kernels fail to build on
       macOS < 15 ONLY in a VM: the paravirtual Metal stack can't specialize them,
       yielding a nil pipeline that crashes during decode.  On real Apple Silicon
       hardware gen works fine on macOS 11+ (verified), and macOS 15+ works even in
       a VM.  embed()/rerank() are unaffected (different Metal path).  So gate only
       the proven-bad combination: macOS < 15 AND running under a hypervisor.
       (x86_64 is not gated -- a no-Metal VM there falls back to CPU in
       build_context.)  RAMPART_FORCE_GEN=1 overrides the gate entirely. */
    if (!getenv("RAMPART_FORCE_GEN"))
    {
        int macos_ver = rp_macos_major();
        if (macos_ver > 0 && macos_ver < 15 && rp_in_vm())
            RP_THROW(ctx,
                "initGen: Apple Silicon macOS < 15 in a VM is unsupported -- the "
                "paravirtual Metal stack can't build the generation kernels "
                "(detected macOS %d in a VM).  It works on real hardware at any "
                "version, or in a VM on macOS 15+.  embed() and rerank() are "
                "unaffected.  Set RAMPART_FORCE_GEN=1 to override.",
                macos_ver);
    }
#endif

    static int bg_ctr = 0;
    int n = __atomic_add_fetch(&bg_ctr, 1, __ATOMIC_SEQ_CST);
    char uid[64];
    snprintf(uid, sizeof uid, "bg%d_%d", (int)getpid(), n);

    /* ASYNC FORM.  initGenAsync(model[, opts], onProgress[, onDone]) takes the
       same coordinator with two callbacks appended: it returns at once and the
       engine arrives at onDone.  Arguments are matched by TYPE so the opts
       object stays optional in both forms. */
    duk_idx_t optsi = -1, progi = -1, donei = -1, ai;
    for (ai = 1; ai < duk_get_top(ctx) && ai < 4; ai++) {
        if (duk_is_function(ctx, ai)) {
            if (progi < 0)      progi = ai;
            else if (donei < 0) donei = ai;
        } else if (duk_is_object(ctx, ai) && optsi < 0) optsi = ai;
    }

    duk_eval_string(ctx, BATCHGEN_SCRIPT);   /* -> wrapper function on the stack */
    duk_push_this(ctx);                       /* mod (this module object, has __rawInitGen) */
    duk_dup(ctx, 0);                          /* model */
    if (optsi > -1) duk_dup(ctx, optsi); else duk_push_object(ctx);
    duk_push_string(ctx, uid);
    if (progi > -1) duk_dup(ctx, progi); else duk_push_undefined(ctx);
    if (donei > -1) duk_dup(ctx, donei); else duk_push_undefined(ctx);
    duk_call(ctx, 6);                         /* wrapper(mod, model, opts, uid, onProgress, onDone) */
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
/* inner teardown: expects the handle object at the TOP of the stack.  The
   destroy() method wrapper pushes `this`; as a finalizer duktape passes the
   object as the argument (duk_push_this would yield undefined there). */
static duk_ret_t emb_free_(duk_context *ctx)
{
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

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

    // rerank_toks: the origin copy owns the one made at init; a copy that
    // REBUILT on its own thread (own_context, not origin) replaced its pointer
    // with a private struct at rebuild -- it owns that one.  A copy that never
    // rebuilt still points at the origin's struct and must not free it.
    if (is_origin || own_context)
    {
        if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rerank_toks")))
        {
            void *toks = duk_get_pointer(ctx, -1);
            if (toks) free(toks);
            duk_pop(ctx);
            duk_push_pointer(ctx, NULL);
            duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rerank_toks"));
        }
        else
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

static duk_ret_t emb_free(duk_context *ctx)
{
    duk_push_this(ctx);
    return emb_free_(ctx);
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

/* ============================================================
 * Structure-aware document embedding (rp-chunker.c, shared source with
 * rampart-onnx).  Text is chunked at semantic boundaries -- one vector per
 * blank-line paragraph (fragments under min_tokens merged), single-newline
 * lines greedily packed to the window, sliding token window with 1/8 overlap
 * as the no-structure / oversized-paragraph fallback -- then each chunk is
 * decoded and pooled.  Used by embedTextTo* (JS) and rp_embed_text (C ABI).
 * ============================================================ */

typedef struct { size_t start, end, n_tokens; } ll_chunk_span;
RP_DOCCACHE_ASSERT_SPAN_LAYOUT(ll_chunk_span);   /* cast to/from the doc cache */

/* rp_chunk_count_fn over llama_tokenize (probe mode; includes specials) */
static size_t ll_chunk_count(void *user, const char *t, size_t l)
{
    const struct llama_vocab *v = (const struct llama_vocab *)user;
    int need = llama_tokenize(v, t, (int)l, NULL, 0,
                              /*add_special*/true, /*parse_special*/true);
    if (need < 0) need = -need;
    return (size_t)need;
}

/* One chunk queued for the next batched decode: a slice of the flat pending
 * token buffer, plus the ROW of `vecs` its pooled vector belongs in.  The
 * destination is an index, not a pointer, because `vecs` is realloc'd as
 * chunks accumulate -- a cached pointer would dangle across a growth. */
typedef struct { int off, n; size_t row; } ll_pend;

/* Tokens that may ride in ONE llama_decode on this context.
 *   n_batch  -- hard cap; exceeding it is a GGML_ASSERT abort, not an error.
 *   n_ubatch -- must also cover the WHOLE batch.  A pooled multi-sequence
 *               decode that splits across ubatches has each sequence's pooled
 *               output overwritten by the last ubatch containing it.  For
 *               non-causal (BERT) models llama.cpp asserts this, but for
 *               CAUSAL embedders (qwen3-embedding) the assert is skipped and
 *               the result is silently wrong -- so we enforce it ourselves.
 *   n_ctx    -- bounds the unified KV cells for KV-backed (causal) embedders.
 *               BERT-style encoders allocate no KV at all, so there it simply
 *               is not the binding constraint.
 * On the embed path these are all equal (new_embed_context), so the min() is
 * belt-and-braces against an explicit nCtx/nUBatch from initEmbed opts. */
static int ll_batch_budget(struct llama_context *lctx)
{
    uint32_t b  = llama_n_batch(lctx);
    uint32_t nu = llama_n_ubatch(lctx);
    uint32_t nc = llama_n_ctx(lctx);
    if (nu && nu < b) b = nu;
    if (nc && nc < b) b = nc;
    return (int)b;
}

/* Decode n_seq INDEPENDENT sequences in ONE llama_decode and write each one's
 * L2-normalized pooled embedding to dst[seqs[j].row * vec_dim].
 *
 * Sequence j gets seq_id j and its OWN positions 0..len-1.  Positions must
 * restart per sequence: BERT-style models index a learned table of
 * n_ctx_train rows with the raw position (no clamping), and CLS/LAST pooling
 * select the min/max position within a sequence.  Every token is flagged as
 * an output because cparams.embeddings requires it (llama.cpp warns and
 * overrides otherwise).  Sequence lengths may differ freely -- with a unified
 * KV cache llama.cpp splits with split_simple, which has no equal-length
 * requirement, so ragged chunks pack with no padding.
 *
 * n_seq == 1 produces a batch byte-identical to the pre-batching single-chunk
 * path, so a one-chunk document is unaffected by this change.
 *
 * The caller guarantees sum(seqs[].n) <= ll_batch_budget(). */
static int ll_decode_pooled_batch(struct llama_context *lctx,
                                  enum llama_pooling_type p,
                                  const llama_token *toks,
                                  const ll_pend *seqs, int n_seq,
                                  int vec_dim, float *dst,
                                  char *err, size_t errlen)
{
    if (n_seq <= 0) return 0;

    int total = 0;
    for (int j = 0; j < n_seq; j++) total += seqs[j].n;
    if (total <= 0) return 0;

    llama_memory_clear(llama_get_memory(lctx), /*clear_kv=*/true);

    struct llama_batch batch = llama_batch_init(/*capacity*/total, /*embd*/0, /*n_seq_max*/1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        if (err) snprintf(err, errlen, "llama_batch_init failed");
        return -1;
    }
    int t = 0;
    for (int j = 0; j < n_seq; j++) {
        for (int i = 0; i < seqs[j].n; i++, t++) {
            batch.token[t]     = toks[seqs[j].off + i];
            batch.pos[t]       = i;                    /* per-sequence, from 0 */
            batch.n_seq_id[t]  = 1;
            batch.seq_id[t][0] = (llama_seq_id)j;
            batch.logits[t]    = 1;
        }
    }
    batch.n_tokens = total;

    /* >0 = KV slot prepare failure, <0 = hard error; both are fatal here. */
    if (llama_decode(lctx, batch) != 0) {
        llama_batch_free(batch);
        if (err) snprintf(err, errlen, "llama_decode failed (%d seqs, %d tokens)",
                          n_seq, total);
        return -1;
    }

    /* Copy EVERY sequence out before returning -- the context's per-sequence
     * embedding map is cleared at the top of the next decode. */
    int rc = 0, base = 0;
    for (int j = 0; j < n_seq; j++) {
        const float *emb = (p == LLAMA_POOLING_TYPE_NONE)
                         ? llama_get_embeddings_ith(lctx, base + seqs[j].n - 1)
                         : llama_get_embeddings_seq(lctx, (llama_seq_id)j);
        base += seqs[j].n;
        if (!emb) {
            if (err) snprintf(err, errlen, "no embedding returned for sequence %d", j);
            rc = -1;
            break;
        }
        float *out = dst + seqs[j].row * (size_t)vec_dim;
        double norm2 = 0.0;
        for (int i = 0; i < vec_dim; ++i) norm2 += (double)emb[i] * (double)emb[i];
        float inv = norm2 > 0.0 ? (float)(1.0 / sqrt(norm2)) : 1.0f;
        for (int i = 0; i < vec_dim; ++i) out[i] = emb[i] * inv;
    }
    llama_batch_free(batch);
    return rc;
}

/* Chunk + decode a whole document.  On success returns vec_dim (>0) and fills
 * (any out pointer may be NULL):
 *   *out_vecs = malloc'd float[k*vec_dim]   unit chunk vectors, row-major
 *   *out_k    = k
 *   *out_spans= malloc'd ll_chunk_span[k]   byte span of each chunk in text
 *               (+ its decoded token count; window sub-chunks share a span)
 *   *out_avg  = malloc'd float[vec_dim]     normalize(mean of unit chunk vecs)
 *   *out_coh  = avg pairwise cosine between unit chunk vecs, [0,1]; 1.0 when k==1
 * Returns 0 with err[0]==0 for empty/whitespace text; err set on real errors. */
static int ll_embed_doc(struct llama_context *lctx, const struct llama_vocab *vocab,
                        const char *text, size_t tlen,
                        const char *dpfx, size_t dpfx_len,   /* per-doc chunk prefix (title) */
                        int vec_dim,
                        int split_mode, int min_tokens, int pack_para,
                        int sentence_split,
                        int spans_only,   /* 1 = skip decode; spans/k only */
                        int max_batch,    /* max chunks per decode; 1 = one each */
                        int max_tokens,   /* soft token cap per decode; 0 = none */
                        float **out_vecs, size_t *out_k, ll_chunk_span **out_spans,
                        float **out_avg, float *out_coh,
                        char *err, size_t errlen)
{
    if (out_vecs) *out_vecs = NULL;
    if (out_k) *out_k = 0;
    if (out_spans) *out_spans = NULL;
    if (out_avg) *out_avg = NULL;
    if (out_coh) *out_coh = 0.0f;
    if (err && errlen) err[0] = '\0';
    if (!lctx || !vocab || !text || tlen == 0 || vec_dim <= 0) return 0;

    const int n_ctx    = llama_n_ctx(lctx);
    int       n_ubatch = llama_n_ubatch(lctx);
    if (n_ubatch <= 0) n_ubatch = n_ctx;
    int chunk_tokens = (n_ctx < n_ubatch ? n_ctx : n_ubatch);
    if (chunk_tokens <= 0) {
        if (err) snprintf(err, errlen, "invalid limits (n_ctx=%d ubatch=%d)", n_ctx, n_ubatch);
        return 0;
    }
    int overlap = chunk_tokens / 8;
    if (overlap >= chunk_tokens) overlap = chunk_tokens - 1;
    int stride = chunk_tokens - overlap;
    if (stride < 1) stride = chunk_tokens;

    /* Per-document prefix tokens (no specials).  Injected into each
     * decoded window below; chunk boundaries / spans / k never depend on
     * it (window budget stays chunk_tokens), so abstract()'s span
     * recomputation -- which doesn't know the prefix -- stays exact.
     * A full window loses its last dp_n tokens of decode input. */
    llama_token *dp_toks = NULL;
    int dp_n = 0;
    if (dpfx && dpfx_len && !spans_only) {
        int pneed = llama_tokenize(vocab, dpfx, (int)dpfx_len, NULL, 0, false, true);
        if (pneed <= 0) pneed = -pneed;
        if (pneed > 0) {
            dp_toks = (llama_token *)calloc((size_t)pneed, sizeof(*dp_toks));
            if (!dp_toks) { if (err) snprintf(err, errlen, "oom prefix"); return 0; }
            dp_n = llama_tokenize(vocab, dpfx, (int)dpfx_len, dp_toks, pneed, false, true);
            if (dp_n <= 0) dp_n = -dp_n;
            if (dp_n > pneed) dp_n = pneed;
            if (dp_n > chunk_tokens - 2) dp_n = chunk_tokens - 2;
        }
    }

    /* 1) structure-aware chunking (token counts include specials, matching
     *    the per-chunk add_special tokenization below) */
    rp_chunk_opts co = { chunk_tokens, min_tokens, pack_para, split_mode, sentence_split };
    rp_chunk_span *spans = NULL;
    size_t nspan = 0;
    if (rp_chunk_text(text, tlen, &co, ll_chunk_count, (void *)vocab, &spans, &nspan) != 0) {
        if (err) snprintf(err, errlen, "chunking failed");
        free(dp_toks);
        return 0;
    }
    if (!nspan) { free(spans); free(dp_toks); return 0; }   /* empty text */

    const enum llama_pooling_type p = llama_pooling_type(lctx);
    if (!spans_only)
        llama_set_embeddings(lctx, true);

    float *vecs = NULL;
    ll_chunk_span *vspans = NULL;
    size_t k = 0, cap = 0;
    int failed = 0;

    /* Pending batch.  Chunks are independent sequences, so instead of one
     * llama_decode each they accumulate here and go through the model
     * together, flushing when the token budget or the sequence cap is
     * reached.  Peak activation memory is unchanged: the context is already
     * sized for a single chunk of `budget` tokens (n_ubatch), and a packed
     * batch never exceeds that. */
    llama_token *pend_toks = NULL;
    ll_pend     *pend      = NULL;
    int          pend_ntok = 0, pend_n = 0, budget = 0, max_seqs = 0, soft_cap = 0;

    if (!spans_only) {
        budget   = ll_batch_budget(lctx);
        /* Caller-resolved (ll_resolve_batch): 1 = one chunk per decode, which
         * builds a batch byte-identical to the pre-batching path. */
        max_seqs = max_batch > 0 ? max_batch : 1;
        /* Soft cap on tokens per decode.  These are no-KV encoders, so
         * attention is computed over the WHOLE packed batch as one NxN matrix
         * (cross-sequence pairs are only masked out) -- cost grows with batch
         * tokens x model width, while the per-decode overhead saved grows
         * only linearly.  Past a few hundred tokens the quadratic wins and
         * batching starts LOSING: measured on an RTX 4070 Ti, bge-m3 went
         * 1.07x at ~670 tok/decode but 0.73x at ~2300.  Never let this reject
         * a lone oversized chunk -- `budget` remains the hard capacity. */
        soft_cap = (max_tokens > 0 && max_tokens < budget) ? max_tokens : budget;
        if (budget < 1) {
            free(spans); free(dp_toks);
            if (err) snprintf(err, errlen, "invalid batch budget");
            return 0;
        }
        pend_toks = (llama_token *)malloc((size_t)budget * sizeof(*pend_toks));
        pend      = (ll_pend *)malloc((size_t)max_seqs * sizeof(*pend));
        if (!pend_toks || !pend) {
            free(pend_toks); free(pend); free(spans); free(dp_toks);
            if (err) snprintf(err, errlen, "oom pending batch");
            return 0;
        }
    }

    for (size_t si = 0; si < nspan && !failed; si++) {
        const char *ct = text + spans[si].start;
        int clen = (int)(spans[si].end - spans[si].start);

        int need = llama_tokenize(vocab, ct, clen, NULL, 0, true, true);
        if (need <= 0) need = -need;
        if (need <= 0) continue;               /* whitespace-only span */
        llama_token *toks = (llama_token *)calloc((size_t)need, sizeof(*toks));
        if (!toks) { failed = 1; if (err) snprintf(err, errlen, "oom tokens"); break; }
        int nw = llama_tokenize(vocab, ct, clen, toks, need, true, true);
        if (nw <= 0) nw = -nw;
        if (nw > need) nw = need;

        /* one decode if it fits; else the sliding-window fallback within
         * this span (oversized paragraph / unstructured text).  In
         * spans_only mode the window walk still runs (span/k parity
         * with the real embed) but no decode happens. */
        for (int start = 0; start < nw; start += stride) {
            int n = nw - start;
            if (n > chunk_tokens) n = chunk_tokens;
            if (k == cap) {
                size_t nc = cap ? cap * 2 : 8;
                float *nv = spans_only ? vecs
                    : (float *)realloc(vecs, nc * (size_t)vec_dim * sizeof(float));
                ll_chunk_span *ns = (ll_chunk_span *)realloc(vspans, nc * sizeof(*ns));
                if (nv || spans_only) vecs = nv;
                if (ns) vspans = ns;
                if ((!nv && !spans_only) || !ns) { failed = 1; if (err) snprintf(err, errlen, "oom vecs"); break; }
                cap = nc;
            }
            if (!spans_only) {
                /* Compose this window's decode input, then QUEUE it -- the
                 * decode happens in ll_decode_pooled_batch once the batch
                 * fills.  The tokens are copied into pend_toks because
                 * `toks` is freed at the end of each span iteration. */
                const llama_token *src;
                llama_token *tmp = NULL;
                int m;
                if (dp_n > 0) {
                    /* [BOS?] prefix window-tokens(trimmed) -- window walk
                     * and spans stay prefix-independent. */
                    llama_token bos = llama_vocab_bos(vocab);
                    int lead = (start == 0 && n > 0 && toks[0] == bos) ? 1 : 0;
                    int keep = n - lead;
                    if (dp_n + keep + lead > chunk_tokens)
                        keep = chunk_tokens - dp_n - lead;
                    if (keep < 0) keep = 0;
                    int q = 0;
                    m = lead + dp_n + keep;
                    tmp = (llama_token *)malloc((size_t)m * sizeof(*tmp));
                    if (!tmp) { failed = 1; if (err) snprintf(err, errlen, "oom prefix buf"); break; }
                    if (lead) tmp[q++] = toks[0];
                    memcpy(tmp + q, dp_toks, (size_t)dp_n * sizeof(*tmp)); q += dp_n;
                    memcpy(tmp + q, toks + start + lead, (size_t)keep * sizeof(*tmp));
                    src = tmp;
                } else {
                    src = toks + start;
                    m   = n;
                }
                /* Flush first if this window would overflow the batch. */
                if (pend_n > 0 && (pend_ntok + m > soft_cap || pend_n >= max_seqs)) {
                    int frc = ll_decode_pooled_batch(lctx, p, pend_toks, pend, pend_n,
                                                     vec_dim, vecs, err, errlen);
                    pend_n = 0; pend_ntok = 0;
                    if (frc != 0) { free(tmp); failed = 1; break; }
                }
                /* m <= chunk_tokens <= budget by construction; guard anyway
                 * rather than let llama.cpp's GGML_ASSERT abort the process. */
                if (m > budget) {
                    free(tmp); failed = 1;
                    if (err) snprintf(err, errlen,
                                      "chunk of %d tokens exceeds batch budget %d", m, budget);
                    break;
                }
                memcpy(pend_toks + pend_ntok, src, (size_t)m * sizeof(*pend_toks));
                pend[pend_n].off = pend_ntok;
                pend[pend_n].n   = m;
                pend[pend_n].row = k;
                pend_n++;
                pend_ntok += m;
                free(tmp);
            }
            vspans[k] = (ll_chunk_span){ spans[si].start, spans[si].end, (size_t)n };
            k++;
            if (start + chunk_tokens >= nw) break;
        }
        free(toks);
    }
    /* Trailing partial batch. */
    if (!failed && pend_n > 0 &&
        ll_decode_pooled_batch(lctx, p, pend_toks, pend, pend_n,
                               vec_dim, vecs, err, errlen) != 0)
        failed = 1;
    free(pend_toks);
    free(pend);
    free(spans);
    free(dp_toks);
    if (failed || k == 0) {
        free(vecs); free(vspans);
        if (!failed && err && errlen) err[0] = '\0';   /* nothing tokenizable */
        return 0;
    }

    /* avgVec + coherence over the unit chunk vectors.  Impossible in
     * spans_only mode -- nothing was decoded, so `vecs' is NULL. */
    if (!spans_only && (out_avg || out_coh)) {
        float *avg = (float *)malloc((size_t)vec_dim * sizeof(float));
        if (!avg) { free(vecs); free(vspans); if (err) snprintf(err, errlen, "oom avg"); return 0; }
        double coh = 1.0;
        if (k == 1) {
            memcpy(avg, vecs, (size_t)vec_dim * sizeof(float));
        } else {
            double n2 = 0.0;
            for (int d = 0; d < vec_dim; d++) {
                double m = 0.0;
                for (size_t i = 0; i < k; i++) m += (double)vecs[i * (size_t)vec_dim + d];
                avg[d] = (float)(m / (double)k);
                n2 += (double)avg[d] * (double)avg[d];
            }
            float inv = n2 > 0.0 ? (float)(1.0 / sqrt(n2)) : 1.0f;
            for (int d = 0; d < vec_dim; d++) avg[d] *= inv;
            /* coherence = AVERAGE PAIRWISE COSINE between the unit chunk vecs
             * (k-independent; raw |mean| has a 1/sqrt(k) floor).  Identity:
             * |mean|^2 = 1/k + (k-1)/k * cbar; clamp to [0,1]. */
            coh = ((double)k * n2 - 1.0) / ((double)k - 1.0);
            if (coh < 0.0) coh = 0.0;
            if (coh > 1.0) coh = 1.0;
        }
        if (out_avg) *out_avg = avg; else free(avg);
        if (out_coh) *out_coh = (float)coh;
    }
    if (out_vecs) *out_vecs = vecs; else free(vecs);
    if (out_k) *out_k = k;
    if (out_spans) *out_spans = vspans; else free(vspans);
    return vec_dim;
}

/* push one vector in the requested pack mode (plain fixed buffer for the
 * fp16/fp32 modes, matching the historical embedTextToBuf return values) */
static void ll_push_vec(duk_context *ctx, const float *v, int dim, int pack)
{
    if (pack == PACK16) {
        uint16_t *o = (uint16_t *)duk_push_fixed_buffer(ctx, (duk_size_t)dim * 2);
        rpvec_f32_to_f16(v, o, (size_t)dim);
    } else if (pack == PACK32) {
        float *o = (float *)duk_push_fixed_buffer(ctx, (duk_size_t)dim * 4);
        memcpy(o, v, (size_t)dim * sizeof(float));
    } else {
        duk_push_array(ctx);
        for (int d = 0; d < dim; d++) {
            duk_push_number(ctx, (double)v[d]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)d);
        }
    }
}

/* Resolve this embed handle (`this`): model, vec_dim, chunk opts, and the
 * per-thread llama_context (rebuilt on a new thread/pid, with a CHECKED model
 * refcount so a copy whose origin was destroyed fails cleanly).  Pushes `this`
 * and LEAVES it on the stack. */
static struct llama_context *emb_resolve(duk_context *ctx, const char *what,
                                         struct llama_model **out_model,
                                         int *out_dim, int *out_split,
                                         int *out_min, int *out_pack,
                                         int *out_sent, int *out_batch, int *out_btok)
{
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
    *out_dim = duk_get_int(ctx, -1);
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

        // Post-fork refusal: GPU runtimes don't survive fork (CPU continues).
        if (pidno != curpid && lt_gpu_in_use())
            RP_THROW(ctx, LT_FORK_REFUSAL);
        duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("cp_buf"));
        struct llama_context_params *cp_buf = duk_get_buffer_data(ctx, -1, NULL);
        duk_pop(ctx);

        // Take this copy's model refcount FIRST, and checked: if the origin copy
        // (sole ref holder) was destroyed before our first use here, the model is
        // already freed -- fail cleanly rather than building a context on freed
        // memory. (Released in emb_free when this copy's context is freed.)
        if (!lgen_model_addref_checked(lmodel))
            RP_THROW(ctx, "rampart-llama-cpp:%s - model was destroyed "
                          "(the originating handle was destroy()ed); create a new handle", what);
        lctx = llama_init_from_model(lmodel, *cp_buf);

        if (!lctx) {
            lgen_model_release(lmodel);
            RP_THROW(ctx, "rampart-llama-cpp:%s - failed to create llama context on this thread", what);
        }

        duk_push_pointer(ctx, lctx);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

        duk_push_int(ctx, curthr);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));
        duk_push_int(ctx, curpid);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_pid"));
    }

    if (!lctx)
        RP_THROW(ctx, "rampart-llama-cpp:%s - NULL llama_context", what);

    /* chunking options stored at initEmbed (split/minTokens/packParagraphs) */
    *out_split = 0; *out_min = 0; *out_pack = 0;
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_split"));
    if (duk_is_number(ctx, -1)) *out_split = duk_get_int(ctx, -1);
    duk_pop(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_min"));
    if (duk_is_number(ctx, -1)) *out_min = duk_get_int(ctx, -1);
    duk_pop(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_pack"));
    if (duk_is_number(ctx, -1)) *out_pack = duk_get_int(ctx, -1);
    duk_pop(ctx);
    *out_sent = 0;
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_sent"));
    if (duk_is_number(ctx, -1)) *out_sent = duk_get_int(ctx, -1);
    duk_pop(ctx);
    /* Resolved at initEmbed (see llamacpp_init_embed); re-resolve here as a
     * fallback so a thread-copy that predates the property still behaves. */
    *out_batch = 0;
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_batch"));
    if (duk_is_number(ctx, -1)) *out_batch = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (*out_batch < 1) *out_batch = ll_resolve_batch(lctx, g_embed_batch_chunks);
    *out_btok = g_embed_batch_tokens;
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_batch_tok"));
    if (duk_is_number(ctx, -1)) *out_btok = duk_get_int(ctx, -1);
    duk_pop(ctx);

    *out_model = lmodel;
    return lctx;
}

/* avgVec + coherence over k unit chunk vectors (mirrors onnx_embed_avg for
 * the custom-split path): avg = normalize(mean); coherence = average
 * pairwise cosine, 1.0 for k == 1.  Returns malloc'd float[dim]. */
static float *ll_avg_coh(const float *vecs, size_t k, int dim, float *out_coh)
{
    float *avg = malloc((size_t)dim * sizeof(float));
    if (!avg) return NULL;
    if (k == 1) {
        memcpy(avg, vecs, (size_t)dim * sizeof(float));
        if (out_coh) *out_coh = 1.0f;
        return avg;
    }
    double norm = 0.0;
    for (int d = 0; d < dim; d++) {
        double m = 0.0;
        for (size_t i = 0; i < k; i++) m += (double)vecs[i * (size_t)dim + d];
        m /= (double)k;
        avg[d] = (float)m;
        norm += m * m;
    }
    norm = sqrt(norm);
    if (norm > 0.0)
        for (int d = 0; d < dim; d++) avg[d] = (float)((double)avg[d] / norm);
    if (out_coh) {
        /* mean pairwise cosine == (|sum|^2 - k) / (k(k-1)) for unit vecs */
        double c = ((double)k * (double)k * norm * norm - (double)k) /
                   ((double)k * ((double)k - 1.0));
        if (c < 0.0) c = 0.0;
        if (c > 1.0) c = 1.0;
        *out_coh = (float)c;
    }
    return avg;
}

/* pack != 0 - return fp16 */
static duk_ret_t embed_text_to_(duk_context *ctx, int pack)
{
    if (duk_is_buffer_data(ctx, 0))
        duk_buffer_to_string(ctx, 0);

    REQUIRE_STRING(ctx, 0, "rampart-llama-cpp:embedTextToBuf - argument must be a String");
    duk_size_t tlen = 0;
    const char *text = duk_get_lstring(ctx, 0, &tlen);   /* NUL-safe length */

    int vec_dim = 0, split = 0, minTok = 0, packPara = 0, sentSpl = 0, maxBatch = 1, maxBTok = 0;
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = emb_resolve(ctx, "embedTextToBuf", &lmodel,
                                             &vec_dim, &split, &minTok, &packPara, &sentSpl,
                                             &maxBatch, &maxBTok);

    const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);

    /* custom splitter (initEmbed split:function): fn(text) -> [String,...];
     * ONE VECTOR PER STRING, always (mirrors chunkembed(strlst)): a string
     * that fits the model window gets its exact vector; an oversized string
     * gets its embed()-style COMBINED (average) vector over its sub-chunks.
     * chunks report {text, tokens} -- no byte spans (the splitter's text
     * needn't appear in the input verbatim). */
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("chunk_splitfn"));
    if (duk_is_function(ctx, -1)) {
        duk_remove(ctx, -2);                        /* this */
        duk_dup(ctx, 0);                            /* text */
        duk_call(ctx, 1);
        if (!duk_is_array(ctx, -1))
            RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - the split function must return an Array of Strings");
        duk_idx_t arr = duk_normalize_index(ctx, -1);
        size_t k = (size_t)duk_get_length(ctx, arr), i;
        if (k == 0) {
            duk_push_object(ctx);
            duk_push_array(ctx);
            duk_put_prop_string(ctx, -2, "vecs");
            return 1;
        }
        /* calloc: overflow-checked k*size (user split array length) */
        float  *all  = calloc(k, (size_t)vec_dim * sizeof(float));
        size_t *ntok = calloc(k, sizeof(size_t));
        if (!all || !ntok) { free(all); free(ntok); RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - oom"); }
        int dim = 0;
        for (i = 0; i < k; i++) {
            duk_get_prop_index(ctx, arr, (duk_uarridx_t)i);
            duk_size_t slen = 0;
            const char *s = duk_is_string(ctx, -1) ? duk_get_lstring(ctx, -1, &slen) : NULL;
            if (!s || !slen) {
                free(all); free(ntok);
                RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - split chunk %lu is not a non-empty String",
                         (unsigned long)i);
            }
            float *v = NULL, *avg1 = NULL;
            float coh1 = 0.0f;
            size_t kk = 0;
            ll_chunk_span *spans = NULL;
            char err2[256] = {0};
            dim = ll_embed_doc(lctx, vocab, s, (size_t)slen, NULL, 0, vec_dim,
                               RP_CHUNK_AUTO, 0, 0, 0, 0, maxBatch, maxBTok,
                               &v, &kk, &spans, &avg1, &coh1, err2, sizeof err2);
            duk_pop(ctx);                           /* the string */
            if (!dim || !kk || !avg1) {
                free(all); free(ntok); free(v); free(avg1); free(spans);
                RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - %s (chunk %lu)",
                         err2[0] ? err2 : "embed failed", (unsigned long)i);
            }
            /* gcc can't see that RP_THROW longjmps and never returns, so it
               flags the lines below as using all/ntok after the frees above
               (-Wuse-after-free, gcc-only flag — hence the guard). */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuse-after-free"
#endif
            /* the combined vector: == the exact vector when the string fits
             * one window (kk == 1), the normalized mean of its sub-chunk
             * vectors otherwise */
            memcpy(all + i * (size_t)dim, avg1, (size_t)dim * sizeof(float));
            ntok[i] = 0;
            if (spans) {
                size_t j;
                for (j = 0; j < kk; j++) ntok[i] += (size_t)spans[j].n_tokens;
            }
            free(v); free(avg1); free(spans);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
        }
        float coh = 0.0f;
        float *avg = ll_avg_coh(all, k, dim, &coh);
        if (!avg) { free(all); free(ntok); RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - oom"); }
        duk_push_object(ctx);
        duk_push_array(ctx);
        for (i = 0; i < k; i++) {
            ll_push_vec(ctx, all + i * (size_t)dim, dim, pack);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
        duk_put_prop_string(ctx, -2, "vecs");
        ll_push_vec(ctx, avg, dim, pack);
        duk_put_prop_string(ctx, -2, "avgVec");
        duk_push_number(ctx, (double)coh);
        duk_put_prop_string(ctx, -2, "coherence");
        duk_push_array(ctx);
        for (i = 0; i < k; i++) {
            duk_push_object(ctx);
            duk_get_prop_index(ctx, arr, (duk_uarridx_t)i);
            duk_put_prop_string(ctx, -2, "text");
            duk_push_number(ctx, (double)ntok[i]);
            duk_put_prop_string(ctx, -2, "tokens");
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
        duk_put_prop_string(ctx, -2, "chunks");
        free(all); free(ntok); free(avg);
        return 1;
    }
    duk_pop_2(ctx);                                 /* fn slot + this */

    /* structure-aware chunk + decode (rp-chunker; see ll_embed_doc) */
    char err[256] = {0};
    float *vecs = NULL, *avg = NULL;
    float coh = 0.0f;
    size_t k = 0;
    ll_chunk_span *spans = NULL;
    int dim = ll_embed_doc(lctx, vocab, text, (size_t)tlen, NULL, 0, vec_dim,
                           split, minTok, packPara, sentSpl, 0, maxBatch, maxBTok,
                           &vecs, &k, &spans, &avg, &coh, err, sizeof err);
    if (!dim) {
        if (err[0])
            RP_THROW(ctx, "rampart-llama-cpp:embedTextToBuf - %s", err);
        /* empty/whitespace input: same { vecs: [] } shape as before */
        duk_push_object(ctx);
        duk_push_array(ctx);
        duk_put_prop_string(ctx, -2, "vecs");
        return 1;
    }

    duk_push_object(ctx);
    duk_push_array(ctx);
    for (size_t i = 0; i < k; i++) {
        ll_push_vec(ctx, vecs + i * (size_t)dim, dim, pack);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    duk_put_prop_string(ctx, -2, "vecs");
    ll_push_vec(ctx, avg, dim, pack);
    duk_put_prop_string(ctx, -2, "avgVec");
    duk_push_number(ctx, (double)coh);
    duk_put_prop_string(ctx, -2, "coherence");
    duk_push_array(ctx);
    for (size_t i = 0; i < k; i++) {
        duk_push_object(ctx);
        duk_push_number(ctx, (double)spans[i].start);
        duk_put_prop_string(ctx, -2, "start");
        duk_push_number(ctx, (double)spans[i].end);
        duk_put_prop_string(ctx, -2, "end");
        duk_push_number(ctx, (double)spans[i].n_tokens);
        duk_put_prop_string(ctx, -2, "tokens");
        duk_push_lstring(ctx, text + spans[i].start,
                         (duk_size_t)(spans[i].end - spans[i].start));
        duk_put_prop_string(ctx, -2, "text");
        /* oversized: one of several token windows over a span that exceeded
         * the model window (sub-windows share their span) */
        if ((i > 0 && spans[i-1].start == spans[i].start
                   && spans[i-1].end   == spans[i].end)
            || (i + 1 < k && spans[i+1].start == spans[i].start
                          && spans[i+1].end   == spans[i].end)) {
            duk_push_true(ctx);
            duk_put_prop_string(ctx, -2, "oversized");
        }
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    duk_put_prop_string(ctx, -2, "chunks");
    free(vecs);
    free(avg);
    free(spans);
    return 1;
}

/* embedTextsToNumbers(texts[, isQuery]) -> [ { avgVec:[...] }, ... ]
 * Return-shape parity with rampart-onnx's embedTextsToNumbers.  Each text gets
 * the full structure-aware chunked embed's avgVec (llamacpp has no cross-text
 * batching; texts run sequentially on this thread's context).  isQuery is
 * accepted for signature parity and ignored (no prefix support here). */
static duk_ret_t embed_texts_to_numbers(duk_context *ctx)
{
    REQUIRE_ARRAY(ctx, 0, "rampart-llama-cpp:embedTextsToNumbers - argument must be an Array of Strings");

    int vec_dim = 0, split = 0, minTok = 0, packPara = 0, sentSpl = 0, maxBatch = 1, maxBTok = 0;
    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = emb_resolve(ctx, "embedTextsToNumbers", &lmodel,
                                             &vec_dim, &split, &minTok, &packPara, &sentSpl,
                                             &maxBatch, &maxBTok);
    const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);

    duk_uarridx_t n = (duk_uarridx_t)duk_get_length(ctx, 0);
    duk_push_array(ctx);   /* result */
    for (duk_uarridx_t i = 0; i < n; i++) {
        duk_get_prop_index(ctx, 0, i);
        duk_size_t tlen = 0;
        const char *text = duk_get_lstring(ctx, -1, &tlen);
        if (!text)
            RP_THROW(ctx, "rampart-llama-cpp:embedTextsToNumbers - texts[%lu] is not a String",
                     (unsigned long)i);

        char err[256] = {0};
        float *avg = NULL;
        int dim = ll_embed_doc(lctx, vocab, text, (size_t)tlen, NULL, 0, vec_dim,
                               split, minTok, packPara, sentSpl, 0, maxBatch, maxBTok,
                               NULL, NULL, NULL, &avg, NULL, err, sizeof err);
        duk_pop(ctx);   /* text */
        if (!dim && err[0])
            RP_THROW(ctx, "rampart-llama-cpp:embedTextsToNumbers - %s (text %lu)",
                     err, (unsigned long)i);

        duk_push_object(ctx);
        if (dim) {
            ll_push_vec(ctx, avg, dim, NOPACK);
            free(avg);
        } else {
            duk_push_array(ctx);   /* empty/whitespace text -> empty avgVec */
        }
        duk_put_prop_string(ctx, -2, "avgVec");
        duk_put_prop_index(ctx, -2, i);
    }
    return 1;
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
 *   rp_embed_doc(h, ...)       → per-chunk vecs + {start,end,n_tokens}
 *                                 spans + avgVec + coherence (mirrors
 *                                 rampart-onnx's rp_onnx_embed_doc)
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
    int                    pid;        /* fork guard: a child must never decode
                                        * on a context inherited from the parent */
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
    int                         init_pid;    /* pid at load: post-fork GPU use is refused */
    int                         batch_chunks; /* raw setting; ll_resolve_batch()s per use */
    int                         batch_tokens; /* soft token cap per packed decode */
    /* Doc-result cache (own mutex, not h->mtx): one model run of a text
     * feeds chunkembed()/chunkavg()/chunkcoherence()/embed(), keyed on
     * the text.  See rp-embed-cache.h. */
    rp_doccache_t            doc_cache;
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

/* Publish `h`, deduping under the lock: if another thread finished loading
 * the same path while we were building, destroy ours and share the winner.
 * Without this, threads racing the FIRST load each keep a private handle --
 * and a private doc cache -- silently defeating cross-thread sharing.
 * (llama.cpp dedups the underlying model internally, so freeing ours just
 * drops a refcount.)  Returns the canonical handle. */
static rp_embed_handle_t *rp_embed_cache_put(rp_embed_handle_t *h)
{
    pthread_mutex_lock(&rp_embed_cache_lock);
    for (rp_embed_handle_t *c = rp_embed_cache_head; c; c = c->next) {
        if (strcmp(c->path, h->path) == 0) {
            c->refcount++;
            pthread_mutex_unlock(&rp_embed_cache_lock);
            if (h->lctx)   llama_free(h->lctx);
            if (h->lmodel) llama_model_free(h->lmodel);
            rp_doccache_destroy(&h->doc_cache);
            pthread_mutex_destroy(&h->mtx);
            free(h->path);
            free(h);
            return c;
        }
    }
    h->next = rp_embed_cache_head;
    rp_embed_cache_head = h;
    pthread_mutex_unlock(&rp_embed_cache_lock);
    return h;
}

/* CUDA graphs (ggml) are cached per shape in a per-context map that only evicts
 * after 10s of non-use. Batched embedding/reranking (and continuous-batch
 * generation) feed it a stream of varying shapes, so the map fills faster than
 * it drains and VRAM climbs until OOM. Graphs only speed up single-stream
 * autoregressive generation -- which embedding and reranking never do -- so we
 * disable them for those paths. ggml reads this env once, lazily, on the first
 * graph evaluation, so it MUST be set before the first llama_decode; the embed/
 * rerank init and the sql-driven rp_embed_load all run before any decode. A pure
 * generation app calls none of these, so its CUDA graphs stay enabled. Opt out
 * (keep graphs for embed/rerank too, accepting the leak) by setting
 * RAMPART_LLAMA_CUDA_GRAPHS. Harmless on CPU/Metal builds (the var is ignored). */
static void lt_disable_cuda_graphs_for_batched(void)
{
    if (!getenv("RAMPART_LLAMA_CUDA_GRAPHS"))
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
}

void *rp_embed_load(const char *path, char *err, size_t errlen)
{
    lt_disable_cuda_graphs_for_batched();   /* sql-loaded embed path */
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
    h->init_pid = (int)getpid();
    pthread_mutex_init(&h->mtx, NULL);
    rp_doccache_init(&h->doc_cache, RP_DOCCACHE_DEFAULT_CAP);

    struct llama_model_params mp = llama_model_default_params();
#if HAVE_CUDA
    {  /* GPU build: always check; lt_gpu_kernel_supported self-gates on has_gpu_backend */
        int dev = mp.main_gpu >= 0 ? mp.main_gpu : 0;
        if (!lt_gpu_kernel_supported(dev, err, errlen)) {
            rp_doccache_destroy(&h->doc_cache);
            free(h->path);
            pthread_mutex_destroy(&h->mtx);
            free(h);
            return NULL;
        }
    }
#endif
    h->lmodel = llama_model_load_from_file(path, mp);
    if (!h->lmodel) {
        if (err && errlen)
            snprintf(err, errlen, "rp_embed_load: could not load '%s': %s",
                     path, strerror(errno));
        rp_doccache_destroy(&h->doc_cache);
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
        rp_doccache_destroy(&h->doc_cache);
        free(h->path);
        pthread_mutex_destroy(&h->mtx);
        free(h);
        return NULL;
    }

    /* Build context params mirroring new_embed_context's defaults
     * (without duktape opts — we don't expose tuning here). */
    h->cp = llama_context_default_params();
    h->cp.embeddings     = true;
    /* Honor the model's OWN declared pooling (mean/cls/last/rank from GGUF
       metadata); UNSPECIFIED lets llama.cpp resolve it. Falls back to MEAN below
       if the model declares none. Mirrors llamacpp_init_embed's default. */
    h->cp.pooling_type   = LLAMA_POOLING_TYPE_UNSPECIFIED;
    /* This path takes no options object -- rampart-sql drives it -- so the
     * only way to tune it is llamacpp.embedDefaults() before the load. */
    h->cp.n_threads       = g_embed_threads;
    h->cp.n_threads_batch = g_embed_threads_batch;
    h->batch_chunks       = g_embed_batch_chunks;
    h->batch_tokens       = g_embed_batch_tokens;
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
        rp_doccache_destroy(&h->doc_cache);
        free(h->path);
        pthread_mutex_destroy(&h->mtx);
        free(h);
        return NULL;
    }

    /* Model declares no pooling (resolves to NONE) -> fall back to MEAN so
       embeddings stay pooled (historical default). h->cp keeps the final pooling
       so per-thread context rebuilds match. */
    if (h->cp.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED &&
        llama_pooling_type(h->lctx) == LLAMA_POOLING_TYPE_NONE)
    {
        h->cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        struct llama_context *l2 = llama_init_from_model(h->lmodel, h->cp);
        if (l2) { llama_free(h->lctx); h->lctx = l2; }
    }

    h->refcount = 1;
    /* May return an equivalent handle that won a concurrent load race
     * (ours is then destroyed) -- callers must use the return value. */
    return rp_embed_cache_put(h);
}

int rp_embed_dim(void *handle)
{
    if (!handle) return 0;
    return ((rp_embed_handle_t *)handle)->vec_dim;
}

/* Resize this handle's doc-result cache (cap == 0 disables it).  Backs
 * sql.set({llamaEmbed:..., likevCache:N}). */
void rp_embed_set_cache_cap(void *handle, size_t cap)
{
    if (!handle) return;
    rp_doccache_set_cap(&((rp_embed_handle_t *)handle)->doc_cache, cap);
}

/* Cached doc-embed core shared by rp_embed_doc (chunkembed/chunkavg/
 * chunkcoherence) and rp_embed_compute_avgvec (embed()).  Checks the
 * handle's doc-result cache first; on a miss runs ll_embed_doc once
 * (forcing vecs+avg+coh+spans so a later call for any other slice of
 * the same text is served from cache), stores the full result, and
 * hands out the pieces the caller asked for.  `lock_compute` wraps the
 * model run in h->mtx (shared-ctx mode); the cache uses its own mutex,
 * so a cache hit never contends with an in-flight embed of a different
 * text.  Takes lctx explicitly (per-thread or shared, resolved by the
 * caller). */
static int ll_embed_doc_cached(rp_embed_handle_t *h, struct llama_context *lctx,
                               const char *text, size_t tlen,
                               const char *prefix, size_t plen,
                               int lock_compute,
                               float **out_vecs, size_t *out_k,
                               ll_chunk_span **out_chunks,
                               float **out_avg, float *out_coh,
                               char *err, size_t errlen)
{
    if (out_vecs) *out_vecs = NULL;
    if (out_k) *out_k = 0;
    if (out_chunks) *out_chunks = NULL;
    if (out_avg) *out_avg = NULL;
    if (out_coh) *out_coh = 0.0f;
    if (!h || !lctx || !text || tlen == 0) return 0;
    if (!prefix) plen = 0;

    {
        float *cv = NULL, *ca = NULL, cc = 0.0f;
        rp_doccache_span *cs = NULL;
        size_t ck = 0; int cd = 0;
        if (rp_doccache_get(&h->doc_cache, text, tlen, prefix, plen,
                               out_vecs ? &cv : NULL, &ck, &cd,
                               out_avg  ? &ca : NULL, &cc,
                               out_chunks ? &cs : NULL)) {
            if (out_vecs)   *out_vecs   = cv;
            if (out_k)      *out_k      = ck;
            if (out_avg)    *out_avg    = ca;
            if (out_coh)    *out_coh    = cc;
            if (out_chunks) *out_chunks = (ll_chunk_span *)cs;
            return cd;
        }
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(h->lmodel);
    /* Always compute the full result (vecs+avg+coh+spans) so the stored
     * entry can serve any of the scalars later, regardless of which one
     * triggered this run.  The spans are a free byproduct of the chunk
     * walk. */
    float *vecs = NULL, *avg = NULL, coh = 0.0f;
    ll_chunk_span *chunks = NULL;
    size_t k = 0;

    if (lock_compute) pthread_mutex_lock(&h->mtx);
    int dim = ll_embed_doc(lctx, vocab, text, tlen, prefix, plen, h->vec_dim,
                           RP_CHUNK_AUTO, 0, 0, 0, 0, ll_resolve_batch(lctx, h->batch_chunks), h->batch_tokens,
                           &vecs, &k, &chunks,
                           &avg, &coh,
                           err, errlen);
    if (lock_compute) pthread_mutex_unlock(&h->mtx);
    if (!dim) { free(vecs); free(avg); free(chunks); return 0; }

    rp_doccache_put(&h->doc_cache, text, tlen, prefix, plen, vecs, k, dim,
                    avg, coh, (const rp_doccache_span *)chunks);

    if (out_vecs)   *out_vecs   = vecs;   else free(vecs);
    if (out_k)      *out_k      = k;
    if (out_avg)    *out_avg    = avg;    else free(avg);
    if (out_coh)    *out_coh    = coh;
    if (out_chunks) *out_chunks = chunks; else free(chunks);
    return dim;
}

/* avgVec compute path (embed()): the L2-normalized mean over a text's
 * chunk vectors, via the cached core. */
static size_t rp_embed_compute_avgvec(rp_embed_handle_t *h,
                                      struct llama_context *lctx,
                                      const char *text, size_t tlen,
                                      float **out_vec,
                                      char *err, size_t errlen)
{
    *out_vec = NULL;
    if (!h || !lctx || !text || tlen == 0) return 0;
    /* lock_compute = 0: callers hold the appropriate ctx discipline
     * (per-thread lctx = no lock; shared = serialized path passes its
     * own lctx and we run unlocked here because rp_embed_text already
     * holds h->mtx in that mode -- see below). */
    float *avg = NULL;
    int dim = ll_embed_doc_cached(h, lctx, text, tlen, NULL, 0, /*lock_compute*/0,
                                  NULL, NULL, NULL, &avg, NULL, err, errlen);
    if (!dim) return 0;
    *out_vec = avg;
    return (size_t)dim;
}

/* Find or create the calling thread's llama_context (shared lmodel
 * comes from h).  Returns NULL on failure.  Mutex briefly during
 * map lookup and lazy llama_init_from_model. */
static struct llama_context *
get_per_thread_ctx(rp_embed_handle_t *h)
{
    int thrno = (int)get_thread_num();
    int pid   = (int)getpid();
    struct llama_context *lctx = NULL;

    /* Post-fork refusal (see fork policy above): a child inheriting this
     * handle may continue only on CPU. */
    if (pid != h->init_pid && lt_gpu_in_use()) {
        static int warned = 0;
        if (!warned) { warned = 1; lt_warn("%s\n", LT_FORK_REFUSAL); }
        return NULL;
    }

    pthread_mutex_lock(&h->mtx);
    for (int i = 0; i < h->n_thread_ctxs; i++) {
        if (h->thread_ctxs[i].thread_num == thrno && h->thread_ctxs[i].pid == pid) {
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
        h->thread_ctxs[h->n_thread_ctxs].pid        = pid;
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
            lt_warn("rp_embed_text: failed to allocate per-thread context\n");
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
        lt_warn("rp_embed_text: %s\n", err);
    return dim;
}

/* Chunk-level variant of rp_embed_text: also yields the per-chunk vectors,
 * the coherence signal, and each chunk's BYTE span in the input text --
 * mirrors rampart-onnx's rp_onnx_embed_doc so a host (rampart-sql) can treat
 * either embedding backend identically.  Chunking is the same rp-chunker
 * structure-aware split (auto mode: one vector per blank-line paragraph,
 * 32-token fragment floor; window+overlap fallback).
 * On success returns dim (>0) and fills (any out pointer may be NULL):
 *   *out_vecs      = malloc'd float[k*dim]  unit chunk vectors, row-major
 *   *out_k         = k
 *   *out_avg       = malloc'd float[dim]    normalize(mean of unit chunk vecs)
 *   *out_coherence = avg pairwise cosine between the unit chunk vectors,
 *                    clamped to [0,1] (k-independent); 1.0 when k==1
 *   *out_chunks    = malloc'd rp_embed_chunk_span[k]: {start,end,n_tokens};
 *                    window sub-chunks of an unstructured region share its span
 * Returns 0 on failure or empty/untokenizable text. */
typedef struct { size_t start, end, n_tokens; } rp_embed_chunk_span;
/* Cast to/from ll_chunk_span AND rp_doccache_span; rampart-sql.c keeps a
 * matching copy (rp_embed_span_t) that only the iface marker below guards. */
RP_DOCCACHE_ASSERT_SPAN_LAYOUT(rp_embed_chunk_span);

/* Interface marker: rampart-sql requires this symbol so a stale module
 * (older rp_embed_doc signature, no per-doc prefix) fails loudly at
 * sql.set instead of crashing on a signature mismatch.  Bump the name
 * (v3 -> v4...) whenever an exported embed function signature changes. */
int rp_embed_iface_v3(void) { return 3; }

size_t rp_embed_doc(void *handle, const char *text, size_t tlen,
                    const char *prefix, size_t plen,
                    float **out_vecs, size_t *out_k,
                    float **out_avg, float *out_coherence,
                    rp_embed_chunk_span **out_chunks)
{
    if (out_vecs) *out_vecs = NULL;
    if (out_k) *out_k = 0;
    if (out_avg) *out_avg = NULL;
    if (out_coherence) *out_coherence = 0.0f;
    if (out_chunks) *out_chunks = NULL;
    if (!handle) return 0;
    rp_embed_handle_t *h = (rp_embed_handle_t *)handle;
    char err[256] = {0};
    int dim = 0;

    /* rp_embed_chunk_span is layout-identical to the internal ll_chunk_span.
     * Per-thread mode: each thread owns its lctx -> no compute lock.
     * Shared-ctx mode: one lctx -> the cached core locks h->mtx around the
     * model run (lock_compute=1). */
    if (g_embed_per_thread) {
        struct llama_context *lctx = get_per_thread_ctx(h);
        if (!lctx) {
            lt_warn("rp_embed_doc: failed to allocate per-thread context\n");
            return 0;
        }
        dim = ll_embed_doc_cached(h, lctx, text, tlen, prefix, plen, /*lock_compute*/0,
                                  out_vecs, out_k, (ll_chunk_span **)out_chunks,
                                  out_avg, out_coherence, err, sizeof err);
    } else {
        dim = ll_embed_doc_cached(h, h->lctx, text, tlen, prefix, plen, /*lock_compute*/1,
                                  out_vecs, out_k, (ll_chunk_span **)out_chunks,
                                  out_avg, out_coherence, err, sizeof err);
    }
    if (dim == 0 && err[0])
        lt_warn("rp_embed_doc: %s\n", err);
    return (size_t)dim;
}

/* Spans-only variant of rp_embed_doc: the byte spans + window walk the
 * doc embed would produce, WITHOUT decoding (tokenize + chunk only).
 * Deterministic w.r.t. the handle's chunking params, so spans line up
 * 1:1 with a chunkembed() of the same text.  Returns k (>= 1) and sets
 * *out_spans (malloc'd, caller frees); 0 on failure. */
size_t rp_embed_spans(void *handle, const char *text, size_t tlen,
                      rp_embed_chunk_span **out_spans)
{
    if (out_spans) *out_spans = NULL;
    if (!handle || !text || tlen == 0 || !out_spans) return 0;
    rp_embed_handle_t *h = (rp_embed_handle_t *)handle;
    const struct llama_vocab *vocab = llama_model_get_vocab(h->lmodel);
    char err[256] = {0};
    size_t k = 0;
    int dim;

    /* Still needs an lctx for the token-window size (n_ctx/n_ubatch);
     * no decode happens. */
    if (g_embed_per_thread) {
        struct llama_context *lctx = get_per_thread_ctx(h);
        if (!lctx) return 0;
        dim = ll_embed_doc(lctx, vocab, text, tlen, NULL, 0, h->vec_dim,
                           RP_CHUNK_AUTO, 0, 0, 0, 1 /* spans_only */, 1, 0,
                           NULL, &k, (ll_chunk_span **)out_spans,
                           NULL, NULL, err, sizeof err);
    } else {
        pthread_mutex_lock(&h->mtx);
        dim = ll_embed_doc(h->lctx, vocab, text, tlen, NULL, 0, h->vec_dim,
                           RP_CHUNK_AUTO, 0, 0, 0, 1 /* spans_only */, 1, 0,
                           NULL, &k, (ll_chunk_span **)out_spans,
                           NULL, NULL, err, sizeof err);
        pthread_mutex_unlock(&h->mtx);
    }
    if (!dim) {
        if (err[0]) lt_warn("rp_embed_spans: %s\n", err);
        return 0;
    }
    return k;
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

/* Read an integer GGUF metadata value "<arch>.<suffix>" from a (vocab_only)
 * model. Returns dflt if the key is absent. */
static long lt_meta_long(struct llama_model *m, const char *arch,
                         const char *suffix, long dflt)
{
    char key[192], val[64] = {0};
    if (!arch || !arch[0]) return dflt;
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    if (llama_model_meta_val_str(m, key, val, sizeof val) >= 0)
        return atol(val);
    return dflt;
}

/* modelInfo(path) -- read a model's metadata WITHOUT loading its weights.
 * Uses a vocab_only load (vocab + metadata only: no tensor data, no GPU upload)
 * and reads the dimensions from the GGUF metadata keys (under vocab_only the
 * hparams accessors like llama_model_n_embd() aren't populated, but the metadata
 * KV are -- and they are exactly what llama.cpp itself reads to fill those
 * hparams). Returns:
 *   { embedDim, hiddenDim, nCtxTrain, nLayer, arch, pooling, nParams }
 * embedDim prefers "<arch>.embedding_length_out" (a projection head's output
 * size) and falls back to "<arch>.embedding_length" -- it is the size of the
 * vector embed()/embedText* produces. pooling is the model's declared pooling
 * type, or "unspecified" if the GGUF doesn't set one. */
static duk_ret_t llamacpp_model_info(duk_context *ctx)
{
    lt_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    const char *path = REQUIRE_STRING(ctx, 0,
        "modelInfo: argument 1 must be a String (path to .gguf)");

    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only   = true;   /* metadata + vocab only -- no weights, no GPU upload */
    mp.n_gpu_layers = 0;      /* belt-and-suspenders: never touch the GPU here */

    struct llama_model *m = llama_model_load_from_file(path, mp);
    if (!m)
        RP_THROW(ctx, "modelInfo: could not load '%s'", path);

    char arch[128] = {0};
    if (llama_model_meta_val_str(m, "general.architecture", arch, sizeof arch) < 0)
        arch[0] = '\0';

    /* pooling type is stored under "<arch>.pooling_type" as a uint enum. */
    const char *pooling = "unspecified";
    if (arch[0]) {
        char pkey[160], pval[32] = {0};
        snprintf(pkey, sizeof pkey, "%s.pooling_type", arch);
        if (llama_model_meta_val_str(m, pkey, pval, sizeof pval) >= 0) {
            switch (atoi(pval)) {
                case 0:  pooling = "none"; break;
                case 1:  pooling = "mean"; break;
                case 2:  pooling = "cls";  break;
                case 3:  pooling = "last"; break;
                case 4:  pooling = "rank"; break;
                default: pooling = pval;   break;
            }
        }
    }

    long     hidden_dim = lt_meta_long(m, arch, "embedding_length", 0);
    long     embed_dim  = lt_meta_long(m, arch, "embedding_length_out", hidden_dim);
    long     n_ctx_tr   = lt_meta_long(m, arch, "context_length", 0);
    long     n_layer    = lt_meta_long(m, arch, "block_count", 0);
    uint64_t nparm      = llama_model_n_params(m);

    llama_model_free(m);

    duk_push_object(ctx);
    duk_push_int(ctx, embed_dim);        duk_put_prop_string(ctx, -2, "embedDim");
    duk_push_int(ctx, hidden_dim);       duk_put_prop_string(ctx, -2, "hiddenDim");
    duk_push_int(ctx, n_ctx_tr);         duk_put_prop_string(ctx, -2, "nCtxTrain");
    duk_push_int(ctx, n_layer);          duk_put_prop_string(ctx, -2, "nLayer");
    duk_push_string(ctx, arch);          duk_put_prop_string(ctx, -2, "arch");
    duk_push_string(ctx, pooling);       duk_put_prop_string(ctx, -2, "pooling");
    duk_push_number(ctx, (double)nparm); duk_put_prop_string(ctx, -2, "nParams");
    return 1;
}

static duk_ret_t llamacpp_init_embed(duk_context *ctx)
{
    lt_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    lt_disable_cuda_graphs_for_batched();
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
    /* Default: honor the model's OWN declared pooling (mean/cls/last/rank baked
       into the GGUF metadata). UNSPECIFIED tells llama.cpp to use the model's
       pooling_type; if the model declares none we fall back to MEAN after the
       context is built (below). An explicit { pooling } option overrides this. */
    cp.pooling_type    = LLAMA_POOLING_TYPE_UNSPECIFIED;
    /* llamacpp.embedDefaults() supplies the defaults; explicit
     * { threads / threadsBatch } options below still win. */
    cp.n_threads       = g_embed_threads;
    cp.n_ctx           = 0;
    cp.n_ubatch        = 0;
    cp.n_threads_batch = g_embed_threads_batch;
    int batch_chunks   = g_embed_batch_chunks;
    int batch_tokens   = g_embed_batch_tokens;
    if (obj_idx > -1) {
        parse_common_opts(ctx, obj_idx, &mp, &cp);
        parse_embed_opts(ctx, obj_idx, &cp);
        /* batchChunks: false = one chunk per decode, true = as many as the
         * context allows, N = cap at N.  Absent = the embedDefaults() value
         * (-1 = auto: on for GPU, off for CPU). */
        if (duk_get_prop_string(ctx, obj_idx, "batchChunks")) {
            if (duk_is_boolean(ctx, -1))     batch_chunks = duk_get_boolean(ctx, -1) ? 0x7fffffff : 1;
            else if (duk_is_number(ctx, -1)) batch_chunks = duk_get_int(ctx, -1);
            else RP_THROW(ctx, "initEmbed: batchChunks must be a Boolean or a Number");
        }
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, obj_idx, "batchTokens"))
            batch_tokens = REQUIRE_INT(ctx, -1, "initEmbed: batchTokens must be an integer");
        duk_pop(ctx);
    }

#if HAVE_CUDA
    {  /* GPU build: always check; lt_gpu_kernel_supported self-gates on has_gpu_backend */
        char eb[512];
        int dev = mp.main_gpu >= 0 ? mp.main_gpu : 0;
        if (!lt_gpu_kernel_supported(dev, eb, sizeof eb))
            RP_THROW(ctx, "initEmbed: %s", eb);
    }
#endif

    // Shared, refcounted load (one llama_model per path even across thread-copies;
    // one refcount per context, released in emb_free). Fixes the cross-copy
    // double-free and shares weights across threads.
    char lerr[256] = {0};
    lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:init - Could not load ggml file '%s': %s", model, lerr[0] ? lerr : strerror(errno));

    int vec_dim = llama_model_n_embd(lmodel);

    if (vec_dim <= 0)
    {
        lgen_model_release(lmodel);
        RP_THROW(ctx, "rampart-llama-cpp:init - Internal error getting vector dimensions");
    }

    lctx = new_embed_context(ctx, lmodel, &cp);

    /* A GPU build (Metal on macOS, CUDA on Linux) grabs a device when a context
     * is created.  In a VM / headless / no-GPU host that fails -- e.g. macOS Metal
     * in a VM: "picking default device: (null) ... failed to create command queue"
     * -- so context creation returns NULL though the model loaded.  Note that
     * n_gpu_layers=0 is NOT enough: the context still selects the Metal device for
     * compute.  Transparently fall back to CPU by pinning the model's device list
     * to the CPU device, then reload + retry.  A real Intel Mac / Apple Silicon
     * with a working Metal device succeeds on the first try and never gets here. */
    if (!lctx)
    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        lgen_model_release(lmodel);
        lmodel = NULL;
        if (cpu_dev)
        {
            ggml_backend_dev_t devs[2] = { cpu_dev, NULL };
            mp.n_gpu_layers = 0;       /* distinct model-cache key from the GPU load */
            mp.devices      = devs;    /* force CPU-only: no Metal/GPU backend */
            lt_warn("rampart-llamacpp: GPU context init failed for '%s'; retrying on CPU\n", model);
            lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);
            if (lmodel)
                lctx = new_embed_context(ctx, lmodel, &cp);
        }
    }

    if (!lctx)
    {
        if (lmodel)
            lgen_model_release(lmodel);
        RP_THROW(ctx, "rampart-llama-cpp:init - Failed to init llama from model");
    }

    /* Caller passed no { pooling } (cp stayed UNSPECIFIED) and the model declares
       no pooling (resolves to NONE) -> fall back to MEAN so embeddings stay pooled
       (the historical default). Rebuild re-stashes cp_buf with the final pooling so
       per-thread context rebuilds match. Models that DO declare a pooling keep
       UNSPECIFIED here, which each thread resolves to the model's type consistently. */
    if (cp.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED &&
        llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_NONE)
    {
        cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        struct llama_context *l2 = new_embed_context(ctx, lmodel, &cp);
        if (l2) { llama_free(lctx); lctx = l2; }
    }

    duk_push_pointer(ctx, lmodel);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("model"));

    duk_dup(ctx, 0);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("model_path"));

    duk_push_pointer(ctx, lctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

    duk_push_int(ctx, vec_dim);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("vec_dim"));

    /* structure-aware chunking options (rp-chunker; mirrors rampart-onnx):
     * split:'auto'(default)|'window', minTokens (paragraph fragment
     * floor, -1 disables merging), packParagraphs (pack to the window
     * instead of one vector per paragraph) */
    {
        int split = RP_CHUNK_AUTO, minTok = 0, packPara = 0, sentSpl = 0;
        if (obj_idx > -1) {
            if (duk_get_prop_string(ctx, obj_idx, "split")) {
                if (duk_is_function(ctx, -1)) {
                    /* custom splitter: fn(text) -> [String,...]; replaces the
                     * built-in chunker (see embed_text_to_) */
                    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_splitfn"));
                    duk_push_undefined(ctx);         /* keep the pop below balanced */
                } else {
                    const char *s = duk_to_string(ctx, -1);
                    if (s && !strcmp(s, "window")) split = RP_CHUNK_WINDOW;
                }
            }
            duk_pop(ctx);
            if (duk_get_prop_string(ctx, obj_idx, "minTokens"))
                minTok = duk_get_int(ctx, -1);
            duk_pop(ctx);
            if (duk_get_prop_string(ctx, obj_idx, "packParagraphs"))
                packPara = duk_to_boolean(ctx, -1);
            duk_pop(ctx);
            if (duk_get_prop_string(ctx, obj_idx, "sentenceSplit"))
                sentSpl = duk_to_boolean(ctx, -1);
            duk_pop(ctx);
        }
        /* Resolve batchChunks ONCE, here: lt_gpu_in_use() is only meaningful
         * after a model load has registered the backends, which has happened
         * by now.  A thread-copy inherits the resolved number. */
        duk_push_int(ctx, ll_resolve_batch(lctx, batch_chunks));
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_batch"));
        duk_push_int(ctx, batch_tokens);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_batch_tok"));
        duk_push_int(ctx, split);    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_split"));
        duk_push_int(ctx, minTok);   duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_min"));
        duk_push_int(ctx, packPara); duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_pack"));
        duk_push_int(ctx, sentSpl);  duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("chunk_sent"));
    }

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

    duk_push_c_function(ctx, embed_texts_to_numbers, 2);
    duk_put_prop_string(ctx, -2, "embedTextsToNumbers");

    duk_push_c_function(ctx, emb_free, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, emb_free_, 1); /* finalizer: object arrives as the argument */
    duk_set_finalizer(ctx, -2);

    return 1;
}


typedef struct rp_rerank_toks {
    const char *bos;
    const char *sep;
    const char *eos;
    size_t len;
    int tmpl;      /* 1 = instruct-template reranker (qwen3-reranker style) */
} rp_rerank_toks;

/* Qwen3-Reranker scores relevance as a yes/no judgement over an instruct
 * chat prompt (the gguf conversion bakes the yes-vs-no logit into a RANK
 * head, but the head only discriminates when the input follows the
 * model's published template).  Without this, every pair scores a
 * constant ~0.5.  The <Instruct> line is the model card's default. */
static const char *Q3RR_PRE =
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer can only be \"yes\" "
    "or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, "
    "retrieve relevant passages that answer the query\n<Query>: ";
static const char *Q3RR_MID  = "\n<Document>: ";
static const char *Q3RR_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

static void get_rr_toks(struct llama_model *lmodel, rp_rerank_toks *toks)
{
    const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);

    if(!toks)
        return;

    /* instruct-style rerankers (qwen3 arch) use the chat template, not
     * the bert-style [bos]query[eos][sep]doc[eos] concatenation */
    {
        char arch[64] = {0};
        toks->tmpl = (llama_model_meta_val_str(lmodel, "general.architecture",
                                               arch, sizeof arch) >= 0 &&
                      strcmp(arch, "qwen3") == 0);
    }
    if (toks->tmpl) {
        toks->bos = toks->sep = toks->eos = "";
        toks->len = strlen(Q3RR_PRE) + strlen(Q3RR_MID) + strlen(Q3RR_POST);
        return;
    }

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
// (or the model's instruct template).  Returns allocated string that
// caller must free()
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

    if (toks->tmpl)
        snprintf(input, total_len, "%s%s%s%s%s", Q3RR_PRE, query, Q3RR_MID, document, Q3RR_POST);
    else
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
        return NULL;   /* caller frees its own buffers and throws */
    }

    *n_tokens = n;
    return tokens;
}

float rerank_one(duk_context *ctx, struct llama_context *lctx, struct llama_model *lmodel,
    const struct llama_vocab *vocab, rp_rerank_toks *toks, const char *query, const char *text)
{
    // Build input string
    char *input = build_rerank_input(toks, query, text);

    if (!input)
        RP_THROW(ctx, "rerank - out of memory building input");

    // Tokenize the input
    int n_tokens = 0;
    int n_ubatch = llama_n_ubatch(lctx);

    /* template mode carries ALL its special tokens in the text (and the
     * vocab may auto-add a bos that doesn't belong mid-template) */
    llama_token *tokens = tokenize_for_rerank(ctx, lctx, vocab, input, &n_tokens,
                                              toks->tmpl ? false : true, true);
    free(input);   /* freed before any throw below */

    if (!tokens)
        RP_THROW(ctx, "Failed to tokenize input for reranking");

    // Clear the KV cache using llama_memory_clear
    llama_memory_clear(llama_get_memory(lctx), true);

    // sanity: must be a rerank model
    if (llama_pooling_type(lctx) != LLAMA_POOLING_TYPE_RANK) {
        // not a reranker; this would return a full embedding vector
        free(tokens);
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

    //clamp to max batch size, preserving the closing tokens: the final
    //special (eos/sep) for bert-style rerankers, or the whole assistant
    //tail of the instruct template -- a RANK head scored without its
    //closing tokens degrades unpredictably
    if(n_tokens > n_ubatch)
    {
        int keep = toks->tmpl ? 16 : 1;
        if (keep > n_ubatch) keep = n_ubatch;
        memmove(tokens + n_ubatch - keep, tokens + n_tokens - keep,
                (size_t)keep * sizeof(llama_token));
        n_tokens = n_ubatch;
    }

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
        // Post-fork refusal: GPU runtimes don't survive fork (CPU continues).
        if (pidno != curpid && lt_gpu_in_use())
            RP_THROW(ctx, LT_FORK_REFUSAL);
        duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("cp_buf"));
        struct llama_context_params *cp_buf = duk_get_buffer_data(ctx, -1, NULL);
        duk_pop(ctx);

        // checked refcount FIRST: if the origin copy was destroyed (its ref was
        // the last), the model is freed -- fail cleanly, no use-after-free.
        if (!lgen_model_addref_checked(lmodel))
            RP_THROW(ctx, "rerank - model was destroyed (the originating handle "
                          "was destroy()ed); create a new handle");
        lctx = llama_init_from_model(lmodel, *cp_buf);

        if (!lctx) {
            lgen_model_release(lmodel);
            RP_THROW(ctx, "rerank - failed to create llama context on this thread");
        }

        // Per-copy rerank_toks: the ORIGIN copy frees its own struct in emb_free,
        // so this copy takes a private one (its strings live in the shared model,
        // which this copy's refcount above now keeps alive).
        {
            rp_rerank_toks *mytoks = NULL;
            REMALLOC(mytoks, sizeof(rp_rerank_toks));
            get_rr_toks(lmodel, mytoks);
            toks = mytoks;
            duk_push_pointer(ctx, mytoks);
            duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rerank_toks"));
        }

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

    int sigmoid = 1;
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rr_sigmoid")))
        sigmoid = duk_get_int_default(ctx, -1, 1);
    duk_pop(ctx);

    duk_pop(ctx); // pop 'this'

    // Get query and text arguments
    const char *query = REQUIRE_STRING(ctx, 0, "rerank: argument 1 (query) must be a String");
    if( duk_is_string(ctx, 1) )
    {
        double score = (double)rerank_one(ctx, lctx, lmodel, vocab, toks, query, duk_get_string(ctx, 1));
        if (sigmoid) score = 1.0 / (1.0 + exp(-score));
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

    /* score all docs (document order) first */
    double *scores = NULL;
    REMALLOC(scores, (len ? len : 1) * sizeof(double));
    for(;i<len;i++)
    {
        duk_get_prop_index(ctx, 1, i);
        text = duk_get_string(ctx, -1);   /* no throw while holding scores[] */
        if (!text) {
            duk_pop(ctx);
            free(scores);
            RP_THROW(ctx, "rerank: argument 2 (documents) must be a String or Array of Strings");
        }
        score = (double) rerank_one(ctx, lctx, lmodel, vocab, toks, query, text);
        duk_pop(ctx);
        scores[i] = sigmoid ? 1.0 / (1.0 + exp(-score)) : score;
    }

    duk_push_array(ctx); //return value
    if(scores_only)
    {
        /* scoresOnly: scores in DOCUMENT order (matches rampart-onnx) */
        for(i=0;i<len;i++)
        {
            duk_push_number(ctx, scores[i]);
            duk_put_prop_index(ctx, -2, i);
        }
        free(scores);
        return 1;
    }

    /* object form: sorted by score desc, with the original index -- matches
     * rampart-onnx's initRerank().rerank() */
    {
        duk_uarridx_t *order = NULL;
        REMALLOC(order, (len ? len : 1) * sizeof(duk_uarridx_t));
        for (i = 0; i < len; i++) order[i] = i;
        /* insertion sort (doc lists are small; stable) */
        for (i = 1; i < len; i++) {
            duk_uarridx_t oi = order[i];
            duk_uarridx_t j = i;
            while (j > 0 && scores[order[j-1]] < scores[oi]) { order[j] = order[j-1]; j--; }
            order[j] = oi;
        }
        for (i = 0; i < len; i++)
        {
            duk_push_object(ctx);
            duk_get_prop_index(ctx, 1, order[i]);
            duk_put_prop_string(ctx, -2, "document");
            duk_push_number(ctx, scores[order[i]]);
            duk_put_prop_string(ctx, -2, "score");
            duk_push_number(ctx, (double)order[i]);
            duk_put_prop_string(ctx, -2, "index");
            duk_put_prop_index(ctx, -2, i);
        }
        free(order);
    }
    free(scores);
    return 1;
}

// Initialize a reranking model
static duk_ret_t llamacpp_init_rerank(duk_context *ctx)
{
    lt_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    lt_disable_cuda_graphs_for_batched();
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

#if HAVE_CUDA
    {  /* GPU build: always check; lt_gpu_kernel_supported self-gates on has_gpu_backend */
        char eb[512];
        int dev = mp.main_gpu >= 0 ? mp.main_gpu : 0;
        if (!lt_gpu_kernel_supported(dev, eb, sizeof eb))
            RP_THROW(ctx, "initRerank: %s", eb);
    }
#endif

    // shared, refcounted load (one refcount per context; released in emb_free)
    char lerr[256] = {0};
    lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:initRerank - Could not load ggml file '%s': %s", model, lerr[0] ? lerr : strerror(errno));

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

    /* GPU context init can fail on a host with no usable device (e.g. Metal in a
     * VM / headless macOS); fall back to a CPU-pinned context.  Pinning the model
     * to the CPU device is required -- n_gpu_layers=0 alone still selects Metal for
     * compute.  See the fuller note in llamacpp_init_embed. */
    if (!lctx)
    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        lgen_model_release(lmodel);
        lmodel = NULL;
        if (cpu_dev)
        {
            ggml_backend_dev_t devs[2] = { cpu_dev, NULL };
            mp.n_gpu_layers = 0;
            mp.devices      = devs;
            lt_warn("rampart-llamacpp: GPU context init failed for '%s'; retrying on CPU\n", model);
            lmodel = lgen_model_acquire(model, &mp, lerr, sizeof lerr);
            if (lmodel)
                lctx = llama_init_from_model(lmodel, cp);
        }
    }

    if (!lctx)
    {
        // the model came from the refcounted cache: release our ref, never
        // llama_model_free it directly (the cache still holds the pointer)
        if (lmodel)
            lgen_model_release(lmodel);
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

    /* sigmoid: onnx-parity default ON (1/(1+e^-x)); { sigmoid:false } for the
     * raw RANK-head score */
    {
        int sig = 1;
        if (obj_idx >= 0)
            lt_opt_bool2(ctx, obj_idx, "sigmoid", NULL, "sigmoid must be boolean", &sig);
        duk_push_int(ctx, sig);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rr_sigmoid"));
    }

    //get bos, sep, eos tokens
    rp_rerank_toks *toks = NULL;
    REMALLOC(toks, sizeof(rp_rerank_toks));

    get_rr_toks(lmodel, toks);

    duk_push_pointer(ctx, toks);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rerank_toks"));

    // Add the rerank function
    duk_push_c_function(ctx, rerank_text, 3);
    duk_put_prop_string(ctx, -2, "rerank");

    duk_push_c_function(ctx, emb_free, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, emb_free_, 1); /* finalizer: object arrives as the argument */
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

/* Most recent ERROR-level output, kept for llamacpp_on_abort() below.  A fatal
 * ggml error logs its reason at ERROR level and then aborts, taking the capture
 * buffer with it, so the abort hook needs the text somewhere it can still reach.
 * Written only here, under cap->mutex.  GGML_LOG_LEVEL_CONT continues the
 * previous line, so it joins whatever group that line belonged to; consecutive
 * ERROR lines accumulate (ggml_cuda_error emits three), and the first ERROR
 * after any other output starts a new group. */
#define LT_LAST_ERR_SZ 2048
static char   last_err[LT_LAST_ERR_SZ];
static size_t last_err_len  = 0;
static int    last_err_open = 0;

static void llamacpp_logger(enum ggml_log_level level, const char *text, void *ud)
{
    struct llog_cap *cap = (struct llog_cap *)ud;

    pthread_mutex_lock(&cap->mutex);

    size_t text_len = strlen(text);

    if (level == GGML_LOG_LEVEL_ERROR ||
        (level == GGML_LOG_LEVEL_CONT && last_err_open))
    {
        if (!last_err_open)
        {
            last_err_len = 0;
            last_err_open = 1;
        }
        if (last_err_len + text_len < sizeof(last_err))
        {
            memcpy(last_err + last_err_len, text, text_len);
            last_err_len += text_len;
            last_err[last_err_len] = '\0';
        }
    }
    else if (level != GGML_LOG_LEVEL_CONT)
        last_err_open = 0;

    // Check if adding new text would exceed maximum (cap->len > 0 implies
    // cap->buf is allocated; skip the trim on a fresh/reset buffer)
    if (cap->len && cap->len + text_len > MAX_LOG_BUFFER)
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

/* The capture buffer above, for llamacpp_on_abort(); ggml's abort callback
 * takes no user data. */
static struct llog_cap *llog_cap_for_abort = NULL;

/* llama.cpp's own words for a failed load.  The shim reports "could not
 * load model"; the reason -- nearly always a size that did not fit -- is
 * in the captured log.  Scanned from a mark taken before the load so a
 * previous failure cannot be reported as this one's. */
static size_t lg_log_mark(void)
{
    size_t at = 0;
    struct llog_cap *cap = llog_cap_for_abort;
    if (!cap) return 0;
    pthread_mutex_lock(&cap->mutex);
    at = cap->len;
    pthread_mutex_unlock(&cap->mutex);
    return at;
}

static void lg_load_reason(size_t from, char *out, size_t outlen)
{
    /* all one size: a shorter destination than `line' truncates silently,
       and gcc is right to say so */
    char why[512] = {0}, oom[512] = {0}, alloc[512] = {0}, line[512];
    const char *detail;
    struct llog_cap *cap = llog_cap_for_abort;

    if (outlen) out[0] = 0;
    if (!cap) return;

    pthread_mutex_lock(&cap->mutex);
    if (cap->buf && cap->len > from)
    {
        const char *p = cap->buf + from, *end = cap->buf + cap->len, *m;
        while (p < end)
        {
            const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
            size_t n = nl ? (size_t)(nl - p) : (size_t)(end - p);
            if (n >= sizeof line) n = sizeof line - 1;
            memcpy(line, p, n);
            line[n] = 0;
            /* keep the LAST of each: later lines sit closer to the failure */
            if ((m = strstr(line, "error loading model: ")))
                snprintf(why, sizeof why, "%s", m + strlen("error loading model: "));
            /* the out-of-memory line carries the size in MiB and the cause;
               the alloc line only repeats it in bytes, so it is the fallback */
            else if (strstr(line, "cudaMalloc failed") || strstr(line, "out of memory"))
                snprintf(oom, sizeof oom, "%s", line);
            else if (strstr(line, "failed to allocate"))
                snprintf(alloc, sizeof alloc, "%s", line);
            p = nl ? nl + 1 : end;
        }
    }
    pthread_mutex_unlock(&cap->mutex);

    detail = oom[0] ? oom : (alloc[0] ? alloc : NULL);
    /* drop a leading "some_function_name: " -- it names the internal that
       noticed, which tells a reader nothing */
    if (detail)
    {
        const char *c = strstr(detail, ": ");
        if (c && !memchr(detail, ' ', (size_t)(c - detail))) detail = c + 2;
    }

    if (why[0] && detail) snprintf(out, outlen, "%s (%s)", why, detail);
    else if (why[0])      snprintf(out, outlen, "%s", why);
    else if (detail)      snprintf(out, outlen, "%s", detail);
}


/* ggml calls this immediately before abort() -- a failed CUDA/Metal call, a
 * GGML_ASSERT.  Nothing else here writes to stderr: the ordinary log goes only
 * to the capture buffer (.getLog()).  But that buffer dies with the process,
 * and what it holds at this moment is the only statement of WHY.  A CUDA
 * failure, for one, logs the driver's reason and the file:line of the call that
 * failed at ERROR level, then aborts from a fixed line inside ggml_cuda_error
 * that reads the same for every such failure -- so without this you get a core
 * and no diagnosis.  Emit the last error group, then the abort message.
 *
 * Installing this callback replaces ggml's own fprintf of `message`, hence
 * printing it here.  We run on the dying thread, so we must not block on a
 * mutex another thread may hold: trylock, and read last_err regardless -- a
 * torn line beats a hung crash. */
static void llamacpp_on_abort(const char *message)
{
    struct llog_cap *cap = llog_cap_for_abort;
    int locked = (cap && pthread_mutex_trylock(&cap->mutex) == 0);

    fflush(stdout);

    if (last_err_len)
    {
        fputs(last_err, stderr);
        if (last_err[last_err_len - 1] != '\n')
            fputc('\n', stderr);
    }

    if (locked)
        pthread_mutex_unlock(&cap->mutex);

    fprintf(stderr, "%s\n", message ? message : "ggml abort");
    fflush(stderr);
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

    if (!caplog)
        RP_THROW(ctx, "Error getting log");

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
/* embedDefaults([{ batchChunks, threads, threadsBatch }]) -> current settings
 *
 * Process-global defaults for embedding.  They seed initEmbed()'s options (an
 * explicit option on the call still wins) and are the ONLY way to configure
 * the rp_embed_* C entry points that rampart-sql drives, which take no
 * options object.  Set them before loading models -- an already-built context
 * keeps the threads it was created with.
 *
 *   batchChunks  false | true | N  -- chunks packed into one llama_decode.
 *                Default (unset/null) = auto: on for a GPU backend, off for
 *                CPU, because batching measured 2.1x on an RTX 4070 Ti and
 *                1.03x (nothing) on a 16-core CPU, while always perturbing
 *                vectors slightly.
 *   threads      n_threads      -- per-token decode.
 *   threadsBatch n_threads_batch -- multi-token decode; the one embedding
 *                uses.  -1 lets ggml pick, and ggml picks the CONSTANT 4
 *                regardless of core count, so set this explicitly.
 *
 * Always returns the settings in effect after applying any changes. */
static duk_ret_t llamacpp_embed_defaults(duk_context *ctx)
{
    if (duk_is_object(ctx, 0)) {
        if (duk_get_prop_string(ctx, 0, "batchChunks")) {
            if (duk_is_boolean(ctx, -1))     g_embed_batch_chunks = duk_get_boolean(ctx, -1) ? 0x7fffffff : 1;
            else if (duk_is_null(ctx, -1))   g_embed_batch_chunks = -1;   /* back to auto */
            else if (duk_is_number(ctx, -1)) g_embed_batch_chunks = duk_get_int(ctx, -1);
            else RP_THROW(ctx, "embedDefaults: batchChunks must be a Boolean, a Number, or null");
        }
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 0, "batchTokens"))
            g_embed_batch_tokens = REQUIRE_INT(ctx, -1, "embedDefaults: batchTokens must be an integer");
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 0, "threads"))
            g_embed_threads = REQUIRE_INT(ctx, -1, "embedDefaults: threads must be an integer");
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 0, "threadsBatch"))
            g_embed_threads_batch = REQUIRE_INT(ctx, -1, "embedDefaults: threadsBatch must be an integer");
        duk_pop(ctx);
    } else if (!duk_is_undefined(ctx, 0) && !duk_is_null(ctx, 0)) {
        RP_THROW(ctx, "embedDefaults: argument must be an Object");
    }

    duk_push_object(ctx);
    if (g_embed_batch_chunks < 0)            duk_push_null(ctx);   /* auto */
    else if (g_embed_batch_chunks <= 1)      duk_push_false(ctx);
    else if (g_embed_batch_chunks == 0x7fffffff) duk_push_true(ctx);
    else duk_push_int(ctx, g_embed_batch_chunks);
    duk_put_prop_string(ctx, -2, "batchChunks");
    duk_push_int(ctx, g_embed_batch_tokens);
    duk_put_prop_string(ctx, -2, "batchTokens");
    duk_push_int(ctx, g_embed_threads);
    duk_put_prop_string(ctx, -2, "threads");
    duk_push_int(ctx, g_embed_threads_batch);
    duk_put_prop_string(ctx, -2, "threadsBatch");
    /* Which way `auto` will resolve on this box.  NB: ggml registers its GPU
     * backend at the first model load, so before any model exists this reads
     * false even on a GPU host.  The batchChunks decision itself is always
     * made after a load (initEmbed / the first rp_embed_doc), so it is
     * unaffected -- only this informational field can be early-false. */
    duk_push_boolean(ctx, lt_gpu_in_use());
    duk_put_prop_string(ctx, -2, "gpuInUse");
    return 1;
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

        /* fatal ggml errors log their reason to `cap` and then abort; this
         * hook flushes it to stderr so it is not lost.  See llamacpp_on_abort. */
        llog_cap_for_abort = cap;
        ggml_set_abort_callback(llamacpp_on_abort);

        duk_push_pointer(ctx, cap);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("caplog"));

        isloaded = 1;
    }

    duk_push_c_function(ctx, llamacpp_init_embed, 2);
    duk_put_prop_string(ctx, -2, "initEmbed");

    duk_push_c_function(ctx, llamacpp_model_info, 1);
    duk_put_prop_string(ctx, -2, "modelInfo");     // read dim/ctx/arch w/o loading weights

    duk_push_c_function(ctx, llamacpp_embed_defaults, 1);
    duk_put_prop_string(ctx, -2, "embedDefaults"); // batchChunks/threads/threadsBatch (also the sql path)

    duk_push_c_function(ctx, lg_init_gen, 2);
    duk_put_prop_string(ctx, -2, "__rawInitGen");   // raw per-thread slot engine (used by initGen's owner thread)

    duk_push_c_function(ctx, lg_load_progress, 1);
    duk_put_prop_string(ctx, -2, "__loadProgress"); // 0..1 while a load is in flight, -1 otherwise

    duk_push_c_function(ctx, lg_cancel_load, 1);
    duk_put_prop_string(ctx, -2, "__cancelLoad");   // stop an in-flight load

    duk_push_c_function(ctx, lg_init_gen_batched, 2);
    duk_put_prop_string(ctx, -2, "initGen");        // transparent cross-thread batching wrapper

    /* same coordinator, non-blocking: initGenAsync(model[, opts], onProgress[, onDone]) */
    duk_push_c_function(ctx, lg_init_gen_batched, 4);
    duk_put_prop_string(ctx, -2, "initGenAsync");

    duk_push_c_function(ctx, llamacpp_init_rerank, 2);
    duk_put_prop_string(ctx, -2, "initRerank");

    duk_push_c_function(ctx, getlog, 0);
    duk_put_prop_string(ctx, -2, "getLog");

    duk_push_c_function(ctx, resetlog, 0);
    duk_put_prop_string(ctx, -2, "resetLog");
    duk_push_c_function(ctx, resetlog, 0);
    duk_put_prop_string(ctx, -2, "clearLog");   /* alias: rampart-onnx naming parity */

    add_exit_func(close_llama_on_exit, NULL);

    /* Remember the module object (per-ctx, so each rampart thread keeps its own):
     * it is where a warning goes when there is no `this` -- see
     * lt_push_errmsg_target(). */
    duk_push_global_stash(ctx);
    duk_dup(ctx, -2);                       /* the module object */
    duk_put_prop_string(ctx, -2, LT_MODULE_STASH);
    duk_pop(ctx);                           /* stash */

    return 1;
}

