/* llama_gen_shim.h — C ABI over llama.cpp + libcommon for multi-session,
 * slot-based, continuous-batching text generation (llama-server style).
 *
 * Implemented in C++ (llama_gen_shim.cc) using libcommon; exposed as a pure C
 * ABI so the duktape module (rampart-llamacpp.c, C) can drive it. All duktape
 * contact stays in the C file; all llama/common contact stays in the .cc.
 *
 * Threading: an engine is single-threaded — created on, and driven from, one
 * rampart thread's libevent loop (a llama_context + Metal command queue is
 * pinned to its creating thread, exactly like the embedding path). The decode
 * runs on that thread; lgen_engine_step() is pumped via a recurring 0-delay
 * timeout, and on_piece/on_done fire there (where duktape is valid). A thread or
 * fork/pid change is handled by lgen_engine_rebind() before the next submit.
 */
#ifndef LLAMA_GEN_SHIM_H
#define LLAMA_GEN_SHIM_H

#include <stddef.h>
#include <stdint.h>
#include "llama.h"   /* lgen_engine_params carries llama_model_params/llama_context_params
                        directly, so the one shared option parser (rampart-llamacpp.c)
                        fills the real structs. Both TUs already include llama.h. */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- shared, refcounted model cache (also used by the embedding/rerank paths)
 * Load a model once and free it once even when a handle is copied across threads.
 * Invariant: ONE refcount per live llama_context that uses the model — call
 * acquire (or addref, if you already hold the pointer) when you create a context,
 * and release when you free that context. */
struct llama_model *lgen_model_acquire(const char *path, const struct llama_model_params *mp,
                                       char *errbuf, size_t errlen);
void                lgen_model_addref(struct llama_model *m);
void                lgen_model_release(struct llama_model *m);

/* ---- opaque handles ---- */
typedef struct lgen_engine  lgen_engine;   /* owns model, shared ctx, slots, templates, batch */
typedef struct lgen_session lgen_session;  /* one logical conversation == one KV sequence      */

/* ---- finish reasons (passed to lgen_on_done) ---- */
enum {
    LGEN_FINISH_STOP   = 0, /* hit a stop string / antiprompt        */
    LGEN_FINISH_EOG    = 1, /* end-of-generation token               */
    LGEN_FINISH_LENGTH = 2, /* hit maxTokens or context limit        */
    LGEN_FINISH_CANCEL = 3, /* cancelled via lgen_session_cancel      */
    LGEN_FINISH_ERROR  = 4  /* error; see err string in lgen_on_done  */
};

/* ---- callbacks (invoked synchronously inside lgen_engine_step, ON the calling
 * thread, where duktape is valid) ----
 * ud is the per-request userdata passed to lgen_engine_submit; the C side uses
 * it to trampoline into the JS piece/done callbacks. */
typedef void (*lgen_on_piece)(void *ud, const char *piece, size_t len);
typedef void (*lgen_on_done)(void *ud, int status, const char *err,
                             int finish_reason, const char *full_text, size_t full_len);

/* ---- engine creation params ----
 * The model/context options are the real llama structs, filled by the single
 * shared option parser in rampart-llamacpp.c (parse_common_opts). Only scalar/enum
 * fields are set there; pointer fields (tensor_split, kv_overrides, devices) are
 * left at their defaults so the struct is safe to memcpy/copy across rampart
 * threads (the per-thread engine rebuild path). model_path and chat_template are
 * const char* owned by the caller; lgen_engine_create copies them into the engine.
 * cparams.n_seq_max is the slot count; cparams.n_ctx == 0 => model's n_ctx_train. */
typedef struct {
    const char *model_path;
    const char *chat_template;   /* gen: custom Jinja template; NULL = model default */
    int         use_jinja;       /* gen: 1 = apply chat template via Jinja (default)  */

    struct llama_model_params   mparams;
    struct llama_context_params cparams;
} lgen_engine_params;

/* ---- per-request params ---- */
typedef struct {
    /* exactly one of prompt / messages_json is set */
    const char *prompt;          /* raw text (wrapped in default template)        */
    const char *messages_json;   /* JSON array [{role,content},...] parsed in shim */
    const char *chat_template;   /* NULL = model default                          */
    int         add_assistant;   /* bool                                          */

    /* sampling (maps onto common_params_sampling) */
    int    max_tokens;
    float  temp;
    float  top_p;
    float  min_p;
    float  typ_p;
    int    top_k;
    float  penalty_repeat;
    int    penalty_last_n;
    float  dry_multiplier;
    float  dry_base;
    int    dry_allowed_length;
    uint32_t seed;

    /* stop control */
    const char *const *stop;     /* antiprompt strings                            */
    size_t            n_stop;
    int   reset_mem;             /* clear this session's KV before the prompt      */
} lgen_request;

/* ---- engine lifecycle ----
 * lgen_engine_create loads the model AND builds the context + slots on the
 * CALLING thread (the rampart thread that will drive the engine). The model is
 * shared read-only (mmap weights); the context is pinned to this thread. Call
 * create/step/submit/free all from the SAME thread (or rebind after a change). */
lgen_engine *lgen_engine_create(const lgen_engine_params *p, char *errbuf, size_t errlen);
void         lgen_engine_free(lgen_engine *e);   /* frees context + model (call on owner thread) */

/* rebuild the context on the current thread after a thread/fork(pid) change.
 * Returns 0 on success, -1 on failure (errbuf filled). */
int          lgen_engine_rebind(lgen_engine *e, char *errbuf, size_t errlen);

/* informational */
uint32_t     lgen_engine_n_ctx(lgen_engine *e);
int32_t      lgen_engine_n_vocab(lgen_engine *e);

/* ---- driving the engine (all ON the owning thread) ----
 * submit: deep-copy + enqueue a request; returns id (>0) or 0 on error (errbuf).
 * step:   advance every active slot one batched decode; fires on_piece/on_done;
 *         returns 1 if work remains (re-arm the pump), 0 if idle.
 * has_active: 1 if any slot is busy or a request is queued. */
uint64_t     lgen_engine_submit(lgen_engine *e, const lgen_request *req,
                                lgen_on_piece on_piece, lgen_on_done on_done,
                                void *ud, char *errbuf, size_t errlen);
int          lgen_engine_step(lgen_engine *e);
int          lgen_engine_has_active(lgen_engine *e);
void         lgen_engine_cancel(lgen_engine *e, uint64_t req_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LLAMA_GEN_SHIM_H */
