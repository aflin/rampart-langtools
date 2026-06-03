/* llama_gen_shim.h — C ABI over llama.cpp + libcommon for multi-session,
 * slot-based, continuous-batching text generation (llama-server style).
 *
 * Implemented in C++ (llama_gen_shim.cc) using libcommon; exposed as a pure C
 * ABI so the duktape module (rampart-llamacpp.c, C) can drive it. All duktape
 * contact stays in the C file; all llama/common contact stays in the .cc.
 *
 * Threading: an engine is single-threaded — created on, and driven from, one
 * rampart thread's libevent loop (a llama_context is pinned to its creating
 * thread). lgen_engine_step() is pumped via a recurring 0-delay timeout.
 */
#ifndef LLAMA_GEN_SHIM_H
#define LLAMA_GEN_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/* ---- callbacks (invoked synchronously inside lgen_engine_step) ----
 * ud is the per-request userdata passed to lgen_session_submit; the C side uses
 * it to trampoline into the JS piece/done callbacks. */
typedef void (*lgen_on_piece)(void *ud, const char *piece, size_t len);
typedef void (*lgen_on_done)(void *ud, int status, const char *err,
                             int finish_reason, const char *full_text, size_t full_len);

/* ---- engine creation params (mirrors the subset of llama params we expose) ---- */
typedef struct {
    const char *model_path;
    const char *mmproj_path;     /* NULL = text-only (P1); vision deferred to P2 */

    uint32_t n_ctx;              /* total context across all slots (0 = auto)    */
    uint32_t n_seq_max;          /* number of slots                              */
    uint32_t n_batch;
    uint32_t n_ubatch;
    int32_t  n_threads;
    int32_t  n_threads_batch;
    int      kv_unified;         /* bool                                          */
    int      flash_attn_type;    /* -1 = auto                                     */
    int      offload_kqv;        /* bool                                          */
    int      op_offload;         /* bool                                          */

    int      type_k;             /* enum ggml_type as int                         */
    int      type_v;

    /* model params */
    int      use_mmap;
    int      use_mlock;
    int      check_tensors;
    int      vocab_only;
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

/* ---- engine lifecycle ---- */
lgen_engine *lgen_engine_create(const lgen_engine_params *p, char *errbuf, size_t errlen);
void         lgen_engine_free(lgen_engine *e);
int          lgen_engine_has_active(lgen_engine *e);                 /* 1 if any slot busy */
int          lgen_engine_rebind(lgen_engine *e, char *errbuf, size_t errlen); /* recreate ctx after thread/pid change */
int          lgen_engine_step(lgen_engine *e);                      /* advance all slots one decode; returns has_active() */

/* informational */
uint32_t     lgen_engine_n_ctx(lgen_engine *e);
int32_t      lgen_engine_n_vocab(lgen_engine *e);

/* ---- sessions / requests ---- */
lgen_session *lgen_session_create(lgen_engine *e);
void          lgen_session_free(lgen_session *s);
uint64_t      lgen_session_submit(lgen_session *s, const lgen_request *req,
                                  lgen_on_piece on_piece, lgen_on_done on_done,
                                  void *ud, char *errbuf, size_t errlen);
void          lgen_session_cancel(lgen_session *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LLAMA_GEN_SHIM_H */
