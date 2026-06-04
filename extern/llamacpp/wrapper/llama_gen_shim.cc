/* llama_gen_shim.cc — multi-session, slot-based, continuous-batching text
 * generation over llama.cpp + libcommon, exposed via the C ABI in
 * llama_gen_shim.h.
 *
 * Threading model (Option C): the engine owns a DEDICATED inference pthread,
 * separate from every rampart JS thread. That thread alone creates and owns the
 * llama_context + slots and runs the continuous-batching loop (lgen_engine_submit
 * is thread-safe and just enqueues + wakes it). The slot scheduler mirrors
 * llama-server's update_slots() text path (tools/server/server-context.cpp).
 * on_piece/on_done fire ON the inference thread (no duktape there); the C caller
 * marshals results to the owning JS thread.
 *
 * Model is loaded eagerly (shared, survives fork via mmap weights); the context
 * is (re)created on the inference thread, lazily and guarded by getpid() so it
 * is rebuilt correctly after the server's daemonizing fork.
 */

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "nlohmann/json.hpp"   /* full definition (common headers only fwd-declare `json`) */

#include <string>
#include <string_view>
#include <vector>
#include <deque>
#include <set>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <pthread.h>
#include <unistd.h>

#include "llama_gen_shim.h"

/* llama.cpp's common headers already define `json` as nlohmann::ordered_json. */

#ifdef __APPLE__
/* defined in llama_gen_macos.mm — flips Cocoa/Foundation to multithreaded mode */
extern "C" void lg_cocoa_make_multithreaded(void);
#endif


/* ============================ submitted request ============================ */

struct gen_request {
    uint64_t    id = 0;
    bool        has_prompt = false, has_messages = false;
    std::string prompt;
    std::string messages_json;
    std::string chat_template;
    int         add_assistant = 1;

    int    max_tokens = 512;
    float  temp = -1, top_p = -1, min_p = -1, typ_p = -1;
    int    top_k = 0;
    float  penalty_repeat = -1; int penalty_last_n = -1;
    float  dry_multiplier = -1, dry_base = -1; int dry_allowed_length = -1;
    uint32_t seed = 0;
    std::vector<std::string> stop;

    lgen_on_piece on_piece = nullptr;
    lgen_on_done  on_done  = nullptr;
    void         *ud       = nullptr;
};

/* ============================== slot state ================================= */

enum slot_state { SLOT_IDLE = 0, SLOT_STARTED, SLOT_PROMPT, SLOT_DONE_PROMPT, SLOT_GENERATING };

struct gen_slot {
    int          id    = 0;          /* == seq_id */
    slot_state   state = SLOT_IDLE;
    uint64_t     req_id = 0;

    std::vector<llama_token> prompt_tokens;
    size_t n_prompt_done = 0;
    int    n_past        = 0;
    int    n_keep        = 0;
    int    n_ctx_slot    = 0;

    int          n_decoded  = 0;
    int          max_tokens = 0;
    llama_token  sampled    = 0;
    int          i_batch    = -1;
    common_sampler *smpl    = nullptr;

    std::string  generated;
    size_t       n_sent = 0;
    std::vector<std::string> antiprompt;
    bool         has_next_token = true;
    int          finish_reason  = LGEN_FINISH_EOG;

    lgen_on_piece on_piece = nullptr;
    lgen_on_done  on_done  = nullptr;
    void         *ud       = nullptr;

    bool is_processing() const { return state != SLOT_IDLE; }
};

/* ================================ engine =================================== */

struct lgen_engine {
    /* config + model (set at create) */
    lgen_engine_params params{};
    std::string  model_path;
    llama_model *model = nullptr;
    const llama_vocab *vocab = nullptr;
    uint32_t  n_ctx_cfg = 0;     /* configured/derived, reported to JS */
    int32_t   n_vocab   = 0;
    uint32_t  n_seq_max = 1;

    std::set<uint64_t> canceled;  /* request ids cancelled before/while running */
    uint64_t next_id = 1;

    /* context + slots, (re)built on the calling thread (build_context).
     * The engine is single-threaded: created on, and driven from, ONE rampart
     * thread's libevent loop (a llama_context + Metal command queue is pinned to
     * its creating thread). lgen_engine_step() is pumped via a 0-delay timeout on
     * that thread; callbacks fire there, where duktape is valid. */
    llama_context *ctx = nullptr;
    llama_batch    batch{};
    std::vector<gen_slot> slots;
    common_chat_templates_ptr templates;
    bool     add_bos = false;
    uint32_t n_ctx = 0;
    uint32_t n_batch = 0;
    std::deque<gen_request*> waitq;   /* pending requests waiting for a free slot */
};

/* ============================== small helpers ============================== */

static void set_err(char *errbuf, size_t errlen, const std::string &msg) {
    if (errbuf && errlen) { std::strncpy(errbuf, msg.c_str(), errlen - 1); errbuf[errlen - 1] = '\0'; }
}

static size_t validate_utf8(const std::string &text) {
    size_t len = text.size();
    if (len == 0) return 0;
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = (unsigned char) text[len - i];
        if ((c & 0xE0) == 0xC0) { if (i < 2) return len - i; }
        else if ((c & 0xF0) == 0xE0) { if (i < 3) return len - i; }
        else if ((c & 0xF8) == 0xF0) { if (i < 4) return len - i; }
    }
    return len;
}

static size_t find_stopping_strings(gen_slot &slot, const std::string &text,
                                    size_t last_token_size, bool is_full_stop) {
    size_t stop_pos = std::string::npos;
    for (const std::string &word : slot.antiprompt) {
        size_t pos;
        if (is_full_stop) {
            const size_t tmp = word.size() + last_token_size;
            const size_t from = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from);
        } else {
            pos = string_find_partial_stop(text, word);
        }
        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            if (is_full_stop) { slot.finish_reason = LGEN_FINISH_STOP; slot.has_next_token = false; }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

/* fully reset the slot BEFORE firing on_done (idempotent / re-entrancy safe) */
static void slot_finish(lgen_engine *e, gen_slot &slot, int status, const char *err) {
    if (slot.state == SLOT_IDLE) return;
    lgen_on_done on_done = slot.on_done;
    void        *ud      = slot.ud;
    int          reason  = slot.finish_reason;

    if (slot.smpl) { common_sampler_free(slot.smpl); slot.smpl = nullptr; }
    llama_memory_seq_rm(llama_get_memory(e->ctx), slot.id, -1, -1);
    slot.on_done  = nullptr;
    slot.on_piece = nullptr;
    slot.i_batch  = -1;
    slot.req_id   = 0;
    slot.state    = SLOT_IDLE;

    if (on_done) on_done(ud, status, err, reason, slot.generated.data(), slot.generated.size());
}

static bool process_token(lgen_engine *e, gen_slot &slot, llama_token id, const std::string &piece) {
    slot.sampled = id;
    slot.generated += piece;
    slot.has_next_token = true;

    const bool incomplete = validate_utf8(slot.generated) < slot.generated.size();
    if (!incomplete) {
        size_t pos = std::min(slot.n_sent, slot.generated.size());
        const std::string str_test = slot.generated.substr(pos);
        bool send_text = true;
        size_t stop_pos = find_stopping_strings(slot, str_test, piece.size(), true);
        if (stop_pos != std::string::npos) {
            slot.generated.erase(slot.generated.begin() + pos + stop_pos, slot.generated.end());
            pos = std::min(slot.n_sent, slot.generated.size());
        } else if (slot.has_next_token && !llama_vocab_is_eog(e->vocab, id)) {
            stop_pos = find_stopping_strings(slot, str_test, piece.size(), false);
            send_text = (stop_pos == std::string::npos);
        }
        if (send_text) {
            const std::string to_send = slot.generated.substr(pos);
            slot.n_sent += to_send.size();
            if (!to_send.empty() && slot.on_piece) slot.on_piece(slot.ud, to_send.data(), to_send.size());
        }
    }
    if (slot.n_decoded > 0 && slot.has_next_token &&
        slot.max_tokens > 0 && slot.n_decoded >= slot.max_tokens) {
        slot.finish_reason = LGEN_FINISH_LENGTH; slot.has_next_token = false;
    }
    if (llama_vocab_is_eog(e->vocab, id)) { slot.finish_reason = LGEN_FINISH_EOG; slot.has_next_token = false; }
    return slot.has_next_token;
}

static common_params_sampling sampling_from_req(const gen_request *req) {
    common_params_sampling sp;
    if (req->temp           >= 0) sp.temp           = req->temp;
    if (req->top_p          >= 0) sp.top_p          = req->top_p;
    if (req->min_p          >= 0) sp.min_p          = req->min_p;
    if (req->typ_p          >= 0) sp.typ_p          = req->typ_p;
    if (req->top_k          >  0) sp.top_k          = req->top_k;
    if (req->penalty_repeat >= 0) sp.penalty_repeat = req->penalty_repeat;
    if (req->penalty_last_n >= 0) sp.penalty_last_n = req->penalty_last_n;
    if (req->dry_multiplier >= 0) sp.dry_multiplier = req->dry_multiplier;
    if (req->dry_base       >= 0) sp.dry_base       = req->dry_base;
    if (req->dry_allowed_length >= 0) sp.dry_allowed_length = req->dry_allowed_length;
    sp.seed = req->seed;
    return sp;
}

static std::string build_prompt(lgen_engine *e, const gen_request *req) {
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = req->add_assistant ? true : false;
    if (req->has_messages) {
        json arr = json::parse(req->messages_json, nullptr, false);
        if (arr.is_array()) {
            for (auto &m : arr) {
                common_chat_msg msg;
                msg.role    = m.value("role", std::string("user"));
                msg.content = m.value("content", std::string(""));
                in.messages.push_back(std::move(msg));
            }
        }
    } else {
        common_chat_msg sys; sys.role = "system"; sys.content = "You are a helpful assistant";
        common_chat_msg usr; usr.role = "user";   usr.content = req->prompt;
        in.messages.push_back(std::move(sys));
        in.messages.push_back(std::move(usr));
    }
    common_chat_params p = common_chat_templates_apply(e->templates.get(), in);
    return p.prompt;
}

/* start a request on a free slot; consumes (frees) req on success or failure */
static void start_slot(lgen_engine *e, gen_slot &slot, gen_request *req) {
    std::string prompt = build_prompt(e, req);
    char errbuf[256];
    const char *err = nullptr;
    std::vector<llama_token> toks;
    if (prompt.empty()) { err = "empty prompt"; }
    else {
        toks = common_tokenize(e->vocab, prompt, true, true);
        if ((int) toks.size() >= slot.n_ctx_slot) { snprintf(errbuf, sizeof errbuf, "prompt longer than slot context (%d >= %d)", (int)toks.size(), slot.n_ctx_slot); err = errbuf; }
    }
    if (err) {
        if (req->on_done) req->on_done(req->ud, 1, err, LGEN_FINISH_ERROR, "", 0);
        delete req;
        return;
    }

    llama_memory_seq_rm(llama_get_memory(e->ctx), slot.id, -1, -1);
    slot.req_id = req->id;
    slot.prompt_tokens = std::move(toks);
    slot.n_prompt_done = 0;
    slot.n_past = 0;
    slot.n_keep = (int) slot.prompt_tokens.size();
    slot.n_decoded = 0;
    slot.max_tokens = req->max_tokens;
    slot.sampled = 0;
    slot.i_batch = -1;
    slot.generated.clear();
    slot.n_sent = 0;
    slot.has_next_token = true;
    slot.finish_reason = LGEN_FINISH_EOG;
    slot.on_piece = req->on_piece; slot.on_done = req->on_done; slot.ud = req->ud;
    slot.antiprompt = req->stop;

    common_params_sampling sp = sampling_from_req(req);
    slot.smpl = common_sampler_init(e->model, sp);
    slot.state = SLOT_STARTED;

    delete req;

    if (!slot.smpl) { slot.finish_reason = LGEN_FINISH_ERROR; slot_finish(e, slot, 1, "failed to init sampler"); }
}

static bool engine_has_active(lgen_engine *e) {
    for (auto &s : e->slots) if (s.is_processing()) return true;
    return !e->waitq.empty();
}

/* one continuous-batching decode step over all active slots (inference thread) */
static void engine_step(lgen_engine *e) {
    /* context shift */
    for (auto &slot : e->slots) {
        if (slot.state == SLOT_GENERATING && slot.n_past + 1 >= slot.n_ctx_slot) {
            int n_keep = slot.n_keep;
            if (e->add_bos) n_keep += 1;
            n_keep = std::min(slot.n_ctx_slot - 4, n_keep);
            const int n_left = slot.n_past - n_keep;
            const int n_discard = n_left > 0 ? n_left / 2 : 0;
            if (n_discard <= 0) { slot.finish_reason = LGEN_FINISH_LENGTH; slot_finish(e, slot, 0, nullptr); continue; }
            llama_memory_t mem = llama_get_memory(e->ctx);
            llama_memory_seq_rm (mem, slot.id, n_keep, n_keep + n_discard);
            llama_memory_seq_add(mem, slot.id, n_keep + n_discard, slot.n_past, -n_discard);
            slot.n_past -= n_discard;
        }
    }

    common_batch_clear(e->batch);

    for (auto &slot : e->slots) {
        if (slot.state != SLOT_GENERATING) continue;
        slot.i_batch = e->batch.n_tokens;
        common_batch_add(e->batch, slot.sampled, slot.n_past, { slot.id }, true);
        slot.n_past++;
    }

    for (auto &slot : e->slots) {
        if (slot.state != SLOT_STARTED && slot.state != SLOT_PROMPT) continue;
        if (slot.state == SLOT_STARTED) slot.state = SLOT_PROMPT;
        while (slot.n_prompt_done < slot.prompt_tokens.size() && (uint32_t) e->batch.n_tokens < e->n_batch) {
            common_batch_add(e->batch, slot.prompt_tokens[slot.n_prompt_done], slot.n_past, { slot.id }, false);
            slot.n_past++; slot.n_prompt_done++;
        }
        if (slot.n_prompt_done == slot.prompt_tokens.size()) {
            slot.state = SLOT_DONE_PROMPT;
            e->batch.logits[e->batch.n_tokens - 1] = true;
            slot.i_batch = e->batch.n_tokens - 1;
            slot.n_decoded = 0;
            common_sampler_reset(slot.smpl);
            for (llama_token t : slot.prompt_tokens) common_sampler_accept(slot.smpl, t, false);
        }
    }
    if (e->batch.n_tokens == 0) return;

    const int32_t n_batch = (int32_t) e->n_batch;
    for (int32_t i = 0; i < e->batch.n_tokens; i += n_batch) {
        const int32_t n = std::min(n_batch, e->batch.n_tokens - i);
        llama_batch view = {
            n, e->batch.token + i, nullptr, e->batch.pos + i,
            e->batch.n_seq_id + i, e->batch.seq_id + i, e->batch.logits + i,
        };
        const int ret = llama_decode(e->ctx, view);
        if (ret != 0) {
            const char *msg = ret == 1 ? "context size exceeded" : ret == -1 ? "invalid input batch" : "compute error";
            for (auto &slot : e->slots)
                if (slot.is_processing() && slot.i_batch >= i && slot.i_batch < i + n) {
                    slot.finish_reason = LGEN_FINISH_ERROR; slot_finish(e, slot, 1, msg);
                }
            continue;
        }
        for (auto &slot : e->slots) {
            if (slot.i_batch < i || slot.i_batch >= i + n) continue;
            if (slot.state == SLOT_DONE_PROMPT) slot.state = SLOT_GENERATING;
            else if (slot.state != SLOT_GENERATING) continue;

            /* cooperative cancel */
            if (e->canceled.count(slot.req_id)) {
                e->canceled.erase(slot.req_id);
                slot.finish_reason = LGEN_FINISH_CANCEL; slot_finish(e, slot, 0, nullptr); continue;
            }

            const int tok_idx = slot.i_batch - i;
            const llama_token id = common_sampler_sample(slot.smpl, e->ctx, tok_idx);
            slot.i_batch = -1;
            common_sampler_accept(slot.smpl, id, true);
            slot.n_decoded++;
            const std::string piece = common_token_to_piece(e->ctx, id, false);
            if (!process_token(e, slot, id, piece)) slot_finish(e, slot, 0, nullptr);
        }
    }
}

static void assign_waitq(lgen_engine *e) {
    for (auto &slot : e->slots) {
        if (e->waitq.empty()) break;
        if (slot.state != SLOT_IDLE) continue;
        gen_request *r = e->waitq.front(); e->waitq.pop_front();
        start_slot(e, slot, r);
    }
}

/* ---- process-global, refcounted model cache --------------------------------
 * A llama_model is read-only weights (mmap), safe to share across threads (the
 * embedding path already does this). We load each (path+load-params) once and
 * hand the same llama_model to every engine that asks; freed when the last
 * engine releases it. This means N per-thread engines cost 1x weights + Nx
 * context, instead of Nx weights. */
namespace {
struct model_cache_entry { std::string key; llama_model *model; int refcount; };
std::mutex                     g_model_cache_mtx;
std::vector<model_cache_entry> g_model_cache;
int                            g_model_cache_pid = -1; /* detect fork */

/* acquire by (path + load flags): load once, or bump refcount + return the
 * cached model. The invariant maintained by all callers is ONE refcount per
 * live llama_context that uses the model. */
llama_model *model_acquire_path(const char *path, int use_mmap, int use_mlock,
                                int check_tensors, char *err, size_t errlen) {
    std::lock_guard<std::mutex> lk(g_model_cache_mtx);
    /* after a fork the inherited GPU/model handles are invalid in the child;
     * drop the cache so this process reloads its own (don't free parent's COW). */
    int pid = (int) getpid();
    if (g_model_cache_pid != pid) { g_model_cache.clear(); g_model_cache_pid = pid; }

    char flags[32];
    snprintf(flags, sizeof flags, "|%d%d%d", use_mmap ? 1 : 0, use_mlock ? 1 : 0, check_tensors ? 1 : 0);
    std::string key = std::string(path) + flags;
    for (auto &e : g_model_cache)
        if (e.key == key) { e.refcount++; return e.model; }

    llama_model_params mp = llama_model_default_params();
    mp.use_mmap      = use_mmap ? true : false;
    mp.use_mlock     = use_mlock ? true : false;
    mp.check_tensors = check_tensors ? true : false;
    llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) { set_err(err, errlen, "could not load model"); return nullptr; }
    g_model_cache.push_back({ key, m, 1 });
    return m;
}

/* bump the refcount of an already-acquired model (caller already holds the
 * pointer, e.g. a thread building an additional context from a shared model). */
void model_addref(llama_model *m) {
    if (!m) return;
    std::lock_guard<std::mutex> lk(g_model_cache_mtx);
    for (auto &e : g_model_cache)
        if (e.model == m) { e.refcount++; return; }
    /* not cached: caller owns it directly; nothing to count (release frees it) */
}

void model_release(llama_model *m) {
    if (!m) return;
    std::lock_guard<std::mutex> lk(g_model_cache_mtx);
    for (size_t i = 0; i < g_model_cache.size(); i++) {
        if (g_model_cache[i].model == m) {
            if (--g_model_cache[i].refcount <= 0) {
                llama_model_free(m);
                g_model_cache.erase(g_model_cache.begin() + i);
            }
            return;
        }
    }
    llama_model_free(m); /* not cached (e.g. dropped on fork) — free directly */
}
} /* anonymous namespace */

/* load the full model (shared via the refcounted cache). Idempotent per engine. */
static bool load_model(lgen_engine *e, char *err, size_t errlen) {
    if (e->model) return true;
    const lgen_engine_params *p = &e->params;
    e->model = model_acquire_path(p->model_path, p->use_mmap, p->use_mlock, p->check_tensors, err, errlen);
    if (!e->model) return false;
    e->vocab   = llama_model_get_vocab(e->model);
    e->n_vocab = llama_vocab_n_tokens(e->vocab);
    return true;
}

/* free the per-thread context resources (keep the shared model) */
static void drop_context(lgen_engine *e) {
    for (auto &slot : e->slots) {
        if (slot.smpl) { common_sampler_free(slot.smpl); slot.smpl = nullptr; }
    }
    e->slots.clear();
    if (e->batch.token) { llama_batch_free(e->batch); e->batch = llama_batch{}; }
    e->templates.reset();
    if (e->ctx) { llama_free(e->ctx); e->ctx = nullptr; }
}

/* create the llama_context + slots + batch + templates ON the calling thread.
 * Must run on the rampart thread that will drive this engine (Metal pins its
 * command queue to the creating thread). Assumes the model is loaded. */
static bool build_context(lgen_engine *e, char *err, size_t errlen) {
    const lgen_engine_params *p = &e->params;
    if (!load_model(e, err, errlen)) return false;

    /* Serialize context creation + warmup across threads. On CUDA, many threads
     * making their first device calls concurrently (lazy ggml-cuda backend/device
     * init, per-context buffer/stream setup) can race and abort(). The model load
     * is already serialized by the cache; this covers the rest of first-use. The
     * lock is held only during one-time engine setup, not steady-state decoding. */
    static std::mutex g_build_mtx;
    std::lock_guard<std::mutex> build_lk(g_build_mtx);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = p->n_ctx ? p->n_ctx : 4096;
    cp.n_seq_max       = e->n_seq_max;
    cp.n_batch         = p->n_batch  ? p->n_batch  : 2048;
    cp.n_ubatch        = p->n_ubatch ? p->n_ubatch : cp.n_batch;
    cp.n_threads       = p->n_threads ? p->n_threads : 1;
    cp.n_threads_batch = p->n_threads_batch ? p->n_threads_batch : cp.n_threads;
    cp.kv_unified      = p->kv_unified ? true : false;
    if (p->flash_attn_type != 0) cp.flash_attn_type = (enum llama_flash_attn_type) p->flash_attn_type;
    cp.offload_kqv     = p->offload_kqv ? true : false;
    cp.op_offload      = p->op_offload ? true : false;
    if (p->type_k) cp.type_k = (enum ggml_type) p->type_k;
    if (p->type_v) cp.type_v = (enum ggml_type) p->type_v;
    cp.embeddings      = false;

    e->ctx = llama_init_from_model(e->model, cp);
    if (!e->ctx) { set_err(err, errlen, "failed to create llama context"); return false; }

    e->n_ctx   = llama_n_ctx(e->ctx);
    e->n_batch = llama_n_batch(e->ctx);
    e->add_bos = llama_vocab_get_add_bos(e->vocab);
    e->templates = common_chat_templates_init(e->model, "");
    e->batch = llama_batch_init(e->n_batch + e->n_seq_max, 0, 1);

    e->slots.assign(e->n_seq_max, gen_slot());
    for (uint32_t i = 0; i < e->n_seq_max; i++) {
        e->slots[i].id = (int) i;
        e->slots[i].n_ctx_slot = cp.kv_unified ? (int) e->n_ctx : (int) (e->n_ctx / e->n_seq_max);
    }

    /* Warm up: one tiny decode forces all lazy Metal/dispatch_once + threadpool
     * init to complete up front, so the first real predict has no init latency. */
    {
        llama_token bos = llama_vocab_bos(e->vocab);
        if (bos == LLAMA_TOKEN_NULL) bos = (llama_token) 0;
        common_batch_clear(e->batch);
        common_batch_add(e->batch, bos, 0, { 0 }, true);
        (void) llama_decode(e->ctx, e->batch);
        llama_memory_clear(llama_get_memory(e->ctx), true);
        common_batch_clear(e->batch);
    }
    return true;
}

/* ================================ C ABI =================================== */

extern "C" {

/* Shared model cache, exposed for the embedding/rerank paths (rampart-llamacpp.c)
 * so a model is loaded once and freed once even when a handle is copied across
 * threads. Invariant: one refcount per live llama_context that uses the model —
 * acquire (or addref) when you create a context, release when you free it. */
struct llama_model *lgen_model_acquire(const char *path, int use_mmap, int use_mlock,
                                       int check_tensors, char *errbuf, size_t errlen) {
    if (!path) { set_err(errbuf, errlen, "missing model_path"); return nullptr; }
    return model_acquire_path(path, use_mmap, use_mlock, check_tensors, errbuf, errlen);
}
void lgen_model_addref(struct llama_model *m)  { model_addref(m); }
void lgen_model_release(struct llama_model *m) { model_release(m); }

lgen_engine *lgen_engine_create(const lgen_engine_params *p, char *errbuf, size_t errlen) {
    if (!p || !p->model_path) { set_err(errbuf, errlen, "missing model_path"); return nullptr; }

    lgen_engine *e = new lgen_engine();
    e->params = *p;
    e->model_path = p->model_path;
    e->params.model_path = e->model_path.c_str();
    e->n_seq_max = p->n_seq_max ? p->n_seq_max : 1;

    /* Load the full model + build the context NOW, on the calling (rampart)
     * thread. This mirrors how embedding (initEmbed) loads on its worker thread
     * and decodes there: many rampart threads each run their own context's Metal
     * concurrently and safely. The context is pinned to this thread; a thread or
     * pid (fork) change is handled via lgen_engine_rebind() before the next use. */
    if (!build_context(e, errbuf, errlen)) { delete e; return nullptr; }
    return e;
}

void lgen_engine_free(lgen_engine *e) {
    if (!e) return;
    /* fail any still-queued requests, then free context + model (all on the
     * owning thread — free must be called from the thread that drove the engine) */
    for (auto &slot : e->slots)
        if (slot.is_processing()) { slot.finish_reason = LGEN_FINISH_CANCEL; slot_finish(e, slot, 0, nullptr); }
    for (gen_request *r : e->waitq) {
        if (r->on_done) r->on_done(r->ud, 1, "engine destroyed", LGEN_FINISH_CANCEL, "", 0);
        delete r;
    }
    e->waitq.clear();
    drop_context(e);
    if (e->model) { model_release(e->model); e->model = nullptr; }
    delete e;
}

/* rebuild the context on the current thread (after a thread or fork/pid change).
 * The caller (rampart-llamacpp.c) detects the change via get_thread_num()/getpid
 * and calls this before submitting. Returns 0 on success, -1 on failure. */
int lgen_engine_rebind(lgen_engine *e, char *errbuf, size_t errlen) {
    if (!e) { set_err(errbuf, errlen, "null engine"); return -1; }
    /* drop in-flight work bound to the old context (slots/samplers are invalid
     * on the new thread); queued requests are kept and will start on the new ctx */
    for (auto &slot : e->slots) if (slot.smpl) { common_sampler_free(slot.smpl); slot.smpl = nullptr; }
    drop_context(e);
    return build_context(e, errbuf, errlen) ? 0 : -1;
}

uint32_t lgen_engine_n_ctx(lgen_engine *e)   { return e ? (e->n_ctx ? e->n_ctx : e->n_ctx_cfg) : 0; }
int32_t  lgen_engine_n_vocab(lgen_engine *e) { return e ? e->n_vocab : 0; }

/* advance every active slot one continuous-batching decode step, ON the calling
 * thread; on_piece/on_done fire here (duktape-safe). Returns 1 if work remains
 * (re-arm the pump), 0 if idle. Pump this via a 0-delay recurring timeout. */
int lgen_engine_step(lgen_engine *e) {
    if (!e || !e->ctx) return 0;
    assign_waitq(e);
    engine_step(e);
    return engine_has_active(e) ? 1 : 0;
}

int lgen_engine_has_active(lgen_engine *e) {
    return (e && engine_has_active(e)) ? 1 : 0;
}

uint64_t lgen_engine_submit(lgen_engine *e, const lgen_request *req,
                            lgen_on_piece on_piece, lgen_on_done on_done,
                            void *ud, char *errbuf, size_t errlen) {
    if (!e || !req) { set_err(errbuf, errlen, "bad submit args"); return 0; }
    if (!e->ctx)    { set_err(errbuf, errlen, "engine has no context (rebind needed)"); return 0; }

    gen_request *r = new gen_request();
    if (req->prompt && req->prompt[0])               { r->has_prompt = true;   r->prompt = req->prompt; }
    if (req->messages_json && req->messages_json[0]) { r->has_messages = true; r->messages_json = req->messages_json; }
    if (req->chat_template)                           r->chat_template = req->chat_template;
    r->add_assistant = req->add_assistant;
    r->max_tokens = req->max_tokens; r->temp = req->temp; r->top_p = req->top_p; r->min_p = req->min_p;
    r->typ_p = req->typ_p; r->top_k = req->top_k; r->penalty_repeat = req->penalty_repeat;
    r->penalty_last_n = req->penalty_last_n; r->dry_multiplier = req->dry_multiplier; r->dry_base = req->dry_base;
    r->dry_allowed_length = req->dry_allowed_length; r->seed = req->seed;
    for (size_t i = 0; i < req->n_stop; i++) if (req->stop[i]) r->stop.emplace_back(req->stop[i]);
    r->on_piece = on_piece; r->on_done = on_done; r->ud = ud;

    uint64_t id = e->next_id++;
    r->id = id;
    e->waitq.push_back(r);
    return id;
}

void lgen_engine_cancel(lgen_engine *e, uint64_t req_id) {
    if (!e || !req_id) return;
    e->canceled.insert(req_id);
}

} /* extern "C" */
