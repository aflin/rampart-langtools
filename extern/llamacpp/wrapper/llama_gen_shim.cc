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
#include "ggml-backend.h"     /* ggml_backend_dev_by_type: CPU-only context fallback */
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
#if defined(__FreeBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>   /* kern.smp.cores; see lgen_default_n_threads */
#endif

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

    /* tool calling (OpenAI-shape JSON, parsed in apply_templates) */
    std::string tools_json;
    int         tool_choice = LGEN_TOOL_CHOICE_AUTO;
    bool        parallel_tool_calls = false;
    bool        reasoning_separate = false;  /* run the parser without tools */
    int         thinking = -1;               /* tri-state; <0 = template default */

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

    /* ---- chat-parser state (tool calling / reasoning).
     * `parsing` is the switch between the two streaming disciplines: false keeps
     * the original byte-offset path (n_sent) exactly as it was, true uses
     * incremental common_chat_parse + compute_diffs.  It is only true when the
     * template produced a non-trivial format, so a plain prompt/messages request
     * is byte-for-byte unchanged from before tool calling existed. */
    bool                     parsing = false;
    common_chat_parser_params parser_params;
    common_chat_msg          prev_msg;   /* last successfully parsed partial */

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
    std::string  chat_template;     /* owned copy of params.chat_template ("" = default) */
    bool         use_jinja = true;  /* apply the chat template via Jinja                  */
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
    bool        supports_tools = false; /* probed at create; see probe_tool_support */
    std::string chat_format;            /* format name of a plain (toolless) apply  */
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

/* chat-parser adapter (defined further down, next to the rest of the
 * common_chat_* contact -- see "chat-template / tool-call adapter") */
static void stream_deltas(gen_slot &slot, std::string &content, std::string &reasoning);
static void        parse_final(gen_slot &slot, std::string &content,
                               std::string &tool_calls_json, std::string &reasoning);

/* fully reset the slot BEFORE firing on_done (idempotent / re-entrancy safe) */
static void slot_finish(lgen_engine *e, gen_slot &slot, int status, const char *err) {
    if (slot.state == SLOT_IDLE) return;
    lgen_on_done on_done = slot.on_done;
    void        *ud      = slot.ud;
    int          reason  = slot.finish_reason;

    /* Final parse BEFORE the slot is reset: strips tool-call markup out of the
     * text the caller sees and yields the structured calls.  Non-parsing
     * requests keep exactly the old behaviour (raw generated text). */
    std::string content, tool_calls_json, reasoning;
    lgen_result_extra extra{};
    const char *full     = slot.generated.data();
    size_t      full_len = slot.generated.size();
    if (slot.parsing && status == 0) {
        parse_final(slot, content, tool_calls_json, reasoning);
        full     = content.data();
        full_len = content.size();
        if (!tool_calls_json.empty()) {
            extra.tool_calls_json = tool_calls_json.c_str();
            if (reason == LGEN_FINISH_EOG || reason == LGEN_FINISH_STOP)
                reason = LGEN_FINISH_TOOL_CALLS;
        }
        if (!reasoning.empty()) {
            extra.reasoning     = reasoning.c_str();
            extra.reasoning_len = reasoning.size();
        }
    }

    if (slot.smpl) { common_sampler_free(slot.smpl); slot.smpl = nullptr; }
    llama_memory_seq_rm(llama_get_memory(e->ctx), slot.id, -1, -1);
    slot.on_done  = nullptr;
    slot.on_piece = nullptr;
    slot.i_batch  = -1;
    slot.req_id   = 0;
    slot.state    = SLOT_IDLE;

    if (on_done) on_done(ud, status, err, reason, full, full_len, &extra);
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
            if (slot.parsing && slot.on_piece) {
                /* Tool-call syntax is generated as ordinary tokens, so the raw
                 * stream would leak exactly the markup we exist to remove.
                 * Emit only the parser's content delta; tool-call and reasoning
                 * text is withheld and delivered structured at completion.
                 *
                 * Only when someone is actually streaming: this re-parses the
                 * whole generated string per token (quadratic, and the PEG
                 * parser is not cheap), which is pure waste for predict(), where
                 * the single parse in slot_finish is all that is needed. */
                std::string cdelta, rdelta;
                stream_deltas(slot, cdelta, rdelta);
                slot.n_sent = slot.generated.size();
                /* reasoning first: it precedes the answer in every format that
                 * separates them, and the flag lets a UI render them apart */
                if (!rdelta.empty()) slot.on_piece(slot.ud, rdelta.data(), rdelta.size(), 1);
                if (!cdelta.empty()) slot.on_piece(slot.ud, cdelta.data(), cdelta.size(), 0);
            } else if (slot.parsing) {
                slot.n_sent = slot.generated.size();   /* parsed once, at finish */
            } else {
                const std::string to_send = slot.generated.substr(pos);
                slot.n_sent += to_send.size();
                if (!to_send.empty() && slot.on_piece) slot.on_piece(slot.ud, to_send.data(), to_send.size(), 0);
            }
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

/* ==================== chat-template / tool-call adapter =====================
 *
 * ALL common_chat_* contact lives in this section, deliberately: it is the
 * fastest-churning API in llama.cpp, and keeping it in one block means the next
 * vendor bump has exactly one place to fix (see extern/llamacpp-vendoring.md,
 * "Re-vendoring procedure").  Do not scatter these calls back into start_slot /
 * process_token.
 *
 * Nothing here may let a C++ exception escape: every entry point is called from
 * code that is ultimately reached through the extern "C" ABI.
 * ========================================================================== */

/* Lenient message parsing, preserving the pre-tool-calling behaviour: role
 * defaults to "user", content to "", unknown fields ignored.  Used as a fallback
 * when the strict oaicompat parser rejects input that used to be accepted. */
static void msgs_parse_lenient(const json &arr, std::vector<common_chat_msg> &out) {
    if (!arr.is_array()) return;
    for (auto &m : arr) {
        common_chat_msg msg;
        msg.role    = m.value("role", std::string("user"));
        msg.content = m.value("content", std::string(""));
        out.push_back(std::move(msg));
    }
}

struct applied_prompt {
    bool                ok = false;
    std::string         err;
    std::string         prompt;
    common_chat_params  params;
    bool                parsing = false;  /* run the chat parser over the output */
};

/* Apply the chat template, keeping the WHOLE common_chat_params (grammar, stops,
 * parser, format) rather than just the prompt string. */
static applied_prompt apply_templates(lgen_engine *e, const gen_request *req) {
    applied_prompt out;
    try {
        common_chat_templates_inputs in;
        in.use_jinja             = e->use_jinja;
        in.add_generation_prompt = req->add_assistant ? true : false;
        /* tri-state: leave the template's own default alone unless asked */
        if (req->thinking >= 0) in.enable_thinking = req->thinking != 0;

        if (req->has_messages) {
            json arr = json::parse(req->messages_json, nullptr, false);
            if (arr.is_discarded()) { out.err = "messages is not valid JSON"; return out; }
            /* Prefer the upstream OpenAI-shape parser: it round-trips tool_calls,
             * tool_call_id, tool_name and reasoning_content, which an agent loop
             * must feed back.  It is stricter than the old lenient loop, so fall
             * back on rejection rather than breaking existing callers. */
            try {
                in.messages = common_chat_msgs_parse_oaicompat(arr);
            } catch (const std::exception &) {
                in.messages.clear();
                msgs_parse_lenient(arr, in.messages);
            }
            if (in.messages.empty()) { out.err = "messages array is empty"; return out; }
        } else {
            common_chat_msg sys; sys.role = "system"; sys.content = "You are a helpful assistant";
            common_chat_msg usr; usr.role = "user";   usr.content = req->prompt;
            in.messages.push_back(std::move(sys));
            in.messages.push_back(std::move(usr));
        }

        const bool want_tools = !req->tools_json.empty();
        if (want_tools) {
            if (!e->use_jinja) { out.err = "tools require the Jinja chat template path (useJinja must be on)"; return out; }
            if (!e->supports_tools) { out.err = "this model's chat template does not support tools"; return out; }
            json tj = json::parse(req->tools_json, nullptr, false);
            if (tj.is_discarded()) { out.err = "tools is not valid JSON"; return out; }
            try {
                in.tools = common_chat_tools_parse_oaicompat(tj);
            } catch (const std::exception &ex) {
                out.err = std::string("tools: ") + ex.what();
                return out;
            }
            switch (req->tool_choice) {
                case LGEN_TOOL_CHOICE_REQUIRED: in.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED; break;
                case LGEN_TOOL_CHOICE_NONE:     in.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;     break;
                default:                        in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;     break;
            }
            in.parallel_tool_calls = req->parallel_tool_calls;
        }

        /* Run the chat parser for tool requests, and for anyone who asked for
         * reasoning to be separated.  The latter is not cosmetic: where a
         * format carries reasoning as a CHANNEL rather than a <think> span,
         * only llama.cpp's parser for that format knows where it ends, so a
         * caller that must machine-read the answer cannot recover it itself.
         * Still opt-in -- parsing unconditionally would move <think> blocks out
         * of fullText for every existing caller. */
        const bool want_parse = want_tools || req->reasoning_separate;
        /* AUTO is upstream's recommended value: thinking models yield
         * reasoning_content instead of inline <think> text. */
        if (want_parse) in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

        out.params = common_chat_templates_apply(e->templates.get(), in);
        out.prompt = out.params.prompt;
        out.parsing = want_parse;
        out.ok = true;
    } catch (const std::exception &ex) {
        out.ok = false;
        out.err = std::string("chat template: ") + ex.what();
    } catch (...) {
        out.ok = false;
        out.err = "chat template: unknown error";
    }
    return out;
}

/* Feed the template's constraints into the sampler + slot.  Mirrors what
 * tools/server does in server-schema.cpp; see the vendoring notes for the two
 * traps here (grammar is a struct now, preserved_tokens needs tokenizing). */
static void apply_constraints(lgen_engine *e, gen_slot &slot,
                              common_params_sampling &sp, const applied_prompt &ap) {
    const common_chat_params &p = ap.params;

    if (!p.grammar.empty()) {
        sp.grammar      = common_grammar(COMMON_GRAMMAR_TYPE_TOOL_CALLS, p.grammar);
        sp.grammar_lazy = p.grammar_lazy;
        /* same type on both sides -- copy directly (the server only round-trips
         * these through JSON because it crosses an HTTP boundary) */
        sp.grammar_triggers = p.grammar_triggers;
        /* REQUIRED, and easy to miss: the assistant generation prompt is already
         * prefilled into the prompt, and a tool-calls grammar must be advanced
         * past those tokens or it starts misaligned -- toolChoice:"required"
         * then fails to force a call and the generation prompt ("assistant\n")
         * leaks into the output.  Only honoured for output-format/tool-calls
         * grammars, which is exactly what we set above. */
        sp.generation_prompt = p.generation_prompt;
    }

    /* vector<string> -> set<llama_token>: only single-token strings are usable,
     * exactly as server-schema.cpp does it. */
    for (const std::string &tok : p.preserved_tokens) {
        std::vector<llama_token> ids = common_tokenize(e->vocab, tok, false, true);
        if (ids.size() == 1) sp.preserved_tokens.insert(ids[0]);
    }

    for (const std::string &s : p.additional_stops) slot.antiprompt.push_back(s);

    slot.parsing = ap.parsing;
    if (slot.parsing) {
        slot.parser_params = common_chat_parser_params(p);  /* copies format + generation_prompt ONLY */
        slot.parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        /* The converting ctor does NOT carry the serialized PEG parser.  Without
         * this load, every PEG-format model silently degrades to content-only
         * parsing: no error, no tool calls. */
        if (!p.parser.empty()) {
            try { slot.parser_params.parser.load(p.parser); }
            catch (const std::exception &) { slot.parsing = false; }
        }
        slot.prev_msg = common_chat_msg();
    }
}

/* Incremental stream: parse what we have so far, diff against the previous
 * parse, and hand back the new content AND reasoning text separately.  Tool-call
 * deltas are accumulated by the parser and delivered whole at completion (no
 * consumer wants those token by token), so the markup never reaches on_piece.
 *
 * Reasoning IS streamed: on a thinking model the deliberation is the longest
 * part of a turn, and withholding it means a reader watches a frozen screen and
 * -- worse -- nothing can be cancelled, because cancellation works by refusing
 * the next token and no token arrives.  Both come back empty when there is
 * nothing new to emit. */
static void stream_deltas(gen_slot &slot, std::string &content, std::string &reasoning) {
    content.clear();
    reasoning.clear();
    common_chat_msg msg;
    try {
        msg = common_chat_parse(slot.generated, /*is_partial=*/true, slot.parser_params);
    } catch (const std::exception &) {
        return;   /* mid-token garbage; wait for more */
    }
    try {
        for (const auto &d : common_chat_msg_diff::compute_diffs(slot.prev_msg, msg)) {
            content   += d.content_delta;
            reasoning += d.reasoning_content_delta;
        }
    } catch (const std::exception &) {
        content.clear(); reasoning.clear();
        return;
    }
    slot.prev_msg = std::move(msg);
}

/* Final parse.  Fills `content` with the markup-free text, `tool_calls_json`
 * with an OpenAI-shape array (empty if none) and `reasoning`. */
static void parse_final(gen_slot &slot, std::string &content,
                        std::string &tool_calls_json, std::string &reasoning) {
    content.clear(); tool_calls_json.clear(); reasoning.clear();
    common_chat_msg msg;
    try {
        msg = common_chat_parse(slot.generated, /*is_partial=*/false, slot.parser_params);
    } catch (const std::exception &) {
        content = slot.generated;   /* unparseable: hand back the raw text */
        return;
    }
    content   = msg.content;
    reasoning = msg.reasoning_content;
    if (!msg.tool_calls.empty()) {
        try {
            json arr = json::array();
            int i = 0;
            for (const auto &tc : msg.tool_calls) {
                /* common_chat_tool_call has no `type`; the OpenAI wrapper shape
                 * is ours to synthesize.  ids are generated when the model's
                 * format doesn't carry them. */
                arr.push_back({
                    { "id",   tc.id.empty() ? ("call_" + std::to_string(i)) : tc.id },
                    { "type", "function" },
                    { "function", { { "name", tc.name }, { "arguments", tc.arguments } } }
                });
                i++;
            }
            tool_calls_json = arr.dump();
        } catch (const std::exception &) { tool_calls_json.clear(); }
    }
}

/* Does this model's template actually render tool definitions?  There is no
 * upstream capability query, so apply the template twice -- with and without a
 * probe tool -- and see whether anything changed.  Also records the format name
 * of the plain apply for diagnostics. */
static void probe_tool_support(lgen_engine *e) {
    e->supports_tools = false;
    e->chat_format    = "";
    if (!e->use_jinja || !e->templates) return;
    try {
        common_chat_templates_inputs in;
        in.use_jinja             = true;
        in.add_generation_prompt = true;
        common_chat_msg usr; usr.role = "user"; usr.content = "hi";
        in.messages.push_back(usr);

        common_chat_params plain = common_chat_templates_apply(e->templates.get(), in);
        const char *fmt = common_chat_format_name(plain.format);
        e->chat_format = fmt ? fmt : "";

        common_chat_tool probe;
        probe.name        = "rampart_probe";
        probe.description = "probe";
        probe.parameters  = "{\"type\":\"object\",\"properties\":{}}";
        in.tools.push_back(probe);
        in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

        common_chat_params withtool = common_chat_templates_apply(e->templates.get(), in);
        /* A template that understands tools either names the probe in the
         * rendered prompt or switches to a tool-aware format/grammar. */
        e->supports_tools = withtool.prompt.find("rampart_probe") != std::string::npos
                         || withtool.format != plain.format
                         || !withtool.grammar.empty();
    } catch (const std::exception &) {
        e->supports_tools = false;
    } catch (...) {
        e->supports_tools = false;
    }
}

/* start a request on a free slot; consumes (frees) req on success or failure */
static void start_slot(lgen_engine *e, gen_slot &slot, gen_request *req) {
    applied_prompt ap = apply_templates(e, req);
    char errbuf[256];
    const char *err = nullptr;
    std::vector<llama_token> toks;
    if (!ap.ok) { err = ap.err.empty() ? "failed to apply chat template" : ap.err.c_str(); }
    else if (ap.prompt.empty()) { err = "empty prompt"; }
    else {
        toks = common_tokenize(e->vocab, ap.prompt, true, true);
        if ((int) toks.size() >= slot.n_ctx_slot) { snprintf(errbuf, sizeof errbuf, "prompt longer than slot context (%d >= %d)", (int)toks.size(), slot.n_ctx_slot); err = errbuf; }
    }
    if (err) {
        lgen_result_extra extra{};
        if (req->on_done) req->on_done(req->ud, 1, err, LGEN_FINISH_ERROR, "", 0, &extra);
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
    slot.parsing = false;                 /* slots are reused; never inherit */
    slot.parser_params = common_chat_parser_params();
    slot.prev_msg = common_chat_msg();

    common_params_sampling sp = sampling_from_req(req);
    /* grammar / triggers / preserved tokens / extra stops / parser setup */
    apply_constraints(e, slot, sp, ap);
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
llama_model *model_acquire_path(const char *path, const llama_model_params *mp,
                                char *err, size_t errlen) {
    std::lock_guard<std::mutex> lk(g_model_cache_mtx);
    /* after a fork the inherited GPU/model handles are invalid in the child;
     * drop the cache so this process reloads its own (don't free parent's COW). */
    int pid = (int) getpid();
    if (g_model_cache_pid != pid) { g_model_cache.clear(); g_model_cache_pid = pid; }

    /* key on path + every load param that changes the resulting model, so two
     * engines differing in GPU offload / split don't wrongly share one model. */
    char flags[96];
    snprintf(flags, sizeof flags, "|lm%d|ct%d|ngl%d|sm%d|mg%d",
             (int) mp->load_mode, mp->check_tensors ? 1 : 0,
             mp->n_gpu_layers, (int) mp->split_mode, mp->main_gpu);
    std::string key = std::string(path) + flags;
    for (auto &e : g_model_cache)
        if (e.key == key) { e.refcount++; return e.model; }

    llama_model *m = llama_model_load_from_file(path, *mp);
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
    e->model = model_acquire_path(p->model_path, &p->mparams, err, errlen);
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

    /* the context options were all parsed by the shared parser into cparams */
    llama_context_params cp = p->cparams;
    /* 0 (unspecified, or nCtx:0/-1) => the model's trained max, same as llama-server */
    if (cp.n_ctx == 0) cp.n_ctx = (uint32_t) llama_model_n_ctx_train(e->model);
    cp.n_seq_max  = e->n_seq_max;   /* slot count */
    cp.embeddings = false;          /* gen never embeds */

    e->ctx = llama_init_from_model(e->model, cp);

    /* GPU context init can fail on a host with no usable device (e.g. Metal in a
     * VM / headless macOS, or a GPU OOM).  Fall back to a CPU-pinned context,
     * matching initEmbed/initRerank: drop the GPU model ref, reload pinned to the
     * CPU device (n_gpu_layers=0 -> distinct model-cache key), and retry.  Pinning
     * the device is required -- n_gpu_layers=0 alone still selects Metal/GPU for
     * compute.  A host with a working GPU succeeds on the first try and skips this. */
    if (!e->ctx) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            model_release(e->model);
            e->model = nullptr;
            llama_model_params cmp = p->mparams;
            ggml_backend_dev_t devs[2] = { cpu_dev, nullptr };
            cmp.n_gpu_layers = 0;
            cmp.devices      = devs;
            lt_warn("rampart-llamacpp: GPU context init failed for '%s'; retrying gen on CPU\n", p->model_path);
            e->model = model_acquire_path(p->model_path, &cmp, err, errlen);
            if (e->model) {
                e->vocab   = llama_model_get_vocab(e->model);
                e->n_vocab = llama_vocab_n_tokens(e->vocab);
                e->ctx     = llama_init_from_model(e->model, cp);
            }
        }
    }

    if (!e->ctx) { set_err(err, errlen, "failed to create llama context"); return false; }

    e->n_ctx   = llama_n_ctx(e->ctx);
    e->n_batch = llama_n_batch(e->ctx);
    e->add_bos = llama_vocab_get_add_bos(e->vocab);
    e->templates = common_chat_templates_init(e->model,
                       e->chat_template.empty() ? "" : e->chat_template);
    probe_tool_support(e);   /* sets supports_tools + chat_format */
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
struct llama_model *lgen_model_acquire(const char *path, const struct llama_model_params *mp,
                                       char *errbuf, size_t errlen) {
    if (!path) { set_err(errbuf, errlen, "missing model_path"); return nullptr; }
    if (!mp)   { set_err(errbuf, errlen, "missing model params"); return nullptr; }
    return model_acquire_path(path, mp, errbuf, errlen);
}
void lgen_model_addref(struct llama_model *m)  { model_addref(m); }
int lgen_model_addref_checked(struct llama_model *m) {
    if (!m) return 0;
    std::lock_guard<std::mutex> lk(g_model_cache_mtx);
    for (auto &e : g_model_cache)
        if (e.model == m) { e.refcount++; return 1; }
    return 0;   /* refcount already hit zero: the model was freed */
}
void lgen_model_release(struct llama_model *m) { model_release(m); }

lgen_engine *lgen_engine_create(const lgen_engine_params *p, char *errbuf, size_t errlen) {
    if (!p || !p->model_path) { set_err(errbuf, errlen, "missing model_path"); return nullptr; }

    lgen_engine *e = new lgen_engine();
    e->params = *p;
    e->model_path = p->model_path;
    e->params.model_path = e->model_path.c_str();
    /* own the chat template string (the caller's pointer won't outlive this call) */
    e->chat_template = p->chat_template ? p->chat_template : "";
    e->params.chat_template = e->chat_template.empty() ? nullptr : e->chat_template.c_str();
    e->use_jinja = p->use_jinja ? true : false;
    e->n_seq_max = p->cparams.n_seq_max ? p->cparams.n_seq_max : 1;

    /* Load the full model + build the context NOW, on the calling (rampart)
     * thread. This mirrors how embedding (initEmbed) loads on its worker thread
     * and decodes there: many rampart threads each run their own context's Metal
     * concurrently and safely. The context is pinned to this thread; on a thread
     * or pid (fork) change the caller (lg_get_info in rampart-llamacpp.c) builds
     * a fresh per-thread engine from the stored params instead of reusing this one. */
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
        lgen_result_extra extra{};
        if (r->on_done) r->on_done(r->ud, 1, "engine destroyed", LGEN_FINISH_CANCEL, "", 0, &extra);
        delete r;
    }
    e->waitq.clear();
    drop_context(e);
    if (e->model) { model_release(e->model); e->model = nullptr; }
    delete e;
}

int lgen_default_n_threads(void) {
#if defined(__FreeBSD__)
    /* libcommon has no FreeBSD branch in common_cpu_get_num_physical_cores():
     * it falls through to the generic tail, hardware_concurrency() and then
     * "/2 if > 4" -- which assumes SMT is always on and therefore HALVES a
     * non-SMT box (8 real cores -> 4).  FreeBSD publishes the topology, so ask
     * it.  (Linux enumerates thread_siblings and macOS uses
     * hw.perflevel0.physicalcpu, both of which are already correct.) */
    {
        int cores = 0;
        size_t len = sizeof(cores);
        if (sysctlbyname("kern.smp.cores", &cores, &len, NULL, 0) == 0 && cores > 0)
            return cores;
    }
#endif
    int n = 0;
    try { n = (int) common_cpu_get_num_math(); } catch (...) { n = 0; }
    return n > 0 ? n : 4;
}

uint32_t lgen_engine_n_ctx(lgen_engine *e)   { return e ? (e->n_ctx ? e->n_ctx : e->n_ctx_cfg) : 0; }
int32_t  lgen_engine_n_vocab(lgen_engine *e) { return e ? e->n_vocab : 0; }
int      lgen_engine_supports_tools(lgen_engine *e) { return (e && e->supports_tools) ? 1 : 0; }
const char *lgen_engine_chat_format(lgen_engine *e) { return e ? e->chat_format.c_str() : ""; }
int lgen_engine_supports_thinking_toggle(lgen_engine *e) {
    if (!e || !e->templates || !e->use_jinja) return 0;
    return common_chat_templates_support_enable_thinking(e->templates.get()) ? 1 : 0;
}

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
    if (!e->ctx)    { set_err(errbuf, errlen, "engine has no context"); return 0; }

    gen_request *r = new gen_request();
    if (req->prompt && req->prompt[0])               { r->has_prompt = true;   r->prompt = req->prompt; }
    if (req->messages_json && req->messages_json[0]) { r->has_messages = true; r->messages_json = req->messages_json; }
    if (req->chat_template)                           r->chat_template = req->chat_template;
    if (req->tools_json && req->tools_json[0])        r->tools_json = req->tools_json;
    r->tool_choice = req->tool_choice;
    r->parallel_tool_calls = req->parallel_tool_calls ? true : false;
    r->reasoning_separate  = req->reasoning_separate ? true : false;
    r->thinking            = req->thinking;
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
