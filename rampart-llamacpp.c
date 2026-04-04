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
#include "llama.h"
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


// LLAMA.CPP Generation funcs:

// Returns 1 if we should stop generation, 0 otherwise.
static int rp_check_stop(const struct llama_vocab *vocab, llama_token tok,
                         int token_index, // how many tokens generated so far
                         int max_tokens,  // hard cap
                         char *tokstr, int tokstrlen)
{

    // --- 1) EOG token ---
    if (llama_vocab_is_eog(vocab, tok))
        return 1;

    // --- 2) EOS token ---
    if (tok == llama_vocab_eos(vocab))
        return 1;

    // --- 3) max tokens cap ---
    if (token_index >= max_tokens)
        return 1;

    // --- 4) check control token
    if (token_index && tokstr)
    {
        // make sure its terminated
        tokstr[tokstrlen] = '\0';
        if (strstr(tokstr, "assistant"))
            return 1;
        if (strstr(tokstr, "end"))
            return 1;
    }

    return 0; // keep going
}

// Try to use llama_chat_apply_template if compiled in; otherwise fallback.
static char *rp_build_prompt(duk_context *ctx, duk_idx_t obj_idx, struct llama_model *lmodel, size_t *out_len)
{
    char *buf = NULL;
    bool add_assistant = true;

    // 1) If plain "prompt" is given, use it directly (but wrap it in a default template)
    if (duk_get_prop_string(ctx, obj_idx, "prompt"))
    {
        const char *p = REQUIRE_STRING(ctx, -1, "prompt must be a string");

        struct llama_chat_message *msgs = NULL;
        REMALLOC(msgs, sizeof(*msgs) * 2);

        msgs[0].role = "system";
        msgs[0].content = "You are a helpful AI assistant";

        msgs[1].role = "user";
        msgs[1].content = p;

        // optional template name
        const char *tmpl = NULL;

        if (!tmpl)
            tmpl = llama_model_chat_template(lmodel, NULL);
        int32_t need = llama_chat_apply_template(tmpl, msgs, 2, add_assistant, NULL, 0);

        if (need == 0)
        {
            free(msgs);
            RP_THROW(ctx, "chat template failed (size query)");
        }

        REMALLOC(buf, (size_t)need + 1);

        // FIXME: don't do this twice if possible.  Look at notes in ../extern/llama.cpp/include/llama.h
        need = llama_chat_apply_template(tmpl, msgs, 2, add_assistant, buf, need);
        free(msgs);
        if (need == 0)
        {
            free(buf);
            RP_THROW(ctx, "chat template failed");
            return NULL;
        }

        buf[need] = '\0';
        if (out_len)
            *out_len = need;
    }

    // 2) Else expect { messages: [ {role, content}, ... ] }
    if (duk_get_prop_string(ctx, obj_idx, "messages"))
    {
        if (buf)
        {
            free(buf);
            RP_THROW(ctx, "Input must be prompt OR messages, not both");
        }

        if (!duk_is_array(ctx, -1))
            RP_THROW(ctx, "messages must be an Array");

        // Build C array of llama_chat_message
        int n = (int)duk_get_length(ctx, -1);

        if (n <= 0)
        {
            duk_pop(ctx);
            RP_THROW(ctx, "messages must be non-empty");
        }

        struct llama_chat_message *msgs = NULL;
        REMALLOC(msgs, sizeof(*msgs) * (size_t)n);

        for (int i = 0; i < n; ++i)
        {
            duk_get_prop_index(ctx, -1, (duk_uarridx_t)i);

            if (!duk_is_object(ctx, -1) || !duk_get_prop_string(ctx, -1, "role") || !duk_is_string(ctx, -1))
            {
                free(msgs);
                RP_THROW(ctx, "messages must contain Object Array members with {role:String, content:String}");
            }
            msgs[i].role = duk_get_string(ctx, -1);
            duk_pop(ctx);

            if (!duk_get_prop_string(ctx, -1, "content") || !duk_is_string(ctx, -1))
            {
                free(msgs);
                RP_THROW(ctx, "messages must contain Object Array members with {role:String, content:String}");
            }

            msgs[i].content = duk_get_string(ctx, -1);
            duk_pop_2(ctx); // string, msg object
        }

        // optional template name
        const char *tmpl = NULL;

        if (duk_get_prop_string(ctx, obj_idx, "template"))
        {
            if (!duk_is_string(ctx, -1))
            {
                free(msgs);
                RP_THROW(ctx, "option template must be a String");
            }
            tmpl = duk_get_string(ctx, -1);
        }
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "addAssistant"))
        {
            if (!duk_is_boolean(ctx, -1))
            {
                free(msgs);
                RP_THROW(ctx, "option addAssistant must be a Boolean");
            }
            add_assistant = (bool)duk_get_boolean(ctx, -1);
        }
        duk_pop(ctx);

        // probe buffer size: call with NULL to get required size (llama.cpp supports this pattern)

        if (!tmpl)
            tmpl = llama_model_chat_template(lmodel, NULL);
        int32_t need = llama_chat_apply_template(tmpl, msgs, (size_t)n, add_assistant, NULL, 0);

        if (need == 0)
        {
            free(msgs);
            RP_THROW(ctx, "chat template failed (size query)");
        }

        REMALLOC(buf, (size_t)need + 1);

        // FIXME: maybe don't do this twice if possible.  Look at notes in ../extern/llama.cpp/include/llama.h
        need = llama_chat_apply_template(tmpl, msgs, (size_t)n, add_assistant, buf, need);
        free(msgs);
        if (need == 0)
        {
            free(buf);
            RP_THROW(ctx, "chat template failed");
            return NULL; // suppress warning
        }

        buf[need] = '\0';
        if (out_len)
            *out_len = need;
    }
    duk_pop(ctx);

    return buf;
}

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
} rp_llama_info;

static rp_llama_info *rp_get_llama_info(duk_context *ctx)
{
    if (!ctx)
    {
        RPTHR *t = get_current_thread();
        ctx = t->ctx;
    }

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("destroyed"));

    if (duk_get_boolean_default(ctx, -1, 0))
        RP_THROW(ctx, "generation object was destroyed");
    duk_pop(ctx);

    rp_llama_info *ret = NULL;

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rp_llama_info")))
        ret = (rp_llama_info *)duk_get_pointer(ctx, -1);
    duk_pop_2(ctx); // info and this

    if (!ret)
    {
        RP_THROW(ctx, "internal error getting llama info");
        return NULL; // silence warning
    }

    if (ret->thr->ctx != ctx)
        RP_THROW(ctx, "cannot use llama from a thread other than where it was created");

    return ret;
}

static int batch_and_decode(duk_context *ctx, struct llama_context *lctx, llama_memory_t mem, llama_token *tok,
                            int32_t len, uint32_t cur_pos)
{
    struct llama_batch b = llama_batch_get_one(tok, len);
    int ret = llama_decode(lctx, b);

    /*
        if(ret == 1)
        {
            printf("\ndoing stupid copy to reset; no idea what I'm doing wrong here\n");
            llama_memory_seq_cp(mem, 0, 1, cur_pos, -1);
            llama_memory_seq_rm(mem, 0, 0, -1);
            llama_memory_seq_cp(mem, 1, 0, cur_pos, -1);
            llama_memory_seq_rm(mem, 1, 0, -1);
            struct llama_batch b = llama_batch_get_one(tok, len);
            ret = llama_decode(lctx, b);
        }
    */
    return ret;
}

static int gen_async_one(void *arg, int unused);

static int gen_cleanup(rp_llama_info *linfo)
{
    duk_context *ctx = linfo->ctx;

    duk_push_global_stash(ctx);

    duk_push_sprintf(ctx, "llamacb_%p", linfo->func_ptr);
    duk_del_prop(ctx, -2);

    duk_push_sprintf(ctx, "llamacb_final_%p", linfo->func_ptr);

    if (duk_get_prop(ctx, -2))
    {
        duk_push_sprintf(ctx, "llamathis_%p", linfo->func_ptr);
        duk_get_prop(ctx, -3);
        if (duk_pcall_method(ctx, 0) != 0)
        {
            const char *errmsg;

            errmsg = rp_push_error(ctx, -1, NULL, rp_print_error_lines);
            fprintf(stderr, "In final callback: %s\n", errmsg);

            duk_pop(ctx);
        }
    }
    duk_pop(ctx); // undef or return value from call

    duk_push_sprintf(ctx, "llamacb_final_%p", linfo->func_ptr);
    duk_del_prop(ctx, -2);

    duk_push_sprintf(ctx, "llamathis_%p", linfo->func_ptr);
    duk_del_prop(ctx, -2);

    if (linfo->smpl)
        llama_sampler_free(linfo->smpl);
    linfo->smpl = NULL;

    duk_pop(ctx); // the stash

    return 0;
}

static int gen_async_one(void *arg, int stage)
{
    rp_llama_info *linfo = (rp_llama_info *)arg;

    if (stage)
    {
        if (linfo->stop)
        {
            gen_cleanup(linfo);
            return 0;
        }
        // return 1 from stage 1 to go again
        return 1;
    }

    // stage 0
    duk_context *ctx = linfo->ctx;
    llama_memory_t mem = linfo->mem;

    duk_idx_t top = duk_get_top(ctx);

    if (linfo->cur_pos + 1 >= linfo->n_ctx)
    {
        int n_keep = linfo->n_keep;
        int cur_pos = linfo->cur_pos;
        const int n_left = cur_pos - n_keep; // tokens after the kept prompt
        const int n_discard = n_left > 0 ? n_left / 2 : 0;
        if (n_discard > 0)
        {

            // remove [n_keep, n_keep + n_discard)
            llama_memory_seq_rm(mem, linfo->seq_id, n_keep, n_keep + n_discard);

            // shift [n_keep + n_discard, cur_pos) left by n_discard
            llama_memory_seq_add(mem, linfo->seq_id, n_keep + n_discard, cur_pos, -n_discard);

            linfo->cur_pos -= n_discard;
        }
        else
        {
            linfo->stop = 1;
            return 1; // continue so we can clean up
        }
    }
    linfo->cur_pos++;

    // sample next token
    llama_token tok = llama_sampler_sample(linfo->smpl, linfo->lctx, /*pos*/ -1);
    llama_sampler_accept(linfo->smpl, tok); // update repetition/mirostat/etc. inside the chain

    // ---- detokenize and output/stream ----
    {
        char piece[256];
        int plen = llama_token_to_piece(linfo->vocab, tok, piece, (int)sizeof(piece), /*lstrip*/ 0, /*special*/ true);
        if (plen < 0)
        {
            linfo->stop = 1;
            duk_set_top(ctx, top);
            linfo->errmsg = "llama_token_to_piece() failed";
            return 1;
        }

        int is_control = llama_vocab_is_control(linfo->vocab, tok);

        if (is_control && rp_check_stop(linfo->vocab, tok, linfo->n_generated + 1, linfo->max_tokens, piece, plen))
        {
            linfo->stop = 1;
            duk_set_top(ctx, top);
            return 1;
        }
        else if (rp_check_stop(linfo->vocab, tok, linfo->n_generated + 1, linfo->max_tokens, NULL, 0))
        {
            linfo->stop = 1;
            duk_set_top(ctx, top);
            return 1;
        }

        if (!is_control)
        {
            // append to output buffer
            if (!linfo->out)
            {
                REMALLOC(linfo->out, 4096);
                linfo->out_cap = 4096;
            }
            if (linfo->out_len + (size_t)plen > linfo->out_cap)
            {
                size_t new_cap = ((linfo->out_len + (size_t)plen) * 3) / 2;
                REMALLOC(linfo->out, new_cap);
                linfo->out_cap = new_cap;
            }
            memcpy(linfo->out + linfo->out_len, piece, (size_t)plen);
            linfo->out_len += (size_t)plen;

            if (linfo->func_ptr) // we are async
            {
                duk_push_global_stash(ctx);
                duk_push_sprintf(ctx, "llamacb_%p", linfo->func_ptr);
                duk_get_prop(ctx, -2);
                duk_push_sprintf(ctx, "llamathis_%p", linfo->func_ptr);
                duk_get_prop(ctx, -3);
                duk_remove(ctx, -3); // stash
                duk_push_lstring(ctx, piece, (duk_size_t)plen);
                if (duk_pcall_method(ctx, 1) != 0)
                {
                    const char *errmsg;

                    errmsg = rp_push_error(ctx, -1, NULL, rp_print_error_lines);
                    fprintf(stderr, "In final callback: %s\n", errmsg);

                    duk_pop(ctx);
                    linfo->stop = 1;
                    duk_set_top(ctx, top);
                    return 1;
                }
            }
            else // we are sync
            {
                duk_dup(ctx, linfo->func_idx);
                duk_push_this(ctx);
                duk_push_lstring(ctx, piece, (duk_size_t)plen);
                duk_call_method(ctx, 1);
            }

            if (duk_get_boolean_default(ctx, -1, 0))
            {
                linfo->stop = 1;
                duk_pop(ctx);
                duk_set_top(ctx, top);
                return 1;
            }
            duk_pop(ctx);
        }
    }

    if (batch_and_decode(ctx, linfo->lctx, mem, &tok, 1, linfo->cur_pos))
    {
        linfo->stop = 1;
        duk_set_top(ctx, top);
        linfo->errmsg = "llama_decode() failed";
        return 1;
    }

    // one more token successfully generated
    linfo->n_generated++;

    duk_set_top(ctx, top);

    return 1;
}

rp_llama_info *prep_predict(duk_context *ctx, int is_async)
{
    rp_llama_info *linfo = rp_get_llama_info(ctx);

    struct llama_context *lctx = linfo->lctx;
    struct llama_model *lmodel = linfo->lmodel;
    const struct llama_vocab *vocab = linfo->vocab;

    uint32_t cur_pos = linfo->cur_pos;

    int cur_thr = get_thread_num();
    int cur_pid = (int)getpid();

    // get a new context if in a new thread.  Model stays the same.
    if (cur_thr != linfo->init_thr || cur_pid != linfo->init_pid )
    {
        // FIXME: save and retrieve options.
#ifdef HAVE_CUDA
        // forking after is bad, mkay
        if(cur_pid != linfo->init_pid && has_gpu_backend() )
        {
            RP_THROW(ctx, "llama.cpp - cannot fork llama.cpp with CUDA initialized");
        }
#endif
        lctx = linfo->lctx = llama_init_from_model(linfo->lmodel, linfo->cp);

        linfo->init_thr = cur_thr;
        linfo->init_pid = cur_pid;

    }


    // generation params (defaults)
    int max_tokens = 12800;
    float temp = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = -1;

    REQUIRE_OBJECT(ctx, 0, "First argument must be an Object");

    if (is_async)
    {
        REQUIRE_FUNCTION(ctx, 1, "Second argument must be a Function (piece callback)");
        // put function where it will not be GCed and store its pointer for easy access
        linfo->func_ptr = duk_get_heapptr(ctx, 1);

        duk_push_global_stash(ctx);

        duk_push_sprintf(ctx, "llamacb_%p", linfo->func_ptr);
        duk_dup(ctx, 1);
        duk_put_prop(ctx, -3);

        duk_push_sprintf(ctx, "llamathis_%p", linfo->func_ptr);
        duk_push_this(ctx);
        duk_put_prop(ctx, -3);

        if (duk_is_function(ctx, 2))
        {
            duk_push_sprintf(ctx, "llamacb_final_%p", linfo->func_ptr);
            duk_dup(ctx, 2);
            duk_put_prop(ctx, -3);
        }

        duk_pop(ctx); // stash
    }
    else
    {
        linfo->func_idx = -1;
        if (duk_is_function(ctx, 1))
        {
            linfo->func_idx = 1;
        }
    }

    if (duk_get_prop_string(ctx, 0, "maxTokens"))
        max_tokens = (int)REQUIRE_UINT(ctx, -1, "maxTokens must be a positive integer");
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, 0, "temp"))
        temp = (float)REQUIRE_NUMBER(ctx, -1, "temp must be a number");
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, 0, "topP"))
        top_p = (float)REQUIRE_NUMBER(ctx, -1, "topP must be a number");
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, 0, "topK"))
        top_k = (int)REQUIRE_UINT(ctx, -1, "topK must be a positive integer");
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, 0, "repeatPenalty"))
        repeat_penalty = (float)REQUIRE_NUMBER(ctx, -1, "repeatPenalty must be a number");
    duk_pop(ctx);

    if (duk_get_prop_string(ctx, 0, "repeatLastN"))
        repeat_last_n = (int)REQUIRE_UINT(ctx, -1, "repeatLastN must be a positive integer");
    duk_pop(ctx);

    // reset
    llama_memory_t mem = llama_get_memory(lctx);

    if (duk_get_prop_string(ctx, 0, "resetMem"))
    {
        if (REQUIRE_BOOL(ctx, -1, "resetMem must be a Boolean"))
        {
            llama_memory_clear(mem, /*clear_kv_data*/ true);
            linfo->cur_pos = cur_pos = 0;
        }
    }
    duk_pop(ctx);

    // Build the prompt string from {prompt} or {messages,...}
    size_t prompt_len = 0;
    char *prompt = rp_build_prompt(ctx, 0, lmodel, &prompt_len);

    duk_push_this(ctx);
    duk_push_string(ctx, prompt);
    duk_put_prop_string(ctx, -2, "lastRawPrompt");
    duk_pop(ctx);

    // Tokenize the prompt
    int cap = (int)(prompt_len + 8);
    llama_token *inp = NULL;
    REMALLOC(inp, sizeof(llama_token) * (size_t)cap);

    int n_inp = llama_tokenize(vocab, prompt, (int)prompt_len, inp, cap, /*add_special*/ true, /*parse_special*/ true);

    if (n_inp >= cap)
    {
        cap = n_inp + 8;
        REMALLOC(inp, sizeof(llama_token) * (size_t)cap);
        n_inp = llama_tokenize(vocab, prompt, (int)prompt_len, inp, cap, true, true);
    }

#ifdef GPT_IS_SMART_HAH
    // 1) Ask the model if it wants a BOS token automatically
    bool need_bos = llama_vocab_get_add_bos(vocab); // same predicate main.cpp uses
    if (need_bos)
    {
        const llama_token bos = llama_token_bos(vocab);

        // 2) If the first token is not already BOS, prepend it
        if (n_inp == 0 || inp[0] != bos)
        {
            // ensure capacity (grow if you're using a fixed buffer)
            if (n_inp == cap)
            {
                // reallocate inp[] to a larger buffer; or assert/grow your vector
                // (do whatever you already do elsewhere on overflow)
            }
            printf("needed bos\n");
            memmove(&inp[1], &inp[0], n_inp * sizeof(inp[0]));
            inp[0] = bos;
            n_inp += 1;
        }
    }
#endif

    int n_keep = n_inp; // keep at least the whole prompt by default
    // track how many tokens are already in the KV after you decoded the prompt:

    cur_pos += n_inp;

    free(prompt);

    if (n_inp < 0)
    {
        free(inp);
        // if (stops){for(int i=0;i<nstops;++i) free(stops[i].ptr); free(stops);}
        RP_THROW(ctx, "tokenize prompt failed");
    }

    // Feed the prompt TODO: if we are near end of n_ctx, we need to compact here too.
    {
        struct llama_batch batch = llama_batch_get_one(inp, n_inp);
        if (llama_decode(lctx, batch) != 0)
        {
            free(inp);
            // if (stops){for(int i=0;i<nstops;++i) free(stops[i].ptr); free(stops);}
            RP_THROW(ctx, "llama_decode(prompt) failed");
        }
    }

    if (repeat_last_n == -1)
        repeat_last_n = 64;

    // sampler
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();

    sparams.no_perf = 0;
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    if (!smpl)
    {
        free(inp);
        RP_THROW(ctx, "sampler_chain_init failed");
    }

    // --- 0) Penalties (same as you had) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                                      /*penalty_last_n*/ repeat_last_n,  // e.g. 64 unless user overrides
                                      /*penalty_repeat*/ repeat_penalty, // e.g. 1.1
                                      /*penalty_freq*/ 0.0f,
                                      /*penalty_present*/ 0.0f));

    // --- 1) DRY (needs vocab + extra args) ---
    const int32_t n_ctx_train = llama_model_n_ctx_train(lmodel); // model's train ctx (or 0)

    // Optional “sequence breakers” like llama-cli defaults:
    static const char *dry_breakers[] = {"\n", ":", "\"", "*"};
    const size_t n_breakers = sizeof(dry_breakers) / sizeof(dry_breakers[0]);

    llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                                      /*vocab*/ vocab,
                                      /*n_ctx_train*/ n_ctx_train, // use model value; 0 is also accepted
                                      /*dry_multiplier*/ 1.00f,    // start modest; tune if needed
                                      /*dry_base*/ 1.75f,          // common default
                                      /*dry_allowed_length*/ 2,
                                      /*dry_penalty_last_n*/ -1,     // -1 = auto (use ctx size)
                                      /*seq_breakers*/ dry_breakers, // or NULL
                                      /*num_breakers*/ n_breakers)); // or 0

    // --- 2) Top-n-sigma (variance cull; 0.0 disables) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_top_n_sigma(/*n_sigma*/ 0.0f));

    // --- 3) Top-K ---
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));

    // --- 4) Typical (needs min_keep) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_typical(/*typ_p*/ 1.0f, /*min_keep*/ 1));
    // set typ_p<1.0f (e.g. 0.95f) to enable

    // --- 5) Top-P (you already had) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, /*min_keep*/ 1));

    // --- 6) Min-P (you already had) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.01f, /*min_keep*/ 1));

    // --- 7) XTC (4 args; keep disabled by default) ---
    uint32_t seed32 = (uint32_t)time(NULL);
    llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
                                      /*probability*/ 0.0f, // 0 = off; e.g. 1.0f to always apply
                                      /*threshold*/ 1.0f,   // 1.0 = off; e.g. 0.10f commonly used
                                      /*min_keep*/ 1,
                                      /*seed*/ seed32));

    // --- 8) Temperature ---
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));

    // --- 9) Final draw (stochastic, not greedy) ---
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed32));

    for (int i = 0; i < n_inp; ++i)
    {
        // no grammar for prompt; llama-cli passes accept_grammar=false there
        llama_sampler_accept(smpl, inp[i]);
    }

    free(inp);

    linfo->n_generated = 0;
    linfo->mem = mem;
    linfo->max_tokens = max_tokens;
    linfo->smpl = smpl;
    linfo->n_keep = n_keep;
    linfo->stop = 0;
    linfo->seq_id = 0;
    linfo->ctx = ctx;

    if (linfo->out)
    {
        free(linfo->out);
        linfo->out = NULL;
        linfo->out_cap = 0;
        linfo->out_len = 0;
    }

    return linfo;
}

static duk_ret_t gen_predict_async(duk_context *ctx)
{

    rp_llama_info *linfo = prep_predict(ctx, 1);

    // get the ball rolling
    (void)duk_rp_insert_timeout(ctx, 0, "predictAsync", gen_async_one, (void *)linfo, DUK_INVALID_INDEX,
                                DUK_INVALID_INDEX, 0.0);

    return 0;
}

static duk_ret_t gen_predict(duk_context *ctx)
{

    rp_llama_info *linfo = prep_predict(ctx, 0);

    while (1)
    {
        gen_async_one(linfo, 0);
        if (linfo->stop)
        {
            gen_async_one(linfo, 1); // cleanup
            break;
        }
    }

    if (linfo->func_idx == -1) // no callback
        duk_push_lstring(ctx, linfo->out, linfo->out_len);
    else
        duk_push_int(ctx, linfo->n_generated);

    return 1;
}

duk_ret_t get_last_gen(duk_context *ctx)
{
    rp_llama_info *linfo = rp_get_llama_info(ctx);

    if (linfo->out)
        duk_push_lstring(ctx, linfo->out, linfo->out_len);
    else
        duk_push_string(ctx, "");

    return 1;
}

static duk_ret_t gen_free(duk_context *ctx)
{
    rp_llama_info *info = rp_get_llama_info(ctx);

    llama_free(info->lctx);

    llama_model_free(info->lmodel);

    if (info->out)
        free(info->out);

    if (info->smpl)
        llama_sampler_free(info->smpl);

    free(info);

    duk_push_this(ctx);

    duk_push_pointer(ctx, NULL);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rp_llama_info"));

    duk_push_true(ctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("destroyed"));

    return 0;
}

// ===================== n_ctx auto-sizer (drop-in) =====================

// ---- System RAM (total/free) ----
static inline uint64_t sys_ram_total_bytes(void)
{
#ifdef __APPLE__
    uint64_t mem = 0;
    size_t sz = sizeof(mem);
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    if (sysctl(mib, 2, &mem, &sz, NULL, 0) == 0)
        return mem;
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    return (pages > 0 && psize > 0) ? (uint64_t)pages * (uint64_t)psize : 0;
#endif
}

static inline uint64_t sys_ram_free_bytes(void)
{
#ifdef __APPLE__
    // macOS “free” is fuzzy; be conservative (40% of total)
    uint64_t tot = sys_ram_total_bytes();
    return tot ? (tot * 40) / 100 : 0;
#else
#ifdef _SC_AVPHYS_PAGES
    long apages = sysconf(_SC_AVPHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (apages > 0 && psize > 0)
        return (uint64_t)apages * (uint64_t)psize;
#endif
    uint64_t tot = sys_ram_total_bytes();
    return tot ? (tot * 40) / 100 : 0;
#endif
}

// ---- CUDA VRAM (free) ----
static inline bool cuda_free_bytes(uint64_t *free_b, uint64_t *total_b)
{
#if HAVE_CUDA
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess)
        return false;
    if (free_b)
        *free_b = (uint64_t)f;
    if (total_b)
        *total_b = (uint64_t)t;
    return true;
#else
    (void)free_b;
    (void)total_b;
    return false;
#endif
}

double kv_elem_bytes(enum ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_F32:
        return 4.0;
    case GGML_TYPE_F16:
        return 2.0;
    case GGML_TYPE_BF16:
        return 2.0;
    case GGML_TYPE_Q8_0:
        return 9.0 / 8.0; // ≈1.125
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_IQ4_NL:
        return 0.55; // rough average, adjust if you want exact
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
        return 0.65; // rough average
    default:
        return 2.0; // safe fallback
    }
}

// ---- KV bytes per token from model hyperparams ----
// head_dim = n_embd / n_head
// bytes/token = n_layer * n_kv_heads * head_dim * (bytes(K)+bytes(V))
static inline uint64_t kv_bytes_per_token(const struct llama_model *m, enum ggml_type k_type, enum ggml_type v_type)
{
    const int n_layer = llama_model_n_layer(m);
    const int n_embd = llama_model_n_embd(m);
    const int n_head = llama_model_n_head(m);
    const int n_head_kv = llama_model_n_head_kv(m);
    if (n_layer <= 0 || n_embd <= 0 || n_head <= 0 || n_head_kv <= 0)
        return 0;

    const int head_dim = n_embd / n_head;
    double bytes_k = kv_elem_bytes(k_type);
    double bytes_v = kv_elem_bytes(v_type);

    double per_lyr = (double)n_head_kv * (double)head_dim * (bytes_k + bytes_v);
    return (uint64_t)((double)n_layer * per_lyr); // bytes/token
}

// ---- Suggest n_ctx from memory budget ----
// kv_on_gpu = true -> size against CUDA VRAM (if available)
// headroom_frac: e.g. 0.15 leaves 15% slack
// headroom_bytes: absolute extra slack (e.g. 512 MiB)
// cpu_free_cap_frac: only use a fraction of free system RAM (e.g. 0.50)
static inline uint32_t suggest_n_ctx_from_mem(const struct llama_model *m, bool kv_on_gpu, enum ggml_type k_type,
                                              enum ggml_type v_type, double headroom_frac, uint64_t headroom_bytes,
                                              double cpu_free_cap_frac, uint64_t *dbg_base_bytes, uint64_t *dbg_kv_bpt)
{
    const uint64_t kv_bpt = kv_bytes_per_token(m, k_type, v_type);
    if (dbg_kv_bpt)
        *dbg_kv_bpt = kv_bpt;
    if (kv_bpt == 0)
        return 0;

    uint64_t base = 0;
    if (kv_on_gpu)
    {
        uint64_t free_b = 0, total_b = 0;
        if (!cuda_free_bytes(&free_b, &total_b))
            return 0;
        base = free_b;
    }
    else
    {
        base = sys_ram_free_bytes();
        if (cpu_free_cap_frac > 0.0 && cpu_free_cap_frac <= 1.0)
        {
            base = (uint64_t)((double)base * cpu_free_cap_frac);
        }
    }
    if (dbg_base_bytes)
        *dbg_base_bytes = base;

    // apply headroom
    uint64_t budget = base;
    if (headroom_frac > 0.0 && headroom_frac < 1.0)
        budget = (uint64_t)((double)budget * (1.0 - headroom_frac));
    if (headroom_bytes > 0 && budget > headroom_bytes)
        budget -= headroom_bytes;

    if (budget < kv_bpt)
        return 0;
    uint64_t nctx64 = budget / kv_bpt;
    if (nctx64 > 0xFFFFFFFFull)
        nctx64 = 0xFFFFFFFFull;
    return (uint32_t)nctx64;
}
// =================== end n_ctx auto-sizer ===================

static duk_ret_t llamacpp_init_gen(duk_context *ctx)
{
    const char *model_path = REQUIRE_STRING(ctx, 0, "initGen: First argument must be a String (path to .gguf)");

    duk_idx_t obj_idx = -1;
    if (duk_is_object(ctx, 1))
        obj_idx = 1;

    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;
    int store_last = 1;

    // ---------------- Model params ----------------
    struct llama_model_params mp = llama_model_default_params();
    mp.use_mmap = false;  // <— key change for ZFS/NFS/SMB or spinning disks
    mp.use_mlock = false; // be explicit

    // (optional) faster failure visibility:
    mp.progress_callback = NULL; // or set your logger if you have one
    mp.progress_callback_user_data = NULL;

    if (obj_idx >= 0)
    {
        if (duk_get_prop_string(ctx, obj_idx, "vocabOnly"))
            mp.vocab_only = REQUIRE_BOOL(ctx, -1, "vocabOnly must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "useMmap"))
            mp.use_mmap = REQUIRE_BOOL(ctx, -1, "useMmap must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "useMlock"))
            mp.use_mlock = REQUIRE_BOOL(ctx, -1, "useMlock must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "checkTensors"))
            mp.check_tensors = REQUIRE_BOOL(ctx, -1, "checkTensors must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "storeLastRawPrompt"))
            store_last = REQUIRE_BOOL(ctx, -1, "storeLastRawPrompt must be boolean");
        duk_pop(ctx);
    }

    lmodel = llama_model_load_from_file(model_path, mp);
    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:initText - Could not load model '%s': %s", model_path, strerror(errno));
    /*
        // Optional LoRA
        const char *lora_path = NULL;
        float lora_scale = 1.0f;
        int lora_threads = (int)get_thread_num();

        if (obj_idx >= 0) {
            if (duk_get_prop_string(ctx, obj_idx, "lora"))
                lora_path = REQUIRE_STRING(ctx, -1, "lora must be a string (path)");
            duk_pop(ctx);

            if (duk_get_prop_string(ctx, obj_idx, "loraScale"))
                lora_scale = (float)REQUIRE_NUMBER(ctx, -1, "loraScale must be a number");
            duk_pop(ctx);

            if (duk_get_prop_string(ctx, obj_idx, "loraThreads"))
                lora_threads = (int)REQUIRE_UINT(ctx, -1, "threads must be a positive integer");
            duk_pop(ctx);
        }
    *
        if (lora_path && lora_path[0]) {
            if (llama_model_apply_lora_from_file(lmodel, lora_path, NULL, lora_scale, lora_threads) != 0)
                RP_THROW(ctx, "rampart-llama-cpp:initText - Failed to apply LoRA '%s'", lora_path);
        }
    */
    // ---------------- Context params ----------------
    struct llama_context_params cp = llama_context_default_params();

    // defaults
    uint32_t ga_n = 1;
    uint32_t ga_w = 512;
    cp.n_threads = 1;
    cp.n_threads_batch = cp.n_threads;
    cp.swa_full = 0;
    cp.n_ctx = 0;
    cp.flash_attn_type = -1;

    int n_train = llama_model_n_ctx_train(lmodel);

    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;

    char typek_str[16];
    char typev_str[16];

    strcpy(typek_str, "F16");
    strcpy(typev_str, "F16");

    if (obj_idx >= 0)
    {
        if (duk_get_prop_string(ctx, obj_idx, "nCtx"))
            cp.n_ctx = (uint32_t)REQUIRE_UINT(ctx, -1, "nCtx must be a positive integer");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "cacheType"))
        {
            const char *tp = REQUIRE_STRING(ctx, -1, "cacheType must be a String ('q8' or 'fp16)");
            if (strcasecmp("F32", tp) == 0)
            {
                cp.type_k = GGML_TYPE_F32;
                cp.type_v = GGML_TYPE_F32;
            }
            else if (strcasecmp("BF16", tp) == 0)
            {
                cp.type_k = GGML_TYPE_BF16;
                cp.type_v = GGML_TYPE_BF16;
            }
            else if (strcasecmp("Q8_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q8_0;
                cp.type_v = GGML_TYPE_Q8_0;
            }
            else if (strcasecmp("Q5_1", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q5_1;
                cp.type_v = GGML_TYPE_Q5_1;
            }
            else if (strcasecmp("Q5_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q5_0;
                cp.type_v = GGML_TYPE_Q5_0;
            }
            else if (strcasecmp("IQ4_NL", tp) == 0)
            {
                cp.type_k = GGML_TYPE_IQ4_NL;
                cp.type_v = GGML_TYPE_IQ4_NL;
            }
            else if (strcasecmp("Q4_1", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q4_1;
                cp.type_v = GGML_TYPE_Q4_1;
            }
            else if (strcasecmp("Q4_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q4_0;
                cp.type_v = GGML_TYPE_Q4_0;
            }
            else if (strcasecmp("F16", tp) != 0)
                RP_THROW(ctx, "cacheType must be a String ('q8' or 'fp16)");
            strcpy(typek_str, tp);
            strcpy(typev_str, tp);
        }
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "kCacheType"))
        {
            const char *tp = REQUIRE_STRING(ctx, -1, "cacheType must be a String ('q8' or 'fp16)");
            if (strcasecmp("F32", tp) == 0)
            {
                cp.type_k = GGML_TYPE_F32;
            }
            else if (strcasecmp("BF16", tp) == 0)
            {
                cp.type_k = GGML_TYPE_BF16;
            }
            else if (strcasecmp("Q8_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q8_0;
            }
            else if (strcasecmp("Q5_1", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q5_1;
            }
            else if (strcasecmp("Q5_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q5_0;
            }
            else if (strcasecmp("IQ4_NL", tp) == 0)
            {
                cp.type_k = GGML_TYPE_IQ4_NL;
            }
            else if (strcasecmp("Q4_1", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q4_1;
            }
            else if (strcasecmp("Q4_0", tp) == 0)
            {
                cp.type_k = GGML_TYPE_Q4_0;
            }
            else if (strcasecmp("F16", tp) != 0)
                RP_THROW(ctx, "kCacheType must be a String ('q8' or 'fp16)");
            strcpy(typek_str, tp);
        }
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "vCacheType"))
        {
            const char *tp = REQUIRE_STRING(ctx, -1, "cacheType must be a String ('q8' or 'fp16)");
            if (strcasecmp("F32", tp) == 0)
            {
                cp.type_v = GGML_TYPE_F32;
            }
            else if (strcasecmp("BF16", tp) == 0)
            {
                cp.type_v = GGML_TYPE_BF16;
            }
            else if (strcasecmp("Q8_0", tp) == 0)
            {
                cp.type_v = GGML_TYPE_Q8_0;
            }
            else if (strcasecmp("Q5_1", tp) == 0)
            {
                cp.type_v = GGML_TYPE_Q5_1;
            }
            else if (strcasecmp("Q5_0", tp) == 0)
            {
                cp.type_v = GGML_TYPE_Q5_0;
            }
            else if (strcasecmp("IQ4_NL", tp) == 0)
            {
                cp.type_v = GGML_TYPE_IQ4_NL;
            }
            else if (strcasecmp("Q4_1", tp) == 0)
            {
                cp.type_v = GGML_TYPE_Q4_1;
            }
            else if (strcasecmp("Q4_0", tp) == 0)
            {
                cp.type_v = GGML_TYPE_Q4_0;
            }
            else if (strcasecmp("F16", tp) != 0)
                RP_THROW(ctx, "cacheType must be a String ('q8' or 'fp16)");
            strcpy(typev_str, tp);
        }
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "nBatch"))
            cp.n_batch = (uint32_t)REQUIRE_UINT(ctx, -1, "nBatch must be a positive integer");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "nUBatch"))
            cp.n_ubatch = (uint32_t)REQUIRE_UINT(ctx, -1, "nUBatch must be a positive integer");
        duk_pop(ctx);

        /*
        if (duk_get_prop_string(ctx, obj_idx, "nSeqMax"))
            cp.n_seq_max = (uint32_t)REQUIRE_UINT(ctx, -1, "nSeqMax must be a positive integer");
        duk_pop(ctx);
        */

        if (duk_get_prop_string(ctx, obj_idx, "threads"))
            cp.n_threads = (int32_t)REQUIRE_UINT(ctx, -1, "threads must be a positive integer");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "threadsBatch"))
            cp.n_threads_batch = (int32_t)REQUIRE_UINT(ctx, -1, "threadsBatch must be a positive integer");
        duk_pop(ctx);

        // enums (pass as integers)
        if (duk_get_prop_string(ctx, obj_idx, "ropeScalingType"))
            cp.rope_scaling_type =
                (enum llama_rope_scaling_type)REQUIRE_UINT(ctx, -1, "ropeScalingType must be an integer enum");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "poolingType"))
            cp.pooling_type = (enum llama_pooling_type)REQUIRE_UINT(ctx, -1, "poolingType must be an integer enum");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "flashAttn"))
            cp.flash_attn_type = REQUIRE_BOOL(ctx, -1, "flashAttn must be a Boolean");
        duk_pop(ctx);

        // RoPE / YaRN
        if (duk_get_prop_string(ctx, obj_idx, "ropeFreqBase"))
            cp.rope_freq_base = (float)REQUIRE_NUMBER(ctx, -1, "ropeFreqBase must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "ropeFreqScale"))
            cp.rope_freq_scale = (float)REQUIRE_NUMBER(ctx, -1, "ropeFreqScale must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "yarnExtFactor"))
            cp.yarn_ext_factor = (float)REQUIRE_NUMBER(ctx, -1, "yarnExtFactor must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "yarnAttnFactor"))
            cp.yarn_attn_factor = (float)REQUIRE_NUMBER(ctx, -1, "yarnAttnFactor must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "yarnBetaFast"))
            cp.yarn_beta_fast = (float)REQUIRE_NUMBER(ctx, -1, "yarnBetaFast must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "yarnBetaSlow"))
            cp.yarn_beta_slow = (float)REQUIRE_NUMBER(ctx, -1, "yarnBetaSlow must be a number");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "yarnOrigCtx"))
            cp.yarn_orig_ctx = (uint32_t)REQUIRE_UINT(ctx, -1, "yarnOrigCtx must be a positive integer");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "defragThold"))
            cp.defrag_thold = (float)REQUIRE_NUMBER(ctx, -1, "defragThold must be a number");
        duk_pop(ctx);

        /*
        // cache types
        if (duk_get_prop_string(ctx, obj_idx, "typeK"))
            cp.type_k = (enum ggml_type)REQUIRE_UINT(ctx, -1, "typeK must be an integer enum");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "typeV"))
            cp.type_v = (enum ggml_type)REQUIRE_UINT(ctx, -1, "typeV must be an integer enum");
        duk_pop(ctx);

        // booleans
        if (duk_get_prop_string(ctx, obj_idx, "embeddings"))
            cp.embeddings = REQUIRE_BOOL(ctx, -1, "embeddings must be boolean");
        duk_pop(ctx);
        */

        // accept either "offloadKQV" or "offloadKqv"
        if (duk_get_prop_string(ctx, obj_idx, "offloadKQV") || duk_get_prop_string(ctx, obj_idx, "offloadKqv"))
            cp.offload_kqv = REQUIRE_BOOL(ctx, -1, "offloadKQV must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "noPerf"))
            cp.no_perf = REQUIRE_BOOL(ctx, -1, "noPerf must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "opOffload"))
            cp.op_offload = REQUIRE_BOOL(ctx, -1, "opOffload must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "swaFull"))
            cp.swa_full = REQUIRE_BOOL(ctx, -1, "swaFull must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "kvUnified"))
            cp.kv_unified = REQUIRE_BOOL(ctx, -1, "kvUnified must be boolean");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "grpAttnN"))
            ga_n = (uint32_t)REQUIRE_UINT(ctx, -1, "grpAttnN must be a positive integer");
        duk_pop(ctx);

        if (duk_get_prop_string(ctx, obj_idx, "grpAttnW"))
            ga_w = (uint32_t)REQUIRE_UINT(ctx, -1, "grpAttnW must be a positive integer");
        duk_pop(ctx);

        if (ga_w % ga_n)
            RP_THROW(ctx, "Error loading model: bad values: grpAttnW must be a multiple of grpAttnN");
    }

    // embeddings is elsewhere (below)
    cp.embeddings = false;

    if (!cp.n_ctx)
    {
        uint64_t base_bytes = 0, kv_bpt = 0;
        uint32_t n_ctx_suggest =
            suggest_n_ctx_from_mem(lmodel, cp.op_offload, cp.type_k, cp.type_v,
                                   /*headroom_frac=*/0.75,          // leave 75% slack
                                   /*headroom_bytes=*/512ull << 20, // +512 MiB slack
                                   /*cpu_free_cap_frac=*/0.50,      // if KV on CPU, use up to 50% of free RAM
                                   &base_bytes, &kv_bpt);

        //printf("suggest = %d\n", (int)n_ctx_suggest);
        // int n_ctx_train = llama_n_ctx_train(model);   // if available in your llama.h
        if (n_ctx_suggest == 0)
            n_ctx_suggest = 2048; // fallback
        if (n_ctx_suggest > (uint32_t)n_train)
            n_ctx_suggest = (uint32_t)n_train;

        /*
        fprintf(stderr, "KV bytes/token ~ %llu, base=%llu, suggested n_ctx=%u\n",
                (unsigned long long)kv_bpt,
                (unsigned long long)base_bytes,
                n_ctx_suggest);
        */
        cp.n_ctx = n_ctx_suggest;

        // cp.offload_kv   = kv_on_gpu;            // true: keep KV on GPU; false: pin in system RAM
    }

    // default ubatch to n_ctx so a full window fits in one micro-batch
    // cp.n_ubatch = cp.n_ctx;

    lctx = llama_init_from_model(lmodel, cp);
    if (!lctx)
        RP_THROW(ctx, "rampart-llama-cpp:initGen - Failed to create llama context");

    // Info
    const struct llama_vocab *vocab = llama_model_get_vocab(lmodel);
    if (!vocab)
        RP_THROW(ctx, "rampart-llama-cpp:initText - Failed to get vocab from model");

    const int32_t n_vocab = (int)llama_vocab_n_tokens(vocab);
    const uint32_t n_ctx = llama_n_ctx(lctx);

    rp_llama_info *info = NULL;
    CALLOC(info, sizeof(rp_llama_info));

    info->thr = get_current_thread();
    info->lctx = lctx;
    info->lmodel = lmodel;
    info->vocab = vocab;
    info->n_ctx = n_ctx;
    info->n_vocab = n_vocab;
    info->ga_n = (uint32_t)ga_n;
    info->ga_w = (uint32_t)ga_w;
    info->store_last = store_last;
    info->cur_pos = 0;
    info->cp = cp;

    duk_push_object(ctx); // return object

    duk_push_int(ctx, (int)n_ctx);
    duk_rp_put_prop_string_ro(ctx, -2, "nCtx");

    duk_push_int(ctx, (int)n_vocab);
    duk_rp_put_prop_string_ro(ctx, -2, "nVocab");

    duk_push_int(ctx, 0);
    duk_rp_put_prop_string_ro(ctx, -2, "position");

    char *s = typek_str;
    while (*s)
    {
        *s = toupper(*s);
        s++;
    }
    s = typev_str;
    while (*s)
    {
        *s = toupper(*s);
        s++;
    }

    duk_push_string(ctx, typek_str);
    duk_rp_put_prop_string_ro(ctx, -2, "kCacheType");

    duk_push_string(ctx, typev_str);
    duk_rp_put_prop_string_ro(ctx, -2, "vCacheType");

    duk_push_pointer(ctx, info);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("rp_llama_info"));

    /*
        maybe later
        duk_push_c_function(ctx, gen_tokenize,   1);
        duk_put_prop_string(ctx, -2, "tokenize");

        duk_push_c_function(ctx, gen_detokenize, 1);
        duk_put_prop_string(ctx, -2, "detokenize");
    */

    duk_push_c_function(ctx, gen_free, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    duk_push_c_function(ctx, gen_free, 1);
    duk_set_finalizer(ctx, -2);

    duk_push_c_function(ctx, get_last_gen, 0);
    duk_put_prop_string(ctx, -2, "getLast");

    duk_push_c_function(ctx, gen_predict, 2);
    duk_put_prop_string(ctx, -2, "predict");

    duk_push_c_function(ctx, gen_predict_async, 3);
    duk_put_prop_string(ctx, -2, "predictAsync");

    return 1;
}

// LLAMA.CPP EMBEDDING MODELS

static duk_ret_t emb_free(duk_context *ctx)
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

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("rerank_toks")))
    {
        void *toks = duk_get_pointer(ctx, -1);
        if(toks)
            free(toks);
    }
    duk_pop(ctx);

    llama_free(lctx);

    llama_model_free(lmodel);

    duk_del_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("model"));
    duk_del_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("llama_ctx"));

    duk_push_true(ctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("destroyed"));

    return 0;
}

static struct llama_context *new_embed_context(duk_context *ctx, struct llama_model *lmodel, duk_idx_t opts_idx)
{
    struct llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cp.n_threads = 1;
    cp.n_ctx = 0;
    cp.n_ubatch = 0;
    cp.n_threads_batch = -1;

    if (opts_idx > -1)
    {
        // nctx
        if (duk_get_prop_string(ctx, opts_idx, "nctx"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "Option nctx must be a Number");
            cp.n_ctx = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // ubatch
        if (duk_get_prop_string(ctx, opts_idx, "ubatch"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "Option ubatch must be a Number");
            cp.n_ubatch = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // nthreads
        if (duk_get_prop_string(ctx, opts_idx, "nthreads"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "Option nthreads must be a Number");
            cp.n_threads = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // nthreads_batch
        if (duk_get_prop_string(ctx, opts_idx, "nthreads_batch"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "Option nthreads_batch must be a Number");
            cp.n_threads_batch = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // embeddings
        if (duk_get_prop_string(ctx, opts_idx, "embeddings"))
        {
            if (!duk_is_boolean(ctx, -1))
                RP_THROW(ctx, "Option embeddings must be a Boolean");
            cp.embeddings = duk_get_boolean(ctx, -1);
        }
        duk_pop(ctx);

        // pooling: "mean" | "cls" | "last" | number (enum)
        if (duk_get_prop_string(ctx, opts_idx, "pooling"))
        {
            if (duk_is_string(ctx, -1))
            {
                const char *s = duk_get_string(ctx, -1);
                if (strcmp(s, "mean") == 0)
                    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                else if (strcmp(s, "cls") == 0)
                    cp.pooling_type = LLAMA_POOLING_TYPE_CLS;
                else if (strcmp(s, "last") == 0)
                    cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
                else
                    RP_THROW(ctx, "Option pooling must be 'mean', 'cls', or 'last'");
            }
            else if (duk_is_number(ctx, -1))
            {
                cp.pooling_type = (enum llama_pooling_type)duk_get_int(ctx, -1);
            }
            else
            {
                RP_THROW(ctx, "Option pooling must be a String or Number");
            }
        }
        duk_pop(ctx);
    }

    // If user didn't specify nctx or ubatch, set both to model's max.
    if (cp.n_ctx <= 0)
    {
        int n_train = llama_model_n_ctx_train(lmodel);

        // don't go crazy and run out of memory, unless we explicitly set n_ctx - then it's user error
        if (n_train > 8192)
            n_train = 8192;

        if (n_train > 0)
            cp.n_ctx = n_train;
    }

    if (cp.n_ubatch <= 0)
    {
        // default ubatch to n_ctx so a full window fits in one micro-batch
        cp.n_ubatch = cp.n_ctx > 0 ? cp.n_ctx : 0;
    }

    // batch must be equal or greater, to prevent clamping
    cp.n_batch = cp.n_ubatch;

    void *cp_buf = duk_push_fixed_buffer(ctx, sizeof(struct llama_context_params));
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("cp_buf"));
    memcpy(cp_buf, &cp, sizeof(struct llama_context_params));

    // printf("setting ctx at %d\n", cp.n_ctx);
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

        duk_push_pointer(ctx, lctx);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

        duk_push_int(ctx, curthr);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));
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
        batch.logits = false;

        if (llama_encode(lctx, batch) != 0)
        {
            llama_batch_free(batch);
            free(toks);
            RP_THROW(ctx, "llama_encode failed on chunk %d (tokens %d..%d)", k, start, start + n - 1);
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

static duk_ret_t llamacpp_init_embed(duk_context *ctx)
{
    const char *model = REQUIRE_STRING(ctx, 0, "init: argument 1 must be a string");
    duk_idx_t obj_idx = -1;

    if (duk_is_object(ctx, 1))
        obj_idx = 1;

    duk_push_object(ctx); // return object

    struct llama_model *lmodel = NULL;
    struct llama_context *lctx = NULL;

    struct llama_model_params mp = llama_model_default_params();

    lmodel = llama_model_load_from_file(model, mp);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:init - Could not load ggml file '%s': %s", model, strerror(errno));

    int vec_dim = llama_model_n_embd(lmodel);

    if (vec_dim <= 0)
        RP_THROW(ctx, "rampart-llama-cpp:init - Internal error getting vector dimensions");

    lctx = new_embed_context(ctx, lmodel, obj_idx);

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

    // Use llama_encode (not llama_decode) for embedding/reranking tasks
    int ret = llama_encode(lctx, batch);

    llama_batch_free(batch);
    free(tokens);

    if (ret != 0)
        RP_THROW(ctx, "llama_encode failed for reranking");

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

        duk_push_pointer(ctx, lctx);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

        duk_push_int(ctx, curthr);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));
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

    // Set up model parameters
    struct llama_model_params mp = llama_model_default_params();

    lmodel = llama_model_load_from_file(model, mp);

    if (!lmodel)
        RP_THROW(ctx, "rampart-llama-cpp:initRerank - Could not load ggml file '%s': %s", model, strerror(errno));

    // Set up context parameters for reranking
    struct llama_context_params cp = llama_context_default_params();

    // Enable embeddings mode and set pooling to RANK
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_RANK;
    cp.n_threads = -1;
    cp.n_ctx = 0;        // will be set later if not specified
    cp.n_ubatch = 0;     // will be set later if not specified
    cp.n_threads_batch = -1;

    // Get options from JavaScript object (matching user's existing option names)
    if (obj_idx >= 0)
    {
        // nctx (context size)
        if (duk_get_prop_string(ctx, obj_idx, "nctx"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "option nctx must be a Number");
            cp.n_ctx = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // ubatch (micro-batch size)
        if (duk_get_prop_string(ctx, obj_idx, "ubatch"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "option ubatch must be a Number");
            cp.n_ubatch = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // nthreads - not really relevant here?
        if (duk_get_prop_string(ctx, obj_idx, "nthreads"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "option nthreads must be a Number");
            cp.n_threads = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

        // nthreads_batch
        if (duk_get_prop_string(ctx, obj_idx, "nthreads_batch"))
        {
            if (!duk_is_number(ctx, -1))
                RP_THROW(ctx, "option nthreads_batch must be a Number");
            cp.n_threads_batch = duk_get_int(ctx, -1);
        }
        duk_pop(ctx);

    }

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
        CALLOC(cap, sizeof(struct llog_cap));
        pthread_mutex_init(&cap->mutex, NULL);
        llama_log_set(llamacpp_logger, cap);

        duk_push_pointer(ctx, cap);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("caplog"));

        isloaded = 1;
    }

    duk_push_c_function(ctx, llamacpp_init_embed, 2);
    duk_put_prop_string(ctx, -2, "initEmbed");

    duk_push_c_function(ctx, llamacpp_init_gen, 2);
    duk_put_prop_string(ctx, -2, "initGen");

    duk_push_c_function(ctx, llamacpp_init_rerank, 2);
    duk_put_prop_string(ctx, -2, "initRerank");

    duk_push_c_function(ctx, getlog, 0);
    duk_put_prop_string(ctx, -2, "getLog");

    duk_push_c_function(ctx, resetlog, 0);
    duk_put_prop_string(ctx, -2, "resetLog");

    add_exit_func(close_llama_on_exit, NULL);

    return 1;
}

