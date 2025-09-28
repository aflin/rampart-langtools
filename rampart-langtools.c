#define _GNU_SOURCE
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

#include "llama.h"
#include "rampart-langtools.h"
#include "rampart.h"
#include "spm_c_wrapper.h"
#include <c_api/AutoTune_c.h>
#include <c_api/Clustering_c.h>
#include <c_api/IndexBinaryIVF_c.h>
#include <c_api/IndexBinary_c.h>
#include <c_api/IndexFlat_c.h>
#include <c_api/IndexIVFFlat_c.h>
#include <c_api/IndexIVF_c.h>
#include <c_api/IndexLSH_c.h>
#include <c_api/IndexPreTransform_c.h>
#include <c_api/IndexReplicas_c.h>
#include <c_api/IndexScalarQuantizer_c.h>
#include <c_api/IndexShards_c.h>
#include <c_api/Index_c.h>
#include <c_api/MetaIndexes_c.h>
#include <c_api/VectorTransform_c.h>
#include <c_api/clone_index_c.h>
#include <c_api/error_c.h>
#include <c_api/index_factory_c.h>
#include <c_api/index_io_c.h>

#include "convert_vec.c"

/* fp32 -> fp16 (portable) */
static inline uint16_t f32_to_f16(float x)
{
    uint32_t f;
    memcpy(&f, &x, sizeof f);
    uint32_t s = (f >> 31) & 1u;
    int32_t e = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t m = f & 0x7FFFFF;
    uint16_t h;
    if (e <= 0)
    {
        if (e < -10)
            h = (uint16_t)(s << 15);
        else
        {
            m |= 0x800000u;
            uint32_t sh = (uint32_t)(14 - e);
            uint32_t mant = m >> (sh + 13);
            if ((m >> (sh + 12)) & 1u)
                mant++;
            h = (uint16_t)((s << 15) | mant);
        }
    }
    else if (e >= 31)
        h = (uint16_t)((s << 15) | (0x1Fu << 10));
    else
    {
        uint16_t mant = (uint16_t)(m >> 13);
        if (m & 0x00001000u)
            mant++;
        h = (uint16_t)((s << 15) | ((uint16_t)e << 10) | mant);
    }
    return h;
}

/*
static inline float f16_to_float(uint16_t h)
{
    uint16_t hs = h & 0x8000u;
    uint16_t he = h & 0x7C00u;
    uint16_t hm = h & 0x03FFu;
    uint32_t sign = ((uint32_t)hs) << 16;
    uint32_t f;

    if (he == 0x7C00u)
    { // Inf/NaN
        f = sign | 0x7F800000u | ((uint32_t)hm << 13);
    }
    else if (he == 0)
    {
        if (hm == 0)
        {
            f = sign; // zero
        }
        else
        {
            int shift = 0;
            while ((hm & 0x0400u) == 0)
            {
                hm <<= 1;
                shift++;
            }
            hm &= 0x03FFu;
            int32_t exp = (127 - 15 - shift);
            f = sign | ((uint32_t)exp << 23) | ((uint32_t)hm << 13);
        }
    }
    else
    {
        int32_t exp = ((he >> 10) - 15 + 127);
        f = sign | ((uint32_t)exp << 23) | ((uint32_t)hm << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static duk_ret_t f16_to_numbers(duk_context *ctx)
{
    duk_double_t res;
    duk_size_t sz=0, i=0;
    uint16_t *buf = (uint16_t *)REQUIRE_BUFFER_DATA(ctx, 0, &sz, "f16BufToNumbers - buffer required as argument");

    sz/=2;
    duk_push_array(ctx);
    for (; i<sz; i++)
    {
        res = (duk_double_t) f16_to_float(buf[i]);
        duk_push_number(ctx, res);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }


    return 1;
}
*/

static struct llama_context *new_embed_context(duk_context *ctx, struct llama_model *lmodel, duk_idx_t opts_idx)
{
    struct llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cp.n_threads = 1;
    cp.n_ctx = 0;
    cp.n_ubatch = 0;
    // optional extras with safe defaults:
    cp.n_threads_batch = 0;

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
        if (n_train > 0)
            cp.n_ctx = n_train;
    }
    if (cp.n_ubatch <= 0)
    {
        // default ubatch to n_ctx so a full window fits in one micro-batch
        cp.n_ubatch = cp.n_ctx > 0 ? cp.n_ctx : 0;
    }
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

    int curthr = get_thread_num();
    // get a new context if in a new thread.  Model stays the same.
    if (curthr != thrno)
    {
        // FIXME: save and retrieve options.

        printf("loading new context\n");
        lctx = new_embed_context(ctx, lmodel, -1);

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
    for (int start = 0; start < nw; start += stride, ++k)
    {
        int n = nw - start;
        if (n > chunk_tokens)
            n = chunk_tokens;

        // encoder path requires: n_tokens <= n_ubatch
        if (n > n_ubatch)
        {
            free(toks);
            RP_THROW(ctx, "chunk too large for micro-batch (n=%d > n_ubatch=%d). Increase cp.n_ubatch.", n, n_ubatch);
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

        if (llama_encode(lctx, batch) != 0)
        {
            llama_batch_free(batch);
            free(toks);
            RP_THROW(ctx, "llama_decode failed on chunk %d (tokens %d..%d)", k, start, start + n - 1);
            return 0;
        }
        llama_batch_free(batch);

        // read pooled embedding
        const enum llama_pooling_type p = llama_pooling_type(lctx);
        const float *emb =
            (p == LLAMA_POOLING_TYPE_NONE) ? llama_get_embeddings_ith(lctx, n - 1) : llama_get_embeddings_seq(lctx, 0);

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
            vec_fp32_to_fp16_buf(v, out, vec_dim);
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
            vec_fp32_to_fp16_buf(avgvec, out, vec_dim);
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

    duk_push_pointer(ctx, lctx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("llama_ctx"));

    duk_push_int(ctx, vec_dim);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("vec_dim"));

    duk_push_int(ctx, (int)get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("ctx_thread"));

    duk_push_c_function(ctx, embed_text_to_buf32, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp32Buf");

    duk_push_c_function(ctx, embed_text_to_buf16, 1);
    duk_put_prop_string(ctx, -2, "embedTextToFp16Buf");

    duk_push_c_function(ctx, embed_text_to_numbers, 1);
    duk_put_prop_string(ctx, -2, "embedTextToNumbers");

    return 1;
}

struct llog_cap
{
    char *buf;
    size_t len;
    size_t alloc;
};

static void llamacpp_logger(enum ggml_log_level level, const char *text, void *ud)
{
    (void)level; // or filter by level
    struct llog_cap *cap = (struct llog_cap *)ud;
    size_t len = strlen(text);

    cap->len += len;
    if (cap->len + 1 > cap->alloc)
    {
        if (cap->alloc == 0)
        {
            cap->alloc = cap->len > 1023 ? cap->len * 2 : 1024;
            REMALLOC(cap->buf, cap->alloc);
            cap->buf[0] = '\0';
        }
        else
        {
            cap->alloc = (3 * cap->alloc) / 2;
            if (cap->len + 1 > cap->alloc)
                cap->alloc = cap->len * 2;
            REMALLOC(cap->buf, cap->alloc);
        }
    }

    cap->len += len;
    strcat(cap->buf, text);
}

static duk_ret_t getlog(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("caplog"));
    struct llog_cap *caplog = duk_get_pointer(ctx, -1);
    if (caplog)
    {
        if (caplog->buf)
            duk_push_string(ctx, caplog->buf);
        else
            duk_push_string(ctx, "");
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

    free(caplog->buf);
    caplog->buf = NULL;
    caplog->alloc = 0;
    caplog->len = 0;

    return 0;
}

static duk_ret_t open_llama(duk_context *ctx)
{
    struct llog_cap *cap = NULL;
    static int isloaded = 0;

    /* the return object */
    duk_push_object(ctx);

    if (!isloaded)
    {
        CALLOC(cap, sizeof(struct llog_cap));

        llama_log_set(llamacpp_logger, cap);

        duk_push_pointer(ctx, cap);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("caplog"));

        isloaded = 1;
    }

    duk_push_c_function(ctx, llamacpp_init_embed, 2);
    duk_put_prop_string(ctx, -2, "initEmbed");

    // duk_push_c_function(ctx, f16_to_numbers, 1);
    // duk_put_prop_string(ctx, -2, "fp16BufToNumbers");

    duk_push_c_function(ctx, getlog, 0);
    duk_put_prop_string(ctx, -2, "getLog");

    duk_push_c_function(ctx, resetlog, 0);
    duk_put_prop_string(ctx, -2, "resetLog");

    return 1;
}

/* *******************************************************
                SENTENCEPIECE
   ******************************************************* */

// U+2581 LOWER ONE EIGHTH BLOCK (▁), used by SentencePiece to mark word starts.
static const char *SP_UNDER = "\xE2\x96\x81";

static duk_ret_t detok_from_pieces(duk_context *ctx, duk_idx_t arridx)
{

    size_t i = 0, n = (size_t)duk_get_length(ctx, arridx);
    size_t mlen = 0;
    int wrote_any = 0;
    const char *p;

    for (; i < n; i++)
    {
        duk_get_prop_index(ctx, arridx, (duk_uarridx_t)i);
        mlen += (size_t)duk_get_length(ctx, -1);
        duk_pop(ctx);
    }

    if (!mlen)
    {
        duk_push_string(ctx, "");
        return 1;
    }

    char *out = NULL;
    REMALLOC(out, mlen + 1);
    out[0] = '\0';

    for (i = 0; i < n; i++)
    {
        duk_get_prop_index(ctx, arridx, (duk_uarridx_t)i);
        p = REQUIRE_STRING(ctx, -1, "sentencepiece.decode - array[%lu] is not a string", i);
        duk_pop(ctx);
        size_t plen = strlen(p);
        if (plen >= 3 && memcmp(p, SP_UNDER, 3) == 0)
        {
            // Leading underline indicates a space before the token (word start)
            if (wrote_any)
                strcat(out, " ");
            // append the remainder (without the marker)
            strncat(out, p + 3, plen - 3);
            wrote_any = 1;
        }
        else
        {
            // No marker; typically continuation of previous piece (subword)
            strcat(out, p);
            wrote_any = 1;
        }
    }
    duk_push_string(ctx, out);
    free(out);
    return 1;
}

duk_ret_t detok_from_piecestring(duk_context *ctx, duk_idx_t stridx)
{

    // First pass: compute output length
    int wrote_any = 0;
    const char *piecestring = duk_get_string(ctx, stridx);
    const char *tok = piecestring;
    size_t plen, mlen = strlen(tok) + 1;
    const char *end;
    char *out = NULL;
    REMALLOC(out, mlen + 1);

    wrote_any = 0;
    tok = piecestring;
    while (*tok)
    {
        end = strchr(tok, ' ');
        plen = (end ? (size_t)(end - tok) : strlen(tok));

        if (plen >= 3 && memcmp(tok, SP_UNDER, 3) == 0)
        {
            if (wrote_any)
                strcat(out, " ");
            strncat(out, tok + 3, plen - 3);
            wrote_any = 1;
        }
        else
        {
            strncat(out, tok, plen);
            wrote_any = 1;
        }

        if (!end)
            break;
        tok = end + 1;
        while (*tok == ' ')
            tok++;
    }
    duk_push_string(ctx, out);
    free(out);
    return 1;
}

static duk_ret_t sp_decode(duk_context *ctx)
{
    if (duk_is_array(ctx, 0))
        return detok_from_pieces(ctx, 0);
    if (duk_is_string(ctx, 0))
        return detok_from_piecestring(ctx, 0);

    RP_THROW(ctx, "sentencepiece.decode - argument must be an Array or String");
    return 0;
}

static duk_ret_t sp_encode(duk_context *ctx)
{
    const char *text = REQUIRE_STRING(ctx, 0, "sentencepiece.encode - argument must be a String (text to encode)");

    int retArray = 1;

    const char **pieces = NULL;
    size_t n_pieces = 0;

    if (!duk_is_undefined(ctx, 1))
        retArray = !REQUIRE_BOOL(ctx, 1, "sentencepiece.encode - second argument must be a Boolean (return String)");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("spmProcessor"));
    spm_processor_t *sp = duk_get_pointer(ctx, -1);

    if (!sp)
        RP_THROW(ctx, "sentencepiece.encode - Internal error getting sentencepiece processor");

    if (spm_encode_as_pieces(sp, text, &pieces, &n_pieces) != 0)
        RP_THROW(ctx, "sentencepiece.encode -Encode failed");

    if (retArray)
    {
        duk_push_array(ctx);

        for (size_t i = 0; i < n_pieces; i++)
        {
            duk_push_string(ctx, pieces[i]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
    }
    else
    {
        char *s, *str = NULL;
        size_t strsz = 0;
        int pos = 0;
        for (size_t i = 0; i < n_pieces; i++)
            strsz += 1 + strlen(pieces[i]);

        CALLOC(str, strsz);
        s = str;
        for (size_t i = 0; i < n_pieces; i++)
        {
            strcpy(s + pos, pieces[i]);
            pos += strlen(pieces[i]);
            *(s + pos++) = ' ';
        }
        *(-1 + s + pos) = '\0';
        duk_push_string(ctx, str);
    }
    return 1;
}

static duk_ret_t sp_init(duk_context *ctx)
{
    const char *model =
        REQUIRE_STRING(ctx, 0, "sentencepiece.init - first argument must be a String (path to model file)");

    duk_push_object(ctx);

    spm_processor_t *sp = spm_create();

    if (!sp)
        RP_THROW(ctx, "sentencepiece.init - failed to create sentencepiece object");

    if (spm_load(sp, model) != 0)
    {
        spm_free(sp);
        RP_THROW(ctx, "Failed to load model: %s\n", model);
    }

    duk_push_pointer(ctx, sp);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("spmProcessor"));
    duk_push_c_function(ctx, sp_encode, 2);
    duk_put_prop_string(ctx, -2, "encode");
    return 1;
}

static void open_sentencepiece(duk_context *ctx)
{
    duk_push_object(ctx);
    duk_push_c_function(ctx, sp_init, 2);
    duk_put_prop_string(ctx, -2, "init");
    duk_push_c_function(ctx, sp_decode, 1);
    duk_put_prop_string(ctx, -2, "decode");
}

/* **************************************************
           FAISS
   ************************************************** */

int faiss_add_one(FaissIndex *idx, idx_t id, float *v, int dim, const char **err)
{
    if (!idx || !v)
    {
        if (err)
            *err = "Null index or vector pointer";
        return -1;
    }

    // probably can get rid of this, as we check below.
    int d = faiss_Index_d(idx);
    if (dim != d)
    {
        if (err)
            *err = "Vector dimension does not match index dimension";
        return -1;
    }

    int rc = faiss_Index_add_with_ids(idx, 1, v, &id);
    if (rc != 0)
    {
        if (err)
        {
            *err = faiss_get_last_error();
            if (!*err)
                *err = "faiss_Index_add_with_ids failed";
        }
        return -1;
    }

    if (err)
        *err = NULL;
    return 0;
}

static duk_ret_t add_fp32(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    idx_t id = (idx_t)REQUIRE_UINT(ctx, 0, "addFp32 requires a positive integer (id) as its first argument");
    float *v = REQUIRE_BUFFER_DATA(ctx, 1, &sz, "addFp32 requires a buffer as its second argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 4 != dim)
        RP_THROW(ctx, "addFp32 - buffer is %lu long, should be %lu (4 * %lu dimensions)", sz, dim * 4, dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);

    if (!idx)
        RP_THROW(ctx, "faiss.train - Internal error getting index handle");

    const char *err = NULL;
    if (faiss_add_one(idx, id, v, (int)dim, &err) == -1)
        RP_THROW(ctx, "addFp32 - %s", err);

    return 0;
}

static duk_ret_t add_fp16(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    idx_t id = (idx_t)REQUIRE_UINT(ctx, 0, "addFp16 requires a positive integer (id) as its first argument");
    const uint16_t *v16 = REQUIRE_BUFFER_DATA(ctx, 1, &sz, "addFp16 requires a buffer as its second argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 2 != dim)
        RP_THROW(ctx, "addFp32 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);

    if (!idx)
        RP_THROW(ctx, "faiss.train - Internal error getting index handle");

    float *v = vec_fp16_to_fp32(v16, (size_t)dim);

    const char *err = NULL;
    if (faiss_add_one(idx, id, v, (int)dim, &err) == -1)
    {
        free(v);
        RP_THROW(ctx, "addFp16 - %s", err);
    }
    free(v);

    return 0;
}

/* Search top-k with optional IVF/HNSW parameters.
 * - If nprobe > 0, uses IVF search params.
 * - Else if efSearch > 0, uses HNSW search params.
 * - Else uses default index settings.
 *
 * Returns malloc'ed array of k labels (idx_t) on success; caller must free().
 * On error returns NULL and sets *err to faiss_get_last_error() or a static message.
 */
static idx_t *faiss_search_topk_ids_params(FaissIndex *idx, const float *q, /* query vector */
                                           idx_t k,                         /* desired number of res */
                                           int nprobe,                      /* set >0 for IVF */
                                           float **Dret,                    /* if not null, set distances to this */
                                           const char **err)
{
    if (err)
        *err = NULL;
    if (!idx || !q || k <= 0)
    {
        if (err)
            *err = "Invalid args: idx/q NULL or k <= 0";
        return NULL;
    }

    float *D = (float *)malloc((size_t)k * sizeof(float));
    idx_t *I = (idx_t *)malloc((size_t)k * sizeof(idx_t));

    if (!D || !I)
    {
        free(D);
        free(I);
        if (err)
            *err = "Out of memory";
        return NULL;
    }

    /* Build one search-params object if requested */
    const FaissSearchParameters *sp = NULL;
    FaissSearchParametersIVF *sp_ivf = NULL;

    int rc = 0;

    if (nprobe > 0)
    {
        if (faiss_SearchParametersIVF_new(&sp_ivf) == 0)
        {
            faiss_SearchParametersIVF_set_nprobe(sp_ivf, nprobe);
            sp = (const FaissSearchParameters *)sp_ivf;
        }
        /* if allocation failed, we just fall back to defaults below */
    }

    if (sp)
    {
        rc = faiss_Index_search_with_params(idx, 1, q, k, sp, D, I);
        /* If the params are incompatible with the index type, try plain search as a fallback */
        if (rc != 0)
        {
            rc = faiss_Index_search(idx, 1, q, k, D, I);
        }
    }
    else
    {
        rc = faiss_Index_search(idx, 1, q, k, D, I);
    }

    /* free params objects (they are independent of the search results) */
    if (sp_ivf)
        faiss_SearchParametersIVF_free(sp_ivf);

    if (rc != 0)
    {
        if (err)
            *err = faiss_get_last_error();
        free(D);
        free(I);
        return NULL;
    }

    if (Dret)
        *Dret = D;
    else
        free(D);

    return I;
}

static void do_search_(duk_context *ctx, const float *v, size_t dim, int is16)
{
    duk_uarridx_t aidx = 0;
    idx_t i = 0, k = 10;
    const char *fname = "searchFp32";

    if (is16)
        fname = "searchFp16";

    if (!duk_is_undefined(ctx, 1))
        k = (idx_t)REQUIRE_UINT(ctx, 1, "%s requires a positive integer (nResults) as its second argument", fname);

    // this is still at -1
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);

    const char *err = NULL;
    float *distances = NULL;

    idx_t *ids = faiss_search_topk_ids_params(idx, v, k, 0, &distances, &err);

    if (err)
        RP_THROW(ctx, "%s - Error: %s", fname, err);
    else if (!ids)
        RP_THROW(ctx, "%s - unknown error while searching index", fname);

    duk_push_array(ctx);
    for (; i < k; i++)
    {
        if (ids[i] == -1)
            break;

        duk_push_object(ctx);
        duk_push_number(ctx, (double)ids[i]);
        duk_put_prop_string(ctx, -2, "id");
        duk_push_number(ctx, (double)distances[i]);
        duk_put_prop_string(ctx, -2, "distance");

        duk_put_prop_index(ctx, -2, aidx++);
    }
    free(ids);
    if(distances)
        free(distances);
}

static duk_ret_t do_search_fp16(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const uint16_t *v16 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addFp16 requires a buffer as its first argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 2 != dim)
        RP_THROW(ctx, "searchFp16 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    float *v = vec_fp16_to_fp32(v16, (size_t)dim);

    do_search_(ctx, v, dim, 1);

    free(v);

    return 1;
}

static duk_ret_t do_search_fp32(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const float *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "searchFp32 requires a buffer as its first argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 4 != dim)
        RP_THROW(ctx, "searchFp32 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    do_search_(ctx, v, dim, 1);

    return 1;
}

static duk_ret_t save_index(duk_context *ctx)
{
    const char *fname = REQUIRE_STRING(ctx, 0, "faiss.save - argument must be a filename");

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    int rc = faiss_write_index_fname(idx, fname);

    if (rc)
        RP_THROW(ctx, "Failed to save file - %s", faiss_get_last_error());
    return 0;
}

void put_prop_readonly(duk_context *ctx, duk_idx_t idx, const char *s)
{
    idx = duk_normalize_index(ctx, idx);
    duk_push_string(ctx, s);
    duk_pull(ctx, -2);
    duk_def_prop(ctx, idx, DUK_DEFPROP_HAVE_VALUE | DUK_DEFPROP_SET_ENUMERABLE | DUK_DEFPROP_CLEAR_WRITABLE);
}

/* training */
int write_vector_to_file(FILE *fh, const float *vec, int dim)
{
    if (!fh || !vec || dim <= 0)
    {
        errno = EINVAL;
        return -1;
    }

    size_t nw = fwrite(vec, sizeof(float), (size_t)dim, fh);
    if (nw != (size_t)dim)
    {
        return -1; /* errno is set by fwrite */
    }
    return 0;
}

static duk_ret_t add_trainvec_fp32(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    FILE *fh = NULL;
    float *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addTrainingFp32 requires a Buffer as an argument");
    int nrows = 0;

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    duk_get_prop_string(ctx, -1, "rowsAdded");
    nrows = duk_get_int_default(ctx, -1, 0);
    duk_pop(ctx);

    if (sz / 4 != dim)
        RP_THROW(ctx, "addFp32 - buffer is %lu long, should be %lu (4 * %lu dimensions)", sz, dim * 4, dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "addTrainingFp32 - Internal error getting file handle");

    if (write_vector_to_file(fh, v, dim) == -1)
        RP_THROW(ctx, "addTrainingFp32 - error writing to tmpfile - %s", strerror(errno));

    nrows++;
    duk_push_int(ctx, nrows);
    duk_put_prop_string(ctx, -2, "rowsAdded");

    return 0;
}

static duk_ret_t add_trainvec_fp16(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const uint16_t *v16 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addTrainingFp16 requires a Buffer as an argument");
    int nrows = 0;
    FILE *fh;

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    duk_get_prop_string(ctx, -1, "rowsAdded");
    nrows = duk_get_int_default(ctx, -1, 0);
    duk_pop(ctx);

    if (sz / 2 != dim)
        RP_THROW(ctx, "addTrainingFp16 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    float *v = vec_fp16_to_fp32(v16, (size_t)dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "addTrainingFp16 - Internal error getting file handle");

    if (write_vector_to_file(fh, v, dim) == -1)
        RP_THROW(ctx, "addTrainingFp16 - error writing to tmpfile - %s", strerror(errno));

    free(v);

    nrows++;
    duk_push_int(ctx, nrows);
    duk_put_prop_string(ctx, -2, "rowsAdded");

    return 0;
}

int close_and_unlink(FILE *fh, const char *path, const char **err)
{
    if (err)
        *err = NULL;

    if (fh)
    {
        if (fclose(fh) != 0)
        {
            if (err)
                *err = strerror(errno);
            return -1;
        }
    }
    if (path)
    {
        if (unlink(path) != 0)
        {
            if (err)
                *err = strerror(errno);
            return -1;
        }
    }
    return 0;
}
#define RP_THROW_AND_CLOSE(ctx, fh, path, ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        const char *__err;                                                                                             \
        if (close_and_unlink(fh, path, &__err))                                                                        \
        {                                                                                                              \
            char tothrow[2048];                                                                                        \
            snprintf(tothrow, 2048, __VA_ARGS__);                                                                      \
            duk_push_error_object(ctx, DUK_ERR_ERROR, "%s - Error closing temp file: %s", tothrow, __err);             \
        }                                                                                                              \
        else                                                                                                           \
            duk_push_error_object(ctx, DUK_ERR_ERROR, __VA_ARGS__);                                                    \
        (void)duk_throw(ctx);                                                                                          \
    } while (0)

static duk_ret_t dotrain(duk_context *ctx)
{
    duk_size_t dim;
    int nrows, fd;
    FILE *fh;

    const char *filename;

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "tempFile");
    filename = duk_get_string(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "faissDim");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    duk_get_prop_string(ctx, -1, "rowsAdded");
    nrows = duk_get_int_default(ctx, -1, 0);
    duk_pop(ctx);

    if (!nrows)
        RP_THROW(ctx, "faiss.train - no vectors have been added yet");

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "faiss.train - Internal error getting file handle");

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!idx)
        RP_THROW(ctx, "faiss.train - Internal error getting index handle");

    fflush(fh);

    fd = fileno(fh);
    if (fd < 0)
    {
        RP_THROW_AND_CLOSE(ctx, fh, filename, "faiss.train - Error :%s", strerror(errno));
    }

    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        RP_THROW_AND_CLOSE(ctx, fh, filename, "faiss.train - Internal error - cannot stat temp file");
    }

    off_t need_bytes = (off_t)nrows * (off_t)dim * sizeof(float);

    if ((off_t)need_bytes != st.st_size)
        RP_THROW_AND_CLOSE(ctx, fh, filename,
                           "faiss.train - Internal error - Training file is not expected size (wanted:%lu vs have:%lu)",
                           (size_t)need_bytes, (size_t)st.st_size);

    void *addr = mmap(NULL, need_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
    {
        RP_THROW_AND_CLOSE(ctx, fh, filename, "faiss.train - Internal error - cannot load temp file - %s",
                           strerror(errno));
    }

    /* optional: advise the kernel about our access pattern */
    // posix_madvise(addr, need_bytes, POSIX_MADV_SEQUENTIAL);

    int rc = faiss_Index_train(idx, nrows, (const float *)addr);

    if (rc != 0)
    {
        const char *err = faiss_get_last_error();
        munmap(addr, need_bytes);
        RP_THROW_AND_CLOSE(ctx, fh, filename, "faiss.train - training failed: %s", err);
    }

    if (munmap(addr, need_bytes) != 0)
    {
        fprintf(stderr, "faiss.train - training completed, but internal error in munmap: %s\n", strerror(errno));
        return 0;
    }

    const char *err;
    if (close_and_unlink(fh, filename, &err))
        fprintf(stderr, "faiss.train - training completed, but error closing temp file '%s' - %s\n", filename, err);

    return 0;
}

static duk_ret_t new_trainer(duk_context *ctx)
{
    const char *traindir = "/tmp";
    char trainfile[PATH_MAX];
    static int counter = 0; /* monotonically increasing */
    FILE *fh = NULL;

    if (!duk_is_undefined(ctx, 0))
        traindir = REQUIRE_STRING(ctx, 0, "faiss.trainer - argument must be a String (directory path)");

    /* strip trailing slashes (except for root "/") */
    size_t tlen = strlen(traindir);
    while (tlen > 1 && traindir[tlen - 1] == '/')
        tlen--;

    /* build filename: <traindir>/faisstrainingdata.<counter>.<pid> */
    pid_t pid = getpid();
    int n = snprintf(trainfile, PATH_MAX, "%.*s/faisstrainingdata.%d.%ld", (int)tlen, traindir, counter++, (long)pid);

    if (n < 0 || n >= PATH_MAX)
        RP_THROW(ctx, "faiss.trainer - training file path too long");

    fh = fopen(trainfile, "w+");
    if (!fh)
        RP_THROW(ctx, "faiss.trainer - failed to open temp file '%s' - %s", trainfile, strerror(errno));

    duk_push_current_function(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("idxthis"));
    duk_push_object(ctx);

    duk_get_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));

    duk_get_prop_string(ctx, -2, "settings");
    duk_put_prop_string(ctx, -2, "settings");

    duk_push_pointer(ctx, fh);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("fh"));

    duk_push_c_function(ctx, add_trainvec_fp16, 1);
    duk_put_prop_string(ctx, -2, "addTrainingfp16");

    duk_push_c_function(ctx, add_trainvec_fp32, 1);
    duk_put_prop_string(ctx, -2, "addTrainingfp32");

    duk_push_c_function(ctx, dotrain, 1);
    duk_put_prop_string(ctx, -2, "train");

    duk_push_string(ctx, trainfile);
    put_prop_readonly(ctx, -2, "tempFile");

    return 1;
}

/* end training */

static void load_index(const char *fname, FaissIndex **out, const char **err)
{
    if (access(fname, R_OK) != 0)
    {
        *out = NULL;
        *err = "Could not access file";
    }
    if (faiss_read_index_fname(fname, 0, out) != 0)
    {
        *out = NULL;
        *err = faiss_get_last_error();
    };
}

static void openidx_fromfile(duk_context *ctx, const char *fname, int addextras)
{

    FaissIndexType type = FAISS_INDEX_FLAT;
    int dim = 1024;
    FaissMetricType mtype = METRIC_INNER_PRODUCT;

    const char *type_str = "Flat";
    const char *mtype_str = "innerProduct";

    FaissIndex *idx = NULL;
    const char *err = NULL;
    load_index(fname, &idx, &err);
    if (!idx)
        RP_THROW(ctx, "faiss open error: %s", err);

    switch (type)
    {
    case FAISS_INDEX_FLAT:
        type_str = "Flat";
        break;
    case FAISS_INDEX_PQ:
        type_str = "PQ";
        break;
    case FAISS_INDEX_IVFFLAT:
        type_str = "IVFFlat";
        break;
    case FAISS_INDEX_IVFPQ:
        type_str = "IVFPQ";
        break;
    case FAISS_INDEX_IVFSCALAR:
        type_str = "IVFScalar";
        break;
    /* seems to be missing from this build?
    case FAISS_INDEX_IVFRSQ:
        type_str =  "IVFRSQ";
        break;
    */
    case FAISS_INDEX_HNSWFLAT:
        type_str = "HNSWFlat";
        break;
    case FAISS_INDEX_HNSWPQ:
        type_str = "HNSWPQ";
        break;
    case FAISS_INDEX_HNSWSQ:
        type_str = "HNSWSQ";
        break;
    case FAISS_INDEX_LSH:
        type_str = "LSH";
        break;
    case FAISS_INDEX_BIN_FLAT:
        type_str = "BinFlat";
        break;
    case FAISS_INDEX_BIN_IVFFLAT:
        type_str = "BinIVFFlat";
        break;
    /*
    case FAISS_INDEX_SHARD:
        type_str =  "Shard";
        break;
    case FAISS_INDEX_PROXY:
        type_str =  "Proxy";
        break;
    */
    default:
        type_str = "Unknown";
        break;
    }

    switch (mtype)
    {
    case METRIC_INNER_PRODUCT:
        mtype_str = "innerProduct";
        break;
    case METRIC_L2:
        mtype_str = "l2";
        break;
    case METRIC_L1:
        mtype_str = "l1";
        break;
    case METRIC_Linf:
        mtype_str = "infinity";
        break;
    case METRIC_Lp:
        mtype_str = "lp";
        break;
    case METRIC_Canberra:
        mtype_str = "canberra";
        break;
    case METRIC_BrayCurtis:
        mtype_str = "brayCurtis";
        break;
    case METRIC_JensenShannon:
        mtype_str = "jensenShannon";
    }

    duk_push_object(ctx);

    duk_push_pointer(ctx, idx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));

    duk_push_int(ctx, (int)type);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissType"));

    duk_push_object(ctx); // settings

    duk_push_int(ctx, dim);
    put_prop_readonly(ctx, -2, "faissDim");

    duk_push_string(ctx, type_str);
    put_prop_readonly(ctx, -2, "faissType");

    duk_push_string(ctx, mtype_str);
    put_prop_readonly(ctx, -2, "faissMtype");

    put_prop_readonly(ctx, -2, "settings");

    duk_push_c_function(ctx, add_fp16, 2);
    duk_put_prop_string(ctx, -2, "addFp16");

    duk_push_c_function(ctx, add_fp32, 2);
    duk_put_prop_string(ctx, -2, "addFp32");

    duk_push_c_function(ctx, save_index, 1);
    duk_put_prop_string(ctx, -2, "save");

    duk_push_c_function(ctx, do_search_fp16, 2);
    duk_put_prop_string(ctx, -2, "searchFp16");

    duk_push_c_function(ctx, do_search_fp32, 2);
    duk_put_prop_string(ctx, -2, "searchFp32");

    if (!faiss_Index_is_trained(idx))
    {
        duk_push_c_function(ctx, new_trainer, 1);
        duk_dup(ctx, -2);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("idxthis"));
        duk_put_prop_string(ctx, -2, "trainer");
    }
}

static duk_ret_t faiss_openidx_fromfile(duk_context *ctx)
{
    const char *fname = REQUIRE_STRING(ctx, 0, "faiss.openIndexFromFile - argument must be a String (filename)");
    openidx_fromfile(ctx, fname, 1);
    return 1;
}

// https://github.com/facebookresearch/faiss/wiki/The-index-factory
static duk_ret_t faiss_open_factory(duk_context *ctx)
{
    const char *desc =
        REQUIRE_STRING(ctx, 0, "faiss.openFactory - first argument must be a String (factory index description)");
    int dim =
        REQUIRE_UINT(ctx, 1, "faiss.openFactory - second argument must be a Positive Integer (vector dimensions)");
    const char *mtype_str = NULL;
    char fn[256];
    pid_t pid = getpid();
    int thrno = get_thread_num();
    FaissIndex *idx;
    FaissMetricType mtype = METRIC_INNER_PRODUCT;

    if (!duk_is_undefined(ctx, 2))
        mtype_str =
            REQUIRE_STRING(ctx, 2, "faiss.openFactory - third argument, if provided, must be a String (metric type)");

    if (mtype_str)
    {
        if (strcasecmp(mtype_str, "innerProduct") == 0 || strcasecmp(mtype_str, "ip") == 0)
        {
            mtype = METRIC_INNER_PRODUCT;
            mtype_str = "innerProduct";
        }
        else if (strcasecmp(mtype_str, "l2") == 0)
        {
            mtype = METRIC_L2;
        }
        else if (strcasecmp(mtype_str, "l1") == 0 || strcasecmp(mtype_str, "manhattan") == 0 ||
                 strcasecmp(mtype_str, "cityBlock") == 0)
        {
            mtype = METRIC_L1;
            mtype_str = "l1";
        }
        else if (strcasecmp(mtype_str, "linf") == 0 || strcasecmp(mtype_str, "infinity") == 0)
        {
            mtype = METRIC_Linf;
            mtype_str = "infinity";
        }
        else if (strcasecmp(mtype_str, "lp") == 0)
        {
            mtype = METRIC_Lp;
        }
        else if (strcasecmp(mtype_str, "canberra") == 0)
        {
            mtype = METRIC_Canberra;
        }
        else if (strcasecmp(mtype_str, "braycurtis") == 0)
        {
            mtype = METRIC_BrayCurtis;
            mtype_str = "brayCurtis";
        }
        else if (strcasecmp(mtype_str, "jensenshannon") == 0)
        {
            mtype = METRIC_JensenShannon;
            mtype_str = "jensenShannon";
        }
    }

    int err = faiss_index_factory(&idx, dim, desc, mtype);
    if (err)
    {
        const char *errmsg = faiss_get_last_error();
        RP_THROW(ctx, "Index creation failed for desc='%s': %s", desc, errmsg ? errmsg : "unknown error");
    }

    snprintf(fn, 256, "/tmp/tmpind_%d_%d", (int)pid, thrno);

    int rc = faiss_write_index_fname(idx, fn);

    if (rc)
        RP_THROW(ctx, "faiss.openFactory - Internal error: Failed to save temp file '%s' - %s", fn,
                 faiss_get_last_error());

    openidx_fromfile(ctx, fn, 0);

    unlink(fn);
    return 1;
}

static void open_faiss(duk_context *ctx)
{
    duk_push_object(ctx);

    duk_push_c_function(ctx, faiss_open_factory, 3);
    duk_put_prop_string(ctx, -2, "openFactory");

    duk_push_c_function(ctx, faiss_openidx_fromfile, 1);
    duk_put_prop_string(ctx, -2, "openIndexFromFile");
}

/* ***********************************************************
                          UTILS
   *********************************************************** */
static duk_ret_t numbers_to(duk_context *ctx, duk_idx_t arridx, int wantfp16)
{
    size_t i = 0, len = (size_t)duk_get_length(ctx, arridx);
    float *out = duk_push_fixed_buffer(ctx, len * 4);

    for (; i < len; i++)
    {
        duk_get_prop_index(ctx, arridx, (duk_uarridx_t)i);
        out[i] = (float)REQUIRE_NUMBER(ctx, -1, "utils.numbersToFP - array[%lu] is not a Number", i);
        duk_pop(ctx);
    }

    if (wantfp16)
    {
        uint16_t *out16 = duk_push_fixed_buffer(ctx, len * 2);
        vec_fp32_to_fp16_buf(out, out16, len);
    }

    return 1;
}

static duk_ret_t num_to_fp32(duk_context *ctx)
{
    REQUIRE_ARRAY(ctx, 0, "utils.numbersToFP32 - argument must be an Array of Numbers");
    return numbers_to(ctx, 0, 0);
}

static duk_ret_t num_to_fp16(duk_context *ctx)
{
    REQUIRE_ARRAY(ctx, 0, "utils.numbersToFP16 - argument must be an Array of Numbers");
    return numbers_to(ctx, 0, 1);
}

static duk_ret_t fp_to_num(duk_context *ctx, void *buf, size_t sz, int havefp16)
{
    float *freebuf = NULL, *fp32 = NULL;
    size_t dim, i = 0;

    if (havefp16)
    {
        if (!sz || sz % 2)
            RP_THROW(ctx, "fp16ToNumbers - invalid buffer data");
        dim = sz / 2;
        freebuf = fp32 = vec_fp16_to_fp32((uint16_t *)buf, dim);
    }
    else
    {
        if (!sz || sz % 4)
            RP_THROW(ctx, "fp32ToNumbers - invalid buffer data");
        dim = sz / 4;
        fp32 = (float *)buf;
    }
    duk_push_array(ctx);
    for (; i < dim; i++)
    {
        duk_push_number(ctx, (double)fp32[i]);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    if (freebuf)
        free(freebuf);
    return 1;
}

static duk_ret_t fp32_to_num(duk_context *ctx)
{
    size_t sz;
    void *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "fp32ToNumbers - argument must be a Buffer");
    return fp_to_num(ctx, v, sz, 0);
}

static duk_ret_t fp16_to_num(duk_context *ctx)
{
    size_t sz;
    void *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "fp16ToNumbers - argument must be a Buffer");
    return fp_to_num(ctx, v, sz, 1);
}

duk_ret_t fp32_to_fp16(duk_context *ctx)
{
    size_t sz;
    float *fp32 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "fp32Tofp16 - argument must be a Buffer");
    uint16_t *fp16 = duk_push_fixed_buffer(ctx, sz / 2);

    vec_fp32_to_fp16_buf(fp32, fp16, sz / 4);

    return 1;
}

duk_ret_t fp16_to_fp32(duk_context *ctx)
{
    size_t sz;
    uint16_t *fp16 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "fp16Tofp32 - argument must be a Buffer");
    float *fp32 = duk_push_fixed_buffer(ctx, sz * 2);

    vec_fp16_to_fp32_buf(fp16, fp32, sz / 2);

    return 1;
}

static void open_utils(duk_context *ctx)
{
    duk_push_object(ctx);

    /* initialize best string */
    float d = 0;
    uint16_t s = 0;
    vec_fp32_to_fp16_buf(&d, &s, 1);

    /* set best string */
    duk_push_string(ctx, fpConverterBackEnd);
    duk_put_prop_string(ctx, -2, "converterBackend");

    duk_push_c_function(ctx, num_to_fp32, 1);
    duk_put_prop_string(ctx, -2, "numbersToFp32");

    duk_push_c_function(ctx, num_to_fp16, 1);
    duk_put_prop_string(ctx, -2, "numbersToFp16");

    duk_push_c_function(ctx, fp32_to_num, 1);
    duk_put_prop_string(ctx, -2, "fp32ToNumbers");

    duk_push_c_function(ctx, fp16_to_num, 1);
    duk_put_prop_string(ctx, -2, "fp16ToNumbers");

    duk_push_c_function(ctx, fp16_to_fp32, 1);
    duk_put_prop_string(ctx, -2, "fp16ToFp32");

    duk_push_c_function(ctx, fp32_to_fp16, 1);
    duk_put_prop_string(ctx, -2, "fp32ToFp16");
}

/* **************************************************
   Initialize module
   ************************************************** */

duk_ret_t duk_open_module(duk_context *ctx)
{
    duk_push_object(ctx);
    open_llama(ctx);
    duk_put_prop_string(ctx, -2, "llamacpp");

    open_sentencepiece(ctx);
    duk_put_prop_string(ctx, -2, "sentencepiece");

    open_faiss(ctx);
    duk_put_prop_string(ctx, -2, "faiss");

    open_utils(ctx);
    duk_put_prop_string(ctx, -2, "utils");

    return 1;
}
