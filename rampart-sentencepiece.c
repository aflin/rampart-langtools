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
#include "rampart.h"
#include "spm_c_wrapper.h"

#endif

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
        duk_size_t plen = 0;
        duk_get_prop_index(ctx, arridx, (duk_uarridx_t)i);
        /* byte length, not duk_get_length's character length: multibyte UTF-8
           pieces are longer in bytes and would overflow the output buffer */
        REQUIRE_LSTRING(ctx, -1, &plen, "sentencepiece.decode - array[%lu] is not a string", i);
        mlen += (size_t)plen;
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
    out[0] = '\0'; /* strcat/strncat below append to this */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
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
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
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
    else if (!n_pieces)
    {
        duk_push_string(ctx, "");
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
        free(str);
    }
    spm_free_pieces(pieces, n_pieces);
    return 1;
}

/* free the processor.  As a finalizer the object is the argument at the top of
   the stack.  Copies of the object on other threads share the pointer, so only
   the copy on the thread+process that created it frees (same ownership
   convention as the llamacpp handles). */
static duk_ret_t sp_free_(duk_context *ctx)
{
    int othr = -1, opid = -1;

    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("spOriginThr")))
        othr = duk_get_int(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("spOriginPid")))
        opid = duk_get_int(ctx, -1);
    duk_pop(ctx);

    if (othr != get_thread_num() || opid != (int)getpid())
        return 0;

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("spmProcessor"));
    spm_processor_t *sp = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (sp)
        spm_free(sp);
    duk_push_pointer(ctx, NULL);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("spmProcessor"));
    return 0;
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
    duk_push_int(ctx, get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("spOriginThr"));
    duk_push_int(ctx, (int)getpid());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("spOriginPid"));
    duk_push_c_function(ctx, sp_encode, 2);
    duk_put_prop_string(ctx, -2, "encode");
    duk_push_c_function(ctx, sp_free_, 1);
    duk_set_finalizer(ctx, -2);
    return 1;
}

#ifdef LANGTOOLS_MAIN_INCLUDE
static duk_ret_t open_sentencepiece(duk_context *ctx)
#else
duk_ret_t duk_open_module(duk_context *ctx)
#endif
{
    duk_push_object(ctx);
    duk_push_c_function(ctx, sp_init, 2);
    duk_put_prop_string(ctx, -2, "init");
    duk_push_c_function(ctx, sp_decode, 1);
    duk_put_prop_string(ctx, -2, "decode");
    return 1;
}
