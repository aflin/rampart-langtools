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
#include "convert_vec.c"

#endif

/* ***********************************************************
                    UTILS
   *********************************************************** */

static duk_ret_t avg_vec16(duk_context *ctx)
{
    REQUIRE_ARRAY(ctx, 0, "avgVecFp16 - first argument must be an array of Buffers");

    duk_uarridx_t j=0, len=duk_get_length(ctx, 0);

    if(!len)
        RP_THROW(ctx, "avgVecFp16 - first argument must be an array of Buffers");

    if(len==1)
    {
        duk_pull(ctx, 0);
        return 1;
    }

    size_t vec_dim=0, lastsz=0;
    float *avgvec=NULL;
    for(;j<len;j++)
    {
        duk_get_prop_index(ctx, 0, j);
        uint16_t *v_in=REQUIRE_BUFFER_DATA(ctx, -1, &lastsz, "avgVecFp16 - first argument must be an array of Buffers");
        if(!vec_dim)
        {
            vec_dim=lastsz/2;
            CALLOC(avgvec, vec_dim * sizeof(float));
        }
        else if (vec_dim != lastsz/2)
        {
            free(avgvec);
            RP_THROW(ctx, "avgVecFp16 - vector size mismatch (expected %lu, got %lu in array[%lu]", vec_dim, lastsz/2, j);
        }

        float *v = vec_fp16_to_fp32(v_in, vec_dim);

        for (int i = 0; i < vec_dim; ++i)
        {
            avgvec[i] += v[i];
        }
        free(v);
        duk_pop(ctx);
    }

    double norm2 = 0.0;

    for (int i = 0; i < vec_dim; ++i)
    {
        avgvec[i] /= (float)len;
        norm2 += (double)avgvec[i] * (double)avgvec[i];
    }

    float inv = norm2 > 0.0 ? (float)(1.0 / (sqrt(norm2))) : 1;

    for (int i = 0; i < vec_dim; ++i)
    {
        avgvec[i] *= inv;
    }

    uint16_t *out = duk_push_fixed_buffer(ctx, lastsz);
    vec_fp32_to_fp16_buf(avgvec, out, vec_dim);

    free(avgvec);

    return 1;
}

static duk_ret_t avg_vec32(duk_context *ctx)
{
    REQUIRE_ARRAY(ctx, 0, "avgVecFp32 - first argument must be an array of Buffers");

    duk_uarridx_t j=0, len=duk_get_length(ctx, 0);

    if(!len)
        RP_THROW(ctx, "avgVecFp32 - first argument must be an array of Buffers");

    if(len==1)
    {
        duk_pull(ctx, 0);
        return 1;
    }

    size_t vec_dim=0, lastsz=0;
    float *avgvec=NULL;
    for(;j<len;j++)
    {
        duk_get_prop_index(ctx, 0, j);
        uint16_t *v=REQUIRE_BUFFER_DATA(ctx, -1, &lastsz, "avgVecFp32 - first argument must be an array of Buffers");
        duk_pop(ctx);
        if(!vec_dim)
        {
            vec_dim=lastsz/4;
            avgvec = duk_push_fixed_buffer(ctx, lastsz);
        }
        else if (vec_dim != lastsz/4)
        {
            RP_THROW(ctx, "avgVecFp32 - vector size mismatch (expected %lu, got %lu in array[%lu]", vec_dim, lastsz/4, j);
        }

        for (int i = 0; i < vec_dim; ++i)
        {
            avgvec[i] += v[i];
        }
    }

    double norm2 = 0.0;

    for (int i = 0; i < vec_dim; ++i)
    {
        avgvec[i] /= (float)len;
        norm2 += (double)avgvec[i] * (double)avgvec[i];
    }

    float inv = norm2 > 0.0 ? (float)(1.0 / (sqrt(norm2))) : 1;

    for (int i = 0; i < vec_dim; ++i)
    {
        avgvec[i] *= inv;
    }

    return 1;
}



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
#ifdef LANGTOOLS_MAIN_INCLUDE
static duk_ret_t open_utils(duk_context *ctx)
#else
duk_ret_t duk_open_module(duk_context *ctx)
#endif
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

    duk_push_c_function(ctx, avg_vec32, 1);
    duk_put_prop_string(ctx, -2, "avgVecFp32");

    duk_push_c_function(ctx, avg_vec16, 1);
    duk_put_prop_string(ctx, -2, "avgVecFp16");

    return 1;
}
