/* rampart-onnx.c -- Rampart (Duktape) bindings for ONNX Runtime.
 *
 * Layer 1: a general inference session.
 *   onnx.onnxVersion()                       -> "1.27.0"
 *   onnx.modelInfo(path)                     -> { inputs:[...], outputs:[...] }
 *   onnx.initSession(path[, opts])           -> session
 *     session.inputs() / session.outputs()   -> [ {name,type,shape}, ... ]
 *     session.run(feeds)                      -> { outName: {data,shape,type}, ... }
 *     session.destroy()                       (+ finalizer)
 *
 * A "tensor" passed to run() is { data, shape, type }:
 *   data  : a buffer (raw bytes, interpreted per `type`) OR an Array of Numbers
 *   shape : Array of ints (optional; defaults to 1-D [n_elems])
 *   type  : 'float32'|'float16'|'float64'|'int64'|'int32'|'int16'|'int8'|'uint8'|'bool'
 * Outputs come back the same shape, with `data` as a fixed buffer of raw bytes.
 *
 * Conventions mirror rampart-llamacpp: camelCase opts, Fp32/Fp16 raw buffers,
 * a factory returning a handle with a destroy() + finalizer. The C ABI to ORT
 * lives in extern/onnxruntime/wrapper/onnx_shim.{h,cc}; this file stays pure C.
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>    /* strcasecmp: computeUnits / RAMPART_ONNX_COREML parsing */
#include <math.h>       /* sqrt for L2-norm in rp_onnx_embed_* */
#include <pthread.h>    /* handle-cache mutex */
#include <sys/stat.h>   /* stat: directory self-config for rp_onnx_embed_* */
#include <dirent.h>     /* opendir/readdir: find vocab.txt / *.onnx in a model dir */
#include "rampart.h"
#include "onnx_shim.h"     /* native tokenizers (onnx_wp / onnx_sp) + session API */
#include "rp-chunker.h"    /* structure-aware text chunking (shared source) */
#include "rp-embed-cache.h"/* content-keyed doc-result LRU (shared source) */

/* get_current_thread() lives in the rampart binary (like the duk_* symbols).
 * Keep the reference WEAK so a non-rampart host that dlopens this module just
 * for the rp_onnx_embed_* C API still loads and runs -- there the JSON-config
 * discovery is skipped (defaults apply) instead of faulting on the call. */
#pragma weak get_current_thread

/* ---- this.errMsg: warnings + non-fatal errors ------------------------------
 * Mirrors rampart-sql (rp_log_copy_to_errMsg there).  Warnings accumulate on
 * `this` -- the module object for onnx.initEmbed(), a handle for handle methods,
 * exactly as Sql/sql share the property.  Cleared at the top of each call, so it
 * always reflects the LAST call.  Kept separate from onnx.getLog(), which is the
 * informational ORT firehose and would bury a warning.
 *
 * Failures throw (RP_THROW); warnings land here; nothing goes to stdout/stderr. */
/* The duk context of the CALLING rampart thread.  This is what makes a pending
 * buffer unnecessary: it is reachable from ANY C code in the module -- the ORT
 * runtime ladder (pthread_once, inside the shim) and the rp_onnx_embed_* exports
 * that rampart-sql calls -- not just from a duk_ret_t with a ctx argument.
 * get_current_thread is a WEAK ref (see below): NULL in a bare non-rampart host
 * that dlopens this module only for the C API, where there is no JS to warn. */
static duk_context *onnx_thr_ctx(void) {
    RPTHR *t = get_current_thread ? get_current_thread() : NULL;
    return t ? t->ctx : NULL;
}

#define ONNX_MODULE_STASH DUK_HIDDEN_SYMBOL("onnx_module")

/* Push the object a warning belongs on: `this` when there is one (the module for
 * onnx.initEmbed(), a handle for handle methods -- as Sql/sql share errMsg), else
 * the module object.  The fallback covers arrivals with no `this` (e.g. the
 * rp_onnx_embed_* exports called from rampart-sql).  0 if neither exists. */
static int onnx_push_errmsg_target(duk_context *ctx) {
    duk_push_this(ctx);
    if (duk_is_object(ctx, -1)) return 1;
    duk_pop(ctx);
    duk_push_global_stash(ctx);
    if (!duk_get_prop_string(ctx, -1, ONNX_MODULE_STASH) || !duk_is_object(ctx, -1)) {
        duk_pop_2(ctx);
        return 0;
    }
    duk_remove(ctx, -2);                 /* drop the stash, leave the module object */
    return 1;
}

static void onnx_errmsg_append(duk_context *ctx, const char *msg) {
    if (!onnx_push_errmsg_target(ctx)) return;
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

/* clear this.errMsg -- call at the top of every JS entry point */
static void onnx_errmsg_clear(duk_context *ctx) {
    if (!onnx_push_errmsg_target(ctx)) return;
    duk_del_prop_string(ctx, -1, "errMsg");
    duk_pop(ctx);
}

/* A warning, from anywhere in the module (including the shim).  Goes straight onto
 * this.errMsg -- no buffer, no drain step.  RAMPART_ONNX_DEBUG=1 additionally
 * echoes to stderr; that opt-in hatch is the only thing that may ever write there. */
void onnx_warn(const char *fmt, ...) {
    duk_context *ctx = onnx_thr_ctx();
    char line[1024];
    size_t l;
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (getenv("RAMPART_ONNX_DEBUG")) fputs(line, stderr);
    if (!ctx) return;                 /* non-rampart host: no JS to carry it */
    l = strlen(line);
    while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = '\0';
    if (l) onnx_errmsg_append(ctx, line);
}

#define ONNX_PTR DUK_HIDDEN_SYMBOL("onnx_session_ptr")

/* The session struct lives once on the heap; every rampart-thread's copy of the
 * handle carries the same pointer to it (deep-copy preserves the duk_pointer).
 * So it must be freed exactly once, and only when NO thread can still be using
 * it. We free it only when this is the LAST live rampart thread
 * (get_thread_count()==1): then no other thread holds a copy or is mid-run, so
 * the free is safe. Earlier destroy()/finalizer calls (count>1) just mark it
 * destroyed (a process-wide flag in the struct) and leave the free to whichever
 * thread is last. A new rampart thread is flagged IN_USE under the thread lock
 * before its creator returns, so count==1 is a race-free "I'm alone" signal.
 * Returns 1 if it freed, 0 otherwise. Nulls this copy's pointer on free. */
static int free_session_if_last(duk_context *ctx, duk_idx_t obj_idx) {
    obj_idx = duk_normalize_index(ctx, obj_idx);
    onnx_session *s = NULL;
    if (duk_get_prop_string(ctx, obj_idx, ONNX_PTR))
        s = (onnx_session *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);
    if (!s) return 0;                            /* already freed */
    if (get_thread_count(NULL) != 1) return 0;  /* other threads may still hold/use it */
    onnx_session_destroy(s);                     /* real free (ORT session + struct) */
    duk_push_pointer(ctx, NULL);
    duk_put_prop_string(ctx, obj_idx, ONNX_PTR);
    return 1;
}

/* ---- float32 -> float16 (round-to-nearest-even), for float16 inputs ------ */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff)                 /* inf/nan */
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);     /* overflow -> inf */
    if (exp <= 0) {                                          /* subnormal/zero */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = (man >> shift);
        uint32_t rem  = man & ((1u << shift) - 1);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1)))
            half++;
        return (uint16_t)(sign | half);
    }
    uint16_t out = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) out++;
    return out;
}

/* product of a shape array (>=0 dims); returns 1 for empty (scalar). */
static int64_t shape_product(const int64_t *shape, size_t n) {
    int64_t p = 1;
    for (size_t i = 0; i < n; ++i) p *= (shape[i] < 0 ? 0 : shape[i]);
    return p;
}

/* push a JS array of { name, type, shape:[...] } from an io list */
static void push_iodesc_array(duk_context *ctx, const onnx_iodesc *list, size_t n) {
    duk_push_array(ctx);
    for (size_t i = 0; i < n; ++i) {
        duk_push_object(ctx);
        duk_push_string(ctx, list[i].name ? list[i].name : "");
        duk_put_prop_string(ctx, -2, "name");
        duk_push_string(ctx, onnx_dtype_name(list[i].dtype));
        duk_put_prop_string(ctx, -2, "type");
        duk_push_array(ctx);
        for (size_t d = 0; d < list[i].n_dims; ++d) {
            duk_push_number(ctx, (double)list[i].shape[d]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)d);
        }
        duk_put_prop_string(ctx, -2, "shape");
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
}

/* convert one element (Number at array index k of array at arr_idx) into the
 * dtype slot of `buf`. Caller guarantees buf has room. */
static void put_typed_elem(duk_context *ctx, duk_idx_t arr_idx, duk_uarridx_t k,
                           int dtype, void *buf, size_t i) {
    duk_get_prop_index(ctx, arr_idx, k);
    /* no duk_to_number: it can re-enter JS (valueOf) and throw out of loops
     * that hold malloc'd buffers.  Plain numbers are required ("data: Array
     * of Numbers"); anything else becomes 0. */
    double v = duk_get_number_default(ctx, -1, 0.0);
    duk_pop(ctx);
    switch (dtype) {
        case ONNX_DT_FLOAT32: ((float   *)buf)[i] = (float)v; break;
        case ONNX_DT_DOUBLE:  ((double  *)buf)[i] = (double)v; break;
        case ONNX_DT_FLOAT16: ((uint16_t*)buf)[i] = f32_to_f16((float)v); break;
        case ONNX_DT_INT64:   ((int64_t *)buf)[i] = (int64_t)v; break;
        case ONNX_DT_INT32:   ((int32_t *)buf)[i] = (int32_t)v; break;
        case ONNX_DT_INT16:   ((int16_t *)buf)[i] = (int16_t)v; break;
        case ONNX_DT_INT8:    ((int8_t  *)buf)[i] = (int8_t)v; break;
        case ONNX_DT_UINT8:   ((uint8_t *)buf)[i] = (uint8_t)v; break;
        case ONNX_DT_BOOL:    ((uint8_t *)buf)[i] = v != 0.0 ? 1 : 0; break;
        default: break;
    }
}

/* Read one input tensor object (at tensor_idx) into *vin, allocating owned
 * data/shape buffers (caller frees).  Returns 0, or -1 with err filled and the
 * duk stack restored -- it must NOT throw, because os_run accumulates
 * allocations across feeds and a longjmp here would leak them all. */
static int read_input_tensor(duk_context *ctx, duk_idx_t tensor_idx,
                             const char *name, onnx_value_in *vin,
                             void **owned_data, int64_t **owned_shape,
                             char *err, size_t errlen) {
    duk_idx_t entry_top = duk_get_top(ctx);
    int64_t *shape = NULL; size_t n_dims = 0;
    void *data = NULL;
#define RIT_FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); \
        free(shape); free(data); duk_set_top(ctx, entry_top); return -1; } while (0)

    tensor_idx = duk_normalize_index(ctx, tensor_idx);
    if (!duk_is_object(ctx, tensor_idx))
        RIT_FAIL("onnx run: feed '%s' must be an object { data, shape, type }", name);

    /* type (required) */
    if (!duk_get_prop_string(ctx, tensor_idx, "type"))
        RIT_FAIL("onnx run: feed '%s' is missing 'type'", name);
    const char *tname = duk_to_string(ctx, -1);
    int dtype = onnx_dtype_from_name(tname);
    duk_pop(ctx);
    if (dtype == ONNX_DT_UNKNOWN)
        RIT_FAIL("onnx run: feed '%s' has unsupported type '%s'", name, tname ? tname : "");
    size_t esz = onnx_dtype_size(dtype);

    /* shape (optional) */
    if (duk_get_prop_string(ctx, tensor_idx, "shape")) {
        if (!duk_is_array(ctx, -1))
            RIT_FAIL("onnx run: feed '%s' shape must be an Array", name);
        n_dims = (size_t)duk_get_length(ctx, -1);
        shape = (int64_t *)malloc((n_dims ? n_dims : 1) * sizeof(int64_t));
        if (!shape) RIT_FAIL("onnx run: out of memory");
        for (size_t d = 0; d < n_dims; ++d) {
            duk_get_prop_index(ctx, -1, (duk_uarridx_t)d);
            shape[d] = (int64_t)duk_to_number(ctx, -1);
            duk_pop(ctx);
        }
    }
    duk_pop(ctx); /* shape value (or undefined) */

    /* data (required): buffer (raw) or Array (converted) */
    if (!duk_get_prop_string(ctx, tensor_idx, "data"))
        RIT_FAIL("onnx run: feed '%s' is missing 'data'", name);
    size_t n_elems = 0;

    if (duk_is_array(ctx, -1)) {
        n_elems = (size_t)duk_get_length(ctx, -1);
        data = malloc(n_elems ? n_elems * esz : 1);
        if (!data) RIT_FAIL("onnx run: out of memory");
        for (size_t i = 0; i < n_elems; ++i)
            put_typed_elem(ctx, -1, (duk_uarridx_t)i, dtype, data, i);
    } else {
        duk_size_t blen = 0;
        void *bdata = duk_get_buffer_data(ctx, -1, &blen);
        if (!bdata && blen == 0) {
            /* allow a zero-length buffer; but a non-buffer non-array is an error */
            if (!duk_is_buffer_data(ctx, -1))
                RIT_FAIL("onnx run: feed '%s' data must be a Buffer or Array", name);
        }
        if (blen % esz != 0)
            RIT_FAIL("onnx run: feed '%s' buffer length %lu not a multiple of %s size %lu",
                     name, (unsigned long)blen, onnx_dtype_name(dtype), (unsigned long)esz);
        n_elems = blen / esz;
        data = malloc(blen ? blen : 1);
        if (!data) RIT_FAIL("onnx run: out of memory");
        memcpy(data, bdata, blen);
    }
    duk_pop(ctx); /* data value */

    /* default shape = 1-D [n_elems] when none was supplied */
    if (!shape) {
        shape = (int64_t *)malloc(sizeof(int64_t));
        if (!shape) RIT_FAIL("onnx run: out of memory");
        shape[0] = (int64_t)n_elems;
        n_dims = 1;
    } else {
        int64_t want = shape_product(shape, n_dims);
        if ((size_t)want != n_elems)
            RIT_FAIL("onnx run: feed '%s' shape implies %lld elems but data has %lu",
                     name, (long long)want, (unsigned long)n_elems);
    }
#undef RIT_FAIL

    vin->name    = name;     /* borrowed: caller keeps the name string alive */
    vin->dtype   = dtype;
    vin->shape   = shape;
    vin->n_dims  = n_dims;
    vin->data    = data;
    vin->n_bytes = n_elems * esz;
    *owned_data  = data;
    *owned_shape = shape;
    return 0;
}

/* dtype -> Duktape typed-array kind for the convenience `array` view on outputs
 * (-1 = no native typed array, e.g. int64: caller uses the raw `data` buffer). */
static int onnx_dtype_bufobj(int dtype) {
    switch (dtype) {
        case ONNX_DT_FLOAT32: return DUK_BUFOBJ_FLOAT32ARRAY;
        case ONNX_DT_DOUBLE:  return DUK_BUFOBJ_FLOAT64ARRAY;
        case ONNX_DT_INT32:   return DUK_BUFOBJ_INT32ARRAY;
        case ONNX_DT_INT16:   return DUK_BUFOBJ_INT16ARRAY;
        case ONNX_DT_INT8:    return DUK_BUFOBJ_INT8ARRAY;
        case ONNX_DT_UINT8:   return DUK_BUFOBJ_UINT8ARRAY;
        case ONNX_DT_BOOL:    return DUK_BUFOBJ_UINT8ARRAY;
        case ONNX_DT_FLOAT16: return DUK_BUFOBJ_UINT16ARRAY; /* raw half bits */
        default:              return -1;                     /* int64: no native type */
    }
}

/* fetch the live session pointer from `this`; throws if destroyed. */
static onnx_session *this_session(duk_context *ctx) {
    duk_push_this(ctx);
    if (!duk_get_prop_string(ctx, -1, ONNX_PTR))
        RP_THROW(ctx, "onnx: session has been destroyed");
    onnx_session *s = (onnx_session *)duk_get_pointer(ctx, -1);
    duk_pop_2(ctx);
    if (!s || onnx_session_is_destroyed(s)) RP_THROW(ctx, "onnx: session has been destroyed");
    return s;
}

/* ---- session methods ----------------------------------------------------- */

static duk_ret_t os_inputs(duk_context *ctx) {
    onnx_session *s = this_session(ctx);
    const onnx_modelinfo *mi = onnx_session_info(s);
    push_iodesc_array(ctx, mi->inputs, mi->n_inputs);
    return 1;
}

static duk_ret_t os_outputs(duk_context *ctx) {
    onnx_session *s = this_session(ctx);
    const onnx_modelinfo *mi = onnx_session_info(s);
    push_iodesc_array(ctx, mi->outputs, mi->n_outputs);
    return 1;
}

static duk_ret_t os_run(duk_context *ctx) {
    onnx_session *s = this_session(ctx);
    /* If we've fork()'d since this session was created, an inherited
     * multi-threaded session is broken -- rebuild it (CPU) or, in a future GPU
     * build, throw. Single-threaded sessions need nothing. */
    char ferr[300] = {0};
    if (onnx_session_ensure_runnable(s, ferr, sizeof ferr) != 0)
        RP_THROW(ctx, "onnx run: %s", ferr[0] ? ferr : "session not runnable");
    if (!duk_is_object(ctx, 0) || duk_is_array(ctx, 0))
        RP_THROW(ctx, "onnx run: argument must be an object mapping input name -> tensor");
    duk_idx_t feeds = 0;

    /* enumerate own keys -> collect names */
    size_t cap = 8, n_ins = 0;
    char         **names = malloc(cap * sizeof(char *));
    onnx_value_in *vins  = malloc(cap * sizeof(onnx_value_in));
    void         **odata = malloc(cap * sizeof(void *));
    int64_t      **oshape= malloc(cap * sizeof(int64_t *));
    if (!names || !vins || !odata || !oshape) {
        free(names); free(vins); free(odata); free(oshape);
        RP_THROW(ctx, "onnx run: out of memory");
    }

    char rerr[512] = {0};
    duk_enum(ctx, feeds, DUK_ENUM_OWN_PROPERTIES_ONLY);
    while (duk_next(ctx, -1, 0 /* key only */)) {
        const char *key = duk_get_string(ctx, -1);
        if (n_ins == cap) {
            size_t ncap = cap * 2;
            char         **nn = realloc(names,  ncap * sizeof(char *));
            onnx_value_in *nv = realloc(vins,   ncap * sizeof(onnx_value_in));
            void         **nd = realloc(odata,  ncap * sizeof(void *));
            int64_t      **ns = realloc(oshape, ncap * sizeof(int64_t *));
            if (nn) names = nn;
            if (nv) vins = nv;
            if (nd) odata = nd;
            if (ns) oshape = ns;
            if (!nn || !nv || !nd || !ns) { snprintf(rerr, sizeof rerr, "onnx run: out of memory"); break; }
            cap = ncap;
        }
        names[n_ins] = strdup(key ? key : "");
        duk_pop(ctx); /* key */
        if (!names[n_ins]) { snprintf(rerr, sizeof rerr, "onnx run: out of memory"); break; }
        /* fetch feeds[key] tensor and read it (non-throwing: we own allocations) */
        duk_get_prop_string(ctx, feeds, names[n_ins]);
        if (read_input_tensor(ctx, -1, names[n_ins], &vins[n_ins], &odata[n_ins],
                              &oshape[n_ins], rerr, sizeof rerr) != 0) {
            duk_pop(ctx); /* tensor */
            free(names[n_ins]);
            break;
        }
        duk_pop(ctx); /* tensor */
        n_ins++;
    }
    duk_pop(ctx); /* enum */
    if (rerr[0]) {
        for (size_t i = 0; i < n_ins; ++i) { free(odata[i]); free(oshape[i]); free(names[i]); }
        free(names); free(vins); free(odata); free(oshape);
        RP_THROW(ctx, "%s", rerr);
    }

    /* run (all outputs) */
    char err[512] = {0};
    onnx_value_out *outs = NULL; size_t n_outs = 0;
    int rc = onnx_session_run(s, vins, n_ins, NULL, 0, &outs, &n_outs, err, sizeof err);

    /* free owned input temporaries + names */
    for (size_t i = 0; i < n_ins; ++i) { free(odata[i]); free(oshape[i]); free(names[i]); }
    free(names); free(vins); free(odata); free(oshape);

    if (rc != 0)
        RP_THROW(ctx, "onnx run: %s", err[0] ? err : "failed");

    /* build result object { name: { data:<buffer>, shape:[...], type } } */
    duk_push_object(ctx);
    for (size_t i = 0; i < n_outs; ++i) {
        duk_push_object(ctx);

        /* Raw bytes in a fixed buffer, exposed two ways over the same storage:
         *   .data  -- an ArrayBuffer (so `new Float32Array(t.data)` reinterprets
         *             bytes; a plain Duktape buffer would copy byte-wise instead)
         *   .array -- a ready typed-array view matching the dtype (omitted for
         *             int64, which has no native Duktape typed array). */
        duk_size_t nb = (duk_size_t)outs[i].n_bytes;
        void *buf = duk_push_fixed_buffer(ctx, nb);          /* [outObj, plainbuf] */
        if (nb) memcpy(buf, outs[i].data, outs[i].n_bytes);
        duk_push_buffer_object(ctx, -1, 0, nb, DUK_BUFOBJ_ARRAYBUFFER);
        duk_put_prop_string(ctx, -3, "data");                /* outObj.data = AB */
        int taf = onnx_dtype_bufobj(outs[i].dtype);
        if (taf >= 0) {
            duk_push_buffer_object(ctx, -1, 0, nb, (duk_uint_t)taf);
            duk_put_prop_string(ctx, -3, "array");           /* outObj.array = typed view */
        }
        duk_pop(ctx); /* plainbuf (kept alive by the buffer objects above) */

        duk_push_array(ctx);
        for (size_t d = 0; d < outs[i].n_dims; ++d) {
            duk_push_number(ctx, (double)outs[i].shape[d]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)d);
        }
        duk_put_prop_string(ctx, -2, "shape");

        duk_push_string(ctx, onnx_dtype_name(outs[i].dtype));
        duk_put_prop_string(ctx, -2, "type");

        duk_put_prop_string(ctx, -2, outs[i].name ? outs[i].name : "");
    }
    onnx_run_free(outs, n_outs);
    return 1;
}

static duk_ret_t os_destroy(duk_context *ctx) {
    duk_push_this(ctx);
    onnx_session *s = NULL;
    if (duk_get_prop_string(ctx, -1, ONNX_PTR))
        s = (onnx_session *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);
    if (s) {
        onnx_session_mark_destroyed(s);    /* later use from ANY thread now throws */
        free_session_if_last(ctx, -1);     /* free now iff last thread, else defer */
    }
    return 0;
}

/* finalizer: object arrives as argument 0 */
static duk_ret_t os_destroy_finalizer(duk_context *ctx) {
    free_session_if_last(ctx, 0);   /* arg 0 is the object; frees iff this is the last thread */
    return 0;
}

/* ---- top-level module functions ------------------------------------------ */

static int parse_graph_opt(duk_context *ctx, duk_idx_t v) {
    if (duk_is_number(ctx, v)) return (int)duk_get_int(ctx, v);
    const char *s = duk_to_string(ctx, v);
    if (!s) return ONNX_OPT_DEFAULT;
    if (!strcmp(s, "disable"))  return 0;
    if (!strcmp(s, "basic"))    return 1;
    if (!strcmp(s, "extended")) return 2;
    if (!strcmp(s, "all"))      return 3;
    return ONNX_OPT_DEFAULT;
}

/* session.metadata() -> { producerName, graphName, domain, description, version } */
static duk_ret_t os_metadata(duk_context *ctx) {
    onnx_session *s = this_session(ctx);
    char *producer = NULL, *graph = NULL, *domain = NULL, *desc = NULL;
    int64_t ver = 0; char err[256] = {0};
    onnx_session_metadata(s, &producer, &graph, &domain, &desc, &ver, err, sizeof err);
    duk_push_object(ctx);
    if (producer) { duk_push_string(ctx, producer); duk_put_prop_string(ctx, -2, "producerName"); free(producer); }
    if (graph)    { duk_push_string(ctx, graph);    duk_put_prop_string(ctx, -2, "graphName");    free(graph); }
    if (domain)   { duk_push_string(ctx, domain);   duk_put_prop_string(ctx, -2, "domain");       free(domain); }
    if (desc)     { duk_push_string(ctx, desc);     duk_put_prop_string(ctx, -2, "description");  free(desc); }
    duk_push_number(ctx, (double)ver); duk_put_prop_string(ctx, -2, "version");
    return 1;
}

/* Fill session opts from a JS opts object at idx (may be undefined/null).
 * Default intraOpThreads = 0 = ORT's own pool (the machine's physical cores):
 * a single embed/rerank call uses the whole CPU, like llamacpp.  Fork safety
 * costs nothing: a threaded session crossing a fork() is transparently
 * REBUILT from its stashed source on first use in the child (see
 * onnx_session_ensure_runnable); a GPU session refuses after fork (a GPU
 * context cannot be rebuilt in a forked child).  intraOpThreads:1 remains
 * available for a session that must be INHERITED across fork() untouched
 * (zero background threads, no rebuild). */
static void parse_session_opts(duk_context *ctx, duk_idx_t idx, onnx_session_opts *so) {
    so->intra_threads = 0; so->inter_threads = 1;
    so->graph_opt = ONNX_OPT_DEFAULT; so->execution_mode = 0;
    /* Auto-GPU by default: on a GPU build with a usable device the JS handles
     * (initSession/initEmbed/initRerank all route here) use CUDA automatically --
     * mirroring the SQL embed() C-ABI path (rp_onnx_embed_load) and llamaEmbed.
     * The caller opts out with gpu:false or provider:'cpu'.  onnx_cuda_ep_available()
     * is 0 on cpu builds and non-CUDA platforms, so this is a no-op there.  macOS
     * CoreML stays OPT-IN (gpu:true / provider:'coreml' / RAMPART_ONNX_COREML)
     * because its MPSGraph backend can abort on these models.  A GPU session that
     * can't actually be created falls back to CPU in onnx_init_session (below). */
    so->use_cuda = onnx_cuda_ep_available(); so->cuda_device_id = 0;
    so->use_coreml = 0; so->coreml_units = 0;
    if (duk_is_undefined(ctx, idx) || duk_is_null(ctx, idx)) return;
    REQUIRE_OBJECT(ctx, idx, "initSession: options, if present, must be an Object");
    if (duk_get_prop_string(ctx, idx, "intraOpThreads"))
        so->intra_threads = (int)REQUIRE_INT(ctx, -1, "initSession: intraOpThreads must be an Integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, idx, "interOpThreads"))
        so->inter_threads = (int)REQUIRE_INT(ctx, -1, "initSession: interOpThreads must be an Integer");
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, idx, "graphOpt"))
        so->graph_opt = parse_graph_opt(ctx, -1);
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, idx, "executionMode")) {
        const char *m = duk_to_string(ctx, -1);
        so->execution_mode = (m && !strcmp(m, "parallel")) ? 1 : 0;
    }
    duk_pop(ctx);
    /* GPU/provider selection. `gpu:true` requests the platform's GPU execution
     * provider: CUDA on Linux, CoreML on macOS. `provider:'cuda'|'coreml'`
     * names one explicitly. The shim appends it and, on a build without that
     * EP, ORT returns a clear "not enabled" error at initSession time. */
    if (duk_get_prop_string(ctx, idx, "gpu")) {
        if (duk_to_boolean(ctx, -1)) {
#ifdef __APPLE__
            so->use_coreml = 1; so->use_cuda = 0;
#else
            so->use_cuda = 1;
#endif
        } else {                 /* gpu:false -> force CPU, overriding the auto default */
            so->use_cuda = 0; so->use_coreml = 0;
        }
    }
    duk_pop(ctx);
    if (duk_get_prop_string(ctx, idx, "provider")) {
        const char *p = duk_to_string(ctx, -1);
        if (p) {
            if (!strcmp(p, "cuda")) { so->use_cuda = 1; so->use_coreml = 0; }
            else if (!strcmp(p, "coreml")) { so->use_coreml = 1; so->use_cuda = 0; }
            else if (!strcmp(p, "gpu")) {
#ifdef __APPLE__
                so->use_coreml = 1;
#else
                so->use_cuda = 1;
#endif
            }
            else if (!strcmp(p, "cpu")) { so->use_cuda = 0; so->use_coreml = 0; }
            else
                RP_THROW(ctx, "initSession: provider '%s' not recognized (use 'cpu', 'cuda' or 'coreml')", p);
        }
    }
    duk_pop(ctx);
    if (so->use_cuda) {
        if (duk_get_prop_string(ctx, idx, "device"))
            so->cuda_device_id = (int)REQUIRE_INT(ctx, -1, "initSession: device must be an Integer");
        duk_pop(ctx);
    }
    if (so->use_coreml) {
        /* Default CPUAndNeuralEngine, NOT ALL: on current macOS (observed on
         * 15.6 / M-series) the MPSGraph (GPU) backend hits a failed assertion
         * ("mps.matmul op contracting dimensions differ") compiling this
         * dynamic-shape model class and ABORTS the process; the ANE compiler
         * handles the same graphs fine.  'all'/'cpuAndGPU' remain selectable
         * for testing on future macOS versions. */
        so->coreml_units = 2;
        /* computeUnits: which Apple compute units CoreML may schedule onto. */
        if (duk_get_prop_string(ctx, idx, "computeUnits")) {
            const char *u = duk_to_string(ctx, -1);
            if (u) {
                if      (!strcasecmp(u, "all"))                  so->coreml_units = 0;
                else if (!strcasecmp(u, "cpuAndGPU"))            so->coreml_units = 1;
                else if (!strcasecmp(u, "cpuAndNeuralEngine"))   so->coreml_units = 2;
                else if (!strcasecmp(u, "cpuOnly"))              so->coreml_units = 3;
                else RP_THROW(ctx, "initSession: computeUnits '%s' not recognized "
                              "(use 'all', 'cpuAndGPU', 'cpuAndNeuralEngine' or 'cpuOnly')", u);
            }
        }
        duk_pop(ctx);
    }
}

/* Build and push the JS session object wrapping s. */
static void push_session_object(duk_context *ctx, onnx_session *s) {
    duk_push_object(ctx);
    duk_push_pointer(ctx, s);            duk_put_prop_string(ctx, -2, ONNX_PTR);
    duk_push_c_function(ctx, os_run, 1);      duk_put_prop_string(ctx, -2, "run");
    duk_push_c_function(ctx, os_inputs, 0);   duk_put_prop_string(ctx, -2, "inputs");
    duk_push_c_function(ctx, os_outputs, 0);  duk_put_prop_string(ctx, -2, "outputs");
    duk_push_c_function(ctx, os_metadata, 0); duk_put_prop_string(ctx, -2, "metadata");
    duk_push_c_function(ctx, os_destroy, 0);  duk_put_prop_string(ctx, -2, "destroy");
    duk_push_c_function(ctx, os_destroy_finalizer, 1);
    duk_set_finalizer(ctx, -2);
}

static duk_ret_t onnx_init_session(duk_context *ctx) {
    onnx_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    const char *path = REQUIRE_STRING(ctx, 0, "initSession: argument 1 must be a String (path to .onnx)");
    onnx_session_opts so; parse_session_opts(ctx, 1, &so);
    char err[512] = {0};
    onnx_session *s = onnx_session_create(path, &so, err, sizeof err);
    /* Auto-GPU (or explicit gpu:true) that can't create a GPU session falls back to
     * CPU with a one-line notice -- mirrors the SQL embed path -- so a GPU handle on
     * a machine with no usable device still works instead of throwing. */
    if (!s && (so.use_cuda || so.use_coreml)) {
        onnx_warn("rampart-onnx initSession: %s: %s build but no usable GPU "
                "(%s); using CPU\n", path, so.use_cuda ? "CUDA" : "CoreML",
                err[0] ? err : "session create failed");
        so.use_cuda = 0; so.use_coreml = 0; err[0] = '\0';
        s = onnx_session_create(path, &so, err, sizeof err);
    }
    if (!s) RP_THROW(ctx, "initSession: %s", err[0] ? err : "failed to load model");
    push_session_object(ctx, s);
    return 1;
}

static duk_ret_t onnx_init_session_from_buffer(duk_context *ctx) {
    onnx_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    duk_size_t len = 0;
    void *data = duk_get_buffer_data(ctx, 0, &len);
    if (!data || !len)
        RP_THROW(ctx, "initSessionFromBuffer: argument 1 must be a non-empty Buffer (model bytes)");
    onnx_session_opts so; parse_session_opts(ctx, 1, &so);
    char err[512] = {0};
    onnx_session *s = onnx_session_create_from_buffer(data, (size_t)len, &so, err, sizeof err);
    if (!s && (so.use_cuda || so.use_coreml)) {   /* GPU unavailable -> CPU (see onnx_init_session) */
        onnx_warn("rampart-onnx initSessionFromBuffer: %s build but no usable GPU "
                "(%s); using CPU\n", so.use_cuda ? "CUDA" : "CoreML", err[0] ? err : "session create failed");
        so.use_cuda = 0; so.use_coreml = 0; err[0] = '\0';
        s = onnx_session_create_from_buffer(data, (size_t)len, &so, err, sizeof err);
    }
    if (!s) RP_THROW(ctx, "initSessionFromBuffer: %s", err[0] ? err : "failed to load model");
    push_session_object(ctx, s);
    return 1;
}

static duk_ret_t onnx_model_info_js(duk_context *ctx) {
    const char *path = REQUIRE_STRING(ctx, 0, "modelInfo: argument 1 must be a String (path to .onnx)");
    char err[512] = {0};
    onnx_modelinfo *mi = NULL;
    if (onnx_model_info(path, &mi, err, sizeof err) != 0)
        RP_THROW(ctx, "modelInfo: %s", err[0] ? err : "failed");

    duk_push_object(ctx);
    push_iodesc_array(ctx, mi->inputs, mi->n_inputs);
    duk_put_prop_string(ctx, -2, "inputs");
    push_iodesc_array(ctx, mi->outputs, mi->n_outputs);
    duk_put_prop_string(ctx, -2, "outputs");
    onnx_modelinfo_free(mi);
    return 1;
}

static duk_ret_t onnx_version_js(duk_context *ctx) {
    duk_push_string(ctx, onnx_version());
    return 1;
}

/* onnx.runtimeInfo() -> which ORT runtime the selection ladder picked
 * ("built-in CPU", or the external onnx-cuNN dir + why).  Forces the
 * (lazy, once-per-process) selection if it hasn't run yet. */
static duk_ret_t onnx_runtime_info_js(duk_context *ctx) {
    (void)onnx_version();                 /* forces shim init / selection */
    duk_push_string(ctx, onnx_runtime_desc());
    return 1;
}

/* onnx.getLog() -> the captured ORT log (warnings + non-fatal errors that would
 * otherwise hit stderr). onnx.clearLog() empties it. */
static duk_ret_t onnx_get_log_js(duk_context *ctx) {
    char *s = onnx_log_dup();
    duk_push_string(ctx, s ? s : "");
    free(s);
    return 1;
}
static duk_ret_t onnx_clear_log_js(duk_context *ctx) {
    onnx_log_clear();
    return 0;
}

/* ============================================================
 * C-callable embed API — mirrors rampart-llamacpp's rp_embed_*
 *
 * Purpose: let rampart-sql's `embed()` SQL builtin dispatch to ONNX
 * models via `sql.set({onnxEmbed:{...}})`, using the same dlsym-based
 * plumbing that llamacpp uses.  These are the ONLY non-duk_open_module
 * exports; see rampart-onnx.map.
 *
 * ABI:
 *   rp_onnx_embed_opts is a versioned struct; callers set .abi_version
 *   to RP_ONNX_EMBED_ABI and load() rejects anything else.  v3: the
 *   exported rp_onnx_embed_doc gained (prefix, plen) after tlen -- the
 *   per-document chunk prefix (e.g. article title).  The caller
 *   (rampart-sql.c) carries its own copy of this struct: a drifted copy
 *   feeds the fields it lacks as stack garbage, so ANY change to the
 *   struct must bump the define in BOTH files.
 *   String pointers in the opts (tokenizer_path, query_prefix, ...)
 *   are borrowed for the duration of the load() call only -- load()
 *   copies whatever it needs.
 *
 * Directory self-config (mirrors JS initEmbed): pass a model DIRECTORY as
 * model_path and everything -- the .onnx, the tokenizer (native WordPiece
 * from *vocab.txt, or SentencePiece/BPE from tokenizer.json), pooling (from
 * 1_Pooling/config.json), bos/eos and normalize -- is discovered from it.
 * No rampart-sentencepiece dependency. opts.tokenizer_path is now optional
 * and only consulted in the legacy "model_path is a bare .onnx file" mode.
 * ============================================================ */

#define RP_ONNX_EMBED_ABI 4   /* v4: + sentence_split */

typedef struct {
    int         abi_version;      /* must be RP_ONNX_EMBED_ABI */
    const char *tokenizer_path;   /* file-mode only: tokenizer path (a
                                   * *vocab.txt or a dir with tokenizer.json);
                                   * NULL/empty in directory mode, which
                                   * self-discovers the tokenizer */
    int         bos_id;           /* -1 to disable prefix token; 0 typical for XLM-R */
    int         eos_id;           /* -1 to disable suffix token; 2 typical for XLM-R */
    int         id_offset;        /* added to each raw SPM id (e.g. XLM-R: 1) */
    int         pad_id;           /* padding id for ragged chunk batches (masked out) */
    int         max_tokens;       /* per-chunk token window; 0 = default 512 */
    int         pooling;          /* 0 = auto, 1 = mean, 2 = cls */
    int         normalize;        /* 0/1 */
    const char *query_prefix;     /* NULL or prefix for isQuery=true; unused in SQL path (isQuery=false) */
    const char *passage_prefix;   /* NULL or prefix for isQuery=false (applied per chunk) */
    int         max_chunk_batch;  /* max chunks per session_run (memory cap); 0 = default 64 */
    int         split_mode;       /* RP_CHUNK_AUTO(0) / _WINDOW(1) / _PARA(2) */
    int         min_split_tokens; /* paragraph fragment floor; 0 = default 32, -1 = off */
    int         pack_paragraphs;  /* 1 = pack paragraphs to the window (default: one chunk per paragraph) */
    int         sentence_split;   /* 1 = sentence-pack oversized pieces (rp-chunker sentence level) */
} rp_onnx_embed_opts;

/* Compile-time-visible dtype for the C output vector: always float32.
 * (JS side layers f16/bf16/... on top; the SQL side does the same.) */

/* ---- directory discovery for the C embed API (mirrors JS initEmbed) ---- */
static int rp_is_dir(const char *p){ struct stat st; return p && stat(p,&st)==0 && S_ISDIR(st.st_mode); }
static int rp_is_file(const char *p){ struct stat st; return p && stat(p,&st)==0 && S_ISREG(st.st_mode); }

/* find a file in `dir` whose name ends with `suffix`; write full path to out. */
static int rp_find_suffix(const char *dir, const char *suffix, char *out, size_t outsz){
    DIR *d = opendir(dir); if(!d) return 0;
    struct dirent *e; size_t sl = strlen(suffix); int found = 0;
    while((e = readdir(d))){
        size_t nl = strlen(e->d_name);
        if(nl >= sl && strcmp(e->d_name + nl - sl, suffix) == 0){
            snprintf(out, outsz, "%s/%s", dir, e->d_name); found = 1; break;
        }
    }
    closedir(d); return found;
}

/* read a small file into a malloc'd NUL-terminated buffer (caller frees). */
static char *rp_slurp(const char *path){
    FILE *f = fopen(path, "rb"); if(!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if(n < 0 || n > 4*1024*1024){ fclose(f); return NULL; }
    char *b = malloc((size_t)n + 1); if(!b){ fclose(f); return NULL; }
    size_t rd = fread(b, 1, (size_t)n, f); fclose(f); b[rd] = '\0'; return b;
}

/* JSON reads go through duktape's real JSON decoder -- even on the pure-C
 * rp_onnx_embed_* path, where the host thread's ctx is reachable via
 * rampart's get_current_thread()->ctx. */
static duk_ret_t rp_json_decode_raw(duk_context *ctx, void *udata)
{
    (void)udata;
    duk_json_decode(ctx, -1);
    return 1;
}

/* Read <path>, JSON-decode it (protected -- malformed JSON = not found), and
 * fetch top-level property `key` into *out (numbers as-is, booleans as 0/1).
 * Returns 1 if found, 0 otherwise. */
static int rp_json_key(duk_context *ctx, const char *path, const char *key, double *out)
{
    if (!ctx) return 0;
    char *txt = rp_slurp(path);
    if (!txt) return 0;
    int found = 0;
    duk_push_string(ctx, txt);
    free(txt);
    if (duk_safe_call(ctx, rp_json_decode_raw, NULL, 1, 1) == DUK_EXEC_SUCCESS
        && duk_is_object(ctx, -1)) {
        duk_get_prop_string(ctx, -1, key);
        if (duk_is_number(ctx, -1))       { *out = duk_get_number(ctx, -1);           found = 1; }
        else if (duk_is_boolean(ctx, -1)) { *out = duk_get_boolean(ctx, -1) ? 1 : 0;  found = 1; }
        duk_pop(ctx);
    }
    duk_pop(ctx);   /* decoded value (or the caught error) */
    return found;
}

/* Pooling mode from <dir>/1_Pooling/config.json: 2 = cls, 1 = mean, 0 = unknown. */
static int rp_discover_pooling(duk_context *ctx, const char *dir)
{
    char p[1100];
    double b = 0;
    snprintf(p, sizeof p, "%s/1_Pooling/config.json", dir);
    if (rp_json_key(ctx, p, "pooling_mode_cls_token", &b) && b)  return 2;
    if (rp_json_key(ctx, p, "pooling_mode_mean_tokens", &b) && b) return 1;
    return 0;
}

/* Auto per-chunk token window, llamacpp-style: the model's positional capacity
 * (config.json max_position_embeddings -- the GGUF n_ctx_train analog; fallback
 * sentence_bert_config.json max_seq_length), auto-capped at 8192 exactly like
 * rampart-llamacpp's embed context.  An explicit maxTokens overrides uncapped.
 * (bge-m3's XLM-R reports max_position_embeddings=8194 = 8192 + the 2-slot pad
 * offset; the cap absorbs it.)  Returns 0 if nothing was found. */
static int rp_discover_window(duk_context *ctx, const char *dir)
{
    char p[1100];
    double v = 0;
    snprintf(p, sizeof p, "%s/config.json", dir);
    if (!rp_json_key(ctx, p, "max_position_embeddings", &v) || v <= 0) {
        snprintf(p, sizeof p, "%s/sentence_bert_config.json", dir);
        if (!rp_json_key(ctx, p, "max_seq_length", &v)) v = 0;
    }
    int w = (int)v;
    if (w > 8192) w = 8192;
    return w > 0 ? w : 0;
}

typedef struct rp_onnx_embed_handle_s {
    /* Identity — used both as cache key and to answer rp_onnx_embed_dim(). */
    char             *model_path;      /* strdup */
    char             *tokenizer_path;  /* strdup */
    rp_onnx_embed_opts opts;           /* EFFECTIVE opts (dir-mode may auto-config
                                        * bos/eos/pad/pooling below); used at embed time */
    rp_onnx_embed_opts key_opts;       /* opts AS REQUESTED (pre auto-config) -- the
                                        * cache dedup key, so two identical sql.set()s
                                        * share one handle+cache even though dir-mode
                                        * mutates `opts` after load. See rp_onnx_embed_load. */
    char             *opt_qprefix;     /* strdup of opts.query_prefix   (NULL if none) */
    char             *opt_pprefix;     /* strdup of opts.passage_prefix (NULL if none) */

    /* Session + tokenizer state. */
    onnx_session      *sess;
    char              *onnx_file;   /* resolved .onnx (may live under <dir>/onnx/) */
    onnx_wp_tokenizer *wp_tok;      /* WordPiece (vocab.txt), or NULL */
    onnx_sp_tokenizer *sp_tok;      /* SentencePiece/BPE (tokenizer.json dir), or NULL */

    /* Cached model I/O layout, populated once at load. */
    int     vec_dim;
    char   *input_ids_name;         /* borrowed from session_info; never freed here */
    char   *attn_mask_name;         /* NULL if model has no attention_mask input */
    char   *token_type_name;        /* NULL if model has no token_type_ids input */
    char   *pooled_out_name;        /* the 2-D float32 output, if any */
    char   *hidden_out_name;        /* the 3-D float32 output, if any */
    int     use_pooled;             /* 1 = use pooled_out; 0 = pool hidden_out */
    int     n_ctx;                  /* per-chunk token window (opts.max_tokens or 512) */
    int     mcb;                    /* max chunks per session_run (opts.max_chunk_batch or 64) */

    /* Doc-result cache: one model run of a text feeds chunkembed() /
     * chunkavg() / chunkcoherence() / embed(), keyed on the text.  Its
     * own mutex (not h->mtx, which guards session state) so cache hits
     * on one text don't serialize embeds of another. */
    rp_doccache_t doc_cache;

    /* Lifecycle. */
    int              refcount;
    pthread_mutex_t  mtx;
    struct rp_onnx_embed_handle_s *next;   /* cache chain */
} rp_onnx_embed_handle_t;

static rp_onnx_embed_handle_t *g_onnx_embed_cache_head = NULL;
static pthread_mutex_t         g_onnx_embed_cache_lock = PTHREAD_MUTEX_INITIALIZER;

/* Two opts sets match for cache-sharing iff every field is equal, INCLUDING
 * the prefix strings.  That way `sql.set({onnxEmbed:{model:X, queryPrefix:'q: '}})`
 * from connection A and `{model:X, queryPrefix:'passage: '}` from B produce
 * distinct handles instead of stomping each other. */
static int rp_onnx_opts_equal(const rp_onnx_embed_opts *a, const rp_onnx_embed_opts *b)
{
    if (a->bos_id != b->bos_id) return 0;
    if (a->eos_id != b->eos_id) return 0;
    if (a->id_offset != b->id_offset) return 0;
    if (a->pad_id != b->pad_id) return 0;
    if (a->max_tokens != b->max_tokens) return 0;
    if (a->pooling != b->pooling) return 0;
    if (a->normalize != b->normalize) return 0;
    if (a->max_chunk_batch != b->max_chunk_batch) return 0;
    if (a->split_mode != b->split_mode) return 0;
    if (a->sentence_split != b->sentence_split) return 0;
    if (a->min_split_tokens != b->min_split_tokens) return 0;
    if (a->pack_paragraphs != b->pack_paragraphs) return 0;
    const char *aq = a->query_prefix   ? a->query_prefix   : "";
    const char *bq = b->query_prefix   ? b->query_prefix   : "";
    const char *ap = a->passage_prefix ? a->passage_prefix : "";
    const char *bp = b->passage_prefix ? b->passage_prefix : "";
    if (strcmp(aq, bq) != 0) return 0;
    if (strcmp(ap, bp) != 0) return 0;
    return 1;
}

static rp_onnx_embed_handle_t *
rp_onnx_embed_cache_get(const char *model, const char *tokenizer,
                        const rp_onnx_embed_opts *opts)
{
    pthread_mutex_lock(&g_onnx_embed_cache_lock);
    rp_onnx_embed_handle_t *h = NULL;
    for (rp_onnx_embed_handle_t *c = g_onnx_embed_cache_head; c; c = c->next) {
        /* Compare against the AS-REQUESTED key_opts, not the effective
         * opts: dir-mode auto-configures bos/eos/pad/pooling on `opts`
         * after load, which would otherwise never match a fresh caller's
         * request and defeat handle+cache sharing across threads. */
        if (strcmp(c->model_path, model) == 0
            && strcmp(c->tokenizer_path, tokenizer) == 0
            && rp_onnx_opts_equal(&c->key_opts, opts)) {
            h = c; h->refcount++; break;
        }
    }
    pthread_mutex_unlock(&g_onnx_embed_cache_lock);
    return h;
}

/* Destroy a fully-built handle (the loser of a load race).  Mirrors the
 * load-failure cleanup; the IO name pointers are owned by the session and
 * freed by onnx_session_destroy. */
static void rp_onnx_handle_destroy_built(rp_onnx_embed_handle_t *h)
{
    if (!h) return;
    if (h->wp_tok) onnx_wp_destroy(h->wp_tok);
    if (h->sp_tok) onnx_sp_destroy(h->sp_tok);
    if (h->sess)   onnx_session_destroy(h->sess);
    rp_doccache_destroy(&h->doc_cache);
    pthread_mutex_destroy(&h->mtx);
    free(h->opt_qprefix);
    free(h->opt_pprefix);
    free(h->model_path);
    free(h->tokenizer_path);
    free(h->onnx_file);
    free(h);
}

/* Publish `h`, deduping under the lock: if another thread finished loading
 * the same (model, tokenizer, key_opts) while we were building, destroy ours
 * and share the winner.  Without this, N threads racing the FIRST load of a
 * model each keep a private handle -- and therefore a private doc cache --
 * silently defeating cross-thread cache sharing (and likevCache would only
 * reach one of them).  Returns the canonical handle. */
static rp_onnx_embed_handle_t *rp_onnx_embed_cache_put(rp_onnx_embed_handle_t *h)
{
    pthread_mutex_lock(&g_onnx_embed_cache_lock);
    for (rp_onnx_embed_handle_t *c = g_onnx_embed_cache_head; c; c = c->next) {
        if (strcmp(c->model_path, h->model_path) == 0
            && strcmp(c->tokenizer_path, h->tokenizer_path) == 0
            && rp_onnx_opts_equal(&c->key_opts, &h->key_opts)) {
            c->refcount++;
            pthread_mutex_unlock(&g_onnx_embed_cache_lock);
            rp_onnx_handle_destroy_built(h);
            return c;
        }
    }
    h->next = g_onnx_embed_cache_head;
    g_onnx_embed_cache_head = h;
    pthread_mutex_unlock(&g_onnx_embed_cache_lock);
    return h;
}

/* Discover input/output layout on the session.  Fills the handle's
 * name pointers + vec_dim + use_pooled.  Returns 0 or -1 (with err).
 * All name pointers are borrowed from onnx_session_info(); they stay
 * valid because the handle (and its session) is process-lifetime —
 * rp_onnx_embed_release deliberately never frees (see its comment). */
static int rp_onnx_discover_io(rp_onnx_embed_handle_t *h, char *err, size_t errlen)
{
    const onnx_modelinfo *mi = onnx_session_info(h->sess);
    if (!mi) {
        if (err) snprintf(err, errlen, "rp_onnx_embed_load: session_info returned NULL");
        return -1;
    }

    /* Inputs: input_ids (or first input), attention_mask, token_type_ids. */
    h->input_ids_name  = NULL;
    h->attn_mask_name  = NULL;
    h->token_type_name = NULL;
    for (size_t i = 0; i < mi->n_inputs; i++) {
        const char *n = mi->inputs[i].name;
        if (!n) continue;
        if (!strcmp(n, "input_ids"))          h->input_ids_name  = mi->inputs[i].name;
        else if (!strcmp(n, "attention_mask")) h->attn_mask_name  = mi->inputs[i].name;
        else if (!strcmp(n, "token_type_ids")) h->token_type_name = mi->inputs[i].name;
    }
    if (!h->input_ids_name && mi->n_inputs > 0)
        h->input_ids_name = mi->inputs[0].name;
    if (!h->input_ids_name) {
        if (err) snprintf(err, errlen, "rp_onnx_embed_load: model has no inputs");
        return -1;
    }

    /* Outputs: first float32 2-D is pre-pooled; first float32 3-D is token-level. */
    h->pooled_out_name = NULL;
    h->hidden_out_name = NULL;
    for (size_t i = 0; i < mi->n_outputs; i++) {
        if (mi->outputs[i].dtype != ONNX_DT_FLOAT32) continue;
        if (mi->outputs[i].n_dims == 2 && !h->pooled_out_name)
            h->pooled_out_name = mi->outputs[i].name;
        else if (mi->outputs[i].n_dims == 3 && !h->hidden_out_name)
            h->hidden_out_name = mi->outputs[i].name;
    }
    if (!h->pooled_out_name && !h->hidden_out_name) {
        if (err) snprintf(err, errlen,
            "rp_onnx_embed_load: no float32 2-D/3-D output to pool");
        return -1;
    }

    /* Match onnx_init_embed: if the caller set pooling explicitly AND a 3-D
     * hidden output is present, prefer pooling the hidden output over a
     * pre-pooled 2-D output whose pooling might not match. */
    if (h->opts.pooling != 0 && h->hidden_out_name) {
        h->use_pooled = 0;
    } else if (h->pooled_out_name) {
        h->use_pooled = 1;
    } else {
        h->use_pooled = 0;
    }

    /* vec_dim: from the chosen output's last shape dim, if known. */
    const onnx_iodesc *out = NULL;
    for (size_t i = 0; i < mi->n_outputs; i++) {
        const char *want = h->use_pooled ? h->pooled_out_name : h->hidden_out_name;
        if (mi->outputs[i].name == want) { out = &mi->outputs[i]; break; }
    }
    if (!out || out->n_dims < 2) {
        if (err) snprintf(err, errlen, "rp_onnx_embed_load: chosen output has bad shape");
        return -1;
    }
    int64_t dim = out->shape[out->n_dims - 1];
    if (dim <= 0) {
        if (err) snprintf(err, errlen,
            "rp_onnx_embed_load: chosen output has dynamic last dim (%lld)",
            (long long)dim);
        return -1;
    }
    h->vec_dim = (int)dim;
    return 0;
}

/* Public API ------------------------------------------------------- */

/* Load or fetch-from-cache a handle.  Thread-safe, path+opts keyed.
 * On error: returns NULL, fills err[].  On success: returns opaque handle. */
void *rp_onnx_embed_load(const char *model_path,
                         const rp_onnx_embed_opts *opts_in,
                         char *err, size_t errlen)
{
    if (!model_path || !opts_in) {
        if (err && errlen) snprintf(err, errlen, "rp_onnx_embed_load: null arg");
        return NULL;
    }
    if (opts_in->abi_version != RP_ONNX_EMBED_ABI) {
        if (err && errlen)
            snprintf(err, errlen,
                "rp_onnx_embed_load: opts.abi_version=%d, expected %d",
                opts_in->abi_version, RP_ONNX_EMBED_ABI);
        return NULL;
    }
    /* --- discover the model + tokenizer + pooling + window from the directory --- */
    /* 1536: must exceed the cand/od[1100] staging buffers plus a dirent
       name, or gcc's -Wformat-truncation flags the copies (and a long
       model path would truncate into a confusing open failure). */
    char onnx_file[1536] = {0}, vocab_file[1536] = {0}, tok_dir[1536] = {0};
    int  is_wp = 0, disc_pooling = 0, disc_win = 0;  /* pooling: 0 none, 1 mean, 2 cls */
    int  dir_mode = rp_is_dir(model_path);
    /* the caller normally reached us from a JS call (sql.set / require), so the
     * host thread's duk ctx is available for the JSON reads; get_current_thread
     * is a weak ref (NULL in a bare non-rampart host -> discovery skipped) */
    RPTHR *rpthr = get_current_thread ? get_current_thread() : NULL;
    duk_context *jctx = rpthr ? rpthr->ctx : NULL;

    if (dir_mode) {
        char cand[1100];
        snprintf(cand, sizeof cand, "%s/onnx/model.onnx", model_path);
        if (rp_is_file(cand)) snprintf(onnx_file, sizeof onnx_file, "%s", cand);
        else {
            snprintf(cand, sizeof cand, "%s/model.onnx", model_path);
            if (rp_is_file(cand)) snprintf(onnx_file, sizeof onnx_file, "%s", cand);
            else {
                char od[1100]; snprintf(od, sizeof od, "%s/onnx", model_path);
                rp_find_suffix(rp_is_dir(od) ? od : model_path, ".onnx", onnx_file, sizeof onnx_file);
            }
        }
        if (!onnx_file[0]) {
            if (err) snprintf(err, errlen, "rp_onnx_embed_load: no .onnx model under %s", model_path);
            return NULL;
        }
        /* tokenizer: *vocab.txt -> WordPiece; else tokenizer.json -> SentencePiece/BPE */
        if (rp_find_suffix(model_path, "vocab.txt", vocab_file, sizeof vocab_file)) {
            is_wp = 1;
        } else {
            char tj[1100]; snprintf(tj, sizeof tj, "%s/tokenizer.json", model_path);
            if (rp_is_file(tj)) snprintf(tok_dir, sizeof tok_dir, "%s", model_path);
            else {
                if (err) snprintf(err, errlen, "rp_onnx_embed_load: no *vocab.txt or tokenizer.json in %s", model_path);
                return NULL;
            }
        }
        disc_pooling = rp_discover_pooling(jctx, model_path);
        disc_win     = rp_discover_window(jctx, model_path);
    } else {
        /* legacy file mode: model_path is a bare .onnx; opts.tokenizer_path picks
         * the tokenizer (a *vocab.txt for WordPiece, or a dir with tokenizer.json). */
        const char *tp = opts_in->tokenizer_path;
        snprintf(onnx_file, sizeof onnx_file, "%s", model_path);
        if (!tp || !tp[0]) {
            if (err) snprintf(err, errlen,
                "rp_onnx_embed_load: pass a model directory, or (file mode) set opts.tokenizer_path");
            return NULL;
        }
        size_t tl = strlen(tp);
        if (tl >= 9 && strcmp(tp + tl - 9, "vocab.txt") == 0) { snprintf(vocab_file, sizeof vocab_file, "%s", tp); is_wp = 1; }
        else if (rp_is_dir(tp)) { snprintf(tok_dir, sizeof tok_dir, "%s", tp); }
        else {
            if (err) snprintf(err, errlen,
                "rp_onnx_embed_load: opts.tokenizer_path must be a *vocab.txt or a dir with tokenizer.json");
            return NULL;
        }
    }
    const char *tok_key = is_wp ? vocab_file : tok_dir;

    /* Cache lookup (keyed on model dir + resolved tokenizer + opts). */
    rp_onnx_embed_handle_t *hit = rp_onnx_embed_cache_get(model_path, tok_key, opts_in);
    if (hit) return hit;

    rp_onnx_embed_handle_t *h = calloc(1, sizeof(*h));
    if (!h) {
        if (err) snprintf(err, errlen, "rp_onnx_embed_load: oom");
        return NULL;
    }
    h->model_path     = strdup(model_path);
    h->tokenizer_path = strdup(tok_key);
    h->onnx_file      = strdup(onnx_file);
    h->opts           = *opts_in;
    h->opt_qprefix    = opts_in->query_prefix   ? strdup(opts_in->query_prefix)   : NULL;
    h->opt_pprefix    = opts_in->passage_prefix ? strdup(opts_in->passage_prefix) : NULL;
    /* Repoint the opts strings into our copies so they outlive the caller's arg. */
    h->opts.query_prefix   = h->opt_qprefix;
    h->opts.passage_prefix = h->opt_pprefix;
    h->opts.tokenizer_path = h->tokenizer_path;
    /* Snapshot the AS-REQUESTED opts NOW, before dir-mode auto-config (below)
     * mutates h->opts.  key_opts is the cache dedup key: two callers passing
     * the same sql.set() share this handle+cache even though their effective
     * opts get auto-filled to model-specific token ids after load. */
    h->key_opts = h->opts;
    /* mutex + doccache are initialized unconditionally so the cleanup path
     * below may always destroy them (destroying an UNinitialized mutex is UB) */
    pthread_mutex_init(&h->mtx, NULL);
    rp_doccache_init(&h->doc_cache, RP_DOCCACHE_DEFAULT_CAP);
    if (!h->model_path || !h->tokenizer_path || !h->onnx_file ||
        (opts_in->query_prefix   && !h->opt_qprefix) ||
        (opts_in->passage_prefix && !h->opt_pprefix)) goto oom;

    /* Directory self-config defaults: bos/eos by tokenizer family, no id offset
     * (native tokenizers emit final ids), normalize on, pooling from the config
     * (opts.pooling still overrides). Must run BEFORE discover_io (it reads pooling). */
    if (dir_mode) {
        h->opts.pad_id    = is_wp ? 0 : 1;        /* [PAD] / <pad> (masked out anyway) */
        h->opts.bos_id    = is_wp ? 101 : 0;      /* [CLS] / <s> */
        h->opts.eos_id    = is_wp ? 102 : 2;      /* [SEP] / </s> */
        h->opts.id_offset = 0;
        h->opts.normalize = 1;
        if (h->opts.pooling == 0) h->opts.pooling = disc_pooling ? disc_pooling : 1;
    }

    /* Session opts: single-threaded intra-op — the SQL path runs in MANY
     * concurrent threads/processes (server workers, index-build threads),
     * so chunk batching + that outer concurrency provide the parallelism;
     * per-session core pools would oversubscribe.  (The JS initEmbed/
     * initRerank handles differ: their sessions default to ORT's full
     * core pool — one interactive call at a time.)  A CUDA build
     * (onnx_cuda_ep_available(): the bundled ORT carries the CUDA EP,
     * i.e. this is rampart-onnx_cu12/_cu13) uses the GPU automatically,
     * mirroring llamaEmbed; cpu builds go straight to CPU, and macOS can
     * opt into CoreML via RAMPART_ONNX_COREML (below).  All of that is
     * silent -- the ONLY messages are the CoreML opt-in notice and a GPU
     * build that cannot use a GPU (no device / driver problem), which
     * falls back to CPU with a one-line warning per model load so it
     * can't go unnoticed. */
    onnx_session_opts so = { .intra_threads  = 1,
                             .inter_threads  = 1,
                             .graph_opt      = ONNX_OPT_DEFAULT,
                             .execution_mode = 0,
                             .use_cuda       = onnx_cuda_ep_available(),
                             .cuda_device_id = 0 };
    /* CoreML (macOS GPU / Neural Engine) is OPT-IN via RAMPART_ONNX_COREML:
     * "1"/"all" = all compute units, "gpu" = CPU+GPU, "ane" = CPU+NeuralEngine.
     * Unset/0 = the (well-benchmarked) CPU EP.  Opt-in rather than automatic
     * because CoreML re-specializes per input shape and the embed path feeds
     * ragged batches -- whether it wins is model-dependent; the env knob makes
     * an A/B on a serving box a restart, not a rebuild. */
    if (!so.use_cuda && onnx_coreml_ep_available()) {
        const char *cml = getenv("RAMPART_ONNX_COREML");
        if (cml && *cml && strcmp(cml, "0") != 0) {
            so.use_coreml = 1;
            /* "1"/"ane" => CPUAndNeuralEngine (the safe, working backend);
             * "gpu"/"all" selectable for testing -- the MPSGraph backend
             * ABORTS on this model class on current macOS (see
             * parse_session_opts). */
            if      (!strcasecmp(cml, "gpu")) so.coreml_units = 1;
            else if (!strcasecmp(cml, "all")) so.coreml_units = 0;
            else                              so.coreml_units = 2;  /* 1/ane */
            onnx_log_note("rampart-onnx embed: %s: CoreML EP enabled "
                    "(RAMPART_ONNX_COREML=%s)\n", model_path, cml);   /* info, not a warning */
        }
    }
    char serr[256] = {0};
    h->sess = onnx_session_create(h->onnx_file, &so, serr, sizeof serr);
    if (!h->sess && (so.use_cuda || so.use_coreml)) {
        onnx_warn("rampart-onnx embed: %s: %s build but no usable "
                "GPU (%s); using CPU\n", model_path,
                so.use_cuda ? "CUDA" : "CoreML", serr);
        so.use_cuda = 0;
        so.use_coreml = 0;
        serr[0] = '\0';
        h->sess = onnx_session_create(h->onnx_file, &so, serr, sizeof serr);
    }
    if (!h->sess) {
        if (err) snprintf(err, errlen, "rp_onnx_embed_load: session_create: %s", serr);
        goto cleanup;
    }

    if (rp_onnx_discover_io(h, err, errlen) != 0) goto cleanup;

    /* Native tokenizer -- no rampart-sentencepiece dependency. */
    if (is_wp) {
        h->wp_tok = onnx_wp_create(vocab_file, 1, 1, 1, err, errlen);
        if (!h->wp_tok) goto cleanup;
    } else {
        h->sp_tok = onnx_sp_create(tok_dir, err, errlen);
        if (!h->sp_tok) goto cleanup;
    }

    /* Per-chunk token window (long text is chunked with 1/8 overlap, not
     * truncated): explicit max_tokens wins (uncapped); else the model's
     * discovered positional capacity (llamacpp n_ctx_train parity, capped
     * 8192 in rp_discover_window); else 512.  The chunk-batch default is a
     * MEMORY cap and follows where the session actually landed after the
     * GPU-first/CPU-fallback dance above: 32 on CUDA (VRAM-conservative,
     * matching the JS layer's gpu default), 64 on CPU (system RAM). */
    h->n_ctx = opts_in->max_tokens > 0 ? opts_in->max_tokens
             : (disc_win > 0 ? disc_win : 512);
    h->mcb   = opts_in->max_chunk_batch > 0 ? opts_in->max_chunk_batch
             : (so.use_cuda ? 32 : 64);

    h->refcount = 1;
    /* May return an equivalent handle that won a concurrent load race
     * (ours is then destroyed) -- callers must use the return value. */
    return rp_onnx_embed_cache_put(h);

oom:
    if (err) snprintf(err, errlen, "rp_onnx_embed_load: oom");
cleanup:
    if (h->wp_tok) onnx_wp_destroy(h->wp_tok);
    if (h->sp_tok) onnx_sp_destroy(h->sp_tok);
    if (h->sess)   onnx_session_destroy(h->sess);
    rp_doccache_destroy(&h->doc_cache);
    free(h->opt_qprefix);
    free(h->opt_pprefix);
    free(h->model_path);
    free(h->tokenizer_path);
    free(h->onnx_file);
    pthread_mutex_destroy(&h->mtx);
    free(h);
    return NULL;
}

int rp_onnx_embed_dim(void *handle)
{
    if (!handle) return 0;
    return ((rp_onnx_embed_handle_t *)handle)->vec_dim;
}

/* Resize this handle's doc-result cache (cap == 0 disables it).  Backs
 * sql.set({onnxEmbed:{...}, likevCache:N}).  Shared handle => process-wide
 * for that model+opts. */
void rp_onnx_embed_set_cache_cap(void *handle, size_t cap)
{
    if (!handle) return;
    rp_doccache_set_cap(&((rp_onnx_embed_handle_t *)handle)->doc_cache, cap);
}

/* Refcount decrement.  v1 never actually frees; embed models are heavy
 * and lifetime matches process lifetime.  Matches rampart-llamacpp's
 * rp_embed_release convention. */
void rp_onnx_embed_release(void *handle)
{
    if (!handle) return;
    rp_onnx_embed_handle_t *h = (rp_onnx_embed_handle_t *)handle;
    pthread_mutex_lock(&g_onnx_embed_cache_lock);
    h->refcount--;
    /* No free path; see comment above. */
    pthread_mutex_unlock(&g_onnx_embed_cache_lock);
}

/* ============================================================
 * Chunked + batched embedding core (C hot path).
 *
 * Long text is split into overlapping token windows (1/8 overlap, like
 * rampart-llamacpp's embed chunker) and ALL windows of a document ride
 * through the model in batched onnx_session_run calls -- sub-batched by
 * `mcb` so a many-chunk document can't blow VRAM/RAM.  Per-chunk vectors
 * are pooled + L2-normalized here in C; avgVec is the L2-normalized mean
 * of the (unit) chunk vectors and `coherence` = the average pairwise cosine
 * between them (clamped [0,1], k-independent) -- ~1 for
 * a topically coherent doc (avgVec trustworthy), ~0 for a diffuse
 * one (prefer the per-chunk vecs for search; avgVec still fine as a
 * coarse sharding/clustering address).
 *
 * Shared by the C ABI (rp_onnx_embed_text / rp_onnx_embed_doc) and by the
 * JS initEmbed methods (via the _embedDoc/_embedBatch duk bindings), so
 * the interpreter never touches per-token or per-dimension loops.
 * ============================================================ */

/* Split content ids into overlapping windows, each wrapped [bos] ids+off [eos].
 * win_tokens is the FULL per-chunk budget (bos/eos count against it).  Returns
 * the chunk count k (>=1) and fills *out_seqs and *out_lens (malloc'd array of
 * malloc'd seqs); 0 on OOM. */
static size_t onnx_chunk_ids(const int64_t *ids, size_t n_ids,
                             int64_t bos, int64_t eos, int64_t id_offset,
                             int win_tokens,
                             int64_t ***out_seqs, size_t **out_lens)
{
    size_t sp = (bos >= 0 ? 1 : 0) + (eos >= 0 ? 1 : 0);
    long win = (long)win_tokens - (long)sp;
    if (win < 1) win = 1;
    long ov = win / 8, stride = win - ov;
    if (stride < 1) stride = win;

    size_t k = (n_ids <= (size_t)win) ? 1
             : 1 + (n_ids - (size_t)win + (size_t)stride - 1) / (size_t)stride;
    int64_t **seqs = calloc(k, sizeof(*seqs));
    size_t   *lens = calloc(k, sizeof(*lens));
    if (!seqs || !lens) { free(seqs); free(lens); return 0; }

    for (size_t c = 0; c < k; c++) {
        size_t start = c * (size_t)stride;
        size_t n = n_ids > start ? n_ids - start : 0;
        if (n > (size_t)win) n = (size_t)win;
        int64_t *s = malloc(((n + sp) ? (n + sp) : 1) * sizeof(int64_t));
        if (!s) {
            while (c) free(seqs[--c]);
            free(seqs); free(lens);
            return 0;
        }
        size_t m = 0;
        if (bos >= 0) s[m++] = bos;
        for (size_t i = 0; i < n; i++) s[m++] = ids[start + i] + id_offset;
        if (eos >= 0) s[m++] = eos;
        seqs[c] = s; lens[c] = m;
    }
    *out_seqs = seqs; *out_lens = lens;
    return k;
}

static void onnx_free_seqs(int64_t **seqs, size_t *lens, size_t k)
{
    if (seqs) { for (size_t i = 0; i < k; i++) free(seqs[i]); free(seqs); }
    free(lens);
}

/* Everything onnx_run_seqs needs to feed + pool one model. */
typedef struct {
    onnx_session *sess;
    const char *ids_name;    /* required */
    const char *mask_name;   /* NULL if the model has no attention_mask input */
    const char *tt_name;     /* NULL if no token_type_ids input */
    const char *out_name;    /* the chosen output */
    int  out_pooled;         /* 1 = out_name is a pre-pooled 2-D [b,dim] output */
    int  pooling;            /* pooling a 3-D output: 2 = cls, else masked mean */
    int  normalize;          /* L2-normalize each per-seq vector */
    int64_t pad_id;
    int  mcb;                /* max seqs per session_run (activation-memory cap) */
    const size_t *seg1;      /* optional, per-seq: token index where segment 1
                              * (the document, in a BERT cross-encoder pair)
                              * begins -- token_type_ids become 1 from there to
                              * the end of the real tokens.  NULL = all zeros. */
} onnx_seqrun;

/* Run n_seqs token sequences through the model -- sub-batched by cf->mcb,
 * padded + masked -- pooling one vector per sequence into a single malloc'd
 * float[n_seqs*dim] block.  Returns dim (>0) on success; 0 on error (err
 * filled, *out_vecs NULL). */
static int onnx_run_seqs(const onnx_seqrun *cf,
                         int64_t *const *seqs, const size_t *lens, size_t n_seqs,
                         float **out_vecs, char *err, size_t errlen)
{
    *out_vecs = NULL;
    if (!n_seqs) { if (err) snprintf(err, errlen, "onnx_run_seqs: no sequences"); return 0; }
    int mcb = cf->mcb > 0 ? cf->mcb : 1;
    float *vecs = NULL;
    int dim = 0;

    /* `cur` is the ADAPTIVE sub-batch cap: it starts at cf->mcb and is
     * halved whenever a Run fails (typically the CUDA BFC arena failing
     * to allocate activation memory for a large batch x long-window
     * shape), then the same span is retried.  It never grows back within
     * this call, so one big document settles to what the device can hold
     * instead of erroring out. */
    int cur = mcb;
    for (size_t base = 0; base < n_seqs; /* advanced at loop end */) {
        size_t bn = n_seqs - base;
        if (bn > (size_t)cur) bn = (size_t)cur;
        /* SHAPE BUCKETING.  Every distinct (bn x maxlen) shape allocates
         * distinct-size activation buffers; on the CUDA EP a stream of
         * per-document shapes splinters the BFC arena until even tiny
         * allocations fail (observed as a permanent AllocateRawInternal
         * spiral after ~20s of wikipedia traffic).  Quantizing both dims
         * to a small fixed vocabulary lets the arena reuse the same
         * buffers forever:
         *   - bn floors to a power of two (fewer sequences per run --
         *     the remainder is picked up next iteration);
         *   - the padded length rounds UP to a power of two <= 512, and
         *     to a multiple of 64 above that (full-window chunks all
         *     share one shape).  Padding is masked out, so results are
         *     unchanged; the pad columns cost some throughput, far less
         *     than the OOM spiral. */
        {
            size_t p2 = 1;
            while (p2 * 2 <= bn) p2 *= 2;
            bn = p2;
        }
        size_t maxlen = 1;
        for (size_t i = 0; i < bn; i++) if (lens[base + i] > maxlen) maxlen = lens[base + i];
        if (maxlen <= 512) {
            size_t l2 = 8;
            while (l2 < maxlen) l2 *= 2;
            maxlen = l2;
        } else {
            maxlen = ((maxlen + 63) / 64) * 64;
        }

        size_t cells = bn * maxlen;
        int64_t *flat = malloc(cells * sizeof(int64_t));
        int64_t *mask = cf->mask_name ? malloc(cells * sizeof(int64_t)) : NULL;
        int64_t *tt   = cf->tt_name   ? calloc(cells, sizeof(int64_t)) : NULL;
        if (!flat || (cf->mask_name && !mask) || (cf->tt_name && !tt)) {
            free(flat); free(mask); free(tt); free(vecs);
            if (err) snprintf(err, errlen, "onnx_run_seqs: oom");
            return 0;
        }
        for (size_t i = 0; i < bn; i++) {
            const int64_t *s = seqs[base + i];
            size_t n = lens[base + i];
            for (size_t j = 0; j < maxlen; j++) {
                int on = j < n;
                flat[i * maxlen + j] = on ? s[j] : cf->pad_id;
                if (mask) mask[i * maxlen + j] = on ? 1 : 0;
                if (tt && cf->seg1 && on && j >= cf->seg1[base + i])
                    tt[i * maxlen + j] = 1;   /* BERT pair: doc segment */
            }
        }

        int64_t shape2[2] = { (int64_t)bn, (int64_t)maxlen };
        onnx_value_in feeds[3];
        size_t n_feeds = 0;
        feeds[n_feeds++] = (onnx_value_in){ cf->ids_name, ONNX_DT_INT64, shape2, 2, flat, cells * sizeof(int64_t) };
        if (mask) feeds[n_feeds++] = (onnx_value_in){ cf->mask_name, ONNX_DT_INT64, shape2, 2, mask, cells * sizeof(int64_t) };
        if (tt)   feeds[n_feeds++] = (onnx_value_in){ cf->tt_name,   ONNX_DT_INT64, shape2, 2, tt,   cells * sizeof(int64_t) };

        onnx_value_out *outs = NULL;
        size_t n_outs = 0;
        int rc = onnx_session_run(cf->sess, feeds, n_feeds, NULL, 0, &outs, &n_outs, err, errlen);
        free(flat); free(mask); free(tt);
        if (rc != 0 || !outs || !n_outs) {
            if (outs) onnx_run_free(outs, n_outs);
            if (bn > 1) {
                /* allocation-or-other Run failure on a multi-sequence
                 * batch: halve the cap and retry this span. */
                cur = (int)(bn / 2);
                continue;
            }
            free(vecs);
            return 0;
        }

        const onnx_value_out *o = NULL;
        for (size_t i = 0; i < n_outs; i++)
            if (outs[i].name && cf->out_name && !strcmp(outs[i].name, cf->out_name)) { o = &outs[i]; break; }
        if (!o || o->n_dims < 2) {
            if (err) snprintf(err, errlen, "onnx_run_seqs: output '%s' missing/bad shape",
                              cf->out_name ? cf->out_name : "(null)");
            onnx_run_free(outs, n_outs); free(vecs);
            return 0;
        }
        if (!vecs) {
            int64_t d = o->shape[o->n_dims - 1];
            if (d <= 0) {
                if (err) snprintf(err, errlen, "onnx_run_seqs: dynamic output dim");
                onnx_run_free(outs, n_outs);
                return 0;
            }
            dim = (int)d;
            vecs = malloc(n_seqs * (size_t)dim * sizeof(float));
            if (!vecs) {
                if (err) snprintf(err, errlen, "onnx_run_seqs: oom");
                onnx_run_free(outs, n_outs);
                return 0;
            }
        }

        const float *hbuf = (const float *)o->data;
        int64_t sl = (o->n_dims >= 3) ? o->shape[1] : 1;   /* padded seq len in the output */
        if (sl <= 0) sl = (int64_t)maxlen;
        for (size_t i = 0; i < bn; i++) {
            float *v = vecs + (base + i) * (size_t)dim;
            if (cf->out_pooled) {
                memcpy(v, hbuf + i * (size_t)dim, (size_t)dim * sizeof(float));
            } else if (cf->pooling == 2) {   /* cls: token 0 of this row */
                memcpy(v, hbuf + (i * (size_t)sl) * (size_t)dim, (size_t)dim * sizeof(float));
            } else {                          /* masked mean over the real tokens */
                size_t n = lens[base + i];
                if ((int64_t)n > sl) n = (size_t)sl;
                if (!n) n = 1;
                for (int d = 0; d < dim; d++) v[d] = 0.0f;
                for (size_t j = 0; j < n; j++) {
                    const float *row = hbuf + (i * (size_t)sl + j) * (size_t)dim;
                    for (int d = 0; d < dim; d++) v[d] += row[d];
                }
                for (int d = 0; d < dim; d++) v[d] /= (float)n;
            }
            if (cf->normalize) {
                double n2 = 0.0;
                for (int d = 0; d < dim; d++) n2 += (double)v[d] * (double)v[d];
                float inv = n2 > 0.0 ? (float)(1.0 / sqrt(n2)) : 1.0f;
                for (int d = 0; d < dim; d++) v[d] *= inv;
            }
        }
        onnx_run_free(outs, n_outs);
        base += bn;
    }
    *out_vecs = vecs;
    return dim;
}

/* avgVec + coherence over k chunk vectors.  Returns malloc'd float[dim].
 * vecs_are_unit: the chunk vecs were L2-normalized (the default) -- then
 * avg = normalize(mean); coherence = avg pairwise cosine.  If not unit, avg = raw
 * mean (unnormalized, honoring normalize:0) and coherence is computed
 * from unit-scaled copies so it stays a meaningful [0,1] signal. */
static float *onnx_embed_avg(const float *vecs, size_t k, int dim,
                             int vecs_are_unit, float *out_coh)
{
    float *avg = malloc((size_t)dim * sizeof(float));
    if (!avg) return NULL;
    if (k == 1) {
        memcpy(avg, vecs, (size_t)dim * sizeof(float));
        if (out_coh) *out_coh = 1.0f;
        return avg;
    }
    for (int d = 0; d < dim; d++) {
        double m = 0.0;
        for (size_t i = 0; i < k; i++) m += (double)vecs[i * (size_t)dim + d];
        avg[d] = (float)(m / (double)k);
    }
    double mag2;   /* |mean of unit chunk vecs|^2 */
    if (vecs_are_unit) {
        double n2 = 0.0;
        for (int d = 0; d < dim; d++) n2 += (double)avg[d] * (double)avg[d];
        mag2 = n2;
        float inv = n2 > 0.0 ? (float)(1.0 / sqrt(n2)) : 1.0f;
        for (int d = 0; d < dim; d++) avg[d] *= inv;
    } else {
        /* coherence from unit-scaled copies; avg itself stays the raw mean */
        double *um = calloc((size_t)dim, sizeof(double));
        if (!um) { if (out_coh) *out_coh = 0.0f; return avg; }
        for (size_t i = 0; i < k; i++) {
            const float *v = vecs + i * (size_t)dim;
            double n2 = 0.0;
            for (int d = 0; d < dim; d++) n2 += (double)v[d] * (double)v[d];
            double inv = n2 > 0.0 ? 1.0 / sqrt(n2) : 1.0;
            for (int d = 0; d < dim; d++) um[d] += (double)v[d] * inv;
        }
        double n2 = 0.0;
        for (int d = 0; d < dim; d++) { um[d] /= (double)k; n2 += um[d] * um[d]; }
        mag2 = n2;
        free(um);
    }
    /* Report coherence as the AVERAGE PAIRWISE COSINE between the unit chunk
     * vectors -- k-independent, unlike the raw |mean| (whose floor is 1/sqrt(k)).
     * Identity: |mean|^2 = 1/k + (k-1)/k * avg_pairwise_cos, inverted here and
     * clamped to [0,1] (negative = chunks actively anti-aligned; call it 0). */
    if (out_coh) {
        double cbar = ((double)k * mag2 - 1.0) / ((double)k - 1.0);
        if (cbar < 0.0) cbar = 0.0;
        if (cbar > 1.0) cbar = 1.0;
        *out_coh = (float)cbar;
    }
    return avg;
}

/* Fill an onnx_seqrun from a C-API handle. */
static void rp_onnx_handle_seqrun(rp_onnx_embed_handle_t *h, onnx_seqrun *cf)
{
    /* Zero first: onnx_seqrun has fields this filler doesn't use (seg1 --
     * the reranker's cross-encoder segment starts).  Callers stack-allocate
     * cf, and an uninitialized seg1 is read by onnx_run_seqs whenever the
     * model has token_type_ids (SIGSEGV on garbage, or -- worse -- silent
     * token_type corruption if the garbage happens to be readable).
     * read_seqrun_cfg (the JS path's filler) already memsets; this one
     * must too. */
    memset(cf, 0, sizeof *cf);
    cf->sess       = h->sess;
    cf->ids_name   = h->input_ids_name;
    cf->mask_name  = h->attn_mask_name;
    cf->tt_name    = h->token_type_name;
    cf->out_name   = h->use_pooled ? h->pooled_out_name : h->hidden_out_name;
    cf->out_pooled = h->use_pooled;
    cf->pooling    = h->opts.pooling;
    cf->normalize  = h->opts.normalize;
    cf->pad_id     = h->opts.pad_id;
    cf->mcb        = h->mcb;
}

/* ---- encode source: one tokenizer front for the doc engine -------------
 * Wraps whichever tokenizer is in play (native WordPiece / SentencePiece, or
 * a custom JS tokenizer via its duk ctx) plus the per-chunk text prefix
 * (e5-style 'passage: ' -- applied to EVERY chunk, since each chunk is an
 * independently embedded sequence). */
typedef struct {
    onnx_wp_tokenizer *wp;
    onnx_sp_tokenizer *sp;
    duk_context *ctx;        /* custom JS tokenizer: object at tok_idx ... */
    duk_idx_t    tok_idx;    /* ... with encodeIds(text) */
    const char  *prefix;     /* NULL/"" or the per-chunk prefix */
    int          with_prefix;/* apply prefix in onnx_enc_text */
} onnx_encsrc;

/* Tokenize text[0..len) (optionally prefixed) -> malloc'd content ids. */
static int onnx_enc_text(onnx_encsrc *E, const char *t, size_t l,
                         int64_t **ids, size_t *n)
{
    *ids = NULL; *n = 0;
    size_t plen = (E->with_prefix && E->prefix && E->prefix[0]) ? strlen(E->prefix) : 0;
    char *tmp = malloc(plen + l + 1);
    if (!tmp) return -1;
    if (plen) memcpy(tmp, E->prefix, plen);
    memcpy(tmp + plen, t, l);
    tmp[plen + l] = '\0';

    int rc = -1;
    if (E->wp)      rc = onnx_wp_encode(E->wp, tmp, ids, n);
    else if (E->sp) rc = onnx_sp_encode(E->sp, tmp, ids, n);
    else if (E->ctx) {
        duk_context *ctx = E->ctx;
        duk_get_prop_string(ctx, E->tok_idx, "encodeIds");
        duk_dup(ctx, E->tok_idx);
        duk_push_string(ctx, tmp);
        if (duk_pcall_method(ctx, 1) != 0) {   /* protected: a throwing
            encodeIds must surface as rc=-1, not longjmp through the
            chunker/encode loops that hold malloc'd state */
            duk_pop(ctx);
            free(tmp);
            return -1;
        }
        if (duk_is_array(ctx, -1)) {
            size_t cnt = duk_get_length(ctx, -1);
            int64_t *o = malloc((cnt ? cnt : 1) * sizeof(int64_t));
            if (o) {
                for (size_t i = 0; i < cnt; i++) {
                    duk_get_prop_index(ctx, -1, (duk_uarridx_t)i);
                    o[i] = (int64_t)duk_get_number(ctx, -1);
                    duk_pop(ctx);
                }
                *ids = o; *n = cnt; rc = 0;
            }
        }
        duk_pop(ctx);
    }
    free(tmp);
    return rc;
}

/* rp_chunk_count_fn: count RAW tokens for a piece (no prefix -- the prefix
 * budget is subtracted from the chunker's window once, up front). */
static size_t onnx_enc_count(void *user, const char *t, size_t l)
{
    onnx_encsrc *E = (onnx_encsrc *)user;
    int saved = E->with_prefix;
    E->with_prefix = 0;
    int64_t *ids = NULL; size_t n = 0;
    int rc = onnx_enc_text(E, t, l, &ids, &n);
    E->with_prefix = saved;
    free(ids);
    return rc ? (size_t)-1 : n;
}

/* Per-vector chunk info returned to callers (byte span in the ORIGINAL text +
 * the actual token count of the embedded sequence, specials included). */
typedef struct { size_t start, end, n_tokens; } rp_onnx_embed_span;
RP_DOCCACHE_ASSERT_SPAN_LAYOUT(rp_onnx_embed_span);   /* cast to/from the doc cache */

/* Everything embedTextTo* / rp_onnx_embed_doc need in one call:
 * structure-aware chunking (rp-chunker) -> per-chunk encode (+prefix) ->
 * token-window fallback for oversized chunks -> batched runs -> pooled,
 * normalized per-vector floats + spans + avgVec + coherence.
 * On success returns dim (>0) and fills R (caller frees R->vecs/avg/chunks);
 * 0 on failure (err filled). */
typedef struct {
    float  *vecs;                /* k * dim */
    size_t  k;
    float  *avg;                 /* dim */
    float   coh;
    rp_onnx_embed_span *chunks;  /* k entries (window sub-chunks share a span) */
} onnx_docres;

/* Inject per-document prefix ids into a built sequence [bos] body [eos],
 * trimming the BODY's tail tokens to stay within win.  Returns the new
 * malloc'd seq (frees the old) and updates *len; NULL on oom (old seq
 * freed).  Deliberately never changes the NUMBER of sequences: chunk
 * boundaries, spans and k are identical with or without the prefix -- it
 * changes only what the model sees.  A full-window chunk loses its last
 * pn tokens of embedding input; abstract()'s span recomputation at query
 * time (which doesn't know the prefix) stays exact. */
static int64_t *onnx_seq_inject_prefix(int64_t *seq, size_t *len,
                                       const int64_t *pids, size_t pn,
                                       int64_t bos, int64_t eos, int64_t off,
                                       int win)
{
    if (!pn) return seq;
    size_t nsp_b = (bos >= 0) ? 1 : 0, nsp_e = (eos >= 0) ? 1 : 0;
    size_t body = *len - nsp_b - nsp_e;
    /* guard: win smaller than the specials (maxTokens:1) must not
     * wrap `room' to SIZE_MAX and emit an uncapped sequence */
    size_t room = ((size_t)win > nsp_b + nsp_e)
                  ? (size_t)win - nsp_b - nsp_e : 0;
    size_t pn_use = pn > room ? room : pn;
    size_t keep = body;
    if (pn_use >= room)            keep = 0;
    else if (body + pn_use > room) keep = room - pn_use;
    size_t nlen = nsp_b + pn_use + keep + nsp_e;
    int64_t *ns = malloc((nlen ? nlen : 1) * sizeof(int64_t));
    if (!ns) { free(seq); return NULL; }
    size_t m = 0;
    if (nsp_b) ns[m++] = bos;
    for (size_t i = 0; i < pn_use; i++) ns[m++] = pids[i] + off;
    for (size_t i = 0; i < keep; i++)  ns[m++] = seq[nsp_b + i];
    if (nsp_e) ns[m++] = eos;
    free(seq);
    *len = nlen;
    return ns;
}

static int onnx_embed_doc_run(const onnx_seqrun *cf, onnx_encsrc *E,
                              const char *text, size_t tlen,
                              const char *dpfx, size_t dpfx_len,
                              int64_t bos, int64_t eos, int64_t off, int win,
                              int split_mode, int min_tokens, int pack,
                              int sentence_split,
                              int want_avg, int normalize,
                              onnx_docres *R, char *err, size_t errlen)
{
    memset(R, 0, sizeof *R);
    size_t nsp = (bos >= 0 ? 1 : 0) + (eos >= 0 ? 1 : 0);

    /* prefix token budget (counted once; boundary effects are absorbed by the
     * oversized-chunk safety windowing below) */
    size_t pfx_n = 0;
    if (E->prefix && E->prefix[0]) {
        int64_t *pids = NULL; size_t pn = 0;
        onnx_encsrc PE = *E; PE.with_prefix = 0;
        if (onnx_enc_text(&PE, E->prefix, strlen(E->prefix), &pids, &pn) == 0) pfx_n = pn;
        free(pids);
    }
    long cwin = (long)win - (long)nsp - (long)pfx_n;
    if (cwin < 1) cwin = 1;

    /* Per-document prefix (e.g. the article title): tokenized once here,
     * injected into each FINAL sequence below.  Unlike the static
     * E->prefix, it takes no part in cwin, chunking, or sub-windowing --
     * see onnx_seq_inject_prefix for why. */
    int64_t *dp_ids = NULL; size_t dp_n = 0;
    if (dpfx && dpfx_len) {
        onnx_encsrc PE = *E; PE.with_prefix = 0;
        if (onnx_enc_text(&PE, dpfx, dpfx_len, &dp_ids, &dp_n) != 0) {
            snprintf(err, errlen, "prefix tokenization failed");
            return 0;
        }
    }

    /* 1) structure-aware chunking of the RAW text */
    rp_chunk_opts co = { (int)cwin, min_tokens, pack, split_mode, sentence_split };
    rp_chunk_span *spans = NULL;
    size_t nspan = 0;
    if (tlen == 0) {
        spans = calloc(1, sizeof *spans);          /* empty doc -> [bos,eos] */
        if (!spans) { free(dp_ids); snprintf(err, errlen, "oom"); return 0; }
        nspan = 1;
    } else if (rp_chunk_text(text, tlen, &co, onnx_enc_count, E, &spans, &nspan) != 0
               || nspan == 0) {
        free(spans);
        free(dp_ids);
        snprintf(err, errlen, "chunking/tokenization failed");
        return 0;
    }

    /* 2) encode each chunk (with prefix); token-window any that still exceed
     * the budget (mis-estimates, oversized paragraphs, unstructured text) */
    int64_t **seqs = NULL; size_t *lens = NULL, *sidx = NULL;
    size_t nseq = 0, cap = 0;
    E->with_prefix = 1;
    for (size_t i = 0; i < nspan; i++) {
        int64_t *ids = NULL; size_t n = 0;
        if (onnx_enc_text(E, text + spans[i].start,
                          spans[i].end - spans[i].start, &ids, &n) != 0) {
            snprintf(err, errlen, "tokenization failed (chunk %zu)", i);
            goto fail;
        }
        int64_t **ws = NULL; size_t *wl = NULL;
        size_t wk;
        if (n + nsp <= (size_t)win) {
            /* single seq [bos] ids+off [eos] */
            wk = 1;
            ws = calloc(1, sizeof *ws);
            wl = calloc(1, sizeof *wl);
            int64_t *s = ws ? malloc(((n + nsp) ? (n + nsp) : 1) * sizeof(int64_t)) : NULL;
            if (!ws || !wl || !s) { free(s); free(ws); free(wl); free(ids); snprintf(err, errlen, "oom"); goto fail; }
            size_t m = 0;
            if (bos >= 0) s[m++] = bos;
            for (size_t j = 0; j < n; j++) s[m++] = ids[j] + off;
            if (eos >= 0) s[m++] = eos;
            ws[0] = s; wl[0] = m;
        } else {
            wk = onnx_chunk_ids(ids, n, bos, eos, off, win, &ws, &wl);
            if (!wk) { free(ids); snprintf(err, errlen, "oom"); goto fail; }
        }
        free(ids);
        for (size_t j = 0; j < wk; j++) {
            if (nseq == cap) {
                size_t nc = cap ? cap * 2 : 16;
                int64_t **ns = realloc(seqs, nc * sizeof(*ns));
                size_t   *nl = realloc(lens, nc * sizeof(*nl));
                size_t   *ni = realloc(sidx, nc * sizeof(*ni));
                if (ns) seqs = ns;
                if (nl) lens = nl;
                if (ni) sidx = ni;
                if (!ns || !nl || !ni) { onnx_free_seqs(ws, wl, wk); snprintf(err, errlen, "oom"); goto fail; }
                cap = nc;
            }
            if (dp_n && cf) {   /* embedding runs only; spans-only mode
                                 * needn't touch the seqs (k already fixed) */
                ws[j] = onnx_seq_inject_prefix(ws[j], &wl[j], dp_ids, dp_n,
                                               bos, eos, off, win);
                if (!ws[j]) { onnx_free_seqs(ws, wl, wk); snprintf(err, errlen, "oom"); goto fail; }
            }
            seqs[nseq] = ws[j];
            ws[j] = NULL;      /* ownership moved: a later onnx_free_seqs(ws,..)
                                * on the error path must not double-free it */
            lens[nseq] = wl[j];
            sidx[nseq] = i;
            nseq++;
        }
        free(ws); free(wl);   /* seq buffers now owned by seqs[] */
    }

    /* 3) batched runs -> pooled per-seq vectors.  spans_only (cf ==
     * NULL) skips the model entirely: the caller wants just the byte
     * spans the embed would produce (abstract()'s snippet lookup) --
     * tokenize + chunk above is all that's needed, and it's µs-cheap. */
    {
        float *vecs = NULL;
        int dim = 1;                      /* spans_only success marker */
        if (cf) {
            dim = onnx_run_seqs(cf, seqs, lens, nseq, &vecs, err, errlen);
            if (!dim) goto fail;
        }

        R->vecs = vecs;
        R->k    = nseq;
        R->chunks = malloc(nseq * sizeof(*R->chunks));
        if (!R->chunks) { snprintf(err, errlen, "oom"); goto fail_res; }
        for (size_t j = 0; j < nseq; j++) {
            const rp_chunk_span *s = &spans[sidx[j]];
            R->chunks[j] = (rp_onnx_embed_span){ s->start, s->end, lens[j] };
        }
        if (cf && want_avg) {
            R->avg = onnx_embed_avg(vecs, nseq, dim, normalize, &R->coh);
            if (!R->avg) { snprintf(err, errlen, "oom"); goto fail_res; }
        }
        onnx_free_seqs(seqs, lens, nseq);
        free(sidx);
        free(spans);
        free(dp_ids);
        return dim;
fail_res:
        free(R->vecs); free(R->chunks); free(R->avg);
        memset(R, 0, sizeof *R);
    }
fail:
    onnx_free_seqs(seqs, lens, nseq);
    free(sidx);
    free(spans);
    free(dp_ids);
    return 0;
}

/* Chunked document embed.  On success returns dim (>0):
 *   *out_vecs      = malloc'd float[k*dim]  (per-chunk vectors, row-major)
 *   *out_k         = k (number of chunks)
 *   *out_avg       = malloc'd float[dim]    (combined document vector)
 *   *out_coherence = avg pairwise cosine between unit chunk vecs, [0,1]
 *                    (k-independent; 1.0 when k==1)
 *   *out_chunks    = malloc'd rp_onnx_embed_span[k]: byte span of each chunk
 *                    in the input text (+ its token count).  Token-window
 *                    sub-chunks of an oversized/unstructured region share
 *                    that region's span.
 * Any of the out pointers may be NULL if the caller doesn't want that part.
 * On failure returns 0 (outputs zeroed). */
size_t rp_onnx_embed_doc(void *handle, const char *text, size_t tlen,
                         const char *prefix, size_t plen,
                         float **out_vecs, size_t *out_k,
                         float **out_avg, float *out_coherence,
                         rp_onnx_embed_span **out_chunks)
{
    if (out_vecs) *out_vecs = NULL;
    if (out_k) *out_k = 0;
    if (out_avg) *out_avg = NULL;
    if (out_coherence) *out_coherence = 0.0f;
    if (out_chunks) *out_chunks = NULL;
    if (!handle || !text || tlen == 0) return 0;
    if (!prefix) plen = 0;
    rp_onnx_embed_handle_t *h = (rp_onnx_embed_handle_t *)handle;

    /* The doc-result cache stores the FULL result {vecs, avg, coh,
     * chunks}, so every caller -- chunkembed (which wants the spans too)
     * included -- shares one model run per text. */
    {
        float *cv = NULL, *ca = NULL, cc = 0.0f;
        rp_doccache_span *cs = NULL;
        size_t ck = 0; int cd = 0;
        if (rp_doccache_get(&h->doc_cache, text, tlen, prefix, plen,
                               out_vecs ? &cv : NULL, &ck, &cd,
                               out_avg  ? &ca : NULL, &cc,
                               out_chunks ? &cs : NULL)) {
            if (out_vecs)      *out_vecs      = cv;
            if (out_k)         *out_k         = ck;
            if (out_avg)       *out_avg       = ca;
            if (out_coherence) *out_coherence = cc;
            if (out_chunks)    *out_chunks    = (rp_onnx_embed_span *)cs;
            return (size_t)cd;
        }
    }

    char err[256] = {0};
    if (onnx_session_ensure_runnable(h->sess, err, sizeof err) != 0) {
        onnx_warn("rp_onnx_embed_doc: %s\n", err);
        return 0;
    }

    onnx_seqrun cf;
    rp_onnx_handle_seqrun(h, &cf);
    onnx_encsrc E;
    memset(&E, 0, sizeof E);
    E.wp = h->wp_tok;
    E.sp = h->sp_tok;
    E.prefix = h->opts.passage_prefix;   /* SQL callers are passage-side */

    /* Always compute avg+coh so a later embed()/chunkavg()/
     * chunkcoherence() on the same text is served from cache. */
    onnx_docres R;
    int dim = onnx_embed_doc_run(&cf, &E, text, tlen, prefix, plen,
                                 h->opts.bos_id, h->opts.eos_id, h->opts.id_offset,
                                 h->n_ctx,
                                 h->opts.split_mode, h->opts.min_split_tokens,
                                 h->opts.pack_paragraphs,
                                 h->opts.sentence_split,
                                 1 /* want_avg */,
                                 h->opts.normalize, &R, err, sizeof err);
    if (!dim) {
        if (err[0]) onnx_warn("rp_onnx_embed_doc: %s\n", err);
        return 0;
    }

    rp_doccache_put(&h->doc_cache, text, tlen, prefix, plen,
                       R.vecs, R.k, dim, R.avg, R.coh,
                       (const rp_doccache_span *)R.chunks);

    if (out_vecs) *out_vecs = R.vecs; else free(R.vecs);
    if (out_k) *out_k = R.k;
    if (out_avg) *out_avg = R.avg; else free(R.avg);
    if (out_coherence) *out_coherence = R.coh;
    if (out_chunks) *out_chunks = R.chunks; else free(R.chunks);
    return (size_t)dim;
}

/* Spans-only variant of rp_onnx_embed_doc: returns the byte spans the
 * doc embed would produce for `text` WITHOUT running the model
 * (tokenize + chunk only).  Deterministic w.r.t. the handle's chunking
 * params, so the spans line up 1:1 with the vectors a chunkembed() of
 * the same text stored.  On success returns k (>= 1) and sets
 * *out_spans (malloc'd, caller frees); 0 on failure. */
size_t rp_onnx_embed_spans(void *handle, const char *text, size_t tlen,
                           rp_onnx_embed_span **out_spans)
{
    if (out_spans) *out_spans = NULL;
    if (!handle || !text || tlen == 0 || !out_spans) return 0;
    rp_onnx_embed_handle_t *h = (rp_onnx_embed_handle_t *)handle;

    onnx_encsrc E;
    memset(&E, 0, sizeof E);
    E.wp = h->wp_tok;
    E.sp = h->sp_tok;
    E.prefix = h->opts.passage_prefix;

    char err[256] = {0};
    onnx_docres R;
    int rc = onnx_embed_doc_run(NULL /* spans only */, &E, text, tlen,
                                NULL, 0 /* per-doc prefix never affects spans */,
                                h->opts.bos_id, h->opts.eos_id,
                                h->opts.id_offset, h->n_ctx,
                                h->opts.split_mode, h->opts.min_split_tokens,
                                h->opts.pack_paragraphs,
                                h->opts.sentence_split,
                                0 /* no avg */, h->opts.normalize,
                                &R, err, sizeof err);
    if (!rc) {
        if (err[0]) onnx_warn("rp_onnx_embed_spans: %s\n", err);
        return 0;
    }
    *out_spans = R.chunks;
    free(R.vecs);   /* NULL in spans-only mode; free for symmetry */
    free(R.avg);
    return R.k;
}

/* Single combined vector for a text (the SQL path).  On success:
 * *out_vec = malloc'd float[dim] and return value = dim; 0 on failure.
 * Long text is chunked (structure-aware) + batched internally; the result
 * is the avgVec. */
size_t rp_onnx_embed_text(void *handle, const char *text, size_t tlen, float **out_vec)
{
    if (!out_vec) return 0;
    *out_vec = NULL;
    float *avg = NULL;
    size_t dim = rp_onnx_embed_doc(handle, text, tlen, NULL, 0, NULL, NULL, &avg, NULL, NULL);
    if (!dim) return 0;
    *out_vec = avg;
    return dim;
}

/* ---- Layer 2: initEmbed (embedded JS over the Layer-1 session) -----------
 * A convenience wrapper whose surface matches rampart-llamacpp's initEmbed
 * (embedTextToFp32Buf / embedTextToFp16Buf / embedTextToNumbers).  The JS here
 * only does the one-time setup that's awkward in C: model-directory discovery
 * (tokenizer / pooling / normalize via rampart.utils file IO), session +
 * tokenizer creation, and prefixing.  Every per-call hot path -- sliding-window
 * chunking, batched runs, pooling, L2-normalize, avgVec+coherence -- runs in C
 * via mod._embedDoc/_embedBatch (see the chunked/batched embedding core above).
 */

/* ------------------------------------------------------------------ *
 * Native tokenizers (onnxruntime-extensions) exposed to JS.
 * wordPieceTokenizer(vocabPath, opts) -> BertTokenizer C++ class;
 * spTokenizer(modelDir)               -> Ortx C API (reads tokenizer.json).
 * Each returns a JS object with encodeIds(text) -> [ids] (CONTENT ids,
 * no special tokens); the native handle is freed by a finalizer.
 * ------------------------------------------------------------------ */
#define ONNX_WP_PTR DUK_HIDDEN_SYMBOL("onnx_wp_ptr")
#define ONNX_SP_PTR DUK_HIDDEN_SYMBOL("onnx_sp_ptr")

static void push_ids_array(duk_context *ctx, const int64_t *ids, size_t n) {
    duk_push_array(ctx);
    for (size_t i = 0; i < n; i++) {
        duk_push_number(ctx, (double)ids[i]);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
}

static duk_ret_t wp_encode_ids(duk_context *ctx) {
    const char *text = duk_to_string(ctx, 0);
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, ONNX_WP_PTR);
    onnx_wp_tokenizer *t = (onnx_wp_tokenizer *)duk_get_pointer(ctx, -1);
    duk_pop_2(ctx);
    int64_t *ids = NULL; size_t n = 0;
    if (!t) RP_THROW(ctx, "wordPieceTokenizer.encodeIds: tokenizer was destroyed");
    if (onnx_wp_encode(t, text ? text : "", &ids, &n) != 0)
        RP_THROW(ctx, "wordPieceTokenizer.encodeIds: tokenization failed");
    push_ids_array(ctx, ids, n);
    free(ids);
    return 1;
}
static duk_ret_t wp_finalizer(duk_context *ctx) {
    /* thread copies share this raw pointer (hidden-symbol props are
     * copied verbatim; finalizers are not).  Same rule as
     * free_session_if_last(): only destroy when no other rampart
     * thread can hold a copy, else the workers' next encode is a
     * use-after-free.  (Deferred frees leak on handle churn -- the
     * module-wide tradeoff, see free_session_if_last comment.) */
    if (get_thread_count(NULL) != 1) return 0;
    if (duk_get_prop_string(ctx, 0, ONNX_WP_PTR)) {
        onnx_wp_tokenizer *t = (onnx_wp_tokenizer *)duk_get_pointer(ctx, -1);
        if (t) onnx_wp_destroy(t);
    }
    return 0;
}
static duk_ret_t onnx_wordpiece_js(duk_context *ctx) {
    const char *vocab = REQUIRE_STRING(ctx, 0, "wordPieceTokenizer: argument 1 must be a String (vocab.txt path)");
    int lower = 1, strip = 1, chinese = 1;
    if (duk_is_object(ctx, 1)) {
        /* duk_get_prop_string pushes a value (undefined if absent) either
           way, so each pop runs unconditionally — split onto its own line
           to make that explicit (-Wmisleading-indentation). */
        if (duk_get_prop_string(ctx, 1, "lowercase"))       lower   = duk_to_boolean(ctx, -1);
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 1, "stripAccents"))    strip   = duk_to_boolean(ctx, -1);
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 1, "tokenizeChinese")) chinese = duk_to_boolean(ctx, -1);
        duk_pop(ctx);
    }
    char err[512] = {0};
    onnx_wp_tokenizer *t = onnx_wp_create(vocab, lower, strip, chinese, err, sizeof err);
    if (!t) RP_THROW(ctx, "wordPieceTokenizer: %s", err[0] ? err : "failed to create");
    duk_push_object(ctx);
    duk_push_pointer(ctx, t);                             duk_put_prop_string(ctx, -2, ONNX_WP_PTR);
    duk_push_number(ctx, (double)onnx_wp_vocab_size(t));  duk_put_prop_string(ctx, -2, "vocabSize");
    duk_push_c_function(ctx, wp_encode_ids, 1);           duk_put_prop_string(ctx, -2, "encodeIds");
    duk_push_c_function(ctx, wp_finalizer, 1);            duk_set_finalizer(ctx, -2);
    return 1;
}

static duk_ret_t sp_encode_ids(duk_context *ctx) {
    const char *text = duk_to_string(ctx, 0);
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, ONNX_SP_PTR);
    onnx_sp_tokenizer *t = (onnx_sp_tokenizer *)duk_get_pointer(ctx, -1);
    duk_pop_2(ctx);
    int64_t *ids = NULL; size_t n = 0;
    if (!t) RP_THROW(ctx, "spTokenizer.encodeIds: tokenizer was destroyed");
    if (onnx_sp_encode(t, text ? text : "", &ids, &n) != 0)
        RP_THROW(ctx, "spTokenizer.encodeIds: tokenization failed");
    push_ids_array(ctx, ids, n);
    free(ids);
    return 1;
}
static duk_ret_t sp_finalizer(duk_context *ctx) {
    /* see wp_finalizer: thread copies share the pointer */
    if (get_thread_count(NULL) != 1) return 0;
    if (duk_get_prop_string(ctx, 0, ONNX_SP_PTR)) {
        onnx_sp_tokenizer *t = (onnx_sp_tokenizer *)duk_get_pointer(ctx, -1);
        if (t) onnx_sp_destroy(t);
    }
    return 0;
}
static duk_ret_t onnx_sptokenizer_js(duk_context *ctx) {
    const char *dir = REQUIRE_STRING(ctx, 0, "spTokenizer: argument 1 must be a String (model dir containing tokenizer.json)");
    char err[512] = {0};
    onnx_sp_tokenizer *t = onnx_sp_create(dir, err, sizeof err);
    if (!t) RP_THROW(ctx, "spTokenizer: %s", err[0] ? err : "failed to create");
    duk_push_object(ctx);
    duk_push_pointer(ctx, t);                    duk_put_prop_string(ctx, -2, ONNX_SP_PTR);
    duk_push_c_function(ctx, sp_encode_ids, 1);  duk_put_prop_string(ctx, -2, "encodeIds");
    duk_push_c_function(ctx, sp_finalizer, 1);   duk_set_finalizer(ctx, -2);
    return 1;
}

/* ------------------------------------------------------------------ *
 * _embedDoc / _embedBatch: duk marshaling over the chunked+batched C
 * embedding core (onnx_chunk_ids / onnx_run_seqs / onnx_embed_avg).
 * The JS initEmbed builds a small cfg object once and these do all the
 * per-call work in C -- the interpreter never loops over tokens or dims.
 * pack: 0 = Array-of-Numbers, 1 = Float32Array, 2 = Uint16Array (fp16).
 * ------------------------------------------------------------------ */

/* Fetch cfg[k] and LEAVE the string on the value stack for the duration of
 * the native call -- that pins it against refcount-free even if user JS (an
 * accessor, or a custom tokenizer's encodeIds) deletes/replaces the property
 * while we're still using the pointer. Non-strings are popped (NULL). */
static const char *cfg_str(duk_context *ctx, duk_idx_t idx, const char *k) {
    const char *s = NULL;
    if (duk_get_prop_string(ctx, idx, k) && duk_is_string(ctx, -1))
        s = duk_get_string(ctx, -1);   /* pinned: value stays on the stack */
    if (!s)
        duk_pop(ctx);
    return s;
}
static double cfg_num(duk_context *ctx, duk_idx_t idx, const char *k, double dflt) {
    double v = dflt;
    if (duk_get_prop_string(ctx, idx, k) && duk_is_number(ctx, -1))
        v = duk_get_number(ctx, -1);
    duk_pop(ctx);
    return v;
}

static void read_seqrun_cfg(duk_context *ctx, duk_idx_t idx, onnx_seqrun *cf,
                            int64_t *bos, int64_t *eos, int64_t *off, int *win)
{
    memset(cf, 0, sizeof *cf);
    cf->ids_name   = cfg_str(ctx, idx, "idsName");
    cf->mask_name  = cfg_str(ctx, idx, "maskName");
    cf->tt_name    = cfg_str(ctx, idx, "ttName");
    cf->out_name   = cfg_str(ctx, idx, "outName");
    cf->out_pooled = (int)cfg_num(ctx, idx, "outPooled", 0);
    cf->pooling    = (int)cfg_num(ctx, idx, "pooling", 1);
    cf->normalize  = (int)cfg_num(ctx, idx, "normalize", 1);
    cf->pad_id     = (int64_t)cfg_num(ctx, idx, "pad", 0);
    cf->mcb        = (int)cfg_num(ctx, idx, "mcb", 64);
    *bos = (int64_t)cfg_num(ctx, idx, "bos", -1);
    *eos = (int64_t)cfg_num(ctx, idx, "eos", -1);
    *off = (int64_t)cfg_num(ctx, idx, "off", 0);
    *win = (int)cfg_num(ctx, idx, "win", 512);
}

/* Content ids for `text` via the tokenizer object at tok_idx: native wp/sp
 * handle -> C encode; anything else -> call its JS encodeIds(text).  Returns
 * malloc'd ids (caller frees), NULL on failure. */
static int64_t *tokobj_encode_ids(duk_context *ctx, duk_idx_t tok_idx,
                                  const char *text, size_t *n_out)
{
    onnx_wp_tokenizer *wp = NULL;
    onnx_sp_tokenizer *sp = NULL;
    if (duk_get_prop_string(ctx, tok_idx, ONNX_WP_PTR))
        wp = (onnx_wp_tokenizer *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);
    if (!wp) {
        if (duk_get_prop_string(ctx, tok_idx, ONNX_SP_PTR))
            sp = (onnx_sp_tokenizer *)duk_get_pointer(ctx, -1);
        duk_pop(ctx);
    }
    int64_t *ids = NULL;
    size_t   n = 0;
    if (wp || sp) {
        int rc = wp ? onnx_wp_encode(wp, text, &ids, &n)
                    : onnx_sp_encode(sp, text, &ids, &n);
        if (rc != 0) return NULL;
    } else {
        /* custom JS tokenizer: ids = tok.encodeIds(text) (protected --
         * callers hold malloc'd sequence arrays across this call) */
        duk_get_prop_string(ctx, tok_idx, "encodeIds");
        duk_dup(ctx, tok_idx);
        duk_push_string(ctx, text);
        if (duk_pcall_method(ctx, 1) != 0) { duk_pop(ctx); return NULL; }
        if (!duk_is_array(ctx, -1)) { duk_pop(ctx); return NULL; }
        n = duk_get_length(ctx, -1);
        ids = malloc((n ? n : 1) * sizeof(int64_t));
        if (!ids) { duk_pop(ctx); return NULL; }
        for (size_t i = 0; i < n; i++) {
            duk_get_prop_index(ctx, -1, (duk_uarridx_t)i);
            ids[i] = (int64_t)duk_get_number(ctx, -1);
            duk_pop(ctx);
        }
        duk_pop(ctx);
    }
    *n_out = n;
    return ids;
}

static void push_packed_vec(duk_context *ctx, const float *v, int dim, int pack)
{
    if (pack == 2) {
        uint16_t *o = (uint16_t *)duk_push_fixed_buffer(ctx, (duk_size_t)dim * 2);
        rpvec_f32_to_f16(v, o, (size_t)dim);
        duk_push_buffer_object(ctx, -1, 0, (duk_size_t)dim * 2, DUK_BUFOBJ_UINT16ARRAY);
        duk_remove(ctx, -2);
    } else if (pack == 1) {
        float *o = (float *)duk_push_fixed_buffer(ctx, (duk_size_t)dim * 4);
        memcpy(o, v, (size_t)dim * sizeof(float));
        duk_push_buffer_object(ctx, -1, 0, (duk_size_t)dim * 4, DUK_BUFOBJ_FLOAT32ARRAY);
        duk_remove(ctx, -2);
    } else {
        duk_push_array(ctx);
        for (int d = 0; d < dim; d++) {
            duk_push_number(ctx, (double)v[d]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)d);
        }
    }
}

static onnx_session *embed_get_session(duk_context *ctx, duk_idx_t idx)
{
    onnx_session *s = NULL;
    if (duk_get_prop_string(ctx, idx, ONNX_PTR))
        s = (onnx_session *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);
    if (!s || onnx_session_is_destroyed(s))
        RP_THROW(ctx, "embed: session destroyed");
    char err[512] = {0};
    if (onnx_session_ensure_runnable(s, err, sizeof err) != 0)
        RP_THROW(ctx, "embed: %s", err);
    return s;
}

/* mod._embedDoc(sessObj, tokObj, cfg, text, pack, prefix)
 *   -> { vecs:[...], avgVec:..., coherence:Number,
 *        chunks:[{start,end,tokens,text}, ...] }   (one entry per vector)
 * Full document: structure-aware chunking (rp-chunker) + batched runs, all in
 * C (onnx_embed_doc_run).  pack: 0 numbers, 1 fp32, 2 fp16.  prefix (optional
 * string) is applied to EVERY chunk at encode time; chunk offsets are byte
 * offsets into the UNPREFIXED text. */
static duk_ret_t onnx_embed_doc_js(duk_context *ctx)
{
    onnx_session *sess = embed_get_session(ctx, 0);
    duk_size_t tlen = 0;
    const char *text = duk_require_lstring(ctx, 3, &tlen);
    int pack = duk_get_int(ctx, 4);
    const char *prefix = duk_is_string(ctx, 5) ? duk_get_string(ctx, 5) : NULL;

    /* empty text: same { vecs: [] } shape as rampart-llamacpp's embedTextTo* */
    if (tlen == 0) {
        duk_push_object(ctx);
        duk_push_array(ctx);
        duk_put_prop_string(ctx, -2, "vecs");
        return 1;
    }

    onnx_seqrun cf;
    int64_t bos, eos, off;
    int win;
    read_seqrun_cfg(ctx, 2, &cf, &bos, &eos, &off, &win);
    cf.sess = sess;
    if (!cf.ids_name || !cf.out_name)
        RP_THROW(ctx, "embed: bad cfg (idsName/outName)");
    int split   = (int)cfg_num(ctx, 2, "split", RP_CHUNK_AUTO);
    int minTok  = (int)cfg_num(ctx, 2, "minTok", 0);
    int packPar = (int)cfg_num(ctx, 2, "packPara", 0);
    int sentSpl = (int)cfg_num(ctx, 2, "sentSplit", 0);

    onnx_encsrc E;
    memset(&E, 0, sizeof E);
    if (duk_get_prop_string(ctx, 1, ONNX_WP_PTR))
        E.wp = (onnx_wp_tokenizer *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);
    if (!E.wp) {
        if (duk_get_prop_string(ctx, 1, ONNX_SP_PTR))
            E.sp = (onnx_sp_tokenizer *)duk_get_pointer(ctx, -1);
        duk_pop(ctx);
    }
    if (!E.wp && !E.sp) { E.ctx = ctx; E.tok_idx = 1; }   /* custom JS tokenizer */
    E.prefix = prefix;

    char err[512] = {0};
    onnx_docres R;
    int dim = onnx_embed_doc_run(&cf, &E, text, (size_t)tlen, NULL, 0,
                                 bos, eos, off, win,
                                 split, minTok, packPar, sentSpl,
                                 1, cf.normalize, &R, err, sizeof err);
    if (!dim) RP_THROW(ctx, "embed: %s", err[0] ? err : "run failed");

    duk_push_object(ctx);
    duk_push_array(ctx);
    for (size_t i = 0; i < R.k; i++) {
        push_packed_vec(ctx, R.vecs + i * (size_t)dim, dim, pack);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    duk_put_prop_string(ctx, -2, "vecs");
    push_packed_vec(ctx, R.avg, dim, pack);
    duk_put_prop_string(ctx, -2, "avgVec");
    duk_push_number(ctx, (double)R.coh);
    duk_put_prop_string(ctx, -2, "coherence");
    duk_push_array(ctx);
    for (size_t i = 0; i < R.k; i++) {
        duk_push_object(ctx);
        duk_push_number(ctx, (double)R.chunks[i].start);
        duk_put_prop_string(ctx, -2, "start");
        duk_push_number(ctx, (double)R.chunks[i].end);
        duk_put_prop_string(ctx, -2, "end");
        duk_push_number(ctx, (double)R.chunks[i].n_tokens);
        duk_put_prop_string(ctx, -2, "tokens");
        duk_push_lstring(ctx, text + R.chunks[i].start,
                         (duk_size_t)(R.chunks[i].end - R.chunks[i].start));
        duk_put_prop_string(ctx, -2, "text");
        /* oversized: this vector is one of SEVERAL token windows over a span
         * that exceeded the model window (sub-windows share their span, so
         * the one-vector-per-paragraph invariant did not hold there) */
        if ((i > 0 && R.chunks[i-1].start == R.chunks[i].start
                   && R.chunks[i-1].end   == R.chunks[i].end)
            || (i + 1 < R.k && R.chunks[i+1].start == R.chunks[i].start
                            && R.chunks[i+1].end   == R.chunks[i].end)) {
            duk_push_true(ctx);
            duk_put_prop_string(ctx, -2, "oversized");
        }
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    duk_put_prop_string(ctx, -2, "chunks");
    free(R.vecs);
    free(R.avg);
    free(R.chunks);
    return 1;
}

/* mod._embedBatch(sessObj, tokObj, cfg, [texts], pack[, full[, prefix]]) ->
 * [vec, ...].  One single-window (truncated, eos kept) vector per text; all
 * texts ride in batched runs together.  With `full` truthy, returns the
 * embedTextTo* doc shape instead: { vecs, avgVec, coherence,
 * chunks:[{text,tokens},...] } -- used by the custom split:function()
 * handles, where each caller-supplied string is ONE chunk, always: in full
 * mode an OVERSIZED string is not truncated but embedded through the doc
 * core and represented by its embed()-style combined (average) vector, so
 * N strings in -> N vectors out (mirrors chunkembed(strlst)). */
static duk_ret_t onnx_embed_batch_js(duk_context *ctx)
{
    onnx_session *sess = embed_get_session(ctx, 0);
    if (!duk_is_array(ctx, 3))
        RP_THROW(ctx, "embed: texts must be an Array");
    int pack = duk_get_int(ctx, 4);
    int full = duk_get_boolean_default(ctx, 5, 0);
    const char *pfx = duk_is_string(ctx, 6) ? duk_get_string(ctx, 6) : NULL;

    onnx_seqrun cf;
    int64_t bos, eos, off;
    int win;
    read_seqrun_cfg(ctx, 2, &cf, &bos, &eos, &off, &win);
    cf.sess = sess;
    if (!cf.ids_name || !cf.out_name)
        RP_THROW(ctx, "embed: bad cfg (idsName/outName)");

    size_t n = (size_t)duk_get_length(ctx, 3);
    if (!n) {
        if (full) {
            duk_push_object(ctx);
            duk_push_array(ctx);
            duk_put_prop_string(ctx, -2, "vecs");
        } else
            duk_push_array(ctx);
        return 1;
    }

    size_t sp = (bos >= 0 ? 1 : 0) + (eos >= 0 ? 1 : 0);
    size_t maxc = (size_t)win > sp ? (size_t)win - sp : 1;   /* content cap per text */

    int64_t **seqs = calloc(n, sizeof(*seqs));
    size_t   *lens = calloc(n, sizeof(*lens));
    size_t   *toks = full ? calloc(n, sizeof(*toks)) : NULL;
    if (!seqs || !lens || (full && !toks)) {
        free(seqs); free(lens); free(toks);
        RP_THROW(ctx, "embed: oom");
    }

    for (size_t i = 0; i < n; i++) {
        duk_get_prop_index(ctx, 3, (duk_uarridx_t)i);
        if (!duk_is_string(ctx, -1)) {
            duk_pop(ctx);
            onnx_free_seqs(seqs, lens, i);
            free(toks);
            RP_THROW(ctx, "embed: texts[%zu] is not a String", i);
        }
        if (pfx) {                       /* encode-only prefix per chunk */
            duk_push_string(ctx, pfx);
            duk_swap_top(ctx, -2);
            duk_concat(ctx, 2);
        }
        const char *t = duk_get_string(ctx, -1);
        size_t nid = 0;
        int64_t *ids = tokobj_encode_ids(ctx, 1, t, &nid);
        duk_pop(ctx);
        if (!ids) {
            onnx_free_seqs(seqs, lens, i);
            free(toks);
            RP_THROW(ctx, "embed: tokenization failed (text %zu)", i);
        }
        if (nid > maxc) {
            if (full) {
                /* oversized caller-chunk: keep ALL content -- embed the
                 * string through the doc core and use its combined
                 * (average) vector; still exactly one vector for this
                 * element.  Marked below via toks[i] (real, uncapped
                 * token count) + the oversized flag. */
                seqs[i] = NULL;           /* filled from the doc run later */
                lens[i] = 0;
                toks[i] = nid;
                free(ids);
                continue;
            }
            nid = maxc;                   /* bare mode: truncate, keep eos */
        }
        if (toks) toks[i] = nid;
        int64_t *s = malloc(((nid + sp) ? (nid + sp) : 1) * sizeof(int64_t));
        if (!s) { free(ids); onnx_free_seqs(seqs, lens, i); free(toks); RP_THROW(ctx, "embed: oom"); }
        size_t m = 0;
        if (bos >= 0) s[m++] = bos;
        for (size_t j = 0; j < nid; j++) s[m++] = ids[j] + off;
        if (eos >= 0) s[m++] = eos;
        free(ids);
        seqs[i] = s;
        lens[i] = m;
    }

    /* compact the fitting sequences for one batched run; remember slots */
    size_t nfit = 0, i2;
    size_t *slot = NULL;
    if (full) {
        slot = calloc(n ? n : 1, sizeof(*slot));
        if (!slot) { onnx_free_seqs(seqs, lens, n); free(toks); RP_THROW(ctx, "embed: oom"); }
        for (i2 = 0; i2 < n; i2++)
            if (seqs[i2]) { seqs[nfit] = seqs[i2]; lens[nfit] = lens[i2]; slot[nfit] = i2; nfit++; }
        for (i2 = nfit; i2 < n; i2++) { seqs[i2] = NULL; lens[i2] = 0; }
    } else
        nfit = n;

    char err[512] = {0};
    float *fitvecs = NULL;
    int dim = 0;
    if (nfit) {
        dim = onnx_run_seqs(&cf, seqs, lens, nfit, &fitvecs, err, sizeof err);
        if (!dim) {
            onnx_free_seqs(seqs, lens, nfit);
            free(toks); free(slot);
            RP_THROW(ctx, "embed: %s", err[0] ? err : "run failed");
        }
    }
    onnx_free_seqs(seqs, lens, nfit);

    float *vecs = NULL;
    if (!full) {
        vecs = fitvecs;
        fitvecs = NULL;
    } else {
        /* assemble per-element vectors: batched results into their slots,
         * then a doc-core combined vector for each oversized element */
        onnx_encsrc E;
        memset(&E, 0, sizeof E);
        if (duk_get_prop_string(ctx, 1, ONNX_WP_PTR))
            E.wp = (onnx_wp_tokenizer *)duk_get_pointer(ctx, -1);
        duk_pop(ctx);
        if (!E.wp) {
            if (duk_get_prop_string(ctx, 1, ONNX_SP_PTR))
                E.sp = (onnx_sp_tokenizer *)duk_get_pointer(ctx, -1);
            duk_pop(ctx);
        }
        if (!E.wp && !E.sp) { E.ctx = ctx; E.tok_idx = 1; }
        E.prefix = pfx;

        int split   = (int)cfg_num(ctx, 2, "split", RP_CHUNK_AUTO);
        int minTokC = (int)cfg_num(ctx, 2, "minTok", 0);
        int packPar = (int)cfg_num(ctx, 2, "packPara", 0);
        int sentSpl = (int)cfg_num(ctx, 2, "sentSplit", 0);
        size_t fi;

        char *fitmask = calloc(n ? n : 1, 1);
        if (!fitmask) { free(fitvecs); free(toks); free(slot); RP_THROW(ctx, "embed: oom"); }
        for (fi = 0; fi < nfit; fi++) fitmask[slot[fi]] = 1;

        for (i2 = 0; i2 < n; i2++) {
            if (fitmask[i2]) continue;
            /* oversized: doc-core run, take the combined vector */
            duk_get_prop_index(ctx, 3, (duk_uarridx_t)i2);
            duk_size_t slen2 = 0;
            const char *s2 = duk_get_lstring(ctx, -1, &slen2);
            onnx_docres R;
            int d2 = onnx_embed_doc_run(&cf, &E, s2, (size_t)slen2, NULL, 0,
                                        bos, eos, off, win,
                                        split, minTokC, packPar, sentSpl,
                                        1, cf.normalize, &R, err, sizeof err);
            duk_pop(ctx);
            if (!d2) {
                free(fitvecs); free(vecs); free(toks); free(slot); free(fitmask);
                RP_THROW(ctx, "embed: %s (oversized chunk %lu)",
                         err[0] ? err : "run failed", (unsigned long)i2);
            }
            if (!dim) dim = d2;
            if (!vecs) {
                /* calloc: overflow-checked n*size (a multi-million-element
                 * user split array could wrap a bare multiply on 32-bit) */
                vecs = calloc(n, (size_t)dim * sizeof(float));
                if (!vecs) { free(R.vecs); free(R.avg); free(R.chunks);
                             free(fitvecs); free(toks); free(slot); free(fitmask);
                             RP_THROW(ctx, "embed: oom"); }
            }
            memcpy(vecs + i2 * (size_t)dim, R.avg, (size_t)dim * sizeof(float));
            free(R.vecs); free(R.avg); free(R.chunks);
        }
        if (!vecs) {
            vecs = calloc(n ? n : 1, (size_t)dim * sizeof(float));
            if (!vecs) { free(fitvecs); free(toks); free(slot); free(fitmask); RP_THROW(ctx, "embed: oom"); }
        }
        for (fi = 0; fi < nfit; fi++)
            memcpy(vecs + slot[fi] * (size_t)dim,
                   fitvecs + fi * (size_t)dim, (size_t)dim * sizeof(float));
        free(fitvecs);
        free(slot);
        free(fitmask);
    }

    if (!full) {
        duk_push_array(ctx);
        for (size_t i = 0; i < n; i++) {
            push_packed_vec(ctx, vecs + i * (size_t)dim, dim, pack);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
        free(vecs);
        return 1;
    }

    float coh = 0.0f;
    float *avg = onnx_embed_avg(vecs, n, dim, cf.normalize, &coh);
    if (!avg) { free(vecs); free(toks); RP_THROW(ctx, "embed: oom"); }

    duk_push_object(ctx);
    duk_push_array(ctx);
    for (size_t i = 0; i < n; i++) {
        push_packed_vec(ctx, vecs + i * (size_t)dim, dim, pack);
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    duk_put_prop_string(ctx, -2, "vecs");
    push_packed_vec(ctx, avg, dim, pack);
    duk_put_prop_string(ctx, -2, "avgVec");
    duk_push_number(ctx, (double)coh);
    duk_put_prop_string(ctx, -2, "coherence");
    /* chunks: the caller's strings ARE the chunks -- text + (truncated)
     * token count; no byte spans (the text needn't appear in any parent
     * document verbatim) */
    duk_push_array(ctx);
    {
        size_t sp2 = (bos >= 0 ? 1 : 0) + (eos >= 0 ? 1 : 0);
        size_t maxc2 = (size_t)win > sp2 ? (size_t)win - sp2 : 1;
        for (size_t i = 0; i < n; i++) {
            duk_push_object(ctx);
            duk_get_prop_index(ctx, 3, (duk_uarridx_t)i);
            duk_put_prop_string(ctx, -2, "text");
            duk_push_number(ctx, (double)toks[i]);
            duk_put_prop_string(ctx, -2, "tokens");
            if (toks[i] > maxc2) {
                /* this single vector is the embed()-style average over the
                 * string's sub-windows */
                duk_push_true(ctx);
                duk_put_prop_string(ctx, -2, "oversized");
            }
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
    }
    duk_put_prop_string(ctx, -2, "chunks");
    free(vecs);
    free(avg);
    free(toks);
    return 1;
}

/* mod._rerank(sessObj, tokObj, cfg, query, docsArray) -> [{document,score,index}]
 * sorted by score desc.  Cross-encoder scoring: the query is encoded ONCE, each
 * doc pair-seq is built in C ([bos] q [eos eos] d [eos], doc ids truncated to
 * keep the trailing eos when cfg.win is set), and ALL docs ride through the
 * model batched via onnx_run_seqs (cfg.outName = the logit output, outPooled=1,
 * dim 1; normalize forced off).  cfg.sigmoid applies 1/(1+e^-x). */
static duk_ret_t onnx_rerank_js(duk_context *ctx)
{
    onnx_session *sess = embed_get_session(ctx, 0);
    const char *query = duk_require_string(ctx, 3);
    if (!duk_is_array(ctx, 4))
        RP_THROW(ctx, "rerank: docs must be an Array");

    onnx_seqrun cf;
    int64_t bos, eos, off;
    int win;
    read_seqrun_cfg(ctx, 2, &cf, &bos, &eos, &off, &win);
    cf.sess = sess;
    cf.out_pooled = 1;      /* logit output: [b,1] -- "pooling" = row copy */
    cf.normalize  = 0;      /* never normalize scores */
    int sigmoid = (int)cfg_num(ctx, 2, "sigmoid", 1);
    /* pair template: 1 = BERT ([CLS] q [SEP] d [SEP] + token_type_ids 0/1),
     * 0 = RoBERTa/XLM-R ([bos] q [eos eos] d [eos], token types all 0) */
    int bert    = (int)cfg_num(ctx, 2, "bertPair", 0);
    if (!cf.ids_name || !cf.out_name)
        RP_THROW(ctx, "rerank: bad cfg (idsName/outName)");

    size_t nq = 0;
    int64_t *qids = tokobj_encode_ids(ctx, 1, query, &nq);
    if (!qids) RP_THROW(ctx, "rerank: query tokenization failed");

    size_t n = (size_t)duk_get_length(ctx, 4);
    if (!n) { free(qids); duk_push_array(ctx); return 1; }

    int64_t **seqs = calloc(n, sizeof(*seqs));
    size_t   *lens = calloc(n, sizeof(*lens));
    size_t   *seg1 = bert ? calloc(n, sizeof(*seg1)) : NULL;
    if (!seqs || !lens || (bert && !seg1)) {
        free(qids); free(seqs); free(lens); free(seg1);
        RP_THROW(ctx, "rerank: oom");
    }

    /* fixed part: [bos] q+off [eos ( eos unless bertPair )]; per-doc: d+off [eos] */
    size_t fixed = (bos >= 0 ? 1 : 0) + nq + (eos >= 0 ? (bert ? 1 : 2) : 0);
    for (size_t i = 0; i < n; i++) {
        duk_get_prop_index(ctx, 4, (duk_uarridx_t)i);
        const char *dtxt = duk_get_string(ctx, -1);   /* no throw: we hold seqs[] */
        if (!dtxt) {
            duk_pop(ctx);
            free(qids); free(seg1); onnx_free_seqs(seqs, lens, i);
            RP_THROW(ctx, "rerank: docs[%zu] is not a String", i);
        }
        size_t nd = 0;
        int64_t *dids = tokobj_encode_ids(ctx, 1, dtxt, &nd);
        duk_pop(ctx);
        if (!dids) {
            free(qids); free(seg1); onnx_free_seqs(seqs, lens, i);
            RP_THROW(ctx, "rerank: doc tokenization failed (doc %zu)", i);
        }
        /* truncate the DOC ids so the pair fits the window, keeping the final eos */
        if (win > 0) {
            size_t budget = (size_t)win > fixed + (eos >= 0 ? 1 : 0)
                          ? (size_t)win - fixed - (eos >= 0 ? 1 : 0) : 0;
            if (nd > budget) nd = budget;
        }
        size_t cap = fixed + nd + (eos >= 0 ? 1 : 0);
        int64_t *s = malloc((cap ? cap : 1) * sizeof(int64_t));
        if (!s) {
            free(dids); free(qids); free(seg1); onnx_free_seqs(seqs, lens, i);
            RP_THROW(ctx, "rerank: oom");
        }
        size_t m = 0;
        if (bos >= 0) s[m++] = bos;
        for (size_t j = 0; j < nq; j++) s[m++] = qids[j] + off;
        if (eos >= 0) { s[m++] = eos; if (!bert) s[m++] = eos; }
        if (seg1) seg1[i] = m;              /* doc segment starts here */
        for (size_t j = 0; j < nd; j++) s[m++] = dids[j] + off;
        if (eos >= 0) s[m++] = eos;
        free(dids);
        if (win > 0 && m > (size_t)win) {                  /* oversized query fallback */
            m = (size_t)win;
            if (eos >= 0 && m > 0) s[m - 1] = eos;         /* keep the final [SEP]/eos */
        }
        seqs[i] = s;
        lens[i] = m;
    }
    free(qids);
    cf.seg1 = seg1;                          /* NULL unless bertPair */

    char err[512] = {0};
    float *vecs = NULL;
    int dim = onnx_run_seqs(&cf, seqs, lens, n, &vecs, err, sizeof err);
    onnx_free_seqs(seqs, lens, n);
    free(seg1);
    if (!dim) RP_THROW(ctx, "rerank: %s", err[0] ? err : "run failed");

    /* scores = first logit per row; sigmoid if configured; sort desc by score */
    typedef struct { float score; size_t idx; } rr_ent;
    rr_ent *ent = malloc(n * sizeof(*ent));
    if (!ent) { free(vecs); RP_THROW(ctx, "rerank: oom"); }
    for (size_t i = 0; i < n; i++) {
        float s = vecs[i * (size_t)dim];
        ent[i].score = sigmoid ? (float)(1.0 / (1.0 + exp(-(double)s))) : s;
        ent[i].idx   = i;
    }
    free(vecs);
    /* insertion sort (docs lists are small; stable, no qsort ctx gymnastics) */
    for (size_t i = 1; i < n; i++) {
        rr_ent e = ent[i];
        size_t j = i;
        while (j > 0 && ent[j-1].score < e.score) { ent[j] = ent[j-1]; j--; }
        ent[j] = e;
    }

    duk_push_array(ctx);
    for (size_t i = 0; i < n; i++) {
        duk_push_object(ctx);
        duk_get_prop_index(ctx, 4, (duk_uarridx_t)ent[i].idx);
        duk_put_prop_string(ctx, -2, "document");
        duk_push_number(ctx, (double)ent[i].score);
        duk_put_prop_string(ctx, -2, "score");
        duk_push_number(ctx, (double)ent[i].idx);
        duk_put_prop_string(ctx, -2, "index");
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    free(ent);
    return 1;
}

/* mod._snacWeave(tokens, off, span, applyOffset) -> [c0, c1, c2]
 * Demux Orpheus 7-codes-per-frame tokens into the SNAC 3-level hierarchy
 * (lvl0=[i], lvl1=[i+1,i+4], lvl2=[i+2,i+3,i+5,i+6]); with applyOffset, first
 * strip the Orpheus per-slot offset (tok - off - (i%7)*span). */
static duk_ret_t onnx_snac_weave_js(duk_context *ctx)
{
    if (!duk_is_array(ctx, 0))
        RP_THROW(ctx, "snac: tokens must be an Array");
    double off  = duk_get_number_default(ctx, 1, 0);
    double span = duk_get_number_default(ctx, 2, 0);
    int    apply = duk_get_boolean_default(ctx, 3, 0);

    size_t nt = (size_t)duk_get_length(ctx, 0);
    size_t nf = nt / 7;
    int64_t *t = malloc((nt ? nt : 1) * sizeof(int64_t));
    if (!t) RP_THROW(ctx, "snac: oom");
    for (size_t i = 0; i < nt; i++) {
        duk_get_prop_index(ctx, 0, (duk_uarridx_t)i);
        double v = duk_get_number(ctx, -1);
        duk_pop(ctx);
        if (apply) v = v - off - (double)(i % 7) * span;
        t[i] = (int64_t)v;
    }

    /* demux in C: lvl0 gets slot 0; lvl1 slots 1,4; lvl2 slots 2,3,5,6 */
    int64_t *c0 = malloc((nf ? nf : 1) * sizeof(int64_t));
    int64_t *c1 = malloc((nf ? nf : 1) * 2 * sizeof(int64_t));
    int64_t *c2 = malloc((nf ? nf : 1) * 4 * sizeof(int64_t));
    if (!c0 || !c1 || !c2) { free(t); free(c0); free(c1); free(c2); RP_THROW(ctx, "snac: oom"); }
    for (size_t f = 0; f < nf; f++) {
        const int64_t *fr = t + f * 7;
        c0[f]         = fr[0];
        c1[f*2]       = fr[1];  c1[f*2+1] = fr[4];
        c2[f*4]       = fr[2];  c2[f*4+1] = fr[3];
        c2[f*4+2]     = fr[5];  c2[f*4+3] = fr[6];
    }
    free(t);

    duk_push_array(ctx);                              /* [c0,c1,c2] */
    const int64_t *lvls[3] = { c0, c1, c2 };
    const size_t   lcnt[3] = { nf, nf * 2, nf * 4 };
    for (int lvl = 0; lvl < 3; lvl++) {
        duk_push_array(ctx);
        for (size_t i = 0; i < lcnt[lvl]; i++) {
            duk_push_number(ctx, (double)lvls[lvl][i]);
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)lvl);
    }
    free(c0); free(c1); free(c2);
    return 1;
}

/* mod._discover(path) -> { isDir, model, vocab, tokDir, pooling }
 * Model-directory discovery, single-sourced with the C rp_onnx_embed_load
 * dir-mode (same rp_is_dir/rp_find_suffix + duktape-JSON discovery helpers):
 * model = onnx/model.onnx | model.onnx | first *.onnx; vocab = first
 * *vocab.txt (WordPiece); tokDir = the dir itself when tokenizer.json is
 * present (SPM/BPE); pooling from 1_Pooling/config.json (0 unknown, 1 mean,
 * 2 cls). */
static duk_ret_t onnx_discover_js(duk_context *ctx)
{
    const char *path = REQUIRE_STRING(ctx, 0, "discover: argument 1 must be a String (path)");
    duk_push_object(ctx);
    int isdir = rp_is_dir(path);
    duk_push_boolean(ctx, isdir);
    duk_put_prop_string(ctx, -2, "isDir");
    if (!isdir) return 1;

    char buf[1536], cand[1536]; /* > od[1100] + dirent name (see rp_find_suffix) */
    snprintf(cand, sizeof cand, "%s/onnx/model.onnx", path);
    if (!rp_is_file(cand)) {
        snprintf(cand, sizeof cand, "%s/model.onnx", path);
        if (!rp_is_file(cand)) {
            char od[1100];
            snprintf(od, sizeof od, "%s/onnx", path);
            if (!rp_find_suffix(rp_is_dir(od) ? od : path, ".onnx", cand, sizeof cand))
                cand[0] = '\0';
        }
    }
    if (cand[0]) { duk_push_string(ctx, cand); duk_put_prop_string(ctx, -2, "model"); }

    if (rp_find_suffix(path, "vocab.txt", buf, sizeof buf)) {
        duk_push_string(ctx, buf);
        duk_put_prop_string(ctx, -2, "vocab");
    } else {
        snprintf(buf, sizeof buf, "%s/tokenizer.json", path);
        if (rp_is_file(buf)) {
            duk_push_string(ctx, path);
            duk_put_prop_string(ctx, -2, "tokDir");
        }
    }

    duk_push_int(ctx, rp_discover_pooling(ctx, path));
    duk_put_prop_string(ctx, -2, "pooling");
    duk_push_int(ctx, rp_discover_window(ctx, path));
    duk_put_prop_string(ctx, -2, "win");
    return 1;
}

/* ==================================================================
 * initRerank — ALL C.  The handle carries EVERYTHING it needs as
 * properties (session object, tokenizer object, cfg data object under
 * hidden symbols) and its methods are C functions reading from `this`.
 * No JS closures, no captured variables — so the handle survives
 * rampart's thread-copy (server preThreadFunc globals -> workers)
 * exactly like rampart-llamacpp's handles do.  A JS-factory
 * predecessor kept sess/tok/cfg in closure variables, which
 * duk_dump_function cannot serialize; copied methods then died on
 * "identifier 'mod' undefined" in worker threads.
 * ================================================================== */

#define ONNX_H_TOK  DUK_HIDDEN_SYMBOL("onnx_h_tok")   /* tokenizer object   */
#define ONNX_H_CFG  DUK_HIDDEN_SYMBOL("onnx_h_cfg")   /* plain-data config  */
#define ONNX_H_QPFX DUK_HIDDEN_SYMBOL("onnx_h_qpfx")  /* embed queryPrefix  */
#define ONNX_H_PPFX DUK_HIDDEN_SYMBOL("onnx_h_ppfx")  /* embed passagePrefix*/
#define ONNX_H_SPLITFN DUK_HIDDEN_SYMBOL("onnx_h_splitfn") /* custom split fn */

/* rr.rerank(query, doc|docs[, scoresOnly]) — state from `this`, scoring
 * via the batched core (onnx_rerank_js: [{document,score,index}] sorted
 * best-first).  Single doc -> Number; scoresOnly -> Numbers in ORIGINAL
 * doc order (llamacpp parity). */
static duk_ret_t onnx_rr_rerank(duk_context *ctx)
{
    int single      = duk_is_string(ctx, 1);
    int scores_only = duk_to_boolean(ctx, 2);   /* undefined -> false */

    duk_push_this(ctx);                              /* idx 3 */

    if (single) {
        duk_push_array(ctx);
        duk_dup(ctx, 1);
        duk_put_prop_index(ctx, -2, 0);
        duk_replace(ctx, 1);                         /* docs = [doc] */
    } else if (!duk_is_array(ctx, 1))
        RP_THROW(ctx, "rerank: second argument must be a String or an Array of Strings");

    /* assemble the core's argument list from our properties */
    duk_push_c_function(ctx, onnx_rerank_js, 5);
    duk_get_prop_string(ctx, 3, "session");
    duk_get_prop_string(ctx, 3, ONNX_H_TOK);
    duk_get_prop_string(ctx, 3, ONNX_H_CFG);
    duk_dup(ctx, 0);                                 /* query */
    duk_dup(ctx, 1);                                 /* docs  */
    duk_call(ctx, 5);

    if (single) {
        duk_get_prop_index(ctx, -1, 0);
        duk_get_prop_string(ctx, -1, "score");
        return 1;
    }
    if (scores_only) {
        duk_size_t n = duk_get_length(ctx, -1), i;
        duk_push_array(ctx);
        for (i = 0; i < n; i++) {
            duk_get_prop_index(ctx, -2, (duk_uarridx_t)i);
            duk_get_prop_string(ctx, -1, "index");
            duk_uarridx_t oi = (duk_uarridx_t)duk_get_uint(ctx, -1);
            duk_pop(ctx);
            duk_get_prop_string(ctx, -1, "score");
            duk_put_prop_index(ctx, -3, oi);
            duk_pop(ctx);                            /* entry */
        }
        return 1;
    }
    return 1;
}

/* shared by the rerank/embed/snac handles: this.session.destroy() */
static duk_ret_t onnx_handle_destroy(duk_context *ctx)
{
    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "session");
    duk_get_prop_string(ctx, -1, "destroy");
    duk_dup(ctx, -2);
    duk_call_method(ctx, 0);
    return 0;
}

/* opts.<k>: boolean defaulting to true unless explicitly false */
static int rr_opt_bool_true(duk_context *ctx, duk_idx_t opts, const char *k)
{
    int v = 1;
    if (duk_get_prop_string(ctx, opts, k) && !duk_is_undefined(ctx, -1))
        v = duk_to_boolean(ctx, -1);
    duk_pop(ctx);
    return v;
}

/* opts.<k>: integer with a default when absent */
static int rr_opt_int(duk_context *ctx, duk_idx_t opts, const char *k, int dflt)
{
    int v = dflt;
    if (duk_get_prop_string(ctx, opts, k) && duk_is_number(ctx, -1))
        v = (int)duk_get_int(ctx, -1);
    duk_pop(ctx);
    return v;
}

/* Resolve the tokenizer for an init* handle: opts.tokenizer (an SPM model
 * path via opts.spModule||require('rampart-sentencepiece'), or an object
 * with encodeIds) wins; else dir-mode auto-detect (*vocab.txt -> WordPiece,
 * tokenizer.json -> SPM/BPE).  Pushes the tokenizer object; sets *isWP. */
static void onnx_handle_tokenizer(duk_context *ctx, duk_idx_t opts_idx,
                                  duk_idx_t d_idx, int isDir,
                                  const char *path, int *isWP,
                                  const char *fname)
{
    *isWP = 0;
    if (duk_get_prop_string(ctx, opts_idx, "tokenizer") &&
        !duk_is_undefined(ctx, -1)) {
        if (duk_is_string(ctx, -1)) {
            /* an SPM model path: opts.spModule || require('rampart-sentencepiece') */
            duk_idx_t sp_path = duk_normalize_index(ctx, -1);
            if (!duk_get_prop_string(ctx, opts_idx, "spModule") ||
                !duk_is_object(ctx, -1)) {
                duk_pop(ctx);
                duk_get_global_string(ctx, "require");
                duk_push_string(ctx, "rampart-sentencepiece");
                duk_call(ctx, 1);
            }
            duk_get_prop_string(ctx, -1, "init");
            duk_dup(ctx, -2);                        /* this = the sp module */
            duk_dup(ctx, sp_path);
            duk_call_method(ctx, 1);                 /* -> tokenizer object */
            duk_remove(ctx, -2);                     /* sp module */
            duk_remove(ctx, sp_path);                /* the path string */
        }
        /* else: a tokenizer object, already on the stack */
    } else {
        duk_pop(ctx);                                /* undefined */
        if (!isDir) {
            /* File mode: rather than fail, try to discover the tokenizer NEAR the
             * model file -- its own directory first, then (for the common HF layout
             * where the .onnx lives in an onnx/ subdir) the parent directory.  Re-run
             * the same directory discovery and continue as if a directory were given.
             * (opts.tokenizer still overrides this; it's only reached when unset.) */
            char tdir[1100];
            snprintf(tdir, sizeof tdir, "%s", path);
            char *sl = strrchr(tdir, '/');
            if (sl) *sl = '\0'; else strcpy(tdir, ".");
            duk_push_c_function(ctx, onnx_discover_js, 1);
            duk_push_string(ctx, tdir);
            duk_call(ctx, 1);                         /* discover(<model's dir>) */
            if (!duk_has_prop_string(ctx, -1, "vocab") &&
                !duk_has_prop_string(ctx, -1, "tokDir")) {
                char *sl2 = strrchr(tdir, '/');
                const char *base = sl2 ? sl2 + 1 : tdir;
                if (!strcmp(base, "onnx")) {          /* .../onnx/model.onnx -> look one up */
                    if (sl2) *sl2 = '\0'; else strcpy(tdir, ".");
                    duk_pop(ctx);                     /* drop the empty discovery */
                    duk_push_c_function(ctx, onnx_discover_js, 1);
                    duk_push_string(ctx, tdir);
                    duk_call(ctx, 1);
                }
            }
            if (!duk_has_prop_string(ctx, -1, "vocab") &&
                !duk_has_prop_string(ctx, -1, "tokDir"))
                RP_THROW(ctx, "%s: no tokenizer found near '%s' -- pass opts.tokenizer "
                         "(an spm path or a tokenizer object), or place a *vocab.txt / "
                         "tokenizer.json beside the model (or in its parent dir)",
                         fname, path);
            d_idx = duk_normalize_index(ctx, -1);     /* use this discovery below */
        }
        if (duk_get_prop_string(ctx, d_idx, "vocab")) {
            /* WordPiece: mod.wordPieceTokenizer(d.vocab, {tokenizer opts}) */
            duk_push_c_function(ctx, onnx_wordpiece_js, 2);
            duk_dup(ctx, -2);
            duk_push_object(ctx);
            duk_push_boolean(ctx, rr_opt_bool_true(ctx, opts_idx, "lowercase"));
            duk_put_prop_string(ctx, -2, "lowercase");
            duk_push_boolean(ctx, rr_opt_bool_true(ctx, opts_idx, "stripAccents"));
            duk_put_prop_string(ctx, -2, "stripAccents");
            duk_push_boolean(ctx, rr_opt_bool_true(ctx, opts_idx, "tokenizeChinese"));
            duk_put_prop_string(ctx, -2, "tokenizeChinese");
            duk_call(ctx, 2);
            duk_remove(ctx, -2);                     /* the vocab path */
            *isWP = 1;
        } else {
            duk_pop(ctx);                            /* undefined vocab */
            if (!duk_get_prop_string(ctx, d_idx, "tokDir"))
                RP_THROW(ctx, "%s: no *vocab.txt or tokenizer.json in "
                         "%s -- pass opts.tokenizer", fname, path);
            duk_push_c_function(ctx, onnx_sptokenizer_js, 1);
            duk_dup(ctx, -2);
            duk_call(ctx, 1);
            duk_remove(ctx, -2);                     /* the tokDir string */
        }
    }
    duk_get_prop_string(ctx, -1, "encodeIds");
    if (!duk_is_function(ctx, -1))
        RP_THROW(ctx, "%s: tokenizer lacks encodeIds()", fname);
    duk_pop(ctx);
}

static duk_ret_t onnx_init_rerank(duk_context *ctx)
{
    onnx_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    const char *mpath = REQUIRE_STRING(ctx, 0,
        "initRerank: argument 1 must be a String (model dir or .onnx file)");
    char path[1024];
    size_t plen;
    int isWP = 0, isDir = 0;

    /* normalize opts to an object at idx 1 */
    if (duk_is_undefined(ctx, 1) || duk_is_null(ctx, 1)) {
        duk_push_object(ctx);
        duk_replace(ctx, 1);
    } else
        REQUIRE_OBJECT(ctx, 1, "initRerank: options, if present, must be an Object");

    /* strip trailing slashes (dir given as "path/") */
    plen = strlen(mpath);
    if (plen >= sizeof(path))
        RP_THROW(ctx, "initRerank: model path too long");
    strcpy(path, mpath);
    while (plen > 1 && path[plen - 1] == '/') path[--plen] = '\0';

    /* --- discovery (same C helper that backs mod._discover) --------- */
    duk_push_c_function(ctx, onnx_discover_js, 1);
    duk_push_string(ctx, path);
    duk_call(ctx, 1);
    duk_idx_t d_idx = duk_normalize_index(ctx, -1);

    duk_get_prop_string(ctx, d_idx, "isDir");
    isDir = duk_get_boolean(ctx, -1);
    duk_pop(ctx);

    /* model file: dir mode discovers it; file mode is the path itself */
    if (isDir) {
        if (!duk_get_prop_string(ctx, d_idx, "model"))
            RP_THROW(ctx, "initRerank: no .onnx model found under %s", path);
        /* leave model string on stack */
    } else
        duk_push_string(ctx, path);
    duk_idx_t model_idx = duk_normalize_index(ctx, -1);

    /* --- tokenizer --------------------------------------------------- */
    onnx_handle_tokenizer(ctx, 1, d_idx, isDir, path, &isWP, "initRerank");
    duk_idx_t tok_idx = duk_normalize_index(ctx, -1);

    /* --- session ------------------------------------------------------ */
    duk_push_c_function(ctx, onnx_init_session, 2);
    duk_push_this(ctx);              /* forward `this` so a warning inside the inner
                                      * call still lands on the caller's errMsg */
    duk_dup(ctx, model_idx);
    duk_dup(ctx, 1);
    duk_call_method(ctx, 2);
    duk_idx_t sess_idx = duk_normalize_index(ctx, -1);

    /* --- io inspection ------------------------------------------------ */
    duk_get_prop_string(ctx, sess_idx, "inputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t ins_idx = duk_normalize_index(ctx, -1);
    duk_get_prop_string(ctx, sess_idx, "outputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t outs_idx = duk_normalize_index(ctx, -1);

    int have_ids = 0, have_mask = 0, have_tt = 0;
    duk_size_t nin = duk_get_length(ctx, ins_idx), oi;
    for (oi = 0; oi < nin; oi++) {
        duk_get_prop_index(ctx, ins_idx, (duk_uarridx_t)oi);
        duk_get_prop_string(ctx, -1, "name");
        const char *nm = duk_get_string(ctx, -1);
        if (nm) {
            if (!strcmp(nm, "input_ids"))            have_ids = 1;
            else if (!strcmp(nm, "attention_mask"))  have_mask = 1;
            else if (!strcmp(nm, "token_type_ids"))  have_tt = 1;
        }
        duk_pop_2(ctx);
    }
    /* cross-encoder sanity: the score output should be [batch,1] */
    duk_get_prop_index(ctx, outs_idx, 0);
    duk_idx_t out0_idx = duk_normalize_index(ctx, -1);
    duk_get_prop_string(ctx, out0_idx, "shape");
    duk_size_t out0_dims = duk_get_length(ctx, -1);
    duk_pop(ctx);
    if (isDir && out0_dims == 3)
        RP_THROW(ctx, "initRerank: %s looks like an embedding model "
                 "(3-D output), not a cross-encoder",
                 duk_get_string(ctx, model_idx));

    /* --- cfg (plain data object: copies cleanly across threads) ------ */
    duk_push_object(ctx);
    duk_idx_t cfg_idx = duk_normalize_index(ctx, -1);
    if (have_ids)
        duk_push_string(ctx, "input_ids");
    else {
        duk_get_prop_index(ctx, ins_idx, 0);
        duk_get_prop_string(ctx, -1, "name");
        duk_remove(ctx, -2);
    }
    duk_put_prop_string(ctx, cfg_idx, "idsName");
    if (have_mask) {
        duk_push_string(ctx, "attention_mask");
        duk_put_prop_string(ctx, cfg_idx, "maskName");
    }
    if (have_tt) {
        duk_push_string(ctx, "token_type_ids");
        duk_put_prop_string(ctx, cfg_idx, "ttName");
    }
    duk_get_prop_string(ctx, out0_idx, "name");
    duk_put_prop_string(ctx, cfg_idx, "outName");
    duk_push_int(ctx, 1);
    duk_put_prop_string(ctx, cfg_idx, "outPooled");
    duk_push_int(ctx, 0);
    duk_put_prop_string(ctx, cfg_idx, "normalize");
    duk_push_int(ctx, rr_opt_int(ctx, 1, "bosId", isWP ? 101 : 0));
    duk_put_prop_string(ctx, cfg_idx, "bos");
    duk_push_int(ctx, rr_opt_int(ctx, 1, "eosId", isWP ? 102 : 2));
    duk_put_prop_string(ctx, cfg_idx, "eos");
    duk_push_int(ctx, rr_opt_int(ctx, 1, "padId", isWP ? 0 : 1));
    duk_put_prop_string(ctx, cfg_idx, "pad");
    duk_push_int(ctx, rr_opt_int(ctx, 1, "idOffset", 0));
    duk_put_prop_string(ctx, cfg_idx, "off");
    {
        /* window: explicit maxTokens > discovered positional capacity > 0 */
        int win = rr_opt_int(ctx, 1, "maxTokens", 0);
        if (!win && isDir) {
            duk_get_prop_string(ctx, d_idx, "win");
            if (duk_is_number(ctx, -1)) win = duk_get_int(ctx, -1);
            duk_pop(ctx);
        }
        duk_push_int(ctx, win);
        duk_put_prop_string(ctx, cfg_idx, "win");
    }
    {
        int gpu = onnx_cuda_ep_available();   /* auto-GPU default; explicit gpu: overrides */
        if (duk_get_prop_string(ctx, 1, "gpu")) gpu = duk_to_boolean(ctx, -1);
        duk_pop(ctx);
        duk_push_int(ctx, rr_opt_int(ctx, 1, "maxChunkBatch", gpu ? 32 : 64));
        duk_put_prop_string(ctx, cfg_idx, "mcb");
    }
    {
        /* pair template: BERT when WordPiece, RoBERTa/XLM-R otherwise;
         * opts.pairTemplate 'bert'|'roberta' overrides */
        int bert = isWP;
        if (duk_get_prop_string(ctx, 1, "pairTemplate") && duk_is_string(ctx, -1))
            bert = !strcmp(duk_get_string(ctx, -1), "bert");
        duk_pop(ctx);
        duk_push_int(ctx, bert);
        duk_put_prop_string(ctx, cfg_idx, "bertPair");
    }
    {
        int sig = 1;
        if (duk_get_prop_string(ctx, 1, "sigmoid") && !duk_is_undefined(ctx, -1))
            sig = duk_to_boolean(ctx, -1);
        duk_pop(ctx);
        duk_push_int(ctx, sig);
        duk_put_prop_string(ctx, cfg_idx, "sigmoid");
    }

    /* --- the handle: everything as properties, methods in C ---------- */
    duk_push_object(ctx);
    duk_dup(ctx, sess_idx);
    duk_put_prop_string(ctx, -2, "session");
    duk_dup(ctx, tok_idx);
    duk_put_prop_string(ctx, -2, ONNX_H_TOK);
    duk_dup(ctx, cfg_idx);
    duk_put_prop_string(ctx, -2, ONNX_H_CFG);
    duk_push_c_function(ctx, onnx_rr_rerank, 3);
    duk_put_prop_string(ctx, -2, "rerank");
    duk_push_c_function(ctx, onnx_handle_destroy, 0);
    duk_put_prop_string(ctx, -2, "destroy");
    return 1;
}

/* ==================================================================
 * initEmbed — ALL C, same handle design as initRerank: state as
 * properties (session, tokenizer, cfg, query/passage prefixes under
 * hidden symbols), methods as C functions reading `this`.  Thread-copy
 * portable.
 * ================================================================== */

/* embedTextTo{Numbers,Fp32Buf,Fp16Buf}(text[, isQuery]) — pack selects
 * the vector format; the prefix (if configured) rides from the hidden
 * prop matching isQuery.  All work happens in the chunked doc core
 * (onnx_embed_doc_js). */
static duk_ret_t onnx_emb_common(duk_context *ctx, int pack)
{
    int isQuery = duk_to_boolean(ctx, 1);
    duk_push_this(ctx);                              /* idx 2 */

    /* custom splitter (initEmbed split:function): fn(text) -> [String,...],
     * each string one single-window chunk; batched via the batch core's
     * `full` mode (avgVec/coherence/chunks like the built-in chunker, minus
     * byte spans -- the splitter's text needn't appear in the input
     * verbatim). */
    if (duk_get_prop_string(ctx, 2, ONNX_H_SPLITFN)) {
        duk_dup(ctx, 0);                             /* text */
        duk_call(ctx, 1);
        if (!duk_is_array(ctx, -1))
            RP_THROW(ctx, "embed: the split function must return an Array of Strings");
        duk_idx_t arr = duk_normalize_index(ctx, -1);
        duk_push_c_function(ctx, onnx_embed_batch_js, 7);
        duk_get_prop_string(ctx, 2, "session");
        duk_get_prop_string(ctx, 2, ONNX_H_TOK);
        duk_get_prop_string(ctx, 2, ONNX_H_CFG);
        duk_dup(ctx, arr);
        duk_push_int(ctx, pack);
        duk_push_true(ctx);                          /* full result shape */
        duk_get_prop_string(ctx, 2, isQuery ? ONNX_H_QPFX : ONNX_H_PPFX);
        duk_call(ctx, 7);
        return 1;
    }
    duk_pop(ctx);

    duk_push_c_function(ctx, onnx_embed_doc_js, 6);
    duk_get_prop_string(ctx, 2, "session");
    duk_get_prop_string(ctx, 2, ONNX_H_TOK);
    duk_get_prop_string(ctx, 2, ONNX_H_CFG);
    duk_dup(ctx, 0);                                 /* text */
    duk_push_int(ctx, pack);
    duk_get_prop_string(ctx, 2, isQuery ? ONNX_H_QPFX : ONNX_H_PPFX);
    duk_call(ctx, 6);                                /* undefined pfx = none */
    return 1;
}
static duk_ret_t onnx_emb_numbers(duk_context *ctx) { return onnx_emb_common(ctx, 0); }
static duk_ret_t onnx_emb_f32(duk_context *ctx)     { return onnx_emb_common(ctx, 1); }
static duk_ret_t onnx_emb_f16(duk_context *ctx)     { return onnx_emb_common(ctx, 2); }

/* embedTextsToNumbers(texts[, isQuery]) — one single-window vector per
 * text via the batched core; prefixes are prepended to the TEXT here
 * (the batch core has no prefix parameter). */
static duk_ret_t onnx_emb_batch_numbers(duk_context *ctx)
{
    int isQuery = duk_to_boolean(ctx, 1);
    duk_size_t n, i;

    if (!duk_is_array(ctx, 0))
        RP_THROW(ctx, "embedTextsToNumbers: first argument must be an Array of Strings");
    duk_push_this(ctx);                              /* idx 2 */
    duk_get_prop_string(ctx, 2, isQuery ? ONNX_H_QPFX : ONNX_H_PPFX); /* 3 */
    n = duk_get_length(ctx, 0);
    duk_push_array(ctx);                             /* 4: prefixed texts */
    for (i = 0; i < n; i++) {
        duk_get_prop_index(ctx, 0, (duk_uarridx_t)i);
        if (duk_is_string(ctx, 3)) {
            duk_dup(ctx, 3);
            duk_swap_top(ctx, -2);                   /* [pfx, text] */
            duk_concat(ctx, 2);
        }
        duk_put_prop_index(ctx, 4, (duk_uarridx_t)i);
    }
    duk_push_c_function(ctx, onnx_embed_batch_js, 5);
    duk_get_prop_string(ctx, 2, "session");
    duk_get_prop_string(ctx, 2, ONNX_H_TOK);
    duk_get_prop_string(ctx, 2, ONNX_H_CFG);
    duk_dup(ctx, 4);
    duk_push_int(ctx, 0);
    duk_call(ctx, 5);                                /* -> [vec, ...] */

    /* llamacpp-parity shape: [{avgVec: vec}, ...] */
    n = duk_get_length(ctx, -1);
    duk_push_array(ctx);
    for (i = 0; i < n; i++) {
        duk_push_object(ctx);
        duk_get_prop_index(ctx, -3, (duk_uarridx_t)i);
        duk_put_prop_string(ctx, -2, "avgVec");
        duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
    }
    return 1;
}

static duk_ret_t onnx_init_embed(duk_context *ctx)
{
    onnx_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    const char *mpath = REQUIRE_STRING(ctx, 0,
        "initEmbed: argument 1 must be a String (model dir or .onnx file)");
    char path[1024];
    char pooled_out[256] = "", hidden_out[256] = "";
    size_t plen;
    int isWP = 0, isDir = 0;

    if (duk_is_undefined(ctx, 1) || duk_is_null(ctx, 1)) {
        duk_push_object(ctx);
        duk_replace(ctx, 1);
    } else
        REQUIRE_OBJECT(ctx, 1, "initEmbed: options, if present, must be an Object");

    plen = strlen(mpath);
    if (plen >= sizeof(path))
        RP_THROW(ctx, "initEmbed: model path too long");
    strcpy(path, mpath);
    while (plen > 1 && path[plen - 1] == '/') path[--plen] = '\0';

    /* --- discovery ---------------------------------------------------- */
    duk_push_c_function(ctx, onnx_discover_js, 1);
    duk_push_string(ctx, path);
    duk_call(ctx, 1);
    duk_idx_t d_idx = duk_normalize_index(ctx, -1);

    duk_get_prop_string(ctx, d_idx, "isDir");
    isDir = duk_get_boolean(ctx, -1);
    duk_pop(ctx);

    if (isDir) {
        if (!duk_get_prop_string(ctx, d_idx, "model"))
            RP_THROW(ctx, "initEmbed: no .onnx model found under %s", path);
    } else
        duk_push_string(ctx, path);
    duk_idx_t model_idx = duk_normalize_index(ctx, -1);

    /* --- tokenizer ---------------------------------------------------- */
    onnx_handle_tokenizer(ctx, 1, d_idx, isDir, path, &isWP, "initEmbed");
    duk_idx_t tok_idx = duk_normalize_index(ctx, -1);

    /* --- session ------------------------------------------------------ */
    duk_push_c_function(ctx, onnx_init_session, 2);
    duk_push_this(ctx);              /* forward `this` so a warning inside the inner
                                      * call still lands on the caller's errMsg */
    duk_dup(ctx, model_idx);
    duk_dup(ctx, 1);
    duk_call_method(ctx, 2);
    duk_idx_t sess_idx = duk_normalize_index(ctx, -1);

    /* --- io inspection ------------------------------------------------ */
    duk_get_prop_string(ctx, sess_idx, "inputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t ins_idx = duk_normalize_index(ctx, -1);
    duk_get_prop_string(ctx, sess_idx, "outputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t outs_idx = duk_normalize_index(ctx, -1);

    int have_ids = 0, have_mask = 0, have_tt = 0;
    duk_size_t nin = duk_get_length(ctx, ins_idx), oi;
    for (oi = 0; oi < nin; oi++) {
        duk_get_prop_index(ctx, ins_idx, (duk_uarridx_t)oi);
        duk_get_prop_string(ctx, -1, "name");
        const char *nm = duk_get_string(ctx, -1);
        if (nm) {
            if (!strcmp(nm, "input_ids"))            have_ids = 1;
            else if (!strcmp(nm, "attention_mask"))  have_mask = 1;
            else if (!strcmp(nm, "token_type_ids"))  have_tt = 1;
        }
        duk_pop_2(ctx);
    }
    /* first float32 2-D output = pre-pooled; first 3-D = token-level */
    duk_size_t nout = duk_get_length(ctx, outs_idx);
    for (oi = 0; oi < nout; oi++) {
        duk_size_t dims;
        const char *tn, *nm;
        duk_get_prop_index(ctx, outs_idx, (duk_uarridx_t)oi);
        duk_get_prop_string(ctx, -1, "type");
        tn = duk_get_string(ctx, -1);
        duk_pop(ctx);
        if (!tn || strcmp(tn, "float32")) { duk_pop(ctx); continue; }
        duk_get_prop_string(ctx, -1, "shape");
        dims = duk_get_length(ctx, -1);
        duk_pop(ctx);
        duk_get_prop_string(ctx, -1, "name");
        nm = duk_get_string(ctx, -1);
        if (nm) {
            if (dims == 2 && !pooled_out[0])
                snprintf(pooled_out, sizeof pooled_out, "%s", nm);
            if (dims == 3 && !hidden_out[0])
                snprintf(hidden_out, sizeof hidden_out, "%s", nm);
        }
        duk_pop_2(ctx);
    }
    if (!pooled_out[0] && !hidden_out[0])
        RP_THROW(ctx, "initEmbed: no float32 2-D/3-D output to pool");

    /* pooling: opts.pooling wins; else 1_Pooling/config.json (discovery);
     * else mean.  When a pooling mode is KNOWN and the model also emits
     * token-level output, pool it ourselves rather than trusting a
     * pre-pooled 2-D output. */
    {
        int pooling_known = 0, pooling_cls = 0;
        if (duk_get_prop_string(ctx, 1, "pooling") && duk_is_string(ctx, -1)) {
            pooling_known = 1;
            pooling_cls = !strcmp(duk_get_string(ctx, -1), "cls");
        }
        duk_pop(ctx);
        if (!pooling_known && isDir) {
            duk_get_prop_string(ctx, d_idx, "pooling");
            if (duk_is_number(ctx, -1) && duk_get_int(ctx, -1) != 0) {
                pooling_known = 1;
                pooling_cls = duk_get_int(ctx, -1) == 2;
            }
            duk_pop(ctx);
        }
        if (pooling_known && hidden_out[0])
            pooled_out[0] = '\0';

        /* --- cfg ------------------------------------------------------ */
        duk_push_object(ctx);
        duk_idx_t cfg_idx = duk_normalize_index(ctx, -1);
        if (have_ids)
            duk_push_string(ctx, "input_ids");
        else {
            duk_get_prop_index(ctx, ins_idx, 0);
            duk_get_prop_string(ctx, -1, "name");
            duk_remove(ctx, -2);
        }
        duk_put_prop_string(ctx, cfg_idx, "idsName");
        if (have_mask) {
            duk_push_string(ctx, "attention_mask");
            duk_put_prop_string(ctx, cfg_idx, "maskName");
        }
        if (have_tt) {
            duk_push_string(ctx, "token_type_ids");
            duk_put_prop_string(ctx, cfg_idx, "ttName");
        }
        duk_push_string(ctx, pooled_out[0] ? pooled_out : hidden_out);
        duk_put_prop_string(ctx, cfg_idx, "outName");
        duk_push_int(ctx, pooled_out[0] ? 1 : 0);
        duk_put_prop_string(ctx, cfg_idx, "outPooled");
        duk_push_int(ctx, pooling_cls ? 2 : 1);
        duk_put_prop_string(ctx, cfg_idx, "pooling");
        {
            int norm = 1;
            if (duk_get_prop_string(ctx, 1, "normalize") &&
                !duk_is_undefined(ctx, -1))
                norm = duk_to_boolean(ctx, -1);
            duk_pop(ctx);
            duk_push_int(ctx, norm);
            duk_put_prop_string(ctx, cfg_idx, "normalize");
        }
        duk_push_int(ctx, rr_opt_int(ctx, 1, "bosId", isWP ? 101 : 0));
        duk_put_prop_string(ctx, cfg_idx, "bos");
        duk_push_int(ctx, rr_opt_int(ctx, 1, "eosId", isWP ? 102 : 2));
        duk_put_prop_string(ctx, cfg_idx, "eos");
        duk_push_int(ctx, rr_opt_int(ctx, 1, "padId", isWP ? 0 : 1));
        duk_put_prop_string(ctx, cfg_idx, "pad");
        duk_push_int(ctx, rr_opt_int(ctx, 1, "idOffset", 0));
        duk_put_prop_string(ctx, cfg_idx, "off");
        {
            /* window: explicit maxTokens (uncapped) > discovered positional
             * capacity (llamacpp n_ctx_train parity, capped 8192 in C) > 512 */
            int win = rr_opt_int(ctx, 1, "maxTokens", 0);
            if (!win && isDir) {
                duk_get_prop_string(ctx, d_idx, "win");
                if (duk_is_number(ctx, -1)) win = duk_get_int(ctx, -1);
                duk_pop(ctx);
            }
            duk_push_int(ctx, win ? win : 512);
            duk_put_prop_string(ctx, cfg_idx, "win");
        }
        {
            int gpu = 0;
            if (duk_get_prop_string(ctx, 1, "gpu")) gpu = duk_to_boolean(ctx, -1);
            duk_pop(ctx);
            duk_push_int(ctx, rr_opt_int(ctx, 1, "maxChunkBatch", gpu ? 32 : 64));
            duk_put_prop_string(ctx, cfg_idx, "mcb");
        }
        {
            /* structure-aware chunking: split 'auto'(default)|'window';
             * minTokens = paragraph fragment floor (-1 disables merging);
             * packParagraphs = pack paragraphs to the window */
            int split = 0, pp = 0;
            if (duk_get_prop_string(ctx, 1, "split") && duk_is_string(ctx, -1))
                split = !strcmp(duk_get_string(ctx, -1), "window");
            duk_pop(ctx);
            duk_push_int(ctx, split);
            duk_put_prop_string(ctx, cfg_idx, "split");
            {
                int ss = 0;
                if (duk_get_prop_string(ctx, 1, "sentenceSplit"))
                    ss = duk_to_boolean(ctx, -1);
                duk_pop(ctx);
                duk_push_int(ctx, ss);
                duk_put_prop_string(ctx, cfg_idx, "sentSplit");
            }
            duk_push_int(ctx, rr_opt_int(ctx, 1, "minTokens", 0));
            duk_put_prop_string(ctx, cfg_idx, "minTok");
            if (duk_get_prop_string(ctx, 1, "packParagraphs"))
                pp = duk_to_boolean(ctx, -1);
            duk_pop(ctx);
            duk_push_int(ctx, pp);
            duk_put_prop_string(ctx, cfg_idx, "packPara");
        }

        /* --- the handle ------------------------------------------------ */
        duk_push_object(ctx);
        duk_dup(ctx, sess_idx);
        duk_put_prop_string(ctx, -2, "session");
        duk_dup(ctx, tok_idx);
        duk_put_prop_string(ctx, -2, ONNX_H_TOK);
        duk_dup(ctx, cfg_idx);
        duk_put_prop_string(ctx, -2, ONNX_H_CFG);
        if (duk_get_prop_string(ctx, 1, "queryPrefix") && duk_is_string(ctx, -1))
            duk_put_prop_string(ctx, -2, ONNX_H_QPFX);
        else
            duk_pop(ctx);
        if (duk_get_prop_string(ctx, 1, "passagePrefix") && duk_is_string(ctx, -1))
            duk_put_prop_string(ctx, -2, ONNX_H_PPFX);
        else
            duk_pop(ctx);
        /* split: a Function replaces the built-in chunker (see
         * onnx_emb_common); a String was already folded into cfg above */
        if (duk_get_prop_string(ctx, 1, "split") && duk_is_function(ctx, -1))
            duk_put_prop_string(ctx, -2, ONNX_H_SPLITFN);
        else
            duk_pop(ctx);
        duk_push_c_function(ctx, onnx_emb_numbers, 2);
        duk_put_prop_string(ctx, -2, "embedTextToNumbers");
        duk_push_c_function(ctx, onnx_emb_f32, 2);
        duk_put_prop_string(ctx, -2, "embedTextToFp32Buf");
        duk_push_c_function(ctx, onnx_emb_f16, 2);
        duk_put_prop_string(ctx, -2, "embedTextToFp16Buf");
        duk_push_c_function(ctx, onnx_emb_batch_numbers, 2);
        duk_put_prop_string(ctx, -2, "embedTextsToNumbers");
        duk_push_c_function(ctx, onnx_handle_destroy, 0);
        duk_put_prop_string(ctx, -2, "destroy");
    }
    return 1;
}

/* ==================================================================
 * initSnacDecoder — ALL C.  Experimental SNAC audio-codec decoder;
 * same portable-handle design.  cfg carries the model's three input
 * names, the output name, and the Orpheus offset/span.
 * ================================================================== */

/* decode(codes): codes = [c0, c1, c2] (Arrays of Numbers) -> audio samples */
static duk_ret_t onnx_snac_decode(duk_context *ctx)
{
    int j;

    if (!duk_is_array(ctx, 0) || duk_get_length(ctx, 0) < 3)
        RP_THROW(ctx, "snac decode: codes must be [c0, c1, c2]");
    duk_push_this(ctx);                              /* 1 */
    duk_get_prop_string(ctx, 1, ONNX_H_CFG);         /* 2 */

    duk_get_prop_string(ctx, 1, "session");          /* 3 */
    duk_get_prop_string(ctx, -1, "run");             /* 4 */
    duk_dup(ctx, 3);                                 /* this for run() */
    duk_push_object(ctx);                            /* feeds */
    for (j = 0; j < 3; j++) {
        char key[8];
        duk_size_t clen;
        snprintf(key, sizeof key, "in%d", j);
        duk_get_prop_string(ctx, 2, key);            /* input name */
        duk_push_object(ctx);                        /* the tensor */
        duk_get_prop_index(ctx, 0, (duk_uarridx_t)j);
        clen = duk_get_length(ctx, -1);
        duk_put_prop_string(ctx, -2, "data");
        duk_push_array(ctx);
        duk_push_int(ctx, 1);
        duk_put_prop_index(ctx, -2, 0);
        duk_push_uint(ctx, (duk_uint_t)clen);
        duk_put_prop_index(ctx, -2, 1);
        duk_put_prop_string(ctx, -2, "shape");
        duk_push_string(ctx, "int64");
        duk_put_prop_string(ctx, -2, "type");
        duk_put_prop(ctx, -3);                       /* feeds[name] = tensor */
    }
    duk_call_method(ctx, 1);                         /* -> outputs object */
    duk_get_prop_string(ctx, 2, "outName");
    duk_get_prop(ctx, -2);                           /* outputs[outName] */
    duk_get_prop_string(ctx, -1, "array");
    return 1;
}

/* framesToCodes(frames) — plain 7-per-frame demux via the C weave */
static duk_ret_t onnx_snac_frames_to_codes(duk_context *ctx)
{
    duk_push_c_function(ctx, onnx_snac_weave_js, 4);
    duk_dup(ctx, 0);
    duk_push_int(ctx, 0);
    duk_push_int(ctx, 0);
    duk_push_false(ctx);
    duk_call(ctx, 4);
    return 1;
}

/* decodeFrames(frames) = decode(framesToCodes(frames)) */
static duk_ret_t onnx_snac_decode_frames(duk_context *ctx)
{
    duk_push_this(ctx);                              /* 1 */
    duk_push_c_function(ctx, onnx_snac_weave_js, 4);
    duk_dup(ctx, 0);
    duk_push_int(ctx, 0);
    duk_push_int(ctx, 0);
    duk_push_false(ctx);
    duk_call(ctx, 4);                                /* codes */
    duk_push_c_function(ctx, onnx_snac_decode, 1);
    duk_dup(ctx, 1);                                 /* this = the handle */
    duk_dup(ctx, -3);                                /* codes */
    duk_call_method(ctx, 1);
    return 1;
}

/* decodeOrpheus(tokens) = decode(weave(tokens, off, span, true)) */
static duk_ret_t onnx_snac_decode_orpheus(duk_context *ctx)
{
    duk_push_this(ctx);                              /* 1 */
    duk_get_prop_string(ctx, 1, ONNX_H_CFG);         /* 2 */
    duk_push_c_function(ctx, onnx_snac_weave_js, 4);
    duk_dup(ctx, 0);
    duk_get_prop_string(ctx, 2, "off");
    duk_get_prop_string(ctx, 2, "span");
    duk_push_true(ctx);
    duk_call(ctx, 4);                                /* codes */
    duk_push_c_function(ctx, onnx_snac_decode, 1);
    duk_dup(ctx, 1);
    duk_dup(ctx, -3);
    duk_call_method(ctx, 1);
    return 1;
}

static duk_ret_t onnx_init_snac(duk_context *ctx)
{
    onnx_errmsg_clear(ctx);   /* errMsg reflects THIS call */
    REQUIRE_STRING(ctx, 0,
        "initSnacDecoder: argument 1 must be a String (path to .onnx)");

    if (duk_is_undefined(ctx, 1) || duk_is_null(ctx, 1)) {
        duk_push_object(ctx);
        duk_replace(ctx, 1);
    } else
        REQUIRE_OBJECT(ctx, 1, "initSnacDecoder: options, if present, must be an Object");

    duk_push_c_function(ctx, onnx_init_session, 2);
    duk_push_this(ctx);              /* forward `this` (see initEmbed) */
    duk_dup(ctx, 0);
    duk_dup(ctx, 1);
    duk_call_method(ctx, 2);
    duk_idx_t sess_idx = duk_normalize_index(ctx, -1);

    duk_get_prop_string(ctx, sess_idx, "inputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t ins_idx = duk_normalize_index(ctx, -1);
    if (duk_get_length(ctx, ins_idx) < 3)
        RP_THROW(ctx, "initSnacDecoder: model must have 3 code inputs");
    duk_get_prop_string(ctx, sess_idx, "outputs");
    duk_dup(ctx, sess_idx);
    duk_call_method(ctx, 0);
    duk_idx_t outs_idx = duk_normalize_index(ctx, -1);

    /* cfg: the three input names, output name, orpheus offset/span */
    duk_push_object(ctx);
    duk_idx_t cfg_idx = duk_normalize_index(ctx, -1);
    {
        int j;
        for (j = 0; j < 3; j++) {
            char key[8];
            snprintf(key, sizeof key, "in%d", j);
            duk_get_prop_index(ctx, ins_idx, (duk_uarridx_t)j);
            duk_get_prop_string(ctx, -1, "name");
            duk_put_prop_string(ctx, cfg_idx, key);
            duk_pop(ctx);
        }
    }
    duk_get_prop_index(ctx, outs_idx, 0);
    duk_get_prop_string(ctx, -1, "name");
    duk_put_prop_string(ctx, cfg_idx, "outName");
    duk_pop(ctx);
    duk_push_int(ctx, rr_opt_int(ctx, 1, "codeOffset", 10));
    duk_put_prop_string(ctx, cfg_idx, "off");
    {
        int span = rr_opt_int(ctx, 1, "slotSpan", 0);
        duk_push_int(ctx, span ? span : 4096);
        duk_put_prop_string(ctx, cfg_idx, "span");
    }

    /* the handle */
    duk_push_object(ctx);
    duk_push_int(ctx, 24000);
    duk_put_prop_string(ctx, -2, "sampleRate");
    duk_dup(ctx, sess_idx);
    duk_put_prop_string(ctx, -2, "session");
    duk_dup(ctx, cfg_idx);
    duk_put_prop_string(ctx, -2, ONNX_H_CFG);
    duk_push_c_function(ctx, onnx_snac_decode, 1);
    duk_put_prop_string(ctx, -2, "decode");
    duk_push_c_function(ctx, onnx_snac_frames_to_codes, 1);
    duk_put_prop_string(ctx, -2, "framesToCodes");
    duk_push_c_function(ctx, onnx_snac_decode_frames, 1);
    duk_put_prop_string(ctx, -2, "decodeFrames");
    duk_push_c_function(ctx, onnx_snac_decode_orpheus, 1);
    duk_put_prop_string(ctx, -2, "decodeOrpheus");
    duk_push_c_function(ctx, onnx_handle_destroy, 0);
    duk_put_prop_string(ctx, -2, "destroy");
    return 1;
}

duk_ret_t duk_open_module(duk_context *ctx) {
    duk_push_object(ctx);

    duk_push_c_function(ctx, onnx_init_session, 2);
    duk_put_prop_string(ctx, -2, "initSession");

    duk_push_c_function(ctx, onnx_init_session_from_buffer, 2);
    duk_put_prop_string(ctx, -2, "initSessionFromBuffer");

    /* initEmbed/initRerank/initSnacDecoder are ALL C: handle state as
     * (hidden) properties, methods as C functions reading `this` — so
     * every handle survives rampart's thread-copy (server preThreadFunc
     * globals -> workers), like rampart-llamacpp's handles. */
    duk_push_c_function(ctx, onnx_init_embed, 2);
    duk_put_prop_string(ctx, -2, "initEmbed");
    duk_push_c_function(ctx, onnx_init_rerank, 2);
    duk_put_prop_string(ctx, -2, "initRerank");
    duk_push_c_function(ctx, onnx_init_snac, 2);
    duk_put_prop_string(ctx, -2, "initSnacDecoder");

    duk_push_c_function(ctx, onnx_model_info_js, 1);
    duk_put_prop_string(ctx, -2, "modelInfo");

    duk_push_c_function(ctx, onnx_version_js, 0);
    duk_put_prop_string(ctx, -2, "onnxVersion");

    duk_push_c_function(ctx, onnx_runtime_info_js, 0);
    duk_put_prop_string(ctx, -2, "runtimeInfo");

    duk_push_c_function(ctx, onnx_get_log_js, 0);
    duk_put_prop_string(ctx, -2, "getLog");
    duk_push_c_function(ctx, onnx_clear_log_js, 0);
    duk_put_prop_string(ctx, -2, "clearLog");
    duk_push_c_function(ctx, onnx_clear_log_js, 0);
    duk_put_prop_string(ctx, -2, "resetLog");   /* alias: rampart-llamacpp naming parity */

    /* native tokenizers */
    duk_push_c_function(ctx, onnx_wordpiece_js, 2);
    duk_put_prop_string(ctx, -2, "wordPieceTokenizer");
    duk_push_c_function(ctx, onnx_sptokenizer_js, 1);
    duk_put_prop_string(ctx, -2, "spTokenizer");

    /* the raw cores behind the init* handles, callable directly:
     * _embedDoc/_embedBatch/_rerank take (sess, tok, cfg, ...) explicitly */
    duk_push_c_function(ctx, onnx_embed_doc_js, 6);
    duk_put_prop_string(ctx, -2, "_embedDoc");
    duk_push_c_function(ctx, onnx_embed_batch_js, 7);
    duk_put_prop_string(ctx, -2, "_embedBatch");
    duk_push_c_function(ctx, onnx_rerank_js, 5);
    duk_put_prop_string(ctx, -2, "_rerank");
    duk_push_c_function(ctx, onnx_snac_weave_js, 4);
    duk_put_prop_string(ctx, -2, "_snacWeave");
    duk_push_c_function(ctx, onnx_discover_js, 1);
    duk_put_prop_string(ctx, -2, "_discover");

    /* Remember the module object (per-ctx, so each rampart thread keeps its own):
     * where a warning goes when there is no `this` -- see onnx_push_errmsg_target(). */
    duk_push_global_stash(ctx);
    duk_dup(ctx, -2);                    /* the module object */
    duk_put_prop_string(ctx, -2, ONNX_MODULE_STASH);
    duk_pop(ctx);                        /* stash */

    return 1;
}
