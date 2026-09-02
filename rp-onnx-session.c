/* rp-onnx-session.c -- implementation of the public session C ABI.
 *
 * A thin veneer over onnx_shim.h (see rp-onnx-session.h for why it exists).
 * Everything here is a translation layer, deliberately dumb: no policy, no
 * caching, no state of its own.  The one job beyond forwarding is to keep the
 * public structs decoupled from the internal ones, so ORT churn in onnx_shim.h
 * cannot silently change the contract a sibling module compiled against.
 *
 * Layout note: rp_onnx_val_in/out are currently field-identical to their
 * onnx_shim.h counterparts, so the copies below look pointless -- and casting
 * would work today.  It is done explicitly on purpose: the moment the internal
 * struct gains or reorders a field, the cast becomes a silent memory bug while
 * the copy becomes a compile error.  The cost is a handful of word moves per
 * run, against inference work measured in milliseconds.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "onnx_shim.h"
#include "rp-onnx-session.h"

/* Inputs are translated into a stack array; real models use very few (each of
 * PP-OCR's three graphs takes exactly one).  Anything larger falls back to the
 * heap rather than imposing a limit. */
#define RP_SESS_STACK_INS 8

static int rp_fail(char *err, size_t errlen, const char *msg)
{
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return -1;
}

/* Translate the public opts to the internal ones.  NULL means "ORT defaults",
 * which is also the only way to call without an abi_version -- safe, because a
 * caller passing NULL is not handing us a struct we could misread.  Returns 0
 * on success, -1 (with err filled) on an ABI mismatch. */
static int rp_opts_in(const rp_onnx_sess_opts *o, onnx_session_opts *so,
                      int *have, char *err, size_t errlen)
{
    if (!o) { *have = 0; return 0; }

    if (o->abi_version != RP_ONNX_SESS_ABI) {
        if (err && errlen)
            snprintf(err, errlen,
                     "rp_onnx_sess: opts.abi_version=%d, expected %d -- "
                     "rampart-onnx and the calling module are from different builds",
                     o->abi_version, RP_ONNX_SESS_ABI);
        return -1;
    }

    memset(so, 0, sizeof *so);
    so->intra_threads  = o->intra_threads;
    so->inter_threads  = o->inter_threads;
    so->graph_opt      = o->graph_opt;
    so->execution_mode = o->execution_mode;
    so->use_cuda       = o->use_cuda;
    so->cuda_device_id = o->cuda_device_id;
    /* CoreML was removed from the build (it bought nothing for our model classes
     * and blocked the older-macOS SDKs), so there is nothing to plumb through. */
    so->use_coreml     = 0;
    so->coreml_units   = 0;
    *have = 1;
    return 0;
}

int         rp_onnx_sess_abi(void)           { return RP_ONNX_SESS_ABI; }
const char *rp_onnx_sess_version(void)       { return onnx_version(); }
const char *rp_onnx_sess_runtime_desc(void)  { return onnx_runtime_desc(); }
int         rp_onnx_sess_cuda_available(void){ return onnx_cuda_ep_available(); }

size_t      rp_onnx_sess_dtype_size(int dt)  { return onnx_dtype_size(dt); }
const char *rp_onnx_sess_dtype_name(int dt)  { return onnx_dtype_name(dt); }

rp_onnx_sess *rp_onnx_sess_open(const char *path, const rp_onnx_sess_opts *opts,
                                char *err, size_t errlen)
{
    onnx_session_opts so;
    int have = 0;

    if (!path || !*path) { rp_fail(err, errlen, "rp_onnx_sess_open: null path"); return NULL; }
    if (rp_opts_in(opts, &so, &have, err, errlen) != 0) return NULL;

    return (rp_onnx_sess *)onnx_session_create(path, have ? &so : NULL, err, errlen);
}

rp_onnx_sess *rp_onnx_sess_open_buf(const void *data, size_t len,
                                    const rp_onnx_sess_opts *opts,
                                    char *err, size_t errlen)
{
    onnx_session_opts so;
    int have = 0;

    if (!data || !len) { rp_fail(err, errlen, "rp_onnx_sess_open_buf: empty model buffer"); return NULL; }
    if (rp_opts_in(opts, &so, &have, err, errlen) != 0) return NULL;

    return (rp_onnx_sess *)onnx_session_create_from_buffer(data, len, have ? &so : NULL, err, errlen);
}

void rp_onnx_sess_close(rp_onnx_sess *s)
{
    onnx_session_destroy((onnx_session *)s);
}

int rp_onnx_sess_ensure_runnable(rp_onnx_sess *s, char *err, size_t errlen)
{
    if (!s) return rp_fail(err, errlen, "rp_onnx_sess: null session");
    return onnx_session_ensure_runnable((onnx_session *)s, err, errlen);
}

size_t rp_onnx_sess_n_inputs(rp_onnx_sess *s)
{
    const onnx_modelinfo *mi = s ? onnx_session_info((onnx_session *)s) : NULL;
    return mi ? mi->n_inputs : 0;
}

size_t rp_onnx_sess_n_outputs(rp_onnx_sess *s)
{
    const onnx_modelinfo *mi = s ? onnx_session_info((onnx_session *)s) : NULL;
    return mi ? mi->n_outputs : 0;
}

/* shared by input()/output(): copy one borrowed description out */
static int rp_desc(const onnx_iodesc *src, rp_onnx_iodesc *dst)
{
    if (!src || !dst) return -1;
    dst->name   = src->name;
    dst->dtype  = src->dtype;
    dst->shape  = src->shape;
    dst->n_dims = src->n_dims;
    return 0;
}

int rp_onnx_sess_input(rp_onnx_sess *s, size_t i, rp_onnx_iodesc *desc)
{
    const onnx_modelinfo *mi = s ? onnx_session_info((onnx_session *)s) : NULL;
    if (!mi || i >= mi->n_inputs) return -1;
    return rp_desc(&mi->inputs[i], desc);
}

int rp_onnx_sess_output(rp_onnx_sess *s, size_t i, rp_onnx_iodesc *desc)
{
    const onnx_modelinfo *mi = s ? onnx_session_info((onnx_session *)s) : NULL;
    if (!mi || i >= mi->n_outputs) return -1;
    return rp_desc(&mi->outputs[i], desc);
}

int rp_onnx_sess_run(rp_onnx_sess *s,
                     const rp_onnx_val_in *ins, size_t n_ins,
                     const char *const *out_names, size_t n_out_names,
                     rp_onnx_val_out **outs, size_t *n_outs,
                     char *err, size_t errlen)
{
    /* zero-initialized so the n_ins==0 case doesn't hand the shim a pointer
     * into unwritten stack (it reads nothing at count 0, but the compiler
     * cannot know that) */
    onnx_value_in  stackv[RP_SESS_STACK_INS] = {{0}};
    onnx_value_in *iv = stackv;
    onnx_value_out *raw = NULL;
    rp_onnx_val_out *pub = NULL;
    size_t n_raw = 0, i;
    int rc;

    if (!s)              return rp_fail(err, errlen, "rp_onnx_sess_run: null session");
    if (!outs || !n_outs)return rp_fail(err, errlen, "rp_onnx_sess_run: null out params");
    if (n_ins && !ins)   return rp_fail(err, errlen, "rp_onnx_sess_run: null inputs");
    *outs = NULL; *n_outs = 0;

    if (n_ins > RP_SESS_STACK_INS) {
        iv = (onnx_value_in *)malloc(n_ins * sizeof *iv);
        if (!iv) return rp_fail(err, errlen, "rp_onnx_sess_run: oom");
    }
    for (i = 0; i < n_ins; i++) {
        iv[i].name    = ins[i].name;
        iv[i].dtype   = ins[i].dtype;
        iv[i].shape   = ins[i].shape;
        iv[i].n_dims  = ins[i].n_dims;
        iv[i].data    = ins[i].data;
        iv[i].n_bytes = ins[i].n_bytes;
    }

    rc = onnx_session_run((onnx_session *)s, iv, n_ins,
                          out_names, n_out_names, &raw, &n_raw, err, errlen);
    if (iv != stackv) free(iv);
    if (rc != 0) return -1;

    /* Take ownership of each output's malloc'd name/shape/data, then free only
     * the shim's container array -- onnx_run_free() would free the parts we are
     * adopting.  rp_onnx_sess_run_free() releases them the same way it does. */
    if (n_raw) {
        pub = (rp_onnx_val_out *)calloc(n_raw, sizeof *pub);
        if (!pub) { onnx_run_free(raw, n_raw); return rp_fail(err, errlen, "rp_onnx_sess_run: oom"); }
        for (i = 0; i < n_raw; i++) {
            pub[i].name    = raw[i].name;
            pub[i].dtype   = raw[i].dtype;
            pub[i].shape   = raw[i].shape;
            pub[i].n_dims  = raw[i].n_dims;
            pub[i].data    = raw[i].data;
            pub[i].n_bytes = raw[i].n_bytes;
            pub[i].n_elems = raw[i].n_elems;
        }
    }
    free(raw);

    *outs = pub;
    *n_outs = n_raw;
    return 0;
}

void rp_onnx_sess_run_free(rp_onnx_val_out *outs, size_t n_outs)
{
    size_t i;
    if (!outs) return;
    for (i = 0; i < n_outs; i++) {
        free(outs[i].name);
        free(outs[i].shape);
        free(outs[i].data);
    }
    free(outs);
}
