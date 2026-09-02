/* rampart-ocr.c -- OCR (PP-OCR / RapidOCR models) over rampart-onnx's engine.
 *
 * See claude-work/rapidocr-plan.md.  The defining property of this module is
 * what it does NOT contain: no ONNX Runtime.  PP-OCR is three ONNX graphs
 * (detection, angle classification, recognition) and they run on the ORT that
 * is statically linked inside rampart-onnx.so, reached through the public C ABI
 * in rp-onnx-session.h.  Two ORTs in one process would mean duplicate
 * protobuf/abseil static initializers -- the hazard extern.cmake documents --
 * plus ~36 MB of redundant binary, so the engine stays in one place.
 *
 * That leaves this module holding only OCR: image decode, the DB detection
 * postprocess, crop/warp, CTC decode, and orchestration.  rampart-sql already
 * consumes rampart-onnx the same way (dlsym of rp_onnx_embed_*); this is that
 * arrangement widened from "embed a string" to "run a session".
 *
 * Status: skeleton.  The binding and the module surface are real; the OCR
 * pipeline lands in ocr-det.c / ocr-rec.c / ocr-cls.c.
 */
/* dladdr()/Dl_info are GNU extensions that <dlfcn.h> only exposes under
 * _GNU_SOURCE.  Must precede every system header.  (onnx_shim.cc gets this for
 * free -- g++ defines _GNU_SOURCE itself; a C TU has to ask.) */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <stdint.h>

#include "rampart.h"
#include "rp-onnx-session.h"
#include "ocr-image.h"
#include "ocr-det.h"
#include "ocr-rec.h"

/* get_current_thread() lives in the rampart binary.  WEAK so a non-rampart host
 * that dlopens this module for a future C API still loads (see rampart-onnx.c,
 * which does the same for the same reason). */
#pragma weak get_current_thread

/* ---- this.errMsg: warnings + non-fatal errors ------------------------------
 * The house convention (rampart-sql's errMsg, mirrored in rampart-onnx and
 * rampart-llamacpp): a FAILURE throws a JS Error; a WARNING accumulates on
 * `this.errMsg`, cleared at the top of each call so it always describes the
 * most recent one; NOTHING is written to stdout/stderr.  RAMPART_OCR_DEBUG=1 is
 * the single opt-in hatch. */
static duk_context *ocr_thr_ctx(void)
{
    RPTHR *t = get_current_thread ? get_current_thread() : NULL;
    return t ? t->ctx : NULL;
}

#define OCR_MODULE_STASH DUK_HIDDEN_SYMBOL("ocr_module")

/* Push the object a warning belongs on: `this` when there is one (the module
 * for ocr.init(), a handle for handle methods), else the stashed module object.
 * 0 if neither exists. */
static int ocr_push_errmsg_target(duk_context *ctx)
{
    duk_push_this(ctx);
    if (duk_is_object(ctx, -1)) return 1;
    duk_pop(ctx);
    duk_push_global_stash(ctx);
    if (!duk_get_prop_string(ctx, -1, OCR_MODULE_STASH) || !duk_is_object(ctx, -1)) {
        duk_pop_2(ctx);
        return 0;
    }
    duk_remove(ctx, -2);
    return 1;
}

static void ocr_errmsg_append(duk_context *ctx, const char *msg)
{
    if (!ocr_push_errmsg_target(ctx)) return;
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

static void ocr_errmsg_clear(duk_context *ctx)
{
    if (!ocr_push_errmsg_target(ctx)) return;
    duk_del_prop_string(ctx, -1, "errMsg");
    duk_pop(ctx);
}

void ocr_warn(const char *fmt, ...)
{
    duk_context *ctx = ocr_thr_ctx();
    char line[1024];
    size_t l;
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (getenv("RAMPART_OCR_DEBUG")) fputs(line, stderr);   /* opt-in hatch */
    if (!ctx) return;
    l = strlen(line);
    while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = '\0';
    if (l) ocr_errmsg_append(ctx, line);
}

/* ---- binding to rampart-onnx's session ABI ---------------------------------
 * Resolved ONCE at require() time, not at first use: a missing engine should
 * fail where the cause is obvious.  We look for rampart-onnx.so beside this
 * module (both are installed into modules/), then fall back to whatever the
 * host already has loaded -- which covers a build tree, an unusual layout, or
 * rampart-sql having pulled the module in first. */
static struct {
    void *handle;
    int   bound;
    int         (*abi)(void);
    const char *(*version)(void);
    const char *(*runtime_desc)(void);
    int         (*cuda_available)(void);
    size_t      (*dtype_size)(int);
    const char *(*dtype_name)(int);
    rp_onnx_sess *(*open)(const char *, const rp_onnx_sess_opts *, char *, size_t);
    rp_onnx_sess *(*open_buf)(const void *, size_t, const rp_onnx_sess_opts *, char *, size_t);
    void        (*close)(rp_onnx_sess *);
    int         (*ensure_runnable)(rp_onnx_sess *, char *, size_t);
    size_t      (*n_inputs)(rp_onnx_sess *);
    size_t      (*n_outputs)(rp_onnx_sess *);
    int         (*input)(rp_onnx_sess *, size_t, rp_onnx_iodesc *);
    int         (*output)(rp_onnx_sess *, size_t, rp_onnx_iodesc *);
    int         (*run)(rp_onnx_sess *, const rp_onnx_val_in *, size_t,
                       const char *const *, size_t,
                       rp_onnx_val_out **, size_t *, char *, size_t);
    void        (*run_free)(rp_onnx_val_out *, size_t);
} ONNX;

/* directory containing this module (.so), via dladdr on a local symbol --
 * same approach onnx_shim.cc uses to find its runtime dirs */
static int ocr_module_dir(char *out, size_t n)
{
    Dl_info di;
    size_t l;
    char *slash;

    if (!dladdr((void *)(intptr_t)&ocr_module_dir, &di) || !di.dli_fname) return 0;
    l = strlen(di.dli_fname);
    if (l >= n) return 0;
    memcpy(out, di.dli_fname, l + 1);
    slash = strrchr(out, '/');
    if (!slash) return 0;
    *slash = '\0';
    return 1;
}

#if defined(__APPLE__)
#define OCR_ONNX_SONAME "rampart-onnx.so"
#else
#define OCR_ONNX_SONAME "rampart-onnx.so"
#endif

/* Resolve every entry point; returns 0 and fills err[] if any is missing. */
static int ocr_bind_onnx(char *err, size_t errlen)
{
    char dir[1024], path[1200], dlerr[512];
    void *h = NULL;
    /* NOT `h != NULL`: RTLD_DEFAULT is ((void *)0) on glibc, so a successful
     * bind to an already-loaded engine leaves h NULL and is indistinguishable
     * from failure.  Track success separately. */
    int have = 0;
    int miss = 0;

    if (ONNX.bound) return 1;
    dir[0] = '\0'; dlerr[0] = '\0';

    /* Already loaded?  Check FIRST: rampart caches module handles per path, and
     * rampart-sql may have pulled rampart-onnx in already.  Reusing that mapping
     * is both cheaper and safer than opening a second one -- ORT's statically
     * linked state (the runtime ladder's pthread_once, its threadpools) should
     * exist once per process, not once per opener. */
    if (dlsym(RTLD_DEFAULT, "rp_onnx_sess_abi")) {
        h = RTLD_DEFAULT;
        have = 1;
    } else if (ocr_module_dir(dir, sizeof dir)) {
        snprintf(path, sizeof path, "%s/%s", dir, OCR_ONNX_SONAME);
        /* Exactly the flags rampart's own module loader uses (module.c):
         * RTLD_GLOBAL because rampart-onnx expects to be reachable that way
         * (rampart-sql dlsym's it from RTLD_DEFAULT), and RTLD_LAZY -- NOT
         * _NOW -- because modules here are permitted to carry intentionally
         * unresolved symbols.  rampart-onnx does: it has a dangling re2
         * reference from the localized extensions object that nothing on any
         * live path calls.  RTLD_NOW refuses to load over it; the host itself
         * would not, so demanding more than the host does is simply wrong. */
        h = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        if (h) {
            have = 1;
        } else {
            const char *e = dlerror();
            snprintf(dlerr, sizeof dlerr, "%s", e ? e : "(no detail from dlerror)");
        }
    }
    if (!have) {
        snprintf(err, errlen,
                 "rampart-ocr requires rampart-onnx, which could not be loaded "
                 "(looked for %s/%s)%s%s. Install rampart-onnx beside rampart-ocr.",
                 dir[0] ? dir : "<module dir>", OCR_ONNX_SONAME,
                 dlerr[0] ? ": " : "", dlerr);
        return 0;
    }

#define OCR_BIND(field, sym) do {                                   \
        *(void **)(&ONNX.field) = dlsym(h, sym);                    \
        if (!ONNX.field) miss++;                                    \
    } while (0)

    OCR_BIND(abi,             "rp_onnx_sess_abi");
    OCR_BIND(version,         "rp_onnx_sess_version");
    OCR_BIND(runtime_desc,    "rp_onnx_sess_runtime_desc");
    OCR_BIND(cuda_available,  "rp_onnx_sess_cuda_available");
    OCR_BIND(dtype_size,      "rp_onnx_sess_dtype_size");
    OCR_BIND(dtype_name,      "rp_onnx_sess_dtype_name");
    OCR_BIND(open,            "rp_onnx_sess_open");
    OCR_BIND(open_buf,        "rp_onnx_sess_open_buf");
    OCR_BIND(close,           "rp_onnx_sess_close");
    OCR_BIND(ensure_runnable, "rp_onnx_sess_ensure_runnable");
    OCR_BIND(n_inputs,        "rp_onnx_sess_n_inputs");
    OCR_BIND(n_outputs,       "rp_onnx_sess_n_outputs");
    OCR_BIND(input,           "rp_onnx_sess_input");
    OCR_BIND(output,          "rp_onnx_sess_output");
    OCR_BIND(run,             "rp_onnx_sess_run");
    OCR_BIND(run_free,        "rp_onnx_sess_run_free");
#undef OCR_BIND

    if (miss) {
        snprintf(err, errlen,
                 "rampart-ocr: the rampart-onnx found does not export the session "
                 "C ABI (%d of 16 symbols missing) -- it is older than this module. "
                 "Rebuild both from the same tree.", miss);
        return 0;
    }
    /* Version gate.  Both modules ship from one tree, so a mismatch is a
     * packaging error -- but catch it here rather than let a drifted opts
     * struct be read as stack garbage on the other side. */
    if (ONNX.abi() != RP_ONNX_SESS_ABI) {
        snprintf(err, errlen,
                 "rampart-ocr: session ABI mismatch -- rampart-onnx provides v%d, "
                 "this module was built for v%d. Rebuild both from the same tree.",
                 ONNX.abi(), RP_ONNX_SESS_ABI);
        return 0;
    }

    ONNX.handle = h;
    ONNX.bound  = 1;
    return 1;
}

/* ---- JS surface ---------------------------------------------------------- */

/* ocr.runtimeInfo() -> { ort, runtime, cuda, sessionAbi }
 *
 * What engine did we actually bind to?  Deliberately the first entry point:
 * it exercises the entire architecture -- module found beside us, symbols
 * resolved, ABI agreed, and a real call reaching ORT. */
static duk_ret_t ocr_runtime_info(duk_context *ctx)
{
    ocr_errmsg_clear(ctx);   /* errMsg reflects THIS call */

    duk_push_object(ctx);
    duk_push_string(ctx, ONNX.version());
    duk_put_prop_string(ctx, -2, "ort");
    duk_push_string(ctx, ONNX.runtime_desc());
    duk_put_prop_string(ctx, -2, "runtime");
    duk_push_boolean(ctx, ONNX.cuda_available());
    duk_put_prop_string(ctx, -2, "cuda");
    duk_push_int(ctx, ONNX.abi());
    duk_put_prop_string(ctx, -2, "sessionAbi");
    return 1;
}

/* ocr.modelInfo(path) -> { inputs:[{name,type,shape}], outputs:[...] }
 *
 * Opens a session on any .onnx, reports its shape, closes it.  Useful on its
 * own, and it is how the det/rec/cls loaders will discover tensor names rather
 * than hardcoding them -- PP-OCR conversions disagree about those.
 * A -1 dim is dynamic (det: page size; rec: crop width). */
static duk_ret_t ocr_model_info(duk_context *ctx)
{
    const char *path = REQUIRE_STRING(ctx, 0, "ocr.modelInfo: argument must be a String (path to a .onnx file)");
    char err[512] = {0};
    rp_onnx_sess_opts o;
    rp_onnx_sess *s;
    size_t i, j, n;

    ocr_errmsg_clear(ctx);

    memset(&o, 0, sizeof o);
    o.abi_version   = RP_ONNX_SESS_ABI;
    o.intra_threads = 1;
    o.inter_threads = 1;
    o.graph_opt     = RP_ONNX_SESS_OPT_DEFAULT;

    s = ONNX.open(path, &o, err, sizeof err);
    if (!s) RP_THROW(ctx, "ocr.modelInfo: %s", err[0] ? err : "could not open the model");

    duk_push_object(ctx);
    for (j = 0; j < 2; j++) {
        int is_in = (j == 0);
        n = is_in ? ONNX.n_inputs(s) : ONNX.n_outputs(s);
        duk_push_array(ctx);
        for (i = 0; i < n; i++) {
            rp_onnx_iodesc d;
            size_t k;
            if ((is_in ? ONNX.input(s, i, &d) : ONNX.output(s, i, &d)) != 0) continue;
            duk_push_object(ctx);
            duk_push_string(ctx, d.name ? d.name : "");
            duk_put_prop_string(ctx, -2, "name");
            duk_push_string(ctx, ONNX.dtype_name(d.dtype));
            duk_put_prop_string(ctx, -2, "type");
            duk_push_array(ctx);
            for (k = 0; k < d.n_dims; k++) {
                duk_push_number(ctx, (duk_double_t)d.shape[k]);
                duk_put_prop_index(ctx, -2, (duk_uarridx_t)k);
            }
            duk_put_prop_string(ctx, -2, "shape");
            duk_put_prop_index(ctx, -2, (duk_uarridx_t)i);
        }
        duk_put_prop_string(ctx, -2, is_in ? "inputs" : "outputs");
    }

    ONNX.close(s);
    return 1;
}

/* ---- the OCR handle ------------------------------------------------------
 * State lives ON the handle behind a hidden-symbol pointer with native methods,
 * never in a JS closure: duktape's bytecode copy into a rampart worker thread
 * cannot carry a lexical environment, so a closure-held session would simply
 * vanish there.  The sessions themselves are process-wide (the same arrangement
 * rampart-onnx uses), so a thread copy shares them rather than reloading. */
#define OCR_HANDLE_PTR DUK_HIDDEN_SYMBOL("ocr_handle_ptr")

#define OCR_NAME_MAX 96

typedef struct {
    rp_onnx_sess *det, *rec, *cls;
    ocr_dict      dict;
    ocr_det_opts  det_opts;
    int           limit_side;
    int           use_cls;
    float         cls_thresh;
    int           rec_h, rec_max_w, rec_batch;
    /* input tensor names are READ FROM THE MODELS at init rather than hardcoded:
     * PP-OCR conversions do not agree on them (this one uses "x" throughout, but
     * others use "images" / "input") */
    char          det_in[OCR_NAME_MAX], rec_in[OCR_NAME_MAX], cls_in[OCR_NAME_MAX];
    int           destroyed;
} ocr_handle;

static void ocr_handle_free(ocr_handle *h)
{
    if (!h) return;
    if (h->det) ONNX.close(h->det);
    if (h->rec) ONNX.close(h->rec);
    if (h->cls) ONNX.close(h->cls);
    ocr_dict_free(&h->dict);
    free(h);
}

/* fetch the handle from `this`, throwing if it is gone or destroyed */
static ocr_handle *ocr_this(duk_context *ctx)
{
    ocr_handle *h = NULL;

    duk_push_this(ctx);
    if (duk_get_prop_string(ctx, -1, OCR_HANDLE_PTR))
        h = (ocr_handle *)duk_get_pointer(ctx, -1);
    duk_pop_2(ctx);

    if (!h) RP_THROW(ctx, "ocr: not an ocr handle (or it was destroyed)");
    if (h->destroyed) RP_THROW(ctx, "ocr: handle was destroyed");
    return h;
}

static int ocr_input_name(rp_onnx_sess *s, char *dst, size_t n)
{
    rp_onnx_iodesc d;
    if (!s || ONNX.n_inputs(s) == 0 || ONNX.input(s, 0, &d) != 0 || !d.name) return -1;
    snprintf(dst, n, "%s", d.name);
    return 0;
}

/* run one already-packed NCHW batch; caller frees *outs with ONNX.run_free */
static int ocr_run_batch(rp_onnx_sess *s, const char *in_name,
                         const float *data, int n, int c, int hgt, int wid,
                         rp_onnx_val_out **outs, size_t *n_outs,
                         char *err, size_t errlen)
{
    rp_onnx_val_in in;
    int64_t shape[4];

    shape[0] = n; shape[1] = c; shape[2] = hgt; shape[3] = wid;
    memset(&in, 0, sizeof in);
    in.name    = in_name;
    in.dtype   = RP_ONNX_DT_FLOAT32;
    in.shape   = shape;
    in.n_dims  = 4;
    in.data    = data;
    in.n_bytes = (size_t)n * c * hgt * wid * sizeof(float);

    if (ONNX.ensure_runnable(s, err, errlen) != 0) return -1;
    return ONNX.run(s, &in, 1, NULL, 0, outs, n_outs, err, errlen);
}

/* ocr.detProbe(imagePath, detModelPath [, {limitSideLen, thresh}])
 *   -> { srcW, srcH, netW, netH, sx, sy, mapW, mapH, min, max, mean, fgFrac }
 *
 * Development probe: run ONLY the detection graph and report statistics of the
 * DB probability map.  It exists because the postprocess (binarize -> connected
 * components -> minAreaRect -> unclip) is the largest piece of new code in this
 * module, and debugging it on top of an unverified tensor is guesswork.  With
 * this, the input side is known-good first: `fgFrac` on a text page should be a
 * small but clearly non-zero fraction, and blank margins should be ~0.
 *
 * Superseded by page() once boxes exist; kept as a debugging tool. */
static duk_ret_t ocr_det_probe(duk_context *ctx)
{
    const char *img_path = REQUIRE_STRING(ctx, 0, "ocr.detProbe: arg 1 must be a String (image path)");
    const char *det_path = REQUIRE_STRING(ctx, 1, "ocr.detProbe: arg 2 must be a String (det .onnx path)");
    int limit_side = 960;
    double thresh = 0.3;
    char err[512] = {0};
    ocr_image im;
    ocr_det_scale sc;
    float *tensor = NULL;
    rp_onnx_sess_opts so;
    rp_onnx_sess *s = NULL;
    rp_onnx_val_in in;
    rp_onnx_val_out *outs = NULL;
    size_t n_outs = 0, i, n;
    int64_t shape[4];
    const float *map;
    double mn = 1e30, mx = -1e30, sum = 0.0;
    size_t fg = 0;
    int mapw = 0, maph = 0;

    ocr_errmsg_clear(ctx);

    if (duk_is_object(ctx, 2)) {
        if (duk_get_prop_string(ctx, 2, "limitSideLen"))
            limit_side = REQUIRE_INT(ctx, -1, "ocr.detProbe: limitSideLen must be an integer");
        duk_pop(ctx);
        if (duk_get_prop_string(ctx, 2, "thresh"))
            thresh = REQUIRE_NUMBER(ctx, -1, "ocr.detProbe: thresh must be a number");
        duk_pop(ctx);
    }

    memset(&im, 0, sizeof im);
    if (ocr_image_load_file(img_path, 0, &im, NULL, err, sizeof err) != 0)
        RP_THROW(ctx, "ocr.detProbe: %s", err);

    if (ocr_det_plan(im.w, im.h, limit_side, &sc) != 0) {
        ocr_image_free(&im);
        RP_THROW(ctx, "ocr.detProbe: could not plan a detection size for %dx%d", im.w, im.h);
    }

    tensor = (float *)malloc((size_t)3 * sc.net_w * sc.net_h * sizeof(float));
    if (!tensor) { ocr_image_free(&im); RP_THROW(ctx, "ocr.detProbe: out of memory"); }

    if (ocr_det_preprocess(&im, limit_side, tensor, &sc) != 0) {
        free(tensor); ocr_image_free(&im);
        RP_THROW(ctx, "ocr.detProbe: preprocessing failed");
    }

    memset(&so, 0, sizeof so);
    so.abi_version   = RP_ONNX_SESS_ABI;
    so.intra_threads  = 1;
    so.inter_threads  = 1;
    so.graph_opt      = RP_ONNX_SESS_OPT_DEFAULT;
    s = ONNX.open(det_path, &so, err, sizeof err);
    if (!s) { free(tensor); ocr_image_free(&im); RP_THROW(ctx, "ocr.detProbe: %s", err); }

    shape[0] = 1; shape[1] = 3; shape[2] = sc.net_h; shape[3] = sc.net_w;
    memset(&in, 0, sizeof in);
    in.name    = "x";
    in.dtype   = RP_ONNX_DT_FLOAT32;
    in.shape   = shape;
    in.n_dims  = 4;
    in.data    = tensor;
    in.n_bytes = (size_t)3 * sc.net_w * sc.net_h * sizeof(float);

    if (ONNX.run(s, &in, 1, NULL, 0, &outs, &n_outs, err, sizeof err) != 0) {
        ONNX.close(s); free(tensor); ocr_image_free(&im);
        RP_THROW(ctx, "ocr.detProbe: det run failed: %s", err);
    }
    if (!n_outs || !outs[0].data || outs[0].dtype != RP_ONNX_DT_FLOAT32) {
        ONNX.run_free(outs, n_outs); ONNX.close(s); free(tensor); ocr_image_free(&im);
        RP_THROW(ctx, "ocr.detProbe: det produced no usable output");
    }

    /* DB output is [1,1,H,W]; be tolerant of a squeezed [1,H,W] */
    if (outs[0].n_dims >= 2) {
        maph = (int)outs[0].shape[outs[0].n_dims - 2];
        mapw = (int)outs[0].shape[outs[0].n_dims - 1];
    }
    map = (const float *)outs[0].data;
    n   = outs[0].n_elems;
    for (i = 0; i < n; i++) {
        double v = map[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        if (v >= thresh) fg++;
    }

    /* opts.writePgm: dump the probability map as a P5 PGM so it can be LOOKED at.
     * Statistics can agree with expectation while the map is shifted, mirrored or
     * inverted; an eye on the actual map rules that out in one glance, and the
     * postprocess is far easier to trust when the input to it has been seen. */
    if (duk_is_object(ctx, 2) && duk_get_prop_string(ctx, 2, "writePgm") && mapw > 0 && maph > 0) {
        const char *pgm = duk_get_string(ctx, -1);
        FILE *f = pgm ? fopen(pgm, "wb") : NULL;
        if (f) {
            unsigned char *row = (unsigned char *)malloc((size_t)mapw);
            fprintf(f, "P5\n%d %d\n255\n", mapw, maph);
            if (row) {
                int yy, xx;
                for (yy = 0; yy < maph; yy++) {
                    for (xx = 0; xx < mapw; xx++) {
                        double v = map[(size_t)yy * mapw + xx];
                        row[xx] = (unsigned char)(v <= 0 ? 0 : (v >= 1 ? 255 : v * 255.0 + 0.5));
                    }
                    fwrite(row, 1, (size_t)mapw, f);
                }
                free(row);
            }
            fclose(f);
        } else {
            ocr_warn("ocr.detProbe: could not write PGM to '%s'\n", pgm ? pgm : "(null)");
        }
    }
    duk_pop(ctx);

    duk_push_object(ctx);
#define OCR_PUTN(k, v) do { duk_push_number(ctx, (duk_double_t)(v)); duk_put_prop_string(ctx, -2, k); } while (0)
    OCR_PUTN("srcW", im.w);      OCR_PUTN("srcH", im.h);
    OCR_PUTN("netW", sc.net_w);  OCR_PUTN("netH", sc.net_h);
    OCR_PUTN("sx", sc.sx);       OCR_PUTN("sy", sc.sy);
    OCR_PUTN("mapW", mapw);      OCR_PUTN("mapH", maph);
    OCR_PUTN("min", n ? mn : 0); OCR_PUTN("max", n ? mx : 0);
    OCR_PUTN("mean", n ? sum / (double)n : 0);
    OCR_PUTN("fgFrac", n ? (double)fg / (double)n : 0);
#undef OCR_PUTN

    ONNX.run_free(outs, n_outs);
    ONNX.close(s);
    free(tensor);
    ocr_image_free(&im);
    return 1;
}

/* h.destroy() -- release the sessions and dictionary now rather than at GC. */
static duk_ret_t ocr_destroy(duk_context *ctx)
{
    ocr_handle *h = NULL;

    duk_push_this(ctx);
    if (duk_get_prop_string(ctx, -1, OCR_HANDLE_PTR))
        h = (ocr_handle *)duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (h && !h->destroyed) {
        h->destroyed = 1;
        ocr_handle_free(h);
        duk_push_pointer(ctx, NULL);
        duk_put_prop_string(ctx, -2, OCR_HANDLE_PTR);
    }
    duk_pop(ctx);
    return 0;
}

/* reader.readText(image [, opts]) -> { text, lines:[{text,score,box:[...]}] }
 *
 * `image` is a path or a Buffer.  One call is 3-4 session runs regardless of how
 * many lines the page holds: one detection, one classification batch, and the
 * recognition batches. */
static duk_ret_t ocr_read_text(duk_context *ctx)
{
    ocr_handle *h = ocr_this(ctx);
    char err[512] = {0};
    ocr_image im;
    ocr_det_scale sc;
    float *tensor = NULL, *batch = NULL;
    rp_onnx_val_out *outs = NULL;
    size_t n_outs = 0;
    ocr_box *boxes = NULL;
    size_t nb = 0, i;
    ocr_image *crops = NULL;
    size_t ncrops = 0;
    char **texts = NULL;
    float *scores = NULL;
    size_t *order = NULL;
    int mapw = 0, maph = 0, page = 0, npages = 1;
    duk_idx_t arr;

    ocr_errmsg_clear(ctx);
    memset(&im, 0, sizeof im);

    /* opts.page selects a directory of a multi-page TIFF (0-based); ignored for
     * single-page formats.  The result's `pages` reports the total. */
    if (duk_is_object(ctx, 1)) {
        if (duk_get_prop_string(ctx, 1, "page"))
            page = REQUIRE_INT(ctx, -1, "ocr.readText: page must be an integer");
        duk_pop(ctx);
        if (page < 0) page = 0;
    }

    /* ---- input: path or Buffer --------------------------------------- */
    if (duk_is_string(ctx, 0)) {
        if (ocr_image_load_file(duk_get_string(ctx, 0), page, &im, &npages, err, sizeof err) != 0)
            RP_THROW(ctx, "ocr.readText: %s", err);
    } else if (duk_is_buffer_data(ctx, 0)) {
        duk_size_t bl;
        const void *bd = duk_get_buffer_data(ctx, 0, &bl);
        if (ocr_image_load_mem(bd, (size_t)bl, page, &im, &npages, err, sizeof err) != 0)
            RP_THROW(ctx, "ocr.readText: %s", err);
    } else {
        RP_THROW(ctx, "ocr.readText: first argument must be a String (path) or a Buffer");
    }

#define OCR_READ_BAIL(...) do {                                   \
        free(tensor); free(batch); free(boxes); free(order);       \
        if (outs) ONNX.run_free(outs, n_outs);                     \
        if (crops) { for (i = 0; i < ncrops; i++) ocr_image_free(&crops[i]); free(crops); } \
        if (texts) { for (i = 0; i < ncrops; i++) free(texts[i]); free(texts); } \
        free(scores);                                              \
        ocr_image_free(&im);                                       \
        RP_THROW(ctx, __VA_ARGS__);                                \
    } while (0)

    /* ---- detection, with a quarter-turn retry ---------------------------
     * A page fed in sideways produces boxes that are TALLER than they are wide
     * -- measured, the fraction of tall boxes is 0.01 upright and 1.00 at both
     * 90 and 270 degrees, so it is an unambiguous trigger.  When it fires, the
     * page is stood up and detection re-run: the crops then come out the right
     * way round for the recognizer, and the residual 90-vs-270 ambiguity is
     * exactly 180 degrees, which the angle vote below resolves.
     * One retry only -- a second would loop on a page that is genuinely all
     * vertical text. */
    {
        int attempt;
        for (attempt = 0; attempt < 2; attempt++) {
            if (ocr_det_plan(im.w, im.h, h->limit_side, &sc) != 0)
                OCR_READ_BAIL("ocr.readText: cannot size a detection pass for %dx%d", im.w, im.h);

            tensor = (float *)malloc((size_t)3 * sc.net_w * sc.net_h * sizeof(float));
            if (!tensor) OCR_READ_BAIL("ocr.readText: out of memory");
            if (ocr_det_preprocess(&im, h->limit_side, tensor, &sc) != 0)
                OCR_READ_BAIL("ocr.readText: detection preprocessing failed");

            if (ocr_run_batch(h->det, h->det_in, tensor, 1, 3, sc.net_h, sc.net_w,
                              &outs, &n_outs, err, sizeof err) != 0)
                OCR_READ_BAIL("ocr.readText: detection failed: %s", err);
            if (!n_outs || !outs[0].data || outs[0].dtype != RP_ONNX_DT_FLOAT32)
                OCR_READ_BAIL("ocr.readText: detection produced no usable output");

            if (outs[0].n_dims >= 2) {
                maph = (int)outs[0].shape[outs[0].n_dims - 2];
                mapw = (int)outs[0].shape[outs[0].n_dims - 1];
            }
            if (mapw <= 0 || maph <= 0)
                OCR_READ_BAIL("ocr.readText: detection output has an unusable shape");

            if (ocr_det_boxes((const float *)outs[0].data, mapw, maph, &sc, &h->det_opts,
                              &boxes, &nb) != 0)
                OCR_READ_BAIL("ocr.readText: out of memory extracting boxes");

            ONNX.run_free(outs, n_outs);
            outs = NULL; n_outs = 0;
            free(tensor); tensor = NULL;

            if (attempt == 0 && nb >= 4 &&
                ocr_det_tall_fraction(boxes, nb) > 0.6f) {
                free(boxes); boxes = NULL; nb = 0;
                if (ocr_image_rot90(&im, 1) != 0) break;   /* oom: keep what we had */
                /* Box coordinates now refer to the stood-up page, not the image
                 * the caller passed.  Say so rather than returning geometry that
                 * silently means something else. */
                ocr_warn("ocr.readText: page appears rotated a quarter turn; re-detected "
                         "on a rotated copy -- box coordinates are relative to that "
                         "rotation, not the input image\n");
                continue;
            }
            break;
        }
    }

    ocr_det_sort_boxes(boxes, nb, 0.0f);

    /* ---- crops --------------------------------------------------------- */
    if (nb) {
        crops = (ocr_image *)calloc(nb, sizeof(ocr_image));
        texts = (char **)calloc(nb, sizeof(char *));
        scores = (float *)calloc(nb, sizeof(float));
        if (!crops || !texts || !scores) OCR_READ_BAIL("ocr.readText: out of memory");
        ncrops = nb;
        for (i = 0; i < nb; i++)
            if (ocr_crop_quad(&im, (const float (*)[2])boxes[i].pt, &crops[i]) != 0)
                memset(&crops[i], 0, sizeof(ocr_image));
    }

    /* ---- angle classification: ONE decision for the page ----------------
     * The classifier is run per crop, but the DECISION is made once for the
     * whole page, by majority vote.  A page is upside down as a whole or it is
     * not -- individual lines cannot genuinely disagree -- so this is one
     * decision supported by many noisy pieces of evidence, not many independent
     * decisions.  Treating it as the latter is measurably bad in both
     * directions: on upright pages ~8% of crops vote 180 (simply the model's
     * error rate) and each spurious flip destroys a line, roughly quadrupling
     * CER; on genuinely rotated pages the classifier MISSES 15-45% of crops and
     * each miss leaves a line unreadable.  Aggregating fixes both, because
     * upright pages poll far below half and rotated pages far above it. */
    if (h->use_cls && h->cls && ncrops) {
        const int CW = 192;
        size_t done = 0, votes = 0, counted = 0;

        while (done < ncrops) {
            size_t take = ncrops - done;
            if (take > (size_t)h->rec_batch) take = (size_t)h->rec_batch;
            batch = (float *)malloc(take * 3 * (size_t)h->rec_h * CW * sizeof(float));
            if (!batch) OCR_READ_BAIL("ocr.readText: out of memory (cls batch)");
            if (ocr_pack_crops(crops + done, take, h->rec_h, CW, batch) == 0 &&
                ocr_run_batch(h->cls, h->cls_in, batch, (int)take, 3, h->rec_h, CW,
                              &outs, &n_outs, err, sizeof err) == 0 &&
                n_outs && outs[0].data && outs[0].n_elems >= take * 2) {
                const float *p = (const float *)outs[0].data;
                size_t k;
                for (k = 0; k < take; k++) {
                    counted++;
                    if (p[k * 2 + 1] > h->cls_thresh && p[k * 2 + 1] > p[k * 2])
                        votes++;
                }
            } else if (err[0]) {
                ocr_warn("ocr.readText: angle classification failed (%s); "
                         "continuing without it\n", err);
                err[0] = '\0';
            }
            if (outs) { ONNX.run_free(outs, n_outs); outs = NULL; n_outs = 0; }
            free(batch); batch = NULL;
            done += take;
        }

        /* Flip everything or nothing.  Flipping ALL crops (not just the ones
         * that voted) is the point: the page-level verdict overrides the
         * individual misses that would otherwise be left upside down. */
        if (counted && votes * 2 > counted) {
            for (i = 0; i < ncrops; i++) ocr_image_rot180(&crops[i]);

            /* ...and REVERSE the order.  Rotating a page 180 degrees maps its
             * reading sequence onto the exact reverse, so flipping the crops
             * alone would return every line correctly transcribed but the page
             * back-to-front -- the footer first and the title last.  Reversing
             * the already-sorted arrays is equivalent to re-sorting after the
             * rotation, and cheaper.  The BOX coordinates are deliberately left
             * alone: they correctly locate the text in the image the caller
             * actually passed, which is the frame the caller can use. */
            for (i = 0; i < ncrops / 2; i++) {
                size_t k = ncrops - 1 - i;
                ocr_image ti = crops[i]; crops[i] = crops[k]; crops[k] = ti;
                if (boxes && nb == ncrops) {
                    ocr_box tb = boxes[i]; boxes[i] = boxes[k]; boxes[k] = tb;
                }
            }
        }
    }

    /* ---- recognition ---------------------------------------------------- */
    if (ncrops) {
        size_t done = 0;
        /* Batch crops of SIMILAR width together: every sample in a batch is
         * padded to the batch's widest, so mixing a 40px word with a 900px line
         * would make the short one 95% padding.  Sort by aspect ratio, batch in
         * that order, and write results back through `order`. */
        order = (size_t *)malloc(ncrops * sizeof(size_t));
        if (!order) OCR_READ_BAIL("ocr.readText: out of memory");
        for (i = 0; i < ncrops; i++) order[i] = i;
        for (i = 0; i + 1 < ncrops; i++) {
            size_t j, mn = i;
            for (j = i + 1; j < ncrops; j++) {
                const ocr_image *a = &crops[order[j]], *b = &crops[order[mn]];
                float ra = (a->h > 0) ? (float)a->w / (float)a->h : 0.0f;
                float rb = (b->h > 0) ? (float)b->w / (float)b->h : 0.0f;
                if (ra < rb) mn = j;
            }
            if (mn != i) { size_t t = order[i]; order[i] = order[mn]; order[mn] = t; }
        }

        while (done < ncrops) {
            size_t take = ncrops - done, k;
            ocr_image *tmp;
            int bw;

            if (take > (size_t)h->rec_batch) take = (size_t)h->rec_batch;

            tmp = (ocr_image *)malloc(take * sizeof(ocr_image));
            if (!tmp) OCR_READ_BAIL("ocr.readText: out of memory");
            for (k = 0; k < take; k++) tmp[k] = crops[order[done + k]];

            bw = ocr_batch_width(tmp, take, h->rec_h, h->rec_h, h->rec_max_w);
            batch = (float *)malloc(take * 3 * (size_t)h->rec_h * bw * sizeof(float));
            if (!batch) { free(tmp); OCR_READ_BAIL("ocr.readText: out of memory (rec batch)"); }

            if (ocr_pack_crops(tmp, take, h->rec_h, bw, batch) == 0 &&
                ocr_run_batch(h->rec, h->rec_in, batch, (int)take, 3, h->rec_h, bw,
                              &outs, &n_outs, err, sizeof err) == 0 &&
                n_outs && outs[0].data && outs[0].n_dims == 3) {
                int T = (int)outs[0].shape[1], C = (int)outs[0].shape[2];
                const float *p = (const float *)outs[0].data;
                char line[8192];
                for (k = 0; k < take; k++) {
                    float s = 0.0f;
                    ocr_ctc_decode(p + (size_t)k * T * C, T, C, &h->dict,
                                   line, sizeof line, &s);
                    texts[order[done + k]] = strdup(line);
                    scores[order[done + k]] = s;
                }
            } else if (err[0]) {
                ocr_warn("ocr.readText: recognition batch failed (%s)\n", err);
                err[0] = '\0';
            }
            if (outs) { ONNX.run_free(outs, n_outs); outs = NULL; n_outs = 0; }
            free(batch); batch = NULL;
            free(tmp);
            done += take;
        }
    }

    /* ---- result --------------------------------------------------------- */
    duk_push_object(ctx);
    duk_push_array(ctx);
    arr = duk_get_top_index(ctx);
    {
        size_t nout = 0;
        size_t total = 0;
        char *pagetext = NULL;

        for (i = 0; i < ncrops; i++) if (texts[i] && texts[i][0]) total += strlen(texts[i]) + 1;
        pagetext = (char *)malloc(total + 1);
        if (pagetext) pagetext[0] = '\0';

        for (i = 0; i < ncrops; i++) {
            int k;
            if (!texts[i] || !texts[i][0]) continue;    /* an empty line is noise, not content */
            duk_push_object(ctx);
            duk_push_string(ctx, texts[i]);
            duk_put_prop_string(ctx, -2, "text");
            duk_push_number(ctx, (duk_double_t)scores[i]);
            duk_put_prop_string(ctx, -2, "score");
            duk_push_number(ctx, (duk_double_t)boxes[i].score);
            duk_put_prop_string(ctx, -2, "detScore");
            duk_push_array(ctx);
            for (k = 0; k < 4; k++) {
                duk_push_number(ctx, (duk_double_t)boxes[i].pt[k][0]);
                duk_put_prop_index(ctx, -2, (duk_uarridx_t)(k * 2));
                duk_push_number(ctx, (duk_double_t)boxes[i].pt[k][1]);
                duk_put_prop_index(ctx, -2, (duk_uarridx_t)(k * 2 + 1));
            }
            duk_put_prop_string(ctx, -2, "box");
            duk_put_prop_index(ctx, arr, (duk_uarridx_t)nout++);

            if (pagetext) { strcat(pagetext, texts[i]); strcat(pagetext, "\n"); }
        }
        duk_put_prop_string(ctx, -2, "lines");
        duk_push_string(ctx, pagetext ? pagetext : "");
        duk_put_prop_string(ctx, -2, "text");
        duk_push_int(ctx, page);
        duk_put_prop_string(ctx, -2, "page");
        duk_push_int(ctx, npages);
        duk_put_prop_string(ctx, -2, "pages");
        free(pagetext);
    }

    free(order);
    free(boxes);
    if (crops) { for (i = 0; i < ncrops; i++) ocr_image_free(&crops[i]); free(crops); }
    if (texts) { for (i = 0; i < ncrops; i++) free(texts[i]); free(texts); }
    free(scores);
    ocr_image_free(&im);
#undef OCR_READ_BAIL
    return 1;
}

/* ocr.init({det, rec, cls, dict, ...}) -> handle
 *
 * Paths come from models.ocrGet("ppocr-v5"), which returns exactly this shape,
 * so the common call is  ocr.init(models.ocrGet("ppocr-v5")). */
/* dst = Object.assign(dst, src) over own enumerable properties */
static void ocr_assign(duk_context *ctx, duk_idx_t dst, duk_idx_t src)
{
    if (!duk_is_object(ctx, src)) return;
    dst = duk_normalize_index(ctx, dst);
    duk_enum(ctx, src, DUK_ENUM_OWN_PROPERTIES_ONLY);
    while (duk_next(ctx, -1, 1))
        duk_put_prop(ctx, dst);          /* pops the key/value pair */
    duk_pop(ctx);
}

static duk_ret_t ocr_init(duk_context *ctx)
{
    ocr_handle *h;
    char err[512] = {0};
    const char *det_p, *rec_p, *cls_p, *dict_p;
    rp_onnx_sess_opts so;
    int use_gpu = 1, gpu_explicit = 0, threads = 1;
    duk_idx_t o = 0;                     /* where options are read from */

    ocr_errmsg_clear(ctx);
    if (!duk_is_object(ctx, 0))
        RP_THROW(ctx, "ocr.init: argument must be an Object ({det, rec, dict[, cls]})");

    /* Two-argument form is Object.assign({}, first, second): the model paths
     * come straight from models.ocrGet() and the settings you actually want to
     * vary go in the second object, second winning.  Without it, turning on the
     * GPU meant mutating the models object in a separate statement. */
    if (duk_is_object(ctx, 1)) {
        duk_push_object(ctx);
        ocr_assign(ctx, -1, 0);
        ocr_assign(ctx, -1, 1);
        o = duk_normalize_index(ctx, -1);
    }

#define OCR_OPT_STR(key) (duk_get_prop_string(ctx, o, key) ? duk_get_string(ctx, -1) : NULL)
    det_p  = OCR_OPT_STR("det");
    rec_p  = OCR_OPT_STR("rec");
    cls_p  = OCR_OPT_STR("cls");
    dict_p = OCR_OPT_STR("dict");
#undef OCR_OPT_STR
    /* the four duk_get_prop_string results stay on the stack until we return;
     * that keeps the strings alive while we use them */

    if (!det_p || !rec_p || !dict_p)
        RP_THROW(ctx, "ocr.init: det, rec and dict paths are required "
                      "(pass the result of models.ocrGet('ppocr-v5'))");

    h = (ocr_handle *)calloc(1, sizeof *h);
    if (!h) RP_THROW(ctx, "ocr.init: out of memory");

    ocr_det_opts_default(&h->det_opts);
    h->limit_side = 960;
    h->use_cls    = 1;          /* on by default: a rotated page decodes to
                                 * garbage that looks like a bad scan, and the
                                 * classifier is tiny */
    /* Confidence a single crop needs before it may CAST a vote.  PP-OCR uses
     * 0.9 because there the per-crop decision acts directly and a wrong flip
     * destroys a line.  Here the decision is made by page-level majority, which
     * inverts the reasoning: individual precision no longer matters much, while
     * suppressing votes does -- a genuinely rotated page whose crops poll only
     * 55% can fall under the majority line and be left upside down.  0.5 on a
     * two-class softmax is effectively "whichever class won", which maximizes
     * the evidence reaching the vote; the aggregate stays cleanly separated
     * (upright pages poll far below half either way). */
    h->cls_thresh = 0.5f;
    h->rec_h      = 48;
    h->rec_max_w  = 1600;
    h->rec_batch  = 6;

#define OCR_NUM(key, dst, cast) do {                                            \
        if (duk_get_prop_string(ctx, o, key)) dst = (cast)duk_get_number(ctx, -1); \
        duk_pop(ctx);                                                           \
    } while (0)
    OCR_NUM("limitSideLen", h->limit_side,          int);
    OCR_NUM("thresh",       h->det_opts.thresh,     float);
    OCR_NUM("boxThresh",    h->det_opts.box_thresh, float);
    OCR_NUM("unclipRatio",  h->det_opts.unclip_ratio, float);
    OCR_NUM("minSize",      h->det_opts.min_size,   int);
    OCR_NUM("maxBoxes",     h->det_opts.max_boxes,  int);
    OCR_NUM("clsThresh",    h->cls_thresh,          float);
    OCR_NUM("recHeight",    h->rec_h,               int);
    OCR_NUM("recMaxWidth",  h->rec_max_w,           int);
    OCR_NUM("recBatch",     h->rec_batch,           int);
    OCR_NUM("threads",      threads,                int);
#undef OCR_NUM
    if (duk_get_prop_string(ctx, o, "cls") && duk_is_boolean(ctx, -1))
        h->use_cls = duk_get_boolean(ctx, -1);
    duk_pop(ctx);
    /* gpu defaults to ON, as rampart-onnx's own handles do: a GPU build with a
     * usable device uses it, anything else lands on the CPU.  Only an EXPLICIT
     * gpu:true that cannot be honored is worth a warning. */
    if (duk_get_prop_string(ctx, o, "gpu") && !duk_is_undefined(ctx, -1)) {
        use_gpu = duk_to_boolean(ctx, -1);
        gpu_explicit = 1;
    }
    duk_pop(ctx);
    if (h->rec_batch < 1) h->rec_batch = 1;

    if (use_gpu && !ONNX.cuda_available()) {
        if (gpu_explicit)
            ocr_warn("ocr.init: gpu requested but this rampart-onnx has no CUDA "
                     "execution provider; using the CPU\n");
        use_gpu = 0;
    }

    memset(&so, 0, sizeof so);
    so.abi_version   = RP_ONNX_SESS_ABI;
    /* Intra-op threads default to 1 because the intended workload is a document
     * pipeline: many pages in flight across many rampart threads, where
     * per-session core pools would oversubscribe (the reasoning rp_onnx_embed_*
     * uses).  That is the wrong default for ONE interactive page, which is
     * latency-bound on a single core -- so it is a knob.  `threads: 0` asks ORT
     * for its own default (all cores). */
    so.intra_threads = threads;
    so.inter_threads = 1;
    so.graph_opt     = RP_ONNX_SESS_OPT_DEFAULT;
    for (;;) {
        so.use_cuda = use_gpu;
        h->det = ONNX.open(det_p, &so, err, sizeof err);
        if (h->det) h->rec = ONNX.open(rec_p, &so, err, sizeof err);
        if (h->det && h->rec) break;
        if (use_gpu) {
            /* the provider exists but a session could not be created on the
             * device (driver, memory, another process holding it): fall back
             * rather than fail, as rampart-onnx's handles do */
            ocr_warn("ocr.init: could not create a GPU session (%s); using the CPU\n", err);
            if (h->det) { ONNX.close(h->det); h->det = NULL; }
            use_gpu = 0;
            continue;
        }
        {
            const char *which = h->det ? "recognition" : "detection";
            ocr_handle_free(h);
            RP_THROW(ctx, "ocr.init: %s model: %s", which, err);
        }
    }
    if (h->use_cls && cls_p) {
        h->cls = ONNX.open(cls_p, &so, err, sizeof err);
        if (!h->cls) {
            /* not fatal: without it, upside-down lines decode badly, but every
             * upright line still works -- a warning, not a failure */
            ocr_warn("ocr.init: angle classifier could not be loaded (%s); "
                     "continuing without angle classification\n", err);
            h->use_cls = 0;
            err[0] = '\0';
        }
    } else {
        h->use_cls = 0;
    }

    if (ocr_input_name(h->det, h->det_in, sizeof h->det_in) != 0 ||
        ocr_input_name(h->rec, h->rec_in, sizeof h->rec_in) != 0) {
        ocr_handle_free(h);
        RP_THROW(ctx, "ocr.init: could not read model input names");
    }
    if (h->cls) ocr_input_name(h->cls, h->cls_in, sizeof h->cls_in);

    if (ocr_dict_load(dict_p, &h->dict, err, sizeof err) != 0) {
        ocr_handle_free(h);
        RP_THROW(ctx, "ocr.init: %s", err);
    }

    /* Cross-check the dictionary against the recognizer: PP-OCR's class list is
     * blank + dict + space, so these must agree exactly.  A mismatch means the
     * wrong dictionary for the model, and every character would decode shifted --
     * a failure worth catching here rather than puzzling over in the output. */
    {
        rp_onnx_iodesc od;
        if (ONNX.n_outputs(h->rec) && ONNX.output(h->rec, 0, &od) == 0 &&
            od.n_dims >= 1) {
            long want = (long)od.shape[od.n_dims - 1];
            if (want > 0 && (size_t)want != h->dict.n_items) {
                size_t got = h->dict.n_items;
                ocr_handle_free(h);
                RP_THROW(ctx, "ocr.init: dictionary has %zu classes (blank + %zu lines "
                              "+ space) but the recognition model emits %ld -- "
                              "wrong dictionary for this model",
                         got, got >= 2 ? got - 2 : 0, want);
            }
        }
    }

    /* build the handle object */
    duk_push_object(ctx);
    duk_push_pointer(ctx, h);
    duk_put_prop_string(ctx, -2, OCR_HANDLE_PTR);
    duk_push_c_function(ctx, ocr_read_text, 2);
    duk_put_prop_string(ctx, -2, "readText");
    duk_push_c_function(ctx, ocr_destroy, 0);
    duk_put_prop_string(ctx, -2, "destroy");

    /* `settings`: what this handle ACTUALLY ended up using, which is not
     * always what was asked for -- `gpu` here is 0 if the CUDA provider was
     * requested but unavailable.  Without it there is no way to answer "did the
     * GPU turn on?": errMsg is empty either way and runtimeInfo() reports only
     * that the engine HAS a provider, not that this handle uses it.
     * (Mirrors idx.settings in rampart-faiss.) */
    duk_push_object(ctx);
#define OCR_SET_N(k, v) do { duk_push_number(ctx, (duk_double_t)(v)); duk_put_prop_string(ctx, -2, k); } while (0)
#define OCR_SET_B(k, v) do { duk_push_boolean(ctx, (v)); duk_put_prop_string(ctx, -2, k); } while (0)
    OCR_SET_B("gpu",          use_gpu);
    OCR_SET_N("threads",      threads);
    OCR_SET_B("cls",          h->use_cls);
    OCR_SET_N("limitSideLen", h->limit_side);
    OCR_SET_N("thresh",       h->det_opts.thresh);
    OCR_SET_N("boxThresh",    h->det_opts.box_thresh);
    OCR_SET_N("unclipRatio",  h->det_opts.unclip_ratio);
    OCR_SET_N("minSize",      h->det_opts.min_size);
    OCR_SET_N("maxBoxes",     h->det_opts.max_boxes);
    OCR_SET_N("clsThresh",    h->cls_thresh);
    OCR_SET_N("recHeight",    h->rec_h);
    OCR_SET_N("recMaxWidth",  h->rec_max_w);
    OCR_SET_N("recBatch",     h->rec_batch);
    OCR_SET_N("dictSize",     (double)h->dict.n_items);
#undef OCR_SET_N
#undef OCR_SET_B
    duk_put_prop_string(ctx, -2, "settings");

    return 1;
}

/* ocr.pageCount(pathOrBuffer) -> Number
 *
 * How many pages the input holds: the directory count of a multi-page TIFF, or 1
 * for every single-page format.  Does not decode pixels. */
static duk_ret_t ocr_page_count(duk_context *ctx)
{
    int n = 0;

    ocr_errmsg_clear(ctx);
    if (duk_is_string(ctx, 0)) {
        const char *path = duk_get_string(ctx, 0);
        n = ocr_image_page_count(path);
        if (n <= 0) RP_THROW(ctx, "ocr.pageCount: cannot open '%s'", path);
    } else if (duk_is_buffer_data(ctx, 0)) {
        duk_size_t bl;
        const void *bd = duk_get_buffer_data(ctx, 0, &bl);
        n = ocr_image_page_count_mem(bd, (size_t)bl);
        if (n <= 0) RP_THROW(ctx, "ocr.pageCount: cannot read image from buffer");
    } else {
        RP_THROW(ctx, "ocr.pageCount: argument must be a String (path) or a Buffer");
    }
    duk_push_int(ctx, n);
    return 1;
}

duk_ret_t duk_open_module(duk_context *ctx)
{
    char err[512] = {0};

    /* Bind before exposing anything: if the engine is missing, require() itself
     * fails with one clear message instead of every later call failing obscurely. */
    if (!ocr_bind_onnx(err, sizeof err))
        RP_THROW(ctx, "%s", err);

    duk_push_object(ctx);
    duk_push_c_function(ctx, ocr_runtime_info, 0);
    duk_put_prop_string(ctx, -2, "runtimeInfo");
    duk_push_c_function(ctx, ocr_model_info, 1);
    duk_put_prop_string(ctx, -2, "modelInfo");
    duk_push_c_function(ctx, ocr_det_probe, 3);
    duk_put_prop_string(ctx, -2, "detProbe");
    duk_push_c_function(ctx, ocr_init, 2);
    duk_put_prop_string(ctx, -2, "init");
    duk_push_c_function(ctx, ocr_page_count, 1);
    duk_put_prop_string(ctx, -2, "pageCount");

    /* Stash the module object: it is where a warning goes when there is no
     * `this` (see ocr_push_errmsg_target). */
    duk_push_global_stash(ctx);
    duk_dup(ctx, -2);
    duk_put_prop_string(ctx, -2, OCR_MODULE_STASH);
    duk_pop(ctx);

    return 1;
}
