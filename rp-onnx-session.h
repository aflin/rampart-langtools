/* rp-onnx-session.h -- the PUBLIC session C ABI exported by rampart-onnx.so.
 *
 * Purpose: let a sibling module run arbitrary ONNX graphs on the ONNX Runtime
 * that is statically linked inside rampart-onnx.so, without linking a second
 * copy of its own.  rampart-ocr is the first consumer (PP-OCR is three graphs:
 * detection, angle classification, recognition).
 *
 * Two ORTs in one process is not merely wasteful, it is a hazard: each bundles
 * protobuf/abseil/onnx, and rampart dlopen's modules RTLD_GLOBAL, so duplicate
 * static initializers collide (the same reason rampart-onnx.map localizes
 * everything and the extensions build sets ONNXEXT_BUNDLE_PROTOBUF=0 on CPU).
 * So the engine stays in one place and this header is how others reach it --
 * the same arrangement rampart-sql already uses for rp_onnx_embed_*, widened
 * from "embed a string" to "run a session".
 *
 * These declarations are a thin, deliberately-stable veneer over the internal
 * onnx_shim.h.  They are NOT that header: the internal one may change with ORT,
 * this one is a contract.  The `rp_` prefix marks it public and keeps generic
 * names (onnx_session_create...) out of the global dynamic namespace, where a
 * host linking its own ORT could collide with them.
 *
 * Binding: consumers dlopen rampart-onnx.so (found beside themselves in
 * modules/) and dlsym these names -- see the "Binding" note at the bottom.
 *
 * Threading / fork: a session pointer is process-wide and may be used from any
 * rampart thread; ORT serializes internally.  Call rp_onnx_sess_ensure_runnable()
 * before each run -- it is the fork gate (a CPU session is transparently rebuilt
 * in the child; a GPU session errors rather than producing garbage).
 */
#ifndef RP_ONNX_SESSION_H
#define RP_ONNX_SESSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version.  Callers set rp_onnx_sess_opts.abi_version to this; open()
 * rejects anything else rather than reading a drifted struct as stack garbage.
 * Both this module and its consumers ship from one tree, so a mismatch is a
 * packaging bug -- but it should be a DETECTED one.  Bump on ANY change to the
 * structs below.  (Mirrors RP_ONNX_EMBED_ABI in rampart-onnx.c.) */
#define RP_ONNX_SESS_ABI 1

/* Element dtypes.  Values match onnx_shim.h's onnx_dtype one-for-one. */
typedef enum {
    RP_ONNX_DT_UNKNOWN = 0,
    RP_ONNX_DT_FLOAT32,
    RP_ONNX_DT_FLOAT16,
    RP_ONNX_DT_DOUBLE,
    RP_ONNX_DT_INT64,
    RP_ONNX_DT_INT32,
    RP_ONNX_DT_INT16,
    RP_ONNX_DT_INT8,
    RP_ONNX_DT_UINT8,
    RP_ONNX_DT_BOOL
} rp_onnx_dtype;

/* Take ORT's own default for a session option. */
#define RP_ONNX_SESS_OPT_DEFAULT (-1)

typedef struct {
    int abi_version;     /* must be RP_ONNX_SESS_ABI */
    int intra_threads;   /* <=0 => ORT default.  1 is right for a module called
                          * from many rampart threads at once: outer concurrency
                          * already fills the cores, and per-session pools would
                          * oversubscribe (rp_onnx_embed_* pins 1 for exactly
                          * this reason). */
    int inter_threads;   /* <=0 => ORT default */
    int graph_opt;       /* 0=disable 1=basic 2=extended 3=all; <0 => all */
    int execution_mode;  /* 0=sequential, 1=parallel; <0 => sequential */
    int use_cuda;        /* !=0 => try the CUDA EP.  Only meaningful when
                          * rp_onnx_sess_cuda_available() is true; on a CPU-only
                          * build open() fails with a clear message, so a caller
                          * that wants graceful degradation should check first
                          * (or retry with use_cuda=0). */
    int cuda_device_id;  /* device ordinal when use_cuda; <0 => 0 */
} rp_onnx_sess_opts;

/* Opaque session.  Lifetime is the caller's: one open(), one close(). */
typedef struct rp_onnx_sess rp_onnx_sess;

/* One input value for a run.  `data` is BORROWED for the duration of the run
 * call only -- ORT wraps it in place and results are copied out, so the caller
 * may reuse or free the buffer as soon as run() returns. */
typedef struct {
    const char    *name;      /* must match a model input name */
    int            dtype;     /* rp_onnx_dtype */
    const int64_t *shape;
    size_t         n_dims;
    const void    *data;
    size_t         n_bytes;   /* must equal prod(shape) * sizeof(dtype) */
} rp_onnx_val_in;

/* One produced output.  name/shape/data are malloc'd; free the whole array
 * with rp_onnx_sess_run_free(). */
typedef struct {
    char    *name;
    int      dtype;
    int64_t *shape;
    size_t   n_dims;
    void    *data;
    size_t   n_bytes;
    size_t   n_elems;
} rp_onnx_val_out;

/* One input/output description from the model.  Strings and shape are BORROWED
 * from the session and valid until rp_onnx_sess_close().  A dim of -1 is
 * dynamic/symbolic -- which is how the PP-OCR det (variable page size) and rec
 * (variable crop width) graphs report themselves; feed whatever concrete shape
 * you want per run. */
typedef struct {
    const char    *name;
    int            dtype;
    const int64_t *shape;
    size_t         n_dims;
} rp_onnx_iodesc;

/* ---- version / capability ------------------------------------------------ */

/* RP_ONNX_SESS_ABI as compiled into rampart-onnx.so.  Call this BEFORE filling
 * in an opts struct if you want to fail cleanly on a mismatched pair rather
 * than have open() reject you. */
int         rp_onnx_sess_abi(void);
/* ORT version string, e.g. "1.27.0". */
const char *rp_onnx_sess_version(void);
/* Which runtime the selection ladder picked ("built-in CPU", or an external
 * onnx-cuNN dir plus why).  Resolved at first use. */
const char *rp_onnx_sess_runtime_desc(void);
/* 1 if the selected runtime carries the CUDA execution provider.  Says nothing
 * about whether a usable GPU is actually present -- that is only known when a
 * session is opened with use_cuda. */
int         rp_onnx_sess_cuda_available(void);

/* ---- lifecycle ----------------------------------------------------------- */

/* Open a session on a .onnx file (open_buf: on an in-memory model).  Returns
 * NULL and fills err[] on failure.  err may be NULL. */
rp_onnx_sess *rp_onnx_sess_open(const char *path, const rp_onnx_sess_opts *opts,
                                char *err, size_t errlen);
rp_onnx_sess *rp_onnx_sess_open_buf(const void *data, size_t len,
                                    const rp_onnx_sess_opts *opts,
                                    char *err, size_t errlen);
void          rp_onnx_sess_close(rp_onnx_sess *s);

/* Pre-run gate; call before each run.  Errors if the session was closed, and
 * handles a fork: an inherited multi-threaded CPU session is rebuilt (ORT's
 * threadpool does not survive fork), a GPU session errors.  0 = ok, -1 = err. */
int rp_onnx_sess_ensure_runnable(rp_onnx_sess *s, char *err, size_t errlen);

/* ---- introspection ------------------------------------------------------- */

size_t rp_onnx_sess_n_inputs(rp_onnx_sess *s);
size_t rp_onnx_sess_n_outputs(rp_onnx_sess *s);
/* Fill *desc for input/output i.  0 on success, -1 if i is out of range. */
int    rp_onnx_sess_input(rp_onnx_sess *s, size_t i, rp_onnx_iodesc *desc);
int    rp_onnx_sess_output(rp_onnx_sess *s, size_t i, rp_onnx_iodesc *desc);

/* ---- run ----------------------------------------------------------------- */

/* Run the graph.  out_names NULL (n_out_names 0) produces every model output.
 * On success returns 0 and sets *outs and *n_outs (free with run_free); on failure
 * returns -1 and fills err[].  Shapes are per-run, so a dynamic-axis model
 * needs no special handling. */
int  rp_onnx_sess_run(rp_onnx_sess *s,
                      const rp_onnx_val_in *ins, size_t n_ins,
                      const char *const *out_names, size_t n_out_names,
                      rp_onnx_val_out **outs, size_t *n_outs,
                      char *err, size_t errlen);
void rp_onnx_sess_run_free(rp_onnx_val_out *outs, size_t n_outs);

/* ---- helpers ------------------------------------------------------------- */

/* Bytes per element of a dtype (0 if unknown), and its canonical lowercase
 * name ("float32", ...).  Handy for sizing/validating a val_in. */
size_t      rp_onnx_sess_dtype_size(int dtype);
const char *rp_onnx_sess_dtype_name(int dtype);

/* ---- Binding ------------------------------------------------------------
 * A consumer module does NOT link against rampart-onnx.so.  It locates it
 * beside itself at load time and resolves these symbols:
 *
 *     Dl_info di;
 *     dladdr((void *)duk_open_module, &di);     // our own .so path
 *     ... replace the basename with "rampart-onnx.so" ...
 *     void *h = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);   // LAZY, as rampart does
 *     fn = dlsym(h, "rp_onnx_sess_open");       // etc.
 *
 * Check dlsym(RTLD_DEFAULT, ...) first -- the module may already be loaded, and
 * ORT's statically linked state should exist once per process.  Use RTLD_LAZY,
 * matching rampart's own loader: modules here may carry intentionally
 * unresolved symbols, and rampart-onnx does, so RTLD_NOW refuses to load it.
 *
 * Resolve every symbol once at require() time and throw a single clear error
 * ("rampart-ocr requires rampart-onnx") if any is missing -- not at first use,
 * where the failure would surface far from its cause.
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* RP_ONNX_SESSION_H */
