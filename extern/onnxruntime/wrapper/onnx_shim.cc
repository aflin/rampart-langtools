/* onnx_shim.cc -- C ABI implementation over the ONNX Runtime C API.
 * See onnx_shim.h for the contract. POSIX only (char model paths); Windows
 * would need the ORTCHAR_T/wchar path, which rampart doesn't target.
 */
#include "onnx_shim.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>      /* runtime selection: dlopen'd ORT core + CUDA driver probe */
#include <limits.h>
#include <sys/stat.h>
#include <atomic>       /* arena-shrinkage run counter */

#include "onnxruntime_c_api.h"

/* ---- ORT api handle + process-global environment ------------------------- */

static const OrtApi     *g_ort  = nullptr;
static const OrtApiBase *g_base = nullptr;   /* internal OR dlopen'd runtime */
static OrtEnv           *g_env  = nullptr;
static char              g_runtime_desc[512] = "uninitialized";
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

/* ---- captured log buffer -------------------------------------------------- */
/* ORT's default logger prints warnings/non-fatal errors straight to stderr (the
 * "[W:onnxruntime:...]" / "[E:onnxruntime:...]" lines -- node-placement notices,
 * the cudnnDestroy teardown gripe, etc.). We install a custom env logger that
 * appends them to a process-global buffer instead, retrievable from JS via
 * onnx.getLog(). Mirrors rampart-llamacpp's llog_cap/getLog. */
#define ONNX_MAX_LOG 65536
static struct {
    char           *buf;
    size_t          len;
    size_t          alloc;
    pthread_mutex_t mtx;
} g_log = { nullptr, 0, 0, PTHREAD_MUTEX_INITIALIZER };

static void onnx_log_append(const char *text) {
    if (!text || !*text) return;
    pthread_mutex_lock(&g_log.mtx);
    size_t tl = strlen(text);
    if (g_log.len && g_log.len + tl > ONNX_MAX_LOG) {   /* drop older half on overflow */
        static const char *warn = "WARN: onnx log overflow (older lines dropped)\n";
        size_t wl = strlen(warn), half = g_log.len / 2, keep = g_log.len - half;
        memmove(g_log.buf + wl, g_log.buf + half, keep);
        memcpy(g_log.buf, warn, wl);
        g_log.len = wl + keep; g_log.buf[g_log.len] = '\0';
    }
    if (g_log.len + tl + 1 > g_log.alloc) {
        size_t na = (g_log.len + tl + 1) < 1024 ? 1024 : (g_log.len + tl + 1) * 2;
        char *nb = (char *)realloc(g_log.buf, na);
        if (!nb) { pthread_mutex_unlock(&g_log.mtx); return; }
        g_log.buf = nb; g_log.alloc = na; if (g_log.len == 0) g_log.buf[0] = '\0';
    }
    memcpy(g_log.buf + g_log.len, text, tl + 1);
    g_log.len += tl;
    pthread_mutex_unlock(&g_log.mtx);
}

/* opt-in debug hatch: the ONLY thing that may reach stderr, and only when the
 * user asks for it (cf. RAMPART_LT_GPU_DEBUG in rampart-llamacpp.c). */
static int onnx_dbg(void) { return getenv("RAMPART_ONNX_DEBUG") != NULL; }

/* Informational: goes to the same captured buffer as ORT's own chatter --
 * onnx.getLog().  A library must never scribble on a program's stdout/stderr:
 * it corrupts piped output and spams every server start. */
void onnx_log_note(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    onnx_log_append(line);
    if (onnx_dbg()) fputs(line, stderr);
}

/* Warnings (onnx_warn) are DEFINED in rampart-onnx.c, which has duktape: they go
 * straight onto `this.errMsg` via the calling rampart thread's ctx
 * (get_current_thread()->ctx), reachable from any C code here -- including this
 * runtime ladder, which runs under pthread_once.  Declared in onnx_shim.h. */

static void ORT_API_CALL onnx_log_cb(void *param, OrtLoggingLevel sev, const char *cat,
                                     const char *logid, const char *loc, const char *msg) {
    (void)param; (void)logid;
    const char *lv = sev == ORT_LOGGING_LEVEL_FATAL   ? "F"
                   : sev == ORT_LOGGING_LEVEL_ERROR   ? "E"
                   : sev == ORT_LOGGING_LEVEL_WARNING ? "W"
                   : sev == ORT_LOGGING_LEVEL_INFO    ? "I" : "V";
    char line[1200];
    snprintf(line, sizeof line, "[%s] %s%s%s: %s\n",
             lv, cat ? cat : "", loc ? " " : "", loc ? loc : "", msg ? msg : "");
    onnx_log_append(line);
}

char *onnx_log_dup(void) {
    pthread_mutex_lock(&g_log.mtx);
    char *r = strdup(g_log.buf ? g_log.buf : "");
    pthread_mutex_unlock(&g_log.mtx);
    return r;
}
void onnx_log_clear(void) {
    pthread_mutex_lock(&g_log.mtx);
    g_log.len = 0; if (g_log.buf) g_log.buf[0] = '\0';
    pthread_mutex_unlock(&g_log.mtx);
}

/* ==================================================================
 * Runtime selection (one rampart-onnx.so for CPU and GPU).
 *
 * The module statically contains a full CPU-only ORT (single-file CPU
 * deployment, unchanged).  At first use we look for an OPTIONAL external
 * ORT runtime directory beside the module:
 *
 *     modules/rampart-onnx.so
 *     modules/onnx-cu13/   libonnxruntime.so.1(.27.0) + providers + sm.list
 *     modules/onnx-cu12/   (may coexist with cu13)
 *
 * Ladder: $RAMPART_ONNX_RUNTIME (cpu | cu12 | cu13 | /abs/dir) overrides
 * everything; else probe the NVIDIA driver (libcuda.so.1, driver API only)
 * and pick the newest runtime the driver supports, preferring a dir whose
 * sm.list contains this GPU's exact compute capability; a candidate that
 * fails to load just drops to the next; the floor is the built-in CPU ORT.
 * The external core is dlopen'd RTLD_LOCAL (its hidden protobuf/abseil
 * can't clash with the internal, version-script-hidden copies) and ORT
 * dlopens its CUDA provider libs from the core's own directory.
 * Everything above the OrtApi pointer (sessions, embed C API, SQL) is
 * agnostic to the choice.  onnx_runtime_desc() reports what was picked.
 * ================================================================== */

/* directory containing this module (.so), via dladdr on a local symbol */
static int rp_module_dir(char *out, size_t n) {
    Dl_info di;
    if (!dladdr((void *)(intptr_t)&rp_module_dir, &di) || !di.dli_fname) return 0;
    size_t l = strlen(di.dli_fname);
    if (l >= n) return 0;
    memcpy(out, di.dli_fname, l + 1);
    char *slash = strrchr(out, '/');
    if (!slash) return 0;
    *slash = '\0';
    return 1;
}

/* NVIDIA driver's max supported CUDA version (e.g. 12080, 13000); 0 = none.
 * Driver API only -- no CUDA toolkit needed on the machine. */
static void *g_libcuda = nullptr;
static int rp_cuda_driver_version(void) {
    g_libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_libcuda) return 0;
    int (*gv)(int *) = (int (*)(int *))dlsym(g_libcuda, "cuDriverGetVersion");
    int v = 0;
    if (!gv || gv(&v) != 0) return 0;
    return v;
}

/* 1 iff the driver initializes AND reports >= 1 device.  A driver LIBRARY
 * alone (userspace package installed, but no GPU / no kernel module --
 * cuInit fails or count is 0) must NOT count: without this gate such a box
 * would select an external CUDA runtime it can never use for the GPU. */
static int rp_cuda_has_device(void) {
    if (!g_libcuda) return 0;
    int (*init)(unsigned) = (int (*)(unsigned))dlsym(g_libcuda, "cuInit");
    int (*cnt)(int *) = (int (*)(int *))dlsym(g_libcuda, "cuDeviceGetCount");
    int n = 0;
    if (!init || !cnt || init(0) != 0 || cnt(&n) != 0) return 0;
    return n > 0;
}

/* device 0's compute capability as maj*10+min (e.g. 89, 121); -1 unknown */
static int rp_cuda_device_cc(void) {
    if (!g_libcuda) return -1;
    int (*init)(unsigned) = (int (*)(unsigned))dlsym(g_libcuda, "cuInit");
    int (*devget)(int *, int) = (int (*)(int *, int))dlsym(g_libcuda, "cuDeviceGet");
    int (*attr)(int *, int, int) = (int (*)(int *, int, int))dlsym(g_libcuda, "cuDeviceGetAttribute");
    if (!init || !devget || !attr || init(0) != 0) return -1;
    int dev = 0, maj = 0, mn = 0;
    if (devget(&dev, 0) != 0) return -1;
    if (attr(&maj, /*CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR*/ 75, dev) != 0) return -1;
    if (attr(&mn,  /*CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR*/ 76, dev) != 0) return -1;
    return maj * 10 + mn;
}

/* does dir/sm.list exist and contain cc exactly?  1 yes, 0 no, -1 no list */
static int rp_smlist_has(const char *dir, int cc) {
    char p[PATH_MAX];
    snprintf(p, sizeof p, "%s/sm.list", dir);
    FILE *f = fopen(p, "r");
    if (!f) return -1;
    int found = 0, v;
    while (fscanf(f, "%d", &v) == 1)
        if (v == cc) { found = 1; break; }
    fclose(f);
    return found;
}

/* try to load dir/libonnxruntime.so.1 as the runtime; returns its ApiBase or
 * NULL.  The dl handle is intentionally kept for the life of the process. */
static const OrtApiBase *rp_try_runtime(const char *dir) {
    char lib[PATH_MAX];
    snprintf(lib, sizeof lib, "%s/libonnxruntime.so.1", dir);
    struct stat st;
    if (stat(lib, &st) != 0) return nullptr;
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        onnx_log_note("rampart-onnx: runtime %s failed to load (%s); trying next\n",
                      dir, dlerror());
        return nullptr;
    }
    const OrtApiBase *(*get)(void) =
        (const OrtApiBase *(*)(void))dlsym(h, "OrtGetApiBase");
    const OrtApiBase *base = get ? get() : nullptr;
    if (!base || !base->GetApi(ORT_API_VERSION)) {
        onnx_log_note("rampart-onnx: runtime %s is ORT %s, incompatible with "
                      "this module (built against %s); trying next\n",
                      dir, base ? base->GetVersionString() : "?",
                      OrtGetApiBase()->GetVersionString());
        dlclose(h);
        return nullptr;
    }
    return base;
}

static const OrtApiBase *rp_pick_runtime(void) {
    char moddir[PATH_MAX];
    int have_dir = rp_module_dir(moddir, sizeof moddir);

    /* 1) explicit override */
    const char *rt = getenv("RAMPART_ONNX_RUNTIME");
    if (rt && *rt) {
        if (!strcmp(rt, "cpu") || !strcmp(rt, "internal")) {
            snprintf(g_runtime_desc, sizeof g_runtime_desc,
                     "built-in CPU (RAMPART_ONNX_RUNTIME=cpu)");
            return OrtGetApiBase();
        }
        char dir[PATH_MAX];
        if (rt[0] == '/') snprintf(dir, sizeof dir, "%s", rt);
        else if (have_dir) snprintf(dir, sizeof dir, "%s/onnx-%s", moddir, rt);
        else dir[0] = '\0';
        const OrtApiBase *b = dir[0] ? rp_try_runtime(dir) : nullptr;
        if (b) {
            snprintf(g_runtime_desc, sizeof g_runtime_desc,
                     "%s (RAMPART_ONNX_RUNTIME=%s)", dir, rt);
            onnx_log_note("rampart-onnx: runtime %s\n", g_runtime_desc);
            return b;
        }
        /* the caller explicitly demanded this runtime and is silently not getting
         * it -- the one case worth interrupting them for */
        onnx_warn("rampart-onnx: RAMPART_ONNX_RUNTIME=%s unusable; "
                      "using built-in CPU\n", rt);
        snprintf(g_runtime_desc, sizeof g_runtime_desc,
                 "built-in CPU (override '%s' unusable)", rt);
        return OrtGetApiBase();
    }

    /* 2) driver-gated auto selection: newest supported first, sm.list-exact
     * candidates ahead of the rest (an sm miss is not fatal -- ORT's PTX may
     * still JIT -- it only demotes the candidate) */
    int drv = have_dir ? rp_cuda_driver_version() : 0;
    if (drv >= 12000 && rp_cuda_has_device()) {
        char cu13[PATH_MAX], cu12[PATH_MAX];
        snprintf(cu13, sizeof cu13, "%s/onnx-cu13", moddir);
        snprintf(cu12, sizeof cu12, "%s/onnx-cu12", moddir);
        const char *cand[4];
        int n = 0;
        int cc = rp_cuda_device_cc();
        /* first pass: driver-compatible dirs whose sm.list has this GPU */
        if (drv >= 13000 && cc > 0 && rp_smlist_has(cu13, cc) == 1) cand[n++] = cu13;
        if (cc > 0 && rp_smlist_has(cu12, cc) == 1)                 cand[n++] = cu12;
        /* second pass: remaining driver-compatible dirs */
        if (drv >= 13000 && (n == 0 || strcmp(cand[0], cu13) != 0)) cand[n++] = cu13;
        {
            int seen12 = 0;
            for (int i = 0; i < n; i++) if (!strcmp(cand[i], cu12)) seen12 = 1;
            if (!seen12) cand[n++] = cu12;
        }
        for (int i = 0; i < n; i++) {
            const OrtApiBase *b = rp_try_runtime(cand[i]);
            if (b) {
                snprintf(g_runtime_desc, sizeof g_runtime_desc,
                         "%s (driver CUDA %d.%d, sm %d)",
                         cand[i], drv / 1000, (drv % 1000) / 10, cc);
                onnx_log_note("rampart-onnx: runtime %s\n", g_runtime_desc);
                return b;
            }
        }
    }

    /* 3) floor: the ORT statically linked into this module */
    snprintf(g_runtime_desc, sizeof g_runtime_desc, "built-in CPU");
    return OrtGetApiBase();
}

const char *onnx_runtime_desc(void) {
    return g_runtime_desc;
}

static void shim_init(void) {
    g_base = rp_pick_runtime();
    g_ort = g_base->GetApi(ORT_API_VERSION);
    if (!g_ort) return;
    /* One env per process; intentionally never released (lives to exit). ORT's
     * statically-linked globals register exit handlers (static dtors) whose code
     * lives in this .so, so the module is linked '-z nodelete' to stay mapped
     * through process exit -- otherwise rampart's dlclose unmaps it and libc's
     * __run_exit_handlers jumps into freed text (segfault on the glibc-2.17
     * runtime; the host's newer libc happened to survive it). The custom logger
     * routes ORT's warnings/non-fatal errors to our buffer (onnx.getLog()) instead
     * of stderr. */
    OrtStatus *st = g_ort->CreateEnvWithCustomLogger(onnx_log_cb, nullptr,
                        ORT_LOGGING_LEVEL_WARNING, "rampart-onnx", &g_env);
    if (st) { g_ort->ReleaseStatus(st); g_env = nullptr; }
}

static const OrtApi *ort(void) {
    pthread_once(&g_once, shim_init);
    return g_ort;
}

/* ---- error helper -------------------------------------------------------- */

/* copy an OrtStatus message into err[], release it, return -1. */
static int fail_status(OrtStatus *st, char *err, size_t errlen) {
    if (err && errlen) {
        const char *m = g_ort ? g_ort->GetErrorMessage(st) : "onnx error";
        snprintf(err, errlen, "%s", m ? m : "onnx error");
    }
    if (g_ort && st) g_ort->ReleaseStatus(st);
    return -1;
}
static int fail_msg(const char *m, char *err, size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "%s", m);
    return -1;
}

/* ---- dtype mapping ------------------------------------------------------- */

size_t onnx_dtype_size(int dt) {
    switch (dt) {
        case ONNX_DT_FLOAT32: return 4;
        case ONNX_DT_FLOAT16: return 2;
        case ONNX_DT_DOUBLE:  return 8;
        case ONNX_DT_INT64:   return 8;
        case ONNX_DT_INT32:   return 4;
        case ONNX_DT_INT16:   return 2;
        case ONNX_DT_INT8:    return 1;
        case ONNX_DT_UINT8:   return 1;
        case ONNX_DT_BOOL:    return 1;
        default:              return 0;
    }
}

const char *onnx_dtype_name(int dt) {
    switch (dt) {
        case ONNX_DT_FLOAT32: return "float32";
        case ONNX_DT_FLOAT16: return "float16";
        case ONNX_DT_DOUBLE:  return "float64";
        case ONNX_DT_INT64:   return "int64";
        case ONNX_DT_INT32:   return "int32";
        case ONNX_DT_INT16:   return "int16";
        case ONNX_DT_INT8:    return "int8";
        case ONNX_DT_UINT8:   return "uint8";
        case ONNX_DT_BOOL:    return "bool";
        default:              return "unknown";
    }
}

int onnx_dtype_from_name(const char *n) {
    if (!n) return ONNX_DT_UNKNOWN;
    if (!strcmp(n, "float32") || !strcmp(n, "float")) return ONNX_DT_FLOAT32;
    if (!strcmp(n, "float16") || !strcmp(n, "half"))  return ONNX_DT_FLOAT16;
    if (!strcmp(n, "float64") || !strcmp(n, "double"))return ONNX_DT_DOUBLE;
    if (!strcmp(n, "int64"))  return ONNX_DT_INT64;
    if (!strcmp(n, "int32") || !strcmp(n, "int")) return ONNX_DT_INT32;
    if (!strcmp(n, "int16"))  return ONNX_DT_INT16;
    if (!strcmp(n, "int8"))   return ONNX_DT_INT8;
    if (!strcmp(n, "uint8"))  return ONNX_DT_UINT8;
    if (!strcmp(n, "bool"))   return ONNX_DT_BOOL;
    return ONNX_DT_UNKNOWN;
}

static int ort_to_dtype(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return ONNX_DT_FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return ONNX_DT_FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return ONNX_DT_DOUBLE;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return ONNX_DT_INT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return ONNX_DT_INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return ONNX_DT_INT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return ONNX_DT_INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return ONNX_DT_UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return ONNX_DT_BOOL;
        default:                                    return ONNX_DT_UNKNOWN;
    }
}

static ONNXTensorElementDataType dtype_to_ort(int dt) {
    switch (dt) {
        case ONNX_DT_FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case ONNX_DT_FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case ONNX_DT_DOUBLE:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case ONNX_DT_INT64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case ONNX_DT_INT32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case ONNX_DT_INT16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case ONNX_DT_INT8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case ONNX_DT_UINT8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case ONNX_DT_BOOL:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default:              return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

const char *onnx_version(void) {
    if (!ort() || !g_base) return "unknown";
    return g_base->GetVersionString();   /* the SELECTED runtime's version */
}

/* ---- introspection ------------------------------------------------------- */

/* onnx_session.flags bits */
#define ONNX_F_BG_THREADS (1u << 0)  /* session has ORT background threads (rebuild on fork) */
#define ONNX_F_USES_GPU   (1u << 1)  /* (future) GPU EP -> throw on fork */
#define ONNX_F_DESTROYED  (1u << 2)  /* destroy() called -- process-wide, set under mtx */

struct onnx_session {
    OrtSession        *sess;
    OrtSessionOptions *opts;
    onnx_modelinfo     info;
    /* fork handling: enough to re-create the session in a forked child. */
    char              *model_path;     /* one of path / data is set */
    void              *model_data;
    size_t             model_len;
    onnx_session_opts  opts_saved;
    unsigned int       flags;          /* ONNX_F_* (bg-threads | uses-gpu | destroyed) */
    int                create_pid;
    pthread_mutex_t    mtx;
};

/* Fill one io list (inputs or outputs) from a live session. Returns 0/-1. */
static int read_io(OrtSession *sess, int is_input,
                   onnx_iodesc **out_list, size_t *out_n,
                   char *err, size_t errlen) {
    const OrtApi *o = g_ort;
    OrtAllocator *alloc = nullptr;
    OrtStatus *st = o->GetAllocatorWithDefaultOptions(&alloc);
    if (st) return fail_status(st, err, errlen);

    size_t n = 0;
    st = is_input ? o->SessionGetInputCount(sess, &n)
                  : o->SessionGetOutputCount(sess, &n);
    if (st) return fail_status(st, err, errlen);

    onnx_iodesc *list = (onnx_iodesc *)calloc(n ? n : 1, sizeof(onnx_iodesc));
    if (!list) return fail_msg("oom", err, errlen);

    /* break-on-error flag rather than goto (C++ forbids jumping across the
     * in-loop initializations of ti/tsi/et/nd). */
    int failed = 0, oom = 0;
    for (size_t i = 0; i < n; ++i) {
        char *nm = nullptr;
        st = is_input ? o->SessionGetInputName(sess, i, alloc, &nm)
                      : o->SessionGetOutputName(sess, i, alloc, &nm);
        if (st) { failed = 1; break; }
        list[i].name = strdup(nm ? nm : "");
        o->AllocatorFree(alloc, nm);

        OrtTypeInfo *ti = nullptr;
        st = is_input ? o->SessionGetInputTypeInfo(sess, i, &ti)
                      : o->SessionGetOutputTypeInfo(sess, i, &ti);
        if (st) { failed = 1; break; }

        const OrtTensorTypeAndShapeInfo *tsi = nullptr;
        st = o->CastTypeInfoToTensorInfo(ti, &tsi);  /* tsi borrowed from ti */
        if (st || !tsi) {
            /* non-tensor io (sequence/map) -- record unknown, no shape */
            if (st) { o->ReleaseStatus(st); st = nullptr; }
            list[i].dtype = ONNX_DT_UNKNOWN;
            o->ReleaseTypeInfo(ti);
            continue;
        }
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        o->GetTensorElementType(tsi, &et);
        list[i].dtype = ort_to_dtype(et);

        size_t nd = 0;
        o->GetDimensionsCount(tsi, &nd);
        list[i].n_dims = nd;
        if (nd) {
            list[i].shape = (int64_t *)malloc(nd * sizeof(int64_t));
            if (!list[i].shape) { o->ReleaseTypeInfo(ti); oom = 1; failed = 1; break; }
            o->GetDimensions(tsi, list[i].shape, nd);
        }
        o->ReleaseTypeInfo(ti);
    }

    if (failed) {
        for (size_t j = 0; j < n; ++j) { free(list[j].name); free(list[j].shape); }
        free(list);
        return oom ? fail_msg("oom", err, errlen) : fail_status(st, err, errlen);
    }

    *out_list = list;
    *out_n = n;
    return 0;
}

static void free_io(onnx_iodesc *list, size_t n) {
    if (!list) return;
    for (size_t i = 0; i < n; ++i) { free(list[i].name); free(list[i].shape); }
    free(list);
}

static int fill_info(OrtSession *sess, onnx_modelinfo *mi, char *err, size_t errlen) {
    memset(mi, 0, sizeof(*mi));
    if (read_io(sess, 1, &mi->inputs, &mi->n_inputs, err, errlen) != 0) return -1;
    if (read_io(sess, 0, &mi->outputs, &mi->n_outputs, err, errlen) != 0) {
        free_io(mi->inputs, mi->n_inputs);
        mi->inputs = nullptr; mi->n_inputs = 0;
        return -1;
    }
    return 0;
}

/* ---- session lifecycle --------------------------------------------------- */

static OrtSessionOptions *make_opts(const onnx_session_opts *so, char *err, size_t errlen) {
    const OrtApi *o = g_ort;
    OrtSessionOptions *opts = nullptr;
    OrtStatus *st = o->CreateSessionOptions(&opts);
    if (st) { fail_status(st, err, errlen); return nullptr; }

    if (so) {
        if (so->intra_threads > 0) o->SetIntraOpNumThreads(opts, so->intra_threads);
        if (so->inter_threads > 0) o->SetInterOpNumThreads(opts, so->inter_threads);
        /* Execution mode: SEQUENTIAL avoids the inter-op threadpool entirely.
         * Combined with intra_threads==1 this means ORT spawns NO background
         * threads -> the session survives fork(). (Parallel/multi-thread is
         * opt-in and is NOT fork-safe; create such sessions post-fork.) */
        o->SetSessionExecutionMode(opts, so->execution_mode == 1 ? ORT_PARALLEL : ORT_SEQUENTIAL);
        GraphOptimizationLevel lvl = ORT_ENABLE_ALL;
        switch (so->graph_opt) {
            case 0: lvl = ORT_DISABLE_ALL;    break;
            case 1: lvl = ORT_ENABLE_BASIC;   break;
            case 2: lvl = ORT_ENABLE_EXTENDED;break;
            case 3: lvl = ORT_ENABLE_ALL;     break;
            default: lvl = ORT_ENABLE_ALL;    break; /* <0 => default */
        }
        o->SetSessionGraphOptimizationLevel(opts, lvl);

        /* CUDA execution provider. These OrtApi entry points exist in every ORT
         * build (CPU included) -- on a CPU-only build the append below returns an
         * error status ("CUDA EP is not enabled in this build") which we surface,
         * rather than failing to link. So the same shim serves both flavors. */
        if (so->use_cuda) {
            OrtCUDAProviderOptionsV2 *cu = nullptr;
            OrtStatus *cst = o->CreateCUDAProviderOptions(&cu);
            if (cst) { o->ReleaseSessionOptions(opts); fail_status(cst, err, errlen); return nullptr; }
            char devbuf[16];
            snprintf(devbuf, sizeof devbuf, "%d", so->cuda_device_id < 0 ? 0 : so->cuda_device_id);
            const char *keys[] = { "device_id" };
            const char *vals[] = { devbuf };
            o->UpdateCUDAProviderOptions(cu, keys, vals, 1);
            OrtStatus *ast = o->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cu);
            o->ReleaseCUDAProviderOptions(cu);
            if (ast) { o->ReleaseSessionOptions(opts); fail_status(ast, err, errlen); return nullptr; }
        }

        /* CoreML execution provider (macOS GPU / Neural Engine).  Appended via
         * the generic named-EP entry point: on an ORT built without CoreML it
         * returns a clear error status ("unknown provider"), never a link
         * failure, so the same shim serves every flavor.  MLProgram format +
         * dynamic shapes: the embed path feeds ragged, per-batch-padded
         * sequences, which the legacy NeuralNetwork format would reject. */
        if (so->use_coreml) {
            const char *units = "ALL";
            switch (so->coreml_units) {
                case 1: units = "CPUAndGPU";           break;
                case 2: units = "CPUAndNeuralEngine";  break;
                case 3: units = "CPUOnly";             break;
                default:                               break;  /* 0 => ALL */
            }
            const char *keys[] = { "ModelFormat", "MLComputeUnits",
                                   "RequireStaticInputShapes", "EnableOnSubgraphs" };
            const char *vals[] = { "MLProgram", units, "0", "0" };
            OrtStatus *ast = o->SessionOptionsAppendExecutionProvider(opts, "CoreML", keys, vals, 4);
            if (ast) { o->ReleaseSessionOptions(opts); fail_status(ast, err, errlen); return nullptr; }
        }
    }
    return opts;
}

/* wrap an already-created OrtSession (+ its opts) into our handle. Takes
 * ownership of both; releases them on failure. */
static onnx_session *wrap_session(OrtSession *sess, OrtSessionOptions *opts,
                                  char *err, size_t errlen) {
    const OrtApi *o = g_ort;
    onnx_session *s = (onnx_session *)calloc(1, sizeof(onnx_session));
    if (!s) { o->ReleaseSession(sess); o->ReleaseSessionOptions(opts);
              fail_msg("oom", err, errlen); return nullptr; }
    s->sess = sess;
    s->opts = opts;
    if (fill_info(sess, &s->info, err, errlen) != 0) {
        o->ReleaseSession(sess); o->ReleaseSessionOptions(opts); free(s);
        return nullptr;
    }
    return s;
}

/* Stash everything needed to re-create the session in a forked child. */
static void record_source(onnx_session *s, const char *path,
                          const void *data, size_t len, const onnx_session_opts *so) {
    if (path) s->model_path = strdup(path);
    if (data && len) {
        s->model_data = malloc(len);
        if (s->model_data) { memcpy(s->model_data, data, len); s->model_len = len; }
    }
    if (so) s->opts_saved = *so;
    else { s->opts_saved.intra_threads = 0; s->opts_saved.inter_threads = 0;
           s->opts_saved.graph_opt = ONNX_OPT_DEFAULT; s->opts_saved.execution_mode = 0; }
    s->flags = 0;
    /* intra_threads != 1 covers <= 0 too: make_opts leaves ORT's default
     * (multi-threaded) pool in that case, so the session DOES have background
     * threads and must be rebuilt after fork.  A NULL so is the same story. */
    if (!so || so->intra_threads != 1 || so->inter_threads > 1 || so->execution_mode == 1)
        s->flags |= ONNX_F_BG_THREADS;
    /* A GPU session (CUDA or CoreML) can't be inherited across fork() (the GPU
     * context dies with the parent), so flag it -> onnx_session_ensure_runnable
     * throws after a fork instead of rebuilding, mirroring rampart-llamacpp's
     * CUDA-fork throw. */
    if (so && (so->use_cuda || so->use_coreml)) s->flags |= ONNX_F_USES_GPU;
    s->create_pid = (int)getpid();
    pthread_mutex_init(&s->mtx, nullptr);
}

static int ep_available(const char *name) {
    const OrtApi *o = ort();
    if (!o) return 0;
    char **provs = nullptr;
    int n = 0;
    if (o->GetAvailableProviders(&provs, &n)) return 0;
    int have = 0;
    for (int i = 0; i < n; i++)
        if (provs[i] && !strcmp(provs[i], name)) { have = 1; break; }
    o->ReleaseAvailableProviders(provs, n);
    return have;
}

int onnx_cuda_ep_available(void) {
    return ep_available("CUDAExecutionProvider");
}

int onnx_coreml_ep_available(void) {
    return ep_available("CoreMLExecutionProvider");
}

onnx_session *onnx_session_create(const char *path, const onnx_session_opts *so,
                                  char *err, size_t errlen) {
    const OrtApi *o = ort();
    if (!o || !g_env) { fail_msg("onnx runtime failed to initialize", err, errlen); return nullptr; }
    OrtSessionOptions *opts = make_opts(so, err, errlen);
    if (!opts) return nullptr;
    OrtSession *sess = nullptr;
    OrtStatus *st = o->CreateSession(g_env, path, opts, &sess);
    if (st) { o->ReleaseSessionOptions(opts); fail_status(st, err, errlen); return nullptr; }
    onnx_session *s = wrap_session(sess, opts, err, errlen);
    if (s) record_source(s, path, nullptr, 0, so);
    return s;
}

onnx_session *onnx_session_create_from_buffer(const void *data, size_t len,
                                              const onnx_session_opts *so,
                                              char *err, size_t errlen) {
    const OrtApi *o = ort();
    if (!o || !g_env) { fail_msg("onnx runtime failed to initialize", err, errlen); return nullptr; }
    if (!data || !len) { fail_msg("empty model buffer", err, errlen); return nullptr; }
    OrtSessionOptions *opts = make_opts(so, err, errlen);
    if (!opts) return nullptr;
    OrtSession *sess = nullptr;
    /* ORT parses the model from the array during this call; the buffer need not
     * outlive it. */
    OrtStatus *st = o->CreateSessionFromArray(g_env, data, len, opts, &sess);
    if (st) { o->ReleaseSessionOptions(opts); fail_status(st, err, errlen); return nullptr; }
    onnx_session *s = wrap_session(sess, opts, err, errlen);
    if (s) record_source(s, nullptr, data, len, so);
    return s;
}

/* Re-create s->sess from the stored source + opts (caller holds s->mtx). The
 * inherited session is intentionally NOT released: in a forked child its
 * threadpool workers no longer exist, so ReleaseSession could hang joining them
 * -- we abandon it (mirrors rampart-llamacpp abandoning the inherited context).
 * The inherited OrtSessionOptions carries no threads and is released. */
static int rebuild_session_locked(onnx_session *s, char *err, size_t errlen) {
    const OrtApi *o = g_ort;
    OrtSessionOptions *opts = make_opts(&s->opts_saved, err, errlen);
    if (!opts) return -1;
    OrtSession *sess = nullptr;
    OrtStatus *st = s->model_path
        ? o->CreateSession(g_env, s->model_path, opts, &sess)
        : o->CreateSessionFromArray(g_env, s->model_data, s->model_len, opts, &sess);
    if (st) { o->ReleaseSessionOptions(opts); return fail_status(st, err, errlen); }
    o->ReleaseSessionOptions(s->opts);   /* options carry no threads -> safe */
    s->sess = sess;                      /* old s->sess deliberately leaked */
    s->opts = opts;
    return 0;
}

int onnx_session_ensure_runnable(onnx_session *s, char *err, size_t errlen) {
    if (!s) return fail_msg("null session", err, errlen);
    pthread_mutex_lock(&s->mtx);
    int rc = 0;
    if (s->flags & ONNX_F_DESTROYED) {
        rc = fail_msg("session has been destroyed", err, errlen);
    } else {
        int pid = (int)getpid();
        if (s->create_pid != pid) {              /* forked since creation */
            if (s->flags & ONNX_F_USES_GPU) {
                rc = fail_msg("cannot use a GPU ONNX session after fork() -- create it in the child process",
                              err, errlen);
            } else if (s->flags & ONNX_F_BG_THREADS) {
                rc = rebuild_session_locked(s, err, errlen);  /* threadpool broken by fork */
                if (rc == 0) s->create_pid = pid;
            } else {
                s->create_pid = pid;   /* single-threaded: inherited session is fork-safe */
            }
        }
    }
    pthread_mutex_unlock(&s->mtx);
    return rc;
}

void onnx_session_mark_destroyed(onnx_session *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mtx);
    s->flags |= ONNX_F_DESTROYED;
    pthread_mutex_unlock(&s->mtx);
}

int onnx_session_is_destroyed(onnx_session *s) {
    return s ? ((s->flags & ONNX_F_DESTROYED) != 0) : 1;  /* plain read; bit set under mtx */
}

/* ---- model metadata ------------------------------------------------------ */

/* Read string metadata fields + version into caller-owned strings (malloc'd;
 * any may be NULL). Returns 0 always (best-effort; missing fields stay NULL). */
int onnx_session_metadata(onnx_session *s,
                          char **producer, char **graph_name, char **domain,
                          char **description, int64_t *version,
                          char *err, size_t errlen) {
    const OrtApi *o = ort();
    if (!o || !s) return fail_msg("null session", err, errlen);
    *producer = *graph_name = *domain = *description = nullptr;
    *version = 0;
    OrtAllocator *alloc = nullptr;
    if (o->GetAllocatorWithDefaultOptions(&alloc)) return 0;
    OrtModelMetadata *md = nullptr;
    if (o->SessionGetModelMetadata(s->sess, &md) || !md) return 0;

    char *tmp = nullptr;
    if (!o->ModelMetadataGetProducerName(md, alloc, &tmp) && tmp) { *producer = strdup(tmp); o->AllocatorFree(alloc, tmp); tmp = nullptr; }
    if (!o->ModelMetadataGetGraphName(md, alloc, &tmp) && tmp)    { *graph_name = strdup(tmp); o->AllocatorFree(alloc, tmp); tmp = nullptr; }
    if (!o->ModelMetadataGetDomain(md, alloc, &tmp) && tmp)       { *domain = strdup(tmp); o->AllocatorFree(alloc, tmp); tmp = nullptr; }
    if (!o->ModelMetadataGetDescription(md, alloc, &tmp) && tmp)  { *description = strdup(tmp); o->AllocatorFree(alloc, tmp); tmp = nullptr; }
    o->ModelMetadataGetVersion(md, version);
    o->ReleaseModelMetadata(md);
    return 0;
}

void onnx_session_destroy(onnx_session *s) {
    if (!s) return;
    const OrtApi *o = g_ort;
    free_io(s->info.inputs, s->info.n_inputs);
    free_io(s->info.outputs, s->info.n_outputs);
    if (o) {
        if (s->sess) o->ReleaseSession(s->sess);
        if (s->opts) o->ReleaseSessionOptions(s->opts);
    }
    free(s->model_path);
    free(s->model_data);
    pthread_mutex_destroy(&s->mtx);
    free(s);
}

const onnx_modelinfo *onnx_session_info(onnx_session *s) {
    return s ? &s->info : nullptr;
}

int onnx_model_info(const char *path, onnx_modelinfo **out, char *err, size_t errlen) {
    onnx_session *s = onnx_session_create(path, nullptr, err, errlen);
    if (!s) return -1;
    onnx_modelinfo *mi = (onnx_modelinfo *)calloc(1, sizeof(onnx_modelinfo));
    if (!mi) { onnx_session_destroy(s); return fail_msg("oom", err, errlen); }
    /* steal the session's info (deep copy by re-reading is wasteful; move it) */
    if (fill_info(s->sess, mi, err, errlen) != 0) { free(mi); onnx_session_destroy(s); return -1; }
    onnx_session_destroy(s);
    *out = mi;
    return 0;
}

void onnx_modelinfo_free(onnx_modelinfo *mi) {
    if (!mi) return;
    free_io(mi->inputs, mi->n_inputs);
    free_io(mi->outputs, mi->n_outputs);
    free(mi);
}

/* ---- run ----------------------------------------------------------------- */

int onnx_session_run(onnx_session *s,
                     const onnx_value_in *ins, size_t n_ins,
                     const char *const *out_names, size_t n_out_names,
                     onnx_value_out **outs_p, size_t *n_outs_p,
                     char *err, size_t errlen) {
    const OrtApi *o = ort();
    if (!o) return fail_msg("onnx runtime not initialized", err, errlen);
    if (!s) return fail_msg("null session", err, errlen);

    OrtMemoryInfo *mem = nullptr;
    OrtStatus *st = o->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
    if (st) return fail_status(st, err, errlen);

    /* Build input OrtValues wrapping caller data (borrowed). */
    OrtValue   **in_vals  = (OrtValue **)calloc(n_ins ? n_ins : 1, sizeof(OrtValue *));
    const char **in_names = (const char **)calloc(n_ins ? n_ins : 1, sizeof(char *));
    if (!in_vals || !in_names) { o->ReleaseMemoryInfo(mem); free(in_vals); free(in_names);
                                 return fail_msg("oom", err, errlen); }

    int rc = 0;
    for (size_t i = 0; i < n_ins; ++i) {
        ONNXTensorElementDataType et = dtype_to_ort(ins[i].dtype);
        if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
            rc = fail_msg("unsupported input dtype", err, errlen); goto cleanup_inputs;
        }
        st = o->CreateTensorWithDataAsOrtValue(
                mem, (void *)ins[i].data, ins[i].n_bytes,
                ins[i].shape, ins[i].n_dims, et, &in_vals[i]);
        if (st) { rc = fail_status(st, err, errlen); goto cleanup_inputs; }
        in_names[i] = ins[i].name;
    }

    /* Resolve output names: caller-given subset, or all model outputs. */
    {
        const char **onames = nullptr;
        size_t       n_out  = 0;
        int          onames_owned = 0;
        if (out_names && n_out_names) {
            onames = (const char **)out_names;
            n_out  = n_out_names;
        } else {
            n_out  = s->info.n_outputs;
            onames = (const char **)malloc((n_out ? n_out : 1) * sizeof(char *));
            if (!onames) { rc = fail_msg("oom", err, errlen); goto cleanup_inputs; }
            onames_owned = 1;
            for (size_t i = 0; i < n_out; ++i) onames[i] = s->info.outputs[i].name;
        }

        OrtValue **out_vals = (OrtValue **)calloc(n_out ? n_out : 1, sizeof(OrtValue *));
        if (!out_vals) { if (onames_owned) free((void *)onames);
                         rc = fail_msg("oom", err, errlen); goto cleanup_inputs; }

        /* CUDA sessions: serialize Run().  ORT allows concurrent Run on
         * one session, but each in-flight run allocates its own
         * activation buffers from the session's BFC GPU arena -- N
         * concurrent big batches multiply the VRAM peak until
         * AllocateRawInternal fails (observed with 2 threads x 64-chunk
         * batches).  The GPU gets its throughput from batch size, not
         * kernel concurrency, so callers lose little; CPU-side tensor
         * prep above and copy-out below stay outside the lock, and CPU
         * sessions stay fully concurrent. */
        int gpu_serialize = (s->flags & ONNX_F_USES_GPU) ? 1 : 0;

        /* Periodic ARENA SHRINKAGE.  ORT's arena allocator retains its
         * high-water mark per distinct input SHAPE, and the embed path
         * feeds ragged (chunk-count x seq-len) batches — every distinct
         * shape ratchets the arena up and nothing ever releases it.
         * Observed: a helper process embedding large multi-chunk
         * documents plateaus GBs above its working set (~120MB retained
         * per distinct document shape).  Shrinking on a cadence caps
         * the ratchet while amortizing the arena re-grow cost to noise.
         * RAMPART_ONNX_ARENA_SHRINK overrides the cadence (0 disables,
         * N = shrink every Nth Run; default 16). */
        static std::atomic<unsigned> arena_run_ctr{0};
        static int arena_cadence = -1;
        if (arena_cadence < 0) {
            const char *e = getenv("RAMPART_ONNX_ARENA_SHRINK");
            arena_cadence = e ? atoi(e) : 16;
            if (arena_cadence < 0) arena_cadence = 0;
        }
        OrtRunOptions *shrink_ro = nullptr;
        if (arena_cadence > 0 &&
            (arena_run_ctr.fetch_add(1) + 1) % (unsigned)arena_cadence == 0) {
            if (!o->CreateRunOptions(&shrink_ro)) {
                o->AddRunConfigEntry(shrink_ro,
                    "memory.enable_memory_arena_shrinkage",
                    (s->flags & ONNX_F_USES_GPU) ? "gpu:0;cpu:0" : "cpu:0");
            }
        }

        if (gpu_serialize) pthread_mutex_lock(&s->mtx);
        st = o->Run(s->sess, shrink_ro, in_names, (const OrtValue *const *)in_vals, n_ins,
                    onames, n_out, out_vals);
        if (shrink_ro) { o->ReleaseRunOptions(shrink_ro); shrink_ro = nullptr; }
        if (st && gpu_serialize) {
            /* GPU Run failure is usually the BFC arena out of (or too
             * fragmented for) VRAM.  Retry once with arena shrinkage:
             * the run-option releases the arena's unused chunks after
             * the retry completes, defragmenting the device so
             * subsequent runs re-grow cleanly.  Small retries (the
             * caller halves its batch on failure) generally fit in the
             * existing fragments, so this turns a permanent OOM spiral
             * into a self-healing blip. */
            o->ReleaseStatus(st);
            st = nullptr;
            OrtRunOptions *ro = nullptr;
            if (!o->CreateRunOptions(&ro)) {
                o->AddRunConfigEntry(ro, "memory.enable_memory_arena_shrinkage", "gpu:0");
                st = o->Run(s->sess, ro, in_names, (const OrtValue *const *)in_vals, n_ins,
                            onames, n_out, out_vals);
                o->ReleaseRunOptions(ro);
            } else {
                st = o->Run(s->sess, nullptr, in_names, (const OrtValue *const *)in_vals, n_ins,
                            onames, n_out, out_vals);
            }
        }
        if (gpu_serialize) pthread_mutex_unlock(&s->mtx);
        if (st) { rc = fail_status(st, err, errlen);
                  /* out_vals is calloc'd; release any outputs ORT populated
                   * before erroring (mirrors the oom path below) */
                  for (size_t i = 0; i < n_out; i++) if (out_vals[i]) o->ReleaseValue(out_vals[i]);
                  free(out_vals); if (onames_owned) free((void *)onames); goto cleanup_inputs; }

        /* Copy each output tensor out into malloc'd buffers. results is fully
         * calloc-zeroed so onnx_run_free can clean a partial fill safely. Use a
         * break-on-error flag rather than goto (C++ forbids jumping across the
         * in-loop initializations). */
        onnx_value_out *results = (onnx_value_out *)calloc(n_out ? n_out : 1, sizeof(onnx_value_out));
        if (!results) { for (size_t i=0;i<n_out;i++) if(out_vals[i]) o->ReleaseValue(out_vals[i]);
                        free(out_vals); if (onames_owned) free((void *)onames);
                        rc = fail_msg("oom", err, errlen); goto cleanup_inputs; }

        int copy_rc = 0;
        for (size_t i = 0; i < n_out; ++i) {
            results[i].name = strdup(onames[i] ? onames[i] : "");
            OrtTensorTypeAndShapeInfo *tsi = nullptr;
            st = o->GetTensorTypeAndShape(out_vals[i], &tsi);
            if (st) { copy_rc = fail_status(st, err, errlen); break; }

            ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            o->GetTensorElementType(tsi, &et);
            results[i].dtype = ort_to_dtype(et);

            size_t nd = 0; o->GetDimensionsCount(tsi, &nd);
            results[i].n_dims = nd;
            if (nd) {
                results[i].shape = (int64_t *)malloc(nd * sizeof(int64_t));
                if (!results[i].shape) { o->ReleaseTensorTypeAndShapeInfo(tsi);
                                         copy_rc = fail_msg("oom", err, errlen); break; }
                o->GetDimensions(tsi, results[i].shape, nd);
            }
            size_t nelem = 0; o->GetTensorShapeElementCount(tsi, &nelem);
            o->ReleaseTensorTypeAndShapeInfo(tsi);
            results[i].n_elems = nelem;

            size_t esz = onnx_dtype_size(results[i].dtype);
            if (esz == 0) { copy_rc = fail_msg("unsupported output dtype", err, errlen); break; }
            results[i].n_bytes = nelem * esz;

            void *src = nullptr;
            st = o->GetTensorMutableData(out_vals[i], &src);
            if (st) { copy_rc = fail_status(st, err, errlen); break; }
            results[i].data = malloc(results[i].n_bytes ? results[i].n_bytes : 1);
            if (!results[i].data) { copy_rc = fail_msg("oom", err, errlen); break; }
            memcpy(results[i].data, src, results[i].n_bytes);
        }

        /* ORT output values + name array are no longer needed either way. */
        for (size_t i = 0; i < n_out; ++i) if (out_vals[i]) o->ReleaseValue(out_vals[i]);
        free(out_vals);
        if (onames_owned) free((void *)onames);

        if (copy_rc != 0) {
            onnx_run_free(results, n_out);
            rc = copy_rc;
            goto cleanup_inputs;
        }

        *outs_p   = results;
        *n_outs_p = n_out;
        rc = 0;
    }

cleanup_inputs:
    for (size_t i = 0; i < n_ins; ++i) if (in_vals[i]) o->ReleaseValue(in_vals[i]);
    free(in_vals);
    free(in_names);
    o->ReleaseMemoryInfo(mem);
    return rc;
}

void onnx_run_free(onnx_value_out *outs, size_t n) {
    if (!outs) return;
    for (size_t i = 0; i < n; ++i) {
        free(outs[i].name);
        free(outs[i].shape);
        free(outs[i].data);
    }
    free(outs);
}

/* ======================= Native tokenizers (extensions) =======================
 * WordPiece uses extensions' BertTokenizer C++ class directly (no session/graph);
 * SentencePiece/BPE uses the Ortx C API. Both return CONTENT ids only. The bundled
 * protobuf/re2 these pull in are localized in onnxext_all.o (rampart-build-ext.sh),
 * so they don't collide with ORT's. */
#ifdef RAMPART_ONNX_EXT
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include "bert_tokenizer.hpp"   /* BertTokenizer, ustring */
#include "ortx_tokenizer.h"      /* OrtxCreateTokenizer / OrtxTokenize */

struct onnx_wp_tokenizer { BertTokenizer *tok; size_t vocab_size; };
struct onnx_sp_tokenizer { OrtxTokenizer *tok; };

onnx_wp_tokenizer *onnx_wp_create(const char *vocab_path, int lowercase,
                                  int strip_accents, int tokenize_chinese,
                                  char *err, size_t errlen) {
    std::ifstream f(vocab_path);
    if (!f) { if (err) snprintf(err, errlen, "cannot open vocab: %s", vocab_path); return nullptr; }
    std::stringstream ss; ss << f.rdbuf();
    std::string vocab = ss.str();
    size_t nlines = 0;
    for (size_t i = 0; i < vocab.size(); i++) if (vocab[i] == '\n') nlines++;
    try {
        /* max_len only matters if Truncate() is called; we only Tokenize+Encode,
         * so it never truncates -- the embed layer owns sequence length. */
        BertTokenizer *t = new BertTokenizer(
            vocab, lowercase != 0, /*do_basic_tokenize*/ true,
            ustring("[UNK]"), ustring("[SEP]"), ustring("[PAD]"),
            ustring("[CLS]"), ustring("[MASK]"),
            tokenize_chinese != 0, strip_accents != 0,
            ustring("##"), /*max_len*/ INT32_MAX, "longest_first");
        return new onnx_wp_tokenizer{ t, nlines };
    } catch (const std::exception &e) {
        if (err) snprintf(err, errlen, "BertTokenizer: %s", e.what());
        return nullptr;
    } catch (...) {
        if (err) snprintf(err, errlen, "BertTokenizer: unknown error");
        return nullptr;
    }
}

int onnx_wp_encode(onnx_wp_tokenizer *t, const char *text, int64_t **ids, size_t *n_ids) {
    if (!t || !t->tok || !ids || !n_ids) return -1;
    try {
        std::list<BertTokenizer::OffsetMappingType> off;
        std::vector<ustring> toks = t->tok->Tokenize(ustring(text), off, false);
        std::vector<int64_t> v = t->tok->Encode(toks);
        int64_t *out = (int64_t *)malloc(sizeof(int64_t) * (v.size() ? v.size() : 1));
        if (!out) return -1;
        for (size_t i = 0; i < v.size(); i++) out[i] = v[i];
        *ids = out; *n_ids = v.size();
        return 0;
    } catch (...) { return -1; }
}

size_t onnx_wp_vocab_size(onnx_wp_tokenizer *t) { return t ? t->vocab_size : 0; }
void   onnx_wp_destroy(onnx_wp_tokenizer *t) { if (t) { delete t->tok; delete t; } }

onnx_sp_tokenizer *onnx_sp_create(const char *model_dir, char *err, size_t errlen) {
    OrtxTokenizer *tok = nullptr;
    extError_t e = OrtxCreateTokenizer(&tok, model_dir);
    if (e != kOrtxOK) {
        if (err) snprintf(err, errlen, "OrtxCreateTokenizer(%s): %s",
                          model_dir, OrtxGetLastErrorMessage());
        return nullptr;
    }
    return new onnx_sp_tokenizer{ tok };
}

int onnx_sp_encode(onnx_sp_tokenizer *t, const char *text, int64_t **ids, size_t *n_ids) {
    if (!t || !t->tok || !ids || !n_ids) return -1;
    const char *inp[1] = { text };
    OrtxTokenId2DArray *out = nullptr;
    if (OrtxTokenize(t->tok, inp, 1, &out) != kOrtxOK) return -1;
    const extTokenId_t *arr = nullptr; size_t n = 0;
    OrtxTokenId2DArrayGetItem(out, 0, &arr, &n);
    int64_t *o = (int64_t *)malloc(sizeof(int64_t) * (n ? n : 1));
    if (!o) { OrtxDispose((OrtxObject **)&out); return -1; }
    for (size_t i = 0; i < n; i++) o[i] = (int64_t)arr[i];
    OrtxDispose((OrtxObject **)&out);
    *ids = o; *n_ids = n;
    return 0;
}

void onnx_sp_destroy(onnx_sp_tokenizer *t) {
    if (t) { OrtxDispose((OrtxObject **)&t->tok); delete t; }
}

#else /* !RAMPART_ONNX_EXT -- stubs so the module links either way */
struct onnx_wp_tokenizer { int _u; };
struct onnx_sp_tokenizer { int _u; };
onnx_wp_tokenizer *onnx_wp_create(const char *, int, int, int, char *err, size_t errlen) {
    if (err) snprintf(err, errlen, "onnx built without extensions"); return nullptr; }
int    onnx_wp_encode(onnx_wp_tokenizer *, const char *, int64_t **, size_t *) { return -1; }
size_t onnx_wp_vocab_size(onnx_wp_tokenizer *) { return 0; }
void   onnx_wp_destroy(onnx_wp_tokenizer *) {}
onnx_sp_tokenizer *onnx_sp_create(const char *, char *err, size_t errlen) {
    if (err) snprintf(err, errlen, "onnx built without extensions"); return nullptr; }
int    onnx_sp_encode(onnx_sp_tokenizer *, const char *, int64_t **, size_t *) { return -1; }
void   onnx_sp_destroy(onnx_sp_tokenizer *) {}
#endif /* RAMPART_ONNX_EXT */
