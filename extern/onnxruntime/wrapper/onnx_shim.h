/* onnx_shim.h -- pure C ABI over the ONNX Runtime C API.
 *
 * Keeps rampart-onnx.c free of any C++/ORT headers (mirrors how
 * llama_gen_shim.h fronts the llama.cpp generation engine). All functions are
 * extern "C"; the implementation lives in onnx_shim.cc.
 *
 * Ownership convention: any pointer returned through an out-param is owned by
 * the caller and freed with the matching *_free() below. Strings inside those
 * structs are plain malloc'd C strings.
 */
#ifndef ONNX_SHIM_H
#define ONNX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Element dtypes we marshal. Values are our own; the .cc maps them to/from
 * ONNXTensorElementDataType. Keep in sync with onnx_dtype_name() in the .cc. */
typedef enum {
    ONNX_DT_UNKNOWN = 0,
    ONNX_DT_FLOAT32,
    ONNX_DT_FLOAT16,
    ONNX_DT_DOUBLE,
    ONNX_DT_INT64,
    ONNX_DT_INT32,
    ONNX_DT_INT16,
    ONNX_DT_INT8,
    ONNX_DT_UINT8,
    ONNX_DT_BOOL
} onnx_dtype;

/* size in bytes of one element of the given dtype (0 if unknown) */
size_t onnx_dtype_size(int dtype);
/* lowercase canonical name ("float32", "int64", ...); "unknown" if unmapped */
const char *onnx_dtype_name(int dtype);
/* parse a name back to a dtype code (ONNX_DT_UNKNOWN if unrecognized) */
int onnx_dtype_from_name(const char *name);

/* ORT version string ("1.27.0"). */
const char *onnx_version(void);
/* which ORT runtime the selection ladder picked ("built-in CPU", or the
 * external runtime dir + why); resolved at first use */
const char *onnx_runtime_desc(void);

/* Captured ORT log: the INFORMATIONAL firehose (ORT's own chatter + which runtime
 * the ladder picked), routed off stderr into a buffer.  onnx_log_dup returns a
 * malloc'd copy ("" if empty) -- caller frees.  Exposed to JS as onnx.getLog(). */
char *onnx_log_dup(void);
void  onnx_log_clear(void);

/* Append an INFORMATIONAL line to that log (not a warning -- see onnx_warn). */
void  onnx_log_note(const char *fmt, ...);

/* WARNINGS + non-fatal errors -- deliberately separate from the log above, which
 * would bury them.  Any C code here may onnx_warn(): it appends to `this.errMsg`
 * (mirroring rampart-sql's errMsg) using the calling rampart thread's duk context,
 * which get_current_thread()->ctx makes reachable from ANY C code -- including the
 * runtime ladder under pthread_once and the rp_onnx_embed_* exports that
 * rampart-sql calls.  Defined in rampart-onnx.c (the side that has duktape).
 * Never reaches stderr unless RAMPART_ONNX_DEBUG is set. */
void onnx_warn(const char *fmt, ...);

/* One input or output tensor description. */
typedef struct {
    char    *name;     /* malloc'd */
    int      dtype;    /* onnx_dtype */
    int64_t *shape;    /* malloc'd; dim == -1 means dynamic/symbolic */
    size_t   n_dims;
} onnx_iodesc;

typedef struct {
    onnx_iodesc *inputs;
    size_t       n_inputs;
    onnx_iodesc *outputs;
    size_t       n_outputs;
} onnx_modelinfo;

/* Introspect a model file (creates a throwaway session internally).
 * Returns 0 and sets *out on success; -1 and fills err[] on failure. */
int  onnx_model_info(const char *path, onnx_modelinfo **out, char *err, size_t errlen);
void onnx_modelinfo_free(onnx_modelinfo *mi);

/* Session options. Use the OPT_DEFAULT sentinels to take ORT defaults. */
#define ONNX_OPT_DEFAULT (-1)
typedef struct {
    int intra_threads;   /* SetIntraOpNumThreads; <=0 => leave ORT default */
    int inter_threads;   /* SetInterOpNumThreads; <=0 => leave ORT default */
    int graph_opt;       /* 0=disable 1=basic 2=extended 3=all; <0 => default(all) */
    int execution_mode;  /* 0=sequential, 1=parallel; <0 => sequential */
    int use_cuda;        /* !=0 => append the CUDA execution provider. On a CPU-only
                          * build this returns a clear runtime error (the EP isn't
                          * present), never a link failure. */
    int cuda_device_id;  /* CUDA device ordinal when use_cuda; <0 => 0 */
    int use_coreml;      /* !=0 => append the CoreML execution provider (macOS
                          * GPU/Neural Engine). Errors clearly at session create
                          * on an ORT built without it. */
    int coreml_units;    /* MLComputeUnits when use_coreml: 0 = ALL (default),
                          * 1 = CPUAndGPU, 2 = CPUAndNeuralEngine, 3 = CPUOnly */
} onnx_session_opts;

typedef struct onnx_session onnx_session;

/* 1 if the bundled ONNX Runtime was built with the CUDA execution
 * provider (i.e. this is a _cu12/_cu13 module), else 0 (cpu / macos).
 * Says nothing about whether a usable GPU is present -- that's only
 * known when a session is created with use_cuda. */
int onnx_cuda_ep_available(void);

/* 1 if the selected ONNX Runtime was built with the CoreML execution
 * provider (macOS builds with --use_coreml), else 0 (Linux, or an older
 * macOS build without it). */
int onnx_coreml_ep_available(void);

onnx_session *onnx_session_create(const char *path, const onnx_session_opts *opts,
                                  char *err, size_t errlen);
onnx_session *onnx_session_create_from_buffer(const void *data, size_t len,
                                              const onnx_session_opts *opts,
                                              char *err, size_t errlen);
void          onnx_session_destroy(onnx_session *s);

/* Best-effort model metadata. Out strings are malloc'd (any may be NULL). */
int onnx_session_metadata(onnx_session *s,
                          char **producer, char **graph_name, char **domain,
                          char **description, int64_t *version,
                          char *err, size_t errlen);

/* Borrowed (owned by the session) -- valid until onnx_session_destroy. */
const onnx_modelinfo *onnx_session_info(onnx_session *s);

/* Pre-run gate. Call before each run. (1) If the session was destroy()'d ->
 * error. (2) If the process fork()'d since creation (pid changed), an inherited
 * multi-threaded session is broken (threadpool workers don't survive fork) -- a
 * CPU session is transparently rebuilt; a single-threaded session needs nothing;
 * a GPU session (future) errors, like rampart-llamacpp's CUDA throw.
 * Returns 0 on success/no-op, -1 and fills err[] otherwise. */
int onnx_session_ensure_runnable(onnx_session *s, char *err, size_t errlen);

/* Shared-state helpers for the cross-thread lifetime model. The session struct
 * lives once on the heap and is pointed at by every rampart-thread's copy of the
 * handle, so "destroyed" is process-wide and must live in the struct (a JS
 * property would be a per-thread copy). */
void onnx_session_mark_destroyed(onnx_session *s);   /* set the shared destroyed flag */
int  onnx_session_is_destroyed(onnx_session *s);     /* read it (1 if NULL/destroyed) */

/* A caller-provided input value for a Run. data is borrowed for the duration
 * of onnx_session_run only (ORT wraps it; results are copied out). */
typedef struct {
    const char    *name;
    int            dtype;
    const int64_t *shape;
    size_t         n_dims;
    const void    *data;
    size_t         n_bytes;
} onnx_value_in;

/* A produced output value. data/shape/name are malloc'd; free with onnx_run_free. */
typedef struct {
    char    *name;
    int      dtype;
    int64_t *shape;
    size_t   n_dims;
    void    *data;
    size_t   n_bytes;
    size_t   n_elems;
} onnx_value_out;

/* Run the session. If out_names is NULL, all model outputs are produced.
 * Returns 0 and sets outs/n_outs on success; -1 and fills err[] on failure. */
int  onnx_session_run(onnx_session *s,
                      const onnx_value_in *ins, size_t n_ins,
                      const char *const *out_names, size_t n_out_names,
                      onnx_value_out **outs, size_t *n_outs,
                      char *err, size_t errlen);
void onnx_run_free(onnx_value_out *outs, size_t n_outs);

/* --- Native tokenizers (onnxruntime-extensions) -----------------------------
 * WordPiece via extensions' BertTokenizer C++ class; SentencePiece/BPE via the
 * Ortx C API (OrtxCreateTokenizer over a model dir holding tokenizer.json +
 * tokenizer_config.json). Both encode to a malloc'd int64 array of CONTENT ids
 * (no [CLS]/[SEP] or <s>/</s>) -- the JS embed layer adds special tokens, which
 * matches the old encodeIds() contract. Built only with RAMPART_ONNX_EXT; the
 * create fns return NULL + fill err[] ("built without extensions") otherwise.
 * *ids from *_encode is caller-freed with free(). */
typedef struct onnx_wp_tokenizer onnx_wp_tokenizer;
typedef struct onnx_sp_tokenizer onnx_sp_tokenizer;

onnx_wp_tokenizer *onnx_wp_create(const char *vocab_path, int lowercase,
                                  int strip_accents, int tokenize_chinese,
                                  char *err, size_t errlen);
int    onnx_wp_encode(onnx_wp_tokenizer *t, const char *text,
                      int64_t **ids, size_t *n_ids);
size_t onnx_wp_vocab_size(onnx_wp_tokenizer *t);
void   onnx_wp_destroy(onnx_wp_tokenizer *t);

onnx_sp_tokenizer *onnx_sp_create(const char *model_dir, char *err, size_t errlen);
int    onnx_sp_encode(onnx_sp_tokenizer *t, const char *text,
                      int64_t **ids, size_t *n_ids);
void   onnx_sp_destroy(onnx_sp_tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_SHIM_H */
