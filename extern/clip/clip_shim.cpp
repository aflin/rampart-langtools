// clip_shim.cpp -- monatis/clip.cpp ported onto the shared modern ggml, wrapped
// in a refcounted path-keyed cache with per-thread compute contexts (the
// rampart-onnx embed-handle lifecycle model).  The transformer graph math is
// verbatim from the original; the scaffolding is modern ggml: no_alloc meta
// context + ggml_gallocr, inputs set after allocation via ggml_backend_tensor_set.
//
// Derived from clip.cpp -- https://github.com/monatis/clip.cpp
//   Copyright (c) 2023 Yusuf Sarigoz -- MIT License.
// Bundled here: stb_image.h (Sean Barrett, MIT OR public domain; its own
// license text is at the bottom of that file).  ggml is NOT bundled -- this
// links the shared copy under extern/llama.cpp/ggml.
// See ./LICENSE in this directory for the full notices.

#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "clip_shim.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

/* GPU (CUDA) builds only: the arch/driver guard needs the CUDA runtime API. */
#if defined(LT_ENABLE_GPU) && !defined(__APPLE__)
  #define CLIP_HAVE_CUDA 1
  #include <cuda_runtime.h>
  #ifndef LT_CUDA_SM_LIST
  #define LT_CUDA_SM_LIST ""
  #endif
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* ---- ggml log capture: route ggml's informational output (the CUDA-init banner,
 * etc.) OFF stderr into a buffer, exposed as clip.getLog().  clip has its OWN ggml
 * (localized symbols), so this ggml_log_set touches only clip's ggml -- never
 * rampart-llamacpp's.  Installed once, before the first backend init. */
static std::mutex g_log_mtx;
static std::string g_log;
static void clip_ggml_logger(ggml_log_level level, const char * text, void * ud) {
    (void) level; (void) ud;
    if (!text) return;
    std::lock_guard<std::mutex> lk(g_log_mtx);
    g_log += text;
    if (g_log.size() > (1u << 20))                     /* cap ~1MB; keep the tail */
        g_log.erase(0, g_log.size() - (1u << 19));
}
char *clip_log_dup(void) {
    std::lock_guard<std::mutex> lk(g_log_mtx);
    char *s = (char *) malloc(g_log.size() + 1);
    if (s) memcpy(s, g_log.c_str(), g_log.size() + 1);
    return s;
}
void clip_log_clear(void) { std::lock_guard<std::mutex> lk(g_log_mtx); g_log.clear(); }
/* Process-once init: route ggml's log into our buffer, and (macOS) defuse the
 * Metal residency-set assert.  Called before anything can create a ggml backend.
 *
 * macOS 15+ ggml-metal "residency sets" GGML_ASSERT([rsets->data count] == 0) in
 * their static destructor at process exit if any Metal buffer is still alive --
 * which is ALWAYS true here, because a loaded model's weights live in the
 * process-lifetime refcounted cache and are deliberately never freed.  Disable
 * the feature (a marginal perf optimization); RAMPART_METAL_RESIDENCY=1 keeps it.
 * rampart-llamacpp does exactly the same for the same reason -- and clip needs its
 * own copy because it links its OWN ggml (symbols are localized), so it creates
 * its own Metal device and cannot rely on llamacpp having been loaded first. */
static void clip_once_init(void) {
    static std::once_flag once;
    std::call_once(once, []{
        ggml_log_set(clip_ggml_logger, nullptr);
#ifdef __APPLE__
        if (!getenv("RAMPART_METAL_RESIDENCY"))
            setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
#endif
    });
}

/* rampart thread id (weak: a non-rampart host that dlopens the .so still links;
 * there every call reports thread 0, which is correct for a single-threaded use). */
extern "C" { int get_thread_num(void) __attribute__((weak)); }
static int cur_thread_num(void) { return get_thread_num ? get_thread_num() : 0; }
#include <unistd.h>   /* getpid */

static std::string vformat(const char * fmt, va_list ap) {
    va_list ap2; va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    return std::string(buf.data(), size);
}
static std::string sformat(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); std::string s = vformat(fmt, ap); va_end(ap); return s;
}
static void set_err(char * err, size_t n, const char * fmt, ...) {
    if (!err || !n) return;
    va_list ap; va_start(ap, fmt);
    std::string s = vformat(fmt, ap); va_end(ap);
    snprintf(err, n, "%s", s.c_str());
}

// ---- key / tensor-name constants (from monatis clip.cpp) ----
#define KEY_HAS_TEXT_ENC "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC "clip.has_vision_encoder"
#define KEY_USE_GELU "clip.use_gelu"
#define KEY_N_EMBD "clip.%s.embedding_length"
#define KEY_N_FF "clip.%s.feed_forward_length"
#define KEY_N_BLOCK "clip.%s.block_count"
#define KEY_N_HEAD "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM "clip.%s.projection_dim"
#define KEY_TOKENS "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS "clip.text.context_length"
#define KEY_IMAGE_SIZE "clip.vision.image_size"
#define KEY_PATCH_SIZE "clip.vision.patch_size"
#define KEY_IMAGE_MEAN "clip.vision.image_mean"
#define KEY_IMAGE_STD "clip.vision.image_std"

#define TN_TOKEN_EMBD "%s.token_embd.weight"
#define TN_POS_EMBD "%s.position_embd.weight"
#define TN_CLASS_EMBD "v.class_embd"
#define TN_PATCH_EMBD "v.patch_embd.weight"
#define TN_ATTN_K "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP "%s.blk.%d.ffn_up.%s"
#define TN_LN_1 "%s.blk.%d.ln1.%s"
#define TN_LN_2 "%s.blk.%d.ln2.%s"
#define TN_LN_PRE "%s.pre_ln.%s"
#define TN_LN_POST "%s.post_ln.%s"
#define TN_TEXT_PROJ "text_projection.weight"
#define TN_VIS_PROJ "visual_projection.weight"

static int64_t key_idx(const gguf_context * g, const char * k) {
    int64_t i = gguf_find_key(g, k);
    if (i == -1) throw std::runtime_error(sformat("gguf key %s not found", k));
    return i;
}
static uint32_t g_u32(const gguf_context * g, const std::string & k) { return gguf_get_val_u32(g, key_idx(g, k.c_str())); }
static float    g_f32(const gguf_context * g, const std::string & k) { return gguf_get_val_f32(g, key_idx(g, k.c_str())); }
static ggml_tensor * g_tensor(ggml_context * c, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(c, name.c_str());
    if (!t) throw std::runtime_error(sformat("tensor %s missing", name.c_str()));
    return t;
}

// ---- model structs ----
struct clip_layer {
    ggml_tensor *k_w,*k_b,*q_w,*q_b,*v_w,*v_b,*o_w,*o_b;
    ggml_tensor *ln_1_w,*ln_1_b,*ff_i_w,*ff_i_b,*ff_o_w,*ff_o_b,*ln_2_w,*ln_2_b;
};
struct clip_text_model {
    int32_t n_vocab, num_positions, hidden_size, n_intermediate, projection_dim, n_head, n_layer;
    float eps;
    ggml_tensor *token_embeddings,*position_embeddings,*post_ln_w,*post_ln_b,*projection;
    std::vector<clip_layer> layers;
};
struct clip_vision_model {
    int32_t image_size, patch_size, hidden_size, n_intermediate, projection_dim, n_head, n_layer;
    float eps;
    ggml_tensor *class_embedding,*patch_embeddings,*position_embeddings;
    ggml_tensor *pre_ln_w,*pre_ln_b,*post_ln_w,*post_ln_b,*projection;
    std::vector<clip_layer> layers;
};

// per-thread compute context (backend + graph allocator); weights are shared
struct clip_thr { int thread_num; int pid; ggml_backend_t backend; ggml_gallocr_t galloc; };

struct clip_handle {
    std::string path;
    bool has_text=false, has_vision=false, use_gelu=false;
    clip_text_model   text;
    clip_vision_model vision;
    std::map<std::string,int32_t> token_to_id;
    float image_mean[3], image_std[3];

    ggml_context        * ctx_data=nullptr;   // tensor structs (read-only, shared)
    ggml_backend_buffer_t weights=nullptr;    // weight data (read-only, shared)

    int  use_gpu=0;                           // chosen backend type (decided at load)
    int  init_pid=0;                          // pid at load (fork guard for GPU handles)
    int refcount=0;
    std::mutex mtx;                           // guards thr[]
    std::vector<clip_thr> thr;                // per-thread compute contexts
    clip_handle * next=nullptr;               // cache chain
};

// ---- process-global path-keyed cache (mirrors rp_onnx_embed_cache_*) ----
static clip_handle * g_cache_head = nullptr;
static std::mutex    g_cache_lock;

#ifdef CLIP_HAVE_CUDA
/* This build ships native-arch SASS (baked LT_CUDA_SM_LIST) via ggml -- the SAME
 * ggml rampart-llamacpp uses, so the SAME guard applies: a device whose compute
 * capability has no matching SASS, or a driver older than the build's CUDA, would
 * abort mid-graph in ggml_cuda_error.  Return 1 if the current GPU can run this
 * build, else 0 (+ reason) so the caller falls back to CPU cleanly. */
static int clip_cuda_supported(char * reason, size_t n) {
    if (LT_CUDA_SM_LIST[0] == '\0') return 1;                 /* list not baked -> don't block */
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return 0;  /* no GPU -> caller uses CPU */
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;          /* can't tell -> allow */
    int cc = p.major * 10 + p.minor;
    int drv = 0;
    if (cudaDriverGetVersion(&drv) == cudaSuccess && drv > 0 && drv < CUDART_VERSION) {
        snprintf(reason, n,
            "GPU '%s' (sm_%d): the NVIDIA driver supports CUDA %d.%d, but this rampart-clip "
            "was built for CUDA %d.%d -- too old to load its GPU kernels (they would abort "
            "mid-graph). Upgrade the driver, or use the cuNN module matching your driver.",
            p.name, cc, drv/1000, (drv%1000)/10, CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
        return 0;
    }
    for (const char * s = LT_CUDA_SM_LIST; *s; ) {
        char * e; long v = strtol(s, &e, 10);
        if (e == s) { s++; continue; }
        if ((int) v == cc) return 1;                          /* native SASS for this device */
        s = e;
    }
    snprintf(reason, n,
        "GPU '%s' (compute %d.%d / sm_%d) has no compatible kernels in this rampart-clip "
        "build (built for sm: %s). Use the cuNN module matching your GPU, or rebuild adding sm_%d.",
        p.name, p.major, p.minor, cc, LT_CUDA_SM_LIST, cc);
    return 0;
}
#endif

/* Does this GPU device actually implement every op our graphs are built from?
 *
 * clip computes a graph on ONE backend -- there is no ggml_backend_sched here to
 * split a graph across devices -- so a single missing op is fatal: the backend calls
 * GGML_ABORT("unsupported op") and takes the whole process down mid-graph.  That is
 * not hypothetical: ggml's Metal backend gates MUL_MAT, NORM and SOFT_MAX behind
 * MTLGPUFamilyApple7 / MTLGPUFamilyMetal3 (i.e. Apple Silicon, or macOS 13+), so on an
 * Intel Mac running Big Sur *every matmul* is unsupported and the first embed aborts.
 *
 * So probe before committing: build a small graph from the same ops (and the model's
 * own weight type) the real vision/text graphs use, and ask the backend about every
 * node.  ggml_backend_supports_op only inspects tensor metadata, so a no_alloc context
 * with unallocated tensors is enough -- nothing is computed and nothing is uploaded.
 * Keep this in step with layer_forward() / clip_embed_text() / embed_pixels(). */
static bool clip_backend_runs_our_graphs(ggml_backend_t b, ggml_type wtype, char * missing, size_t nmiss) {
    const size_t bufsz = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ggml_init_params ip = { bufsz, NULL, /*no_alloc*/ true };
    ggml_context * c = ggml_init(ip);
    if (!c) return true;                         /* can't probe -> don't block the GPU */
    ggml_cgraph * gf = ggml_new_graph(c);

    /* a quantized row must be a whole number of blocks (32 for q4_0/q8_0/..., 256 for
     * the k-quants), so round the probe's hidden size up to the type's block size. */
    const int64_t blk = ggml_blck_size(wtype);
    const int D = (int) (blk * ((64 + blk - 1) / blk));
    const int HEADS = 4, DH = D / HEADS, N = 8, S = 32, P = 16, NP = (S/P) * (S/P);
    ggml_tensor * w    = ggml_new_tensor_2d(c, wtype, D, D);
    ggml_tensor * bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    ggml_tensor * ids  = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, N);
    ggml_tensor * img  = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, 1);
    ggml_tensor * kern = ggml_new_tensor_4d(c, GGML_TYPE_F16, P, P, 3, D);
    ggml_tensor * cls  = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, 1, 1);

    /* text-tower head, then one attention block (the shape of layer_forward) */
    ggml_tensor * t = ggml_add(c, ggml_get_rows(c, w, ids), ggml_get_rows(c, w, ids));
    t = ggml_norm(c, t, 1e-5f);
    t = ggml_add(c, ggml_mul(c, ggml_repeat(c, bias, t), t), ggml_repeat(c, bias, t));
    ggml_tensor * Q = ggml_scale(c, ggml_add(c, ggml_repeat(c, bias, t), ggml_mul_mat(c, w, t)), 1.0f/8.0f);
    Q = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, Q, DH, HEADS, N, 1), 0, 2, 1, 3));
    Q = ggml_reshape_3d(c, Q, DH, N, HEADS);
    ggml_tensor * kq = ggml_mul_mat(c, Q, Q);
    ggml_build_forward_expand(gf, ggml_soft_max_ext(c, kq, mask, 1.0f, 0.0f));   /* text: causal */
    ggml_build_forward_expand(gf, ggml_soft_max_inplace(c, ggml_mul_mat(c, Q, Q)));  /* vision */
    ggml_build_forward_expand(gf, ggml_gelu_inplace(c, ggml_mul_mat(c, w, t)));
    ggml_build_forward_expand(gf, ggml_gelu_quick_inplace(c, ggml_mul_mat(c, w, t)));

    /* vision-tower head: conv_2d patch embedding (expands to IM2COL + MUL_MAT) + concat */
    ggml_tensor * v = ggml_conv_2d(c, kern, img, P, P, 0, 0, 1, 1);
    v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, NP, D, 1), 1, 0, 2, 3));
    ggml_build_forward_expand(gf, ggml_concat(c, ggml_repeat(c, bias, cls), v, 1));

    bool ok = true;
    for (int i = 0, n = ggml_graph_n_nodes(gf); i < n && ok; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(b, node)) {
            snprintf(missing, nmiss, "%s", ggml_op_desc(node));
            ok = false;
        }
    }
    ggml_free(c);
    return ok;
}

// GPU-ready backend pick: prefer a GPU device, fall back to CPU.  In a CPU-only
// build no GPU device is registered, so this yields CPU automatically; a cu* build
// selects the GPU -- unless the arch/driver guard or the op probe rules it out
// (-> CPU + warn once).  wtype is the model's matmul weight type, so the probe asks
// about the quantization this model will really run.
// Sets *is_gpu to 1 iff a GPU backend was chosen.
static ggml_backend_t clip_backend_init(int * is_gpu, ggml_type wtype) {
    clip_once_init();          /* must precede any ggml device creation */
    if (is_gpu) *is_gpu = 0;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) {
        int ok = 1;
#ifdef CLIP_HAVE_CUDA
        char reason[512] = {0};
        ok = clip_cuda_supported(reason, sizeof reason);
        if (!ok && reason[0]) {
            static int warned = 0;
            if (!warned) { warned = 1; clip_warn("rampart-clip: %s  Using CPU.\n", reason); }
        }
#endif
        if (ok) {
            ggml_backend_t b = ggml_backend_dev_init(dev, NULL);
            if (b) {
                char missing[64] = {0};
                if (clip_backend_runs_our_graphs(b, wtype, missing, sizeof missing)) {
                    if (is_gpu) *is_gpu = 1;
                    return b;
                }
                static int warned = 0;
                if (!warned) {
                    warned = 1;
                    clip_warn("rampart-clip: GPU '%s' does not implement '%s', which CLIP's graphs "
                        "need, so it cannot run this model (on macOS this means a GPU/OS older than "
                        "MTLGPUFamilyApple7 or Metal 3 -- e.g. an Intel Mac, or macOS before 13). "
                        "Using CPU.\n", ggml_backend_dev_description(dev), missing);
                }
                ggml_backend_free(b);
            }
        }
    }
    return ggml_backend_cpu_init();
}

/* a compute backend of the load-chosen type (GPU weights need GPU compute). */
static ggml_backend_t clip_backend_of_type(int use_gpu) {
    clip_once_init();          /* must precede any ggml device creation */
    if (use_gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) return ggml_backend_dev_init(dev, NULL);
    }
    return ggml_backend_cpu_init();
}

#define CLIP_FORK_REFUSAL "rampart-clip: this model was loaded before a fork() and uses " \
    "the GPU -- using it in the child would crash the CUDA runtime. Load the model after " \
    "forking (rampart-server postForkFunc), or run it on CPU."

// find or lazily create the calling thread's compute context (of the load-chosen
// backend type).  Refuses in a forked child if the handle uses the GPU.
static clip_thr * get_thr(clip_handle * h, char * err, size_t errlen) {
    int thrno = cur_thread_num();
    int pid = (int) getpid();
    if (h->use_gpu && pid != h->init_pid) { set_err(err, errlen, "%s", CLIP_FORK_REFUSAL); return nullptr; }
    std::lock_guard<std::mutex> lk(h->mtx);
    for (auto & t : h->thr)
        if (t.thread_num == thrno && t.pid == pid) return &t;
    clip_thr t; t.thread_num = thrno; t.pid = pid;
    t.backend = clip_backend_of_type(h->use_gpu);
    if (!t.backend) { set_err(err, errlen, "failed to init compute backend"); return nullptr; }
    t.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t.backend));
    if (!t.galloc) { ggml_backend_free(t.backend); set_err(err, errlen, "failed to init graph allocator"); return nullptr; }
    h->thr.push_back(t);
    return &h->thr.back();
}

static void l2norm(float * v, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += v[i]*v[i];
    float inv = 1.0f / sqrtf(s); for (int i = 0; i < n; i++) v[i] *= inv;
}

// ================= load =================
static clip_handle * clip_build(const char * path, char * err, size_t errlen) {
    clip_handle * h = nullptr;
    ggml_context * meta = nullptr;
    gguf_context * g = nullptr;
    ggml_backend_t load_backend = nullptr;
    try {
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
        g = gguf_init_from_file(path, gp);
        if (!g) throw std::runtime_error("gguf_init_from_file failed (not a CLIP gguf?)");

        h = new clip_handle;
        h->path = path;
        h->init_pid = (int) getpid();
        clip_once_init();        /* log capture + macOS Metal residency guard */
        /* Allocate the weights on the SAME backend type the per-thread compute
         * uses: CPU in a CPU build, the GPU device in a cu* build (unless the
         * arch/driver guard rules the GPU out).  On GPU the weights land in VRAM,
         * where the per-thread CUDA compute contexts read them -- a CPU weight
         * buffer would be unreadable by a GPU graph.  h->use_gpu records the choice
         * so every per-thread context matches; the buffer outlives this handle.
         * The GPU is also probed for the ops these graphs need, using this model's own
         * matmul weight type -- the biggest 2D tensor in the file is always one (an
         * ffn/projection/embedding matrix), and they all share a quantization. */
        {
            ggml_type wtype = GGML_TYPE_F16;
            size_t    best  = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t))
                if (ggml_n_dims(t) == 2 && ggml_nbytes(t) > best) { best = ggml_nbytes(t); wtype = t->type; }
            load_backend = clip_backend_init(&h->use_gpu, wtype);
        }

        h->has_text   = gguf_get_val_bool(g, key_idx(g, KEY_HAS_TEXT_ENC));
        h->has_vision = gguf_get_val_bool(g, key_idx(g, KEY_HAS_VIS_ENC));
        h->use_gelu   = gguf_get_val_bool(g, key_idx(g, KEY_USE_GELU));

        const int64_t n_tensors = gguf_get_n_tensors(g);
        ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 1), NULL, /*no_alloc*/ true };
        h->ctx_data = ggml_init(ip);
        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * cur = ggml_dup_tensor(h->ctx_data, ggml_get_tensor(meta, name));
            ggml_set_name(cur, name);
        }
        h->weights = ggml_backend_alloc_ctx_tensors(h->ctx_data, load_backend);
        if (!h->weights) throw std::runtime_error("weight buffer allocation failed");
        {
            std::ifstream fin(path, std::ios::binary);
            if (!fin) throw std::runtime_error("cannot open model file for tensor data");
            const size_t data_off = gguf_get_data_offset(g);
            std::vector<char> buf;
            for (int64_t i = 0; i < n_tensors; i++) {
                const char * name = gguf_get_tensor_name(g, i);
                ggml_tensor * cur = ggml_get_tensor(h->ctx_data, name);
                const size_t nb = ggml_nbytes(cur);
                fin.seekg(data_off + gguf_get_tensor_offset(g, i), std::ios::beg);
                buf.resize(nb);
                fin.read(buf.data(), nb);
                if (!fin) throw std::runtime_error(sformat("read failed for tensor %s", name));
                ggml_backend_tensor_set(cur, buf.data(), 0, nb);
            }
        }

        if (h->has_text) {
            auto & m = h->text;
            m.hidden_size    = g_u32(g, sformat(KEY_N_EMBD, "text"));
            m.n_head         = g_u32(g, sformat(KEY_N_HEAD, "text"));
            m.n_intermediate = g_u32(g, sformat(KEY_N_FF, "text"));
            m.n_layer        = g_u32(g, sformat(KEY_N_BLOCK, "text"));
            m.num_positions  = g_u32(g, KEY_N_POSITIONS);
            m.projection_dim = g_u32(g, sformat(KEY_PROJ_DIM, "text"));
            m.eps            = g_f32(g, sformat(KEY_LAYER_NORM_EPS, "text"));
            const int64_t it = key_idx(g, KEY_TOKENS);
            m.n_vocab = (int32_t) gguf_get_arr_n(g, it);
            for (int id = 0; id < m.n_vocab; id++)
                h->token_to_id[gguf_get_arr_str(g, it, id)] = id;
            m.token_embeddings    = g_tensor(h->ctx_data, sformat(TN_TOKEN_EMBD, "t"));
            m.position_embeddings = g_tensor(h->ctx_data, sformat(TN_POS_EMBD, "t"));
            m.post_ln_w = g_tensor(h->ctx_data, sformat(TN_LN_POST, "t", "weight"));
            m.post_ln_b = g_tensor(h->ctx_data, sformat(TN_LN_POST, "t", "bias"));
            m.projection = g_tensor(h->ctx_data, TN_TEXT_PROJ);
            m.layers.resize(m.n_layer);
            for (int il = 0; il < m.n_layer; il++) {
                auto & L = m.layers[il];
                L.k_w=g_tensor(h->ctx_data,sformat(TN_ATTN_K,"t",il,"weight"));
                L.q_w=g_tensor(h->ctx_data,sformat(TN_ATTN_Q,"t",il,"weight"));
                L.v_w=g_tensor(h->ctx_data,sformat(TN_ATTN_V,"t",il,"weight"));
                L.o_w=g_tensor(h->ctx_data,sformat(TN_ATTN_OUTPUT,"t",il,"weight"));
                L.ln_1_w=g_tensor(h->ctx_data,sformat(TN_LN_1,"t",il,"weight"));
                L.ln_2_w=g_tensor(h->ctx_data,sformat(TN_LN_2,"t",il,"weight"));
                L.ff_i_w=g_tensor(h->ctx_data,sformat(TN_FFN_DOWN,"t",il,"weight"));
                L.ff_o_w=g_tensor(h->ctx_data,sformat(TN_FFN_UP,"t",il,"weight"));
                L.k_b=g_tensor(h->ctx_data,sformat(TN_ATTN_K,"t",il,"bias"));
                L.q_b=g_tensor(h->ctx_data,sformat(TN_ATTN_Q,"t",il,"bias"));
                L.v_b=g_tensor(h->ctx_data,sformat(TN_ATTN_V,"t",il,"bias"));
                L.o_b=g_tensor(h->ctx_data,sformat(TN_ATTN_OUTPUT,"t",il,"bias"));
                L.ln_1_b=g_tensor(h->ctx_data,sformat(TN_LN_1,"t",il,"bias"));
                L.ln_2_b=g_tensor(h->ctx_data,sformat(TN_LN_2,"t",il,"bias"));
                L.ff_i_b=g_tensor(h->ctx_data,sformat(TN_FFN_DOWN,"t",il,"bias"));
                L.ff_o_b=g_tensor(h->ctx_data,sformat(TN_FFN_UP,"t",il,"bias"));
            }
        }
        if (h->has_vision) {
            auto & m = h->vision;
            m.hidden_size    = g_u32(g, sformat(KEY_N_EMBD, "vision"));
            m.n_head         = g_u32(g, sformat(KEY_N_HEAD, "vision"));
            m.n_intermediate = g_u32(g, sformat(KEY_N_FF, "vision"));
            m.n_layer        = g_u32(g, sformat(KEY_N_BLOCK, "vision"));
            m.image_size     = g_u32(g, KEY_IMAGE_SIZE);
            m.patch_size     = g_u32(g, KEY_PATCH_SIZE);
            m.projection_dim = g_u32(g, sformat(KEY_PROJ_DIM, "vision"));
            m.eps            = g_f32(g, sformat(KEY_LAYER_NORM_EPS, "vision"));
            const int64_t im = key_idx(g, KEY_IMAGE_MEAN), is = key_idx(g, KEY_IMAGE_STD);
            for (int i = 0; i < 3; i++) {
                h->image_mean[i] = ((const float *) gguf_get_arr_data(g, im))[i];
                h->image_std[i]  = ((const float *) gguf_get_arr_data(g, is))[i];
            }
            m.patch_embeddings    = g_tensor(h->ctx_data, TN_PATCH_EMBD);
            m.class_embedding     = g_tensor(h->ctx_data, TN_CLASS_EMBD);
            m.position_embeddings = g_tensor(h->ctx_data, sformat(TN_POS_EMBD, "v"));
            m.pre_ln_w  = g_tensor(h->ctx_data, sformat(TN_LN_PRE, "v", "weight"));
            m.pre_ln_b  = g_tensor(h->ctx_data, sformat(TN_LN_PRE, "v", "bias"));
            m.post_ln_w = g_tensor(h->ctx_data, sformat(TN_LN_POST, "v", "weight"));
            m.post_ln_b = g_tensor(h->ctx_data, sformat(TN_LN_POST, "v", "bias"));
            m.projection = g_tensor(h->ctx_data, TN_VIS_PROJ);
            m.layers.resize(m.n_layer);
            for (int il = 0; il < m.n_layer; il++) {
                auto & L = m.layers[il];
                L.k_w=g_tensor(h->ctx_data,sformat(TN_ATTN_K,"v",il,"weight"));
                L.q_w=g_tensor(h->ctx_data,sformat(TN_ATTN_Q,"v",il,"weight"));
                L.v_w=g_tensor(h->ctx_data,sformat(TN_ATTN_V,"v",il,"weight"));
                L.o_w=g_tensor(h->ctx_data,sformat(TN_ATTN_OUTPUT,"v",il,"weight"));
                L.ln_1_w=g_tensor(h->ctx_data,sformat(TN_LN_1,"v",il,"weight"));
                L.ln_2_w=g_tensor(h->ctx_data,sformat(TN_LN_2,"v",il,"weight"));
                L.ff_i_w=g_tensor(h->ctx_data,sformat(TN_FFN_DOWN,"v",il,"weight"));
                L.ff_o_w=g_tensor(h->ctx_data,sformat(TN_FFN_UP,"v",il,"weight"));
                L.k_b=g_tensor(h->ctx_data,sformat(TN_ATTN_K,"v",il,"bias"));
                L.q_b=g_tensor(h->ctx_data,sformat(TN_ATTN_Q,"v",il,"bias"));
                L.v_b=g_tensor(h->ctx_data,sformat(TN_ATTN_V,"v",il,"bias"));
                L.o_b=g_tensor(h->ctx_data,sformat(TN_ATTN_OUTPUT,"v",il,"bias"));
                L.ln_1_b=g_tensor(h->ctx_data,sformat(TN_LN_1,"v",il,"bias"));
                L.ln_2_b=g_tensor(h->ctx_data,sformat(TN_LN_2,"v",il,"bias"));
                L.ff_i_b=g_tensor(h->ctx_data,sformat(TN_FFN_DOWN,"v",il,"bias"));
                L.ff_o_b=g_tensor(h->ctx_data,sformat(TN_FFN_UP,"v",il,"bias"));
            }
        }
        ggml_backend_free(load_backend);   // weight buffer is independent host memory now
        gguf_free(g);
        ggml_free(meta);
        return h;
    } catch (const std::exception & e) {
        set_err(err, errlen, "clip_load: %s", e.what());
        if (load_backend) ggml_backend_free(load_backend);
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
        if (h) {
            if (h->weights) ggml_backend_buffer_free(h->weights);
            if (h->ctx_data) ggml_free(h->ctx_data);
            delete h;
        }
        return nullptr;
    }
}

clip_handle * clip_load(const char * path, char * err, size_t errlen) {
    {   // cache hit?
        std::lock_guard<std::mutex> lk(g_cache_lock);
        for (clip_handle * c = g_cache_head; c; c = c->next)
            if (c->path == path) { c->refcount++; return c; }
    }
    clip_handle * h = clip_build(path, err, errlen);
    if (!h) return nullptr;
    std::lock_guard<std::mutex> lk(g_cache_lock);
    for (clip_handle * c = g_cache_head; c; c = c->next)   // lost a load race -> share winner
        if (c->path == path) {
            c->refcount++;
            if (h->weights) ggml_backend_buffer_free(h->weights);
            if (h->ctx_data) ggml_free(h->ctx_data);
            delete h;
            return c;
        }
    h->refcount = 1;
    h->next = g_cache_head;
    g_cache_head = h;
    return h;
}

void clip_acquire(clip_handle * h) { if (h) { std::lock_guard<std::mutex> lk(g_cache_lock); h->refcount++; } }
void clip_release(clip_handle * h) { if (h) { std::lock_guard<std::mutex> lk(g_cache_lock); if (h->refcount > 0) h->refcount--; } }
int clip_dim(clip_handle * h) {
    if (h->has_vision && h->vision.projection_dim > 0) return h->vision.projection_dim;
    if (h->has_text) return h->text.projection_dim;
    return 0;
}
int clip_has_text(clip_handle * h)   { return h->has_text ? 1 : 0; }
int clip_has_vision(clip_handle * h) { return h->has_vision ? 1 : 0; }
int clip_on_gpu(clip_handle * h)     { return h->use_gpu ? 1 : 0; }

// ================= transformer layer (verbatim math; scaffolding modernized) =================
/* kq_mask: NULL for the (bidirectional) vision tower; for the text tower it is an
 * [seq, seq] F32 additive causal mask (0 keep / -INF mask) applied inside
 * ggml_soft_max_ext.  This replaces ggml_diag_mask_inf, which the Metal backend
 * does NOT implement at all (it aborts with "unsupported op") -- soft_max_ext with
 * an explicit mask is the path llama.cpp itself uses for attention, so it is
 * supported on CPU, CUDA and Metal alike, and is mathematically identical:
 * diag_mask_inf(x,0) then soft_max == soft_max_ext(x, mask) with that mask. */
static ggml_tensor * layer_forward(ggml_context * c, const clip_layer & L, ggml_tensor * embeddings,
                                   int hidden_size, int n_head, int d_head, int seq, int batch,
                                   float eps, bool use_gelu, ggml_tensor * kq_mask) {
    ggml_tensor * cur = ggml_norm(c, embeddings, eps);
    cur = ggml_add(c, ggml_mul(c, ggml_repeat(c, L.ln_1_w, cur), cur), ggml_repeat(c, L.ln_1_b, cur));
    {
        ggml_tensor * Q = ggml_add(c, ggml_repeat(c, L.q_b, cur), ggml_mul_mat(c, L.q_w, cur));
        Q = ggml_scale(c, Q, 1.0f / sqrtf((float) d_head));
        Q = ggml_reshape_4d(c, Q, d_head, n_head, seq, batch);
        Q = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(c, Q, d_head, seq, n_head * batch);
        ggml_tensor * K = ggml_add(c, ggml_repeat(c, L.k_b, cur), ggml_mul_mat(c, L.k_w, cur));
        K = ggml_reshape_4d(c, K, d_head, n_head, seq, batch);
        K = ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(c, K, d_head, seq, n_head * batch);
        ggml_tensor * V = ggml_add(c, ggml_repeat(c, L.v_b, cur), ggml_mul_mat(c, L.v_w, cur));
        V = ggml_reshape_4d(c, V, d_head, n_head, seq, batch);
        V = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3));
        V = ggml_reshape_3d(c, V, seq, d_head, n_head * batch);
        ggml_tensor * KQ = ggml_mul_mat(c, K, Q);
        /* Q is already pre-scaled by 1/sqrt(d_head) above, hence scale = 1.0 here. */
        KQ = kq_mask ? ggml_soft_max_ext(c, KQ, kq_mask, 1.0f, 0.0f)
                     : ggml_soft_max_inplace(c, KQ);
        ggml_tensor * KQV = ggml_mul_mat(c, V, KQ);
        KQV = ggml_reshape_4d(c, KQV, d_head, seq, n_head, batch);
        KQV = ggml_cont(c, ggml_permute(c, KQV, 0, 2, 1, 3));
        cur = ggml_reshape_3d(c, KQV, hidden_size, seq, batch);
    }
    cur = ggml_add(c, ggml_repeat(c, L.o_b, cur), ggml_mul_mat(c, L.o_w, cur));
    cur = ggml_add(c, cur, embeddings);
    embeddings = cur;
    cur = ggml_norm(c, cur, eps);
    cur = ggml_add(c, ggml_mul(c, ggml_repeat(c, L.ln_2_w, cur), cur), ggml_repeat(c, L.ln_2_b, cur));
    cur = ggml_mul_mat(c, L.ff_i_w, cur);
    cur = ggml_add(c, ggml_repeat(c, L.ff_i_b, cur), cur);
    cur = use_gelu ? ggml_gelu_inplace(c, cur) : ggml_gelu_quick_inplace(c, cur);
    cur = ggml_mul_mat(c, L.ff_o_w, cur);
    cur = ggml_add(c, ggml_repeat(c, L.ff_o_b, cur), cur);
    cur = ggml_add(c, embeddings, cur);
    return cur;
}

// ================= text encode =================
int clip_embed_text(clip_handle * h, const char * text, float * out, int normalize, char * err, size_t errlen) {
    if (!h->has_text) { set_err(err, errlen, "model has no text encoder"); return -1; }
    clip_thr * T = get_thr(h, err, errlen);
    if (!T) return -1;   /* get_thr set err (fork refusal / backend / allocator) */
    const auto & m = h->text;

    // tokenize (CLIP BPE; greedy longest-match over the gguf vocab)
    std::vector<int32_t> v; v.push_back(49406);
    {
        std::vector<std::string> words;
        std::string str = text;
        std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)");
        std::smatch mt;
        while (std::regex_search(str, mt, re)) { for (auto x : mt) words.push_back(x); str = mt.suffix(); }
        for (const auto & word : words) {
            std::string full = (word.find(" ") == 0) ? word.substr(1) : word;
            full += "</w>";
            auto wit = h->token_to_id.find(full);
            if (wit != h->token_to_id.end()) { v.push_back(wit->second); continue; }
            for (int i = 0; i < (int) word.size();) {
                for (int j = (int) word.size() - 1; j >= i; j--) {
                    auto it = h->token_to_id.find(word.substr(i, j - i + 1));
                    if (it != h->token_to_id.end()) { v.push_back(it->second); i = j + 1; break; }
                    else if (j == i) { i++; }
                }
            }
        }
    }
    v.push_back(49407);
    const int N = (int) v.size();
    const int hidden_size = m.hidden_size, n_head = m.n_head, d_head = hidden_size / n_head;

    size_t bufsz = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ggml_init_params ip = { bufsz, NULL, true };
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(c);

    ggml_tensor * input_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_input(input_ids);
    ggml_tensor * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_input(positions);
    ggml_tensor * eot       = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(eot);
    /* additive causal mask, [key, query]: 0 where key <= query, -INF above */
    ggml_tensor * kq_mask   = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, N); ggml_set_input(kq_mask);

    ggml_tensor * emb = ggml_get_rows(c, m.token_embeddings, input_ids);
    emb = ggml_add(c, ggml_get_rows(c, m.position_embeddings, positions), emb);
    for (int il = 0; il < m.n_layer; il++)
        emb = layer_forward(c, m.layers[il], emb, hidden_size, n_head, d_head, N, 1, m.eps, h->use_gelu, kq_mask);
    emb = ggml_norm(c, emb, m.eps);
    emb = ggml_add(c, ggml_mul(c, ggml_repeat(c, m.post_ln_w, emb), emb), ggml_repeat(c, m.post_ln_b, emb));
    emb = ggml_get_rows(c, emb, eot);
    emb = ggml_mul_mat(c, m.projection, emb);
    ggml_set_output(emb);

    ggml_build_forward_expand(gf, emb);
    if (!ggml_gallocr_alloc_graph(T->galloc, gf)) { ggml_free(c); set_err(err, errlen, "graph alloc failed"); return -1; }

    std::vector<int32_t> pos(N); for (int i = 0; i < N; i++) pos[i] = i;
    int32_t e = N - 1;
    /* ne[0] is the key index (fastest), ne[1] the query index -- mask a key that
     * lies AFTER the query, matching ggml_diag_mask_inf(x, n_past=0). */
    std::vector<float> maskbuf((size_t) N * N);
    for (int q = 0; q < N; q++)
        for (int k = 0; k < N; k++)
            maskbuf[(size_t) q * N + k] = (k > q) ? -INFINITY : 0.0f;
    ggml_backend_tensor_set(input_ids, v.data(),   0, N * sizeof(int32_t));
    ggml_backend_tensor_set(positions, pos.data(), 0, N * sizeof(int32_t));
    ggml_backend_tensor_set(eot, &e, 0, sizeof(int32_t));
    ggml_backend_tensor_set(kq_mask, maskbuf.data(), 0, maskbuf.size() * sizeof(float));

    if (ggml_backend_graph_compute(T->backend, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); set_err(err, errlen, "compute failed"); return -1; }
    ggml_backend_tensor_get(emb, out, 0, m.projection_dim * sizeof(float));
    if (normalize) l2norm(out, m.projection_dim);
    ggml_free(c);
    return 0;
}

// ================= image preprocess (bicubic, verbatim math) =================
static inline double bicubic_filter(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}
static bool precompute_coeffs(int inSize, int outSize, double ** kkp, int ** boundsp, int * ksize) {
    double support = 2.0, filterscale = (double) inSize / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    support *= filterscale;
    int ks = (int) ceil(support) * 2 + 1;
    double * kk = (double *) malloc((size_t) outSize * ks * sizeof(double));
    int * bounds = (int *) malloc((size_t) outSize * 2 * sizeof(int));
    if (!kk || !bounds) { free(kk); free(bounds); return false; }
    for (int xx = 0; xx < outSize; xx++) {
        double center = (xx + 0.5) * (double) inSize / outSize, ww = 0.0, ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize; xmax -= xmin;
        double * k = &kk[xx * ks];
        for (int x = 0; x < xmax; x++) { double w = bicubic_filter((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (int x = 0; x < xmax; x++) if (ww != 0.0) k[x] /= ww;
        for (int x = xmax; x < ks; x++) k[x] = 0.0;
        bounds[xx*2] = xmin; bounds[xx*2+1] = xmax;
    }
    *kkp = kk; *boundsp = bounds; *ksize = ks; return true;
}
// returns malloc'd float[3*S*S] normalized planar-source (NHWC) or NULL
static float * preprocess(clip_handle * h, const uint8_t * img, int nx, int ny, int S) {
    const int nx3i = (int)(nx / (std::min((float)nx,(float)ny)/(float)S) + 0.5f);
    const int ny3i = (int)(ny / (std::min((float)nx,(float)ny)/(float)S) + 0.5f);
    double *kh,*kv; int *bh,*bv,ksh,ksv;
    if (!precompute_coeffs(nx, nx3i, &kh, &bh, &ksh) || !precompute_coeffs(ny, ny3i, &kv, &bv, &ksv)) return NULL;
    float * temp = new float[(size_t)3*nx3i*ny]();
    for (int y = 0; y < ny; y++) for (int xx = 0; xx < nx3i; xx++) {
        int xmin=bh[xx*2],xmax=bh[xx*2+1]; double*k=&kh[xx*ksh];
        for (int cc=0;cc<3;cc++){ double ss=0; for(int x=0;x<xmax;x++) ss+=(double)img[3*(y*nx+(x+xmin))+cc]*k[x];
            temp[3*(y*nx3i+xx)+cc]=std::min(std::max((float)ss,0.0f),255.0f); }
    }
    float * rs = new float[(size_t)3*nx3i*ny3i]();
    for (int yy=0;yy<ny3i;yy++){ int ymin=bv[yy*2],ymax=bv[yy*2+1]; double*k=&kv[yy*ksv];
        for(int x=0;x<nx3i;x++) for(int cc=0;cc<3;cc++){ double ss=0; for(int y=0;y<ymax;y++) ss+=(double)temp[3*((y+ymin)*nx3i+x)+cc]*k[y];
            rs[3*(yy*nx3i+x)+cc]=std::min(std::max((float)ss,0.0f),255.0f); }
    }
    float * out = (float *) malloc((size_t)3*S*S*sizeof(float));
    int xo=(nx3i-S)/2, yo=(ny3i-S)/2;
    for (int yy=0;yy<S;yy++) for(int x=0;x<S;x++){ int src=3*((yy+yo)*nx3i+(x+xo)),dst=3*(yy*S+x);
        for(int cc=0;cc<3;cc++) out[dst+cc]=((rs[src+cc]/255.0f)-h->image_mean[cc])/h->image_std[cc]; }
    delete[] rs; delete[] temp; free(kh); free(bh); free(kv); free(bv);
    return out;
}

// ================= image encode =================
// The two public entry points (file, memory buffer) differ ONLY in how they
// obtain the decoded RGB pixels; everything after -- preprocess, graph, encode,
// L2-normalize -- is this one shared function, so a disk image and a column blob
// of the same picture always embed identically.
static int embed_pixels(clip_handle * h, const uint8_t * raw, int nx, int ny,
                        float * out, int normalize, char * err, size_t errlen) {
    clip_thr * T = get_thr(h, err, errlen);
    if (!T) return -1;   /* get_thr set err (fork refusal / backend / allocator) */
    const auto & m = h->vision;
    const int S = m.image_size, patch_size = m.patch_size;
    const int num_patches = (S / patch_size) * (S / patch_size), num_positions = num_patches + 1;
    const int hidden_size = m.hidden_size, n_head = m.n_head, d_head = hidden_size / n_head, batch = 1;

    float * px = preprocess(h, raw, nx, ny, S);
    if (!px) { set_err(err, errlen, "image preprocess failed"); return -1; }

    size_t bufsz = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ggml_init_params ip = { bufsz, NULL, true };
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(c);

    ggml_tensor * inp_raw   = ggml_new_tensor_4d(c, GGML_TYPE_F32, S, S, 3, batch); ggml_set_input(inp_raw);
    ggml_tensor * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, num_positions);  ggml_set_input(positions);
    ggml_tensor * cls       = ggml_new_tensor_1d(c, GGML_TYPE_I32, batch);          ggml_set_input(cls);

    ggml_tensor * inp = ggml_conv_2d(c, m.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(c, inp, num_patches, hidden_size, batch);
    inp = ggml_cont(c, ggml_permute(c, inp, 1, 0, 2, 3));
    ggml_tensor * class_tmpl = ggml_new_tensor_3d(c, GGML_TYPE_F32, hidden_size, 1, batch);
    ggml_tensor * class_emb  = ggml_repeat(c, m.class_embedding, class_tmpl);
    ggml_tensor * emb = ggml_concat(c, class_emb, inp, 1);
    emb = ggml_add(c, emb, ggml_get_rows(c, m.position_embeddings, positions));
    emb = ggml_norm(c, emb, m.eps);
    emb = ggml_add(c, ggml_mul(c, ggml_repeat(c, m.pre_ln_w, emb), emb), ggml_repeat(c, m.pre_ln_b, emb));
    for (int il = 0; il < m.n_layer; il++)
        emb = layer_forward(c, m.layers[il], emb, hidden_size, n_head, d_head, num_positions, batch, m.eps, h->use_gelu, /*kq_mask*/ NULL);
    emb = ggml_reshape_2d(c, emb, hidden_size, num_positions * batch);
    emb = ggml_get_rows(c, emb, cls);
    emb = ggml_norm(c, emb, m.eps);
    emb = ggml_add(c, ggml_mul(c, ggml_repeat(c, m.post_ln_w, emb), emb), ggml_repeat(c, m.post_ln_b, emb));
    emb = ggml_mul_mat(c, m.projection, emb);
    ggml_set_output(emb);

    ggml_build_forward_expand(gf, emb);
    if (!ggml_gallocr_alloc_graph(T->galloc, gf)) { ggml_free(c); free(px); set_err(err, errlen, "graph alloc failed"); return -1; }

    {   // NHWC -> planar NCHW
        const int n = S * S;
        std::vector<float> data((size_t) 3 * n);
        for (int k = 0; k < 3; k++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++)
            data[k*n + y*S + x] = px[3*(y*S + x) + k];
        ggml_backend_tensor_set(inp_raw, data.data(), 0, data.size() * sizeof(float));
    }
    free(px);
    std::vector<int32_t> pos(num_positions); for (int i = 0; i < num_positions; i++) pos[i] = i;
    int32_t cls0 = 0;
    ggml_backend_tensor_set(positions, pos.data(), 0, num_positions * sizeof(int32_t));
    ggml_backend_tensor_set(cls, &cls0, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(T->backend, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); set_err(err, errlen, "compute failed"); return -1; }
    ggml_backend_tensor_get(emb, out, 0, m.projection_dim * sizeof(float));
    if (normalize) l2norm(out, m.projection_dim);
    ggml_free(c);
    return 0;
}

/* decode an image FILE, then embed its pixels.  Distinguish a missing/unreadable
 * file (stat) from a file that exists but isn't a decodable image (stbi_load NULL),
 * so the two failures don't share one vague message. */
int clip_embed_image_file(clip_handle * h, const char * path, float * out, int normalize, char * err, size_t errlen) {
    if (!h->has_vision) { set_err(err, errlen, "model has no vision encoder"); return -1; }
    struct stat st;
    if (stat(path, &st) != 0) { set_err(err, errlen, "cannot open image file '%s': %s", path, strerror(errno)); return -1; }
    int nx, ny, nc;
    uint8_t * raw = stbi_load(path, &nx, &ny, &nc, 3);
    if (!raw) { set_err(err, errlen, "failed to decode image '%s' (unrecognized format?)", path); return -1; }
    int rc = embed_pixels(h, raw, nx, ny, out, normalize, err, errlen);
    stbi_image_free(raw);
    return rc;
}

/* decode in-memory image BYTES (borrowed, const, not NUL-terminated; length is
 * authoritative), then embed.  stbi_load_from_memory copies into its own decode
 * buffer, so `buf` is never retained past this call. */
int clip_embed_image_mem(clip_handle * h, const void * buf, size_t len, float * out, int normalize, char * err, size_t errlen) {
    if (!h->has_vision) { set_err(err, errlen, "model has no vision encoder"); return -1; }
    if (!buf || len == 0) { set_err(err, errlen, "empty image buffer"); return -1; }
    /* stbi_load_from_memory takes an int length; a size_t > INT_MAX would wrap
     * negative and silently misparse -- reject it explicitly. */
    if (len > (size_t) INT_MAX) { set_err(err, errlen, "image buffer too large (%zu bytes; max %d)", len, INT_MAX); return -1; }
    int nx, ny, nc;
    uint8_t * raw = stbi_load_from_memory((const stbi_uc *) buf, (int) len, &nx, &ny, &nc, 3);
    if (!raw) { set_err(err, errlen, "failed to decode image buffer (%zu bytes; unrecognized format?)", len); return -1; }
    int rc = embed_pixels(h, raw, nx, ny, out, normalize, err, errlen);
    stbi_image_free(raw);
    return rc;
}

float clip_similarity(const float * a, const float * b, int dim) {
    float d = 0.0f; for (int i = 0; i < dim; i++) d += a[i]*b[i]; return d;
}
