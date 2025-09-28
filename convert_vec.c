#include <stdint.h>

/* ====================================================================
           BEGIN fp16 <--> fp32 optimized conversions
   ==================================================================== */

char *fpConverterBackEnd="scalar";

/* ---------------------------------------------------------
   Scalar fallback: works everywhere
--------------------------------------------------------- */
static inline float fp16_to_fp32_scalar(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15);
    uint32_t e = (uint32_t)((h >> 10) & 0x1F);
    uint32_t f = (uint32_t)(h & 0x3FF);
    uint32_t out_e, out_f;

    if (e == 0)
    {
        if (f == 0)
        {
            out_e = 0;
            out_f = 0;
        }
        else
        {
            /* subnormal */
            while ((f & 0x400) == 0)
            {
                f <<= 1;
                e--;
            }
            f &= 0x3FFu;
            out_e = (127 - 15 + 1);
            out_f = f << 13;
        }
    }
    else if (e == 31)
    {
        out_e = 255;
        out_f = (f ? 0x7FFFFF : 0);
    }
    else
    {
        out_e = e - 15 + 127;
        out_f = f << 13;
    }
    uint32_t u = (s << 31) | (out_e << 23) | out_f;
    float out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

float *vec_fp16_to_fp32_scalar_buf(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        dst[i] = fp16_to_fp32_scalar(src[i]);
    return dst;
}

float *vec_fp16_to_fp32_scalar(const uint16_t *src, size_t n)
{
    float *dst = (float *)malloc(n * sizeof(float));
    if (!dst)
        return NULL;
    return vec_fp16_to_fp32_scalar_buf(src, dst, n);
}

static inline uint16_t fp32_to_fp16_scalar(float x)
{
    union { float f; uint32_t u; } in = { x };
    uint32_t f = in.u;

    uint32_t sign = (f >> 16) & 0x8000u;      /* sign bit */
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;

    if (exp <= 0) {
        /* Subnormal or underflow */
        if (exp < -10) {
            /* Too small, becomes signed zero */
            return (uint16_t)sign;
        }
        /* Subnormal: add implicit 1, shift to fit mantissa */
        mant = (mant | 0x800000u) >> (1 - exp);
        /* Round */
        if (mant & 0x1000u) mant += 0x2000u;
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        /* Overflow or Inf/NaN */
        if ((f & 0x7FFFFFu) != 0) {
            /* NaN: propagate top mantissa bits */
            return (uint16_t)(sign | 0x7C00u | (mant >> 13));
        } else {
            /* Inf */
            return (uint16_t)(sign | 0x7C00u);
        }
    } else {
        /* Normalized number */
        /* Round mantissa */
        if (mant & 0x1000u) {
            mant += 0x2000u;
            if (mant & 0x800000u) {  /* mantissa overflow */
                mant = 0;
                exp += 1;
                if (exp >= 31) {
                    /* exponent overflow to Inf */
                    return (uint16_t)(sign | 0x7C00u);
                }
            }
        }
        return (uint16_t)(sign | (exp << 10) | (mant >> 13));
    }
}

static void vec_fp32_to_fp16_scalar_buf(const float *src, uint16_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        dst[i] = fp32_to_fp16_scalar(src[i]);
}

uint16_t *vec_fp32_to_fp16_scalar(const float *src, size_t n)
{
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst)
        return NULL;
    vec_fp32_to_fp16_scalar_buf(src, dst, n);
    return dst;
}


/* =========================================================
   macOS: use Accelerate
========================================================= */
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
float *vec_fp16_to_fp32_buf(const uint16_t *src, float *dst, size_t n)
{
    vImage_Buffer srcBuf = {
        .data     = (void *)src,
        .height   = 1,
        .width    = n,
        .rowBytes = n * sizeof(uint16_t)
    };
    vImage_Buffer dstBuf = {
        .data     = dst,
        .height   = 1,
        .width    = n,
        .rowBytes = n * sizeof(float)
    };

    /* Convert IEEE-754 half (Planar16F) -> float (PlanarF) */
    vImage_Error err = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, kvImageNoFlags);
    /* Optional: handle/propagate err; kvImageNoError == 0 */
    (void)err;

    fpConverterBackEnd="accelerate";

    return dst;
}

#elif defined(__aarch64__) || defined(__ARM_NEON)
/* =========================================================
   ARM Linux / FreeBSD: assume NEON
========================================================= */
#include <arm_neon.h>

float *vec_fp16_to_fp32_buf(const uint16_t *src, float *dst, size_t n)
{
    float *dst = (float *)malloc(n * sizeof(float));
    if (!dst)
        return NULL;

    size_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    fpConverterBackEnd="neon";
    for (; i + 4 <= n; i += 4)
    {
        uint16x4_t u16 = vld1_u16(src + i);
        float16x4_t h4 = vreinterpret_f16_u16(u16);
        float32x4_t f32 = vcvt_f32_f16(h4);
        vst1q_f32(dst + i, f32);
    }
#endif
    for (; i < n; ++i)
        dst[i] = fp16_to_fp32_scalar(src[i]);
    return dst;
}

#else
/* =========================================================
   x86/x86_64: runtime dispatch among SSE2 / AVX / AVX2
========================================================= */
#include <cpuid.h>
#include <immintrin.h>

static int cpu_has_avx_f16c(void) {
    uint32_t eax, ebx, ecx, edx;

    // CPUID.(EAX=1):ECX bits
    // 27 = OSXSAVE, 28 = AVX, 29 = F16C
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(1), "c"(0));

    int has_osxsave = (ecx >> 27) & 1;
    int has_avx     = (ecx >> 28) & 1;
    int has_f16c    = (ecx >> 29) & 1;
    if (!(has_osxsave && has_avx && has_f16c)) return 0;

    // Verify OS enabled XMM/YMM via XGETBV(EAX=0), need XCR0[1:0] == SSE+AVX set
    uint32_t xcr0_lo, xcr0_hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0"   // xgetbv
                     : "=a"(xcr0_lo), "=d"(xcr0_hi)
                     : "c"(0));
    uint64_t xcr0 = ((uint64_t)xcr0_hi << 32) | xcr0_lo;
    int os_sse = (xcr0 & 0x2) != 0;
    int os_avx = (xcr0 & 0x4) != 0;

    return os_sse && os_avx;
}


// typedef float *(*fp16_convert_fn)(const uint16_t *, size_t);
typedef float *(*fp16_convert_fn_buf)(const uint16_t *, float *dst, size_t);


/* --- AVX2 + F16C path --- */
__attribute__((target("f16c,avx")))
static float *fp16_to_fp32_avx2_buf(const uint16_t *src, float *dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i)
        dst[i] = fp16_to_fp32_scalar(src[i]);

    return dst;
}


/* --- runtime selector --- */
static fp16_convert_fn_buf best_converter_buf = vec_fp16_to_fp32_scalar_buf;

static void init_fp16_dispatch(void)
{
    if(cpu_has_avx_f16c())
    {
        fpConverterBackEnd="avx2";
        best_converter_buf = fp16_to_fp32_avx2_buf;
        return;
    }
    best_converter_buf = vec_fp16_to_fp32_scalar_buf;
}

static int converter_inited = 0;

float *vec_fp16_to_fp32_buf(const uint16_t *src, float *dst, size_t n)
{
    static int inited = 0;
    if (!inited)
    {
        init_fp16_dispatch();
        converter_inited = 1;
    }
    return best_converter_buf(src, dst, n);
}


#endif /* platform */

float *vec_fp16_to_fp32(const uint16_t *src, size_t n)
{
    float *dst = (float *)malloc(n * sizeof(float));
    if (!dst)
        return NULL;
    return vec_fp16_to_fp32_buf(src, dst, n);
}


/* ====================================================================
   FP32 -> FP16 (uint16_t bits) — runtime-dispatched implementations
   ==================================================================== */

/* ---------- Per-ISA implementations (static) ---------- */

#if (defined(__aarch64__) || defined(__ARM_NEON)) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void vec_fp32_to_fp16_buf(const float *src, uint16_t *dst, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t f32 = vld1q_f32(src + i);
        float16x4_t h4 = vcvt_f16_f32(f32); /* RN-even */
        uint16x4_t u16 = vreinterpret_u16_f16(h4);
        vst1_u16(dst + i, u16);
    }
    for (; i < n; ++i)
        dst[i] = fp32_to_fp16_scalar(src[i]);
}

#elif defined(__i386__) || defined(__x86_64__)

// ---- F16C + AVX fast path (8 floats -> 8 half) ----
__attribute__((target("f16c,avx")))
static void fp32_to_fp16_f16c_avx(const float *src, uint16_t *dst, size_t n) {
    size_t i = 0;
    const size_t k = 8;
    for (; i + k <= n; i += k) {
        __m256  f  = _mm256_loadu_ps(src + i);
        __m128i h8 = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(dst + i), h8); // stores 8 x uint16_t
    }
    // tail
    for (; i < n; ++i) dst[i] = fp32_to_fp16_scalar(src[i]);
}

typedef void (*fp32_to_fp16_fn)(const float*, uint16_t*, size_t);


typedef void (*fp32_to_fp16_fn)(const float*, uint16_t*, size_t);

static fp32_to_fp16_fn resolve_impl(void) {
    if (cpu_has_avx_f16c())
    {
        fpConverterBackEnd="avx";
        return fp32_to_fp16_f16c_avx;
    }
    return vec_fp32_to_fp16_scalar_buf;
}


uint16_t *vec_fp32_to_fp16_buf(const float *src, uint16_t *dst, size_t n) {
    static fp32_to_fp16_fn impl = NULL;
    // thread-safe enough for most use; benign race resolves to same value
    if (!impl) impl = resolve_impl();
    impl(src, dst, n);
    return dst;
}
#endif


uint16_t *vec_fp32_to_fp16(const float *src, size_t n)
{
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst)
        return NULL;
    vec_fp32_to_fp16_buf(src, dst, n);
    return dst;
}

/* ====================================================================
           END fp16 <--> fp32 optimized conversions
   ==================================================================== */
