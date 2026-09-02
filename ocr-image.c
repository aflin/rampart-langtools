/* ocr-image.c -- image decode / resize / detection preprocessing.
 *
 * stb_image is vendored here (the same public-domain header rampart-clip uses,
 * extern/clip/stb_image.h).  Its symbols are kept out of the global namespace by
 * rampart-ocr.map -- clip publishes its own copy of the same header, and two
 * exported copies binding to each other is a bug waiting for the day one side
 * updates and the other does not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <sys/stat.h>

/* Every decoder stb offers is enabled.  An earlier version trimmed the list on
 * the theory that a decoder not compiled cannot have a bug -- true, but far too
 * cheap a trade when the formats are ones real documents arrive in.  GIF in
 * particular carries a great deal of screenshot and simple-graphic text, and
 * PNM is what `pdftoppm` emits by DEFAULT: rasterizing a PDF page to PPM takes
 * 182 ms against 430 ms for PNG, because PNG spends more than half that time in
 * compression we immediately undo.  PPM is also lossless, where JPEG (173 ms)
 * costs accuracy -- measured CER 0.0250 against 0.0168 clean.  So the format we
 * most want for the rampart-totext path was the one being excluded. */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <tiffio.h>

#include "ocr-image.h"

/* ImageNet normalization PP-OCR's detector was trained with. */
static const float DET_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float DET_STD[3]  = { 0.229f, 0.224f, 0.225f };

static int img_fail(char *err, size_t errlen, const char *fmt, ...)
{
    va_list ap;
    if (err && errlen) {
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}


/* ---- TIFF ----------------------------------------------------------------
 * stb has no TIFF decoder and multi-page Group 4 is the standard archive format
 * for scans, so libtiff owns this format.  Its RGBA reader covers bilevel, LZW,
 * PackBits, Deflate, palette and CMYK behind one call.
 */

/* libtiff's default handlers print to stderr; this module never does. */
static void ocr_tiff_quiet(const char *mod, const char *fmt, va_list ap)
{
    (void)mod; (void)fmt; (void)ap;
}

static int ocr_is_tiff(const unsigned char *m, size_t n)
{
    if (n < 4) return 0;
    /* II*\0 little-endian, MM\0* big-endian; 43 is BigTIFF */
    if (m[0] == 'I' && m[1] == 'I' && (m[2] == 42 || m[2] == 43) && m[3] == 0) return 1;
    if (m[0] == 'M' && m[1] == 'M' && m[2] == 0 && (m[3] == 42 || m[3] == 43)) return 1;
    return 0;
}

/* Read-only memory source for TIFFClientOpen, so a Buffer decodes without a
 * temp file.  map() hands libtiff the buffer itself: zero-copy. */
typedef struct { const unsigned char *p; size_t n, off; } ocr_tiff_mem;

static tmsize_t tmem_read(thandle_t h, void *buf, tmsize_t n)
{
    ocr_tiff_mem *m = (ocr_tiff_mem *)h;
    size_t left = m->n - m->off;
    if ((size_t)n > left) n = (tmsize_t)left;
    memcpy(buf, m->p + m->off, (size_t)n);
    m->off += (size_t)n;
    return n;
}
static tmsize_t tmem_write(thandle_t h, void *buf, tmsize_t n) { (void)h; (void)buf; (void)n; return -1; }
static toff_t tmem_seek(thandle_t h, toff_t off, int whence)
{
    ocr_tiff_mem *m = (ocr_tiff_mem *)h;
    uint64_t base = whence == SEEK_CUR ? m->off : whence == SEEK_END ? m->n : 0;
    if (base + off > m->n) return (toff_t)-1;
    m->off = (size_t)(base + off);
    return (toff_t)m->off;
}
static int    tmem_close(thandle_t h) { (void)h; return 0; }
static toff_t tmem_size(thandle_t h)  { return (toff_t)((ocr_tiff_mem *)h)->n; }
static int    tmem_map(thandle_t h, void **base, toff_t *size)
{
    ocr_tiff_mem *m = (ocr_tiff_mem *)h;
    *base = (void *)m->p; *size = (toff_t)m->n;
    return 1;
}
static void   tmem_unmap(thandle_t h, void *base, toff_t size) { (void)h; (void)base; (void)size; }

static TIFF *ocr_tiff_open_file(const char *path)
{
    TIFFSetErrorHandler(ocr_tiff_quiet);
    TIFFSetWarningHandler(ocr_tiff_quiet);
    return TIFFOpen(path, "r");
}

static TIFF *ocr_tiff_open_mem(ocr_tiff_mem *m)
{
    TIFFSetErrorHandler(ocr_tiff_quiet);
    TIFFSetWarningHandler(ocr_tiff_quiet);
    return TIFFClientOpen("buffer", "rm", (thandle_t)m, tmem_read, tmem_write, tmem_seek,
                          tmem_close, tmem_size, tmem_map, tmem_unmap);
}

/* Directory count.  Walks the IFD chain (cheap: headers only) and leaves the
 * current directory at the end, so callers reposition afterwards. */
static int ocr_tiff_ndirs(TIFF *t)
{
    int n = 0;
    do { n++; } while (TIFFReadDirectory(t));
    return n;
}

/* Decode directory `page` of an open TIFF into `out`; reports the page count. */
static int ocr_tiff_page(TIFF *t, int page, ocr_image *out, int *npages, char *err, size_t errlen)
{
    uint32_t w = 0, h = 0, *raster = NULL;
    size_t n, i;
    int ndirs = ocr_tiff_ndirs(t);

    if (npages) *npages = ndirs;
    if (page >= ndirs)
        return img_fail(err, errlen, "TIFF has %d page%s; there is no page %d",
                        ndirs, ndirs == 1 ? "" : "s", page);
    if (!TIFFSetDirectory(t, (tdir_t)page))
        return img_fail(err, errlen, "cannot select TIFF page %d", page);

    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
    if (!w || !h) return img_fail(err, errlen, "TIFF page %d has no usable size", page);
    if ((uint64_t)w * h > 512ull * 1024 * 1024)
        return img_fail(err, errlen, "TIFF page %d is implausibly large (%ux%u)", page, w, h);

    n = (size_t)w * h;
    raster = (uint32_t *)_TIFFmalloc((tmsize_t)(n * sizeof(uint32_t)));
    if (!raster) return img_fail(err, errlen, "out of memory decoding TIFF");

    /* TOPLEFT: the default is bottom-up, which would OCR the page upside down
     * rather than fail. */
    if (!TIFFReadRGBAImageOriented(t, w, h, raster, ORIENTATION_TOPLEFT, 0)) {
        _TIFFfree(raster);
        return img_fail(err, errlen, "could not decode TIFF page %d", page);
    }

    out->px = (unsigned char *)malloc(n * 3);
    if (!out->px) { _TIFFfree(raster); return img_fail(err, errlen, "out of memory"); }
    out->w = (int)w;
    out->h = (int)h;
    for (i = 0; i < n; i++) {
        uint32_t p = raster[i];
        out->px[i * 3 + 0] = (unsigned char)TIFFGetR(p);
        out->px[i * 3 + 1] = (unsigned char)TIFFGetG(p);
        out->px[i * 3 + 2] = (unsigned char)TIFFGetB(p);
    }
    _TIFFfree(raster);
    return 0;
}

/* ---- ASCII PNM (P1/P2/P3) --------------------------------------------------
 * stb reads only the binary forms (P5/P6).  Ghostscript's pnm devices write the
 * ASCII ones, and the parse is trivial, so accept them here. */
static int ocr_is_ascii_pnm(const unsigned char *b, size_t n)
{
    return n >= 3 && b[0] == 'P' && b[1] >= '1' && b[1] <= '3' && isspace(b[2]);
}

static int pnm_tok(const unsigned char *b, size_t n, size_t *i, long *v)
{
    while (*i < n) {
        if (b[*i] == '#') { while (*i < n && b[*i] != '\n') (*i)++; }
        else if (isspace(b[*i])) (*i)++;
        else break;
    }
    if (*i >= n || !isdigit(b[*i])) return -1;
    *v = 0;
    while (*i < n && isdigit(b[*i])) {
        *v = *v * 10 + (b[*i] - '0');
        (*i)++;
        if (*v > 1000000000L) return -1;
    }
    return 0;
}

static int ocr_pnm_ascii_load(const unsigned char *b, size_t n, ocr_image *out, char *err, size_t errlen)
{
    size_t i = 2, k, npx, nval;
    long w, h, maxv = 1, v;
    int kind = b[1] - '0';              /* 1 bitmap, 2 gray, 3 rgb */
    int ch = kind == 3 ? 3 : 1;

    if (pnm_tok(b, n, &i, &w) || pnm_tok(b, n, &i, &h) || (kind != 1 && pnm_tok(b, n, &i, &maxv)))
        return img_fail(err, errlen, "bad PNM header");
    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 65535 || (uint64_t)w * h > 512ull * 1024 * 1024)
        return img_fail(err, errlen, "bad PNM dimensions (%ldx%ld)", w, h);
    npx = (size_t)w * h;
    nval = npx * ch;
    out->px = (unsigned char *)malloc(npx * 3);
    if (!out->px) return img_fail(err, errlen, "out of memory");
    for (k = 0; k < nval; k++) {
        unsigned char c;
        if (kind == 1) {                /* P1 digits may run together */
            while (i < n && (isspace(b[i]) || b[i] == '#')) {
                if (b[i] == '#') while (i < n && b[i] != '\n') i++;
                else i++;
            }
            if (i >= n) break;
            c = b[i++] == '1' ? 0 : 255;
        } else {
            if (pnm_tok(b, n, &i, &v)) break;
            c = (unsigned char)(v * 255 / maxv);
        }
        if (ch == 3) out->px[k] = c;
        else out->px[k * 3] = out->px[k * 3 + 1] = out->px[k * 3 + 2] = c;
    }
    if (k < nval) { free(out->px); out->px = NULL; return img_fail(err, errlen, "truncated PNM data"); }
    out->w = (int)w;
    out->h = (int)h;
    return 0;
}

static int ocr_read_all(const char *path, unsigned char **out, size_t *n)
{
    struct stat st;
    FILE *f;
    if (stat(path, &st) != 0 || st.st_size <= 0) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    *n = (size_t)st.st_size;
    *out = (unsigned char *)malloc(*n);
    if (!*out || fread(*out, 1, *n, f) != *n) { free(*out); *out = NULL; fclose(f); return -1; }
    fclose(f);
    return 0;
}

static int ocr_file_is_tiff(const char *path)
{
    unsigned char magic[4] = {0};
    FILE *f = fopen(path, "rb");
    int r = 0;
    if (!f) return 0;
    r = (fread(magic, 1, 4, f) == 4) && ocr_is_tiff(magic, 4);
    fclose(f);
    return r;
}

int ocr_image_page_count(const char *path)
{
    TIFF *t;
    int n;

    if (!path || !*path) return 0;
    if (!ocr_file_is_tiff(path)) {
        struct stat st;
        return stat(path, &st) == 0 ? 1 : 0;   /* every other format is one page */
    }
    t = ocr_tiff_open_file(path);
    if (!t) return 0;
    n = ocr_tiff_ndirs(t);
    TIFFClose(t);
    return n;
}

int ocr_image_page_count_mem(const void *buf, size_t len)
{
    ocr_tiff_mem m = { (const unsigned char *)buf, len, 0 };
    TIFF *t;
    int n;

    if (!buf || !len) return 0;
    if (!ocr_is_tiff(m.p, len)) return 1;
    t = ocr_tiff_open_mem(&m);
    if (!t) return 0;
    n = ocr_tiff_ndirs(t);
    TIFFClose(t);
    return n;
}

int ocr_image_load_file(const char *path, int page, ocr_image *out, int *npages, char *err, size_t errlen)
{
    struct stat st;
    int nx, ny, nc;

    if (npages) *npages = 1;
    if (!path || !*path || !out) return img_fail(err, errlen, "ocr_image_load_file: null path");
    /* distinguish "no such file" from "not a decodable image" -- stb reports
     * both as a NULL return, and the two need different fixes */
    if (stat(path, &st) != 0)
        return img_fail(err, errlen, "cannot open image '%s': %s", path, strerror(errno));

    if (ocr_file_is_tiff(path)) {
        TIFF *t = ocr_tiff_open_file(path);
        int rc;
        if (!t) return img_fail(err, errlen, "cannot open TIFF '%s'", path);
        rc = ocr_tiff_page(t, page, out, npages, err, errlen);
        TIFFClose(t);
        return rc;
    }

    {
        unsigned char m3[3] = {0};
        FILE *f = fopen(path, "rb");
        size_t got = f ? fread(m3, 1, 3, f) : 0;
        if (f) fclose(f);
        if (got == 3 && ocr_is_ascii_pnm(m3, 3)) {
            unsigned char *all = NULL;
            size_t n = 0;
            int rc;
            if (ocr_read_all(path, &all, &n) != 0)
                return img_fail(err, errlen, "cannot read '%s'", path);
            rc = ocr_pnm_ascii_load(all, n, out, err, errlen);
            free(all);
            return rc;
        }
    }

    out->px = stbi_load(path, &nx, &ny, &nc, 3);
    if (!out->px)
        return img_fail(err, errlen, "failed to decode image '%s' (unrecognized format?)", path);
    out->w = nx;
    out->h = ny;
    return 0;
}

int ocr_image_load_mem(const void *buf, size_t len, int page, ocr_image *out, int *npages, char *err, size_t errlen)
{
    int nx, ny, nc;

    if (npages) *npages = 1;
    if (!buf || !len || !out) return img_fail(err, errlen, "ocr_image_load_mem: empty buffer");

    if (ocr_is_tiff((const unsigned char *)buf, len)) {
        ocr_tiff_mem m = { (const unsigned char *)buf, len, 0 };
        TIFF *t = ocr_tiff_open_mem(&m);
        int rc;
        if (!t) return img_fail(err, errlen, "cannot open TIFF from buffer");
        rc = ocr_tiff_page(t, page, out, npages, err, errlen);
        TIFFClose(t);
        return rc;
    }

    if (ocr_is_ascii_pnm((const unsigned char *)buf, len))
        return ocr_pnm_ascii_load((const unsigned char *)buf, len, out, err, errlen);

    /* stbi takes an int length; a size_t past INT_MAX would wrap negative and
     * silently misparse (same guard clip_shim.cpp uses) */
    if (len > (size_t)INT_MAX)
        return img_fail(err, errlen, "image buffer too large (%zu bytes; max %d)", len, INT_MAX);

    out->px = stbi_load_from_memory((const stbi_uc *)buf, (int)len, &nx, &ny, &nc, 3);
    if (!out->px)
        return img_fail(err, errlen, "failed to decode image buffer (%zu bytes; unrecognized format?)", len);
    out->w = nx;
    out->h = ny;
    return 0;
}

void ocr_image_free(ocr_image *im)
{
    if (!im) return;
    if (im->px) stbi_image_free(im->px);
    im->px = NULL;
    im->w = im->h = 0;
}

/* Bilinear resize.  Uses the half-pixel center convention
 * (src = (d + 0.5) * scale - 0.5), which is what OpenCV's INTER_LINEAR does --
 * the naive `d * scale` convention shifts the image by half a destination pixel
 * and, on the small crops the recognizer sees, that is a visible error. */
int ocr_image_resize(const ocr_image *src, int dw, int dh, ocr_image *dst)
{
    int x, y, c;
    float xs, ys;

    if (!src || !src->px || !dst || dw <= 0 || dh <= 0) return -1;

    dst->px = (unsigned char *)malloc((size_t)dw * dh * 3);
    if (!dst->px) return -1;
    dst->w = dw;
    dst->h = dh;

    xs = (float)src->w / (float)dw;
    ys = (float)src->h / (float)dh;

    for (y = 0; y < dh; y++) {
        float sy = ((float)y + 0.5f) * ys - 0.5f;
        int   y0, y1;
        float fy;
        if (sy < 0.0f) sy = 0.0f;
        y0 = (int)sy;
        y1 = y0 + 1 < src->h ? y0 + 1 : src->h - 1;
        fy = sy - (float)y0;

        for (x = 0; x < dw; x++) {
            float sx = ((float)x + 0.5f) * xs - 0.5f;
            int   x0, x1;
            float fx;
            const unsigned char *p00, *p01, *p10, *p11;
            unsigned char *o;

            if (sx < 0.0f) sx = 0.0f;
            x0 = (int)sx;
            x1 = x0 + 1 < src->w ? x0 + 1 : src->w - 1;
            fx = sx - (float)x0;

            p00 = src->px + ((size_t)y0 * src->w + x0) * 3;
            p01 = src->px + ((size_t)y0 * src->w + x1) * 3;
            p10 = src->px + ((size_t)y1 * src->w + x0) * 3;
            p11 = src->px + ((size_t)y1 * src->w + x1) * 3;
            o   = dst->px + ((size_t)y * dw + x) * 3;

            for (c = 0; c < 3; c++) {
                float top = (float)p00[c] + ((float)p01[c] - (float)p00[c]) * fx;
                float bot = (float)p10[c] + ((float)p11[c] - (float)p10[c]) * fx;
                float v   = top + (bot - top) * fy;
                o[c] = (unsigned char)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
            }
        }
    }
    return 0;
}

/* Round up to a multiple of 32, with a floor of 32: the DB backbone downsamples
 * by 32, so a ragged size changes the output stride and silently misplaces every
 * box.  Rounding UP (rather than to-nearest) never discards page content. */
static int round32_up(int v)
{
    int r = ((v + 31) / 32) * 32;
    return r < 32 ? 32 : r;
}

int ocr_det_plan(int src_w, int src_h, int limit_side_len, ocr_det_scale *out)
{
    float ratio = 1.0f;
    int rw, rh;

    if (src_w <= 0 || src_h <= 0 || !out) return -1;
    if (limit_side_len <= 0) limit_side_len = 960;

    /* Cap the LONG side.  RapidOCR's default limit_type is "max": scale down
     * only when the longer side exceeds the cap, never scale up -- upscaling a
     * small image just costs time and invents no detail. */
    {
        int longest = src_w > src_h ? src_w : src_h;
        if (longest > limit_side_len) ratio = (float)limit_side_len / (float)longest;
    }

    rw = (int)((float)src_w * ratio + 0.5f);
    rh = (int)((float)src_h * ratio + 0.5f);
    out->net_w = round32_up(rw);
    out->net_h = round32_up(rh);

    /* Map network coords back to the ORIGINAL image.  Per-axis, because the
     * round-to-32 differs by axis; a single averaged scale shifts every box. */
    out->sx = (float)src_w / (float)out->net_w;
    out->sy = (float)src_h / (float)out->net_h;
    return 0;
}

int ocr_det_preprocess(const ocr_image *src, int limit_side_len,
                       float *dst, ocr_det_scale *scale)
{
    ocr_image tmp;
    const ocr_image *use;
    ocr_det_scale s;
    size_t plane;
    int x, y, c;

    if (!src || !src->px || !dst || !scale) return -1;
    if (ocr_det_plan(src->w, src->h, limit_side_len, &s) != 0) return -1;

    memset(&tmp, 0, sizeof tmp);
    if (s.net_w == src->w && s.net_h == src->h) {
        use = src;                                   /* already an exact fit */
    } else {
        if (ocr_image_resize(src, s.net_w, s.net_h, &tmp) != 0) return -1;
        use = &tmp;
    }

    /* interleaved RGB u8 -> planar CHW float32, scaled to [0,1] then normalized */
    plane = (size_t)s.net_w * s.net_h;
    for (c = 0; c < 3; c++) {
        float *o = dst + (size_t)c * plane;
        float m = DET_MEAN[c], iv = 1.0f / DET_STD[c];
        for (y = 0; y < s.net_h; y++) {
            const unsigned char *p = use->px + ((size_t)y * use->w) * 3 + c;
            for (x = 0; x < s.net_w; x++, p += 3)
                *o++ = ((float)*p / 255.0f - m) * iv;
        }
    }

    if (use == &tmp) ocr_image_free(&tmp);
    *scale = s;
    return 0;
}

/* ---- text-line crop ------------------------------------------------------ */

/* Solve an 8x8 linear system in place by Gaussian elimination with partial
 * pivoting.  Returns 0, or -1 if the matrix is singular (degenerate quad). */
static int solve8(double A[8][9], double *x)
{
    int i, j, k;

    for (i = 0; i < 8; i++) {
        int piv = i;
        double mx = fabs(A[i][i]);
        for (k = i + 1; k < 8; k++)
            if (fabs(A[k][i]) > mx) { mx = fabs(A[k][i]); piv = k; }
        if (mx < 1e-12) return -1;
        if (piv != i) for (j = 0; j <= 8; j++) { double t = A[i][j]; A[i][j] = A[piv][j]; A[piv][j] = t; }
        for (k = i + 1; k < 8; k++) {
            double f = A[k][i] / A[i][i];
            if (f == 0.0) continue;
            for (j = i; j <= 8; j++) A[k][j] -= f * A[i][j];
        }
    }
    for (i = 7; i >= 0; i--) {
        double s = A[i][8];
        for (j = i + 1; j < 8; j++) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return 0;
}

/* Perspective transform mapping DESTINATION (x,y) -> SOURCE (u,v).
 *
 * Built in that direction deliberately: warping is an inverse-map operation
 * (for each output pixel, find where it came from), so solving dst->src
 * directly avoids inverting a 3x3 afterwards. */
static int persp_dst_to_src(const float quad[4][2], float w, float h, double H[9])
{
    const double dx[4] = { 0.0, (double)w, (double)w, 0.0 };
    const double dy[4] = { 0.0, 0.0, (double)h, (double)h };
    double A[8][9], x[8];
    int i;

    memset(A, 0, sizeof A);
    for (i = 0; i < 4; i++) {
        double u = quad[i][0], v = quad[i][1];
        double *r0 = A[i * 2], *r1 = A[i * 2 + 1];
        r0[0] = dx[i]; r0[1] = dy[i]; r0[2] = 1.0;
        r0[6] = -dx[i] * u; r0[7] = -dy[i] * u; r0[8] = u;
        r1[3] = dx[i]; r1[4] = dy[i]; r1[5] = 1.0;
        r1[6] = -dx[i] * v; r1[7] = -dy[i] * v; r1[8] = v;
    }
    if (solve8(A, x) != 0) return -1;
    for (i = 0; i < 8; i++) H[i] = x[i];
    H[8] = 1.0;
    return 0;
}

static float dist2f(const float a[2], const float b[2])
{
    float dx = a[0] - b[0], dy = a[1] - b[1];
    return sqrtf(dx * dx + dy * dy);
}

void ocr_image_rot180(ocr_image *im)
{
    size_t n, i;
    unsigned char *p;

    if (!im || !im->px) return;
    p = im->px;
    n = (size_t)im->w * im->h;
    for (i = 0; i < n / 2; i++) {
        size_t j = n - 1 - i;
        unsigned char t[3];
        memcpy(t,        p + i * 3, 3);
        memcpy(p + i * 3, p + j * 3, 3);
        memcpy(p + j * 3, t,        3);
    }
}

int ocr_crop_quad(const ocr_image *src, const float quad[4][2], ocr_image *dst)
{
    float wa, wb, ha, hb;
    int w, h, x, y, c;
    double H[9];

    if (!src || !src->px || !quad || !dst) return -1;

    /* Output size from the quad's own edges (max of the opposing pair, as
     * RapidOCR does) so a skewed line is deskewed, not squashed. */
    wa = dist2f(quad[0], quad[1]);
    wb = dist2f(quad[3], quad[2]);
    ha = dist2f(quad[0], quad[3]);
    hb = dist2f(quad[1], quad[2]);
    w = (int)((wa > wb ? wa : wb) + 0.5f);
    h = (int)((ha > hb ? ha : hb) + 0.5f);
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    if ((long)w * h > 64L * 1024 * 1024) return -1;      /* absurd quad guard */

    if (persp_dst_to_src(quad, (float)w, (float)h, H) != 0) return -1;

    dst->px = (unsigned char *)malloc((size_t)w * h * 3);
    if (!dst->px) return -1;
    dst->w = w; dst->h = h;

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            double dxc = (double)x + 0.5, dyc = (double)y + 0.5;
            double den = H[6] * dxc + H[7] * dyc + H[8];
            double su, sv;
            int x0, y0, x1, y1;
            float fx, fy;
            unsigned char *o = dst->px + ((size_t)y * w + x) * 3;

            if (fabs(den) < 1e-12) { memset(o, 0, 3); continue; }
            su = (H[0] * dxc + H[1] * dyc + H[2]) / den - 0.5;
            sv = (H[3] * dxc + H[4] * dyc + H[5]) / den - 0.5;

            if (su < 0) su = 0;
            if (sv < 0) sv = 0;
            if (su > src->w - 1) su = src->w - 1;
            if (sv > src->h - 1) sv = src->h - 1;

            x0 = (int)su; y0 = (int)sv;
            x1 = x0 + 1 < src->w ? x0 + 1 : src->w - 1;
            y1 = y0 + 1 < src->h ? y0 + 1 : src->h - 1;
            fx = (float)(su - x0); fy = (float)(sv - y0);

            for (c = 0; c < 3; c++) {
                const unsigned char *p00 = src->px + ((size_t)y0 * src->w + x0) * 3 + c;
                const unsigned char *p01 = src->px + ((size_t)y0 * src->w + x1) * 3 + c;
                const unsigned char *p10 = src->px + ((size_t)y1 * src->w + x0) * 3 + c;
                const unsigned char *p11 = src->px + ((size_t)y1 * src->w + x1) * 3 + c;
                float top = (float)*p00 + ((float)*p01 - (float)*p00) * fx;
                float bot = (float)*p10 + ((float)*p11 - (float)*p10) * fx;
                float v   = top + (bot - top) * fy;
                o[c] = (unsigned char)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v + 0.5f));
            }
        }
    }

    /* Markedly taller than wide => vertical text; stand it up so the
     * horizontal-only recognizer can read it (RapidOCR's 1.5 threshold). */
    if ((float)dst->h / (float)dst->w >= 1.5f) {
        ocr_image rot;
        int xx, yy;
        rot.w = dst->h; rot.h = dst->w;
        rot.px = (unsigned char *)malloc((size_t)rot.w * rot.h * 3);
        if (!rot.px) return 0;                    /* keep the unrotated crop */
        /* rotate 90 counter-clockwise */
        for (yy = 0; yy < rot.h; yy++)
            for (xx = 0; xx < rot.w; xx++) {
                int sx2 = yy, sy2 = dst->h - 1 - xx;
                memcpy(rot.px + ((size_t)yy * rot.w + xx) * 3,
                       dst->px + ((size_t)sy2 * dst->w + sx2) * 3, 3);
            }
        free(dst->px);
        *dst = rot;
    }
    return 0;
}

int ocr_image_rot90(ocr_image *im, int ccw)
{
    ocr_image out;
    int x, y;

    if (!im || !im->px) return -1;
    out.w = im->h;
    out.h = im->w;
    out.px = (unsigned char *)malloc((size_t)out.w * out.h * 3);
    if (!out.px) return -1;

    for (y = 0; y < out.h; y++) {
        for (x = 0; x < out.w; x++) {
            int sx, sy;
            if (ccw) { sx = im->w - 1 - y; sy = x; }
            else     { sx = y;             sy = im->h - 1 - x; }
            memcpy(out.px + ((size_t)y * out.w + x) * 3,
                   im->px + ((size_t)sy * im->w + sx) * 3, 3);
        }
    }
    /* the source may have come from stb (stbi_image_free) or from our own
     * malloc; both are plain free() underneath, so releasing it here is safe */
    ocr_image_free(im);
    *im = out;
    return 0;
}
