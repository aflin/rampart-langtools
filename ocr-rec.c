/* ocr-rec.c -- recognition/classification preprocessing, dictionary, CTC decode. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ocr-rec.h"

/* ---- dictionary ---------------------------------------------------------- */

void ocr_dict_free(ocr_dict *d)
{
    size_t i;
    if (!d || !d->item) return;
    for (i = 0; i < d->n_items; i++) free(d->item[i]);
    free(d->item);
    d->item = NULL;
    d->n_items = 0;
}

int ocr_dict_load(const char *path, ocr_dict *d, char *err, size_t errlen)
{
    FILE *f;
    char *buf = NULL, **items = NULL;
    long sz;
    size_t n = 0, cap = 0, i, start;
    int rc = -1;

    if (!path || !d) { if (err) snprintf(err, errlen, "ocr_dict_load: null path"); return -1; }
    memset(d, 0, sizeof *d);

    f = fopen(path, "rb");
    if (!f) { if (err) snprintf(err, errlen, "cannot open dictionary '%s'", path); return -1; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 32L * 1024 * 1024) { fclose(f); if (err) snprintf(err, errlen, "dictionary '%s' has an implausible size", path); return -1; }
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); if (err) snprintf(err, errlen, "oom"); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); if (err) snprintf(err, errlen, "short read on '%s'", path); return -1; }
    fclose(f);
    buf[sz] = '\0';

/* slot 0 is the CTC blank, and a trailing space is appended after the file's
 * lines -- see the header; both are part of PP-OCR's class list, not the file */
#define DICT_PUSH(s, len) do {                                              \
        if (n == cap) {                                                     \
            size_t nc = cap ? cap * 2 : 512;                                \
            char **ni = (char **)realloc(items, nc * sizeof(char *));       \
            if (!ni) goto oom;                                              \
            items = ni; cap = nc;                                           \
        }                                                                   \
        items[n] = (char *)malloc((len) + 1);                               \
        if (!items[n]) goto oom;                                            \
        memcpy(items[n], (s), (len));                                       \
        items[n][(len)] = '\0';                                             \
        n++;                                                                \
    } while (0)

    DICT_PUSH("", 0);                                    /* 0: CTC blank */

    start = 0;
    for (i = 0; i <= (size_t)sz; i++) {
        if (i == (size_t)sz || buf[i] == '\n') {
            size_t end = i;
            /* the shipped dictionary is CRLF; leaving the \r attached would put
             * a carriage return on every decoded character */
            while (end > start && (buf[end - 1] == '\r')) end--;
            if (i == (size_t)sz && end == start) break;  /* ignore trailing newline */
            DICT_PUSH(buf + start, end - start);
            start = i + 1;
        }
    }

    DICT_PUSH(" ", 1);                                   /* N+1: literal space */
#undef DICT_PUSH

    d->item = items;
    d->n_items = n;
    items = NULL;
    rc = 0;
    goto out;

oom:
    if (err) snprintf(err, errlen, "out of memory loading '%s'", path);
out:
    free(buf);
    if (items) { for (i = 0; i < n; i++) free(items[i]); free(items); }
    return rc;
}

/* ---- preprocessing ------------------------------------------------------- */

int ocr_batch_width(const ocr_image *crops, size_t n, int height,
                    int min_width, int max_width)
{
    size_t i;
    int w = min_width > 0 ? min_width : height;

    if (!crops || !n || height <= 0) return w;
    for (i = 0; i < n; i++) {
        int cw;
        if (crops[i].h <= 0) continue;
        cw = (int)((float)height * (float)crops[i].w / (float)crops[i].h + 0.5f);
        if (cw > w) w = cw;
    }
    if (max_width > 0 && w > max_width) w = max_width;
    /* multiple of 8: the recognizer downsamples width by 8, so a ragged width
     * wastes a partial output column */
    w = ((w + 7) / 8) * 8;
    if (w < 8) w = 8;
    return w;
}

int ocr_pack_crops(const ocr_image *crops, size_t n, int height, int width, float *dst)
{
    size_t i;
    size_t plane = (size_t)height * width;

    if (!crops || !dst || height <= 0 || width <= 0) return -1;

    for (i = 0; i < n; i++) {
        float *base = dst + i * 3 * plane;
        ocr_image r;
        const ocr_image *use;
        int rw, x, y, c;

        memset(&r, 0, sizeof r);
        /* zero the whole sample first: everything past the resized width is
         * padding, and PP-OCR pads with zeros (mid-grey after normalization) */
        memset(base, 0, 3 * plane * sizeof(float));

        if (crops[i].h <= 0 || crops[i].w <= 0 || !crops[i].px) continue;

        rw = (int)((float)height * (float)crops[i].w / (float)crops[i].h + 0.5f);
        if (rw < 1) rw = 1;
        if (rw > width) rw = width;

        if (crops[i].w == rw && crops[i].h == height) {
            use = &crops[i];
        } else {
            if (ocr_image_resize(&crops[i], rw, height, &r) != 0) continue;
            use = &r;
        }

        /* interleaved u8 -> planar CHW, x/255 then (x-0.5)/0.5 == x/127.5 - 1 */
        for (c = 0; c < 3; c++) {
            float *o = base + (size_t)c * plane;
            for (y = 0; y < height; y++) {
                const unsigned char *p = use->px + ((size_t)y * use->w) * 3 + c;
                float *row = o + (size_t)y * width;
                for (x = 0; x < rw; x++, p += 3)
                    row[x] = (float)*p / 127.5f - 1.0f;
            }
        }
        if (use == &r) ocr_image_free(&r);
    }
    return 0;
}

/* ---- CTC decode ---------------------------------------------------------- */

int ocr_ctc_decode(const float *logits, int T, int C, const ocr_dict *d,
                   char *out, size_t outlen, float *score)
{
    int t, prev = -1;
    size_t used = 0, kept = 0;
    double conf = 0.0;

    if (!logits || !d || !out || !outlen || T <= 0 || C <= 0) {
        if (out && outlen) out[0] = '\0';
        if (score) *score = 0.0f;
        return -1;
    }
    out[0] = '\0';

    for (t = 0; t < T; t++) {
        const float *row = logits + (size_t)t * C;
        int c, best = 0;
        float bv = row[0];

        for (c = 1; c < C; c++) if (row[c] > bv) { bv = row[c]; best = c; }

        /* collapse repeats, then drop blanks -- in that order: a repeated
         * character separated by a blank is two characters, which is the whole
         * point of the blank symbol */
        if (best != prev && best != 0) {
            const char *s = (size_t)best < d->n_items ? d->item[best] : NULL;
            if (s && *s) {
                size_t l = strlen(s);
                if (used + l + 1 <= outlen) { memcpy(out + used, s, l); used += l; }
            }
            conf += bv;
            kept++;
        }
        prev = best;
    }
    out[used] = '\0';
    if (score) *score = kept ? (float)(conf / (double)kept) : 0.0f;
    return 0;
}
