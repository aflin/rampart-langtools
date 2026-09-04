/* ocr-layout.c -- PP-DocLayoutV3 preprocessing, decoding and line ordering. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>

#include "ocr-layout.h"
#include "ocr-det.h"

const char *const ocr_layout_labels[OCR_LAYOUT_NCLASS] = {
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote"
};

static int lay_fail(char *err, size_t errlen, const char *fmt, ...)
{
    va_list ap;
    if (err && errlen) {
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

int ocr_layout_preprocess(const ocr_image *src, float *dst, char *err, size_t errlen)
{
    ocr_image r;
    size_t n = (size_t)OCR_LAYOUT_SIDE * OCR_LAYOUT_SIDE, i;

    if (!src || !src->px || !dst) return lay_fail(err, errlen, "layout: no image");

    memset(&r, 0, sizeof r);
    if (ocr_image_resize(src, OCR_LAYOUT_SIDE, OCR_LAYOUT_SIDE, &r) != 0)
        return lay_fail(err, errlen, "layout: could not resize to %dx%d",
                        OCR_LAYOUT_SIDE, OCR_LAYOUT_SIDE);

    /* CHW, RGB, 0..1.  See the normalisation note in the header: 1/255 only. */
    for (i = 0; i < n; i++) {
        dst[i]         = (float)r.px[i * 3 + 0] / 255.0f;
        dst[n + i]     = (float)r.px[i * 3 + 1] / 255.0f;
        dst[2 * n + i] = (float)r.px[i * 3 + 2] / 255.0f;
    }
    ocr_image_free(&r);
    return 0;
}

static int cmp_region_order(const void *a, const void *b)
{
    const ocr_region *p = (const ocr_region *)a, *q = (const ocr_region *)b;
    if (p->order != q->order) return p->order < q->order ? -1 : 1;
    /* same key: fall back to position, so the result is deterministic */
    if (p->y0 != q->y0) return p->y0 < q->y0 ? -1 : 1;
    return p->x0 < q->x0 ? -1 : (p->x0 > q->x0);
}

int ocr_layout_decode(const float *det, size_t rows, size_t cols, float thresh,
                      ocr_region **out, size_t *n_out)
{
    ocr_region *r = NULL;
    size_t i, k = 0;

    if (out)   *out = NULL;
    if (n_out) *n_out = 0;
    if (!det || !rows || cols < 7) return 0;

    r = (ocr_region *)malloc(rows * sizeof *r);
    if (!r) return -1;

    for (i = 0; i < rows; i++) {
        const float *o = det + i * cols;
        int lab = (int)o[0];
        if (!(o[1] > thresh)) continue;
        /* a box with no area is a padding row, not a region */
        if (!(o[4] > o[2]) || !(o[5] > o[3])) continue;
        r[k].label = (lab >= 0 && lab < OCR_LAYOUT_NCLASS) ? lab : -1;
        r[k].score = o[1];
        r[k].x0 = o[2]; r[k].y0 = o[3];
        r[k].x1 = o[4]; r[k].y1 = o[5];
        r[k].order = (int)o[6];
        k++;
    }
    if (k) qsort(r, k, sizeof *r, cmp_region_order);
    *out = r;
    *n_out = k;
    return 0;
}

/* axis-aligned bounds of a quad */
static void quad_bounds(const ocr_box *b, float *x0, float *y0, float *x1, float *y1)
{
    int i;
    *x0 = *x1 = b->pt[0][0];
    *y0 = *y1 = b->pt[0][1];
    for (i = 1; i < 4; i++) {
        if (b->pt[i][0] < *x0) *x0 = b->pt[i][0];
        if (b->pt[i][0] > *x1) *x1 = b->pt[i][0];
        if (b->pt[i][1] < *y0) *y0 = b->pt[i][1];
        if (b->pt[i][1] > *y1) *y1 = b->pt[i][1];
    }
}

typedef struct { size_t idx; int rank; size_t seq; } lay_key;

static int cmp_lay_key(const void *a, const void *b)
{
    const lay_key *p = (const lay_key *)a, *q = (const lay_key *)b;
    if (p->rank != q->rank) return p->rank < q->rank ? -1 : 1;
    return p->seq < q->seq ? -1 : (p->seq > q->seq);
}


void ocr_layout_order(const ocr_box *b, size_t n,
                      const ocr_region *r, size_t nr,
                      size_t *order_out)
{
    lay_key *k;
    size_t i, j;

    if (!order_out || !n) return;
    for (i = 0; i < n; i++) order_out[i] = i;
    if (!b || !r || !nr) return;                 /* nothing found: leave as is */

    k = (lay_key *)malloc(n * sizeof *k);
    if (!k) return;                              /* oom: the existing order stands */

    for (i = 0; i < n; i++) {
        float x0, y0, x1, y1, best_ov = 0.0f, best_d = FLT_MAX;
        int best = -1, nearest = -1;
        float cx, cy;

        quad_bounds(&b[i], &x0, &y0, &x1, &y1);
        cx = 0.5f * (x0 + x1);
        cy = 0.5f * (y0 + y1);

        {
            float line_area = (x1 - x0) * (y1 - y0);
            float best_area = 0.0f;
            if (line_area <= 0.0f) line_area = 1.0f;

            for (j = 0; j < nr; j++) {
                float ox = (x1 < r[j].x1 ? x1 : r[j].x1) - (x0 > r[j].x0 ? x0 : r[j].x0);
                float oy = (y1 < r[j].y1 ? y1 : r[j].y1) - (y0 > r[j].y0 ? y0 : r[j].y0);
                float d, dx, dy;
                if (ox > 0.0f && oy > 0.0f) {
                    float frac = (ox * oy) / line_area;
                    float area = (r[j].x1 - r[j].x0) * (r[j].y1 - r[j].y0);
                    /* Most SPECIFIC wins, not largest overlap.  A page-sized
                     * region (this model likes to call a columned body a
                     * "table") contains every line just as fully as the line's
                     * own text box does, so absolute overlap ties and the
                     * container swallows the page -- taking the whole reading
                     * order with it. */
                    if (frac > best_ov + 0.01f ||
                        (frac > best_ov - 0.01f && best >= 0 && area < best_area)) {
                        best_ov = frac > best_ov ? frac : best_ov;
                        best_area = area;
                        best = (int)j;
                    }
                }
                dx = cx - 0.5f * (r[j].x0 + r[j].x1);
                dy = cy - 0.5f * (r[j].y0 + r[j].y1);
                d = dx * dx + dy * dy;
                if (d < best_d) { best_d = d; nearest = (int)j; }
            }
        }
        if (best < 0) best = nearest;            /* no overlap: nearest region */
        k[i].idx  = i;
        k[i].rank = best >= 0 ? r[best].order : 0;
        k[i].seq  = i;                           /* keep the within-block order */
    }

    qsort(k, n, sizeof *k, cmp_lay_key);
    for (i = 0; i < n; i++) order_out[i] = k[i].idx;
    free(k);
}

/* Thresholds, tuned on the pages in claude-work/layout-fixtures.  See the
 * comment in the sweep below for what each rejects. */
#define OCR_COL_MAX_CROSS 0.15f
#define OCR_COL_MIN_SIDE  0.25f

int ocr_layout_looks_multicolumn(const ocr_box *b, size_t n, float min_gap,
                                 float *cross_out, float *side_out)
{
    float best_cross = 1.0f, best_side = 0.0f;
    float *lo = NULL, *hi = NULL;
    float minx = FLT_MAX, maxx = -FLT_MAX, content;
    size_t i;
    int t, found = 0;

    (void)min_gap;                               /* kept for ABI; the test is
                                                  * now a ratio, not a width */
    if (cross_out) *cross_out = 1.0f;
    if (side_out)  *side_out  = 0.0f;
    if (!b || n < 8) return 0;                   /* too little to judge */

    lo = (float *)malloc(n * sizeof *lo);
    hi = (float *)malloc(n * sizeof *hi);
    if (!lo || !hi) { free(lo); free(hi); return 0; }

    for (i = 0; i < n; i++) {
        float x0, y0, x1, y1;
        quad_bounds(&b[i], &x0, &y0, &x1, &y1);
        lo[i] = x0; hi[i] = x1;
        if (x0 < minx) minx = x0;
        if (x1 > maxx) maxx = x1;
    }
    content = maxx - minx;
    if (content <= 1.0f) { free(lo); free(hi); return 0; }

    /* Sweep the interior and count the lines STRADDLING each point.  In one
     * column every full line straddles the middle; a gutter is where almost
     * nothing does.
     *
     * Two thresholds, both chosen from measurement rather than taste:
     *
     *   crossing < 10%   An earlier version demanded that NOTHING cross, which
     *                    no real newspaper satisfies -- a headline spanning two
     *                    of six columns crosses that gutter, and with enough
     *                    such headings every gutter is dirty.  Measured on a
     *                    1920 broadsheet: 0.067 crossing at the best split.
     *
     *   weaker side >=25%  Crossing alone is not enough: a single-column page
     *                    with equation numbers down the right margin has a
     *                    clean vertical line through it (measured 0.000) but
     *                    the "column" beyond it is three numerals.  Requiring
     *                    both sides to carry a real share of the page rejects
     *                    that (0.169) while the true multi-column pages sit at
     *                    0.33 to 0.49.
     */
    for (t = 15; t <= 85; t++) {
        float x = minx + content * (float)t / 100.0f;
        size_t cross = 0, left = 0, right = 0, side;

        for (i = 0; i < n; i++) {
            if (lo[i] < x && hi[i] > x) cross++;
            else if (hi[i] <= x) left++;
            else right++;
        }
        if (left < 3 || right < 3) continue;
        side = left < right ? left : right;
        {
            float cf = (float)cross / (float)n, sf = (float)side / (float)n;
            int passes = (cf < OCR_COL_MAX_CROSS && sf >= OCR_COL_MIN_SIDE);
            /* Report the split the DECISION rests on: the best passing one, or
             * failing that the most gutter-like one seen.  Reporting the
             * minimum-crossing split regardless would describe a different
             * split from the one that fired, which is how this was first
             * mis-tuned. */
            if (passes) {
                if (!found || cf < best_cross) { best_cross = cf; best_side = sf; }
                found = 1;
            } else if (!found && cf < best_cross) {
                best_cross = cf; best_side = sf;
            }
        }
    }

    if (cross_out) *cross_out = best_cross;
    if (side_out)  *side_out  = best_side;
    free(lo); free(hi);
    return found;
}
