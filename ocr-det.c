/* ocr-det.c -- DB detection postprocessing (see ocr-det.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ocr-det.h"

void ocr_det_opts_default(ocr_det_opts *o)
{
    if (!o) return;
    o->thresh       = 0.3f;
    o->box_thresh   = 0.5f;
    o->unclip_ratio = 1.6f;
    o->min_size     = 3;
    o->max_boxes    = 1000;
}

/* ---- small geometry ------------------------------------------------------ */

typedef struct { float x, y; } pt2;

/* A minimum-area rectangle, in the same parameterization OpenCV's minAreaRect
 * returns: centre, side lengths, and the unit direction of the `w` side.  The
 * direction is kept as a vector rather than an angle -- unclip and corner
 * generation both want the vector, and going through an angle would add two
 * trig calls and a wrap-around case for nothing. */
typedef struct {
    pt2   c;
    float w, h;
    pt2   u;        /* unit vector along w; the h axis is (-u.y, u.x) */
} min_rect;

static int cmp_pt(const void *a, const void *b)
{
    const pt2 *p = (const pt2 *)a, *q = (const pt2 *)b;
    if (p->x < q->x) return -1;
    if (p->x > q->x) return 1;
    if (p->y < q->y) return -1;
    if (p->y > q->y) return 1;
    return 0;
}

static float cross_o(pt2 o, pt2 a, pt2 b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/* Convex hull, Andrew's monotone chain.  `p` is sorted in place; `hull` must
 * hold at least n+1 points.  Returns the hull size (counter-clockwise). */
static int convex_hull(pt2 *p, int n, pt2 *hull)
{
    int i, k = 0;

    if (n < 3) {
        for (i = 0; i < n; i++) hull[i] = p[i];
        return n;
    }
    qsort(p, (size_t)n, sizeof *p, cmp_pt);

    for (i = 0; i < n; i++) {                       /* lower */
        while (k >= 2 && cross_o(hull[k - 2], hull[k - 1], p[i]) <= 0.0f) k--;
        hull[k++] = p[i];
    }
    {
        int lower = k + 1;
        for (i = n - 2; i >= 0; i--) {              /* upper */
            while (k >= lower && cross_o(hull[k - 2], hull[k - 1], p[i]) <= 0.0f) k--;
            hull[k++] = p[i];
        }
    }
    return k - 1;                                    /* last == first */
}

/* Minimum-area enclosing rectangle by rotating calipers: the optimal rectangle
 * always has a side flush with a hull edge, so try each edge as the axis and
 * keep the smallest area.  Hulls here are tiny (tens of points), so the plain
 * O(hull * hull) projection is not worth optimizing away. */
static int min_area_rect(const pt2 *hull, int n, min_rect *out)
{
    int i, j;
    float best = -1.0f;

    if (n <= 0) return -1;
    if (n == 1) {
        out->c = hull[0]; out->w = out->h = 0.0f;
        out->u.x = 1.0f; out->u.y = 0.0f;
        return 0;
    }

    for (i = 0; i < n; i++) {
        pt2 a = hull[i], b = hull[(i + 1) % n];
        float dx = b.x - a.x, dy = b.y - a.y;
        float len = sqrtf(dx * dx + dy * dy);
        float ux, uy, min_u, max_u, min_v, max_v, area;

        if (len < 1e-6f) continue;
        ux = dx / len; uy = dy / len;

        min_u = max_u = hull[0].x * ux + hull[0].y * uy;
        min_v = max_v = -hull[0].x * uy + hull[0].y * ux;
        for (j = 1; j < n; j++) {
            float pu =  hull[j].x * ux + hull[j].y * uy;
            float pv = -hull[j].x * uy + hull[j].y * ux;
            if (pu < min_u) min_u = pu;
            if (pu > max_u) max_u = pu;
            if (pv < min_v) min_v = pv;
            if (pv > max_v) max_v = pv;
        }
        area = (max_u - min_u) * (max_v - min_v);
        if (best < 0.0f || area < best) {
            float cu = (min_u + max_u) * 0.5f, cv = (min_v + max_v) * 0.5f;
            best   = area;
            out->w = max_u - min_u;
            out->h = max_v - min_v;
            out->u.x = ux; out->u.y = uy;
            /* centre back from (u,v) axes into image coords */
            out->c.x = cu * ux - cv * uy;
            out->c.y = cu * uy + cv * ux;
        }
    }
    return best < 0.0f ? -1 : 0;
}

/* The rect's four corners, clockwise in image coordinates. */
static void rect_corners(const min_rect *r, pt2 *q)
{
    pt2 a, b;
    float hw = r->w * 0.5f, hh = r->h * 0.5f;

    a.x = r->u.x * hw;  a.y = r->u.y * hw;      /* half-extent along w */
    b.x = -r->u.y * hh; b.y = r->u.x * hh;      /* half-extent along h */

    q[0].x = r->c.x - a.x - b.x;  q[0].y = r->c.y - a.y - b.y;
    q[1].x = r->c.x + a.x - b.x;  q[1].y = r->c.y + a.y - b.y;
    q[2].x = r->c.x + a.x + b.x;  q[2].y = r->c.y + a.y + b.y;
    q[3].x = r->c.x - a.x + b.x;  q[3].y = r->c.y - a.y + b.y;
}

/* Order 4 points clockwise starting from the top-left, which is the corner
 * order the recognizer's perspective crop assumes.  Sort by x, then decide
 * top/bottom within each pair by y -- the same construction PaddleOCR uses. */
static void order_quad(pt2 *q)
{
    pt2 s[4], tl, tr, br, bl;
    int i, j;

    memcpy(s, q, sizeof s);
    for (i = 0; i < 3; i++)                    /* tiny insertion sort by x */
        for (j = i + 1; j < 4; j++)
            if (s[j].x < s[i].x) { pt2 t = s[i]; s[i] = s[j]; s[j] = t; }

    if (s[0].y <= s[1].y) { tl = s[0]; bl = s[1]; } else { tl = s[1]; bl = s[0]; }
    if (s[2].y <= s[3].y) { tr = s[2]; br = s[3]; } else { tr = s[3]; br = s[2]; }

    q[0] = tl; q[1] = tr; q[2] = br; q[3] = bl;
}

/* Mean probability inside the quad -- PaddleOCR's `box_score_fast`: mask the
 * quad inside its bounding box and average the map there.  Implemented as a
 * scanline fill of the (convex) quad, which needs no mask buffer at all. */
static float box_score(const float *prob, int mw, int mh, const pt2 *q)
{
    int ymin = mh, ymax = -1, y, i;
    double sum = 0.0;
    size_t cnt = 0;

    for (i = 0; i < 4; i++) {
        int yi = (int)floorf(q[i].y), ya = (int)ceilf(q[i].y);
        if (yi < ymin) ymin = yi;
        if (ya > ymax) ymax = ya;
    }
    if (ymin < 0) ymin = 0;
    if (ymax > mh - 1) ymax = mh - 1;
    if (ymin > ymax) return 0.0f;

    for (y = ymin; y <= ymax; y++) {
        float cy = (float)y + 0.5f;
        float xs[8];
        int nx = 0, x, x0, x1;

        /* crossings of the scanline with the four edges */
        for (i = 0; i < 4; i++) {
            pt2 a = q[i], b = q[(i + 1) & 3];
            float lo = a.y < b.y ? a.y : b.y, hi = a.y < b.y ? b.y : a.y;
            if (cy < lo || cy >= hi) continue;
            if (fabsf(b.y - a.y) < 1e-9f) continue;
            xs[nx++] = a.x + (cy - a.y) * (b.x - a.x) / (b.y - a.y);
        }
        if (nx < 2) continue;
        /* convex quad: the span is simply min..max crossing */
        {
            float lo = xs[0], hi = xs[0];
            for (i = 1; i < nx; i++) { if (xs[i] < lo) lo = xs[i]; if (xs[i] > hi) hi = xs[i]; }
            x0 = (int)floorf(lo); x1 = (int)ceilf(hi) - 1;
            if (x0 < 0) x0 = 0;
            if (x1 > mw - 1) x1 = mw - 1;
            for (x = x0; x <= x1; x++) { sum += prob[(size_t)y * mw + x]; cnt++; }
        }
    }
    return cnt ? (float)(sum / (double)cnt) : 0.0f;
}

/* Unclip: expand the box so it covers the glyphs, not just the shrunk kernel DB
 * was trained to predict.
 *
 * PaddleOCR offsets the polygon by  d = area * ratio / perimeter  using Clipper
 * with round joins, then takes minAreaRect of the result.  For a RECTANGLE --
 * which is all minAreaRect ever produces -- that composition has a closed form:
 * round-offsetting a rectangle by d yields a rounded rectangle whose minimum-
 * area rectangle is the original grown by d on every side, i.e. w+2d, h+2d
 * about the same centre and angle.  So the exact result is reachable in three
 * lines, and vendoring Clipper buys nothing.  (It WOULD be needed for genuinely
 * polygonal, curved-text boxes -- not a v1 concern.) */
static void unclip_rect(min_rect *r, float ratio)
{
    float area = r->w * r->h;
    float peri = 2.0f * (r->w + r->h);
    float d;

    if (peri < 1e-6f) return;
    d = area * ratio / peri;
    r->w += 2.0f * d;
    r->h += 2.0f * d;
}

/* ---- connected components ------------------------------------------------ */

/* 8-connected flood fill with an explicit stack (recursion would blow the stack
 * on a full-page component).  Collects the component's pixels into `buf`. */
static int flood(const unsigned char *bin, int mw, int mh, int *lab, int seed,
                 int label, pt2 **buf, size_t *cap, size_t *n,
                 int *stack, size_t stack_cap)
{
    size_t sp = 0;
    static const int dx[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static const int dy[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

    stack[sp++] = seed;
    lab[seed] = label;
    *n = 0;

    while (sp) {
        int idx = stack[--sp];
        int x = idx % mw, y = idx / mw, k;

        if (*n == *cap) {
            size_t nc = *cap ? *cap * 2 : 256;
            pt2 *nb = (pt2 *)realloc(*buf, nc * sizeof(pt2));
            if (!nb) return -1;
            *buf = nb; *cap = nc;
        }
        (*buf)[*n].x = (float)x;
        (*buf)[*n].y = (float)y;
        (*n)++;

        for (k = 0; k < 8; k++) {
            int nx2 = x + dx[k], ny2 = y + dy[k], ni;
            if (nx2 < 0 || ny2 < 0 || nx2 >= mw || ny2 >= mh) continue;
            ni = ny2 * mw + nx2;
            if (lab[ni] || !bin[ni]) continue;
            lab[ni] = label;
            if (sp < stack_cap) stack[sp++] = ni;
            else return -1;                   /* stack exhausted: bail cleanly */
        }
    }
    return 0;
}

int ocr_det_boxes(const float *prob, int mw, int mh,
                  const ocr_det_scale *sc, const ocr_det_opts *opt,
                  ocr_box **out, size_t *n_out)
{
    ocr_det_opts d;
    unsigned char *bin = NULL;
    int *lab = NULL, *stack = NULL;
    pt2 *pts = NULL, *hull = NULL;
    size_t pcap = 0, npx, i;
    ocr_box *boxes = NULL;
    size_t nb = 0, bcap = 0;
    int label = 0, rc = -1;
    float ssx, ssy;

    if (!prob || mw <= 0 || mh <= 0 || !out || !n_out) return -1;
    *out = NULL; *n_out = 0;

    if (opt) d = *opt; else ocr_det_opts_default(&d);
    ssx = sc ? sc->sx : 1.0f;
    ssy = sc ? sc->sy : 1.0f;

    npx = (size_t)mw * mh;
    bin   = (unsigned char *)malloc(npx);
    lab   = (int *)calloc(npx, sizeof(int));
    stack = (int *)malloc(npx * sizeof(int));
    hull  = (pt2 *)malloc((npx + 1) * sizeof(pt2));
    if (!bin || !lab || !stack || !hull) goto done;

    for (i = 0; i < npx; i++) bin[i] = prob[i] >= d.thresh ? 1u : 0u;

    for (i = 0; i < npx; i++) {
        size_t np = 0;
        int nh;
        min_rect r;
        pt2 quad[4];
        float shorter, score;
        ocr_box *bx;

        if (!bin[i] || lab[i]) continue;
        if (d.max_boxes > 0 && (int)nb >= d.max_boxes) break;

        if (flood(bin, mw, mh, lab, (int)i, ++label, &pts, &pcap, &np, stack, npx) != 0)
            goto done;
        /* A component smaller than the min side can never survive the size
         * filter; skipping the hull for it is pure savings on noisy scans,
         * which produce a great many single-pixel specks. */
        if (np < (size_t)(d.min_size * d.min_size) && np < 4) continue;

        nh = convex_hull(pts, (int)np, hull);
        if (nh < 2) continue;
        if (min_area_rect(hull, nh, &r) != 0) continue;

        shorter = r.w < r.h ? r.w : r.h;
        if (shorter < (float)d.min_size) continue;

        /* Score BEFORE unclipping: the probability mass belongs to the shrunk
         * kernel DB predicts, and scoring the expanded box would dilute every
         * score with the background the expansion just swept in. */
        rect_corners(&r, quad);
        score = box_score(prob, mw, mh, quad);
        if (score < d.box_thresh) continue;

        unclip_rect(&r, d.unclip_ratio);
        shorter = r.w < r.h ? r.w : r.h;
        if (shorter < (float)d.min_size + 2.0f) continue;

        rect_corners(&r, quad);
        order_quad(quad);

        if (nb == bcap) {
            size_t nc = bcap ? bcap * 2 : 64;
            ocr_box *nbx = (ocr_box *)realloc(boxes, nc * sizeof(ocr_box));
            if (!nbx) goto done;
            boxes = nbx; bcap = nc;
        }
        bx = &boxes[nb++];
        bx->score = score;
        {
            int k;
            for (k = 0; k < 4; k++) {
                /* map network coords -> original image, per axis, and clamp:
                 * unclip can push a box past the page edge */
                float X = quad[k].x * ssx, Y = quad[k].y * ssy;
                float maxx = (float)mw * ssx, maxy = (float)mh * ssy;
                bx->pt[k][0] = X < 0.0f ? 0.0f : (X > maxx ? maxx : X);
                bx->pt[k][1] = Y < 0.0f ? 0.0f : (Y > maxy ? maxy : Y);
            }
        }
    }

    *out = boxes; *n_out = nb;
    boxes = NULL;
    rc = 0;

done:
    free(bin); free(lab); free(stack); free(hull); free(pts); free(boxes);
    return rc;
}

/* ---- reading order ------------------------------------------------------- */

static void box_center(const ocr_box *b, float *cx, float *cy)
{
    int i;
    float sx = 0.0f, sy = 0.0f;
    for (i = 0; i < 4; i++) { sx += b->pt[i][0]; sy += b->pt[i][1]; }
    *cx = sx * 0.25f;
    *cy = sy * 0.25f;
}

/* The quad's OWN width and height (along its edges), not the axis-aligned
 * bounding box.  The distinction matters as soon as a page is tilted: a 1200 px
 * line at 15 degrees has an axis-aligned height of ~344 px against 35 px of
 * actual text, so any threshold derived from the bounding box balloons and
 * merges neighbouring lines together. */
static void box_dims(const ocr_box *b, float *w, float *h)
{
    float dx1 = b->pt[1][0] - b->pt[0][0], dy1 = b->pt[1][1] - b->pt[0][1];
    float dx2 = b->pt[3][0] - b->pt[0][0], dy2 = b->pt[3][1] - b->pt[0][1];
    *w = sqrtf(dx1 * dx1 + dy1 * dy1);
    *h = sqrtf(dx2 * dx2 + dy2 * dy2);
}

float ocr_det_tall_fraction(const ocr_box *b, size_t n)
{
    size_t i, tall = 0;
    float w, h;

    if (!b || !n) return 0.0f;
    for (i = 0; i < n; i++) {
        box_dims(&b[i], &w, &h);
        if (h > w) tall++;
    }
    return (float)tall / (float)n;
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

void ocr_det_sort_boxes(ocr_box *b, size_t n, float line_tol)
{
    size_t i, j;
    float *tmp = NULL, theta = 0.0f, ct = 1.0f, st = 0.0f;

    if (!b || n < 2) return;

    /* Reading order must survive a ROTATED page, not merely a slightly skewed
     * one.  Each detected box is a rotated quad, so it already knows the local
     * text direction; take the MEDIAN of those directions (median, not mean: a
     * few oddly-rotated boxes are normal and would drag an average) and sort in
     * a frame rotated to match.
     *
     * A true rotation is used rather than the shear `cy - cx*slope` that was
     * here before.  The shear is a small-angle approximation: it is fine at 3
     * degrees, visibly wrong by 15, and meaningless approaching 90.  Rotating
     * the centres costs one sin/cos for the whole page and is exact at any
     * angle.  `perp` orders lines down the page, `along` orders boxes within a
     * line; at theta = 0 they degenerate to plain y and x. */
    tmp = (float *)malloc(n * sizeof *tmp);
    if (tmp) {
        size_t m = 0;
        for (i = 0; i < n; i++) {
            float dx = b[i].pt[1][0] - b[i].pt[0][0];
            float dy = b[i].pt[1][1] - b[i].pt[0][1];
            float a;
            if (dx * dx + dy * dy < 4.0f) continue;         /* too short to aim */
            a = atan2f(dy, dx);
            /* fold onto (-90, 90]: a text line and the same line read backwards
             * describe the same page orientation */
            if (a >  1.57079633f) a -= 3.14159265f;
            if (a <= -1.57079633f) a += 3.14159265f;
            tmp[m++] = a;
        }
        if (m >= 3) {
            qsort(tmp, m, sizeof *tmp, cmp_float);
            theta = tmp[m / 2];
        }
        free(tmp);
    }
    ct = cosf(theta);
    st = sinf(theta);

#define OCR_PERP(bx)  do {                                   \
        float _cx, _cy;                                      \
        box_center((bx), &_cx, &_cy);                        \
        perp = -st * _cx + ct * _cy;                         \
        along =  ct * _cx + st * _cy;                        \
    } while (0)

    /* insertion sort on (perp, along) */
    for (i = 1; i < n; i++) {
        ocr_box key = b[i];
        float perp, along, kp, ka, pp, pa;
        OCR_PERP(&key); kp = perp; ka = along;
        j = i;
        while (j > 0) {
            OCR_PERP(&b[j - 1]); pp = perp; pa = along;
            if (pp < kp - 0.001f) break;
            if (fabsf(pp - kp) < 0.001f && pa <= ka) break;
            b[j] = b[j - 1];
            j--;
        }
        b[j] = key;
    }

    /* Group into lines and order each along the text direction.  The tolerance
     * is a quarter of the median QUAD height -- the quad's own height, not the
     * axis-aligned box, and median rather than mean.  Two earlier versions of
     * this got it wrong in ways that scrambled order: half the height merged
     * adjacent lines (unclip inflates a 35 px line to ~89 px, which is the whole
     * line pitch), and using the bounding box inflated it further on any tilted
     * page. */
    if (line_tol <= 0.0f) {
        float med = 0.0f;
        float *hs = (float *)malloc(n * sizeof *hs);
        if (hs) {
            float w, h;
            for (i = 0; i < n; i++) { box_dims(&b[i], &w, &h); hs[i] = h; }
            qsort(hs, n, sizeof *hs, cmp_float);
            med = hs[n / 2];
            free(hs);
        }
        line_tol = med * 0.25f;
        if (line_tol < 1.0f) line_tol = 1.0f;
    }

    i = 0;
    while (i < n) {
        float perp, along, ref;
        OCR_PERP(&b[i]); ref = perp;
        j = i + 1;
        while (j < n) {
            OCR_PERP(&b[j]);
            if (perp - ref > line_tol) break;
            j++;
        }
        if (j - i > 1) {
            size_t a, c;
            for (a = i; a + 1 < j; a++)
                for (c = a + 1; c < j; c++) {
                    float aa, ca;
                    OCR_PERP(&b[a]); aa = along;
                    OCR_PERP(&b[c]); ca = along;
                    if (ca < aa) { ocr_box t = b[a]; b[a] = b[c]; b[c] = t; }
                }
        }
        i = j;
    }
#undef OCR_PERP
}
