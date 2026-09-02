/* ocr-det.h -- DB (Differentiable Binarization) detection postprocessing.
 *
 * Turns the detector's probability map into quadrilaterals in ORIGINAL image
 * coordinates.  The pipeline follows PaddleOCR/RapidOCR exactly:
 *
 *   binarize at `thresh`
 *     -> connected components (8-connected, as cv2.findContours treats fg)
 *     -> convex hull -> minimum-area rectangle (rotating calipers)
 *     -> drop rects whose short side < min_size
 *     -> score = mean probability inside the quad; drop if < box_thresh
 *     -> unclip (expand) by unclip_ratio
 *     -> map back to original image coordinates
 *
 * The OpenCV calls that reference implementation leans on (connectedComponents,
 * minAreaRect, fillPoly, boxPoints) are reimplemented here in plain C; the
 * surrounding logic and every constant come from the RapidOCR source, because
 * these are precisely the small numeric decisions where reimplementing "from the
 * paper" produces something that quietly disagrees with published results.
 */
#ifndef OCR_DET_H
#define OCR_DET_H

#include <stddef.h>
#include "ocr-image.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One detected text quad, corners clockwise from the top-left, in ORIGINAL
 * image pixel coordinates. */
typedef struct {
    float pt[4][2];
    float score;
} ocr_box;

typedef struct {
    float thresh;        /* probability -> foreground        (PP-OCR: 0.3) */
    float box_thresh;    /* mean-probability floor per box   (PP-OCR: 0.5) */
    float unclip_ratio;  /* box expansion                    (PP-OCR: 1.6) */
    int   min_size;      /* drop rects with a shorter side   (PP-OCR: 3)   */
    int   max_boxes;     /* candidate cap                    (PP-OCR: 1000)*/
} ocr_det_opts;

/* Fill in the PP-OCR defaults. */
void ocr_det_opts_default(ocr_det_opts *o);

/* Extract boxes from a [mh][mw] probability map.  `sc` carries the per-axis
 * scale back to original coordinates (from ocr_det_plan).  On success returns 0
 * and sets *out (malloc'd, free with free()) and *n_out; returns -1 on oom.
 * Zero boxes is a success with *n_out == 0. */
int ocr_det_boxes(const float *prob, int mw, int mh,
                  const ocr_det_scale *sc, const ocr_det_opts *opt,
                  ocr_box **out, size_t *n_out);

/* Sort boxes into reading order.  Handles a rotated page: the median text
 * direction is taken from the boxes themselves and the sort is done in a frame
 * rotated to match, so this is correct at any angle rather than only for small
 * skew.  `line_tol` is the across-line slack in pixels; <= 0 derives one from
 * the median quad height. */
void ocr_det_sort_boxes(ocr_box *b, size_t n, float line_tol);

/* Fraction of boxes taller than they are wide (using the quads' own edges, not
 * bounding boxes).  Text lines are wide, so a high value means the PAGE is a
 * quarter-turn out: measured 0.01 upright versus 1.00 at both 90 and 270
 * degrees, which makes it a clean trigger for re-running detection on a rotated
 * copy.  It cannot distinguish 90 from 270 -- that ambiguity is 180 degrees,
 * which is exactly what the angle classifier resolves. */
float ocr_det_tall_fraction(const ocr_box *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* OCR_DET_H */
