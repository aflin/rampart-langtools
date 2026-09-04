/* ocr-layout.h -- document layout analysis (PP-DocLayoutV3) for rampart-ocr.
 *
 * WHY THIS EXISTS.  Detection gives text lines; it says nothing about how the
 * page is organised, so lines are ordered top-to-bottom and a multi-column page
 * is read ACROSS instead of down.  Measured on a 1918 newspaper front page:
 * character error 0.5974 while token-level F1 was 0.99 -- every word right, the
 * sequence wrong.  A layout model fixes the sequence, and hands us region
 * labels (table, figure, header) that a caller can act on.
 *
 * WHY A MODEL AND NOT GEOMETRY.  Recursive XY-cut over the detection boxes was
 * the obvious cheap alternative, but PP-StructureV3 already runs XY-cut over
 * its OWN regions and refines it, so a hand-rolled version would be the weaker
 * half of what this model already does -- and over raw unlabelled lines, which
 * is the harder input.  This model also emits a reading order directly.
 */
#ifndef OCR_LAYOUT_H
#define OCR_LAYOUT_H

#include <stddef.h>
#include "ocr-image.h"
#include "ocr-det.h"     /* ocr_box: the line quads this orders */

#ifdef __cplusplus
extern "C" {
#endif

/* PP-DocLayoutV3 is trained at a fixed 800x800 with the aspect ratio NOT
 * preserved (config.json: Resize keep_ratio false).  A page is squashed to the
 * square, which is what the model saw in training. */
#define OCR_LAYOUT_SIDE 800

/* The 25 classes, in the model's own label order (config.json label_list). */
extern const char *const ocr_layout_labels[];
#define OCR_LAYOUT_NCLASS 25

typedef struct {
    int   label;           /* index into ocr_layout_labels, -1 if out of range */
    float score;
    float x0, y0, x1, y1;  /* ORIGINAL image pixels: the model applies
                            * scale_factor internally, so no mapping back */
    int   order;           /* the model's own reading-order key; sort ascending */
} ocr_region;

/* Build the network input: resize to 800x800 and scale to 0..1, CHW RGB.
 *
 * NORMALISATION.  Only a 1/255 scale -- no ImageNet mean/std.  config.json says
 * norm_type "none" with mean 0 / std 1, and the conversion's own example script
 * contradicts it by applying ImageNet values.  Settled by measurement on a real
 * page: 1/255 scored 0.919 on the region ImageNet scored 0.842 on, with better
 * labels, and raw 0..255 detected nothing at all.  The config is right.
 *
 * `dst` must hold 3 * 800 * 800 floats.  Returns 0, or -1 with err[] filled. */
int ocr_layout_preprocess(const ocr_image *src, float *dst, char *err, size_t errlen);

/* Decode the model's detection output into regions above `thresh`, sorted by
 * reading order.  `det` is the (rows x 7) tensor: label, score, x0, y0, x1, y1,
 * order.  Returns 0 and sets *out (malloc'd; free with free()) and *n_out; -1
 * on allocation failure.  Zero regions is a success with *n_out == 0. */
int ocr_layout_decode(const float *det, size_t rows, size_t cols, float thresh,
                      ocr_region **out, size_t *n_out);

/* Reading-order permutation for `n` text-line quads, given `nr` layout regions.
 *
 * Each line is assigned to the MOST SPECIFIC region containing it (smallest
 * area, not largest overlap: this model labels a columned body a "table"
 * spanning the whole page, and by absolute overlap that container ties with
 * every line's own text box and swallows the reading order); a line overlapping
 * none
 * (marginal notes, a line the layout model missed) is assigned to the nearest
 * region by centre distance, so it stays where it belongs rather than being
 * dumped at the end.  Lines are then ordered by their region's reading order,
 * ties broken by the order they already had -- which is correct WITHIN a block,
 * since that is exactly the case top-to-bottom already handles.
 *
 * Writes `n` indices into `order_out` (caller-allocated).  With nr == 0 this is
 * the identity, so a page the model found nothing on is left untouched. */
void ocr_layout_order(const ocr_box *b, size_t n,
                      const ocr_region *r, size_t nr,
                      size_t *order_out);

/* Does this page look like it has more than one column?
 *
 * The cheap trigger for running layout at all.  It sweeps x and measures the
 * fraction of lines STRADDLING each split: in one column every full line
 * straddles the middle, while a gutter is a place almost nothing crosses.
 * That is a far weaker question than "what is the reading order", which is why
 * it is safe to answer geometrically.
 *
 * An earlier version demanded a band that NOTHING crossed.  Real newspapers
 * never satisfy that -- a headline spanning two of six columns crosses that
 * gutter -- and the trigger missed every genuine broadsheet.
 *
 * Being wrong is cheap in both directions, which is why the thresholds lean
 * towards firing: measured against a PDF's own text layer, a page that fires
 * wrongly cost 0.006 character error, while a page I had mislabelled as
 * single-column gained 0.166 by firing.
 *
 * `min_gap` is retained for the ABI and ignored: the test is a ratio now, not a
 * width.  `cross_out` / `side_out` (both may be NULL) report the split the
 * decision rests on -- the crossing fraction and the weaker side's share -- so
 * a caller, or a tuning run, can see WHY the answer was no. */
int ocr_layout_looks_multicolumn(const ocr_box *b, size_t n, float min_gap,
                                 float *cross_out, float *side_out);

#ifdef __cplusplus
}
#endif

#endif /* OCR_LAYOUT_H */
