/* ocr-rec.h -- text recognition (CRNN/SVTR + CTC) and angle classification.
 *
 * Both models consume the same thing -- an upright text-line crop, height-
 * normalized, scaled to [-1,1] -- so they share the preprocessing here.  The
 * recognizer's width is variable and the classifier's is fixed; that is the
 * only real difference.
 */
#ifndef OCR_REC_H
#define OCR_REC_H

#include <stddef.h>
#include "ocr-image.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- character dictionary ------------------------------------------------
 * PP-OCR's class list is NOT the dictionary file: it is
 *
 *     index 0            CTC blank
 *     index 1..N         the N lines of the dictionary
 *     index N+1          a literal space
 *
 * which is why a 18383-line dictionary pairs with an 18385-class recognizer.
 * Getting this wrong shifts every decoded character by one, so the loader
 * checks the count against the model and says so if they disagree. */
typedef struct {
    char  **item;     /* item[0] = "" (blank); item[n_items-1] = " " */
    size_t  n_items;
} ocr_dict;

/* Load a PP-OCR dictionary file (one character per line, CR/LF tolerated --
 * the shipped file is CRLF, and an unstripped \r would ride along on every
 * decoded character).  0 on success. */
int  ocr_dict_load(const char *path, ocr_dict *d, char *err, size_t errlen);
void ocr_dict_free(ocr_dict *d);

/* ---- preprocessing -------------------------------------------------------
 * Pack `n` crops into one NCHW float batch of shape [n, 3, height, width].
 * Each crop is aspect-preserved to `height`, then right-padded with zeros to
 * the common `width` (PP-OCR pads rather than stretches: stretching a short
 * word to the batch width distorts glyph shapes and costs accuracy).
 * `dst` must hold n*3*height*width floats. */
int ocr_pack_crops(const ocr_image *crops, size_t n, int height, int width, float *dst);

/* Width this batch should use: max over crops of round(height * w/h), clamped
 * to [min_width, max_width] and rounded up to a multiple of 8. */
int ocr_batch_width(const ocr_image *crops, size_t n, int height,
                    int min_width, int max_width);

/* ---- CTC decode ----------------------------------------------------------
 * Greedy best-path decode of one [T, C] logit/probability slice: argmax per
 * timestep, collapse runs of the same index, drop blanks.  Writes UTF-8 into
 * `out` and reports the mean winning probability over the kept timesteps,
 * which is the per-line confidence PP-OCR reports. */
int ocr_ctc_decode(const float *logits, int T, int C, const ocr_dict *d,
                   char *out, size_t outlen, float *score);

#ifdef __cplusplus
}
#endif

#endif /* OCR_REC_H */
