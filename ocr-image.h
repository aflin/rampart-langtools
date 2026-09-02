/* ocr-image.h -- image decode, resize and tensor packing for rampart-ocr.
 *
 * Everything here is plain C over an interleaved 8-bit RGB buffer.  The
 * RapidOCR C++ reference does this work with OpenCV (cv::imread, cv::resize,
 * cv::warpPerspective); vendoring OpenCV for four primitives would be wildly
 * disproportionate, so they are reimplemented here and the surrounding logic
 * and constants are taken from the RapidOCR source.
 */
#ifndef OCR_IMAGE_H
#define OCR_IMAGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved RGB, 3 bytes per pixel, row-major, no row padding. */
typedef struct {
    unsigned char *px;
    int            w, h;
} ocr_image;

/* Decode from a file / from memory.  Always yields 3-channel RGB regardless of
 * what the source had (stb handles gray, RGBA, palette).  0 on success; -1 with
 * err[] filled otherwise.  Free with ocr_image_free().
 *
 * TIFF is handled by libtiff rather than stb, which has never supported it.
 * `page` selects a directory (0-based) of a multi-page TIFF and is ignored for
 * every other format; `npages` (may be NULL) receives the page count. */
int  ocr_image_load_file(const char *path, int page, ocr_image *out, int *npages, char *err, size_t errlen);
int  ocr_image_load_mem(const void *buf, size_t len, int page, ocr_image *out, int *npages, char *err, size_t errlen);
void ocr_image_free(ocr_image *im);

/* Number of pages: TIFF directory count, or 1 for any other format (and 0 if
 * the input cannot be opened at all).  Cheap -- it does not decode pixels. */
int  ocr_image_page_count(const char *path);
int  ocr_image_page_count_mem(const void *buf, size_t len);

/* Bilinear resize into a freshly allocated image (dst is overwritten, not
 * freed).  0 on success, -1 on bad args / oom. */
int  ocr_image_resize(const ocr_image *src, int dw, int dh, ocr_image *dst);

/* ---- detection preprocessing --------------------------------------------
 * PP-OCR's DB detector wants an NCHW float32 tensor, ImageNet-normalized, whose
 * spatial dims are MULTIPLES OF 32 (the backbone downsamples by 32; a ragged
 * size silently changes the output stride).  The long side is also capped --
 * `limit_side_len`, 960 in RapidOCR's default config -- because cost grows with
 * area and detection quality does not, past that.
 *
 * Fills `dst` (caller-allocated, 3*out_w*out_h floats) and reports the size fed
 * to the network plus the scale factors needed to map boxes BACK to the original
 * image.  sx/sy are separate on purpose: rounding each axis up to a multiple of
 * 32 makes them differ slightly, and using a single average shifts every box. */
typedef struct {
    int   net_w, net_h;   /* what the network actually saw */
    float sx, sy;         /* original = network * s{x,y} */
} ocr_det_scale;

/* How big a tensor det preprocessing will produce for this image (so the caller
 * can allocate).  Returns 0 on success. */
int ocr_det_plan(int src_w, int src_h, int limit_side_len, ocr_det_scale *out);

/* Resize + normalize into `dst` (3 * net_w * net_h floats, NCHW).
 * mean/std are the ImageNet constants PP-OCR trained with. */
int ocr_det_preprocess(const ocr_image *src, int limit_side_len,
                       float *dst, ocr_det_scale *scale);

/* ---- text-line crop ------------------------------------------------------
 * Perspective-warp a detected quad (4 corners, clockwise from top-left, in
 * source pixel coordinates) to an upright image -- RapidOCR's
 * GetRotateCropImage.  Output size is taken from the quad's own edge lengths,
 * so a rotated or slightly skewed line is deskewed rather than merely cropped.
 * A crop that comes out much taller than wide is rotated 90 degrees, which is
 * how vertical text is handed to a recognizer that only reads horizontally.
 * 0 on success; dst is overwritten (not freed) and owns new memory. */
int ocr_crop_quad(const ocr_image *src, const float quad[4][2], ocr_image *dst);

/* Rotate an image 180 degrees in place (angle classification says upside down). */
void ocr_image_rot180(ocr_image *im);

/* Rotate an image a quarter turn (ccw != 0 => counter-clockwise), replacing it.
 * Used to stand a quarter-turned PAGE upright before re-detecting.  0 on
 * success; the image is left untouched on allocation failure. */
int  ocr_image_rot90(ocr_image *im, int ccw);

#ifdef __cplusplus
}
#endif

#endif /* OCR_IMAGE_H */
