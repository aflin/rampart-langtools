// spm_c_wrapper.h — minimal C API shim for SentencePiece (Apache-2.0 compatible)
#ifndef SPM_C_WRAPPER_H_
#define SPM_C_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spm_processor spm_processor_t;

// Create / destroy
spm_processor_t* spm_create(void);
void spm_free(spm_processor_t* p);

// Load a .model file; returns 0 on success
int spm_load(spm_processor_t* p, const char* model_path);

// Encode to ids; caller must free(*out_ids) with spm_free_array
int spm_encode_as_ids(spm_processor_t* p, const char* text,
                      const int32_t** out_ids, size_t* out_len);

// Encode to pieces; caller must free(*out_pieces) with spm_free_pieces
int spm_encode_as_pieces(spm_processor_t* p, const char* text,
                         const char*** out_pieces, size_t* out_len);

// Free arrays returned by encode functions
void spm_free_array(const void* ptr);
void spm_free_pieces(const char** pieces, size_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPM_C_WRAPPER_H_
