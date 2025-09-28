// spm_c_wrapper.cc — minimal C API shim for SentencePiece
#include "spm_c_wrapper.h"
#include "../src/sentencepiece_processor.h"
#include <vector>
#include <string>
#include <cstdlib>

struct spm_processor {
  sentencepiece::SentencePieceProcessor proc;
};

extern "C" {

spm_processor_t* spm_create(void) {
  return new (std::nothrow) spm_processor_t();
}

void spm_free(spm_processor_t* p) {
  delete p;
}

int spm_load(spm_processor_t* p, const char* model_path) {
  if (!p || !model_path) return 1;
  auto status = p->proc.Load(model_path);
  return status.ok() ? 0 : 1;
}

int spm_encode_as_ids(spm_processor_t* p, const char* text,
                      const int32_t** out_ids, size_t* out_len) {
  if (!p || !text || !out_ids || !out_len) return 1;
  std::vector<int> ids;
  auto status = p->proc.Encode(text, &ids);
  if (!status.ok()) return 1;
  size_t n = ids.size();
  int32_t* buf = static_cast<int32_t*>(std::malloc(n * sizeof(int32_t)));
  if (!buf && n) return 1;
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int32_t>(ids[i]);
  *out_ids = buf;
  *out_len = n;
  return 0;
}

int spm_encode_as_pieces(spm_processor_t* p, const char* text,
                         const char*** out_pieces, size_t* out_len) {
  if (!p || !text || !out_pieces || !out_len) return 1;
  std::vector<std::string> pieces;
  auto status = p->proc.Encode(text, &pieces);
  if (!status.ok()) return 1;
  size_t n = pieces.size();
  const char** arr = static_cast<const char**>(std::malloc(n * sizeof(char*)));
  if (!arr && n) return 1;
  for (size_t i = 0; i < n; ++i) {
    char* s = static_cast<char*>(std::malloc(pieces[i].size() + 1));
    if (!s) { // cleanup partial
      for (size_t k = 0; k < i; ++k) std::free((void*)arr[k]);
      std::free(arr);
      return 1;
    }
    std::memcpy(s, pieces[i].c_str(), pieces[i].size() + 1);
    arr[i] = s;
  }
  *out_pieces = arr;
  *out_len = n;
  return 0;
}

void spm_free_array(const void* ptr) {
  std::free((void*)ptr);
}

void spm_free_pieces(const char** pieces, size_t len) {
  if (!pieces) return;
  for (size_t i = 0; i < len; ++i) std::free((void*)pieces[i]);
  std::free((void*)pieces);
}

} // extern "C"
