#ifndef RAMPART_LANGTOOLS_H
#define RAMPART_LANGTOOLS_H

typedef enum {
    FAISS_INDEX_FLAT = 0,           /* Exact search */
    FAISS_INDEX_PQ,                 /* Product Quantization index */
    FAISS_INDEX_IVFFLAT,            /* Inverted file with flat vectors */
    FAISS_INDEX_IVFPQ,              /* Inverted file with product quantization */
    FAISS_INDEX_IVFSCALAR,          /* IVF with scalar quantizer */
    FAISS_INDEX_IVFRSQ,             /* IVF with residual scalar quantizer */
    FAISS_INDEX_HNSWFLAT,           /* HNSW graph + flat vectors */
    FAISS_INDEX_HNSWPQ,             /* HNSW + product quantization */
    FAISS_INDEX_HNSWSQ,             /* HNSW + scalar quantizer */
    FAISS_INDEX_LSH,                /* Locality Sensitive Hashing */
    FAISS_INDEX_BIN_FLAT,           /* Flat binary vectors */
    FAISS_INDEX_BIN_IVFFLAT,        /* IVF for binary vectors */
    FAISS_INDEX_SHARD,              /* Sharded index (distributed across nodes) */
    FAISS_INDEX_PROXY,              /* Proxy to another index */
    FAISS_INDEX_UNKNOWN             /* Fallback / uninitialized */
} FaissIndexType;

#ifdef RP_FAISS_SIMPLE_OPEN

#ifdef __cplusplus
extern "C" {
#endif


typedef struct FaissIndex_H        FaissIndex;
typedef struct FaissIndexBinary_H  FaissIndexBinary;

FaissIndexType faiss_load_and_detect_type(const char* filename, FaissIndex** idx, int* pqM, int* pqBits);

#ifdef __cplusplus
}
#endif

#endif // RP_FAISS_SIMPLE_OPEN

#endif /* RAMPART_LANGTOOLS_H */