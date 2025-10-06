#include "rampart-langtools.h"


#include <c_api/index_io_c.h>   // faiss_read_index_fname
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>
#include <faiss/IndexIDMap.h>          // IndexIDMap, IndexIDMap2
#include <faiss/IndexRefine.h>         // IndexRefineFlat
#include <faiss/IndexPreTransform.h>   // IndexPreTransform
#include <faiss/IndexShards.h>         // IndexShards (optional)


#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
//#include <faiss/index_io_binary.h>

#include <memory>

/* unwrap common wrappers so we can see the inner concrete index */
static const faiss::Index* unwrap_wrappers(const faiss::Index* idx, int *mapped) {
    using namespace faiss;
    *mapped=0;
    for (int i = 0; i < 8 && idx; ++i) {
        if (auto m  = dynamic_cast<const IndexIDMap*>(idx))        { idx = m->index; *mapped = 1; continue; }
        if (auto m2 = dynamic_cast<const IndexIDMap2*>(idx))       { idx = m2->index;*mapped = 2; continue; }
        if (auto rf = dynamic_cast<const IndexRefineFlat*>(idx))   { idx = rf->base_index; continue; }
        if (auto pt = dynamic_cast<const IndexPreTransform*>(idx)) { idx = pt->index; continue; }
        /*
        if (auto px = dynamic_cast<const IndexProxy*>(idx)) {
            if (px->proxies.size() == 1 && px->proxies[0]) { idx = px->proxies[0]; continue; }
        }
        */
        if (dynamic_cast<const IndexShards*>(idx)) break; /* container, stop */
        break;
    }
    return idx;
}

/* map dense index to enum; also fill pqM/pqBits when PQ-based (else leave 0) */
static FaissIndexType map_dense_index(const faiss::Index* idx, int* pqM, int* pqBits) {
    using namespace faiss;
    if (pqM) *pqM = 0;
    if (pqBits) *pqBits = 0;


    // Flat family
    if (dynamic_cast<const IndexFlatL2*>(idx)) return FAISS_INDEX_FLAT;
    if (dynamic_cast<const IndexFlatIP*>(idx)) return FAISS_INDEX_FLAT;
    if (dynamic_cast<const IndexFlat*>(idx))   return FAISS_INDEX_FLAT;

    // IVF families
    if (dynamic_cast<const IndexIVFFlat*>(idx))         return FAISS_INDEX_IVFFLAT;
    if (dynamic_cast<const IndexScalarQuantizer*>(idx)) return FAISS_INDEX_IVFSCALAR;

    // PQ-bearing families: fill pqM/pqBits
    if (auto p = dynamic_cast<const IndexIVFPQ*>(idx)) {
        if (pqM)   *pqM   = (int)p->pq.M;
        if (pqBits)*pqBits= (p->pq.M > 0 ? (int)(p->pq.code_size * 8 / p->pq.M) : 0);
        return FAISS_INDEX_IVFPQ;
    }
    if (auto p = dynamic_cast<const IndexPQ*>(idx)) {
        if (pqM)   *pqM   = (int)p->pq.M;
        if (pqBits)*pqBits= (p->pq.M > 0 ? (int)(p->pq.code_size * 8 / p->pq.M) : 0);
        return FAISS_INDEX_PQ;
    }

    // HNSW variants
    if (dynamic_cast<const faiss::IndexHNSWFlat*>(idx)) {
        return FAISS_INDEX_HNSWFLAT;
    }

    if (auto hpq = dynamic_cast<const faiss::IndexHNSWPQ*>(idx)) {
        // Extract PQ params via the underlying storage, which is an IndexPQ
        if (pqM || pqBits) {
            if (auto storage_pq = dynamic_cast<const faiss::IndexPQ*>(hpq->storage)) {
                int M = (int)storage_pq->pq.M;
                int code_size = (int)storage_pq->pq.code_size; // bytes per vector
                if (pqM)   *pqM   = M;
                if (pqBits)*pqBits= (M > 0) ? (code_size * 8 / M) : 0;
            } else {
                // Unexpected, but fail-safe to zeros
                if (pqM)   *pqM   = 0;
                if (pqBits)*pqBits= 0;
            }
        }
        return FAISS_INDEX_HNSWPQ;
    }

    if (dynamic_cast<const faiss::IndexHNSWSQ*>(idx)) {
        return FAISS_INDEX_HNSWSQ;
    }
    // Others
    if (dynamic_cast<const IndexLSH*>(idx))     return FAISS_INDEX_LSH;

    // Containers as their own types (optional)
    //if (dynamic_cast<const IndexShards*>(raw))  return FAISS_INDEX_SHARD;
    //if (dynamic_cast<const IndexProxy*>(raw))   return FAISS_INDEX_PROXY;

    return FAISS_INDEX_UNKNOWN;
}
/*
static FaissIndexType map_binary_index(const faiss::IndexBinary* bidx) {
    using namespace faiss;
    if (dynamic_cast<const IndexBinaryIVF*>(bidx))  return FAISS_INDEX_BIN_IVFFLAT;
    if (dynamic_cast<const IndexBinaryFlat*>(bidx)) return FAISS_INDEX_BIN_FLAT;
    return FAISS_INDEX_UNKNOWN;
}
*/
/* CHANGED: now also sets pqM/pqBits (0 when not PQ-based) */
extern "C"
FaissIndexType faiss_detect_type(FaissIndex* c_idx, int* pqM, int* pqBits, int* mapped)
{
    if (pqM) *pqM = 0;
    if (pqBits) *pqBits = 0;

    if (!c_idx) return FAISS_INDEX_UNKNOWN;


    const faiss::Index* idx = reinterpret_cast<const faiss::Index*>(c_idx);

    // Detect via C++ (dense first)
    try {
            const faiss::Index* idx_unwrapped = unwrap_wrappers(idx, mapped);
            return map_dense_index(idx_unwrapped, pqM, pqBits);
    } catch (...) {
        /* fall through to binary */
    }
/*
    // Try binary
    try {
        return map_binary_index(idx);
    } catch (...) {
        // give up
    }
*/
    return FAISS_INDEX_UNKNOWN;
}
