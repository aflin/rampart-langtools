/* rp-embed-cache.h -- content-keyed LRU cache of document embed results.
 *
 * Shared at the SOURCE level by rampart-onnx and rampart-llamacpp (compile
 * rp-embed-cache.c into each, like rp-chunker.c).  One cache lives on each
 * embed handle (which IS a (model, tokenizer, opts) identity), so entries
 * are automatically scoped to the model + options that produced them.
 *
 * Purpose: a document's per-chunk vectors, its avgVec, and its coherence
 * are all produced by ONE model run (rp_*_embed_doc).  The SQL scalars
 * chunkembed()/chunkavg()/chunkcoherence()/embed() each want a different
 * slice of that result, and are frequently called on the SAME text within
 * one statement (e.g. `insert ... values (chunkavg(?text), chunkembed(?text))`)
 * or across statements at search time.  Keying on the text content means
 * whichever scalar runs first pays the model cost and the rest are lookups,
 * regardless of call order or which SQL parameter carried the text.
 *
 * Threading: a dedicated mutex (NOT the handle's model/context mutex) guards
 * the cache.  The model run happens OUTSIDE the lock; the lock is held only
 * for the hash lookup + a bounded copy-out / insert, so concurrent embeds of
 * DIFFERENT texts don't serialize.  Two threads racing the SAME text both
 * compute; the second insert dedups.  Because embedding a given text under a
 * given model is deterministic, a hit computed by one thread is valid for any
 * other.
 *
 * Correctness: the key is (FNV-1a hash, byte length) and a full memcmp of the
 * stored text confirms a hash-bucket hit -- a collision can never return the
 * wrong result.
 */
#ifndef RP_DOCCACHE_H
#define RP_DOCCACHE_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rp_doccache_entry_s rp_doccache_entry_t;

typedef struct {
    rp_doccache_entry_t *head;   /* MRU */
    rp_doccache_entry_t *tail;   /* LRU (eviction end) */
    size_t                  n;
    size_t                  cap;    /* max entries; 0 => disabled */
    pthread_mutex_t         mtx;
    int                     initialized;
} rp_doccache_t;

/* Default capacity when nothing sets one.  Runtime-adjustable via
 * rp_doccache_set_cap(), which the modules export as
 * rp_*_embed_set_cache_cap and rampart-sql wires to
 * sql.set({likevCache: N}). */
#define RP_DOCCACHE_DEFAULT_CAP 10

/* Initialize a cache in-place with `cap` entries (0 => use the default,
 * a negative-cast/huge value is clamped by the caller).  Idempotent:
 * a second call on an initialized cache only updates cap. */
void rp_doccache_init(rp_doccache_t *c, size_t cap);

/* Change capacity (evicts LRU entries if shrinking).  cap == 0 DISABLES
 * the cache.  Backs the sql.set({...,likevCache:N}) knob; safe anytime
 * after init. */
void rp_doccache_set_cap(rp_doccache_t *c, size_t cap);

/* Free all entries + the mutex.  (Handles are process-lifetime in v1, so
 * this is mainly for completeness / tests.) */
void rp_doccache_destroy(rp_doccache_t *c);

/* Look up (text[0..tlen), prefix[0..plen)) -- the per-document embed
 * prefix (e.g. the title) is part of the key: the same text under a
 * different prefix embeds differently.  prefix may be NULL/0.  On a hit,
 * fills whichever out-params are non-NULL with FRESH malloc'd copies
 * (caller frees out_vecs / out_avg) and returns 1.  On a miss returns 0
 * and touches nothing.  out_vecs is k*dim floats, out_avg is dim floats,
 * out_dim/out_k/out_coh are scalars. */
int rp_doccache_get(rp_doccache_t *c, const char *text, size_t tlen,
                       const char *prefix, size_t plen,
                       float **out_vecs, size_t *out_k, int *out_dim,
                       float **out_avg, float *out_coh);

/* Insert (or refresh) text -> {vecs[k*dim], avg[dim], coh}.  Copies
 * everything it stores; the caller retains ownership of its buffers.
 * avg may be NULL (then a cache_get with out_avg on this entry misses the
 * avg and returns 0 for the whole call -- but callers always store avg). */
void rp_doccache_put(rp_doccache_t *c, const char *text, size_t tlen,
                        const char *prefix, size_t plen,
                        const float *vecs, size_t k, int dim,
                        const float *avg, float coh);

#ifdef __cplusplus
}
#endif

#endif /* RP_DOCCACHE_H */
