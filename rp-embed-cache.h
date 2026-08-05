/* rp-embed-cache.h -- content-keyed LRU cache of document embed results.
 *
 * Shared at the SOURCE level by rampart-onnx and rampart-llamacpp (compile
 * rp-embed-cache.c into each, like rp-chunker.c).  One cache lives on each
 * embed handle (which IS a (model, tokenizer, opts) identity), so entries
 * are automatically scoped to the model + options that produced them.
 *
 * Purpose: a document's per-chunk vectors, their byte spans, its avgVec,
 * and its coherence are all produced by ONE model run (rp_*_embed_doc).
 * The SQL scalars chunkembed()/chunkavg()/chunkcoherence()/embed() each
 * want a different slice of that result, and are frequently called on the
 * SAME text within one statement (e.g.
 * `insert ... values (chunkavg(?text), chunkembed(?text))`)
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

/* Per-chunk byte span (+ decoded token count).  Layout-identical to the
 * engines' span types (ll_chunk_span / rp_onnx_embed_span / the exported
 * rp_*_embed_chunk_span), so callers cast pointers rather than copy. */
typedef struct { size_t start, end, n_tokens; } rp_doccache_span;

/* Enforce that layout-identical claim.  The engines cast span pointers to
 * and from rp_doccache_span instead of copying field by field, so a field
 * added, removed or reordered in ANY of these types would silently
 * reinterpret byte offsets -- and those offsets are stored in chunkembed()
 * values and drive abstract()'s snippet extraction, so the damage would be
 * invisible until someone noticed snippets had gone wrong.  Invoke once per
 * engine span type, at file scope, immediately after that type is defined.
 *
 * NOTE: rampart-sql.c carries its own copy of this layout (rp_embed_span_t)
 * on the far side of a dlsym boundary; nothing here can check that one. */
#define RP_DOCCACHE_ASSERT_SPAN_LAYOUT(T)                                      \
    _Static_assert(sizeof(T) == sizeof(rp_doccache_span),                      \
                   #T " size must match rp_doccache_span");                    \
    _Static_assert(offsetof(T, start) == offsetof(rp_doccache_span, start),    \
                   #T ".start must match rp_doccache_span.start");             \
    _Static_assert(offsetof(T, end) == offsetof(rp_doccache_span, end),        \
                   #T ".end must match rp_doccache_span.end");                 \
    _Static_assert(offsetof(T, n_tokens) == offsetof(rp_doccache_span, n_tokens), \
                   #T ".n_tokens must match rp_doccache_span.n_tokens")

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
 * (caller frees out_vecs / out_avg / out_chunks) and returns 1.  On a
 * miss returns 0 and touches nothing.  out_vecs is k*dim floats, out_avg
 * is dim floats, out_chunks is k spans, out_dim/out_k/out_coh are
 * scalars. */
int rp_doccache_get(rp_doccache_t *c, const char *text, size_t tlen,
                       const char *prefix, size_t plen,
                       float **out_vecs, size_t *out_k, int *out_dim,
                       float **out_avg, float *out_coh,
                       rp_doccache_span **out_chunks);

/* Insert (or refresh) text -> {vecs[k*dim], avg[dim], coh, chunks[k]}.
 * Copies everything it stores; the caller retains ownership of its
 * buffers.  avg/chunks may be NULL (then a cache_get requesting that
 * part treats the entry as a miss and recomputes -- but callers always
 * store both). */
void rp_doccache_put(rp_doccache_t *c, const char *text, size_t tlen,
                        const char *prefix, size_t plen,
                        const float *vecs, size_t k, int dim,
                        const float *avg, float coh,
                        const rp_doccache_span *chunks);

#ifdef __cplusplus
}
#endif

#endif /* RP_DOCCACHE_H */
