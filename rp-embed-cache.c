/* rp-embed-cache.c -- see rp-embed-cache.h. */
#include "rp-embed-cache.h"
#include <stdlib.h>
#include <string.h>

struct rp_doccache_entry_s {
    char     *text;         /* malloc'd (tlen bytes + NUL); for memcmp confirm */
    size_t    tlen;
    char     *prefix;       /* malloc'd (plen bytes + NUL); NULL when plen==0 */
    size_t    plen;
    uint64_t  hash;
    float    *vecs;         /* k*dim, malloc'd */
    size_t    k;
    int       dim;
    float    *avg;          /* dim, malloc'd (may be NULL) */
    float     coh;
    rp_doccache_entry_t *lru_prev, *lru_next;
};

static uint64_t fnv1a(const char *p, size_t n)
{
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (uint8_t)p[i]) * 1099511628211ULL;
    return h;
}

void rp_doccache_init(rp_doccache_t *c, size_t cap)
{
    if (!c) return;
    /* Re-init keeps init's 0-means-default semantics (set_cap's 0 means
     * DISABLE; only an explicit set_cap(0) may disable). */
    if (c->initialized) { rp_doccache_set_cap(c, cap ? cap : RP_DOCCACHE_DEFAULT_CAP); return; }
    c->head = c->tail = NULL;
    c->n = 0;
    c->cap = cap ? cap : RP_DOCCACHE_DEFAULT_CAP;
    pthread_mutex_init(&c->mtx, NULL);
    c->initialized = 1;
}

/* mtx held */
static void entry_unlink(rp_doccache_t *c, rp_doccache_entry_t *e)
{
    if (e->lru_prev) e->lru_prev->lru_next = e->lru_next;
    else             c->head = e->lru_next;
    if (e->lru_next) e->lru_next->lru_prev = e->lru_prev;
    else             c->tail = e->lru_prev;
    e->lru_prev = e->lru_next = NULL;
}

/* mtx held */
static void entry_push_front(rp_doccache_t *c, rp_doccache_entry_t *e)
{
    e->lru_prev = NULL;
    e->lru_next = c->head;
    if (c->head) c->head->lru_prev = e;
    else         c->tail = e;
    c->head = e;
}

static void entry_free(rp_doccache_entry_t *e)
{
    if (!e) return;
    free(e->text);
    free(e->prefix);
    free(e->vecs);
    free(e->avg);
    free(e);
}

/* mtx held */
static void evict_tail(rp_doccache_t *c)
{
    rp_doccache_entry_t *e = c->tail;
    if (!e) return;
    entry_unlink(c, e);
    c->n--;
    entry_free(e);
}

void rp_doccache_set_cap(rp_doccache_t *c, size_t cap)
{
    if (!c || !c->initialized) return;
    pthread_mutex_lock(&c->mtx);
    /* Explicit cap wins verbatim -- cap == 0 DISABLES the cache (get/put
     * early-return on cap==0).  Only rp_doccache_init maps 0 to the
     * default; a caller who sets 0 means "off". */
    c->cap = cap;
    while (c->n > c->cap) evict_tail(c);
    pthread_mutex_unlock(&c->mtx);
}

void rp_doccache_destroy(rp_doccache_t *c)
{
    if (!c || !c->initialized) return;
    pthread_mutex_lock(&c->mtx);
    rp_doccache_entry_t *e = c->head;
    while (e) { rp_doccache_entry_t *nx = e->lru_next; entry_free(e); e = nx; }
    c->head = c->tail = NULL;
    c->n = 0;
    pthread_mutex_unlock(&c->mtx);
    pthread_mutex_destroy(&c->mtx);
    c->initialized = 0;
}

/* hash covers prefix + text (lengths separate the fields) */
static uint64_t key_hash(const char *text, size_t tlen,
                         const char *prefix, size_t plen)
{
    uint64_t h = fnv1a(text, tlen);
    if (plen) h = (h ^ fnv1a(prefix, plen)) * 1099511628211ULL + plen;
    return h;
}

/* mtx held */
static rp_doccache_entry_t *find_locked(rp_doccache_t *c, uint64_t h,
                                           const char *text, size_t tlen,
                                           const char *prefix, size_t plen)
{
    for (rp_doccache_entry_t *e = c->head; e; e = e->lru_next)
        if (e->hash == h && e->tlen == tlen && e->plen == plen
            && memcmp(e->text, text, tlen) == 0
            && (plen == 0 || memcmp(e->prefix, prefix, plen) == 0))
            return e;
    return NULL;
}

int rp_doccache_get(rp_doccache_t *c, const char *text, size_t tlen,
                       const char *prefix, size_t plen,
                       float **out_vecs, size_t *out_k, int *out_dim,
                       float **out_avg, float *out_coh)
{
    if (out_vecs) *out_vecs = NULL;
    if (out_avg)  *out_avg  = NULL;
    if (!c || !c->initialized || c->cap == 0 || !text || tlen == 0) return 0;
    if (!prefix) plen = 0;

    uint64_t h = key_hash(text, tlen, prefix, plen);
    pthread_mutex_lock(&c->mtx);
    rp_doccache_entry_t *e = find_locked(c, h, text, tlen, prefix, plen);
    if (!e) { pthread_mutex_unlock(&c->mtx); return 0; }
    /* If the caller wants avg but this entry has none, treat as a miss so
     * it recomputes with avg (shouldn't happen -- puts always store avg). */
    if (out_avg && !e->avg) { pthread_mutex_unlock(&c->mtx); return 0; }

    float  *vc = NULL, *av = NULL;
    if (out_vecs) {
        size_t nfloat = e->k * (size_t)e->dim;
        vc = (float *)malloc(nfloat * sizeof(float));
        if (!vc) { pthread_mutex_unlock(&c->mtx); return 0; }
        memcpy(vc, e->vecs, nfloat * sizeof(float));
    }
    if (out_avg) {
        av = (float *)malloc((size_t)e->dim * sizeof(float));
        if (!av) { free(vc); pthread_mutex_unlock(&c->mtx); return 0; }
        memcpy(av, e->avg, (size_t)e->dim * sizeof(float));
    }
    if (out_vecs) *out_vecs = vc;
    if (out_k)    *out_k    = e->k;
    if (out_dim)  *out_dim  = e->dim;
    if (out_avg)  *out_avg  = av;
    if (out_coh)  *out_coh  = e->coh;

    /* Promote to MRU. */
    if (c->head != e) { entry_unlink(c, e); entry_push_front(c, e); }
    pthread_mutex_unlock(&c->mtx);
    return 1;
}

void rp_doccache_put(rp_doccache_t *c, const char *text, size_t tlen,
                        const char *prefix, size_t plen,
                        const float *vecs, size_t k, int dim,
                        const float *avg, float coh)
{
    if (!c || !c->initialized || c->cap == 0 || !text || tlen == 0) return;
    if (!vecs || k == 0 || dim <= 0) return;
    if (!prefix) plen = 0;

    uint64_t h = key_hash(text, tlen, prefix, plen);

    /* Build the entry OUTSIDE the lock (allocations + copies). */
    rp_doccache_entry_t *e = (rp_doccache_entry_t *)calloc(1, sizeof *e);
    if (!e) return;
    e->text = (char *)malloc(tlen + 1);
    size_t nfloat = k * (size_t)dim;
    e->vecs = (float *)malloc(nfloat * sizeof(float));
    e->avg  = avg ? (float *)malloc((size_t)dim * sizeof(float)) : NULL;
    if (plen) {
        e->prefix = (char *)malloc(plen + 1);
        if (!e->prefix) { entry_free(e); return; }
        memcpy(e->prefix, prefix, plen); e->prefix[plen] = '\0';
        e->plen = plen;
    }
    if (!e->text || !e->vecs || (avg && !e->avg)) { entry_free(e); return; }
    memcpy(e->text, text, tlen); e->text[tlen] = '\0';
    e->tlen = tlen; e->hash = h;
    memcpy(e->vecs, vecs, nfloat * sizeof(float));
    if (avg) memcpy(e->avg, avg, (size_t)dim * sizeof(float));
    e->k = k; e->dim = dim; e->coh = coh;

    pthread_mutex_lock(&c->mtx);
    /* Dedup: another thread may have inserted the same text meanwhile.
     * Exception: if the existing entry lacks avg and ours has it, replace
     * it -- otherwise an avg-less entry would pin (get with out_avg treats
     * it as a miss, and this dedup would discard every recompute's result,
     * recomputing forever until eviction). */
    rp_doccache_entry_t *dup = find_locked(c, h, text, tlen, prefix, plen);
    if (dup) {
        if (!dup->avg && e->avg) {
            entry_unlink(c, dup);
            c->n--;
            entry_free(dup);
            /* fall through: insert the richer entry */
        } else {
            if (c->head != dup) { entry_unlink(c, dup); entry_push_front(c, dup); }
            pthread_mutex_unlock(&c->mtx);
            entry_free(e);
            return;
        }
    }
    entry_push_front(c, e);
    c->n++;
    while (c->n > c->cap) evict_tail(c);
    pthread_mutex_unlock(&c->mtx);
}
