/* rp-chunker.c -- structure-aware text chunking (see rp-chunker.h).
 *
 * Pure C99 + libc; tokenizer-agnostic via the count callback.  Shared at the
 * source level between rampart modules (rampart-onnx today, rampart-llamacpp
 * later): compile this file into each module.
 */
#include "rp-chunker.h"

#include <stdlib.h>
#include <string.h>

/* ---------- growable span list ---------- */
typedef struct {
    rp_chunk_span *v;
    size_t n, cap;
    int oom;
} spanlist;

static void sl_push(spanlist *sl, rp_chunk_span s)
{
    if (sl->oom) return;
    if (sl->n == sl->cap) {
        size_t nc = sl->cap ? sl->cap * 2 : 16;
        rp_chunk_span *nv = realloc(sl->v, nc * sizeof(*nv));
        if (!nv) { sl->oom = 1; return; }
        sl->v = nv; sl->cap = nc;
    }
    sl->v[sl->n++] = s;
}

/* ---------- newline-run scanning ---------- */

/* If text[i] starts a [\r\n]+ run containing at least one '\n', set *run_end
 * (one past the run) and *nl (count of '\n' in it, i.e. the run's semantic
 * level: "\r\n\r\n" == 2) and return 1.  Otherwise return 0. */
static int nl_run(const char *t, size_t end, size_t i, size_t *run_end, int *nl)
{
    size_t j = i;
    int count = 0;
    while (j < end && (t[j] == '\n' || t[j] == '\r')) {
        if (t[j] == '\n') count++;
        j++;
    }
    if (!count) return 0;
    *run_end = j;
    *nl = count;
    return 1;
}

/* Longest newline-run level present in [s,e), capped at 3. */
static int max_level(const char *t, size_t s, size_t e)
{
    int best = 0;
    size_t i = s;
    while (i < e) {
        if (t[i] == '\n' || t[i] == '\r') {
            size_t re; int nl;
            if (nl_run(t, e, i, &re, &nl)) {
                if (nl > best) best = nl;
                if (best >= 3) return 3;
                i = re;
                continue;
            }
        }
        i++;
    }
    return best;
}

/* Trim [*s,*e) to drop leading/trailing whitespace (incl. newlines). */
static void trim(const char *t, size_t *s, size_t *e)
{
    while (*s < *e && (t[*s] == ' ' || t[*s] == '\t' || t[*s] == '\r' || t[*s] == '\n'))
        (*s)++;
    while (*e > *s && (t[*e - 1] == ' ' || t[*e - 1] == '\t' || t[*e - 1] == '\r' || t[*e - 1] == '\n'))
        (*e)--;
}

/* ---------- chunking core ---------- */
typedef struct {
    const char        *t;
    rp_chunk_opts      o;        /* defaults already applied */
    rp_chunk_count_fn  count;
    void              *user;
    spanlist           out;
    int                err;      /* count-callback failure */
} cctx;

static size_t piece_count(cctx *c, size_t s, size_t e)
{
    size_t n = c->count(c->user, c->t + s, e - s);
    if (n == (size_t)-1) c->err = 1;
    return n;
}

/* Emit pending same-level units [u0..un) as chunks.
 *
 * level >= 2 (blank-line paragraphs), default: one chunk per paragraph, but a
 * paragraph under min_tokens is combined FORWARD with following units until
 * the accumulation reaches the floor (Unstructured's combine-under).  With
 * pack_paragraphs, paragraphs are instead greedily packed to win_tokens
 * (LangChain/LlamaIndex style).
 *
 * level 1 (single-newline lines): lines are packing units -- greedily packed
 * to win_tokens with cuts at line boundaries, no overlap.  Hard-wrapped prose
 * self-heals into window-sized chunks; line-structured data keeps line-aligned
 * boundaries.
 *
 * Tail rule (Haystack split_threshold): an under-floor final chunk merges back
 * into the previous chunk from this flush when it fits. */
static void flush_units(cctx *c, const rp_chunk_span *u, size_t un, int level)
{
    if (!un || c->err) return;
    size_t win    = (size_t)c->o.win_tokens;
    size_t minTok = (size_t)c->o.min_tokens;
    int    pack   = (level <= 1) || c->o.pack_paragraphs;
    size_t first_out = c->out.n;

    size_t i = 0;
    while (i < un) {
        rp_chunk_span cur = u[i];
        i++;
        if (pack) {
            while (i < un && cur.n_tokens + u[i].n_tokens <= win) {
                cur.end = u[i].end;
                cur.n_tokens += u[i].n_tokens;
                i++;
            }
        } else if (minTok > 0) {
            while (i < un && cur.n_tokens < minTok
                   && cur.n_tokens + u[i].n_tokens <= win) {
                cur.end = u[i].end;
                cur.n_tokens += u[i].n_tokens;
                i++;
            }
        }
        sl_push(&c->out, cur);
    }

    if (minTok > 0 && c->out.n - first_out >= 2) {
        rp_chunk_span *last = &c->out.v[c->out.n - 1];
        rp_chunk_span *prev = &c->out.v[c->out.n - 2];
        if (last->n_tokens < minTok && prev->n_tokens + last->n_tokens <= win) {
            prev->end = last->end;
            prev->n_tokens += last->n_tokens;
            c->out.n--;
        }
    }
}


/* ---------- sentence-level splitting (opts.sentence_split) ----------
 *
 * A pragmatic subset of UAX #29 sentence segmentation (see rp-chunker.h
 * for the design rationale).  All matching is raw UTF-8 byte-sequence
 * comparison; boundaries land after the terminator + any closing
 * quotes/brackets, so spans remain valid UTF-8 slices. */

/* Match one closing quote/bracket at t[i]; return its byte length or 0. */
static size_t close_len(const unsigned char *t, size_t i, size_t e)
{
    unsigned char c = t[i];
    if (c == '"' || c == '\'' || c == ')' || c == ']' || c == '}')
        return 1;
    if (i + 3 <= e && c == 0xE3 && t[i+1] == 0x80 &&
        (t[i+2] == 0x8D || t[i+2] == 0x8F ||     /* 」 』 */
         t[i+2] == 0x8B || t[i+2] == 0x89 ||     /* 》 〉 */
         t[i+2] == 0x91))                        /* 】 */
        return 3;
    if (i + 3 <= e && c == 0xEF && t[i+1] == 0xBC && t[i+2] == 0x89)
        return 3;                                /* ） */
    if (i + 3 <= e && c == 0xE2 && t[i+1] == 0x80 &&
        (t[i+2] == 0x9D || t[i+2] == 0x99))      /* " ' */
        return 3;
    return 0;
}

static int is_ws(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Fullwidth digit ０-９ (EF BC 90..99)? */
static int fw_digit(const unsigned char *t, size_t i, size_t e)
{
    return i + 3 <= e && t[i] == 0xEF && t[i+1] == 0xBC &&
           t[i+2] >= 0x90 && t[i+2] <= 0x99;
}

/* If a sentence terminator starts at t[i], classify it and return its
 * byte length; else return 0.  *cls: 1 = ASCII (trailing whitespace
 * required), 2 = self-delimiting, 3 = digit-guarded (fullwidth ．). */
static size_t term_len(const unsigned char *t, size_t i, size_t e, int *cls)
{
    unsigned char c = t[i];
    if (c == '.' || c == '!' || c == '?') { *cls = 1; return 1; }
    if (i + 2 <= e) {
        if (c == 0xD8 && t[i+1] == 0x9F) { *cls = 2; return 2; }  /* ؟ */
        if (c == 0xDB && t[i+1] == 0x94) { *cls = 2; return 2; }  /* ۔ */
        if (c == 0xD6 && t[i+1] == 0x89) { *cls = 2; return 2; }  /* ։ */
    }
    if (i + 3 <= e) {
        if (c == 0xE3 && t[i+1] == 0x80 && t[i+2] == 0x82)
            { *cls = 2; return 3; }                               /* 。 */
        if (c == 0xEF && t[i+1] == 0xBC) {
            if (t[i+2] == 0x81 || t[i+2] == 0x9F)
                { *cls = 2; return 3; }                           /* ！ ？ */
            if (t[i+2] == 0x8E)
                { *cls = 3; return 3; }                           /* ． */
        }
        if (c == 0xE0 && t[i+1] == 0xA5 &&
            (t[i+2] == 0xA4 || t[i+2] == 0xA5))
            { *cls = 2; return 3; }                               /* । ॥ */
        if (c == 0xE1 && t[i+1] == 0x81 && t[i+2] == 0x8B)
            { *cls = 2; return 3; }                               /* ။ */
        if (c == 0xE1 && t[i+1] == 0x8D && t[i+2] == 0xA2)
            { *cls = 2; return 3; }                               /* ። */
    }
    return 0;
}

/* Find the end of the sentence unit starting at `us` in [us,e): the
 * position just past the next boundary (terminator + closers), or e. */
static size_t sent_unit_end(const char *text, size_t us, size_t e)
{
    const unsigned char *t = (const unsigned char *)text;
    size_t i = us;
    while (i < e) {
        int cls = 0;
        size_t tl = term_len(t, i, e, &cls);
        if (!tl) { i++; continue; }
        size_t j = i + tl, cl;
        while (j < e && (cl = close_len(t, j, e)) != 0) j += cl;
        if (cls == 2)
            return j;                                  /* self-delimiting */
        if (cls == 3) {                                /* ．: digit guard */
            if (j >= e ||
                (!(t[j] >= '0' && t[j] <= '9') && !fw_digit(t, j, e)))
                return j;
        } else {                                       /* ASCII: ws guard */
            if (j >= e || is_ws(t[j]))
                return j;
        }
        i = j > i ? j : i + 1;
    }
    return e;
}

/* Split an oversized structureless piece [s,e) at sentence boundaries and
 * greedily pack the sentences to win_tokens.
 *
 * Backoff rule: a chunk never ends on TINY trailing units (under
 * RP_SENT_TAIL_MIN tokens) -- they move whole to the next chunk,
 * iteratively.  A false abbreviation boundary at the START of a sentence
 * ("Mr. Smith arrived...") makes a tiny "Mr." unit, so cuts can never
 * land there -- the same placement a lexicon-tailored segmenter gives.
 * The threshold is deliberately tiny (NOT min_tokens, which exceeds many
 * real sentences and would unravel the packing): a false boundary LATE in
 * a sentence ("...according to Mr." -- a 15-token fragment) is
 * indistinguishable from a real short sentence without a lexicon, and we
 * accept that cut exactly as untailored UAX #29 does.
 *
 * A single sentence over the window is emitted with .oversized=1 (caller
 * token-windows it).  Falls back to one oversized span when no boundaries
 * exist (Thai etc.). */
#define RP_SENT_TAIL_MIN 6
static void split_sentences(cctx *c, size_t s, size_t e)
{
    size_t minTok = (size_t)c->o.min_tokens;
    size_t win    = (size_t)c->o.win_tokens;

    /* phase 1: collect the sentence units */
    rp_chunk_span *u = NULL;
    size_t nu = 0, ucap = 0;
    size_t us = s;
    while (us < e) {
        size_t ue = sent_unit_end(c->t, us, e);
        size_t as = us, ae = ue;
        us = ue;
        trim(c->t, &as, &ae);
        if (as >= ae) continue;
        size_t n = piece_count(c, as, ae);
        if (c->err) { free(u); return; }
        if (nu == ucap) {
            size_t nc = ucap ? ucap * 2 : 32;
            rp_chunk_span *nv = realloc(u, nc * sizeof(*nv));
            if (!nv) { free(u); c->out.oom = 1; return; }
            u = nv; ucap = nc;
        }
        u[nu++] = (rp_chunk_span){ as, ae, n, 0 };
    }
    if (nu == 0) {                       /* no units at all */
        size_t n = piece_count(c, s, e);
        if (!c->err)
            sl_push(&c->out, (rp_chunk_span){ s, e, n, 1 });
        return;
    }

    /* phase 2: greedy pack with iterative sub-minimal-tail backoff */
    size_t first_out = c->out.n;
    size_t i = 0;
    while (i < nu) {
        if (u[i].n_tokens > win) {       /* single monster sentence */
            rp_chunk_span m = u[i];
            m.oversized = 1;
            sl_push(&c->out, m);
            i++;
            continue;
        }
        size_t j = i, tok = 0;
        while (j < nu && u[j].n_tokens <= win && tok + u[j].n_tokens <= win) {
            tok += u[j].n_tokens;
            j++;
        }
        if (j < nu) {                    /* a cut (not the tail): backoff */
            /* Pop tiny trailing units so a cut can't land after an
             * abbreviation fragment -- but never unwind the chunk below
             * half the window: a document made ENTIRELY of tiny
             * sentences (dialogue, chat logs, CJK short-sentence lists)
             * would otherwise unravel to one-unit chunks, emitting one
             * vector per sentence instead of packed windows. */
            while (j - i >= 2 && u[j - 1].n_tokens < RP_SENT_TAIL_MIN &&
                   tok - u[j - 1].n_tokens >= win / 2) {
                tok -= u[j - 1].n_tokens;
                j--;
            }
        }
        sl_push(&c->out, (rp_chunk_span){ u[i].start, u[j - 1].end, tok, 0 });
        i = j;
    }
    free(u);

    /* Haystack tail rule, same as flush_units: an under-floor final chunk
     * merges back into the previous chunk from this split when it fits */
    if (minTok > 0 && c->out.n - first_out >= 2) {
        rp_chunk_span *last = &c->out.v[c->out.n - 1];
        rp_chunk_span *prev = &c->out.v[c->out.n - 2];
        if (!last->oversized && !prev->oversized &&
            last->n_tokens < minTok &&
            prev->n_tokens + last->n_tokens <= win) {
            prev->end = last->end;
            prev->n_tokens += last->n_tokens;
            c->out.n--;
        }
    }
}

/* Split [s,e) at newline runs of >= `level` newlines; pieces that fit the
 * window are same-level units (packed/merged by flush_units); an oversized
 * piece flushes the pending units first (LangChain's rule: recursed output
 * never merges with siblings) and recurses at the next-finer level.  Below
 * level 1 the piece is emitted with .oversized=1 for the caller to
 * token-window. */
static void split_region(cctx *c, size_t s, size_t e, int level)
{
    if (c->err || c->out.oom) return;
    trim(c->t, &s, &e);
    if (s >= e) return;

    if (level <= 0) {
        size_t n = piece_count(c, s, e);
        if (c->err) return;
        if (n > (size_t)c->o.win_tokens && c->o.sentence_split) {
            split_sentences(c, s, e);
            return;
        }
        sl_push(&c->out, (rp_chunk_span){ s, e, n, n > (size_t)c->o.win_tokens });
        return;
    }

    /* don't split at a level with no separators in this region */
    int lvl = max_level(c->t, s, e);
    if (lvl > level) lvl = level;
    if (lvl == 0) { split_region(c, s, e, 0); return; }

    rp_chunk_span *pend = NULL;
    size_t pn = 0, pcap = 0;
    size_t ps = s;
    size_t i = s;
    for (;;) {
        int at_end = (i >= e);
        size_t re = 0;
        int is_sep = 0;
        if (!at_end && (c->t[i] == '\n' || c->t[i] == '\r')) {
            int nl;
            if (nl_run(c->t, e, i, &re, &nl)) {
                if (nl >= lvl) is_sep = 1;
                else { i = re; continue; }   /* finer run: ordinary content */
            }
        }
        if (at_end || is_sep) {
            size_t as = ps, ae = at_end ? e : i;
            trim(c->t, &as, &ae);
            if (as < ae) {
                size_t n = piece_count(c, as, ae);
                if (c->err) { free(pend); return; }
                if (n <= (size_t)c->o.win_tokens) {
                    if (pn == pcap) {
                        size_t nc = pcap ? pcap * 2 : 16;
                        rp_chunk_span *np = realloc(pend, nc * sizeof(*np));
                        if (!np) { free(pend); c->out.oom = 1; return; }
                        pend = np; pcap = nc;
                    }
                    pend[pn++] = (rp_chunk_span){ as, ae, n, 0 };
                } else {
                    flush_units(c, pend, pn, lvl);
                    pn = 0;
                    if (lvl > 1)
                        split_region(c, as, ae, lvl - 1);
                    else if (c->o.sentence_split)
                        split_sentences(c, as, ae);
                    else
                        sl_push(&c->out, (rp_chunk_span){ as, ae, n, 1 });
                    if (c->err || c->out.oom) { free(pend); return; }
                }
            }
            if (at_end) break;
            i = re;
            ps = re;
        } else {
            i++;
        }
    }
    flush_units(c, pend, pn, lvl);
    free(pend);
}

/* ---------- public entry ---------- */
int rp_chunk_text(const char *text, size_t len,
                  const rp_chunk_opts *opts,
                  rp_chunk_count_fn count, void *count_user,
                  rp_chunk_span **out_spans, size_t *out_n)
{
    if (out_spans) *out_spans = NULL;
    if (out_n) *out_n = 0;
    if (!text || !opts || !count || !out_spans || !out_n || opts->win_tokens <= 0)
        return -1;
    if (!len) return 0;

    cctx c;
    memset(&c, 0, sizeof c);
    c.t = text;
    c.o = *opts;
    c.count = count;
    c.user = count_user;
    if (c.o.min_tokens == 0)  c.o.min_tokens = 32;          /* default floor */
    if (c.o.min_tokens < 0)   c.o.min_tokens = 0;           /* disabled */
    if (c.o.min_tokens > c.o.win_tokens) c.o.min_tokens = c.o.win_tokens;

    int lvl = (c.o.mode == RP_CHUNK_WINDOW) ? 0 : max_level(text, 0, len);
    if (lvl == 0) {
        /* window mode, or no newline structure at all: one span; the caller
         * token-windows it if it exceeds the budget */
        size_t n = count(count_user, text, len);
        if (n == (size_t)-1) return -1;
        if (n > (size_t)c.o.win_tokens && c.o.sentence_split &&
            c.o.mode != RP_CHUNK_WINDOW)
            split_sentences(&c, 0, len);   /* structureless AUTO text */
        else
            sl_push(&c.out, (rp_chunk_span){ 0, len, n, n > (size_t)c.o.win_tokens });
    } else {
        split_region(&c, 0, len, lvl);
        if (!c.err && !c.out.oom && c.out.n == 0) {
            /* pathological (e.g. whitespace-only): fall back to one span */
            size_t n = count(count_user, text, len);
            if (n == (size_t)-1) { free(c.out.v); return -1; }
            sl_push(&c.out, (rp_chunk_span){ 0, len, n, n > (size_t)c.o.win_tokens });
        }
    }

    if (c.err || c.out.oom) { free(c.out.v); return -1; }
    *out_spans = c.out.v;
    *out_n = c.out.n;
    return 0;
}
