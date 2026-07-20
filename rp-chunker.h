/* rp-chunker.h -- structure-aware text chunking for embedding pipelines.
 *
 * Splits a UTF-8 document into token-budgeted chunks at SEMANTIC boundaries,
 * following the consensus design of LangChain's RecursiveCharacterTextSplitter,
 * LlamaIndex's SentenceSplitter, the Rust text-splitter crate and Unstructured's
 * chunkers (researched 2026-07):
 *
 *   - boundary hierarchy by descending newline-run length (\n\n\n+ > \n\n > \n),
 *     picking the longest level PRESENT in the text (text-splitter's "ascending
 *     newline sequence" levels);
 *   - per-piece recursion: an oversized piece is re-split at the next-finer
 *     level; its output never merges with siblings (LangChain's flush rule);
 *   - blank-line paragraphs (\n\n+) become one chunk each ("a vector per
 *     paragraph"); fragments under min_tokens merge with neighbors
 *     (Unstructured's combine_text_under_n_chars / Haystack's split_threshold);
 *   - single-\n level: lines are packing units, greedily packed to the window
 *     with cuts at line boundaries, NO overlap (hard-wrapped prose self-heals);
 *   - no text structure at all -> ONE span covering the text; the CALLER
 *     token-windows it (with overlap) exactly as before.  The chunker never
 *     cuts inside a line; token-level windowing stays the caller's fallback.
 *
 * The chunker is tokenizer-agnostic: sizing goes through a token-count
 * callback, so the same code serves rampart-onnx (onnxruntime-extensions
 * tokenizers) and later rampart-llamacpp (llama_tokenize).  This file is
 * shared at the SOURCE level -- compile rp-chunker.c into each module.
 *
 * Spans are BYTE offsets into the input.  All cuts land on '\n' (ASCII), so
 * spans are always valid UTF-8 boundaries.  \r\n is understood ("\r\n\r\n"
 * counts as a 2-newline run).
 */
#ifndef RP_CHUNKER_H
#define RP_CHUNKER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Count the tokens the tokenizer would produce for text[0..len).  Return
 * (size_t)-1 on tokenizer failure (aborts the chunking with -1). */
typedef size_t (*rp_chunk_count_fn)(void *user, const char *text, size_t len);

typedef struct {
    size_t start;       /* byte span [start, end) in the input text */
    size_t end;
    size_t n_tokens;    /* counted content tokens for the span (merged spans:
                           sum of their pieces -- an estimate; re-tokenize the
                           span for the exact count) */
    int    oversized;   /* 1 = exceeds win_tokens even at the finest text
                           level; the caller should token-window it */
} rp_chunk_span;

/* All int options; 0 = default. */
typedef struct {
    int win_tokens;       /* REQUIRED: content-token budget per chunk (the
                             caller subtracts its bos/eos overhead first) */
    int min_tokens;       /* fragment floor at the paragraph level: spans
                             smaller than this merge with neighbors.
                             default 32; pass -1 to disable merging */
    int pack_paragraphs;  /* 1 = pack paragraphs up to win_tokens (framework
                             style, fewer vectors); 0 = one chunk per
                             paragraph (retrieval granularity; default) */
    int mode;             /* RP_CHUNK_AUTO / _WINDOW / _PARA */
    int sentence_split;   /* 1 = an oversized structureless piece is split at
                             SENTENCE boundaries and the sentences greedily
                             packed to win_tokens (LlamaIndex SentenceSplitter
                             / Elastic semantic_text behavior) instead of
                             being handed to the caller's token-windowing.
                             Boundaries come from a curated multi-script
                             terminator table (a pragmatic subset of Unicode
                             UAX #29): ASCII .!? require trailing whitespace
                             (which subsumes the decimal-point rule);
                             self-delimiting terminators need none
                             (CJK 。！？, Arabic ؟ ۔, Devanagari ।॥, Myanmar ။,
                             Ethiopic ።, Armenian ։); fullwidth ．is digit-
                             guarded (３．１４).  Closing quotes/brackets ride
                             with their sentence.  A chunk never ends on a
                             sub-min_tokens final sentence (backoff: it moves
                             whole to the next chunk), which neutralizes most
                             abbreviation false-boundaries ("Mr.") without a
                             lexicon.  A single sentence over the window still
                             falls back to caller token-windowing.  Default 0
                             (off): flipping it changes chunk boundaries,
                             which headerless (pre-header) tables' abstract()
                             span recomputation depends on. */
} rp_chunk_opts;

#define RP_CHUNK_AUTO   0   /* detect structure; window fallback */
#define RP_CHUNK_WINDOW 1   /* no structure split: one span, caller windows */
#define RP_CHUNK_PARA   2   /* DEPRECATED: behaves exactly like RP_CHUNK_AUTO.
                               Kept so the numbering of the versioned opts
                               structs (rp_onnx_embed_opts.split_mode) is
                               stable; a "force paragraphs" mode has no sane
                               failure semantics (throw? truncate?) -- callers
                               that need the one-vector-per-paragraph invariant
                               should check the per-chunk `oversized` flag
                               instead and apply their own policy. */

/* Chunk text[0..len).  On success returns 0 and sets *out_spans (malloc'd,
 * caller frees) and *out_n (>= 1 for non-empty text).  Returns -1 on OOM or
 * count-callback failure.  len == 0 yields *out_n == 0. */
int rp_chunk_text(const char *text, size_t len,
                  const rp_chunk_opts *opts,
                  rp_chunk_count_fn count, void *count_user,
                  rp_chunk_span **out_spans, size_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* RP_CHUNKER_H */
