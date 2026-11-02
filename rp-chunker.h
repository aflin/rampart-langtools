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
 *   - a piece still over the window with no finer structure (or no text
 *     structure at all) is split at SENTENCE boundaries, and anything still
 *     over the window -- a monster sentence, or sentence splitting turned
 *     off -- is cut at WORD boundaries into windows overlapping by ~1/8.
 *     Every span therefore fits the window and is exactly the bytes its
 *     vector is embedded from.  (Callers used to token-window over-window
 *     spans themselves, giving every window the whole span.)
 *
 * The chunker is tokenizer-agnostic: sizing goes through a token-count
 * callback, so the same code serves rampart-onnx (onnxruntime-extensions
 * tokenizers) and later rampart-llamacpp (llama_tokenize).  This file is
 * shared at the SOURCE level -- compile rp-chunker.c into each module.
 *
 * Spans are BYTE offsets into the input, always on UTF-8 character
 * boundaries (cuts land on newlines, sentence ends, whitespace, or as a last
 * resort a character boundary).  \r\n is understood ("\r\n\r\n" counts as a
 * 2-newline run).  Word-level windows overlap their neighbours; no other
 * spans overlap.
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
    int    oversized;   /* 1 = a word-level window cut from a piece that
                           exceeded win_tokens (it fits; it overlaps its
                           neighbours) */
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
    int sentence_split;   /* 0 (default) or 1 = a piece over win_tokens with
                             no finer text structure is split at SENTENCE
                             boundaries and the sentences greedily packed to
                             win_tokens (LlamaIndex SentenceSplitter / Elastic
                             semantic_text behavior); -1 = off, cut such
                             pieces at word boundaries instead.  Measured
                             2026-11 (Wikipedia + a long essay, bge-small and
                             MiniLM at 256/512-token windows): sentences beat
                             word cuts and beat the old token windows on
                             retrieval and on locating the matching passage.
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
                             lexicon.  A single sentence over the window is
                             cut at word boundaries.  (Before 2026-11 the
                             default was off and over-window pieces went to
                             caller token-windowing; headerless pre-header
                             rows holding such pieces now re-chunk
                             differently, and abstract() falls back to a
                             plain snippet for them.) */
} rp_chunk_opts;

#define RP_CHUNK_AUTO   0   /* detect structure; sentence / word fallback */
#define RP_CHUNK_WINDOW 1   /* no structure split: word-level windows only */
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

/* Byte range of a caller-side token window over span [s,e) that still
 * tokenized past the model window (a count mismatch; the chunker itself no
 * longer emits over-window spans): window tokens [tok_start, tok_start +
 * tok_n) of tok_total, mapped proportionally to bytes and snapped inward to
 * UTF-8 character boundaries.  An estimate, but never the whole span. */
void rp_chunk_window_span(const char *text, size_t s, size_t e,
                          size_t tok_start, size_t tok_n, size_t tok_total,
                          size_t *out_s, size_t *out_e);

#ifdef __cplusplus
}
#endif

#endif /* RP_CHUNKER_H */
