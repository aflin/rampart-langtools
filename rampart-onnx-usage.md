# Using rampart-onnx

`rampart-onnx` runs ONNX models (embeddings, rerankers, audio decoders, or any
general ONNX graph) from Rampart JavaScript. ONNX Runtime and its tokenizers
are statically linked into the module — there is nothing else to install and no
external `libonnxruntime.so`.

```js
var onnx = require('rampart-onnx');
```

The module ships as `rampart-onnx_cpu.so` (with a `rampart-onnx.so` symlink) in
`<rampart>/modules/`. A GPU flavor (`rampart-onnx_cu12.so` / `_cu13.so`) exists
for CUDA machines; the API is identical, plus the `gpu`/`provider`/`device`
session options below.

Models live under `~/.rampart/models/` by convention. Embedding models are
**directories** mirroring a HuggingFace download (see
[Model directories](#model-directories)).

All functions **throw** a JavaScript `Error` on failure (bad path, bad options,
model/run errors). There are no error-return values to check.

---

## Quick start

```js
var onnx = require('rampart-onnx');
var M    = rampart.utils.homedir() + '/.rampart/models';

/* embeddings: pass the model DIRECTORY -- everything self-configures */
var emb = onnx.initEmbed(M + '/embed/bge-m3');
var r   = emb.embedTextToNumbers('How do I bake bread at home?');
// r.avgVec    -> [1024 numbers], unit length
// r.vecs      -> one vector per chunk (1 for short text)
// r.coherence -> 1.0
emb.destroy();
```

---

## Module functions

### `onnx.onnxVersion()`
Returns the ONNX Runtime version string, e.g. `"1.27.0"`.

### `onnx.modelInfo(path)`
Inspect a model without creating a session.

**Returns** `{ inputs: [ {name, type, shape}, ... ], outputs: [ ... ] }`.
`type` is a dtype string (`"float32"`, `"int64"`, ...); `shape` is an array of
integers with `-1` for dynamic dimensions.

### `onnx.getLog()` / `onnx.clearLog()`
ONNX Runtime warnings and non-fatal errors are captured into an in-memory
buffer instead of being printed to stderr. `getLog()` returns the buffered text
(a string, `""` if empty); `clearLog()` empties it. The buffer keeps the most
recent ~64 KB.

---

## Layer 1 — sessions (any ONNX model)

### `onnx.initSession(path, options)` / `onnx.initSessionFromBuffer(buffer, options)`

Load a model from a file path, or from bytes already in memory.

| option | default | meaning |
|---|---|---|
| `intraOpThreads` | `1` | ORT intra-op threads. `1` = no background threads (**fork-safe**). Raise for faster single-run CPU inference, but create such sessions only after any `fork()`. |
| `interOpThreads` | `1` | ORT inter-op threads (only used with `executionMode:'parallel'`). |
| `executionMode` | `'sequential'` | `'sequential'` or `'parallel'`. |
| `graphOpt` | `'all'` | Graph optimization: `'disable'`, `'basic'`, `'extended'`, `'all'`. |
| `provider` | `'cpu'` | `'cpu'` or `'cuda'` (GPU flavor only). |
| `gpu` | `false` | `gpu:true` is shorthand for `provider:'cuda'`. |
| `device` | `0` | CUDA device index (with `gpu`/`provider:'cuda'`). |

**Returns** a session object:

#### `session.run(feeds)`
Run the model. `feeds` is an object keyed by **input name**; every value is
`{ data, shape, type }`:

- `data` — a JS Array of numbers, or any Buffer/TypedArray with the raw bytes
- `shape` — array of integers, e.g. `[1, 128]`
- `type` — dtype string: `float32 float16 float64 int64 int32 int16 int8 uint8 bool`

```js
var out = sess.run({
    input_ids:      { data: ids,  shape: [1, ids.length],  type: 'int64' },
    attention_mask: { data: mask, shape: [1, mask.length], type: 'int64' },
});
```

**Returns** an object keyed by **output name**. Each output is:

| field | contents |
|---|---|
| `.data`  | ArrayBuffer with the raw tensor bytes |
| `.array` | a ready typed-array view of `.data` (Float32Array, Int32Array, ...). **Omitted for int64** outputs (no native typed array); read `.data` yourself. |
| `.shape` | array of integers (the concrete output shape) |
| `.type`  | dtype string |

```js
var logits = out.logits.array;       // Float32Array
var shape  = out.logits.shape;       // e.g. [1, 128, 30522]
```

#### `session.inputs()` / `session.outputs()`
`[ {name, type, shape}, ... ]` — same shape info as `modelInfo()`, from the
live session. `-1` marks dynamic dims.

#### `session.metadata()`
`{ producerName, graphName, domain, description, version }` (fields present
only when the model provides them; `version` is a number).

#### `session.destroy()`
Free the session. Optional — also runs on garbage collection. Idempotent;
using a destroyed session throws.

**Thread/fork safety:** with the defaults, sessions survive `fork()` and
`session.run()` may be called from multiple Rampart threads concurrently. The
destroy/GC logic only frees on the creating thread + pid, so session objects
copied across threads don't double-free.

---

## Tokenizers

Native C++ tokenizers (from onnxruntime-extensions). Both return **content
ids only** — no `[CLS]`/`[SEP]`/`<s>`/`</s>`; the embed/rerank layers add
those.

### `onnx.wordPieceTokenizer(vocabPath, options)`
BERT WordPiece from a standard `vocab.txt` (one token per line; line number =
id). Options (all default `true`): `lowercase`, `stripAccents`,
`tokenizeChinese`.

**Returns** `{ encodeIds(text) -> [ids], vocabSize }`.

```js
var wp = onnx.wordPieceTokenizer(dir + '/vocab.txt');
wp.encodeIds('Hello, world!');   // [7592, 1010, 2088, 999]
wp.vocabSize;                    // 30522
```

### `onnx.spTokenizer(modelDir)`
SentencePiece (Unigram/XLM-R) and BPE tokenizers. Takes a **directory**
containing HuggingFace `tokenizer.json` + `tokenizer_config.json` (an
embedding-model directory works directly).

**Returns** `{ encodeIds(text) -> [ids] }`.

```js
var sp = onnx.spTokenizer(M + '/embed/bge-m3');
sp.encodeIds('Hello, world!');   // [35378, 4, 8999, 174128]
```

**Custom tokenizers:** anywhere a tokenizer is accepted, you may pass your own
object with an `encodeIds(text) -> [ids]` method instead.

---

## Embeddings — `onnx.initEmbed(modelDir, options)`

Create an embedder from a **model directory**; tokenizer, pooling and
normalization are discovered from the directory's files, so no options are
required:

```js
var emb = onnx.initEmbed(M + '/embed/all-minilm-l6-v2');
```

Discovery (every field overridable via options):

| what | discovered from | fallback |
|---|---|---|
| model file | `onnx/model.onnx`, else `model.onnx`, else first `*.onnx` | error |
| tokenizer | `*vocab.txt` → WordPiece; else `tokenizer.json` → SentencePiece/BPE | error (pass `tokenizer`) |
| pooling | `1_Pooling/config.json` (`cls` or `mean`) | model's pooled output, else `mean` |
| bos/eos ids | tokenizer family | WordPiece `101/102`, SPM `0/2` |
| normalize | — | `true` |

Options:

| option | default | meaning |
|---|---|---|
| `tokenizer` | auto | A tokenizer object (`encodeIds()`), or a SentencePiece `.model` path (loads via rampart-sentencepiece). Overrides auto-detection. Required if `modelDir` is a bare `.onnx` file. |
| `pooling` | auto | `'mean'` or `'cls'`. |
| `normalize` | `true` | L2-normalize each vector. |
| `maxTokens` | auto | Per-chunk token window (bos/eos count against it). Auto = the model's `max_position_embeddings` (from `config.json`; fallback `sentence_bert_config.json` `max_seq_length`), capped at 8192 — llamacpp parity. Explicit values are not capped. |
| `maxChunkBatch` | `64` cpu / `32` gpu | Max chunks per batched model run (memory cap). |
| `split` | `'auto'` | `'auto'` = structure-aware chunking with window fallback; `'window'` = always the sliding token window (pre-chunker behavior). |
| `minTokens` | `32` | Paragraph fragment floor: smaller paragraphs merge with neighbors. `-1` disables merging (strictly one vector per paragraph). |
| `packParagraphs` | `false` | Pack consecutive paragraphs up to the window (fewer, larger chunks — LangChain-style) instead of one vector per paragraph. |
| `bosId` / `eosId` | auto | Special-token ids; `-1` disables. |
| `idOffset` | `0` | Added to every token id (legacy tokenizers only). |
| `padId` | auto | Padding id for ragged batches (masked out anyway). |
| `queryPrefix` / `passagePrefix` | none | Prefix prepended when embedding queries / passages (e5-style models). |
| `lowercase`, `stripAccents`, `tokenizeChinese` | `true` | WordPiece config (dir mode). |
| *session options* | | `gpu`, `intraOpThreads`, `graphOpt`, ... are passed through to `initSession`. |

**Long text is never truncated — and chunking is structure-aware.** Text is
split at semantic boundaries (the method used by LangChain / LlamaIndex /
Unstructured, implemented in `rp-chunker.c`):

1. **Blank-line paragraphs** (`\n\n`) → one chunk (one vector) per paragraph,
   no overlap. Paragraphs under `minTokens` merge with neighbors so fragments
   (titles, bylines) don't become tiny vectors.
2. **Single-newline text** → lines are packing units: greedily packed up to the
   token window with cuts at line boundaries, no overlap.
3. **No structure** (or an oversized single paragraph) → sliding token window
   with 1/8 overlap — the classic fallback.

All chunks of a document run through the model **batched**; you get every
chunk's vector, its text span, and a combined document vector.

**Returns** an embedder object:

#### `emb.embedTextToNumbers(text [, isQuery])`
**Returns** `{ vecs, avgVec, coherence, chunks }`:

| field | contents |
|---|---|
| `vecs` | Array of per-chunk vectors (each an Array of Numbers, unit length). Short text → one entry. |
| `avgVec` | The combined document vector (Array of Numbers, unit length): L2-normalized mean of the chunk vectors. |
| `coherence` | Number in [0,1]: the **average pairwise cosine similarity** between the chunk vectors (k-independent). `1.0` = single chunk or perfectly aligned chunks; `~0` = unrelated topics. Rule of thumb: **search** against the per-chunk `vecs` when it's low; use `avgVec` as a coarse address for **sharding/clustering** either way. Note it inherits the model's similarity baseline (e5 scores even unrelated text ~0.7; all-MiniLM ~0.0) — calibrate thresholds per model. |
| `chunks` | Array parallel to `vecs`: `{ start, end, tokens, text }` — the **byte** span of the chunk in the input text, its embedded token count, and the chunk text itself (use `.text`; the byte offsets differ from JS char indices for non-ASCII). Token-window sub-chunks of an unstructured/oversized region share that region's span and carry `oversized: true` — the "one vector per paragraph" invariant did not hold there; check it if your schema depends on 1:1 span↔vector mapping. |

`isQuery:true` applies `queryPrefix` instead of `passagePrefix` (the prefix is
applied to **every chunk** at encode time; spans refer to the unprefixed text).

#### `emb.embedTextToFp32Buf(text [, isQuery])`
Same `{ vecs, avgVec, coherence }`, but every vector is a **Float32Array**.

#### `emb.embedTextToFp16Buf(text [, isQuery])`
Same shape; vectors are **Uint16Array** holding IEEE half-precision bits
(ready to store as compact fp16 blobs).

#### `emb.embedTextsToNumbers(texts [, isQuery])`
Batch many **separate** texts (one vector each; each text is truncated to a
single window — use the single-text calls for long documents). All texts ride
through the model batched.

**Returns** `[ { avgVec: [numbers] }, ... ]` in input order.

#### `emb.session` / `emb.destroy()`
The underlying session (usable directly), and its destructor.

```js
var doc = emb.embedTextToNumbers(longArticle);
printf('%d chunks, coherence %.2f\n', doc.vecs.length, doc.coherence);
// store doc.vecs for search, doc.avgVec for routing
```

---

## Reranking — `onnx.initRerank(modelPath, options)`

Cross-encoder relevance scoring (e.g. bge-reranker-base, ms-marco-MiniLM).
Like `initEmbed`, **pass a model directory and everything self-configures**:
the `.onnx`, the tokenizer (`*vocab.txt` → WordPiece, else `tokenizer.json` →
SPM/BPE), the special tokens, the token window, and the pair template —
BERT-style `[CLS] q [SEP] d [SEP]` with `token_type_ids` for WordPiece models,
RoBERTa/XLM-R-style `<s> q </s></s> d </s>` otherwise.

```js
var rr = onnx.initRerank(M + '/rerank/bge-reranker-v2-m3');       // that's it
// or: onnx.initRerank(models.onnxGet('bge-reranker-v2-m3'))      // with rampart-models
```

A bare `.onnx` path still works, but then `tokenizer` is required and the
family defaults below apply.

| option | default (dir mode) | meaning |
|---|---|---|
| `tokenizer` | discovered | Tokenizer object or SentencePiece `.model` path (required in file mode). |
| `bosId` / `eosId` | by family (WP `101`/`102`, else `0`/`2`) | Pair framing tokens. |
| `pairTemplate` | by family | `'bert'` (single SEP + token types) or `'roberta'` (doubled eos). |
| `padId` | by family (WP `0`, else `1`) | Batch padding id (masked out). |
| `idOffset` | `0` | Added to every token id. |
| `sigmoid` | `true` | Map the raw logit through 1/(1+e⁻ˣ) to a 0..1 score. |
| `maxTokens` | model's discovered window | Cap the pair length (doc side is truncated, closing eos kept). |
| `maxChunkBatch` | `64` cpu / `32` gpu | Docs per batched model run. |
| *session options* | | passed through to `initSession`. |

Passing an **embedding** model directory throws ("looks like an embedding
model, not a cross-encoder").

Scoring is batched: one call scores **all** documents in as few model runs as
possible.

**Returns** a reranker object:

#### `rr.rerank(query, docs [, scoresOnly])`

| call | returns |
|---|---|
| `rr.rerank(q, [d1, d2, ...])` | Array sorted by score, **descending**: `[ { document, score, index }, ... ]` — `index` is the position in your input array. |
| `rr.rerank(q, [d1, d2, ...], true)` | Just the scores (Array of Numbers, in **document order**). |
| `rr.rerank(q, "single doc")` | A single Number. |

```js
var ranked = rr.rerank('capital of France',
                       ['Paris is the capital of France.', 'Bread needs flour.']);
// [ {document:'Paris is...', score:0.98, index:0},
//   {document:'Bread...',    score:0.02, index:1} ]
```

#### `rr.session` / `rr.destroy()`

---

## SNAC audio decode — `onnx.initSnacDecoder(modelPath, options)`

Decodes SNAC neural-audio codes (the Orpheus TTS vocoder) to PCM.

```js
var snac = onnx.initSnacDecoder(M + '/tts/snac_decoder.onnx');
```

Options: `codeOffset` (default `10`) and `slotSpan` (default `4096`) — the
Orpheus token-numbering layout; plus session options.

**Returns:**

| member | meaning |
|---|---|
| `snac.sampleRate` | `24000` |
| `snac.decodeOrpheus(tokens)` | Orpheus token numbers (flat, 7 per frame) → **Float32Array** PCM samples. The usual entry point after llamacpp generates Orpheus audio tokens. |
| `snac.decodeFrames(frames)` | Flat SNAC codes (7 per frame, offsets already stripped) → Float32Array PCM. |
| `snac.framesToCodes(frames)` | Demux flat frames into the 3-level SNAC hierarchy `[c0, c1, c2]` (1/2/4 codes per frame). |
| `snac.decode([c0,c1,c2])` | Raw 3-codebook decode → Float32Array PCM. |
| `snac.session` / `snac.destroy()` | |

Note: the SNAC decoder contains a stochastic noise block — output audio
differs slightly run-to-run by design.

---

## Model directories

Embedding model directories mirror what
`huggingface_hub.snapshot_download("<repo>")` produces:

```
~/.rampart/models/embed/<model>/
    onnx/model.onnx           (+ model.onnx_data for large models)
    1_Pooling/config.json     pooling mode (mean vs cls)
    tokenizer.json            fast-tokenizer definition (SPM/BPE)
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt                 (WordPiece/BERT models only)
    config.json  modules.json  sentence_bert_config.json
```

Verified models: `all-MiniLM-L6-v2` (WordPiece, mean-pool, 384-dim),
`bge-m3` (XLM-R SentencePiece, CLS-pool, 1024-dim),
`multilingual-e5-small` (SentencePiece, mean-pool, 384-dim, uses
`passagePrefix:'passage: '` / `queryPrefix:'query: '`).

---

## C API (for C modules, e.g. rampart-sql)

`rampart-onnx.so` also exports a small C ABI so host modules can embed text
without a JS context: `rp_onnx_embed_load(dir, opts, err, errlen)` →
`rp_onnx_embed_text(h, text, len, &vec)` (one combined vector) and
`rp_onnx_embed_doc(...)` (per-chunk vectors + avgVec + coherence + per-chunk
`{start,end,n_tokens}` spans), plus `rp_onnx_embed_dim` /
`rp_onnx_embed_release`. Both use the same structure-aware chunking as the JS
side (`split_mode`/`min_split_tokens`/`pack_paragraphs` in the opts struct). Like the JS side, passing a
model **directory** with only `abi_version` set self-configures everything.
(The JSON-config discovery — pooling, token window — uses the host rampart
thread's duktape context; in a non-rampart process that dlopens the module,
discovery is skipped and you must set `pooling`/`max_tokens` in the opts.)
Full struct and semantics: see the "C API" section of `rampart-onnx.md`.

---

## Behavior notes

- **Errors:** every call throws `Error` with a descriptive message; nothing
  returns null/undefined on failure.
- **Fork safety:** default sessions spawn no threads and survive `fork()`.
  `intraOpThreads > 1` or `executionMode:'parallel'` sessions must be created
  after any fork.
- **GC:** `destroy()` is optional everywhere; finalizers release native
  handles when objects are collected. Calling into a destroyed
  session/embedder throws.
- **Warnings:** ORT never writes to stderr; check `onnx.getLog()` if something
  seems off.
- **Performance:** all per-call work (tokenization, chunking, batching,
  pooling, normalization) runs in native code; on CPU, raise `intraOpThreads`
  for lower latency at the cost of fork safety. On the GPU flavor, batching
  across chunks/docs is where the speedup comes from.
