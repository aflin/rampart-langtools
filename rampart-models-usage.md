# Using rampart-models

`rampart-models` downloads and locates GGUF and ONNX models, returning the
model **file** — which feeds `rampart-llamacpp` directly.  `rampart-onnx`
wants the model's **directory** instead (that is where it finds the
tokenizer, pooling and token window), so derive it from the returned path:

```js
var models   = require('rampart-models');
var onnx     = require('rampart-onnx');
var llamacpp = require('rampart-llamacpp');

/* .../embed/bge-m3/onnx/model_fp16.onnx  ->  .../embed/bge-m3 */
function modelDir(f) {
    var d = f.replace(/\/[^\/]*$/, '');
    return /\/onnx$/.test(d) ? d.replace(/\/onnx$/, '') : d;
}

var emb = onnx.initEmbed( modelDir(models.onnxGet('bge-m3')) );      // onnx: a DIRECTORY
var em2 = llamacpp.initEmbed( models.ggufGet('bge-m3') );           // gguf: a FILE
var gen = llamacpp.initGen(   models.ggufGet('qwen3-4b:q4_k_m') );  // exact quant
```

Everything is **synchronous**: the call blocks until the model is on disk and
returns the path.  If the model is already present under
`~/.rampart/models/<category>/`, the path is returned immediately with no
network access — so it is safe (and intended) to call at the top of any script.

It is a single pure-JS file (`rampart-models.js`) that needs only
`rampart-curl` and `rampart-crypto`.  The catalog comes in two halves: a
stable set built into the script, and a fetched layer that overrides and
extends it — see [The catalog](#the-catalog).

---

## What you can ask for

| call | you get |
|---|---|
| `models.get('bge-m3')` | the onnx **model file**, fp16 (default for embed/rerank models); its directory is what `rampart-onnx` wants |
| `models.get('bge-m3:q8_0')` | the Q8_0 **gguf file** (a `:quant` suffix always means gguf) |
| `models.get('qwen3-4b')` | a gguf file (gen models only exist as gguf; default quant) |
| `models.ggufGet('bge-m3')` | gguf file, default quant — **explicit**, use with llamacpp |
| `models.onnxGet('bge-m3')` | onnx model file, fp16 — **explicit**; pass its directory to rampart-onnx |
| `models.get('BAAI/bge-m3')` | exact HuggingFace repo (`org/repo`), no search |
| `models.get('org/repo:q4_k_m')` | exact repo + quant |
| `models.get('https://host/x.gguf')` | plain download of any URL, returns the file |
| `models.url(u, opts)` | same as above, explicit form |

Default quant preference when none is given: `Q4_K_M` → `Q8_0` → `Q5_K_M` →
`Q6_K` → `F16`.  Asking for a quant a model doesn't have throws an error that
**lists the available quants**.

## How names resolve

For a short name like `'bge-m3'`, in order:

1. **Already on disk** — `~/.rampart/models/<category>/<name>` (onnx dir) or
   the known gguf filename: returned instantly.
2. **The catalog** — ~100 curated embedding / reranker / generation / clip
   models with pinned repos, commit revisions, and full quant maps.
   `models.list()` shows them; `models.catalog` is the raw data, and
   `models.variants(name)` lists a model's quants with real file sizes.
3. **`org/repo` passthrough** — a `/` in the name skips all lookup.
4. **Live HuggingFace discovery** — for names the catalog doesn't know
   (including models released after the catalog was generated).  Ranked by
   exact name match, the model family's own org, established converter orgs,
   quant coverage, and downloads.  The chosen source is announced, and the
   resolution is remembered in `~/.rampart/models/.resolved.json` so the same
   name resolves the same way next time.

## Where things go

Everything lives under `~/.rampart/models/` (override per call with `dest`):

```
~/.rampart/models/
  embed/bge-m3/                  onnx model directories (HF snapshot layout:
                                 onnx/model.onnx, tokenizer.json, 1_Pooling/, ...)
  embed/bge-m3-FP16.gguf         gguf files keep their source filename
  rerank/bge-reranker-v2-m3/
  gen/qwen3-4b-Q4_K_M.gguf
  other/...                      url() downloads with no category
```

Directory names use the catalog's lowercase alias (`embed/all-minilm-l6-v2`).
Each downloaded onnx directory gets a `.source.json` recording the repo,
commit revision, and file list it came from.

The onnx directories are exactly what `onnx.initEmbed(dir)` self-configures
from (tokenizer, pooling, token window all discovered).

## Options (second argument to `get`/`ggufGet`/`onnxGet`/`url`)

| option | default | meaning |
|---|---|---|
| `format` | by model | `'gguf'` or `'onnx'` (the `*Get` variants set this for you) |
| `quant` | preference order | gguf quantization; same as the `:quant` suffix |
| `category` | from catalog | subdirectory: `'embed'`, `'rerank'`, `'gen'`, ... (`'other'` for URLs) |
| `dest` | category layout | exact destination file/dir, overrides everything |
| `progress` | stdout | `false` (silent), a file handle opened with `fopen` (written via `fprintf`), or `function(info)` with `{file, got, total, mbps}` |
| `force` | `false` | re-download even if present |
| `token` | `$HF_TOKEN` | HuggingFace token for gated/private repos |
| `revision` | catalog-pinned sha, else `main` | git revision to fetch |
| `sha256` | — | (`url()` only) verify the downloaded file's digest |

Progress renders as a single self-updating line:

```
  bge-m3/onnx/model.onnx_data      42.3%    958.1 / 2161.8 MB   38.2 MB/s
```

## Reliability behavior

- **Streamed to `dest.part`**, atomically renamed on success — an interrupted
  download never leaves a half-written file under the real name.
- **Resume**: an interrupted `.part` continues with an HTTP `Range` request;
  a `.part` that is already complete is finalized without any network.
- **Retries** with backoff on transient failures; sizes verified against the
  repo listing; large-file safe (>2 GiB tested).
- **Compressed responses are refused, never stored**: the transport requests
  identity encoding and fails loudly if a server sends compressed bytes
  anyway (a raw gzip stream on disk is worse than an error).
- **Gated repos**: a 401/403 produces "set HF_TOKEN", not a mystery failure.
- `HF_ENDPOINT` env overrides the HuggingFace host (mirrors, internal proxies).

## Command line

```
rampart rampart-models.js bge-m3                 # resolve/download, print the path
rampart rampart-models.js bge-m3 gguf q8_0       # format + quant
rampart rampart-models.js Qwen/Qwen3-4B-GGUF:q4_k_m
rampart rampart-models.js https://host/model.gguf
rampart rampart-models.js --list                 # catalog by category
rampart rampart-models.js --update               # refresh the catalog now
```

The path is printed to stdout (progress goes to the same terminal), so it
composes: `rampart run-something.js $(rampart rampart-models.js bge-m3)`.

## Other exports

```js
models.resolve('some-model')   // -> catalog-shaped entry {category, onnx:{...}, gguf:{...}}
                               //    without downloading anything
models.list()                  // -> { embed:[...], rerank:[...], gen:[...] }
models.catalog                 // the catalog object (name -> entry)
models.variants('qwen3-30b-a3b')// -> [{quant, bytes, files, installed}, ...] smallest first
models.catalogInfo()           // -> {source, entries, generated, fetched, etag, error, ...}
models.updateCatalog()         // force a catalog refresh now
models.modelsDir               // "~/.rampart/models"
models.pull(...)               // alias of get()
```

## The catalog

Models turn over much faster than this script, but some model families barely
move at all.  The catalog is split accordingly:

**Built in** — compiled into `rampart-models.js`: every `clip`, `ocr` and
`rerank` model, plus the stable embedding and generation workhorses.  These
resolve with **no network and no cache** — if github is unreachable, asking
for `bge-m3` or `qwen3-4b` still works, from disk or by downloading straight
from its pinned repo.  Only a release changes this set.

**Fetched** — `rampart-models-catalog.json`, downloaded from the project repo
and cached under the model store; deliberately *not* installed with the
module.  It carries the **whole** catalog, built-ins included, and each entry
**overrides the built-in of the same name**.  That is what lets a moved repo,
a re-pinned revision or a new quant reach installed scripts without a release,
and it is where newly discovered and hand-added models arrive.

The fetched half is resolved in this order:

1. `rampart-models-catalog.json` **beside the script** — a repo checkout, or a
   file you place there to pin a catalog.  Never present in an installed tree,
   so this is the development path: regenerate, test locally, then push.
2. `~/.rampart/models/.rampart-models-catalog.json` — the cache, revalidated
   against the repo with a conditional GET (`ETag`): `304` keeps it, `200`
   replaces it.
3. The network alone, on the very first run.

So a catalog we push reaches every install on its next use.  A failed refresh
is never fatal — the cache keeps working offline, and a download that is
truncated, an error page, or implausibly small is rejected rather than allowed
to replace a good cache.  With no catalog at all, the built-ins still resolve;
anything else falls through to live HuggingFace search, and one warning is
printed.

| Environment variable | Effect |
| --- | --- |
| `RAMPART_MODELS_CATALOG_URL` | fetch from a different URL (a mirror, or a pinned raw URL) |
| `RAMPART_MODELS_CATALOG_TTL` | seconds between revalidations (default `0` — check on every load) |
| `RAMPART_MODELS_CATALOG_OFFLINE` | never touch the network; use the cache only |

`models.catalogInfo()` reports which source is live, how many models it holds,
when it was generated and last fetched, and why the last refresh failed if it
did.  `rampart rampart-models.js --update` forces a refresh.

## Regenerating the catalog

`rampart-models-catalog.json` is generated — do not edit it by hand.  To
refresh it (new models, new quants, moved repos):

```
rampart gen-model-list.js
```

`gen-model-list.js` (same directory) sweeps the HuggingFace API by category,
resolves both formats per model with the same ranking heuristics the module
uses at runtime, and writes `rampart-models-catalog.json` (one model per line,
so the diff is reviewable).  Its run report shows a diff against the previous
catalog and every ranking choice with runners-up; pin corrections or add
`"skip"` entries in its `OVERRIDES` table and rerun.

One sweep writes both halves: the full `rampart-models-catalog.json`, and the
built-in subset spliced into `rampart-models.js` between the `==== BEGIN/END
GENERATED BUILTIN CATALOG ====` markers.  Membership of the built-in set is
the `BUILTIN_CATEGORIES` / `BUILTIN_MODELS` table at the top of
`gen-model-list.js` — add an alias there to promote a model, and keep the list
to things that are genuinely stable, since every entry is script weight.

A regenerated catalog takes effect locally as soon as it is written (the file
beside the script wins), and reaches other installs when it is **pushed** —
that is the release step for models.  Models that discovery cannot surface
(CLIP, OCR) are hand-pinned in the generator's `OVERRIDES`; an OCR entry
carries its whole role/variant map verbatim via `ocr:`.  To add a single model
without a full sweep, add its override and run
`rampart gen-model-list.js --pin <alias>`.
Repos are recorded with their commit sha, so catalog downloads use immutable
`resolve/{sha}/` URLs.

## Notes

- Generation models are gguf-only (rampart runs generation through llama.cpp).
- Vision/multimodal models (llava, moondream, `mmproj` projector files) are
  not in the catalog yet — fetch those with `models.url(...)` for now.
- Licenses ride along in the catalog (`models.catalog[name].gguf.license`);
  note e.g. `jina-reranker-v2-base-multilingual` is `cc-by-nc-4.0`.
