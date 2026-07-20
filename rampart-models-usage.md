# Using rampart-models

`rampart-models` downloads and locates GGUF and ONNX models, returning a path
that feeds directly into `rampart-llamacpp` / `rampart-onnx`:

```js
var models   = require('rampart-models');
var onnx     = require('rampart-onnx');
var llamacpp = require('rampart-llamacpp');

var emb = onnx.initEmbed(     models.onnxGet('bge-m3') );           // onnx: a DIRECTORY
var em2 = llamacpp.initEmbed( models.ggufGet('bge-m3') );           // gguf: a FILE
var gen = llamacpp.initGen(   models.ggufGet('qwen3-4b:q4_k_m') );  // exact quant
```

Everything is **synchronous**: the call blocks until the model is on disk and
returns the path.  If the model is already present under
`~/.rampart/models/<category>/`, the path is returned immediately with no
network access — so it is safe (and intended) to call at the top of any script.

It is a single pure-JS file (`rampart-models.js`) with the curated model
catalog embedded; it needs only `rampart-curl`.

---

## What you can ask for

| call | you get |
|---|---|
| `models.get('bge-m3')` | the onnx **directory** (default for embed/rerank models) |
| `models.get('bge-m3:q8_0')` | the Q8_0 **gguf file** (a `:quant` suffix always means gguf) |
| `models.get('qwen3-4b')` | a gguf file (gen models only exist as gguf; default quant) |
| `models.ggufGet('bge-m3')` | gguf file, default quant — **explicit**, use with llamacpp |
| `models.onnxGet('bge-m3')` | onnx directory — **explicit**, use with rampart-onnx |
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
2. **The embedded catalog** — ~70 curated embedding / reranker / generation
   models with pinned repos, commit revisions, and full quant maps.
   `models.list()` shows them; `models.catalog` is the raw data.
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
```

The path is printed to stdout (progress goes to the same terminal), so it
composes: `rampart run-something.js $(rampart rampart-models.js bge-m3)`.

## Other exports

```js
models.resolve('some-model')   // -> catalog-shaped entry {category, onnx:{...}, gguf:{...}}
                               //    without downloading anything
models.list()                  // -> { embed:[...], rerank:[...], gen:[...] }
models.catalog                 // the embedded catalog object
models.modelsDir               // "~/.rampart/models"
models.pull(...)               // alias of get()
```

## Updating the embedded catalog

The catalog section of `rampart-models.js` is generated — do not edit it by
hand.  To refresh it (new models, new quants, moved repos):

```
rampart gen-model-list.js
```

`gen-model-list.js` (same directory) sweeps the HuggingFace API by category,
resolves both formats per model with the same ranking heuristics the module
uses at runtime, and splices the result between the
`==== BEGIN/END GENERATED CATALOG ====` markers.  Its run report shows a diff
against the previous catalog and every ranking choice with runners-up; pin
corrections or add `"skip"` entries in its `OVERRIDES` table and rerun.
Repos are recorded with their commit sha, so catalog downloads use immutable
`resolve/{sha}/` URLs.

## Notes

- Generation models are gguf-only (rampart runs generation through llama.cpp).
- Vision/multimodal models (llava, moondream, `mmproj` projector files) are
  not in the catalog yet — fetch those with `models.url(...)` for now.
- Licenses ride along in the catalog (`models.catalog[name].gguf.license`);
  note e.g. `jina-reranker-v2-base-multilingual` is `cc-by-nc-4.0`.
