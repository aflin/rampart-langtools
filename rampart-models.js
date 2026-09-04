/* rampart-models.js -- download and locate GGUF / ONNX models.
 *
 *   var models = require('rampart-models');
 *
 *   models.get('bge-m3')                      -> .../embed/bge-m3/onnx/model_fp16.onnx  (onnx FILE)
 *   models.get('bge-m3:q8_0')                 -> ~/.rampart/models/embed/bge-m3-Q8_0.gguf (gguf FILE)
 *   models.get('qwen3-4b')                    -> gguf file (gen models are gguf-only)
 *   models.get('BAAI/bge-m3')                 -> exact HF repo, no search
 *   models.get('https://host/x.gguf')         -> plain download, returns the file
 *   models.ggufGet('bge-m3')  models.onnxGet('bge-m3')  -> format-explicit variants
 *
 * ONNX precision: onnx models default to fp16 (the GPU sweet spot).  Pass
 * {precision:'fp32'|'int8'|'q4'} for the full-precision original or a
 * CPU-oriented quantization (int8/q4 are smaller + faster on CPU, poor on the
 * CUDA EP).  Official repos usually ship only fp32; the variants come from the
 * onnx-community / Xenova mirrors, which getOnnx searches automatically.
 *   models.onnxGet('bge-m3')                    -> model_fp16.onnx  (default)
 *   models.onnxGet('bge-m3', {precision:'q4'})  -> model_q4.onnx    (CPU)
 *
 * The returned path feeds llamacpp/onnx initGen / initEmbed / initRerank
 * directly.  If the model already exists under ~/.rampart/models/<category>/
 * the path is returned immediately (no network).  Downloads stream to a
 * .part file with resume + retries and print single-line progress via
 * rampart.utils.fprintf (default stdout; opts.progress = false | fileHandle |
 * function(info)).
 *
 * Resolution order for a short name:
 *   1. already on disk            (~/.rampart/models/<category>/...)
 *   2. the model CATALOG          (built into this script, plus the fetched
 *                                  rampart-models-catalog.json overriding
 *                                  and extending it -- see "The model
 *                                  catalog" below)
 *   3. 'org/repo' passthrough     (name contains '/')
 *   4. live HuggingFace search    (same ranking heuristics as the generator:
 *                                  exact name ignoring punctuation, family-org
 *                                  first, then converter orgs, quant coverage)
 * Live resolutions are remembered in ~/.rampart/models/.resolved.json so a
 * name resolves the same way next time.
 *
 * Retrieval prompts: asymmetric embed models expect prefixes on queries and/or
 * documents (nomic's 'search_query: ', bge's instruction, ...).  The catalog
 * carries them per model ("prompts", from gen-model-list.js); org/repo and
 * live resolutions fetch them from the repo's config_sentence_transformers.json.
 * get() writes them as a sidecar next to the model file it returns:
 *     <model-file>.prompts.json   ->   { "prompts": { "query": "...",
 *                                        "document": "...", ... } }
 * so embed loaders (sql.set/initEmbed) pick them up with one stat.  Existing
 * on-disk models get the sidecar backfilled on their next get().  Models with
 * no prompts (all-minilm, bge-m3) get no sidecar and behave as before.
 *
 * options for get(name, opts):
 *   format    'onnx' | 'gguf'   (default: onnx when available, else gguf;
 *                                a ':quant' suffix implies gguf)
 *   quant     'Q4_K_M' etc      (same as the ':quant' suffix)
 *   category  'embed'|'rerank'|'gen'|...  (subdir; default from catalog, else
 *                                detected from the repo's HF pipeline tag
 *                                (text-generation -> gen, sentence-similarity/
 *                                feature-extraction -> embed, text-ranking ->
 *                                rerank), else 'embed'; URLs default to 'other')
 *   dest      exact destination file/dir (overrides category layout)
 *   progress  false | FILE handle | function(info)   (default: stdout)
 *   force     re-download even if present
 *   token     HuggingFace token (default: $HF_TOKEN) for gated repos
 *   revision  git revision (default: catalog-pinned sha, else 'main')
 *   confirm   function(info)->bool   called ONLY when a download is actually
 *                                needed (the model isn't already on disk), so
 *                                a caller can prompt first; return falsy to
 *                                skip -> get() returns null.  info = {name,
 *                                format, dest, size, bytes, precision|quant,
 *                                repo}.  Omit it for the default silent fetch.
 *
 * Other exports: models.list(), models.url(u, opts), models.resolve(name, opts),
 * models.catalogInfo() -> where the catalog came from and how current it is,
 * models.updateCatalog() -> force a refresh from the catalog URL,
 * models.variants(name[, opts]) -> [{quant, bytes, files, installed}] smallest
 * first (real file sizes for memory-aware selection, no download).
 * Quants: 'UD-*' (Unsloth Dynamic) is a distinct build, keyed with its
 * prefix; requesting the plain name errors with a did-you-mean unless
 * opts.allowVariant is true.
 * CLI:  rampart rampart-models.js <name|url> [gguf|onnx] [quant]
 *       rampart rampart-models.js --list
 *       rampart rampart-models.js --update   (refresh the catalog now)
 *
 * Only huggingface_hub's own stable URL patterns are used (api/models,
 * resolve/{rev}/) -- never CDN URLs.  $HF_ENDPOINT overrides the host.
 */

var u = rampart.utils;
var curl = require("rampart-curl");
var crypto = require("rampart-crypto");

var HF = u.getenv("HF_ENDPOINT") || "https://huggingface.co";

/* stdout/stderr are bare globals in current rampart; older builds expose them
 * only as rampart.utils.stdout/.stderr -- resolve whichever exists. */
var STDOUT = (typeof stdout !== "undefined") ? stdout : u.stdout;
var STDERR = (typeof stderr !== "undefined") ? stderr : (u.stderr || STDOUT);
/* The model store lives under the CURRENT user's home.  Resolve it on
 * every use, never at load time: a server started as root may require this
 * module while still root and only then drop privileges (rampart-server
 * repoints $HOME to the unprivileged user at the drop).  A value captured
 * here would keep pointing at root's home and fail with EACCES on the
 * first download. */
function modelsDir() { return u.homedir() + "/.rampart/models"; }

/* files copied for an onnx model directory, beyond the model itself */
var ONNX_AUX = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "config.json", "modules.json", "sentence_bert_config.json",
    "config_sentence_transformers.json", "vocab.txt", "merges.txt",
    "vocab.json", "sentencepiece.bpe.model", "spiece.model"
];
var ONNX_AUX_DIRS = /^1_Pooling\//;

/* ONNX precision -> model_*.onnx filename suffix (Xenova / onnx-community naming).
 * Default fp16 = the GPU sweet spot (tensor cores, ~half the fp32 size).  int8/q4 are
 * CPU-oriented -- smaller and usually FASTER on the CPU's quantized-matmul kernels, but
 * they map poorly to ORT's CUDA EP (CPU-fallback/memcpy), so prefer them only for CPU.
 * fp32 = the full-precision original.  Official model repos usually ship only fp32; the
 * variants live in the converter mirrors (onnx-community / Xenova), which getOnnx also
 * searches. */
var ONNX_PRECISION = {
    "fp16": "_fp16", "f16": "_fp16",
    "fp32": "",      "f32": "",      "full": "",
    "int8": "_int8", "i8": "_int8",  "uint8": "_uint8",
    "q4":   "_q4",   "q4f16": "_q4f16"
};

/* converter/mirror orgs for ranking live-search results (generator parity) */
var CONVERTER_ORGS = [
    "ggml-org", "unsloth", "bartowski", "lmstudio-community", "gpustack",
    "second-state", "onnx-community", "Xenova", "CompendiumLabs", "cstr",
    "leliuga", "mradermacher", "TheBloke"
];

/* ------------------------------------------------------------------ *
 * The model catalog
 *
 * The catalog ships in two halves.
 *
 *   BUILT-IN   the GENERATED BUILTIN CATALOG block above: clip, ocr and
 *              rerank models plus the stable embed/gen workhorses.  These
 *              resolve with NO network and NO cache -- if github is
 *              unreachable, asking for them still works, on disk or by
 *              download straight from their pinned repos.
 *   FETCHED    rampart-models-catalog.json, downloaded and cached under
 *              the model store, refreshed with a conditional GET (ETag)
 *              each time this module loads.  It carries the whole catalog,
 *              built-ins included, and each entry OVERRIDES the built-in
 *              of the same name -- so a moved repo, a re-pinned revision
 *              or a new quant reaches installed scripts without a release,
 *              and newly discovered or hand-added models arrive the same
 *              way.  It is NOT installed with the module, by design.
 *
 * Fetched source order:
 *   1. rampart-models-catalog.json BESIDE this script -- a repo checkout
 *      or a hand-placed pin.  Never present in an installed tree (the
 *      file is deliberately not installed), so this is the development
 *      path: regenerate, test, then push.
 *   2. ~/.rampart/models/.rampart-models-catalog.json -- the cache,
 *      revalidated against the repo (304 = keep, 200 = replace).
 *   3. Network only, when there is no cache yet (first run).
 * A refresh that fails is not fatal: the cached copy keeps working
 * offline, and with no cache either the built-ins still resolve (a warning
 * is printed once, and unknown names fall through to live HuggingFace
 * resolution, tier 4).
 *
 * Environment:
 *   RAMPART_MODELS_CATALOG_URL      override the source URL (mirror, or a
 *                                   file:///... path for an air gap)
 *   RAMPART_MODELS_CATALOG_TTL      seconds between revalidations
 *                                   (default 0 = check on every load)
 *   RAMPART_MODELS_CATALOG_OFFLINE  never touch the network; cache only
 * ------------------------------------------------------------------ */
var CATALOG_FILE = "rampart-models-catalog.json";
var CATALOG_URL  = u.getenv("RAMPART_MODELS_CATALOG_URL") ||
    "https://raw.githubusercontent.com/aflin/rampart-langtools/main/" + CATALOG_FILE;
var CATALOG_MIN  = 20;          /* sanity floor: a real catalog is ~100 */
var CATALOG_STALE_DAYS = 30;    /* warn when a cache this old can't refresh */

/* populated by initCatalog() below, before anything reads it */
var CATALOG = {};
var catalogState = {
    source: "none",   /* local | cache | network | none */
    url: CATALOG_URL,
    builtin: 0,       /* models compiled into this script */
    entries: 0,       /* models in the fetched catalog (0 if unavailable) */
    models: 0,        /* effective total after the fetched layer is merged */
    version: null,
    generated: null,  /* when the catalog was generated (from the file) */
    fetched: null,    /* when this machine last downloaded it */
    checked: null,    /* when this machine last revalidated it */
    etag: null,
    error: null       /* why the last refresh didn't happen, if it didn't */
};

/* ==== BEGIN GENERATED BUILTIN CATALOG (do not edit; run gen-model-list.js) ==== */
var BUILTIN_CATALOG = {
    "all-minilm-l6-v2": {"category":"embed","onnx":{"repo":"sentence-transformers/all-MiniLM-L6-v2","revision":"1110a243fdf4706b3f48f1d95db1a4f5529b4d41","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"second-state/All-MiniLM-L6-v2-Embedding-GGUF","revision":"544f204f2eaa2d71361ffc74d6df7170285b286a","quants":{"Q2_K":{"file":"all-MiniLM-L6-v2-Q2_K.gguf","size":19229632},"Q3_K_L":{"file":"all-MiniLM-L6-v2-Q3_K_L.gguf","size":20473792},"Q3_K_M":{"file":"all-MiniLM-L6-v2-Q3_K_M.gguf","size":19939264},"Q3_K_S":{"file":"all-MiniLM-L6-v2-Q3_K_S.gguf","size":19229632},"Q4_0":{"file":"all-MiniLM-L6-v2-Q4_0.gguf","size":19699648},"Q4_K_M":{"file":"all-MiniLM-L6-v2-Q4_K_M.gguf","size":20999104},"Q4_K_S":{"file":"all-MiniLM-L6-v2-Q4_K_S.gguf","size":20694976},"Q5_0":{"file":"all-MiniLM-L6-v2-Q5_0.gguf","size":21026752},"Q5_K_M":{"file":"all-MiniLM-L6-v2-Q5_K_M.gguf","size":21717952},"Q5_K_S":{"file":"all-MiniLM-L6-v2-Q5_K_S.gguf","size":21469120},"Q6_K":{"file":"all-MiniLM-L6-v2-Q6_K.gguf","size":24150976},"Q8_0":{"file":"all-MiniLM-L6-v2-Q8_0.gguf","size":25008064},"F16":{"file":"all-MiniLM-L6-v2-ggml-model-f16.gguf","size":45949216}},"license":"apache-2.0"}},
    "bge-small-en-v1.5": {"category":"embed","onnx":{"repo":"BAAI/bge-small-en-v1.5","revision":"5c38ec7c405ec4b44b94cc5a9bb96e735b38267a","model":"onnx/model.onnx","license":"mit","dim":384},"gguf":{"repo":"CompendiumLabs/bge-small-en-v1.5-gguf","revision":"d32f8c040ea3b516330eeb75b72bcc2d3a780ab7","quants":{"F16":{"file":"bge-small-en-v1.5-f16.gguf","size":67308128},"F32":{"file":"bge-small-en-v1.5-f32.gguf","size":133609568},"Q4_K_M":{"file":"bge-small-en-v1.5-q4_k_m.gguf","size":24808576},"Q8_0":{"file":"bge-small-en-v1.5-q8_0.gguf","size":36806944}},"license":"mit"},"prompts":{"query":"Represent this sentence for searching relevant passages: "}},
    "bge-m3": {"category":"embed","onnx":{"repo":"BAAI/bge-m3","revision":"5617a9f61b028005a4858fdac845db406aefb181","model":"onnx/model.onnx","license":"mit","dim":1024},"gguf":{"repo":"gpustack/bge-m3-GGUF","revision":"2d48f1737679ad900d5c26c5aad5410e9c70fdca","quants":{"FP16":{"file":"bge-m3-FP16.gguf","size":1157671200},"Q2_K":{"file":"bge-m3-Q2_K.gguf","size":366114880},"Q3_K":{"file":"bge-m3-Q3_K.gguf","size":402290752},"Q4_0":{"file":"bge-m3-Q4_0.gguf","size":421558336},"Q4_K_M":{"file":"bge-m3-Q4_K_M.gguf","size":437778496},"Q5_0":{"file":"bge-m3-Q5_0.gguf","size":459307072},"Q5_K_M":{"file":"bge-m3-Q5_K_M.gguf","size":467662912},"Q6_K":{"file":"bge-m3-Q6_K.gguf","size":499415104},"Q8_0":{"file":"bge-m3-Q8_0.gguf","size":634553760}},"license":"mit"}},
    "all-mpnet-base-v2": {"category":"embed","onnx":{"repo":"sentence-transformers/all-mpnet-base-v2","revision":"e8c3b32edf5434bc2275fc9bab85f82640a19130","model":"onnx/model.onnx","license":"apache-2.0","dim":768},"gguf":{"repo":"cstr/all-mpnet-base-v2-GGUF","revision":"413030ffeb1e47954094742eaf0f4454b4ca919b","quants":{"IQ4_XS":{"file":"all-mpnet-base-v2-iq4_xs.gguf","size":71576672},"DEFAULT":{"file":"all-mpnet-base-v2-q4_k-imatrix.gguf","size":74230880},"Q4_K":{"file":"all-mpnet-base-v2-q4_k.gguf","size":74230880},"Q8_0":{"file":"all-mpnet-base-v2-q8_0.gguf","size":116698208}},"license":"apache-2.0"}},
    "nomic-embed-text-v1.5": {"category":"embed","onnx":{"repo":"nomic-ai/nomic-embed-text-v1.5","revision":"e9b6763023c676ca8431644204f50c2b100d9aab","model":"onnx/model.onnx","license":"apache-2.0","dim":768},"gguf":{"repo":"nomic-ai/nomic-embed-text-v1.5-GGUF","revision":"0188c9bf409793f810680a5a431e7b899c46104c","quants":{"Q2_K":{"file":"nomic-embed-text-v1.5.Q2_K.gguf","size":49361088},"Q3_K_L":{"file":"nomic-embed-text-v1.5.Q3_K_L.gguf","size":71593088},"Q3_K_M":{"file":"nomic-embed-text-v1.5.Q3_K_M.gguf","size":67169408},"Q3_K_S":{"file":"nomic-embed-text-v1.5.Q3_K_S.gguf","size":59649152},"Q4_0":{"file":"nomic-embed-text-v1.5.Q4_0.gguf","size":77802880},"Q4_K_M":{"file":"nomic-embed-text-v1.5.Q4_K_M.gguf","size":84106624},"Q4_K_S":{"file":"nomic-embed-text-v1.5.Q4_K_S.gguf","size":78097792},"Q5_0":{"file":"nomic-embed-text-v1.5.Q5_0.gguf","size":94888768},"Q5_K_M":{"file":"nomic-embed-text-v1.5.Q5_K_M.gguf","size":99588928},"Q5_K_S":{"file":"nomic-embed-text-v1.5.Q5_K_S.gguf","size":94888768},"Q6_K":{"file":"nomic-embed-text-v1.5.Q6_K.gguf","size":113042528},"Q8_0":{"file":"nomic-embed-text-v1.5.Q8_0.gguf","size":146146432},"F16":{"file":"nomic-embed-text-v1.5.f16.gguf","size":274290560},"F32":{"file":"nomic-embed-text-v1.5.f32.gguf","size":547664768}},"license":"apache-2.0"},"prompts":{"query":"search_query: ","document":"search_document: "}},
    "multilingual-e5-small": {"category":"embed","onnx":{"repo":"intfloat/multilingual-e5-small","revision":"614241f622f53c4eeff9890bdc4f31cfecc418b3","model":"onnx/model.onnx","license":"mit","dim":384},"prompts":{"query":"query: ","document":"passage: "}},
    "bge-large-en-v1.5": {"category":"embed","onnx":{"repo":"BAAI/bge-large-en-v1.5","revision":"d4aa6901d3a41ba39fb536a557fa166f842b0e09","model":"onnx/model.onnx","license":"mit","dim":1024},"gguf":{"repo":"CompendiumLabs/bge-large-en-v1.5-gguf","revision":"03b9bf107964236f1de7c217f866d1bf27ea677d","quants":{"F16":{"file":"bge-large-en-v1.5-f16.gguf","size":669603712},"F32":{"file":"bge-large-en-v1.5-f32.gguf","size":1337141120},"Q4_K_M":{"file":"bge-large-en-v1.5-q4_k_m.gguf","size":207833664},"Q8_0":{"file":"bge-large-en-v1.5-q8_0.gguf","size":358235712}},"license":"mit"},"prompts":{"query":"Represent this sentence for searching relevant passages: "}},
    "bge-base-en-v1.5": {"category":"embed","onnx":{"repo":"BAAI/bge-base-en-v1.5","revision":"a5beb1e3e68b9ab74eb54cfd186867f64f240e1a","model":"onnx/model.onnx","license":"mit","dim":768},"gguf":{"repo":"CompendiumLabs/bge-base-en-v1.5-gguf","revision":"24914efdfa9ee54e815c3fcaa78a617031251c5c","quants":{"F16":{"file":"bge-base-en-v1.5-f16.gguf","size":218789984},"F32":{"file":"bge-base-en-v1.5-f32.gguf","size":436327520},"Q4_K_M":{"file":"bge-base-en-v1.5-q4_k_m.gguf","size":68348448},"Q8_0":{"file":"bge-base-en-v1.5-q8_0.gguf","size":117974304}},"license":"mit"},"prompts":{"query":"Represent this sentence for searching relevant passages: "}},
    "qwen3-embedding-0.6b": {"category":"embed","onnx":{"repo":"onnx-community/Qwen3-Embedding-0.6B-ONNX","revision":"c25a394dd583836952667c12f008335071b3f43d","model":"onnx/model.onnx","license":null,"dim":1024},"gguf":{"repo":"mradermacher/Qwen3-Embedding-0.6B-GGUF","revision":"8c605f43dcb0b43cf6e4afc7203888d912a67ace","quants":{"IQ4_XS":{"file":"Qwen3-Embedding-0.6B.IQ4_XS.gguf","size":369048224},"Q2_K":{"file":"Qwen3-Embedding-0.6B.Q2_K.gguf","size":296008352},"Q3_K_L":{"file":"Qwen3-Embedding-0.6B.Q3_K_L.gguf","size":368261792},"Q3_K_M":{"file":"Qwen3-Embedding-0.6B.Q3_K_M.gguf","size":346897056},"Q3_K_S":{"file":"Qwen3-Embedding-0.6B.Q3_K_S.gguf","size":322845344},"Q4_K_M":{"file":"Qwen3-Embedding-0.6B.Q4_K_M.gguf","size":396475040},"Q4_K_S":{"file":"Qwen3-Embedding-0.6B.Q4_K_S.gguf","size":383040160},"Q5_K_M":{"file":"Qwen3-Embedding-0.6B.Q5_K_M.gguf","size":444185248},"Q5_K_S":{"file":"Qwen3-Embedding-0.6B.Q5_K_S.gguf","size":436386464},"Q6_K":{"file":"Qwen3-Embedding-0.6B.Q6_K.gguf","size":494877344},"Q8_0":{"file":"Qwen3-Embedding-0.6B.Q8_0.gguf","size":639151072},"F16":{"file":"Qwen3-Embedding-0.6B.f16.gguf","size":1197630112}},"license":"apache-2.0"},"prompts":{"query":"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"}},
    "multilingual-e5-large": {"category":"embed","onnx":{"repo":"intfloat/multilingual-e5-large","revision":"3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3","model":"onnx/model.onnx","license":"mit","dim":1024},"gguf":{"repo":"phate334/multilingual-e5-large-gguf","revision":"080da2e028f5efe9f584b62bfd4f98e3b34ec916","quants":{"F16":{"file":"multilingual-e5-large-f16.gguf","size":1126215072},"Q4_K_M":{"file":"multilingual-e5-large-q4_k_m.gguf","size":406322368}},"license":"mit"},"prompts":{"query":"query: ","document":"passage: "}},
    "multilingual-e5-base": {"category":"embed","onnx":{"repo":"intfloat/multilingual-e5-base","revision":"d128750597153bb5987e10b1c3493a34e5a4502a","model":"onnx/model.onnx","license":"mit","dim":768},"gguf":{"repo":"dinab/multilingual-e5-base-Q4_K_M-GGUF","revision":"bef187287ba4f8cd4b6acfa8e501e45c89aca753","quants":{"Q4_K_M":{"file":"multilingual-e5-base-q4_k_m.gguf","size":218732864}},"license":"mit"},"prompts":{"query":"query: ","document":"passage: "}},
    "mxbai-embed-large-v1": {"category":"embed","onnx":{"repo":"mixedbread-ai/mxbai-embed-large-v1","revision":"b33106f585b9ce46904ad7443a3b52b7a63e231c","model":"onnx/model.onnx","license":"apache-2.0","dim":1024},"gguf":{"repo":"mixedbread-ai/mxbai-embed-large-v1","revision":"b33106f585b9ce46904ad7443a3b52b7a63e231c","quants":{"F16":{"file":"gguf/mxbai-embed-large-v1-f16.gguf","size":669603712}},"license":"apache-2.0"},"prompts":{"query":"Represent this sentence for searching relevant passages: "}},
    "all-minilm-l12-v2": {"category":"embed","onnx":{"repo":"sentence-transformers/all-MiniLM-L12-v2","revision":"a50ef00143b4d5391434df20ae11632588ac25be","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"cstr/all-MiniLM-L12-v2-GGUF","revision":"7fe71bd6429728bd2a4d384cadb53a48984de8f5","quants":{"IQ4_XS":{"file":"all-MiniLM-L12-v2-iq4_xs.gguf","size":25171328},"DEFAULT":{"file":"all-MiniLM-L12-v2-q4_k-imatrix.gguf","size":25392512},"Q4_K":{"file":"all-MiniLM-L12-v2-q4_k.gguf","size":25390272},"Q8_0":{"file":"all-MiniLM-L12-v2-q8_0.gguf","size":36080832}},"license":"apache-2.0"}},
    "jina-embeddings-v3": {"category":"embed","onnx":{"repo":"jinaai/jina-embeddings-v3","revision":"ab036b023d30b4d1138c4c3bfa9f0c445ab455d6","model":"onnx/model.onnx","license":"cc-by-nc-4.0","dim":1024},"gguf":{"repo":"second-state/jina-embeddings-v3-GGUF","revision":"61b5399b6dab4af55fd305574625d88f50241ca8","quants":{"Q2_K":{"file":"jina-embeddings-v3-Q2_K.gguf","size":330000640},"Q3_K_L":{"file":"jina-embeddings-v3-Q3_K_L.gguf","size":390359296},"Q3_K_M":{"file":"jina-embeddings-v3-Q3_K_M.gguf","size":375154944},"Q3_K_S":{"file":"jina-embeddings-v3-Q3_K_S.gguf","size":347891968},"Q4_0":{"file":"jina-embeddings-v3-Q4_0.gguf","size":388000000},"Q4_K_M":{"file":"jina-embeddings-v3-Q4_K_M.gguf","size":410413312},"Q4_K_S":{"file":"jina-embeddings-v3-Q4_K_S.gguf","size":389572864},"Q5_0":{"file":"jina-embeddings-v3-Q5_0.gguf","size":425748736},"Q5_K_M":{"file":"jina-embeddings-v3-Q5_K_M.gguf","size":442460416},"Q5_K_S":{"file":"jina-embeddings-v3-Q5_K_S.gguf","size":425748736},"Q6_K":{"file":"jina-embeddings-v3-Q6_K.gguf","size":465856768},"Q8_0":{"file":"jina-embeddings-v3-Q8_0.gguf","size":600995424},"F16":{"file":"lora-classification-jina-embeddings-v3-f16.gguf","size":10338016}},"license":null}},
    "ms-marco-minilm-l6-v2": {"category":"rerank","onnx":{"repo":"cross-encoder/ms-marco-MiniLM-L6-v2","revision":"233902d25c440f23af6f7d6e94d2946bac0bee0a","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"cstr/ms-marco-MiniLM-L-6-v2-GGUF","revision":"1a9ef5ce8cb08936338233731314f3ff61ce0930","quants":{"DEFAULT":{"file":"ms-marco-MiniLM-L-6-v2-q4_k-imatrix.gguf","size":19394624},"IQ4_XS":{"file":"ms-marco-MiniLM-L-6-v2-iq4_xs.gguf","size":19284032},"Q4_K":{"file":"ms-marco-MiniLM-L-6-v2-q4_k.gguf","size":19394624},"Q8_0":{"file":"ms-marco-MiniLM-L-6-v2-q8_0.gguf","size":24703040}},"license":"apache-2.0"}},
    "ms-marco-minilm-l4-v2": {"category":"rerank","onnx":{"repo":"cross-encoder/ms-marco-MiniLM-L4-v2","revision":"777b2f369bc1c2f850df8bd367ed1654bda4497b","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"mradermacher/ms-marco-MiniLM-L4-v2-GGUF","revision":"3c271e62e3e247e1a0ca5278a9ca531f307adc1c","quants":{"IQ4_XS":{"file":"ms-marco-MiniLM-L4-v2.IQ4_XS.gguf","size":17866912},"Q2_K":{"file":"ms-marco-MiniLM-L4-v2.Q2_K.gguf","size":17627296},"Q3_K_L":{"file":"ms-marco-MiniLM-L4-v2.Q3_K_L.gguf","size":18456736},"Q3_K_M":{"file":"ms-marco-MiniLM-L4-v2.Q3_K_M.gguf","size":18106528},"Q3_K_S":{"file":"ms-marco-MiniLM-L4-v2.Q3_K_S.gguf","size":17627296},"Q4_K_M":{"file":"ms-marco-MiniLM-L4-v2.Q4_K_M.gguf","size":18945184},"Q4_K_S":{"file":"ms-marco-MiniLM-L4-v2.Q4_K_S.gguf","size":18567328},"Q5_K_M":{"file":"ms-marco-MiniLM-L4-v2.Q5_K_M.gguf","size":19369120},"Q5_K_S":{"file":"ms-marco-MiniLM-L4-v2.Q5_K_S.gguf","size":19120288},"Q6_K":{"file":"ms-marco-MiniLM-L4-v2.Q6_K.gguf","size":20908192},"Q8_0":{"file":"ms-marco-MiniLM-L4-v2.Q8_0.gguf","size":21479584},"F16":{"file":"ms-marco-MiniLM-L4-v2.f16.gguf","size":39103008}},"license":"apache-2.0"}},
    "qwen3-reranker-4b": {"category":"rerank"},
    "qwen3-reranker-0.6b": {"category":"rerank","gguf":{"repo":"ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF","revision":"a02f48bb4f057028298c21fa033da2b30d7742d5","quants":{"Q8_0":{"file":"qwen3-reranker-0.6b-q8_0.gguf","size":639153184}},"license":"apache-2.0"}},
    "ms-marco-minilm-l12-v2": {"category":"rerank","onnx":{"repo":"cross-encoder/ms-marco-MiniLM-L12-v2","revision":"7b0235231ca2674cb8ca8f022859a6eba2b1c968","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"cstr/ms-marco-MiniLM-L-12-v2-GGUF","revision":"2b6301cfa22006cb9eb5563c2e673e7f3976f00e","quants":{"DEFAULT":{"file":"ms-marco-MiniLM-L-12-v2-q4_k-imatrix.gguf","size":25491872},"IQ4_XS":{"file":"ms-marco-MiniLM-L-12-v2-iq4_xs.gguf","size":25270688},"Q4_K":{"file":"ms-marco-MiniLM-L-12-v2-q4_k.gguf","size":25491872},"Q8_0":{"file":"ms-marco-MiniLM-L-12-v2-q8_0.gguf","size":36108704}},"license":"apache-2.0"}},
    "gte-reranker-modernbert-base": {"category":"rerank","onnx":{"repo":"Alibaba-NLP/gte-reranker-modernbert-base","revision":"f7481e6055501a30fb19d090657df9ec1f79ab2c","model":"onnx/model.onnx","license":"apache-2.0","dim":768},"gguf":{"repo":"jolleyboy/gte-reranker-modernbert-base-GGUF","revision":"574c2d4ed7565a72b965d491cc70f284292de864","quants":{"Q4_K_M":{"file":"gte-reranker-modernbert-base-Q4_K_M.gguf","size":106340352},"Q6_K":{"file":"gte-reranker-modernbert-base-Q6_K.gguf","size":129329664},"Q8_0":{"file":"gte-reranker-modernbert-base-Q8_0.gguf","size":160839552},"F16":{"file":"gte-reranker-modernbert-base-f16.gguf","size":301060992}},"license":"apache-2.0"}},
    "mmarco-mminilmv2-l12-h384-v1": {"category":"rerank","onnx":{"repo":"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1","revision":"1427fd652930e4ba29e8149678df786c240d8825","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"mono-of-pg/mmarco-mMiniLMv2-L12-H384-v1-gguf","revision":"704d22d515a70f0e42f7b7c70fc7b3fd74b59c7c","quants":{"F32":{"file":"mMiniLMv2-L12-H384-distilled-from-XLMR-Large-F32.gguf","size":477394048}},"license":"apache-2.0"}},
    "ms-marco-minilm-l2-v2": {"category":"rerank","onnx":{"repo":"cross-encoder/ms-marco-MiniLM-L2-v2","revision":"1b5cd67b15209f24824c50370e0397743aa9b787","model":"onnx/model.onnx","license":"apache-2.0","dim":384},"gguf":{"repo":"mradermacher/ms-marco-MiniLM-L2-v2-GGUF","revision":"050c4fef72897a1ac89d1f5b473a32c4b7d026af","quants":{"IQ4_XS":{"file":"ms-marco-MiniLM-L2-v2.IQ4_XS.gguf","size":15871360},"Q2_K":{"file":"ms-marco-MiniLM-L2-v2.Q2_K.gguf","size":15751552},"Q3_K_L":{"file":"ms-marco-MiniLM-L2-v2.Q3_K_L.gguf","size":16166272},"Q3_K_M":{"file":"ms-marco-MiniLM-L2-v2.Q3_K_M.gguf","size":16000384},"Q3_K_S":{"file":"ms-marco-MiniLM-L2-v2.Q3_K_S.gguf","size":15751552},"Q4_K_M":{"file":"ms-marco-MiniLM-L2-v2.Q4_K_M.gguf","size":16410496},"Q4_K_S":{"file":"ms-marco-MiniLM-L2-v2.Q4_K_S.gguf","size":16221568},"Q5_K_M":{"file":"ms-marco-MiniLM-L2-v2.Q5_K_M.gguf","size":16622464},"Q5_K_S":{"file":"ms-marco-MiniLM-L2-v2.Q5_K_S.gguf","size":16498048},"Q6_K":{"file":"ms-marco-MiniLM-L2-v2.Q6_K.gguf","size":17392000},"Q8_0":{"file":"ms-marco-MiniLM-L2-v2.Q8_0.gguf","size":17677696},"F16":{"file":"ms-marco-MiniLM-L2-v2.f16.gguf","size":31983360}},"license":"apache-2.0"}},
    "jina-reranker-v2-base-multilingual": {"category":"rerank","onnx":{"repo":"jinaai/jina-reranker-v2-base-multilingual","revision":"9cfeff2df7d40d1b78e75e5e9cebec92a99813c9","model":"onnx/model.onnx","license":"cc-by-nc-4.0","dim":768},"gguf":{"repo":"gpustack/jina-reranker-v2-base-multilingual-GGUF","revision":"09a0e5b9f3d193a4f1e771ba6ceccdf1153d3a9a","quants":{"FP16":{"file":"jina-reranker-v2-base-multilingual-FP16.gguf","size":565520224},"Q2_K":{"file":"jina-reranker-v2-base-multilingual-Q2_K.gguf","size":199626400},"Q3_K":{"file":"jina-reranker-v2-base-multilingual-Q3_K.gguf","size":212238496},"Q4_0":{"file":"jina-reranker-v2-base-multilingual-Q4_0.gguf","size":216076960},"Q4_K_M":{"file":"jina-reranker-v2-base-multilingual-Q4_K_M.gguf","size":222380704},"Q5_0":{"file":"jina-reranker-v2-base-multilingual-Q5_0.gguf","size":226767520},"Q5_K_M":{"file":"jina-reranker-v2-base-multilingual-Q5_K_M.gguf","size":231467680},"Q6_K":{"file":"jina-reranker-v2-base-multilingual-Q6_K.gguf","size":238126240},"Q8_0":{"file":"jina-reranker-v2-base-multilingual-Q8_0.gguf","size":305339552}},"license":"cc-by-nc-4.0"}},
    "jina-reranker-v3": {"category":"rerank","gguf":{"repo":"jinaai/jina-reranker-v3-GGUF","revision":"4bbace80cf59987f6fec850519012341c06810d5","quants":{"BF16":{"file":"jina-reranker-v3-BF16.gguf","size":1198785888},"IQ1_M":{"file":"jina-reranker-v3-IQ1_M.gguf","size":216655456},"IQ1_S":{"file":"jina-reranker-v3-IQ1_S.gguf","size":208619104},"IQ2_M":{"file":"jina-reranker-v3-IQ2_M.gguf","size":265512544},"IQ2_XXS":{"file":"jina-reranker-v3-IQ2_XXS.gguf","size":230049376},"IQ3_M":{"file":"jina-reranker-v3-IQ3_M.gguf","size":336630368},"IQ3_S":{"file":"jina-reranker-v3-IQ3_S.gguf","size":323678816},"IQ3_XS":{"file":"jina-reranker-v3-IQ3_XS.gguf","size":313356896},"IQ3_XXS":{"file":"jina-reranker-v3-IQ3_XXS.gguf","size":279619168},"IQ4_NL":{"file":"jina-reranker-v3-IQ4_NL.gguf","size":382169696},"IQ4_XS":{"file":"jina-reranker-v3-IQ4_XS.gguf","size":368407136},"Q2_K":{"file":"jina-reranker-v3-Q2_K.gguf","size":296841824},"Q3_K_M":{"file":"jina-reranker-v3-Q3_K_M.gguf","size":347730528},"Q4_K_M":{"file":"jina-reranker-v3-Q4_K_M.gguf","size":397308512},"Q5_K_M":{"file":"jina-reranker-v3-Q5_K_M.gguf","size":445018720},"Q5_K_S":{"file":"jina-reranker-v3-Q5_K_S.gguf","size":437219936},"Q6_K":{"file":"jina-reranker-v3-Q6_K.gguf","size":495710816},"Q8_0":{"file":"jina-reranker-v3-Q8_0.gguf","size":640050528}},"license":"cc-by-nc-4.0"}},
    "gte-multilingual-reranker-base": {"category":"rerank","onnx":{"repo":"onnx-community/gte-multilingual-reranker-base","revision":"ee64367e35a2db0da46bb6497e13a18f8bd585cb","model":"onnx/model.onnx","license":null,"dim":768},"gguf":{"repo":"gpustack/gte-multilingual-reranker-base-GGUF","revision":"ca8f873f19fa20a4c2051166acbcf39c7cbe32a0","quants":{"FP16":{"file":"gte-multilingual-reranker-base-FP16.gguf","size":618922176},"Q2_K":{"file":"gte-multilingual-reranker-base-Q2_K.gguf","size":205653312},"Q3_K":{"file":"gte-multilingual-reranker-base-Q3_K.gguf","size":221140800},"Q4_0":{"file":"gte-multilingual-reranker-base-Q4_0.gguf","size":228739392},"Q4_K_M":{"file":"gte-multilingual-reranker-base-Q4_K_M.gguf","size":235043136},"Q5_0":{"file":"gte-multilingual-reranker-base-Q5_0.gguf","size":242968896},"Q5_K_M":{"file":"gte-multilingual-reranker-base-Q5_K_M.gguf","size":247669056},"Q6_K":{"file":"gte-multilingual-reranker-base-Q6_K.gguf","size":258087744},"Q8_0":{"file":"gte-multilingual-reranker-base-Q8_0.gguf","size":332166336}},"license":"apache-2.0"}},
    "qwen3-coder-30b-a3b-instruct": {"category":"gen","gguf":{"repo":"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF","revision":"b17cb02dd882d5b6ab62fc777ad2995f19668350","quants":{"IQ4_NL":{"file":"Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf","size":17310784672},"IQ4_XS":{"file":"Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf","size":16378076320},"Q2_K":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf","size":11258612896},"Q2_K_L":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf","size":11331542176},"Q3_K_M":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf","size":14711850144},"Q3_K_S":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q3_K_S.gguf","size":13292471456},"Q4_0":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf","size":17379990688},"Q4_1":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q4_1.gguf","size":19192503456},"Q4_K_M":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf","size":18556689568},"Q4_K_S":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf","size":17456012448},"Q5_K_M":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf","size":21725584544},"Q5_K_S":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q5_K_S.gguf","size":21080513696},"Q6_K":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf","size":25092535456},"Q8_0":{"file":"Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf","size":32483935392},"UD-IQ1_M":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf","size":9627540640},"UD-IQ1_S":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf","size":8914328736},"UD-IQ2_M":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_M.gguf","size":10837007520},"UD-IQ2_XXS":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf","size":10333691040},"UD-IQ3_XXS":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf","size":12848766112},"UD-Q2_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q2_K_XL.gguf","size":11788590240},"UD-Q3_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf","size":13806312608},"UD-Q4_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf","size":17665334432},"UD-Q5_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf","size":21740305568},"UD-Q6_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf","size":26340328608},"UD-Q8_K_XL":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-Q8_K_XL.gguf","size":35989947552},"DEFAULT":{"file":"Qwen3-Coder-30B-A3B-Instruct-UD-TQ1_0.gguf","size":8005213344},"BF16":{"file":"BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf","size":61095806048,"parts":[{"file":"BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf","size":49655154016},{"file":"BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00002-of-00002.gguf","size":11440652032}]}},"license":"apache-2.0"}},
    "gpt-oss-20b": {"category":"gen","gguf":{"repo":"ggml-org/gpt-oss-20b-GGUF","revision":"ef9b12f2ff56c69cf32153a02784e7a3c88bf524","quants":{"MXFP4":{"file":"gpt-oss-20b-MXFP4.gguf","size":12109566624}},"license":"apache-2.0"}},
    "llama-3.1-8b-instruct": {"category":"gen","gguf":{"repo":"bartowski/Meta-Llama-3.1-8B-Instruct-GGUF","revision":"bf5b95e96dac0462e2a09145ec66cae9a3f12067","quants":{"IQ2_M":{"file":"Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf","size":2948285856},"IQ3_M":{"file":"Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf","size":3784828320},"IQ3_XS":{"file":"Meta-Llama-3.1-8B-Instruct-IQ3_XS.gguf","size":3518752160},"IQ4_NL":{"file":"Meta-Llama-3.1-8B-Instruct-IQ4_NL.gguf","size":4677993888},"IQ4_XS":{"file":"Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf","size":4447667616},"Q2_K":{"file":"Meta-Llama-3.1-8B-Instruct-Q2_K.gguf","size":3179136416},"Q2_K_L":{"file":"Meta-Llama-3.1-8B-Instruct-Q2_K_L.gguf","size":3692160416},"Q3_K_L":{"file":"Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf","size":4321961376},"Q3_K_M":{"file":"Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf","size":4018922912},"Q3_K_S":{"file":"Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf","size":3664504224},"Q3_K_XL":{"file":"Meta-Llama-3.1-8B-Instruct-Q3_K_XL.gguf","size":4781630880},"Q4_0_4_4":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf","size":4661216672},"Q4_0_4_8":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_0_4_8.gguf","size":4661216672},"Q4_0_8_8":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_0_8_8.gguf","size":4661216672},"Q4_K_L":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf","size":5310637472},"Q4_K_M":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf","size":4920739232},"Q4_K_S":{"file":"Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf","size":4692673952},"Q5_K_L":{"file":"Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf","size":6057223584},"Q5_K_M":{"file":"Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf","size":5732992416},"Q5_K_S":{"file":"Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf","size":5599298976},"Q6_K":{"file":"Meta-Llama-3.1-8B-Instruct-Q6_K.gguf","size":6596011424},"Q6_K_L":{"file":"Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf","size":6850471328},"Q8_0":{"file":"Meta-Llama-3.1-8B-Instruct-Q8_0.gguf","size":8540775840},"F32":{"file":"Meta-Llama-3.1-8B-Instruct-f32.gguf","size":32128885888}},"license":"llama3.1"}},
    "qwen3-4b": {"category":"gen","gguf":{"repo":"Qwen/Qwen3-4B-GGUF","revision":"bc640142c66e1fdd12af0bd68f40445458f3869b","quants":{"Q4_K_M":{"file":"Qwen3-4B-Q4_K_M.gguf","size":2497280256},"Q5_0":{"file":"Qwen3-4B-Q5_0.gguf","size":2823710976},"Q5_K_M":{"file":"Qwen3-4B-Q5_K_M.gguf","size":2889513184},"Q6_K":{"file":"Qwen3-4B-Q6_K.gguf","size":3306260704},"Q8_0":{"file":"Qwen3-4B-Q8_0.gguf","size":4280404704}},"license":"apache-2.0"}},
    "qwen3-8b": {"category":"gen","gguf":{"repo":"Qwen/Qwen3-8B-GGUF","revision":"7c41481f57cb95916b40956ab2f0b139b296d974","quants":{"Q4_K_M":{"file":"Qwen3-8B-Q4_K_M.gguf","size":5027783488},"Q5_0":{"file":"Qwen3-8B-Q5_0.gguf","size":5720761152},"Q5_K_M":{"file":"Qwen3-8B-Q5_K_M.gguf","size":5851112224},"Q6_K":{"file":"Qwen3-8B-Q6_K.gguf","size":6725899040},"Q8_0":{"file":"Qwen3-8B-Q8_0.gguf","size":8709518112}},"license":"apache-2.0"}},
    "qwen2.5-7b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-7B-Instruct-GGUF","revision":"bb5d59e06d9551d752d08b292a50eb208b07ab1f","quants":{"Q2_K":{"file":"qwen2.5-7b-instruct-q2_k.gguf","size":3015940000},"Q3_K_M":{"file":"qwen2.5-7b-instruct-q3_k_m.gguf","size":3808391072},"FP16":{"file":"qwen2.5-7b-instruct-fp16-00001-of-00004.gguf","size":15237853536,"parts":[{"file":"qwen2.5-7b-instruct-fp16-00001-of-00004.gguf","size":3951521376},{"file":"qwen2.5-7b-instruct-fp16-00002-of-00004.gguf","size":3864909312},{"file":"qwen2.5-7b-instruct-fp16-00003-of-00004.gguf","size":3864894976},{"file":"qwen2.5-7b-instruct-fp16-00004-of-00004.gguf","size":3556527872}]},"Q4_0":{"file":"qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf","size":4431390848,"parts":[{"file":"qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf","size":3983228352},{"file":"qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf","size":448162496}]},"Q4_K_M":{"file":"qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf","size":4683073632,"parts":[{"file":"qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf","size":3993201344},{"file":"qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf","size":689872288}]},"Q5_0":{"file":"qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf","size":5315176576,"parts":[{"file":"qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf","size":4001112160},{"file":"qwen2.5-7b-instruct-q5_0-00002-of-00002.gguf","size":1314064416}]},"Q5_K_M":{"file":"qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf","size":5444831360,"parts":[{"file":"qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf","size":3989841792},{"file":"qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf","size":1454989568}]},"Q6_K":{"file":"qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf","size":6254198880,"parts":[{"file":"qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf","size":3950642464},{"file":"qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf","size":2303556416}]},"Q8_0":{"file":"qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf","size":8098525408,"parts":[{"file":"qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf","size":3980069248},{"file":"qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf","size":3942935680},{"file":"qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf","size":175520480}]}},"license":"apache-2.0"}},
    "qwen2.5-1.5b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-1.5B-Instruct-GGUF","revision":"91cad51170dc346986eccefdc2dd33a9da36ead9","quants":{"FP16":{"file":"qwen2.5-1.5b-instruct-fp16.gguf","size":3560416288},"Q2_K":{"file":"qwen2.5-1.5b-instruct-q2_k.gguf","size":752880160},"Q3_K_M":{"file":"qwen2.5-1.5b-instruct-q3_k_m.gguf","size":924455968},"Q4_0":{"file":"qwen2.5-1.5b-instruct-q4_0.gguf","size":1066227232},"Q4_K_M":{"file":"qwen2.5-1.5b-instruct-q4_k_m.gguf","size":1117320736},"Q5_0":{"file":"qwen2.5-1.5b-instruct-q5_0.gguf","size":1259173408},"Q5_K_M":{"file":"qwen2.5-1.5b-instruct-q5_k_m.gguf","size":1285494304},"Q6_K":{"file":"qwen2.5-1.5b-instruct-q6_k.gguf","size":1464178720},"Q8_0":{"file":"qwen2.5-1.5b-instruct-q8_0.gguf","size":1894532128}},"license":"apache-2.0"}},
    "qwen2.5-3b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-3B-Instruct-GGUF","revision":"7dabda4d13d513e3e842b20f0d435c732f172cbe","quants":{"Q2_K":{"file":"qwen2.5-3b-instruct-q2_k.gguf","size":1376856480},"Q3_K_M":{"file":"qwen2.5-3b-instruct-q3_k_m.gguf","size":1724178848},"Q4_0":{"file":"qwen2.5-3b-instruct-q4_0.gguf","size":1997879712},"Q4_K_M":{"file":"qwen2.5-3b-instruct-q4_k_m.gguf","size":2104932768},"Q5_0":{"file":"qwen2.5-3b-instruct-q5_0.gguf","size":2383591840},"Q5_K_M":{"file":"qwen2.5-3b-instruct-q5_k_m.gguf","size":2438740384},"Q6_K":{"file":"qwen2.5-3b-instruct-q6_k.gguf","size":2793410976},"Q8_0":{"file":"qwen2.5-3b-instruct-q8_0.gguf","size":3616088480},"FP16":{"file":"qwen2.5-3b-instruct-fp16-00001-of-00002.gguf","size":6800646784,"parts":[{"file":"qwen2.5-3b-instruct-fp16-00001-of-00002.gguf","size":3980517664},{"file":"qwen2.5-3b-instruct-fp16-00002-of-00002.gguf","size":2820129120}]}},"license":"qwen-research"}},
    "qwen2.5-coder-7b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-Coder-7B-Instruct-GGUF","revision":"13fb94bfda8c8cf22497dc57b78f391a9acb426a","quants":{"FP16":{"file":"qwen2.5-coder-7b-instruct-fp16.gguf","size":15237853184},"Q2_K":{"file":"qwen2.5-coder-7b-instruct-q2_k.gguf","size":3015940032},"Q3_K_M":{"file":"qwen2.5-coder-7b-instruct-q3_k_m.gguf","size":3808391104},"Q4_0":{"file":"qwen2.5-coder-7b-instruct-q4_0.gguf","size":4431390720},"Q4_K_M":{"file":"qwen2.5-coder-7b-instruct-q4_k_m.gguf","size":4683073536},"Q5_K_M":{"file":"qwen2.5-coder-7b-instruct-q5_k_m.gguf","size":5444831232},"Q6_K":{"file":"qwen2.5-coder-7b-instruct-q6_k.gguf","size":6254198784},"Q8_0":{"file":"qwen2.5-coder-7b-instruct-q8_0.gguf","size":8098525184},"Q5_0":{"file":"qwen2.5-coder-7b-instruct-q5_0-00001-of-00002.gguf","size":5315176576,"parts":[{"file":"qwen2.5-coder-7b-instruct-q5_0-00001-of-00002.gguf","size":4001112160},{"file":"qwen2.5-coder-7b-instruct-q5_0-00002-of-00002.gguf","size":1314064416}]}},"license":"apache-2.0"}},
    "qwen2.5-0.5b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-0.5B-Instruct-GGUF","revision":"9217f5db79a29953eb74d5343926648285ec7e67","quants":{"FP16":{"file":"qwen2.5-0.5b-instruct-fp16.gguf","size":1266425696},"Q2_K":{"file":"qwen2.5-0.5b-instruct-q2_k.gguf","size":415182688},"Q3_K_M":{"file":"qwen2.5-0.5b-instruct-q3_k_m.gguf","size":432041824},"Q4_0":{"file":"qwen2.5-0.5b-instruct-q4_0.gguf","size":428730208},"Q4_K_M":{"file":"qwen2.5-0.5b-instruct-q4_k_m.gguf","size":491400032},"Q5_0":{"file":"qwen2.5-0.5b-instruct-q5_0.gguf","size":490475360},"Q5_K_M":{"file":"qwen2.5-0.5b-instruct-q5_k_m.gguf","size":522186592},"Q6_K":{"file":"qwen2.5-0.5b-instruct-q6_k.gguf","size":650379104},"Q8_0":{"file":"qwen2.5-0.5b-instruct-q8_0.gguf","size":675710816}},"license":"apache-2.0"}},
    "llama-3.2-3b-instruct": {"category":"gen","gguf":{"repo":"unsloth/Llama-3.2-3B-Instruct-GGUF","revision":"e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31","quants":{"BF16":{"file":"Llama-3.2-3B-Instruct-BF16.gguf","size":6433687744},"F16":{"file":"Llama-3.2-3B-Instruct-F16.gguf","size":6433687616},"IQ4_NL":{"file":"Llama-3.2-3B-Instruct-IQ4_NL.gguf","size":1917190592},"IQ4_XS":{"file":"Llama-3.2-3B-Instruct-IQ4_XS.gguf","size":1829110208},"Q2_K":{"file":"Llama-3.2-3B-Instruct-Q2_K.gguf","size":1363935680},"Q2_K_L":{"file":"Llama-3.2-3B-Instruct-Q2_K_L.gguf","size":1363935680},"Q3_K_M":{"file":"Llama-3.2-3B-Instruct-Q3_K_M.gguf","size":1687159232},"Q3_K_S":{"file":"Llama-3.2-3B-Instruct-Q3_K_S.gguf","size":1542848960},"Q4_0":{"file":"Llama-3.2-3B-Instruct-Q4_0.gguf","size":1921909184},"Q4_1":{"file":"Llama-3.2-3B-Instruct-Q4_1.gguf","size":2093351360},"Q4_K_M":{"file":"Llama-3.2-3B-Instruct-Q4_K_M.gguf","size":2019377600},"Q4_K_S":{"file":"Llama-3.2-3B-Instruct-Q4_K_S.gguf","size":1928200640},"Q5_K_M":{"file":"Llama-3.2-3B-Instruct-Q5_K_M.gguf","size":2322153920},"Q5_K_S":{"file":"Llama-3.2-3B-Instruct-Q5_K_S.gguf","size":2269512128},"Q6_K":{"file":"Llama-3.2-3B-Instruct-Q6_K.gguf","size":2643853760},"Q8_0":{"file":"Llama-3.2-3B-Instruct-Q8_0.gguf","size":3421898816},"UD-IQ1_M":{"file":"Llama-3.2-3B-Instruct-UD-IQ1_M.gguf","size":960416192},"UD-IQ1_S":{"file":"Llama-3.2-3B-Instruct-UD-IQ1_S.gguf","size":912345536},"UD-IQ2_M":{"file":"Llama-3.2-3B-Instruct-UD-IQ2_M.gguf","size":1256262080},"UD-IQ2_XXS":{"file":"Llama-3.2-3B-Instruct-UD-IQ2_XXS.gguf","size":1046579648},"UD-IQ3_XXS":{"file":"Llama-3.2-3B-Instruct-UD-IQ3_XXS.gguf","size":1370147264},"UD-Q2_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q2_K_XL.gguf","size":1403380160},"UD-Q3_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q3_K_XL.gguf","size":1742430656},"UD-Q4_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf","size":2060886464},"UD-Q5_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q5_K_XL.gguf","size":2327781824},"UD-Q6_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q6_K_XL.gguf","size":2967833024},"UD-Q8_K_XL":{"file":"Llama-3.2-3B-Instruct-UD-Q8_K_XL.gguf","size":4204153280}},"license":"llama3.2"}},
    "phi-3.5-mini-instruct": {"category":"gen","gguf":{"repo":"bartowski/Phi-3.5-mini-instruct-GGUF","revision":"6d70da17e749a471ccb62ade694486011a75cda3","quants":{"IQ2_M":{"file":"Phi-3.5-mini-instruct-IQ2_M.gguf","size":1316395296},"IQ3_M":{"file":"Phi-3.5-mini-instruct-IQ3_M.gguf","size":1855600416},"IQ3_XS":{"file":"Phi-3.5-mini-instruct-IQ3_XS.gguf","size":1625175840},"IQ4_XS":{"file":"Phi-3.5-mini-instruct-IQ4_XS.gguf","size":2059853088},"Q2_K":{"file":"Phi-3.5-mini-instruct-Q2_K.gguf","size":1416204576},"Q2_K_L":{"file":"Phi-3.5-mini-instruct-Q2_K_L.gguf","size":1512396576},"Q3_K_L":{"file":"Phi-3.5-mini-instruct-Q3_K_L.gguf","size":2087597856},"Q3_K_M":{"file":"Phi-3.5-mini-instruct-Q3_K_M.gguf","size":1955477280},"Q3_K_S":{"file":"Phi-3.5-mini-instruct-Q3_K_S.gguf","size":1681798944},"Q3_K_XL":{"file":"Phi-3.5-mini-instruct-Q3_K_XL.gguf","size":2173785888},"Q4_0":{"file":"Phi-3.5-mini-instruct-Q4_0.gguf","size":2182468896},"Q4_0_4_4":{"file":"Phi-3.5-mini-instruct-Q4_0_4_4.gguf","size":2176177440},"Q4_0_4_8":{"file":"Phi-3.5-mini-instruct-Q4_0_4_8.gguf","size":2176177440},"Q4_0_8_8":{"file":"Phi-3.5-mini-instruct-Q4_0_8_8.gguf","size":2176177440},"Q4_K_L":{"file":"Phi-3.5-mini-instruct-Q4_K_L.gguf","size":2466338592},"Q4_K_M":{"file":"Phi-3.5-mini-instruct-Q4_K_M.gguf","size":2393232672},"Q4_K_S":{"file":"Phi-3.5-mini-instruct-Q4_K_S.gguf","size":2188760352},"Q5_K_L":{"file":"Phi-3.5-mini-instruct-Q5_K_L.gguf","size":2876069664},"Q5_K_M":{"file":"Phi-3.5-mini-instruct-Q5_K_M.gguf","size":2815276320},"Q5_K_S":{"file":"Phi-3.5-mini-instruct-Q5_K_S.gguf","size":2641474848},"Q6_K":{"file":"Phi-3.5-mini-instruct-Q6_K.gguf","size":3135853344},"Q6_K_L":{"file":"Phi-3.5-mini-instruct-Q6_K_L.gguf","size":3183564576},"Q8_0":{"file":"Phi-3.5-mini-instruct-Q8_0.gguf","size":4061222688},"F32":{"file":"Phi-3.5-mini-instruct-f32.gguf","size":15285057024}},"license":"mit"}},
    "llama-3.2-1b-instruct": {"category":"gen","gguf":{"repo":"bartowski/Llama-3.2-1B-Instruct-GGUF","revision":"067b946cf014b7c697f3654f621d577a3e3afd1c","quants":{"IQ3_M":{"file":"Llama-3.2-1B-Instruct-IQ3_M.gguf","size":657289344},"IQ4_XS":{"file":"Llama-3.2-1B-Instruct-IQ4_XS.gguf","size":743141504},"Q3_K_L":{"file":"Llama-3.2-1B-Instruct-Q3_K_L.gguf","size":732524672},"Q3_K_XL":{"file":"Llama-3.2-1B-Instruct-Q3_K_XL.gguf","size":796139648},"Q4_0":{"file":"Llama-3.2-1B-Instruct-Q4_0.gguf","size":773025920},"Q4_0_4_4":{"file":"Llama-3.2-1B-Instruct-Q4_0_4_4.gguf","size":770928768},"Q4_0_4_8":{"file":"Llama-3.2-1B-Instruct-Q4_0_4_8.gguf","size":770928768},"Q4_0_8_8":{"file":"Llama-3.2-1B-Instruct-Q4_0_8_8.gguf","size":770928768},"Q4_K_L":{"file":"Llama-3.2-1B-Instruct-Q4_K_L.gguf","size":871309440},"Q4_K_M":{"file":"Llama-3.2-1B-Instruct-Q4_K_M.gguf","size":807694464},"Q4_K_S":{"file":"Llama-3.2-1B-Instruct-Q4_K_S.gguf","size":775647360},"Q5_K_L":{"file":"Llama-3.2-1B-Instruct-Q5_K_L.gguf","size":975118464},"Q5_K_M":{"file":"Llama-3.2-1B-Instruct-Q5_K_M.gguf","size":911503488},"Q5_K_S":{"file":"Llama-3.2-1B-Instruct-Q5_K_S.gguf","size":892563584},"Q6_K":{"file":"Llama-3.2-1B-Instruct-Q6_K.gguf","size":1021800576},"Q6_K_L":{"file":"Llama-3.2-1B-Instruct-Q6_K_L.gguf","size":1085415552},"Q8_0":{"file":"Llama-3.2-1B-Instruct-Q8_0.gguf","size":1321083008},"F16":{"file":"Llama-3.2-1B-Instruct-f16.gguf","size":2479595360}},"license":"llama3.2"}},
    "gpt-oss-120b": {"category":"gen","gguf":{"repo":"ggml-org/gpt-oss-120b-GGUF","revision":"238abdd290bb874b90a5da1b4549881b7d05c091","quants":{"MXFP4":{"file":"gpt-oss-120b-MXFP4.gguf","size":63387346208}},"license":"apache-2.0"}},
    "qwen2.5-coder-32b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-Coder-32B-Instruct-GGUF","revision":"9d3053fce650fe1cdbdb75998c2a87add9d178ef","quants":{"Q2_K":{"file":"qwen2.5-coder-32b-instruct-q2_k.gguf","size":12313098432},"Q3_K_M":{"file":"qwen2.5-coder-32b-instruct-q3_k_m.gguf","size":15935047872},"Q4_0":{"file":"qwen2.5-coder-32b-instruct-q4_0.gguf","size":18640230592},"Q4_K_M":{"file":"qwen2.5-coder-32b-instruct-q4_k_m.gguf","size":19851335872},"Q5_0":{"file":"qwen2.5-coder-32b-instruct-q5_0.gguf","size":22638254272},"Q5_K_M":{"file":"qwen2.5-coder-32b-instruct-q5_k_m.gguf","size":23262156992},"Q6_K":{"file":"qwen2.5-coder-32b-instruct-q6_k.gguf","size":26886154432},"Q8_0":{"file":"qwen2.5-coder-32b-instruct-q8_0.gguf","size":34820884672},"FP16":{"file":"qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf","size":65535970464,"parts":[{"file":"qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf","size":7823519328},{"file":"qwen2.5-coder-32b-instruct-fp16-00002-of-00009.gguf","size":7801988608},{"file":"qwen2.5-coder-32b-instruct-fp16-00003-of-00009.gguf","size":7801968128},{"file":"qwen2.5-coder-32b-instruct-fp16-00004-of-00009.gguf","size":7801968128},{"file":"qwen2.5-coder-32b-instruct-fp16-00005-of-00009.gguf","size":7801968128},{"file":"qwen2.5-coder-32b-instruct-fp16-00006-of-00009.gguf","size":7801968128},{"file":"qwen2.5-coder-32b-instruct-fp16-00007-of-00009.gguf","size":7801968128},{"file":"qwen2.5-coder-32b-instruct-fp16-00008-of-00009.gguf","size":7801947584},{"file":"qwen2.5-coder-32b-instruct-fp16-00009-of-00009.gguf","size":3098674304}]}},"license":"apache-2.0"}},
    "deepseek-r1-0528-qwen3-8b": {"category":"gen","gguf":{"repo":"unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF","revision":"eb48357c179d34dbf515983f798dfb8752a0f261","quants":{"BF16":{"file":"DeepSeek-R1-0528-Qwen3-8B-BF16.gguf","size":16388045056},"IQ4_NL":{"file":"DeepSeek-R1-0528-Qwen3-8B-IQ4_NL.gguf","size":4793625088},"IQ4_XS":{"file":"DeepSeek-R1-0528-Qwen3-8B-IQ4_XS.gguf","size":4581288448},"Q2_K":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q2_K.gguf","size":3281734144},"Q2_K_L":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q2_K_L.gguf","size":3427592704},"Q3_K_M":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q3_K_M.gguf","size":4124162560},"Q3_K_S":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q3_K_S.gguf","size":3769612800},"Q4_0":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf","size":4787333632},"Q4_1":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q4_1.gguf","size":5247756800},"Q4_K_M":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf","size":5027785216},"Q4_K_S":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q4_K_S.gguf","size":4802013696},"Q5_K_M":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf","size":5851113984},"Q5_K_S":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q5_K_S.gguf","size":5720762880},"Q6_K":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf","size":6725900800},"Q8_0":{"file":"DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf","size":8709519872},"UD-IQ1_M":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-IQ1_M.gguf","size":2390985216},"UD-IQ1_S":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-IQ1_S.gguf","size":2268695040},"UD-IQ2_M":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-IQ2_M.gguf","size":3110504960},"UD-IQ2_XXS":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-IQ2_XXS.gguf","size":2601683456},"UD-IQ3_XXS":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-IQ3_XXS.gguf","size":3410266624},"UD-Q2_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q2_K_XL.gguf","size":3501976064},"UD-Q3_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q3_K_XL.gguf","size":4313737728},"UD-Q4_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q4_K_XL.gguf","size":5122746880},"UD-Q5_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q5_K_XL.gguf","size":5883587072},"UD-Q6_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q6_K_XL.gguf","size":7490550272},"UD-Q8_K_XL":{"file":"DeepSeek-R1-0528-Qwen3-8B-UD-Q8_K_XL.gguf","size":10824038912}},"license":"mit"}},
    "qwen3-14b": {"category":"gen","gguf":{"repo":"Qwen/Qwen3-14B-GGUF","revision":"530227a7d994db8eca5ab5ced2fb692b614357fd","quants":{"Q4_K_M":{"file":"Qwen3-14B-Q4_K_M.gguf","size":9001752960},"Q5_0":{"file":"Qwen3-14B-Q5_0.gguf","size":10263894400},"Q5_K_M":{"file":"Qwen3-14B-Q5_K_M.gguf","size":10514569568},"Q6_K":{"file":"Qwen3-14B-Q6_K.gguf","size":12121937248},"Q8_0":{"file":"Qwen3-14B-Q8_0.gguf","size":15698533728}},"license":"apache-2.0"}},
    "gemma-2-2b-it": {"category":"gen","gguf":{"repo":"bartowski/gemma-2-2b-it-GGUF","revision":"855f67caed130e1befc571b52bd181be2e858883","quants":{"IQ3_M":{"file":"gemma-2-2b-it-IQ3_M.gguf","size":1393561440},"IQ4_XS":{"file":"gemma-2-2b-it-IQ4_XS.gguf","size":1566250848},"Q3_K_L":{"file":"gemma-2-2b-it-Q3_K_L.gguf","size":1550436192},"Q4_K_M":{"file":"gemma-2-2b-it-Q4_K_M.gguf","size":1708582752},"Q4_K_S":{"file":"gemma-2-2b-it-Q4_K_S.gguf","size":1638651744},"Q5_K_M":{"file":"gemma-2-2b-it-Q5_K_M.gguf","size":1923278688},"Q5_K_S":{"file":"gemma-2-2b-it-Q5_K_S.gguf","size":1882543968},"Q6_K":{"file":"gemma-2-2b-it-Q6_K.gguf","size":2151393120},"Q6_K_L":{"file":"gemma-2-2b-it-Q6_K_L.gguf","size":2294241120},"Q8_0":{"file":"gemma-2-2b-it-Q8_0.gguf","size":2784495456},"F32":{"file":"gemma-2-2b-it-f32.gguf","size":10463413856}},"license":"gemma"}},
    "qwen3-1.7b": {"category":"gen","gguf":{"repo":"unsloth/Qwen3-1.7B-GGUF","revision":"d7f544eead698dbd1f15126ef60b45a1e1933222","quants":{"BF16":{"file":"Qwen3-1.7B-BF16.gguf","size":3447349568},"IQ4_NL":{"file":"Qwen3-1.7B-IQ4_NL.gguf","size":1054423616},"IQ4_XS":{"file":"Qwen3-1.7B-IQ4_XS.gguf","size":1010383424},"Q2_K":{"file":"Qwen3-1.7B-Q2_K.gguf","size":777796160},"Q2_K_L":{"file":"Qwen3-1.7B-Q2_K_L.gguf","size":777796160},"Q3_K_M":{"file":"Qwen3-1.7B-Q3_K_M.gguf","size":939539008},"Q3_K_S":{"file":"Qwen3-1.7B-Q3_K_S.gguf","size":867252800},"Q4_0":{"file":"Qwen3-1.7B-Q4_0.gguf","size":1056782912},"Q4_1":{"file":"Qwen3-1.7B-Q4_1.gguf","size":1142504000},"Q4_K_M":{"file":"Qwen3-1.7B-Q4_K_M.gguf","size":1107409472},"Q4_K_S":{"file":"Qwen3-1.7B-Q4_K_S.gguf","size":1060190784},"Q5_K_M":{"file":"Qwen3-1.7B-Q5_K_M.gguf","size":1257880128},"Q5_K_S":{"file":"Qwen3-1.7B-Q5_K_S.gguf","size":1230584384},"Q6_K":{"file":"Qwen3-1.7B-Q6_K.gguf","size":1417755200},"Q8_0":{"file":"Qwen3-1.7B-Q8_0.gguf","size":1834426944},"UD-IQ1_M":{"file":"Qwen3-1.7B-UD-IQ1_M.gguf","size":561947200},"UD-IQ1_S":{"file":"Qwen3-1.7B-UD-IQ1_S.gguf","size":537829952},"UD-IQ2_M":{"file":"Qwen3-1.7B-UD-IQ2_M.gguf","size":708715072},"UD-IQ2_XXS":{"file":"Qwen3-1.7B-UD-IQ2_XXS.gguf","size":605790784},"UD-IQ3_XXS":{"file":"Qwen3-1.7B-UD-IQ3_XXS.gguf","size":765174336},"UD-Q2_K_XL":{"file":"Qwen3-1.7B-UD-Q2_K_XL.gguf","size":797866560},"UD-Q3_K_XL":{"file":"Qwen3-1.7B-UD-Q3_K_XL.gguf","size":968899136},"UD-Q4_K_XL":{"file":"Qwen3-1.7B-UD-Q4_K_XL.gguf","size":1132952128},"UD-Q5_K_XL":{"file":"Qwen3-1.7B-UD-Q5_K_XL.gguf","size":1262991936},"UD-Q6_K_XL":{"file":"Qwen3-1.7B-UD-Q6_K_XL.gguf","size":1610949184},"UD-Q8_K_XL":{"file":"Qwen3-1.7B-UD-Q8_K_XL.gguf","size":2332582464}},"license":"apache-2.0"}},
    "qwen3-0.6b": {"category":"gen","gguf":{"repo":"unsloth/Qwen3-0.6B-GGUF","revision":"50968a4468ef4233ed78cd7c3de230dd1d61a56b","quants":{"BF16":{"file":"Qwen3-0.6B-BF16.gguf","size":1198182848},"IQ4_NL":{"file":"Qwen3-0.6B-IQ4_NL.gguf","size":381566656},"IQ4_XS":{"file":"Qwen3-0.6B-IQ4_XS.gguf","size":367804096},"Q2_K":{"file":"Qwen3-0.6B-Q2_K.gguf","size":296238784},"Q2_K_L":{"file":"Qwen3-0.6B-Q2_K_L.gguf","size":296238784},"Q3_K_M":{"file":"Qwen3-0.6B-Q3_K_M.gguf","size":347127488},"Q3_K_S":{"file":"Qwen3-0.6B-Q3_K_S.gguf","size":323075776},"Q4_0":{"file":"Qwen3-0.6B-Q4_0.gguf","size":382156480},"Q4_1":{"file":"Qwen3-0.6B-Q4_1.gguf","size":409091776},"Q4_K_M":{"file":"Qwen3-0.6B-Q4_K_M.gguf","size":396705472},"Q4_K_S":{"file":"Qwen3-0.6B-Q4_K_S.gguf","size":383270592},"Q5_K_M":{"file":"Qwen3-0.6B-Q5_K_M.gguf","size":444415680},"Q5_K_S":{"file":"Qwen3-0.6B-Q5_K_S.gguf","size":436616896},"Q6_K":{"file":"Qwen3-0.6B-Q6_K.gguf","size":495107776},"Q8_0":{"file":"Qwen3-0.6B-Q8_0.gguf","size":639447744},"UD-IQ1_M":{"file":"Qwen3-0.6B-UD-IQ1_M.gguf","size":220754624},"UD-IQ1_S":{"file":"Qwen3-0.6B-UD-IQ1_S.gguf","size":214643392},"UD-IQ2_M":{"file":"Qwen3-0.6B-UD-IQ2_M.gguf","size":268702400},"UD-IQ2_XXS":{"file":"Qwen3-0.6B-UD-IQ2_XXS.gguf","size":234074816},"UD-IQ3_XXS":{"file":"Qwen3-0.6B-UD-IQ3_XXS.gguf","size":282088128},"UD-Q2_K_XL":{"file":"Qwen3-0.6B-UD-Q2_K_XL.gguf","size":301727424},"UD-Q3_K_XL":{"file":"Qwen3-0.6B-UD-Q3_K_XL.gguf","size":356622016},"UD-Q4_K_XL":{"file":"Qwen3-0.6B-UD-Q4_K_XL.gguf","size":405372608},"UD-Q5_K_XL":{"file":"Qwen3-0.6B-UD-Q5_K_XL.gguf","size":446381760},"UD-Q6_K_XL":{"file":"Qwen3-0.6B-UD-Q6_K_XL.gguf","size":576467648},"UD-Q8_K_XL":{"file":"Qwen3-0.6B-UD-Q8_K_XL.gguf","size":844288704}},"license":"apache-2.0"}},
    "gemma-3-270m-it": {"category":"gen","gguf":{"repo":"unsloth/gemma-3-270m-it-GGUF","revision":"c90975dbd40c0c7b275fefaae758c3415c906238","quants":{"F16":{"file":"gemma-3-270m-it-F16.gguf","size":542835488},"IQ4_NL":{"file":"gemma-3-270m-it-IQ4_NL.gguf","size":241964064},"IQ4_XS":{"file":"gemma-3-270m-it-IQ4_XS.gguf","size":240858144},"Q2_K":{"file":"gemma-3-270m-it-Q2_K.gguf","size":237079584},"Q2_K_L":{"file":"gemma-3-270m-it-Q2_K_L.gguf","size":237079584},"Q3_K_M":{"file":"gemma-3-270m-it-Q3_K_M.gguf","size":241964064},"Q3_K_S":{"file":"gemma-3-270m-it-Q3_K_S.gguf","size":236710944},"Q4_0":{"file":"gemma-3-270m-it-Q4_0.gguf","size":241574944},"Q4_1":{"file":"gemma-3-270m-it-Q4_1.gguf","size":247677984},"Q4_K_M":{"file":"gemma-3-270m-it-Q4_K_M.gguf","size":253115424},"Q4_K_S":{"file":"gemma-3-270m-it-Q4_K_S.gguf","size":249889824},"Q5_K_M":{"file":"gemma-3-270m-it-Q5_K_M.gguf","size":260027424},"Q5_K_S":{"file":"gemma-3-270m-it-Q5_K_S.gguf","size":257999904},"Q6_K":{"file":"gemma-3-270m-it-Q6_K.gguf","size":282975264},"Q8_0":{"file":"gemma-3-270m-it-Q8_0.gguf","size":291546144},"UD-IQ2_M":{"file":"gemma-3-270m-it-UD-IQ2_M.gguf","size":182787104},"UD-IQ2_XXS":{"file":"gemma-3-270m-it-UD-IQ2_XXS.gguf","size":180104224},"UD-IQ3_XXS":{"file":"gemma-3-270m-it-UD-IQ3_XXS.gguf","size":184517664},"UD-Q2_K_XL":{"file":"gemma-3-270m-it-UD-Q2_K_XL.gguf","size":237745184},"UD-Q3_K_XL":{"file":"gemma-3-270m-it-UD-Q3_K_XL.gguf","size":242619424},"UD-Q4_K_XL":{"file":"gemma-3-270m-it-UD-Q4_K_XL.gguf","size":253934624},"UD-Q5_K_XL":{"file":"gemma-3-270m-it-UD-Q5_K_XL.gguf","size":260027424},"UD-Q6_K_XL":{"file":"gemma-3-270m-it-UD-Q6_K_XL.gguf","size":286149664},"UD-Q8_K_XL":{"file":"gemma-3-270m-it-UD-Q8_K_XL.gguf","size":471104544}},"license":"gemma"}},
    "gte-multilingual-base": {"category":"embed","onnx":{"repo":"onnx-community/gte-multilingual-base","revision":"2edbf5e672aab465f9ed4c154a8b61791c082c69","model":"onnx/model.onnx","license":null,"dim":768},"license":"apache-2.0"},
    "snowflake-arctic-embed-m-v1.5": {"category":"embed","onnx":{"repo":"Snowflake/snowflake-arctic-embed-m-v1.5","revision":"e58a8f756156a1293d763f17e3aae643474e9b8a","model":"onnx/model.onnx","license":"apache-2.0","dim":768},"gguf":{"repo":"Snowflake/snowflake-arctic-embed-m-v1.5","revision":"e58a8f756156a1293d763f17e3aae643474e9b8a","quants":{"BF16":{"file":"gguf/snowflake-arctic-embed-m-v1.5-bf16.gguf","size":219454752},"F16":{"file":"gguf/snowflake-arctic-embed-m-v1.5-f16.gguf","size":219454752},"F32":{"file":"gguf/snowflake-arctic-embed-m-v1.5-f32.gguf","size":436205856},"Q8_0":{"file":"gguf/snowflake-arctic-embed-m-v1.5-q8_0.gguf","size":117852672},"DEFAULT":{"file":"gguf/snowflake-arctic-embed-m-v1.5-tq1_0.gguf","size":67501344}},"license":"apache-2.0"},"prompts":{"query":"Represent this sentence for searching relevant passages: "}},
    "bge-reranker-base": {"category":"rerank","onnx":{"repo":"BAAI/bge-reranker-base","revision":"2cfc18c9415c912f9d8155881c133215df768a70","model":"onnx/model.onnx","license":"mit","dim":768},"gguf":{"repo":"cstr/bge-reranker-base-GGUF","revision":"357f3d22b5edf9dfbe11e2f5fbf33e27affdaad7","quants":{"DEFAULT":{"file":"bge-reranker-base-iq4_xs-f7.gguf","size":257989344},"IQ4_XS":{"file":"bge-reranker-base-iq4_xs.gguf","size":257989344},"Q4_K":{"file":"bge-reranker-base-q4_k.gguf","size":260643552},"Q8_0":{"file":"bge-reranker-base-q8_0.gguf","size":303110880}},"license":"mit"}},
    "bge-reranker-v2-m3": {"category":"rerank","onnx":{"repo":"onnx-community/bge-reranker-v2-m3-ONNX","revision":"6f5ff65298512715a1e669753bc754d2bc8f367b","model":"onnx/model.onnx","license":null,"dim":1024},"gguf":{"repo":"gpustack/bge-reranker-v2-m3-GGUF","revision":"3093af03b1a635e67b084b1d8c03c5f5e020fd05","quants":{"FP16":{"file":"bge-reranker-v2-m3-FP16.gguf","size":1159776896},"Q2_K":{"file":"bge-reranker-v2-m3-Q2_K.gguf","size":366467488},"Q3_K_M":{"file":"bge-reranker-v2-m3-Q3_K_M.gguf","size":402749856},"Q4_0":{"file":"bge-reranker-v2-m3-Q4_0.gguf","size":422156704},"Q4_K_M":{"file":"bge-reranker-v2-m3-Q4_K_M.gguf","size":438376864},"Q5_0":{"file":"bge-reranker-v2-m3-Q5_0.gguf","size":460036512},"Q5_K_M":{"file":"bge-reranker-v2-m3-Q5_K_M.gguf","size":468392352},"Q6_K":{"file":"bge-reranker-v2-m3-Q6_K.gguf","size":500283808},"Q8_0":{"file":"bge-reranker-v2-m3-Q8_0.gguf","size":635676416}},"license":"apache-2.0"}},
    "mxbai-rerank-base-v1": {"category":"rerank","onnx":{"repo":"mixedbread-ai/mxbai-rerank-base-v1","revision":"800f24c113213a187e65bde9db00c15a2bb12738","model":"onnx/model.onnx","license":"apache-2.0","dim":768},"gguf":{"repo":"cstr/mxbai-rerank-base-v1-GGUF","revision":"5b165473678425c25a5790218905e1bd7bf07712","quants":{"DEFAULT":{"file":"mxbai-rerank-base-v1-iq4_xs-f7.gguf","size":153700896},"IQ4_XS":{"file":"mxbai-rerank-base-v1-iq4_xs.gguf","size":153700896},"Q4_K":{"file":"mxbai-rerank-base-v1-q4_k.gguf","size":156355104},"Q8_0":{"file":"mxbai-rerank-base-v1-q8_0.gguf","size":198822432}},"license":"apache-2.0"}},
    "qwen3-32b": {"category":"gen","gguf":{"repo":"unsloth/Qwen3-32B-GGUF","revision":"931c84066f88693a02ab8de820cfcd066d913241","quants":{"IQ4_NL":{"file":"Qwen3-32B-IQ4_NL.gguf","size":18679495328},"IQ4_XS":{"file":"Qwen3-32B-IQ4_XS.gguf","size":17714805408},"Q2_K":{"file":"Qwen3-32B-Q2_K.gguf","size":12344652448},"Q2_K_L":{"file":"Qwen3-32B-Q2_K_L.gguf","size":12526975648},"Q3_K_M":{"file":"Qwen3-32B-Q3_K_M.gguf","size":15971778208},"Q3_K_S":{"file":"Qwen3-32B-Q3_K_S.gguf","size":14389739168},"Q4_0":{"file":"Qwen3-32B-Q4_0.gguf","size":18703088288},"Q4_1":{"file":"Qwen3-32B-Q4_1.gguf","size":20636523168},"Q4_K_M":{"file":"Qwen3-32B-Q4_K_M.gguf","size":19762150048},"Q4_K_S":{"file":"Qwen3-32B-Q4_K_S.gguf","size":18771245728},"Q5_K_M":{"file":"Qwen3-32B-Q5_K_M.gguf","size":23214832288},"Q5_K_S":{"file":"Qwen3-32B-Q5_K_S.gguf","size":22635494048},"Q6_K":{"file":"Qwen3-32B-Q6_K.gguf","size":26883307168},"Q8_0":{"file":"Qwen3-32B-Q8_0.gguf","size":34817719968},"UD-IQ1_M":{"file":"Qwen3-32B-UD-IQ1_M.gguf","size":8302084768},"UD-IQ1_S":{"file":"Qwen3-32B-UD-IQ1_S.gguf","size":7738884768},"UD-IQ2_M":{"file":"Qwen3-32B-UD-IQ2_M.gguf","size":11553366688},"UD-IQ2_XXS":{"file":"Qwen3-32B-UD-IQ2_XXS.gguf","size":9277096608},"UD-IQ3_XXS":{"file":"Qwen3-32B-UD-IQ3_XXS.gguf","size":12967551648},"UD-Q2_K_XL":{"file":"Qwen3-32B-UD-Q2_K_XL.gguf","size":12797352608},"UD-Q3_K_XL":{"file":"Qwen3-32B-UD-Q3_K_XL.gguf","size":16403102368},"UD-Q4_K_XL":{"file":"Qwen3-32B-UD-Q4_K_XL.gguf","size":20021713568},"UD-Q5_K_XL":{"file":"Qwen3-32B-UD-Q5_K_XL.gguf","size":23233551008},"UD-Q6_K_XL":{"file":"Qwen3-32B-UD-Q6_K_XL.gguf","size":28961586848},"UD-Q8_K_XL":{"file":"Qwen3-32B-UD-Q8_K_XL.gguf","size":39481015968},"BF16":{"file":"BF16/Qwen3-32B-BF16-00001-of-00002.gguf","size":65531575936,"parts":[{"file":"BF16/Qwen3-32B-BF16-00001-of-00002.gguf","size":49871764512},{"file":"BF16/Qwen3-32B-BF16-00002-of-00002.gguf","size":15659811424}]}},"license":"apache-2.0"}},
    "qwen3-30b-a3b": {"category":"gen","gguf":{"repo":"unsloth/Qwen3-30B-A3B-GGUF","revision":"d5b1d57bd0b504ac62ae6c725904e96ef228dc74","quants":{"IQ4_NL":{"file":"Qwen3-30B-A3B-IQ4_NL.gguf","size":17310782016},"IQ4_XS":{"file":"Qwen3-30B-A3B-IQ4_XS.gguf","size":16378073664},"Q2_K":{"file":"Qwen3-30B-A3B-Q2_K.gguf","size":11258610240},"Q2_K_L":{"file":"Qwen3-30B-A3B-Q2_K_L.gguf","size":11331539520},"Q3_K_M":{"file":"Qwen3-30B-A3B-Q3_K_M.gguf","size":14711847488},"Q3_K_S":{"file":"Qwen3-30B-A3B-Q3_K_S.gguf","size":13292468800},"Q4_0":{"file":"Qwen3-30B-A3B-Q4_0.gguf","size":17379988032},"Q4_1":{"file":"Qwen3-30B-A3B-Q4_1.gguf","size":19192500800},"Q4_K_M":{"file":"Qwen3-30B-A3B-Q4_K_M.gguf","size":18556686912},"Q4_K_S":{"file":"Qwen3-30B-A3B-Q4_K_S.gguf","size":17456009792},"Q5_K_M":{"file":"Qwen3-30B-A3B-Q5_K_M.gguf","size":21725581888},"Q5_K_S":{"file":"Qwen3-30B-A3B-Q5_K_S.gguf","size":21080511040},"Q6_K":{"file":"Qwen3-30B-A3B-Q6_K.gguf","size":25092532800},"Q8_0":{"file":"Qwen3-30B-A3B-Q8_0.gguf","size":32483932736},"UD-IQ1_M":{"file":"Qwen3-30B-A3B-UD-IQ1_M.gguf","size":9666859584},"UD-IQ1_S":{"file":"Qwen3-30B-A3B-UD-IQ1_S.gguf","size":9043300928},"UD-IQ2_M":{"file":"Qwen3-30B-A3B-UD-IQ2_M.gguf","size":10865578560},"UD-IQ2_XXS":{"file":"Qwen3-30B-A3B-UD-IQ2_XXS.gguf","size":10362262080},"UD-IQ3_XXS":{"file":"Qwen3-30B-A3B-UD-IQ3_XXS.gguf","size":12888085056},"UD-Q2_K_XL":{"file":"Qwen3-30B-A3B-UD-Q2_K_XL.gguf","size":11814277696},"UD-Q3_K_XL":{"file":"Qwen3-30B-A3B-UD-Q3_K_XL.gguf","size":13833048640},"UD-Q4_K_XL":{"file":"Qwen3-30B-A3B-UD-Q4_K_XL.gguf","size":17715663424},"UD-Q5_K_XL":{"file":"Qwen3-30B-A3B-UD-Q5_K_XL.gguf","size":21740302912},"UD-Q6_K_XL":{"file":"Qwen3-30B-A3B-UD-Q6_K_XL.gguf","size":26340325952},"UD-Q8_K_XL":{"file":"Qwen3-30B-A3B-UD-Q8_K_XL.gguf","size":35989944896},"BF16":{"file":"BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf","size":61095803424,"parts":[{"file":"BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf","size":49693950144},{"file":"BF16/Qwen3-30B-A3B-BF16-00002-of-00002.gguf","size":11401853280}]}},"license":"apache-2.0"}},
    "qwen2.5-14b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-14B-Instruct-GGUF","revision":"b466e1f8c07172155743e8e1307507d8a4f91fbd","quants":{"FP16":{"file":"qwen2.5-14b-instruct-fp16-00001-of-00008.gguf","size":29547716864,"parts":[{"file":"qwen2.5-14b-instruct-fp16-00001-of-00008.gguf","size":3891239200},{"file":"qwen2.5-14b-instruct-fp16-00002-of-00008.gguf","size":3995566912},{"file":"qwen2.5-14b-instruct-fp16-00003-of-00008.gguf","size":3995566976},{"file":"qwen2.5-14b-instruct-fp16-00004-of-00008.gguf","size":3995608064},{"file":"qwen2.5-14b-instruct-fp16-00005-of-00008.gguf","size":3979867360},{"file":"qwen2.5-14b-instruct-fp16-00006-of-00008.gguf","size":3995566976},{"file":"qwen2.5-14b-instruct-fp16-00007-of-00008.gguf","size":3995546432},{"file":"qwen2.5-14b-instruct-fp16-00008-of-00008.gguf","size":1698754944}]},"Q2_K":{"file":"qwen2.5-14b-instruct-q2_k-00001-of-00002.gguf","size":5770497600,"parts":[{"file":"qwen2.5-14b-instruct-q2_k-00001-of-00002.gguf","size":4004118304},{"file":"qwen2.5-14b-instruct-q2_k-00002-of-00002.gguf","size":1766379296}]},"Q3_K_M":{"file":"qwen2.5-14b-instruct-q3_k_m-00001-of-00002.gguf","size":7339204160,"parts":[{"file":"qwen2.5-14b-instruct-q3_k_m-00001-of-00002.gguf","size":4000429472},{"file":"qwen2.5-14b-instruct-q3_k_m-00002-of-00002.gguf","size":3338774688}]},"Q4_0":{"file":"qwen2.5-14b-instruct-q4_0-00001-of-00003.gguf","size":8517725856,"parts":[{"file":"qwen2.5-14b-instruct-q4_0-00001-of-00003.gguf","size":4003609056},{"file":"qwen2.5-14b-instruct-q4_0-00002-of-00003.gguf","size":3835573056},{"file":"qwen2.5-14b-instruct-q4_0-00003-of-00003.gguf","size":678543744}]},"Q4_K_M":{"file":"qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf","size":8988110496,"parts":[{"file":"qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf","size":3991999872},{"file":"qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf","size":3989373504},{"file":"qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf","size":1006737120}]},"Q5_0":{"file":"qwen2.5-14b-instruct-q5_0-00001-of-00003.gguf","size":10266554048,"parts":[{"file":"qwen2.5-14b-instruct-q5_0-00001-of-00003.gguf","size":3997359360},{"file":"qwen2.5-14b-instruct-q5_0-00002-of-00003.gguf","size":3975436576},{"file":"qwen2.5-14b-instruct-q5_0-00003-of-00003.gguf","size":2293758112}]},"Q5_K_M":{"file":"qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf","size":10508873376,"parts":[{"file":"qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf","size":4005690208},{"file":"qwen2.5-14b-instruct-q5_k_m-00002-of-00003.gguf","size":3997407296},{"file":"qwen2.5-14b-instruct-q5_k_m-00003-of-00003.gguf","size":2505775872}]},"Q6_K":{"file":"qwen2.5-14b-instruct-q6_k-00001-of-00004.gguf","size":12124684064,"parts":[{"file":"qwen2.5-14b-instruct-q6_k-00001-of-00004.gguf","size":3985204096},{"file":"qwen2.5-14b-instruct-q6_k-00002-of-00004.gguf","size":3945074624},{"file":"qwen2.5-14b-instruct-q6_k-00003-of-00004.gguf","size":3497613920},{"file":"qwen2.5-14b-instruct-q6_k-00004-of-00004.gguf","size":696791424}]},"Q8_0":{"file":"qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf","size":15701597984,"parts":[{"file":"qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf","size":3975711104},{"file":"qwen2.5-14b-instruct-q8_0-00002-of-00004.gguf","size":3953267776},{"file":"qwen2.5-14b-instruct-q8_0-00003-of-00004.gguf","size":3986695424},{"file":"qwen2.5-14b-instruct-q8_0-00004-of-00004.gguf","size":3785923680}]}},"license":"apache-2.0"}},
    "qwen2.5-32b-instruct": {"category":"gen","gguf":{"repo":"Qwen/Qwen2.5-32B-Instruct-GGUF","revision":"a15e3cc10f8bbb2c0af6f8f1f34a32e3b060c09d","quants":{"FP16":{"file":"qwen2.5-32b-instruct-fp16-00001-of-00017.gguf","size":65535971488,"parts":[{"file":"qwen2.5-32b-instruct-fp16-00001-of-00017.gguf","size":3922555904},{"file":"qwen2.5-32b-instruct-fp16-00002-of-00017.gguf","size":3900963552},{"file":"qwen2.5-32b-instruct-fp16-00003-of-00017.gguf","size":3901004608},{"file":"qwen2.5-32b-instruct-fp16-00004-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00005-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00006-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00007-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00008-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00009-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00010-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00011-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00012-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00013-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00014-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00015-of-00017.gguf","size":3900984128},{"file":"qwen2.5-32b-instruct-fp16-00016-of-00017.gguf","size":3900963584},{"file":"qwen2.5-32b-instruct-fp16-00017-of-00017.gguf","size":3098674304}]},"Q2_K":{"file":"qwen2.5-32b-instruct-q2_k-00001-of-00004.gguf","size":12313098784,"parts":[{"file":"qwen2.5-32b-instruct-q2_k-00001-of-00004.gguf","size":4003345536},{"file":"qwen2.5-32b-instruct-q2_k-00002-of-00004.gguf","size":3987016768},{"file":"qwen2.5-32b-instruct-q2_k-00003-of-00004.gguf","size":3398403296},{"file":"qwen2.5-32b-instruct-q2_k-00004-of-00004.gguf","size":924333184}]},"Q3_K_M":{"file":"qwen2.5-32b-instruct-q3_k_m-00001-of-00005.gguf","size":15935048320,"parts":[{"file":"qwen2.5-32b-instruct-q3_k_m-00001-of-00005.gguf","size":3980600608},{"file":"qwen2.5-32b-instruct-q3_k_m-00002-of-00005.gguf","size":3953507872},{"file":"qwen2.5-32b-instruct-q3_k_m-00003-of-00005.gguf","size":3953507872},{"file":"qwen2.5-32b-instruct-q3_k_m-00004-of-00005.gguf","size":3955324544},{"file":"qwen2.5-32b-instruct-q3_k_m-00005-of-00005.gguf","size":92107424}]},"Q4_0":{"file":"qwen2.5-32b-instruct-q4_0-00001-of-00005.gguf","size":18640231072,"parts":[{"file":"qwen2.5-32b-instruct-q4_0-00001-of-00005.gguf","size":3992579744},{"file":"qwen2.5-32b-instruct-q4_0-00002-of-00005.gguf","size":3938084672},{"file":"qwen2.5-32b-instruct-q4_0-00003-of-00005.gguf","size":3999991456},{"file":"qwen2.5-32b-instruct-q4_0-00004-of-00005.gguf","size":3955824704},{"file":"qwen2.5-32b-instruct-q4_0-00005-of-00005.gguf","size":2753750496}]},"Q4_K_M":{"file":"qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf","size":19851336384,"parts":[{"file":"qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf","size":3961498272},{"file":"qwen2.5-32b-instruct-q4_k_m-00002-of-00005.gguf","size":3948996064},{"file":"qwen2.5-32b-instruct-q4_k_m-00003-of-00005.gguf","size":3993478688},{"file":"qwen2.5-32b-instruct-q4_k_m-00004-of-00005.gguf","size":3950347744},{"file":"qwen2.5-32b-instruct-q4_k_m-00005-of-00005.gguf","size":3997015616}]},"Q5_0":{"file":"qwen2.5-32b-instruct-q5_0-00001-of-00006.gguf","size":22638254880,"parts":[{"file":"qwen2.5-32b-instruct-q5_0-00001-of-00006.gguf","size":3994998752},{"file":"qwen2.5-32b-instruct-q5_0-00002-of-00006.gguf","size":3922513920},{"file":"qwen2.5-32b-instruct-q5_0-00003-of-00006.gguf","size":3983769504},{"file":"qwen2.5-32b-instruct-q5_0-00004-of-00006.gguf","size":3922493408},{"file":"qwen2.5-32b-instruct-q5_0-00005-of-00006.gguf","size":3926122624},{"file":"qwen2.5-32b-instruct-q5_0-00006-of-00006.gguf","size":2888356672}]},"Q5_K_M":{"file":"qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf","size":23262157568,"parts":[{"file":"qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf","size":3952703168},{"file":"qwen2.5-32b-instruct-q5_k_m-00002-of-00006.gguf","size":3960827616},{"file":"qwen2.5-32b-instruct-q5_k_m-00003-of-00006.gguf","size":3982458784},{"file":"qwen2.5-32b-instruct-q5_k_m-00004-of-00006.gguf","size":3982438240},{"file":"qwen2.5-32b-instruct-q5_k_m-00005-of-00006.gguf","size":3906789440},{"file":"qwen2.5-32b-instruct-q5_k_m-00006-of-00006.gguf","size":3476940320}]},"Q6_K":{"file":"qwen2.5-32b-instruct-q6_k-00001-of-00007.gguf","size":26886155104,"parts":[{"file":"qwen2.5-32b-instruct-q6_k-00001-of-00007.gguf","size":3986912000},{"file":"qwen2.5-32b-instruct-q6_k-00002-of-00007.gguf","size":3978922688},{"file":"qwen2.5-32b-instruct-q6_k-00003-of-00007.gguf","size":3996162848},{"file":"qwen2.5-32b-instruct-q6_k-00004-of-00007.gguf","size":3884305120},{"file":"qwen2.5-32b-instruct-q6_k-00005-of-00007.gguf","size":3996146464},{"file":"qwen2.5-32b-instruct-q6_k-00006-of-00007.gguf","size":3978939104},{"file":"qwen2.5-32b-instruct-q6_k-00007-of-00007.gguf","size":3064766880}]},"Q8_0":{"file":"qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf","size":34820885632,"parts":[{"file":"qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf","size":3941956768},{"file":"qwen2.5-32b-instruct-q8_0-00002-of-00009.gguf","size":3927757760},{"file":"qwen2.5-32b-instruct-q8_0-00003-of-00009.gguf","size":3994654144},{"file":"qwen2.5-32b-instruct-q8_0-00004-of-00009.gguf","size":3994654144},{"file":"qwen2.5-32b-instruct-q8_0-00005-of-00009.gguf","size":3927737280},{"file":"qwen2.5-32b-instruct-q8_0-00006-of-00009.gguf","size":3994654144},{"file":"qwen2.5-32b-instruct-q8_0-00007-of-00009.gguf","size":3994654144},{"file":"qwen2.5-32b-instruct-q8_0-00008-of-00009.gguf","size":3927737280},{"file":"qwen2.5-32b-instruct-q8_0-00009-of-00009.gguf","size":3117079968}]}},"license":"apache-2.0"}},
    "gemma-3-1b-it": {"category":"gen","gguf":{"repo":"ggml-org/gemma-3-1b-it-GGUF","revision":"f9c28bcd85737ffc5aef028638d3341d49869c27","quants":{"Q4_K_M":{"file":"gemma-3-1b-it-Q4_K_M.gguf","size":806058240},"Q8_0":{"file":"gemma-3-1b-it-Q8_0.gguf","size":1069306368},"F16":{"file":"gemma-3-1b-it-f16.gguf","size":2006573568}},"license":"gemma"}},
    "gemma-3-4b-it": {"category":"gen","gguf":{"repo":"ggml-org/gemma-3-4b-it-GGUF","revision":"d0976223747697cb51e056d85c532013931fe52e","quants":{"Q4_K_M":{"file":"gemma-3-4b-it-Q4_K_M.gguf","size":2489757856},"Q8_0":{"file":"gemma-3-4b-it-Q8_0.gguf","size":4130226336},"F16":{"file":"gemma-3-4b-it-f16.gguf","size":7767474336}},"license":"gemma"}},
    "phi-4": {"category":"gen","gguf":{"repo":"microsoft/phi-4-gguf","revision":"6edc2ef6664b739a8e11e62f2672ff6afe0c15ac","quants":{"IQ3_M":{"file":"phi-4-IQ3_M.gguf","size":6913835200},"IQ3_S":{"file":"phi-4-IQ3_S.gguf","size":6504747200},"IQ3_XS":{"file":"phi-4-IQ3_XS.gguf","size":6246699200},"IQ3_XXS":{"file":"phi-4-IQ3_XXS.gguf","size":6079643840},"IQ4_NL":{"file":"phi-4-IQ4_NL.gguf","size":8440762560},"IQ4_XS":{"file":"phi-4-IQ4_XS.gguf","size":8013058240},"Q2_K":{"file":"phi-4-Q2_K.gguf","size":5547348160},"Q3_K":{"file":"phi-4-Q3_K.gguf","size":7363268800},"Q3_K_L":{"file":"phi-4-Q3_K_L.gguf","size":7930155200},"Q3_K_S":{"file":"phi-4-Q3_K_S.gguf","size":6504747200},"Q4_0":{"file":"phi-4-Q4_0.gguf","size":8383418560},"Q4_1":{"file":"phi-4-Q4_1.gguf","size":9267499200},"Q4_K":{"file":"phi-4-Q4_K.gguf","size":9053114560},"Q4_K_S":{"file":"phi-4-Q4_K_S.gguf","size":8440762560},"Q5_0":{"file":"phi-4-Q5_0.gguf","size":10151579840},"Q5_1":{"file":"phi-4-Q5_1.gguf","size":11035660480},"Q5_K":{"file":"phi-4-Q5_K.gguf","size":10604187840},"Q5_K_S":{"file":"phi-4-Q5_K_S.gguf","size":10151579840},"Q6_K":{"file":"phi-4-Q6_K.gguf","size":12030251200},"Q8_0":{"file":"phi-4-Q8_0.gguf","size":15580500160},"DEFAULT":{"file":"phi-4-TQ1_0.gguf","size":3591098560},"BF16":{"file":"phi-4-bf16.gguf","size":29323399360}},"license":"mit"}},
    "phi-4-mini-instruct": {"category":"gen","gguf":{"repo":"unsloth/Phi-4-mini-instruct-GGUF","revision":"78eb92a46fc37e6b524df991ed9aca9bc6aa7b80","quants":{"Q2_K":{"file":"Phi-4-mini-instruct-Q2_K.gguf","size":1682635744},"Q2_K_L":{"file":"Phi-4-mini-instruct-Q2_K_L.gguf","size":1682635744},"Q3_K_M":{"file":"Phi-4-mini-instruct-Q3_K_M.gguf","size":2117532640},"Q4_K_M":{"file":"Phi-4-mini-instruct-Q4_K_M.gguf","size":2491874272},"Q5_K_M":{"file":"Phi-4-mini-instruct-Q5_K_M.gguf","size":2848127968},"Q6_K":{"file":"Phi-4-mini-instruct-Q6_K.gguf","size":3155622880},"BF16":{"file":"Phi-4-mini-instruct.BF16.gguf","size":7680694240},"Q8_0":{"file":"Phi-4-mini-instruct.Q8_0.gguf","size":4084611040}},"license":"mit"}},
    "mistral-7b-instruct-v0.3": {"category":"gen","gguf":{"repo":"bartowski/Mistral-7B-Instruct-v0.3-GGUF","revision":"61fd4167fff3ab01ee1cfe0da183fa27a944db48","quants":{"IQ1_M":{"file":"Mistral-7B-Instruct-v0.3-IQ1_M.gguf","size":1757663456},"IQ1_S":{"file":"Mistral-7B-Instruct-v0.3-IQ1_S.gguf","size":1615319264},"IQ2_M":{"file":"Mistral-7B-Instruct-v0.3-IQ2_M.gguf","size":2504249568},"IQ2_S":{"file":"Mistral-7B-Instruct-v0.3-IQ2_S.gguf","size":2314457312},"IQ2_XS":{"file":"Mistral-7B-Instruct-v0.3-IQ2_XS.gguf","size":2201473248},"IQ2_XXS":{"file":"Mistral-7B-Instruct-v0.3-IQ2_XXS.gguf","size":1994903776},"IQ3_M":{"file":"Mistral-7B-Instruct-v0.3-IQ3_M.gguf","size":3288846560},"IQ3_S":{"file":"Mistral-7B-Instruct-v0.3-IQ3_S.gguf","size":3186348256},"IQ3_XS":{"file":"Mistral-7B-Instruct-v0.3-IQ3_XS.gguf","size":3022770400},"IQ3_XXS":{"file":"Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf","size":2830880992},"IQ4_NL":{"file":"Mistral-7B-Instruct-v0.3-IQ4_NL.gguf","size":4130066656},"IQ4_XS":{"file":"Mistral-7B-Instruct-v0.3-IQ4_XS.gguf","size":3911962848},"Q2_K":{"file":"Mistral-7B-Instruct-v0.3-Q2_K.gguf","size":2722877664},"Q3_K_L":{"file":"Mistral-7B-Instruct-v0.3-Q3_K_L.gguf","size":3825979616},"Q3_K_M":{"file":"Mistral-7B-Instruct-v0.3-Q3_K_M.gguf","size":3522941152},"Q3_K_S":{"file":"Mistral-7B-Instruct-v0.3-Q3_K_S.gguf","size":3168522464},"Q4_K_M":{"file":"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf","size":4372812000},"Q4_K_S":{"file":"Mistral-7B-Instruct-v0.3-Q4_K_S.gguf","size":4144746720},"Q5_K_M":{"file":"Mistral-7B-Instruct-v0.3-Q5_K_M.gguf","size":5136175328},"Q5_K_S":{"file":"Mistral-7B-Instruct-v0.3-Q5_K_S.gguf","size":5002481888},"Q6_K":{"file":"Mistral-7B-Instruct-v0.3-Q6_K.gguf","size":5947248864},"Q8_0":{"file":"Mistral-7B-Instruct-v0.3-Q8_0.gguf","size":7702565088},"F32":{"file":"Mistral-7B-Instruct-v0.3-f32.gguf","size":28992851904}},"license":"apache-2.0"}},
    "smollm2-1.7b-instruct": {"category":"gen","gguf":{"repo":"bartowski/SmolLM2-1.7B-Instruct-GGUF","revision":"1f03464768bfcc0319fc50da8ff5fb20b6417ba2","quants":{"IQ3_M":{"file":"SmolLM2-1.7B-Instruct-IQ3_M.gguf","size":810243040},"IQ3_XS":{"file":"SmolLM2-1.7B-Instruct-IQ3_XS.gguf","size":739070944},"IQ4_XS":{"file":"SmolLM2-1.7B-Instruct-IQ4_XS.gguf","size":940397536},"Q2_K":{"file":"SmolLM2-1.7B-Instruct-Q2_K.gguf","size":674583520},"Q2_K_L":{"file":"SmolLM2-1.7B-Instruct-Q2_K_L.gguf","size":698962912},"Q3_K_L":{"file":"SmolLM2-1.7B-Instruct-Q3_K_L.gguf","size":932533216},"Q3_K_M":{"file":"SmolLM2-1.7B-Instruct-Q3_K_M.gguf","size":860181472},"Q3_K_S":{"file":"SmolLM2-1.7B-Instruct-Q3_K_S.gguf","size":776819680},"Q3_K_XL":{"file":"SmolLM2-1.7B-Instruct-Q3_K_XL.gguf","size":956912608},"Q4_0":{"file":"SmolLM2-1.7B-Instruct-Q4_0.gguf","size":993874912},"Q4_0_4_4":{"file":"SmolLM2-1.7B-Instruct-Q4_0_4_4.gguf","size":990729184},"Q4_0_4_8":{"file":"SmolLM2-1.7B-Instruct-Q4_0_4_8.gguf","size":990729184},"Q4_0_8_8":{"file":"SmolLM2-1.7B-Instruct-Q4_0_8_8.gguf","size":990729184},"Q4_K_L":{"file":"SmolLM2-1.7B-Instruct-Q4_K_L.gguf","size":1079989216},"Q4_K_M":{"file":"SmolLM2-1.7B-Instruct-Q4_K_M.gguf","size":1055609824},"Q4_K_S":{"file":"SmolLM2-1.7B-Instruct-Q4_K_S.gguf","size":999117792},"Q5_K_L":{"file":"SmolLM2-1.7B-Instruct-Q5_K_L.gguf","size":1249858528},"Q5_K_M":{"file":"SmolLM2-1.7B-Instruct-Q5_K_M.gguf","size":1225479136},"Q5_K_S":{"file":"SmolLM2-1.7B-Instruct-Q5_K_S.gguf","size":1192055776},"Q6_K":{"file":"SmolLM2-1.7B-Instruct-Q6_K.gguf","size":1405965280},"Q6_K_L":{"file":"SmolLM2-1.7B-Instruct-Q6_K_L.gguf","size":1430344672},"Q8_0":{"file":"SmolLM2-1.7B-Instruct-Q8_0.gguf","size":1820414944},"F16":{"file":"SmolLM2-1.7B-Instruct-f16.gguf","size":3424735936}},"license":"apache-2.0"}},
    "deepseek-r1-qwen-7b": {"category":"gen","gguf":{"repo":"bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF","revision":"361004151d4f4f6b446dc5e6d46fbf4422a80d5f","quants":{"IQ2_M":{"file":"DeepSeek-R1-Distill-Qwen-7B-IQ2_M.gguf","size":2780342240},"IQ3_M":{"file":"DeepSeek-R1-Distill-Qwen-7B-IQ3_M.gguf","size":3574011872},"IQ3_XS":{"file":"DeepSeek-R1-Distill-Qwen-7B-IQ3_XS.gguf","size":3346255840},"IQ4_NL":{"file":"DeepSeek-R1-Distill-Qwen-7B-IQ4_NL.gguf","size":4437813216},"IQ4_XS":{"file":"DeepSeek-R1-Distill-Qwen-7B-IQ4_XS.gguf","size":4218472416},"Q2_K":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf","size":3015940064},"Q2_K_L":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q2_K_L.gguf","size":3548164064},"Q3_K_L":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q3_K_L.gguf","size":4088459232},"Q3_K_M":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf","size":3808391136},"Q3_K_S":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q3_K_S.gguf","size":3492368352},"Q3_K_XL":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q3_K_XL.gguf","size":4565331936},"Q4_0":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf","size":4444121056},"Q4_1":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q4_1.gguf","size":4873283552},"Q4_K_L":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q4_K_L.gguf","size":5087563744},"Q4_K_M":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf","size":4683073504},"Q4_K_S":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q4_K_S.gguf","size":4457768928},"Q5_K_L":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q5_K_L.gguf","size":5781196768},"Q5_K_M":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf","size":5444831200},"Q5_K_S":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q5_K_S.gguf","size":5315176416},"Q6_K":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf","size":6254198752},"Q6_K_L":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q6_K_L.gguf","size":6518181856},"Q8_0":{"file":"DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf","size":8098525152},"F16":{"file":"DeepSeek-R1-Distill-Qwen-7B-f16.gguf","size":15237853152},"F32":{"file":"DeepSeek-R1-Distill-Qwen-7B-f32.gguf","size":30468419264}},"license":null},"license":"mit"},
    "granite-3.3-8b-instruct": {"category":"gen","gguf":{"repo":"ibm-granite/granite-3.3-8b-instruct-GGUF","revision":"e40e9dd739c7be00fa965c16ce167088190ce114","quants":{"Q2_K":{"file":"granite-3.3-8b-instruct-Q2_K.gguf","size":3103605504},"Q3_K_L":{"file":"granite-3.3-8b-instruct-Q3_K_L.gguf","size":4349444864},"Q3_K_M":{"file":"granite-3.3-8b-instruct-Q3_K_M.gguf","size":3996599040},"Q3_K_S":{"file":"granite-3.3-8b-instruct-Q3_K_S.gguf","size":3592504064},"Q4_0":{"file":"granite-3.3-8b-instruct-Q4_0.gguf","size":4650910464},"Q4_1":{"file":"granite-3.3-8b-instruct-Q4_1.gguf","size":5148984064},"Q4_K_M":{"file":"granite-3.3-8b-instruct-Q4_K_M.gguf","size":4942873344},"Q4_K_S":{"file":"granite-3.3-8b-instruct-Q4_K_S.gguf","size":4685775616},"Q5_0":{"file":"granite-3.3-8b-instruct-Q5_0.gguf","size":5647057664},"Q5_1":{"file":"granite-3.3-8b-instruct-Q5_1.gguf","size":6145131264},"Q5_K_M":{"file":"granite-3.3-8b-instruct-Q5_K_M.gguf","size":5797462784},"Q5_K_S":{"file":"granite-3.3-8b-instruct-Q5_K_S.gguf","size":5647057664},"Q6_K":{"file":"granite-3.3-8b-instruct-Q6_K.gguf","size":6705464064},"Q8_0":{"file":"granite-3.3-8b-instruct-Q8_0.gguf","size":8684264992},"F16":{"file":"granite-3.3-8b-instruct-f16.gguf","size":16344139552}},"license":"apache-2.0"}},
    "tinyllama-1.1b-chat": {"category":"gen","gguf":{"repo":"TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF","revision":"52e7645ba7c309695bec7ac98f4f005b139cf465","quants":{"Q2_K":{"file":"tinyllama-1.1b-chat-v1.0.Q2_K.gguf","size":483116416},"Q3_K_L":{"file":"tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf","size":592500096},"Q3_K_M":{"file":"tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf","size":550819200},"Q3_K_S":{"file":"tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf","size":500315520},"Q4_0":{"file":"tinyllama-1.1b-chat-v1.0.Q4_0.gguf","size":637699456},"Q4_K_M":{"file":"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf","size":668788096},"Q4_K_S":{"file":"tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf","size":643728768},"Q5_0":{"file":"tinyllama-1.1b-chat-v1.0.Q5_0.gguf","size":767001984},"Q5_K_M":{"file":"tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf","size":783017344},"Q5_K_S":{"file":"tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf","size":767001984},"Q6_K":{"file":"tinyllama-1.1b-chat-v1.0.Q6_K.gguf","size":904385920},"Q8_0":{"file":"tinyllama-1.1b-chat-v1.0.Q8_0.gguf","size":1170781568}},"license":"apache-2.0"}},
    "clip-vit-b-32-laion": {"category":"clip","gguf":{"repo":"mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K","revision":"26ebd3e1648320e965df9e69ca01963d144cb380","quants":{"F16":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf","size":303810560},"F32":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f32.gguf","size":601237504},"Q4_0":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q4_0.gguf","size":90035008},"Q4_1":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q4_1.gguf","size":99329600},"Q5_0":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q5_0.gguf","size":108624192},"Q5_1":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q5_1.gguf","size":117918784},"Q8_0":{"file":"CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-q8_0.gguf","size":164391744}},"license":"mit"},"dim":512},
    "clip-vit-b-32": {"category":"clip","gguf":{"repo":"mys/ggml_clip-vit-base-patch32","revision":"3d547b9f9411f696e94febaf81048c9304c14ec7","quants":{"F16":{"file":"clip-vit-base-patch32_ggml-model-f16.gguf","size":303810624},"F32":{"file":"clip-vit-base-patch32_ggml-model-f32.gguf","size":601237568},"Q4_0":{"file":"clip-vit-base-patch32_ggml-model-q4_0.gguf","size":90035040},"Q4_1":{"file":"clip-vit-base-patch32_ggml-model-q4_1.gguf","size":99329632},"Q5_0":{"file":"clip-vit-base-patch32_ggml-model-q5_0.gguf","size":108624224},"Q5_1":{"file":"clip-vit-base-patch32_ggml-model-q5_1.gguf","size":117918816},"Q8_0":{"file":"clip-vit-base-patch32_ggml-model-q8_0.gguf","size":164391776}},"license":"mit"},"dim":512},
    "clip-vit-l-14-laion": {"category":"clip","gguf":{"repo":"mys/ggml_CLIP-ViT-L-14-laion2B-s32B-b82K","revision":"b0f40e33c510fec00770ddb5436d20b8daa413c4","quants":{"F16":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-f16.gguf","size":856982496},"F32":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-f32.gguf","size":1710119392},"Q4_0":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q4_0.gguf","size":243790400},"Q4_1":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q4_1.gguf","size":270450912},"Q5_0":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q5_0.gguf","size":297111456},"Q5_1":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q5_1.gguf","size":323771968},"Q8_0":{"file":"CLIP-ViT-L-14-laion2B-s32B-b82K_ggml-model-q8_0.gguf","size":457074624}},"license":"mit"},"dim":768},
    "clip-vit-l-14": {"category":"clip","gguf":{"repo":"mys/ggml_clip-vit-large-patch14","revision":"428a41bc7ff7e0a5b9673c4794588a04855e80b7","quants":{"F16":{"file":"clip-vit-large-patch14_ggml-model-f16.gguf","size":856982528},"F32":{"file":"clip-vit-large-patch14_ggml-model-f32.gguf","size":1710119424},"Q4_0":{"file":"clip-vit-large-patch14_ggml-model-q4_0.gguf","size":243790464},"Q4_1":{"file":"clip-vit-large-patch14_ggml-model-q4_1.gguf","size":270450976},"Q5_0":{"file":"clip-vit-large-patch14_ggml-model-q5_0.gguf","size":297111520},"Q5_1":{"file":"clip-vit-large-patch14_ggml-model-q5_1.gguf","size":323772032},"Q8_0":{"file":"clip-vit-large-patch14_ggml-model-q8_0.gguf","size":457074688}},"license":"mit"},"dim":768},
    "clip-vit-h-14-laion": {"category":"clip","gguf":{"repo":"mys/ggml_CLIP-ViT-H-14-laion2B-s32B-b79K","revision":"0f537fcbcb0ef8e0c4522801be05e767dd39808e","quants":{"F16":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-f16.gguf","size":1974815104},"F32":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-f32.gguf","size":3943807872},"Q4_0":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-q4_0.gguf","size":559601632},"Q4_1":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-q4_1.gguf","size":621132640},"Q5_0":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-q5_0.gguf","size":682663680},"Q5_1":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-q5_1.gguf","size":744194688},"Q8_0":{"file":"CLIP-ViT-H-14-laion2B-s32B-b79K_ggml-model-q8_0.gguf","size":1051849824}},"license":"mit"},"dim":1024},
    "ppocr-v5": {"category":"ocr","ocr":{"repo":"bukuroo/PPOCRv5-ONNX","revision":"47b3e1b4e90c79737cb71f562a6c85809067c7a5","license":"apache-2.0","defaultVariant":"multi_mobile","common":{"cls":{"file":"ppocrv5-cls.onnx","size":582663}},"variants":{"multi_mobile":{"det":{"file":"ppocrv5-mobile-det.onnx","size":4748769},"rec":{"file":"ppocrv5-mobile-rec.onnx","size":16517247},"dict":{"file":"ppocrv5_dict.txt","size":92395}},"multi_server":{"det":{"file":"ppocrv5-server-det.onnx","size":87697340},"rec":{"file":"ppocrv5-server-rec.onnx","size":84137438},"dict":{"file":"ppocrv5_dict.txt","size":92395}},"en_mobile":{"det":{"file":"ppocrv5-mobile-det.onnx","size":4748769},"rec":{"repo":"PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx","revision":"3fafbc3b5dcf","file":"inference.onnx","as":"ppocrv5-en-rec.onnx","size":7848423},"dict":{"repo":"PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx","revision":"3fafbc3b5dcf","file":"inference.yml","as":"ppocrv5-en-dict.txt","dictFromYml":true,"size":3964}}}}},
    "ppocr-layout": {"category":"ocr","ocr":{"repo":"alex-dinh/PP-DocLayoutV3-ONNX","revision":"7952bce3e684c7aa90aa7bf47798e8efae3c0921","license":"apache-2.0","defaultVariant":"v3","common":{},"variants":{"v3":{"layout":{"file":"PP-DocLayoutV3.onnx","size":130502049}}}}}
};
/* ==== END GENERATED BUILTIN CATALOG ==== */
/* ------------------------------------------------------------------ *
 * catalog loading (see "The model catalog" above)
 * ------------------------------------------------------------------ */
function catalogCachePath() { return modelsDir() + "/." + CATALOG_FILE; }
function catalogMetaPath()  { return modelsDir() + "/." + CATALOG_FILE + ".meta"; }

/* this script's own directory: module.id when require()'d, the script dir
 * when run from the command line */
function selfDir() {
    var id = (typeof module !== "undefined" && module && module.id) || null;
    return id ? id.replace(/\/[^\/]*$/, "") : process.scriptPath;
}

function envOn(name) {
    var v = u.getenv(name);
    return !!v && !/^(0|false|no|off)$/i.test(v);
}

/* parse + sanity-check a catalog document.  Throws on anything that isn't
 * a plausible catalog, so a truncated download or an error page served as
 * 200 can never replace a good cache. */
function parseCatalogDoc(txt) {
    var j = JSON.parse(txt);
    var c = j && j.catalog;
    if (!c || typeof c !== "object" || Array.isArray(c))
        throwErr("catalog document has no 'catalog' object");
    var n = 0, k;
    for (k in c) {
        if (!c[k] || typeof c[k] !== "object") throwErr("catalog entry '%s' is not an object", k);
        n++;
    }
    if (n < CATALOG_MIN)
        throwErr("catalog has only %d entries (expected >= %d)", n, CATALOG_MIN);
    return { catalog: c, entries: n, version: j.version || null, generated: j.generated || null };
}

/* conditional GET.  Returns {notModified}, {catalog,...} or {error}. */
function fetchCatalogDoc(etag) {
    var opts = { location: true, headers: { "Accept-Encoding": "identity" } };
    if (etag) opts.headers["If-None-Match"] = etag;
    var res;
    try { res = curl.fetch(CATALOG_URL, opts); }
    catch (e) { return { error: e.message }; }
    if (!res || !res.status) return { error: "no response" };
    if (res.status === 304) return { notModified: true };
    if (res.status !== 200) return { error: u.sprintf("HTTP %d", res.status) };
    var got;
    try { got = parseCatalogDoc(res.text); }
    /* throwErr already prefixes "rampart-models: "; catalogWarn adds its
     * own, so strip it here rather than printing it twice */
    catch (e) { return { error: (e.message || "").replace(/^rampart-models: /, "") }; }
    got.etag = (res.headers && (res.headers.ETag || res.headers.etag)) || null;
    got.text = res.text;
    return got;
}

function saveCatalogCache(text, meta) {
    try {
        u.mkDir(modelsDir());
        /* atomic: a concurrent reader sees the old or the new file, never
         * a half-written one */
        var tmp = catalogCachePath() + ".tmp" + Math.floor(Math.random() * 1e9);
        u.writeFile(tmp, text);
        u.rename(tmp, catalogCachePath());
        u.writeFile(catalogMetaPath(), u.sprintf("%4J", meta));
        return true;
    } catch (e) {
        catalogState.error = "cannot write cache: " + e.message;
        return false;
    }
}

function nBuiltin() { var n = 0; for (var k in BUILTIN_CATALOG) n++; return n; }

var catalogWarned = false;
function catalogWarn(fmt) {
    if (catalogWarned) return;
    catalogWarned = true;
    var args = Array.prototype.slice.call(arguments);
    u.fprintf(STDERR, "rampart-models: %s\n", u.sprintf.apply(null, args));
}

/* Load the catalog into CATALOG, then overlay the hand-maintained OCR
 * entries.  Never throws: a missing catalog degrades to live resolution. */
function initCatalog(o) {
    o = o || {};
    var now = Math.floor(Date.now() / 1000), doc = null;

    /* start from a clean slate: a reload (updateCatalog) must not report
     * the previous load's source, url or timestamps */
    catalogState.source = "none";
    catalogState.url = CATALOG_URL;
    catalogState.entries = catalogState.builtin = catalogState.models = 0;
    catalogState.version = catalogState.generated = null;
    catalogState.fetched = catalogState.checked = catalogState.etag = null;
    catalogState.error = null;

    /* 1. beside the script (repo checkout / hand-placed pin) */
    var local = selfDir() + "/" + CATALOG_FILE;
    if (!o.remote && isFile(local)) {
        try {
            doc = parseCatalogDoc(u.readFile(local, true));
            catalogState.source = "local";
            catalogState.url = local;
        } catch (e) {
            catalogWarn("ignoring %s: %s", local, (e.message || "").replace(/^rampart-models: /, ""));
            doc = null;
        }
    }

    if (!doc) {
        /* 2. the cache, plus its metadata (etag + when we last checked) */
        var cached = null, meta = {};
        try { meta = JSON.parse(u.readFile(catalogMetaPath(), true)) || {}; } catch (e) {}
        try { cached = parseCatalogDoc(u.readFile(catalogCachePath(), true)); } catch (e) {}

        var ttl = parseInt(u.getenv("RAMPART_MODELS_CATALOG_TTL") || "0", 10);
        if (!(ttl >= 0)) ttl = 0;
        var fresh = cached && meta.checked && (now - meta.checked) < ttl;
        var offline = envOn("RAMPART_MODELS_CATALOG_OFFLINE");

        if (cached && (offline || (fresh && !o.force))) {
            doc = cached;
            catalogState.source = "cache";
            catalogState.error = offline ? "offline (RAMPART_MODELS_CATALOG_OFFLINE)" : null;
        } else if (offline) {
            catalogState.error = "offline (RAMPART_MODELS_CATALOG_OFFLINE), no cache";
        } else {
            /* 3. revalidate (or fetch for the first time) */
            var got = fetchCatalogDoc(cached ? meta.etag : null);
            if (got.error) {
                doc = cached;                                  /* keep working offline */
                catalogState.source = cached ? "cache" : "none";
                catalogState.error = got.error;
                if (!cached)
                    catalogWarn("catalog update unavailable (%s): using the %d models built into this script; " +
                                "others will resolve by live HuggingFace search.  %s",
                                got.error, nBuiltin(), CATALOG_URL);
                else if (meta.fetched && (now - meta.fetched) > CATALOG_STALE_DAYS * 86400)
                    catalogWarn("model catalog is %d days old and cannot refresh (%s)",
                                Math.floor((now - meta.fetched) / 86400), got.error);
            } else if (got.notModified) {
                doc = cached;
                catalogState.source = "cache";
                meta.checked = now;
                try { u.writeFile(catalogMetaPath(), u.sprintf("%4J", meta)); } catch (e) {}
            } else {
                doc = got;
                catalogState.source = "network";
                saveCatalogCache(got.text, { etag: got.etag, fetched: now, checked: now,
                                             url: CATALOG_URL, generated: got.generated,
                                             version: got.version, entries: got.entries });
                meta = { etag: got.etag, fetched: now, checked: now };
            }
        }
        catalogState.etag = meta.etag || null;
        catalogState.fetched = meta.fetched || null;
        catalogState.checked = meta.checked || null;
    }

    /* clear IN PLACE: module.exports.catalog holds this object's identity,
     * so a later updateCatalog() must not swap in a different object */
    for (var old in CATALOG) delete CATALOG[old];

    /* built-ins first: clip / ocr / rerank and the stable embed+gen
     * workhorses resolve with no network and no cache */
    var nb = 0, k;
    for (k in BUILTIN_CATALOG) { CATALOG[k] = BUILTIN_CATALOG[k]; nb++; }
    catalogState.builtin = nb;

    /* then the fetched catalog, which OVERRIDES a built-in of the same
     * name (moved repo, re-pinned revision, new quants) and adds the rest */
    if (doc) {
        for (k in doc.catalog) CATALOG[k] = doc.catalog[k];
        catalogState.entries = doc.entries;
        catalogState.version = doc.version;
        catalogState.generated = doc.generated;
    } else {
        catalogState.entries = 0;
    }
    var tot = 0;
    for (k in CATALOG) tot++;
    catalogState.models = tot;
    return catalogState;
}

/* force a refresh from the catalog URL (ignores the TTL).  Pass
 * {remote:true} to bypass a catalog sitting beside the script. */
function updateCatalog(o) {
    o = o ? Object.assign({}, o) : {};
    o.force = true;
    initCatalog(o);
    return catalogInfo();
}

/* where the active catalog came from, and how current it is */
function catalogInfo() {
    var s = {}, k;
    for (k in catalogState) s[k] = catalogState[k];
    s.cache = catalogCachePath();
    return s;
}

initCatalog();

/* ------------------------------------------------------------------ *
 * small helpers
 * ------------------------------------------------------------------ */
function mb(n) { return n / 1048576; }
function sizeStr(b) {
    if (!(b > 0)) return "size unknown";
    return b >= 1073741824 ? (b / 1073741824).toFixed(1) + " GB" : Math.round(mb(b)) + " MB";
}

/* If o.confirm is set, ask it before an actual download; a falsy answer means
 * "skip", and get() returns null.  info = {name, format, dest, size, bytes,
 * precision|quant, repo}.  Kept non-interactive by default: the module never
 * touches stdin -- the caller's callback owns any prompt. */
function confirmed(o, info) { return !o.confirm || !!o.confirm(info); }

function isDir(p)  { var s = u.stat(p); return s && s.isDirectory; }
function isFile(p) { var s = u.stat(p); return s && s.isFile; }
function fsize(p)  { var s = u.stat(p); return s ? s.size : -1; }

function compact(s) { return s.replace(/[-_.]/g, "").toLowerCase(); }
function famWord(alias) { return (alias.match(/^[a-z]+/i) || [alias])[0].toLowerCase(); }

function normName(repoId) {
    var b = repoId.split("/").pop();
    for (;;) {
        var before = b;
        b = b.replace(/[-_.](gguf|onnx)$/i, "");
        b = b.replace(/[-_.]?(embedding|text-embedding)$/i, "");
        b = b.replace(/[-_.](i?q[0-9][a-z0-9_]*|f16|f32|bf16|fp16)$/i, "");
        if (b === before) break;
    }
    b = b.replace(/^(meta|microsoft|sentence-transformers)[-_]/i, "");
    return b.toLowerCase();
}

function apiGet(path, token) {
    var opts = {};
    if (token) opts.headers = { "Authorization": "Bearer " + token };
    var res, backoff = [2, 8];
    for (var a = 0; ; a++) {
        res = curl.fetch(HF + path, opts);
        if (res.status === 200) break;
        if (a < backoff.length && (res.status === 429 || res.status >= 500 || res.status <= 0)) {
            u.sleep(backoff[a]);
            continue;
        }
        return { _status: res.status };
    }
    try { return JSON.parse(res.text); } catch (e) { return { _status: -1 }; }
}

function repoTree(repo, rev, token) {
    var j = apiGet("/api/models/" + encodeURI(repo) + "/tree/" +
                   encodeURI(rev || "main") + "?recursive=true", token);
    if (!Array.isArray(j)) return null;
    return j.map(function (e) {
        return { path: e.path, size: (e.lfs && e.lfs.size) || e.size || 0,
                 sha256: (e.lfs && e.lfs.oid) || null };
    });
}

function resolveUrl(repo, rev, path) {
    return HF + "/" + repo + "/resolve/" + (rev || "main") + "/" + path;
}

/* store category from a HF pipeline tag -- the same signal the generator's
 * discovery sweeps key on.  Checked on pipeline_tag first, then the tags[]
 * array (converter repos often carry the tag only there). */
var TAG_CATEGORY = {
    "text-generation": "gen", "text2text-generation": "gen",
    "sentence-similarity": "embed", "feature-extraction": "embed",
    "text-ranking": "rerank"
};
function categoryFromApi(info) {
    if (!info) return null;
    if (TAG_CATEGORY[info.pipeline_tag]) return TAG_CATEGORY[info.pipeline_tag];
    /* before the tags[] fallback: converter repos of cross-encoder rerankers
     * predate the text-ranking tag (text-classification pipeline, tags that
     * include feature-extraction) -- the repo name is the reliable signal */
    if (info.id && /rerank/i.test(info.id)) return "rerank";
    if (Array.isArray(info.tags))
        for (var i = 0; i < info.tags.length; i++)
            if (TAG_CATEGORY[info.tags[i]]) return TAG_CATEGORY[info.tags[i]];
    return null;
}

/* Retrieval prompts from a repo's config_sentence_transformers.json "prompts"
 * dict (generator parity: keys normalized to query / document / documentWithTitle,
 * "passage" -> document, empty strings dropped, unknown task keys ignored). */
function fetchPrompts(repo, token) {
    var opts = { location: true };                 /* resolve/ URLs 307-redirect */
    if (token) opts.headers = { "Authorization": "Bearer " + token };
    var res = curl.fetch(HF + "/" + repo + "/resolve/main/config_sentence_transformers.json", opts);
    if (res.status !== 200) return null;
    var out = null;
    try {
        var p = JSON.parse(res.text).prompts;
        var map = { query: "query", document: "document", passage: "document",
                    documentWithTitle: "documentWithTitle" };
        if (p && typeof p === "object")
            for (var k in p) {
                if (!map[k] || typeof p[k] !== "string" || !p[k].length) continue;
                out = out || {};
                if (!(map[k] in out)) out[map[k]] = p[k];
            }
    } catch (e) {}
    return out;
}

/* write <model-file>.prompts.json when the entry knows the model's retrieval
 * prompts; no-op (and no sidecar) otherwise.  Idempotent -- unchanged content
 * is not rewritten, so mtimes stay stable for loader caches. */
function writePromptSidecar(modelFile, entry) {
    if (!entry || !entry.prompts) return;
    var f = modelFile + ".prompts.json";
    var body = u.sprintf("%4J", { prompts: entry.prompts });
    try { if (u.readFile(f, true) === body) return; } catch (e) {}
    try { u.writeFile(f, body); }
    catch (e) {
        u.fprintf(STDERR, "rampart-models: cannot write %s (%s) -- prompt prefixes will not be auto-applied\n",
                  f, e.message);
    }
}

/* remembered live resolutions: name -> {category, onnx?, gguf?} (catalog shape) */
function resolvedPath() { return modelsDir() + "/.resolved.json"; }
function loadResolved() {
    try { return JSON.parse(u.readFile(resolvedPath(), true)); } catch (e) { return {}; }
}
function saveResolved(map) {
    u.mkDir(modelsDir());
    u.writeFile(resolvedPath(), u.sprintf("%4J", map));
}

/* ------------------------------------------------------------------ *
 * transport: one file, streamed, resumable, with progress
 * ------------------------------------------------------------------ */
function throwErr(fmt) {
    var args = Array.prototype.slice.call(arguments);
    throw new Error("rampart-models: " + u.sprintf.apply(null, args));
}

function progressLine(prog, label, got, total, secs) {
    if (prog === false) return;
    var rate = secs > 0 ? mb(got) / secs : 0;
    if (typeof prog === "function") {
        prog({ file: label, got: got, total: total, mbps: rate });
        return;
    }
    var fh = (prog === undefined || prog === true) ? STDOUT : prog;
    if (total > 0)
        u.fprintf(fh, "\r  %-44s %5.1f%%  %8.1f / %.1f MB  %6.2f MB/s   ",
                  label, 100 * got / total, mb(got), mb(total), rate);
    else
        u.fprintf(fh, "\r  %-44s %8.1f MB  %6.2f MB/s   ", label, mb(got), rate);
    u.fflush(fh);
}
function progressEnd(prog, label, got) {
    if (prog === false || typeof prog === "function") return;
    var fh = (prog === undefined || prog === true) ? STDOUT : prog;
    u.fprintf(fh, "\r  %-44s done (%.1f MB)%*s\n", label, mb(got), 24, "");
}

/* normalize an expected sha256 (HF LFS oid; tolerate a "sha256:" prefix) */
function wantSha(o) {
    return o.sha256 ? String(o.sha256).replace(/^sha256:/i, "").toLowerCase() : null;
}

/* stream len bytes of an existing file into a HashStream, 8MB at a time */
function hashFeedFile(h, path, len) {
    var fh = u.fopen(path, "r"), b;
    while (len > 0 && (b = fh.fread(Math.min(8388608, len))) && b.byteLength) {
        h.update(b);
        len -= b.byteLength;
    }
    fh.fclose();
}

/* sha256 of an existing file (verifying an already-complete .part) */
function sha256File(path) {
    var h = new crypto.HashStream("sha256");
    hashFeedFile(h, path, fsize(path));
    return h.final();
}

/* download url -> dest (file).  size/sha256 optional (verified when known;
 * sha256 is hashed incrementally in chunkCallback -- no second read).
 * Resumes an existing dest.part; retries transient failures. */
function fetchFile(url, dest, o) {
    o = o || {};
    if (!o.force && isFile(dest) && (o.size === undefined || o.size <= 0 || fsize(dest) === o.size))
        return dest;                                   /* already here */

    var slash = dest.lastIndexOf("/");
    if (slash > 0) u.mkDir(dest.substring(0, slash));
    var part = dest + ".part";
    var label = o.label || dest.split("/").pop();
    var want = wantSha(o);

    /* a previous run may have fully downloaded the payload and died before
     * the rename (or between): finalize it instead of re-downloading */
    if (!o.force && o.size > 0 && isFile(part) && fsize(part) === o.size) {
        if (want && sha256File(part) !== want)
            u.rmFile(part);                            /* corrupt: re-download */
        else {
            u.rename(part, dest);
            return dest;
        }
    }

    for (var attempt = 0; attempt < 4; attempt++) {
        var base = 0;
        /* identity encoding is essential: HF gzip-compresses small text files
         * when the client advertises gzip, and chunkCallback receives the RAW
         * wire bytes (no inflate) -- we'd write a .gz to disk. */
        var headers = { "Accept-Encoding": "identity" };
        if (o.token) headers["Authorization"] = "Bearer " + o.token;
        var mode = "w";
        if (isFile(part) && attempt < 3) {             /* final attempt: start clean */
            base = fsize(part);
            if (base > 0) { headers["Range"] = "bytes=" + base + "-"; mode = "a"; }
        }
        /* hash incrementally as chunks arrive; on a resume, feed the bytes
         * already on disk first so the digest covers the whole file */
        var hash = want ? new crypto.HashStream("sha256") : null;
        if (hash && base > 0)
            hashFeedFile(hash, part, base);
        var f = u.fopen(part, mode);
        var got = base, lastDraw = 0;
        var res = { status: 0 };
        curl.fetch(url, {
            location: true,
            headers: headers,
            /* essential for large files: without it rampart-curl ALSO
             * accumulates the whole body in memory for the final result,
             * and duktape buffers cap at ~2GiB (RangeError: buffer too long) */
            skipFinalRes: true,
            chunkCallback: function (r) {
                f.fprintf("%s", r.body);
                if (hash) hash.update(r.body);
                /* count bytes ourselves: r.progress is 32-bit and wraps
                 * negative past 2GiB */
                var n = r.body.byteLength;
                if (n === undefined) n = r.body.length || 0;
                got += n;
            },
            progressCallback: function (r) {
                var now = r.totalTime;
                if (now - lastDraw >= 0.2) {
                    lastDraw = now;
                    progressLine(o.progress, label, got, o.size || 0, now);
                }
            },
            /* with chunkCallback, fetch is still blocking but the result
             * object arrives via the callback rather than the return value */
            callback: function (r) { res = r; }
        });
        f.fclose();

        /* We request identity encoding, but a non-compliant server could
         * still send compressed bytes -- and chunkCallback writes RAW wire
         * bytes, so that would silently store a .gz.  Fail loudly instead. */
        var enc = res.headers && (res.headers["Content-Encoding"] || res.headers["content-encoding"]);
        if (enc && !/^identity$/i.test(enc)) {
            u.rmFile(part);
            throwErr("%s: server sent Content-Encoding '%s' despite identity request -- refusing to store compressed bytes", url, enc);
        }
        if (res.status === 416) {
            /* range past EOF: if the part is in fact complete, finalize it.
             * Re-hash from disk rather than trusting `hash`: a 416 can carry
             * an error body that chunkCallback fed into the digest. */
            if (o.size > 0 && fsize(part) === o.size && (!want || sha256File(part) === want)) {
                u.rename(part, dest);
                progressEnd(o.progress, label, o.size);
                return dest;
            }
            u.rmFile(part);
            continue;
        }
        if (base > 0 && res.status === 200) {
            /* we sent a Range but the server sent the FULL body -- our append
             * just corrupted the part.  Start clean. */
            u.rmFile(part);
            continue;
        }
        if (res.status !== 200 && res.status !== 206) {
            if (res.status === 401 || res.status === 403)
                throwErr("%s: HTTP %d -- gated/private? set HF_TOKEN or pass opts.token", url, res.status);
            if (attempt < 3) { u.sleep(2 + attempt * 4); continue; }
            throwErr("%s: HTTP %d after %d attempts", url, res.status, attempt + 1);
        }

        var have = fsize(part);
        /* belt-and-suspenders: some servers compress without declaring it in
         * a header we see.  If the payload starts with the gzip magic and the
         * destination isn't itself a compressed-format file, reject it. */
        if (!/\.(gz|tgz|zip|bz2|xz|zst|7z)$/i.test(dest)) {
            try {
                var fh2 = u.fopen(part, "r");
                var magic = new Uint8Array(fh2.fread(2));
                fh2.fclose();
                if (magic.length === 2 && magic[0] === 0x1f && magic[1] === 0x8b) {
                    u.rmFile(part);
                    throwErr("%s: payload arrived gzip-compressed (server ignored Accept-Encoding: identity)", url);
                }
            } catch (e) { if (/gzip-compressed/.test(e.message)) throw e; }
        }
        if (o.size > 0 && have !== o.size) {
            if (attempt < 3) continue;                 /* resume the remainder */
            throwErr("%s: size mismatch (%d != %d)", dest, have, o.size);
        }
        if (hash) {
            var got256 = hash.final();
            if (got256 !== want) {
                u.rmFile(part);                        /* never keep corrupt bytes */
                if (attempt < 3) continue;             /* clean re-download */
                throwErr("%s: sha256 mismatch (%s != %s)", dest, got256, want);
            }
        }
        u.rename(part, dest);
        progressEnd(o.progress, label, have);
        return dest;
    }
    throwErr("%s: download failed", url);
}

/* ------------------------------------------------------------------ *
 * gguf / onnx materialization
 * ------------------------------------------------------------------ */
function pickQuant(quants, want, allowVariant) {
    var keys = Object.keys(quants);
    if (want) {
        var w = want.toUpperCase();
        if (quants[w]) return w;
        for (var i = 0; i < keys.length; i++)
            if (keys[i].toUpperCase() === w) return keys[i];
        /* no exact match: same quant under a vendor prefix (UD-Q4_K_M for
         * Q4_K_M, or vice versa)?  Different quantisation strategy, so
         * substituting silently would be wrong -- name it in the error, or
         * take it when the caller opted in with allowVariant. */
        var near = [];
        for (i = 0; i < keys.length; i++) {
            var k = keys[i].toUpperCase();
            if (k !== w && k.replace(/^UD-/, "") === w.replace(/^UD-/, ""))
                near.push(keys[i]);
        }
        if (near.length && allowVariant) return near[0];
        throwErr("quant '%s' not available%s; have: %s", want,
                 near.length ? " (did you mean '" + near.join("' / '") +
                               "'? pass allowVariant:true to accept)" : "",
                 keys.sort().join(", "));
    }
    /* each preference tier in plain then UD form before the next tier:
     * a repo shipping UD-Q4_K_M + plain Q8_0 should default to the 4-bit
     * K-M build, not jump tiers because only UD carries it */
    var prefs = ["Q4_K_M", "Q8_0", "Q5_K_M", "Q6_K", "F16", "FP16", "Q4_0", "DEFAULT"];
    for (i = 0; i < prefs.length; i++) {
        if (quants[prefs[i]]) return prefs[i];
        if (quants["UD-" + prefs[i]]) return "UD-" + prefs[i];
    }
    return keys.sort()[0];
}

function getGguf(name, entry, o) {
    var g = entry.gguf;
    if (!g) throwErr("'%s' has no gguf source (formats: %s)", name,
                     entry.onnx ? "onnx" : "none");
    var q = pickQuant(g.quants, o.quant, o.allowVariant);
    var qi = g.quants[q];
    var cat = o.category || entry.category || "embed";
    /* a split quant is several part files that must sit side by side
     * (llama.cpp opens the first and derives its siblings' names).
     * o.dest: an existing directory (or any dest for a split quant) is the
     * directory the file(s) go into; otherwise the exact file path. */
    var files = qi.parts || [qi];
    var destIsDir = o.dest && (qi.parts || isDir(o.dest));
    var dests = files.map(function (f) {
        var base = f.file.split("/").pop();
        if (o.dest) return destIsDir ? o.dest + "/" + base : o.dest;
        return modelsDir() + "/" + cat + "/" + base;
    });
    var dest = dests[0];
    var complete = !o.force && dests.every(function (d, i) {
        return isFile(d) && (files[i].size <= 0 || fsize(d) === files[i].size);
    });
    if (complete) {
        writePromptSidecar(dest, entry);           /* backfill for older downloads */
        return dest;
    }
    if (!confirmed(o, { name: name, format: "gguf", quant: q, dest: dest,
                       bytes: qi.size, size: sizeStr(qi.size), repo: g.repo,
                       files: files.length }))
        return null;
    /* the catalog carries no digest; the tree API's lfs.oid is the file's
     * sha256.  Best-effort: an unreachable tree just skips verification. */
    var shaOf = {}, tree = repoTree(g.repo, o.revision || g.revision, o.token);
    if (tree)
        for (var ti = 0; ti < tree.length; ti++)
            shaOf[tree[ti].path] = tree[ti].sha256;
    for (var fi = 0; fi < files.length; fi++)
        fetchFile(resolveUrl(g.repo, o.revision || g.revision, files[fi].file), dests[fi], {
            size: files[fi].size, sha256: shaOf[files[fi].file] || null,
            progress: o.progress, token: o.token, force: o.force
        });
    writePromptSidecar(dest, entry);
    return dest;
}

function getOnnx(name, entry, o) {
    var x = entry.onnx;
    if (!x) throwErr("'%s' has no onnx source (formats: %s)", name,
                     entry.gguf ? "gguf (add :quant or format:'gguf')" : "none");
    var cat = o.category || entry.category || "embed";
    var dir = o.dest || (modelsDir() + "/" + cat + "/" + name);
    var rev = o.revision || x.revision;

    /* which precision variant to fetch (default fp16) */
    var prec = ("" + (o.precision || "fp16")).toLowerCase();
    if (!(prec in ONNX_PRECISION))
        throwErr("unknown onnx precision '%s' (use fp16, fp32, int8, q4)", prec);

    /* the model's directory within the repo (e.g. "onnx/"); base name is "model" */
    var mdir = x.model.replace(/[^\/]*$/, "");

    /* repos to search: the catalog repo first, then the converter mirrors, which
     * carry the whole fp16/int8/q4 matrix that official repos usually lack. */
    var b = x.repo.split("/").pop().replace(/-onnx$/i, "");
    var repos = [x.repo];
    ["onnx-community/" + b + "-ONNX", "onnx-community/" + b, "Xenova/" + b]
        .forEach(function (r) { if (repos.indexOf(r) === -1) repos.push(r); });

    /* precision fallback order: requested -> fp16 -> fp32 (fp32 always exists) */
    var order = [prec];
    if (order.indexOf("fp16") === -1) order.push("fp16");
    if (order.indexOf("fp32") === -1) order.push("fp32");

    var trees = {}, hit = null;
    function treeOf(r) {
        if (!(r in trees)) trees[r] = repoTree(r, r === x.repo ? rev : null, o.token) || null;
        return trees[r];
    }
    for (var pi = 0; pi < order.length && !hit; pi++) {
        var want = mdir + "model" + ONNX_PRECISION[order[pi]] + ".onnx";
        for (var ri = 0; ri < repos.length && !hit; ri++) {
            var t = treeOf(repos[ri]);
            if (!t) continue;
            for (var ti = 0; ti < t.length; ti++)
                if (t[ti].path === want) { hit = { repo: repos[ri], tree: t, model: want, prec: order[pi] }; break; }
        }
    }
    if (!hit) throwErr("%s: no onnx model found (precision '%s' or fp16/fp32) in %s",
                       name, prec, repos.join(", "));
    if (hit.prec !== prec)
        u.fprintf(STDERR, "rampart-models: %s: onnx precision '%s' unavailable; using '%s'\n",
                  name, prec, hit.prec);

    var repoRev = hit.repo === x.repo ? rev : null;
    var modelFile = dir + "/" + hit.model;

    /* already complete for this precision? (model variant + a tokenizer present) */
    if (!o.force && isFile(modelFile) &&
        (isFile(dir + "/tokenizer.json") || isFile(dir + "/vocab.txt"))) {
        writePromptSidecar(modelFile, entry);      /* backfill for older downloads */
        writePromptSidecar(dir, entry);            /* dir form: <dir>.prompts.json */
        return modelFile;
    }

    /* download the chosen model (+ its external-data sidecar, if any) + tokenizer/config */
    var wanted = [];
    for (var i = 0; i < hit.tree.length; i++) {
        var p = hit.tree[i].path;
        if (p === hit.model || p === hit.model + "_data" ||
            ONNX_AUX_DIRS.test(p) || ONNX_AUX.indexOf(p) !== -1)
            wanted.push(hit.tree[i]);
    }
    if (!wanted.length) throwErr("%s: nothing to download from %s?", name, hit.repo);
    var total = 0;
    for (i = 0; i < wanted.length; i++) total += (wanted[i].size || 0);
    if (!confirmed(o, { name: name, format: "onnx", precision: hit.prec, dest: dir,
                       model: modelFile, bytes: total, size: sizeStr(total), repo: hit.repo }))
        return null;
    for (i = 0; i < wanted.length; i++) {
        fetchFile(resolveUrl(hit.repo, repoRev, wanted[i].path), dir + "/" + wanted[i].path, {
            size: wanted[i].size, sha256: wanted[i].sha256,
            progress: o.progress, token: o.token, force: o.force,
            label: name + "/" + wanted[i].path
        });
    }
    /* record provenance (also marks this precision complete for future calls) */
    u.writeFile(dir + "/.source.json", u.sprintf("%4J", {
        repo: hit.repo, revision: repoRev, endpoint: HF, precision: hit.prec,
        files: wanted.map(function (w) { return w.path; })
    }));
    writePromptSidecar(modelFile, entry);
    writePromptSidecar(dir, entry);               /* dir form: <dir>.prompts.json */
    return modelFile;
}

/* Fetch an OCR model SET (PP-OCR: det + rec + cls + dict) into one directory.
 *
 * Unlike getGguf/getOnnx this returns an OBJECT, not a path:
 *
 *     { dir, det, rec, cls, dict, variant }     (absolute paths)
 *
 * because "the model" here is genuinely four files with distinct roles, and the
 * consumer (ocr.init) needs each by role.  Guessing roles from filenames later
 * would be fragile -- conversions do not agree on naming -- so the roles are
 * resolved here, where the catalog states them, and also recorded in
 * .source.json so an already-downloaded directory is self-describing.
 *
 * o.variant selects mobile (default) or server.  Returns null if o.confirm
 * declines, matching the other fetchers. */
function getOcr(name, entry, o) {
    var x = entry.ocr, i;
    if (!x) throwErr("'%s' has no ocr source", name);

    var cat = o.category || entry.category || "ocr";
    var dir = o.dest || (modelsDir() + "/" + cat + "/" + name);
    var rev = o.revision || x.revision;
    var variant = ("" + (o.variant || x.defaultVariant || "mobile")).toLowerCase();

    var vset = x.variants && x.variants[variant];
    if (!vset)
        throwErr("unknown ocr variant '%s' for %s (have: %s)", variant, name,
                 Object.keys(x.variants || {}).join(", "));

    /* role -> {file,size}: the variant's det/rec plus the shared cls/dict */
    var roles = {}, r;
    for (r in x.common)  roles[r] = x.common[r];
    for (r in vset)      roles[r] = vset[r];

    /* A role may name its OWN repo.  PaddleOCR publishes each model as a
     * separate repository, every one of them holding a file called
     * `inference.onnx`, so a role also carries `as` -- the name to store it
     * under locally -- and may ask for its character dictionary to be pulled
     * out of the accompanying `inference.yml`, which is where PaddleOCR keeps
     * it rather than in a text file. */
    var names = Object.keys(roles);
    var out = { dir: dir, variant: variant };
    function localName(role) { return roles[role].as || roles[role].file; }
    for (i = 0; i < names.length; i++)
        out[names[i]] = dir + "/" + localName(names[i]);

    /* Provenance.  Variants of one entry SHARE a directory, so a single
     * "which variant is this" field is wrong the moment a second variant is
     * fetched -- and with per-role repos a single "repo" field is wrong too.
     * Record it per FILE instead, merging with whatever is already there, so
     * the record describes the directory rather than the last call. */
    function writeProv() {
        var prov = { endpoint: HF, variants: [], files: {} }, j, k, prev;
        try { prev = JSON.parse(u.readFile(dir + "/.source.json", true)); } catch (e) { prev = null; }
        if (prev && prev.files) {
            prov.files = prev.files;
            if (prev.variants) prov.variants = prev.variants;
        }
        for (j = 0; j < names.length; j++) {
            var ro = roles[names[j]];
            prov.files[localName(names[j])] = {
                repo: ro.repo || x.repo,
                revision: ro.revision || (ro.repo ? "main" : rev)
            };
        }
        /* drop entries for files no longer on disk, so a hand-cleaned
         * directory does not keep claiming them */
        for (k in prov.files) if (!isFile(dir + "/" + k)) delete prov.files[k];
        if (prov.variants.indexOf(variant) < 0) prov.variants.push(variant);
        prov.variants.sort();
        u.writeFile(dir + "/.source.json", u.sprintf("%4J", prov));
    }

    /* already complete?  (every role present on disk) */
    var missing = [];
    for (i = 0; i < names.length; i++)
        if (!isFile(out[names[i]])) missing.push(names[i]);
    /* refresh provenance even when nothing is fetched: that is what lets a
     * stale record left by an older version heal on the next call */
    if (!o.force && !missing.length) { writeProv(); return out; }

    /* size the prompt on what is actually missing, not the whole set */
    var total = 0;
    for (i = 0; i < missing.length; i++) total += (roles[missing[i]].size || 0);
    if (!confirmed(o, { name: name, format: "ocr", variant: variant, dest: dir,
                        files: missing, bytes: total, size: sizeStr(total),
                        repo: x.repo }))
        return null;

    /* sha256 comes from the repo tree (the catalog carries sizes, not hashes);
     * one tree per distinct repo, fetched once. */
    var trees = {}, sha = {};
    function shaFor(repo, revision, path) {
        var key = repo + "@" + revision;
        if (!trees[key]) {
            trees[key] = {};
            var t = repoTree(repo, revision, o.token) || [];
            for (var j = 0; j < t.length; j++)
                if (t[j].sha256) trees[key][t[j].path] = t[j].sha256;
        }
        return trees[key][path];
    }

    var fetchList = o.force ? names : missing;
    for (i = 0; i < fetchList.length; i++) {
        var role = roles[fetchList[i]];
        var repo = role.repo || x.repo;
        var rrev = role.revision || (role.repo ? "main" : rev);
        var dst  = dir + "/" + localName(fetchList[i]);
        /* PaddleOCR keeps the character list inside inference.yml; ocr.init
         * wants a plain one-per-line file, so derive it here rather than make
         * every caller parse YAML.  The yml lands under a temporary name: if
         * extraction fails, the dictionary path stays ABSENT and the next call
         * re-fetches, rather than leaving YAML parked where the dictionary
         * should be, which nothing would ever retry. */
        var tmp = role.dictFromYml ? dst + ".yml" : dst;
        fetchFile(resolveUrl(repo, rrev, role.file), tmp, {
            size: role.size, sha256: shaFor(repo, rrev, role.file),
            progress: o.progress, token: o.token, force: o.force,
            label: name + "/" + localName(fetchList[i])
        });
        if (role.dictFromYml) {
            var y = u.readFile(tmp, true), lines = y.split("\n"), keep = [], on = false, c, k;
            for (k = 0; k < lines.length; k++) {
                if (/^\s*character_dict:\s*$/.test(lines[k])) { on = true; continue; }
                if (!on) continue;
                if (!/^\s*-\s/.test(lines[k])) break;
                c = lines[k].replace(/^\s*-\s/, "");
                if (c.length >= 2 && c.charAt(0) === c.charAt(c.length - 1) &&
                    (c.charAt(0) === "'" || c.charAt(0) === '"'))
                    c = c.substring(1, c.length - 1);
                keep.push(c);
            }
            if (!keep.length) {
                try { u.rmFile(tmp); } catch (e) {}
                throwErr("%s: no character_dict in %s", name, role.file);
            }
            u.writeFile(dst, keep.join("\n") + "\n");
            try { u.rmFile(tmp); } catch (e) {}
        }
    }

    writeProv();
    return out;
}

/* ------------------------------------------------------------------ *
 * resolution: catalog / repo-id / live search
 * ------------------------------------------------------------------ */
function entryFromRepo(repo, o, tree) {
    /* build a catalog-shaped entry by inspecting one explicit repo.
     * `tree` optional: a caller that already fetched the repo tree
     * (discover) passes it in to avoid a second API call. */
    if (!tree) tree = repoTree(repo, o.revision, o.token);
    if (!tree) throwErr("cannot list %s (typo? gated? set HF_TOKEN)", repo);
    var entry = { category: o.category || "embed" };
    if (!o.category) {
        /* no caller-pinned category: take it from the repo's pipeline tag
         * (a gen model must land in gen/, not the embed/ default) */
        var pcat = categoryFromApi(apiGet("/api/models/" + encodeURI(repo), o.token));
        if (pcat) entry.category = pcat;
    }
    var quants = {}, model = null, groups = {};
    /* the optional UD- prefix (Unsloth Dynamic) is part of the key: a UD
     * build uses a different quantisation strategy than the plain build of
     * the same name, and repos may ship both */
    var QUANT_RE = /(?:^|[-_.])((?:ud-)?(?:i?q[0-9][a-z0-9_]*|mxfp[0-9][a-z0-9_]*|f16|f32|bf16|fp16|fp32))\.gguf$/i;
    var SPLIT_RE = /-(\d{5})-of-(\d{5})\.gguf$/i;
    for (var i = 0; i < tree.length; i++) {
        var p = tree[i].path;
        /* mmproj = vision projectors; eagle = EAGLE speculative-decoding
         * draft heads (ggml-org ships them beside the real model, and a
         * draft's Q8_0 would win the default-quant pick); imatrix with no
         * quant token = bare importance-matrix data, not a model (a
         * "*-q4_k-imatrix.gguf" IS a model and is kept) */
        if (/\.gguf$/i.test(p) && !/mmproj/i.test(p) && !/(^|\/)eagle\d*-/i.test(p) &&
            !(/imatrix/i.test(p) && !/[-_.](i?q[0-9]|f16|f32|bf16|fp16|fp32|mxfp[0-9])/i.test(p))) {
            var sm = SPLIT_RE.exec(p);
            if (sm) {
                /* a quant split into parts (per-file upload limits): group
                 * under the reassembled name; big repos ship every larger
                 * quant this way */
                var gk = p.replace(SPLIT_RE, ".gguf");
                var grp = groups[gk] = groups[gk] || { parts: [], size: 0, total: parseInt(sm[2], 10) };
                grp.parts.push({ file: p, size: tree[i].size });
                grp.size += tree[i].size;
            } else {
                var m = QUANT_RE.exec(p);
                quants[m ? m[1].toUpperCase() : "DEFAULT"] = { file: p, size: tree[i].size };
            }
        }
        if (p === "onnx/model.onnx") model = p;
        if (!model && /^[^\/]+\.onnx$/.test(p)) model = p;
    }
    /* fold complete split groups in as one quant each: file = the first
     * part (what llama.cpp opens; it finds the siblings next to it),
     * size = the sum, parts = the full download list for getGguf */
    for (var gk2 in groups) {
        var g2 = groups[gk2];
        if (g2.parts.length !== g2.total) continue;         /* incomplete upload */
        g2.parts.sort(function (a, b) { return a.file < b.file ? -1 : 1; });
        var m2 = QUANT_RE.exec(gk2);
        quants[m2 ? m2[1].toUpperCase() : "DEFAULT"] =
            { file: g2.parts[0].file, size: g2.size, parts: g2.parts };
    }
    if (Object.keys(quants).length) entry.gguf = { repo: repo, revision: o.revision, quants: quants };
    if (model) entry.onnx = { repo: repo, revision: o.revision, model: model };
    if (!entry.gguf && !entry.onnx) throwErr("%s has no .gguf or .onnx files", repo);
    if (entry.category === "embed") {
        var p = fetchPrompts(repo, o.token);
        if (p) entry.prompts = p;
    }
    return entry;
}

function searchRank(alias, cands) {
    var fam = famWord(alias);
    var conv = CONVERTER_ORGS.map(function (c) { return c.toLowerCase(); });
    function tier(id) {
        var owner = id.split("/")[0].toLowerCase();
        if (owner.indexOf(fam) !== -1) return 0;
        var ci = conv.indexOf(owner);
        return ci >= 0 ? 1 + ci : 1000;
    }
    return cands.slice().sort(function (a, b) {
        var ta = tier(a.id), tb = tier(b.id);
        if (ta !== tb) return ta - tb;
        return (b.downloads || 0) - (a.downloads || 0);
    });
}

function discover(name, o) {
    /* live HF resolution for a name the catalog doesn't know (tier 3).
     * gguf: exact-name gguf repos, family/converter-ranked, adequate quants
     * preferred.  onnx: the original repo when it ships onnx, else the
     * onnx-community / Xenova mirrors. */
    var entry = { category: o.category || "embed", discovered: true };
    var quiet = o.progress === false;

    /* category from a search item's pipeline tag, unless caller-pinned.
     * Carried in o so the entryFromRepo calls below inherit it (and skip
     * their own lookup); a gen model must land in gen/, not embed/. */
    function noteCat(info) {
        if (o.category) return;
        var c = categoryFromApi(info);
        if (c) { entry.category = c; o = Object.assign({}, o, { category: c }); }
    }

    /* -- gguf -- */
    var j = apiGet("/api/models?search=" + encodeURIComponent(name) +
                   "&filter=gguf&sort=downloads&direction=-1&limit=10", o.token);
    var cands = [];
    if (Array.isArray(j))
        for (var i = 0; i < j.length; i++)
            if (compact(normName(j[i].id)) === compact(name)) cands.push(j[i]);
    for (i = 0; i < cands.length && !o.category; i++) noteCat(cands[i]);
    var ranked = searchRank(name, cands), viable = [];
    for (i = 0; i < ranked.length && viable.length < 3; i++) {
        var tree = repoTree(ranked[i].id, null, o.token);
        if (!tree) continue;
        var e = null;
        try { e = entryFromRepo(ranked[i].id, o, tree); } catch (err) { continue; }
        if (e.gguf) viable.push({ id: ranked[i].id, gguf: e.gguf,
                                  nq: Object.keys(e.gguf.quants).length });
    }
    viable.sort(function (a, b) {
        var aa = a.nq >= 4 ? 0 : 1, ba = b.nq >= 4 ? 0 : 1;
        return aa !== ba ? aa - ba : b.nq - a.nq;
    });
    if (viable.length) entry.gguf = viable[0].gguf;

    /* -- onnx: original repo, else the mirror orgs -- */
    j = apiGet("/api/models?search=" + encodeURIComponent(name) +
               "&sort=downloads&direction=-1&limit=10", o.token);
    var orig = null;
    if (Array.isArray(j))
        for (i = 0; i < j.length; i++)
            if (compact(normName(j[i].id)) === compact(name)) { orig = j[i].id; noteCat(j[i]); break; }
    var tries = [];
    if (orig) tries.push(orig, "onnx-community/" + orig.split("/").pop() + "-ONNX",
                         "onnx-community/" + orig.split("/").pop(),
                         "Xenova/" + orig.split("/").pop());
    for (i = 0; i < tries.length && !entry.onnx; i++) {
        try {
            var e2 = entryFromRepo(tries[i], o);
            if (e2.onnx) entry.onnx = e2.onnx;
        } catch (err) {}
    }

    if (!entry.gguf && !entry.onnx)
        throwErr("could not resolve '%s' on HuggingFace -- try an explicit org/repo or URL", name);

    /* retrieval prompts: the original repo publishes them when anyone does;
     * fall back to the repos we resolved.  (Remembered with the entry in
     * .resolved.json, so this fetch happens once per name.) */
    if (entry.category === "embed") {
        var pRepos = [], pi;
        if (orig) pRepos.push(orig);
        if (entry.onnx && pRepos.indexOf(entry.onnx.repo) === -1) pRepos.push(entry.onnx.repo);
        if (entry.gguf && pRepos.indexOf(entry.gguf.repo) === -1) pRepos.push(entry.gguf.repo);
        for (pi = 0; pi < pRepos.length && !entry.prompts; pi++) {
            var pp = fetchPrompts(pRepos[pi], o.token);
            if (pp) entry.prompts = pp;
        }
    }
    if (!quiet) {
        var fh = (o.progress === undefined || o.progress === true) ? STDOUT
               : (typeof o.progress === "function" ? STDOUT : o.progress);
        u.fprintf(fh, "  resolved '%s': %s%s\n", name,
                  entry.onnx ? "onnx=" + entry.onnx.repo + " " : "",
                  entry.gguf ? "gguf=" + entry.gguf.repo : "");
    }
    return entry;
}

/* ------------------------------------------------------------------ *
 * public API
 * ------------------------------------------------------------------ */
function urlGet(theUrl, o) {
    o = o || {};
    var base = theUrl.split("?")[0].split("/").pop() || "download";
    var dest = o.dest ||
        (modelsDir() + "/" + (o.category || "other") + "/" + base);
    return fetchFile(theUrl, dest, {
        progress: o.progress, token: o.token, force: o.force,
        sha256: o.sha256, size: o.size
    });
}

function get(name, o) {
    if (typeof name !== "string" || !name.length)
        throwErr("get: model name or url required");
    o = o ? Object.assign({}, o) : {};
    if (o.token === undefined) o.token = u.getenv("HF_TOKEN") || undefined;

    /* full URL: raw transport */
    if (/^https?:\/\//i.test(name)) return urlGet(name, o);

    /* name:quant shorthand (quant implies gguf) */
    var m = /^(.*):([A-Za-z0-9_.]+)$/.exec(name);
    if (m && !/^https?$/i.test(m[1])) {
        name = m[1];
        o.quant = o.quant || m[2];
    }
    if (o.quant && !o.format) o.format = "gguf";

    var entry, alias;
    if (name.indexOf("/") !== -1) {
        /* explicit org/repo */
        alias = normName(name);
        entry = entryFromRepo(name, o);
    } else {
        alias = name.toLowerCase();
        entry = CATALOG[alias];
        if (!entry) {
            var res = loadResolved();
            entry = res[alias];
            if (!entry) {
                entry = discover(alias, o);
                res[alias] = entry;
                saveResolved(res);
            }
        }
    }

    /* an ocr entry is a role-named SET of files, and no other format can serve
     * it -- so it selects itself rather than waiting to be asked for */
    var format = o.format ||
        (entry.ocr ? "ocr"
                   : (entry.onnx && (entry.category === "embed" || entry.category === "rerank") ? "onnx"
                     : (entry.gguf ? "gguf" : "onnx")));
    if (format === "ocr")  return getOcr(alias, entry, o);
    return format === "gguf" ? getGguf(alias, entry, o)
                             : getOnnx(alias, entry, o);
}

function list() {
    var out = {};
    for (var k in CATALOG) {
        var e = CATALOG[k], f = [];
        if (e.onnx) f.push("onnx");
        if (e.gguf) f.push("gguf");
        if (e.ocr)  f.push("ocr");       /* det+rec+cls+dict set */
        (out[e.category] = out[e.category] || []).push(k + " [" + f.join("+") + "]");
    }
    return out;
}

/* What of a catalog model is already on disk (under ~/.rampart/models/), or null
 * if nothing.  onnx records its precision in .source.json; a gguf is identified
 * by the quant embedded in its downloaded filename. */
function installedVariants(name, entry) {
    var cat = entry.category || "embed", parts = [];
    if (entry.onnx) {
        var odir = modelsDir() + "/" + cat + "/" + name, src = odir + "/.source.json";
        if (isFile(src)) {
            var prec = null;
            try { prec = JSON.parse(u.readFile(src, true)).precision; } catch (e) {}
            parts.push("onnx" + (prec ? " " + prec : ""));
        } else if (isFile(odir + "/onnx/model.onnx") || isFile(odir + "/model.onnx")) {
            parts.push("onnx");
        }
    }
    if (entry.gguf) {
        var q, found = [];
        for (q in entry.gguf.quants) {
            var qq = entry.gguf.quants[q], fl = qq.parts || [qq], all = true;
            for (var fi = 0; fi < fl.length; fi++)
                if (!isFile(modelsDir() + "/" + cat + "/" + fl[fi].file.split("/").pop())) { all = false; break; }
            if (all) found.push(q);
        }
        if (found.length) parts.push("gguf " + found.join("/"));
    }
    if (entry.ocr) {
        /* an ocr set counts as installed per VARIANT: shared cls/dict plus that
         * variant's det+rec all on disk */
        var odir2 = modelsDir() + "/" + cat + "/" + name, vfound = [], v, rr;
        for (v in entry.ocr.variants) {
            var all2 = true;
            for (rr in entry.ocr.common)
                if (!isFile(odir2 + "/" + entry.ocr.common[rr].file)) { all2 = false; break; }
            if (all2) for (rr in entry.ocr.variants[v])
                if (!isFile(odir2 + "/" + entry.ocr.variants[v][rr].file)) { all2 = false; break; }
            if (all2) vfound.push(v);
        }
        if (vfound.length) parts.push("ocr " + vfound.join("/"));
    }
    return parts.length ? parts.join(", ") : null;
}

/* short human phrase for an entry's license: entry.license (an OVERRIDES
 * pin, authoritative for the model) wins over the repo tags */
var LIC_PHRASE = {
    "apache-2.0": "Apache", "mit": "MIT", "gemma": "Gemma",
    "nvidia-open-model-license": "NOML", "nvidia-nemotron-open-model-license": "NOML",
    "lfm1.0": "LFM", "openmdw-1.1": "OpenMDW",
    "cc-by-4.0": "CC-BY", "cc-by-sa-4.0": "CC-BY-SA", "cc-by-nc-4.0": "CC-BY-NC"
};
function licPhrase(entry) {
    var l = entry.license ||
            (entry.gguf && entry.gguf.license) || (entry.onnx && entry.onnx.license) ||
            (entry.ocr && entry.ocr.license);
    if (!l) return "";
    if (LIC_PHRASE[l]) return LIC_PHRASE[l];
    if (/^bsd/i.test(l)) return "BSD";
    if (/^llama/i.test(l)) return "Llama";
    if (/^cc-by-nc/i.test(l)) return "CC-BY-NC";
    return l;                                  /* unmapped: show the raw slug */
}

/* resolve a name to its catalog-shaped entry without downloading */
function resolve(name, o) {
    o = o || {};
    if (name.indexOf("/") !== -1) return entryFromRepo(name, o);
    var alias = name.toLowerCase().replace(/:.*$/, "");
    return CATALOG[alias] || loadResolved()[alias] || discover(alias, o);
}

/* the gguf quants available for a name, with real file sizes, WITHOUT
 * downloading -- for memory-aware pickers ("largest quant that fits in
 * N GB").  Sizes must come from here, not from parameter-count math:
 * hybrid architectures don't shrink uniformly (nemotron-3-nano-30b is
 * 33.6 GB at Q8_0 and 33.5 GB at Q6_K).
 *   models.variants('qwen3-30b-a3b')
 *   -> [ {quant:'Q4_K_M', bytes:18e9, files:1, installed:false}, ... ]
 * sorted smallest first; [] for a model with no gguf side. */
function variants(name, o) {
    o = o || {};
    var entry = resolve(name, o);
    if (!entry.gguf) return [];
    var cat = o.category || entry.category || "embed";
    var out = [];
    for (var q in entry.gguf.quants) {
        var qi = entry.gguf.quants[q], fl = qi.parts || [qi], inst = true;
        for (var i = 0; i < fl.length; i++)
            if (!isFile(modelsDir() + "/" + cat + "/" + fl[i].file.split("/").pop())) { inst = false; break; }
        out.push({ quant: q, bytes: qi.size, files: fl.length, installed: inst });
    }
    out.sort(function (a, b) { return a.bytes - b.bytes; });
    return out;
}

/* format-explicit variants, so the call site reads as the engine it feeds:
 *   llamacpp.initEmbed( models.ggufGet('bge-m3') )        // file (default quant)
 *   llamacpp.initGen(   models.ggufGet('qwen3-4b:q4_k_m'))
 *
 * onnxGet fetches the whole model directory but RETURNS THE .onnx FILE
 * inside it (<store>/embed/bge-m3/onnx/model_fp16.onnx) -- the file, not
 * the directory, is what names the precision that was fetched.
 * rampart-onnx, though, wants the DIRECTORY: initEmbed()/initRerank()
 * discover the tokenizer, pooling and token window from the directory's
 * json files, while a bare .onnx puts them in file mode, where
 * opts.tokenizer_path is mandatory and pooling is never discovered -- so
 * a file path that does load embeds with the wrong pooling instead of
 * failing.  Fetch with onnxGet(), hand the engine the directory it came
 * from (same derivation as onnx-test.js's modelDir()):
 *   var f   = models.onnxGet('bge-m3');                    // fetch -> file
 *   var dir = f.replace(/[^\/]*$/, '').replace(/\/onnx\/?$/, '');
 *   onnx.initEmbed(dir);
 * CAVEAT: directory mode picks the model file itself -- onnx/model.onnx,
 * else model.onnx, else the first *.onnx -- and nothing overrides that.
 * So a directory holding several precisions (fp16 fetched once, q4
 * fetched later) may not load the one this call asked for; keep one
 * precision per alias, or point initEmbed at a dedicated {dest:} dir.
 */
function ggufGet(name, o) { o = o ? Object.assign({}, o) : {}; o.format = "gguf"; return get(name, o); }
function onnxGet(name, o) { o = o ? Object.assign({}, o) : {}; o.format = "onnx"; return get(name, o); }
/* NB: returns an OBJECT of role paths ({dir,det,rec,cls,dict,variant}), not a
 * single path -- an OCR model is a set.  Feed it straight to ocr.init(). */
function ocrGet(name, o)  { o = o ? Object.assign({}, o) : {}; o.format = "ocr";  return get(name, o); }

/* Module, or command line?  (standard rampart idiom: module.exports is set
 * when require()'d, falsy when run directly as the entry script.) */
if (module && module.exports) {
    module.exports = {
        get: get,
        pull: get,             /* alias */
        ggufGet: ggufGet,
        onnxGet: onnxGet,
        ocrGet: ocrGet,
        url: urlGet,
        resolve: resolve,
        variants: variants,
        list: list,
        catalog: CATALOG,
        catalogInfo: catalogInfo,     /* where the catalog came from, how current */
        updateCatalog: updateCatalog, /* force a refresh from the catalog URL */
        modelsDir: modelsDir()  /* snapshot taken at require(); the paths this
                                   module builds internally resolve per call,
                                   so they follow a later $HOME change */
    };
} else {
    /* ------------------------------ CLI ------------------------------ */
    /* rampart flags (-t, -g, ...) sit before the script path in argv, so a
     * fixed slice(2) would eat the script path as the model name.  Our
     * args start after the first non-flag element past the binary. */
    var argi = 1;
    while (argi < process.argv.length && process.argv[argi][0] === "-") argi++;
    var argv = process.argv.slice(argi + 1);
    if (!argv.length || argv[0] === "--help" || argv[0] === "-h") {
        u.printf("usage: rampart rampart-models.js <name[:quant] | org/repo[:quant] | url> [gguf|onnx] [quant|precision]\n" +
                 "       (onnx precision: fp16 (default) | fp32 | int8 | q4)\n" +
                 "       rampart rampart-models.js --list\n" +
                 "       rampart rampart-models.js --update   (refresh the model catalog)\n");
        process.exit(argv.length ? 0 : 1);
    }
    if (argv[0] === "--update") {
        var ci = updateCatalog({ remote: argv[1] === "--remote" });
        u.printf("catalog: %d models (%d built in, %d fetched from %s)\n",
                 ci.models, ci.builtin, ci.entries, ci.source);
        u.printf("  url:       %s\n", ci.url);
        if (ci.generated) u.printf("  generated: %s\n", ci.generated);
        if (ci.fetched)   u.printf("  fetched:   %s\n", u.dateFmt("%Y-%m-%d %H:%M:%S", ci.fetched));
        u.printf("  cache:     %s\n", ci.cache);
        if (ci.source === "local")
            u.printf("  note:      a catalog beside the script is in use; --update --remote to bypass it\n");
        if (ci.error) u.printf("  note:      %s\n", ci.error);
        /* built-ins alone are a working catalog; only a failed refresh is
         * worth a non-zero exit */
        process.exit(ci.error && !ci.entries ? 1 : 0);
    }
    if (argv[0] === "--list") {
        /* group catalog by category; annotate installed models + colorize
         * (%a: color only on a color terminal, plain when piped). */
        var groups = {};
        for (var k in CATALOG)
            (groups[CATALOG[k].category] = groups[CATALOG[k].category] || []).push(k);
        var cats = Object.keys(groups).sort();
        for (var ci = 0; ci < cats.length; ci++) {
            u.printf("%as\n", "yellow", cats[ci] + ":");
            groups[cats[ci]].sort().forEach(function (name) {
                var e = CATALOG[name];
                var ff = [];
                if (e.onnx) ff.push("onnx");
                if (e.gguf) ff.push("gguf");
                if (e.ocr)  ff.push("ocr");
                var fmt = "[" + ff.join("+") + "]";
                u.printf("  %-38s %as %as", name, "gray", u.sprintf("%-11s", fmt),
                         "cyan", u.sprintf("%-8s", licPhrase(e)));
                var inst = installedVariants(name, e);
                if (inst) u.printf(" %as", "green", "[installed (" + inst + ")]");
                u.printf("\n");
            });
        }
        process.exit(0);
    }
    var cliOpts = {};
    if (argv[1] === "gguf" || argv[1] === "onnx") cliOpts.format = argv[1];
    if (argv[2]) { if (cliOpts.format === "onnx") cliOpts.precision = argv[2]; else cliOpts.quant = argv[2]; }
    try {
        var p = get(argv[0], cliOpts);
        u.printf("%s\n", p);
    } catch (e) {
        u.fprintf(STDERR, "%s\n", e.message);
        process.exit(1);
    }
}
