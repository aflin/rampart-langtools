/* gen-model-list.js -- DISCOVER models on HuggingFace and generate
 * rampart-models-list.js (the catalog rampart-models.js uses to resolve short
 * names like 'bge-m3' or 'qwen3-4b:q4_k_m' to canonical repos/files).
 *
 * Discovery-driven: the catalog is built by sweeping the HF API --
 *   embed  : pipeline_tag=sentence-similarity (+ feature-extraction with
 *            library=sentence-transformers), by downloads
 *   rerank : pipeline_tag=text-ranking, by downloads
 *   gen    : pipeline_tag=text-generation&filter=gguf, by downloads,
 *            deduped to one canonical GGUF repo per base model
 * For every embed/rerank model BOTH formats are resolved:
 *   onnx : the original repo when it ships onnx/, else the
 *          onnx-community / Xenova mirror orgs
 *   gguf : the original repo when it ships .gguf, else a downloads-ranked
 *          search preferring the model family's own org, then established
 *          converter orgs
 * gen models are gguf-only (rampart runs generation through llama.cpp; there
 * is no ONNX generation engine in rampart-onnx).
 *
 * The same ranking heuristics power rampart-models.js's live (tier-3)
 * resolution at runtime -- this generator is also their test bed.
 *
 * OVERRIDES is the small human escape hatch: pin a repo discovery ranks
 * wrong, add a model discovery misses, or 'skip' junk.  To update the
 * catalog: adjust CONFIG/OVERRIDES if needed, then
 *     rampart gen-model-list.js
 * and review the git diff (the run report shows adds/removes/changes vs the
 * existing catalog, the chosen repo per model, and the runners-up).
 *
 * Embed models also get a "prompts" field -- the retrieval prefixes an
 * asymmetric model expects on queries/documents.  Discovered from the repo's
 * config_sentence_transformers.json "prompts" dict when published (snowflake,
 * qwen3); supplied via OVERRIDES[alias].prompts for models that document
 * their prefixes only in the README (nomic, bge, e5).  rampart-models.js
 * writes them as a .prompts.json sidecar next to downloaded model files.
 *     rampart gen-model-list.js --prompts
 * refreshes ONLY the prompts on the existing catalog (no discovery sweep,
 * no repo/revision churn).
 *
 * Only URL patterns that huggingface_hub itself hardcodes are used (api/
 * models, resolve/) -- never CDN URLs.  HF_TOKEN honored for gated repos;
 * HF_ENDPOINT overrides the host.
 */
rampart.globalize(rampart.utils);
var curl = require("rampart-curl");

var HF = getenv("HF_ENDPOINT") || "https://huggingface.co";
var TOKEN = getenv("HF_TOKEN");
var OUT = process.scriptPath + "/rampart-models.js";   /* catalog spliced between markers */

var CONFIG = {
    embedTop:  20,     /* how many models per category to keep, by downloads */
    rerankTop: 10,
    genTop:    30,
    scanLimit: 250     /* how many search results to consider per query */
};

/* Established orgs, in preference order, for ranking format-companion repos.
 * A repo whose owner matches the model's own family (qwen -> Qwen) always
 * outranks these. */
var CONVERTER_ORGS = [
    "ggml-org", "unsloth", "bartowski", "lmstudio-community", "gpustack",
    "second-state", "onnx-community", "Xenova", "CompendiumLabs", "cstr",
    "leliuga", "mradermacher", "TheBloke"
];

/* ---- human overrides: name -> 'skip' | { category?, onnx?, gguf? }
 * (repo string pins it; null drops that format; absent = discovered) ---- */
var OVERRIDES = {
    /* --- taxonomy gaps: models HF's pipeline tags don't surface --- */
    "gte-multilingual-base":   { category: "embed",  onnx: "onnx-community/gte-multilingual-base", gguf: null,
                                 license: "apache-2.0" },   /* mirror repo untagged */
    "snowflake-arctic-embed-m-v1.5": { category: "embed", onnx: "Snowflake/snowflake-arctic-embed-m-v1.5", gguf: "Snowflake/snowflake-arctic-embed-m-v1.5" },
    "bge-reranker-base":       { category: "rerank", onnx: "BAAI/bge-reranker-base",                 gguf: "cstr/bge-reranker-base-GGUF" },
    "bge-reranker-v2-m3":      { category: "rerank", onnx: "onnx-community/bge-reranker-v2-m3-ONNX", gguf: "gpustack/bge-reranker-v2-m3-GGUF" },
    "mxbai-rerank-base-v1":    { category: "rerank", onnx: "mixedbread-ai/mxbai-rerank-base-v1",     gguf: "cstr/mxbai-rerank-base-v1-GGUF" },
    /* qwen3-reranker: a causal yes/no judge scored through an instruct
     * chat template (rampart-llamacpp applies the template when
     * general.architecture is "qwen3").  Only ggml-org's 0.6B gguf is a
     * real reranker conversion (rank head + tokenizer.chat_template.rerank
     * in the metadata -- grep the header for "qwen3.pooling_type");
     * EVERY public 4B/8B gguf (mradermacher, QuantFactory, Mungert,
     * DevQuasar, dean2155, camelliah, greenwich157, aotsukiqx -- surveyed
     * 2026-07-17) is a plain causal-LM conversion whose rank pooling
     * scores a constant 0.5 for all inputs.  onnx is nulled too:
     * rampart-onnx's rerank has no template support, so an onnx-community
     * mirror would be equally broken.  The nulls keep the 4b/8b entries
     * format-less so get() fails cleanly instead of serving a broken
     * model; re-pin when reranker-aware conversions appear. */
    "qwen3-reranker-0.6b":     { gguf: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", onnx: null },
    "qwen3-reranker-4b":       { gguf: null, onnx: null },
    "qwen3-reranker-8b":       { gguf: null, onnx: null },
    /* --- 2026-08 curated additions: gen models the last sweep predates,
     * gguf-only, from ungated repos.  License policy (2026-08-10, Aaron):
     * any license is acceptable as long as the files download without a
     * login; --list shows a short license phrase so the license is
     * visible at a glance.  Still excluded: hy3 (no trustworthy gguf
     * conversion -- only an unaffiliated mirror), laguna-s-2.1
     * (originating lab unidentifiable), abliterated/distill-merge
     * variants.  ov.license overrides the repo tag where converter repos
     * are untagged (nemotron nano-v2). --- */
    "lfm2.5-230m":          { category: "gen", gguf: "LiquidAI/LFM2.5-230M-GGUF" },
    "lfm2.5-1.2b-instruct": { category: "gen", gguf: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF" },
    "lfm2.5-1.2b-thinking": { category: "gen", gguf: "LiquidAI/LFM2.5-1.2B-Thinking-GGUF" },
    "lfm2.5-2.6b":          { category: "gen", gguf: "LiquidAI/LFM2.5-2.6B-GGUF" },
    "lfm2.5-8b-a1b":        { category: "gen", gguf: "LiquidAI/LFM2.5-8B-A1B-GGUF" },
    "nemotron-nano-9b-v2":  { category: "gen", gguf: "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF",
                              license: "nvidia-open-model-license" },
    "nemotron-nano-12b-v2": { category: "gen", gguf: "MaziyarPanahi/NVIDIA-Nemotron-Nano-12B-v2-GGUF",
                              license: "nvidia-open-model-license" },
    /* nemotron-3 license set explicitly: converter repos tag old/new
     * nvidia license names inconsistently; nvidia's canonical repos all
     * carry the (newer, apache-shaped) nemotron open model license */
    "nemotron-3-nano-4b":   { category: "gen", gguf: "lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                              license: "nvidia-nemotron-open-model-license" },
    "nemotron-3-nano-30b-a3b": { category: "gen", gguf: "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
                              license: "nvidia-nemotron-open-model-license" },
    "nemotron-3-super-120b-a12b": { category: "gen", gguf: "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
                              license: "nvidia-nemotron-open-model-license" },
    /* bonsai: prism-ml is the family org but famWord("bonsai") never
     * matches it, so the sweep's owner filter drops these -- pin.
     * ternary = BitNet-class quants; not yet load-validated on b9494. */
    "bonsai-27b":           { category: "gen", gguf: "prism-ml/Bonsai-27B-gguf" },
    "ternary-bonsai-8b":    { category: "gen", gguf: "prism-ml/Ternary-Bonsai-8B-gguf" },
    "ternary-bonsai-27b":   { category: "gen", gguf: "prism-ml/Ternary-Bonsai-27B-gguf" },
    /* embeddinggemma prompts per google's model card (query prefix +
     * untitled-document prefix) */
    "embeddinggemma-300m":  { category: "embed", gguf: "unsloth/embeddinggemma-300m-GGUF", onnx: null,
                              prompts: { query: "task: search result | query: ", document: "title: none | text: " } },
    "qwen3.5-0.8b":           { category: "gen", gguf: "unsloth/Qwen3.5-0.8B-GGUF" },
    "qwen3.5-2b":             { category: "gen", gguf: "unsloth/Qwen3.5-2B-GGUF" },
    "qwen3.5-27b":            { category: "gen", gguf: "unsloth/Qwen3.5-27B-GGUF" },
    "qwen3.5-122b-a10b":      { category: "gen", gguf: "unsloth/Qwen3.5-122B-A10B-GGUF" },
    "qwen3-1.7b":             { category: "gen", gguf: "unsloth/Qwen3-1.7B-GGUF" },
    "qwen3-32b":              { category: "gen", gguf: "unsloth/Qwen3-32B-GGUF" },
    "qwen3-235b-a22b":        { category: "gen", gguf: "unsloth/Qwen3-235B-A22B-GGUF" },
    "qwen3-4b-instruct-2507": { category: "gen", gguf: "unsloth/Qwen3-4B-Instruct-2507-GGUF" },
    "glm-5.2":                { category: "gen", gguf: "unsloth/GLM-5.2-GGUF" },
    "deepseek-v4-flash":      { category: "gen", gguf: "unsloth/DeepSeek-V4-Flash-GGUF" },
    "ornith-1.0-9b":          { category: "gen", gguf: "ornith-ai/Ornith-1.0-9B-GGUF" },
    "ornith-1.0-35b":         { category: "gen", gguf: "ornith-ai/Ornith-1.0-35B-GGUF" },
    "kat-coder-v2.5":         { category: "gen", gguf: "bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF" },
    "gemma-4-12b-it":         { category: "gen", gguf: "unsloth/gemma-4-12b-it-GGUF" },
    "gemma-4-26b-a4b-it":     { category: "gen", gguf: "unsloth/gemma-4-26B-A4B-it-GGUF" },
    /* nomic v2: apache MoE embedder, first-party gguf; prompts are the
     * README-documented nomic prefixes (same as v1).  gguf-only per the
     * 2026-08 curation pass. */
    "nomic-embed-text-v2-moe": { category: "embed", gguf: "nomic-ai/nomic-embed-text-v2-moe-GGUF", onnx: null,
                                 prompts: { query: "search_query: ", document: "search_document: " } },
    /* gpt-oss ships natively in MXFP4 (the MoE expert weights); ggml-org's
     * single-file MXFP4 ggufs ARE the original release.  Discovery ranks
     * unsloth's requantized ladders first (more downloads/quants), but every
     * rung below "F16" degrades the native weights, and the 120b entry it
     * finds is F16-only.  Pin the canonical repos. */
    "gpt-oss-20b":             { category: "gen", gguf: "ggml-org/gpt-oss-20b-GGUF" },
    "gpt-oss-120b":            { category: "gen", gguf: "ggml-org/gpt-oss-120b-GGUF" },
    /* --- churn guard (2026-08): models the download-rank sweep dropped as
     * newer families took the top slots, pinned back deliberately -- they
     * anchor memory tiers in downstream pickers (qwen3-30b-a3b = the 24GB
     * recommendation).  qwen2.5-72b stays retired (superseded at that
     * tier by qwen3-32b/qwen3.5). --- */
    "qwen3-30b-a3b":        { category: "gen", gguf: "unsloth/Qwen3-30B-A3B-GGUF" },
    "qwen2.5-14b-instruct": { category: "gen", gguf: "Qwen/Qwen2.5-14B-Instruct-GGUF" },
    "qwen2.5-32b-instruct": { category: "gen", gguf: "Qwen/Qwen2.5-32B-Instruct-GGUF" },
    /* qwen3.6: real Qwen family (apache, ungated) -- the earlier
     * qwen3.6-27b skip covers only batiai's name-squat repo */
    "qwen3.6-35b-a3b":      { category: "gen", gguf: "unsloth/Qwen3.6-35B-A3B-GGUF" },
    /* --- classics below the popularity window --- */
    "qwen2.5-7b-instruct":     { category: "gen", gguf: "Qwen/Qwen2.5-7B-Instruct-GGUF" },
    "gemma-3-1b-it":           { category: "gen", gguf: "ggml-org/gemma-3-1b-it-GGUF" },
    "gemma-3-4b-it":           { category: "gen", gguf: "ggml-org/gemma-3-4b-it-GGUF" },
    "phi-4":                   { category: "gen", gguf: "microsoft/phi-4-gguf" },
    "phi-4-mini-instruct":     { category: "gen", gguf: "unsloth/Phi-4-mini-instruct-GGUF" },
    "mistral-7b-instruct-v0.3":{ category: "gen", gguf: "bartowski/Mistral-7B-Instruct-v0.3-GGUF" },
    "smollm2-1.7b-instruct":   { category: "gen", gguf: "bartowski/SmolLM2-1.7B-Instruct-GGUF" },
    "deepseek-r1-qwen-7b":     { category: "gen", gguf: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                                 license: "mit" },          /* conversion repo untagged */
    "granite-3.3-8b-instruct": { category: "gen", gguf: "ibm-granite/granite-3.3-8b-instruct-GGUF" },
    "tinyllama-1.1b-chat":     { category: "gen", gguf: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" },
    /* --- CLIP (image+text shared space, for rampart-clip): the monatis/clip.cpp
     * two-tower GGUFs.  Discovery can't surface these -- they carry no
     * sentence-similarity / text-ranking / text-generation pipeline tag -- so they
     * are pinned by hand.  gguf-only (onnx:null); dim is the shared embedding size.
     * Each repo ALSO ships _ggml-text-model- / _ggml-vision-model- single-tower
     * files; ggufQuants() filters those out so only the two-tower _ggml-model-
     * quants are kept (rampart-clip needs both towers for cross-modal search). --- */
    "clip-vit-b-32-laion": { category: "clip", gguf: "mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K", onnx: null, dim: 512 },
    "clip-vit-b-32":       { category: "clip", gguf: "mys/ggml_clip-vit-base-patch32",           onnx: null, dim: 512 },
    "clip-vit-l-14-laion": { category: "clip", gguf: "mys/ggml_CLIP-ViT-L-14-laion2B-s32B-b82K", onnx: null, dim: 768 },
    "clip-vit-l-14":       { category: "clip", gguf: "mys/ggml_clip-vit-large-patch14",          onnx: null, dim: 768 },
    "clip-vit-h-14-laion": { category: "clip", gguf: "mys/ggml_CLIP-ViT-H-14-laion2B-s32B-b79K", onnx: null, dim: 1024 },
    /* --- junk / non-text models the sweeps surface --- */
    "w2v-bert-2.0": "skip",                    /* speech embedder, not text */
    "stories15m_moe": "skip",                  /* ggml-org test model */
    /* --- 2026-08 sweep guard: high-download entries that must NOT enter
     * the catalog.  Gguf only from unaffiliated mirrors (hy3,
     * deepseek-v4), unidentifiable provenance (laguna), name-squatting
     * merges (qwen3.6/3.8, deepwen), Claude-distill merges, abliterated
     * variants, and dataset-junk (automotive). --- */
    "hy3": "skip",                             /* tencent, apache, but only mirror ggufs */
    "deepseek-v4": "skip",                     /* only a personal-mirror gguf; flash is pinned */
    "laguna-s-2.1": "skip",
    /* discovery aliases of repos already pinned under cleaner names below */
    "nvidia-nemotron-nano-9b-v2": "skip", "nvidia-nemotron-nano-12b-v2": "skip",
    "nvidia-nemotron-3-nano-4b": "skip", "nvidia-nemotron-3-super-120b-a12b": "skip",
    "gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2": "skip",
    "gemma-4-12b-coder-fable5-composer2.5-v1": "skip",
    "minicpm5-1b-claude-opus-fable5-thinking": "skip",
    "minicpm5-1b-claude-opus-fable5-v2-thinking": "skip",
    "parable-qwen3-4b-claude-fable-5": "skip",
    "huihui-deepseek-v4-flash-0731-abliterated": "skip",
    "gemma-4-12b-heretic-abliterated": "skip",
    "qwen3.6-27b": "skip", "qwen3.8_4b_distilled": "skip", "deepwen-3.6": "skip",
    "dolphin3-cyber-8b": "skip",
    "automotive": "skip",
    /* discovery's alias for the SAME bartowski repo the kat-coder-v2.5
     * override pins -- skip the ugly duplicate */
    "kwaipilot_kat-coder-v2.5-dev": "skip",
    /* bartowski org_Model naming produces a dup alias of qwen3-0.6b */
    "qwen_qwen3-0.6b": "skip",
    /* discovery alias for the SAME repo the deepseek-r1-qwen-7b classic pins */
    "deepseek-r1-distill-qwen-7b": "skip",
    /* --- retrieval prompts documented only in the README (not in the
     * repo's config_sentence_transformers.json) -- exact strings, trailing
     * whitespace/colons significant.  Discovery still fills the rest of the
     * entry; these merge in via attachPrompts(). --- */
    "nomic-embed-text-v1.5": { prompts: { query: "search_query: ", document: "search_document: " } },
    "nomic-embed-text-v1":   { prompts: { query: "search_query: ", document: "search_document: " } },
    "bge-small-en-v1.5":     { prompts: { query: "Represent this sentence for searching relevant passages: " } },
    "bge-base-en-v1.5":      { prompts: { query: "Represent this sentence for searching relevant passages: " } },
    "bge-large-en-v1.5":     { prompts: { query: "Represent this sentence for searching relevant passages: " } },
    "bge-small-zh-v1.5":     { prompts: { query: "为这个句子生成表示以用于检索相关文章：" } },
    /* multilingual-e5 gguf pins: the discovery-ranked cstr conversions
     * predate llama.cpp's "bert model needs to define token type count"
     * requirement and no longer load; these community conversions do
     * (verified 2026-07-14).  Rerun with --pin to re-apply. */
    /* e5-small: NO known-good gguf conversion exists on HF (2026-07-15
     * survey: cstr fails llama.cpp's token-type-count check; keisuke
     * q8_0 loads but embeds everything near-identically; milimyname
     * q8_0 GGML_ABORTs -- token-type embeddings quantized; rodion-m
     * fp32 has a degenerate tokenizer -- wiki_00 retrieval 3/8 vs the
     * official onnx's 8/8).  onnx-only until someone converts it with
     * a current llama.cpp. */
    "multilingual-e5-small": { gguf: null,
                               prompts: { query: "query: ", document: "passage: " } },
    "multilingual-e5-base":  { gguf: "dinab/multilingual-e5-base-Q4_K_M-GGUF",
                               prompts: { query: "query: ", document: "passage: " } },
    "multilingual-e5-large": { gguf: "phate334/multilingual-e5-large-gguf",
                               prompts: { query: "query: ", document: "passage: " } },
    "e5-large-v2":           { prompts: { query: "query: ", document: "passage: " } },
    "mxbai-embed-large-v1":  { prompts: { query: "Represent this sentence for searching relevant passages: " } },
    /* qwen3's catalog repos are converter mirrors without the config; these
     * are the strings Qwen/Qwen3-Embedding-0.6B publishes (document is "") */
    "qwen3-embedding-0.6b":  { prompts: { query: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" } },
};

/* =================================================================== */
var failures = [], report = [];

function apiGet(path) {
    var opts = {};
    if (TOKEN) opts.headers = { "Authorization": "Bearer " + TOKEN };
    var res, backoff = [2, 8, 20];
    for (var attempt = 0; ; attempt++) {
        sleep(0.15);                       /* stay well under HF's rate limits */
        res = curl.fetch(HF + path, opts);
        if (res.status === 200) break;
        /* 429/5xx/network hiccup: back off and retry */
        if (attempt < backoff.length && (res.status === 429 || res.status >= 500 || res.status <= 0)) {
            fprintf(stderr, "  (HTTP %d on %s -- retrying in %ds)\n", res.status, path, backoff[attempt]);
            sleep(backoff[attempt]);
            continue;
        }
        return { _status: res.status };
    }
    try { return JSON.parse(res.text); } catch (e) { return { _status: -1 }; }
}

function repoMeta(repo) {
    var j = apiGet("/api/models/" + encodeURI(repo));
    if (j._status) return null;
    /* "other" is a placeholder tag; the real name lives in license_name
     * (e.g. nvidia-nemotron-open-model-license, lfm1.0) -- record that */
    var lic = (j.cardData && j.cardData.license) || null;
    if ((!lic || lic === "other") && j.cardData && j.cardData.license_name)
        lic = j.cardData.license_name;
    return {
        sha: j.sha,
        gated: j.gated === true || (typeof j.gated === "string" && j.gated !== "false"),
        license: lic,
        downloads: j.downloads || 0
    };
}

function repoTree(repo) {
    var j = apiGet("/api/models/" + encodeURI(repo) + "/tree/main?recursive=true");
    if (!Array.isArray(j)) return null;
    return j.map(function (e) {
        return { path: e.path, size: (e.lfs && e.lfs.size) || e.size || 0 };
    });
}

function repoDim(repo) {
    /* location:true is required: resolve/ URLs 307-redirect */
    var res = curl.fetch(HF + "/" + repo + "/resolve/main/config.json",
                         TOKEN ? { location: true, headers: { "Authorization": "Bearer " + TOKEN } }
                               : { location: true });
    if (res.status !== 200) return null;
    try {
        var c = JSON.parse(res.text);
        return c.hidden_size || c.d_model || null;
    } catch (e) { return null; }
}

/* Retrieval prompt prefixes from a repo's config_sentence_transformers.json
 * "prompts" dict.  Keys normalized to the sidecar contract: query, document
 * ("passage" -> document), documentWithTitle (rampart extension: a document
 * template with a {title} slot).  Empty-string prompts are dropped (no-op);
 * task-name keys we don't understand (jina's "retrieval.query" etc.) are
 * ignored rather than guessed at. */
var promptsCache = {};
function repoPrompts(repo) {
    if (repo in promptsCache) return promptsCache[repo];
    var res = curl.fetch(HF + "/" + repo + "/resolve/main/config_sentence_transformers.json",
                         TOKEN ? { location: true, headers: { "Authorization": "Bearer " + TOKEN } }
                               : { location: true });
    var out = null;
    if (res.status === 200) {
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
    }
    promptsCache[repo] = out;
    return out;
}

/* set entry.prompts for an embed model: OVERRIDES wins, else the published
 * config (original/onnx repo first, then the gguf repo).  Returns the source
 * ('override' | 'config') or null. */
function attachPrompts(alias, entry) {
    if (entry.category !== "embed") return null;
    var ov = OVERRIDES[alias];
    var p = (ov && ov !== "skip" && ov.prompts) || null;
    var src = p ? "override" : null;
    if (!p && entry.onnx && entry.onnx.repo) { p = repoPrompts(entry.onnx.repo); if (p) src = "config"; }
    if (!p && entry.gguf && entry.gguf.repo) { p = repoPrompts(entry.gguf.repo); if (p) src = "config"; }
    if (p) entry.prompts = p; else delete entry.prompts;
    return src;
}

/* strip org, -GGUF/-ONNX/quant suffixes; lowercase -> the catalog alias */
function normName(repoId) {
    var b = repoId.split("/").pop();
    /* strip -GGUF/-ONNX/quant/"Embedding" suffixes in any order until stable
     * (handles names like foo-gguf-q8_0 and foo-Q4_K_M-GGUF alike) */
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

/* the model "family" word, for owner-match ranking (qwen3-4b -> qwen) */
function famWord(alias) { return (alias.match(/^[a-z]+/) || [alias])[0]; }

function compact(s) { return s.replace(/[-_.]/g, ""); }

/* rank candidate repos for a given alias: family-owner first, then converter
 * orgs by list order, then anything, downloads desc within a tier */
function rankCandidates(alias, cands) {
    var fam = famWord(alias);
    function tier(id) {
        var owner = id.split("/")[0].toLowerCase();
        if (owner.indexOf(fam) !== -1) return 0;
        var ci = CONVERTER_ORGS.map(function (o) { return o.toLowerCase(); }).indexOf(owner);
        return ci >= 0 ? 1 + ci : 1000;
    }
    return cands.slice().sort(function (a, b) {
        var ta = tier(a.id), tb = tier(b.id);
        if (ta !== tb) return ta - tb;
        return (b.downloads || 0) - (a.downloads || 0);
    });
}

/* the optional UD- prefix (Unsloth Dynamic) is part of the key: a UD build
 * uses a different quantisation strategy than the plain build of the same
 * name, and repos may ship both */
var QUANT_RE = /(?:^|[-_.])((?:ud-)?(?:i?q[0-9][a-z0-9_]*|mxfp[0-9][a-z0-9_]*|f16|f32|bf16|fp16|fp32))\.gguf$/i;

var SPLIT_RE = /-(\d{5})-of-(\d{5})\.gguf$/i;

function ggufQuants(tree) {
    var quants = {}, groups = {};
    for (var i = 0; i < tree.length; i++) {
        var p = tree[i].path;
        if (!/\.gguf$/i.test(p)) continue;
        if (/mmproj/i.test(p)) continue;                    /* vision projectors */
        if (/(^|\/)eagle\d*-/i.test(p)) continue;           /* EAGLE draft heads (a
             draft's Q8_0 would win the default-quant pick over e.g. MXFP4) */
        if (/imatrix/i.test(p) &&
            !/[-_.](i?q[0-9]|f16|f32|bf16|fp16|fp32|mxfp[0-9])/i.test(p)) continue;
             /* bare importance-matrix data (imatrix_unsloth.gguf, *-imatrix.gguf
              * with no quant token); "*-q4_k-imatrix.gguf" IS a model -- kept */
        if (/[-_.](text|vision)-model[-_.]/i.test(p)) continue;  /* CLIP single-tower
             variants: keep only the two-tower _ggml-model- files (else the smaller
             text-only file would win a quant slot).  No-op for non-CLIP repos. */
        var sm = SPLIT_RE.exec(p);
        if (sm) {
            /* a quant split into parts (per-file upload limits): group under
             * the reassembled name; big repos ship every larger quant this way */
            var gk = p.replace(SPLIT_RE, ".gguf");
            var g = groups[gk] = groups[gk] || { parts: [], size: 0, total: parseInt(sm[2], 10) };
            g.parts.push({ file: p, size: tree[i].size });
            g.size += tree[i].size;
            continue;
        }
        var m = QUANT_RE.exec(p);
        var q = m ? m[1].toUpperCase() : "DEFAULT";
        if (!quants[q] || tree[i].size < quants[q].size)
            quants[q] = { file: p, size: tree[i].size };
    }
    /* fold complete split groups in as one quant each: file = first part
     * (what llama.cpp opens), size = sum, parts = the full download list.
     * Same smaller-file-wins rule on a key collision. */
    for (var gk2 in groups) {
        var g2 = groups[gk2];
        if (g2.parts.length !== g2.total) continue;         /* incomplete upload */
        g2.parts.sort(function (a, b) { return a.file < b.file ? -1 : 1; });
        var m2 = QUANT_RE.exec(gk2);
        var q2 = m2 ? m2[1].toUpperCase() : "DEFAULT";
        if (!quants[q2] || g2.size < quants[q2].size)
            quants[q2] = { file: g2.parts[0].file, size: g2.size, parts: g2.parts };
    }
    return Object.keys(quants).length ? quants : null;
}

function onnxModelIn(tree) {
    var root = null;
    for (var i = 0; i < tree.length; i++) {
        if (tree[i].path === "onnx/model.onnx") return "onnx/model.onnx";
        if (/^[^\/]+\.onnx$/.test(tree[i].path) && !root) root = tree[i].path;
    }
    return root;
}

/* build the onnx side for a model: original repo, else the mirror orgs */
function resolveOnnx(alias, origRepo, origTree) {
    var tries = [];
    if (origRepo) tries.push({ repo: origRepo, tree: origTree });
    var base = origRepo ? origRepo.split("/").pop() : alias;
    tries.push({ repo: "onnx-community/" + base + "-ONNX" });
    tries.push({ repo: "onnx-community/" + base });
    tries.push({ repo: "Xenova/" + base });
    for (var i = 0; i < tries.length; i++) {
        var t = tries[i];
        var tree = t.tree || repoTree(t.repo);
        if (!tree) continue;
        var model = onnxModelIn(tree);
        if (!model) continue;
        var meta = repoMeta(t.repo);
        if (!meta) continue;
        var out = { repo: t.repo, revision: meta.sha, model: model,
                    license: meta.license, downloads: meta.downloads };
        if (meta.gated) out.gated = true;
        var dim = repoDim(t.repo);
        if (dim) out.dim = dim;
        return out;
    }
    return null;
}

/* build the gguf side: original repo if it ships gguf, else ranked search */
function resolveGguf(alias, origRepo, origTree, wantReport) {
    var cands = [];
    if (origTree && ggufQuants(origTree))
        cands.push({ id: origRepo, downloads: 1e12 });   /* first-party wins */
    else {
        var j = apiGet("/api/models?search=" + encodeURIComponent(alias) +
                       "&filter=gguf&sort=downloads&direction=-1&limit=10");
        if (Array.isArray(j))
            for (var i = 0; i < j.length; i++) {
                /* candidate must actually be this model, not a lookalike:
                 * exact match ignoring punctuation (L-6 == l6), so v1 never
                 * grabs a v1.5 repo */
                if (compact(normName(j[i].id)) === compact(alias))
                    cands.push(j[i]);
            }
    }
    if (!cands.length) return null;
    var best = pickBestGguf(alias, rankCandidates(alias, cands), origRepo, origTree, wantReport);
    return best;
}

/* Inspect up to the first 5 viable ranked candidates and pick by
 * (rank tier, quant coverage desc, downloads desc) -- so an official
 * single-quant repo doesn't beat a converter repo carrying every quant. */
function pickBestGguf(alias, ranked, origRepo, origTree, wantReport) {
    var fam = famWord(alias);
    function tierOf(id) {
        var owner = id.split("/")[0].toLowerCase();
        if (owner.indexOf(fam) !== -1) return 0;
        var ci = CONVERTER_ORGS.map(function (o) { return o.toLowerCase(); }).indexOf(owner);
        return ci >= 0 ? 1 + ci : 1000;
    }
    var viable = [];
    for (var i = 0; i < ranked.length && viable.length < 5; i++) {
        var tree = (ranked[i].id === origRepo) ? origTree : repoTree(ranked[i].id);
        if (!tree) continue;
        var quants = ggufQuants(tree);
        if (!quants) continue;
        var meta = repoMeta(ranked[i].id);
        if (!meta) continue;
        viable.push({ id: ranked[i].id, tier: tierOf(ranked[i].id),
                      nq: Object.keys(quants).length, quants: quants, meta: meta });
    }
    if (!viable.length) return null;
    /* adequate quant coverage (>=4) trumps tier: an official repo shipping
     * only Q8_0 shouldn't be chosen over a converter carrying every quant
     * (the catalog records ONE gguf repo per model, so coverage decides
     * whether name:q4_k_m can resolve at all) */
    viable.sort(function (a, b) {
        var aa = a.nq >= 4 ? 0 : 1, ba = b.nq >= 4 ? 0 : 1;
        if (aa !== ba) return aa - ba;
        if (a.tier !== b.tier) return a.tier - b.tier;
        if (a.nq !== b.nq) return b.nq - a.nq;
        return (b.meta.downloads || 0) - (a.meta.downloads || 0);
    });
    var w = viable[0];
    var out = { repo: w.id, revision: w.meta.sha, quants: w.quants,
                license: w.meta.license, downloads: w.meta.downloads };
    if (w.meta.gated) out.gated = true;
    if (wantReport && viable.length > 1)
        report.push(alias + " gguf: chose " + w.id + " (" + w.nq + "q)  [also: " +
            viable.slice(1).map(function (v) { return v.id + " (" + v.nq + "q)"; }).join(", ") + "]");
    return out;
}

/* ------------------------- discovery sweeps ------------------------- */
function sweep(url) {
    var j = apiGet(url);
    return Array.isArray(j) ? j : [];
}

function discoverEmbedRerank(category, queries, topN, catalog) {
    var seen = {};
    var cands = [];
    for (var q = 0; q < queries.length; q++) {
        var list = sweep(queries[q]);
        for (var i = 0; i < list.length; i++) {
            if (!seen[list[i].id]) { seen[list[i].id] = 1; cands.push(list[i]); }
        }
    }
    cands.sort(function (a, b) { return (b.downloads || 0) - (a.downloads || 0); });

    var kept = 0;
    for (i = 0; i < cands.length && kept < topN; i++) {
        var repo = cands[i].id;
        var alias = normName(repo);
        if (OVERRIDES[alias] === "skip") continue;
        if (catalog[alias]) continue;                       /* alias collision: first (most dl) wins */
        /* skip mirror-org originals; they'll be reached via resolveOnnx */
        var owner = repo.split("/")[0];
        if (owner === "onnx-community" || owner === "Xenova") continue;

        printf("%-34s ", alias); fflush(stdout);
        var tree = repoTree(repo);
        if (!tree) { printf("(tree unreadable, skipped)\n"); continue; }

        var entry = { category: category };
        var ov = OVERRIDES[alias] || {};
        var o = (ov.onnx === null) ? null
              : ov.onnx ? resolveOnnx(alias, ov.onnx, null)
              : resolveOnnx(alias, repo, tree);
        var g = (ov.gguf === null) ? null
              : ov.gguf ? resolveGguf(alias, ov.gguf, repoTree(ov.gguf), false)
              : resolveGguf(alias, repo, tree, true);
        if (o) entry.onnx = o;
        if (g) entry.gguf = g;
        if (!o && !g) {
            /* deliberately format-less (OVERRIDES nulls both sides, or
             * nulls one and the other doesn't resolve): keep a stub so
             * get() fails with a clear "no source" error instead of the
             * live-discovery fallback re-finding a known-broken repo */
            if (ov.gguf === null || ov.onnx === null) {
                catalog[alias] = entry;
                printf("(no usable source -- stub entry kept)\n");
                continue;
            }
            printf("(no onnx, no gguf -- skipped)\n"); continue;
        }
        var psrc = attachPrompts(alias, entry);
        printf("onnx:%s gguf:%s%s\n", o ? "ok" : "-", g ? Object.keys(g.quants).length + "q" : "-",
               psrc ? " prompts:" + Object.keys(entry.prompts).join("+") + "(" + psrc + ")" : "");
        catalog[alias] = entry;
        kept++;
    }
}

function discoverGen(topN, catalog) {
    var list = sweep("/api/models?pipeline_tag=text-generation&filter=gguf" +
                     "&sort=downloads&direction=-1&limit=" + CONFIG.scanLimit);
    /* group by normalized base name, rank within each group.  Only accept
     * repos owned by the model's own family org or an established converter
     * org -- downloads-sorted gen results are polluted with fine-tune slop
     * from unknown owners. */
    var convLower = CONVERTER_ORGS.map(function (o) { return o.toLowerCase(); });
    var groups = {};
    for (var i = 0; i < list.length; i++) {
        var alias = normName(list[i].id);
        if (OVERRIDES[alias] === "skip") continue;
        var owner = list[i].id.split("/")[0].toLowerCase();
        if (owner.indexOf(famWord(alias)) === -1 && convLower.indexOf(owner) === -1)
            continue;
        (groups[alias] = groups[alias] || []).push(list[i]);
    }
    /* order groups by their best downloads */
    var names = Object.keys(groups).sort(function (a, b) {
        function best(g) { var m = 0; for (var i = 0; i < g.length; i++) m = Math.max(m, g[i].downloads || 0); return m; }
        return best(groups[b]) - best(groups[a]);
    });

    var kept = 0;
    for (var n = 0; n < names.length && kept < topN; n++) {
        var alias = names[n];
        if (catalog[alias]) continue;
        printf("%-34s ", alias); fflush(stdout);
        var ov = OVERRIDES[alias] || {};
        var g = null;
        if (ov.gguf) g = resolveGguf(alias, ov.gguf, repoTree(ov.gguf), false);
        else g = pickBestGguf(alias, rankCandidates(alias, groups[alias]), null, null, true);
        if (!g) { printf("(no usable gguf -- skipped)\n"); continue; }
        catalog[alias] = { category: "gen", gguf: g };
        printf("gguf:%dq (%s)\n", Object.keys(g.quants).length, g.repo);
        kept++;
    }
}

/* ------------------------------ run ------------------------------ */
var catalog = {};
var PROMPTS_ONLY = process.argv.indexOf("--prompts") !== -1;
var PIN_IX = process.argv.indexOf("--pin");

if (PIN_IX !== -1) {
    /* re-resolve ONLY the named models (or every model with a repo pin
     * in OVERRIDES when none are named) against the existing catalog --
     * for applying new OVERRIDES pins without a discovery sweep. */
    catalog = require(OUT).catalog;
    var pins = process.argv.slice(PIN_IX + 1).filter(function (a) { return a[0] !== "-"; });
    if (!pins.length)
        pins = Object.keys(OVERRIDES).filter(function (k) {
            var ov = OVERRIDES[k];
            return ov !== "skip" && (ov.gguf || ov.onnx) && catalog[k];
        });
    printf("== pin refresh: %s ==\n", pins.join(", "));
    for (var pi = 0; pi < pins.length; pi++) {
        var pname = pins[pi];
        var pov = OVERRIDES[pname];
        var pentry = catalog[pname];
        if (!pentry || !pov || pov === "skip") {
            printf("%-34s (not in catalog / no override -- skipped)\n", pname);
            failures.push(pname + ": nothing to pin");
            continue;
        }
        printf("%-34s ", pname); fflush(stdout);
        if (pov.gguf === null) delete pentry.gguf;      /* format dropped */
        else if (pov.gguf) {
            var pg = resolveGguf(pname, pov.gguf, repoTree(pov.gguf), false);
            if (pg) pentry.gguf = pg;
            else failures.push(pname + ": pinned gguf repo unusable: " + pov.gguf);
        }
        if (pov.onnx === null) delete pentry.onnx;
        else if (pov.onnx) {
            var po = resolveOnnx(pname, pov.onnx, null);
            if (po) pentry.onnx = po;
            else failures.push(pname + ": pinned onnx repo unusable: " + pov.onnx);
        }
        if (pov.dim) pentry.dim = pov.dim;    /* static dim for a gguf-only model (CLIP) */
        if (pov.license) pentry.license = pov.license;  /* model license when the
                                                           conversion repo is untagged */
        attachPrompts(pname, pentry);
        printf("gguf:%s onnx:%s\n",
               pentry.gguf ? pentry.gguf.repo : "-",
               pentry.onnx ? pentry.onnx.repo : "-");
    }
} else if (PROMPTS_ONLY) {
    /* refresh ONLY the prompts on the existing catalog: no discovery sweep,
     * no repo/revision churn.  Adjust OVERRIDES[..].prompts and rerun. */
    catalog = require(OUT).catalog;
    printf("== prompts refresh (existing catalog, no re-discovery) ==\n");
    for (var pk in catalog) {
        if (catalog[pk].category !== "embed") continue;
        var pBefore = sprintf("%J", catalog[pk].prompts || null);
        var pSrc = attachPrompts(pk, catalog[pk]);
        var pAfter = sprintf("%J", catalog[pk].prompts || null);
        printf("%-38s %s%s\n", pk,
               pSrc ? Object.keys(catalog[pk].prompts).join("+") + " (" + pSrc + ")" : "(none)",
               pAfter !== pBefore ? "  [changed]" : "");
    }
} else {

printf("== embed (sentence-similarity / feature-extraction) ==\n");
discoverEmbedRerank("embed", [
    "/api/models?pipeline_tag=sentence-similarity&sort=downloads&direction=-1&limit=" + CONFIG.scanLimit,
    "/api/models?pipeline_tag=feature-extraction&library=sentence-transformers&sort=downloads&direction=-1&limit=" + CONFIG.scanLimit
], CONFIG.embedTop, catalog);

printf("\n== rerank (text-ranking) ==\n");
discoverEmbedRerank("rerank", [
    "/api/models?pipeline_tag=text-ranking&sort=downloads&direction=-1&limit=" + CONFIG.scanLimit
], CONFIG.rerankTop, catalog);

printf("\n== gen (text-generation + gguf) ==\n");
discoverGen(CONFIG.genTop, catalog);

/* overrides that ADD models discovery didn't surface */
for (var name in OVERRIDES) {
    var ov = OVERRIDES[name];
    if (ov === "skip" || catalog[name] || !ov.category) continue;
    printf("%-34s (override) ", name); fflush(stdout);
    var entry = { category: ov.category };
    if (ov.onnx) { var o = resolveOnnx(name, ov.onnx, null); if (o) entry.onnx = o; }
    if (ov.gguf) { var g = resolveGguf(name, ov.gguf, repoTree(ov.gguf), false); if (g) entry.gguf = g; }
    if (ov.dim) entry.dim = ov.dim;    /* static dim for a gguf-only model (e.g. CLIP) */
    if (ov.license) entry.license = ov.license;  /* model license when the conversion
                                                    repo is untagged */
    if (entry.onnx || entry.gguf) { attachPrompts(name, entry); catalog[name] = entry; printf("ok\n"); }
    else { printf("FAILED\n"); failures.push(name + ": override repos unusable"); }
}

}   /* !PROMPTS_ONLY */

/* ------------------------- diff vs existing ------------------------- */
var prior = null;
try { prior = require(OUT).catalog; } catch (e) {}
if (prior) {
    var added = [], removed = [], changed = [];
    for (var k in catalog) {
        if (!prior[k]) added.push(k);
        else {
            var pr = (prior[k].onnx && prior[k].onnx.repo) + "|" + (prior[k].gguf && prior[k].gguf.repo);
            var nr = (catalog[k].onnx && catalog[k].onnx.repo) + "|" + (catalog[k].gguf && catalog[k].gguf.repo);
            if (pr !== nr) changed.push(k + "  (" + pr + " -> " + nr + ")");
        }
    }
    for (k in prior) if (!catalog[k]) removed.push(k);
    printf("\n-- diff vs existing catalog --\n  added:   %s\n  removed: %s\n", added.join(", ") || "(none)", removed.join(", ") || "(none)");
    if (changed.length) printf("  changed:\n    %s\n", changed.join("\n    "));
}

/* refuse to clobber a good catalog with a degraded run (rate limiting,
 * network trouble): a real update never shrinks the catalog by 30%+ */
if (prior) {
    var np = Object.keys(prior).length, nn = Object.keys(catalog).length;
    if (nn < np * 0.7) {
        printf("\nABORT: new catalog has %d models vs %d existing (degraded run?) -- NOT writing %s\n", nn, np, OUT);
        process.exit(1);
    }
}

/* ------------------------------ emit: splice into rampart-models.js ---- */
var names2 = Object.keys(catalog);
var lines = [];
for (var li = 0; li < names2.length; li++)
    lines.push('    ' + JSON.stringify(names2[li]) + ': ' + sprintf("%J", catalog[names2[li]]) + (li < names2.length - 1 ? ',' : ''));
var body = "var CATALOG = {\n" + lines.join("\n") + "\n};";
var modsrc = readFile(OUT, true);
var BM = "/* ==== BEGIN GENERATED CATALOG (do not edit; run gen-model-list.js) ==== */";
var EM = "/* ==== END GENERATED CATALOG ==== */";
var bi = modsrc.indexOf(BM), ei = modsrc.indexOf(EM);
if (bi < 0 || ei < 0) { printf("ERROR: catalog markers not found in %s\n", OUT); process.exit(1); }
modsrc = modsrc.substring(0, bi + BM.length) + "\n" + body + "\n" + modsrc.substring(ei);
writeFile(OUT, modsrc);
var counts = {};
for (k in catalog) counts[catalog[k].category] = (counts[catalog[k].category] || 0) + 1;
printf("\nwrote %s  (%d models: %s)\n", OUT, Object.keys(catalog).length, sprintf("%J", counts));

if (report.length) {
    printf("\n-- ranking choices (review) --\n");
    for (var i = 0; i < report.length; i++) printf("  %s\n", report[i]);
}
if (failures.length) {
    printf("\n-- FAILURES --\n");
    for (i = 0; i < failures.length; i++) printf("  %s\n", failures[i]);
    process.exit(1);
}
