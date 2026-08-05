# Rampart-langtools

## Build instructions:
```
git clone https://github.com/aflin/rampart-langtools.git
mkdir rampart-langtools/build && cd rampart-langtools/build
cmake ..
## or for CUDA build
cmake -DLT_ENABLE_GPU=1 ..
# make and copy modules to current rampart install dir
make install
```

### macOS support

Supported on **macOS 11 (Big Sur) and newer**, on both Apple Silicon and Intel.
embed, rerank, and text generation are all verified on macOS 11+.
rampart-onnx builds and passes its full suite on macOS too (all objects at the
11.0 deployment target).

### FreeBSD support

rampart-llamacpp, rampart-faiss, rampart-sentencepiece and **rampart-onnx** all
build and pass their suites on FreeBSD (amd64).  One extra package is needed
for the onnx build: `pkg install patch` (GNU patch -- see
`extern/onnxruntime-vendoring.md` for the details and the two vendored ORT
patches that make FreeBSD work).

## rampart-llamacpp use:

The primary use cases for rampart-llamacpp are **embedding generation**
(`initEmbed`) and **reranking** (`initRerank`), where the model runs
directly inside the rampart process.

Text generation (`initGen` / `predict`) is also supported but is
**experimental** — its API and internals are still evolving.  `initGen`
runs a single shared, continuously-batched context: when a `gen` handle
is shared across rampart threads, their requests are transparently pooled
into one engine (one copy of the model in memory, batched decoding).

### Embeddings:
```
// load module
var llamacpp=require('rampart-llamacpp');

// load model downloaded from huggingface
var emb = llamacpp.initEmbed('all-minilm-l6-v2_f16.gguf');

var mytext = "about a paragraph of text follows...";
// create a semantic vector from text:
// also available: embedTextToFp32Buf(), embedTextToNumbers(), and
// embedTextsToNumbers([texts]) -> [ {avgVec:[...]}, ... ] (one per text)
var v = emb.embedTextToFp16Buf(mytext);

// v = { vecs: [vec1, vec2, ...],           one vector per chunk
//       avgVec: avgOfVecs,                  renormalized average of vecs[]
//       coherence: 0..1,                    avg pairwise cosine between chunks (1 = single chunk)
//       chunks: [{start,end,tokens,text,oversized?}] } text span of each chunk
//       (oversized:true = one of several sub-windows of a too-big paragraph)
//
// Chunking is structure-aware (rp-chunker.c, shared with rampart-onnx):
// one vector per blank-line paragraph (fragments under minTokens merged),
// single-newline lines packed to the model window at line boundaries, and a
// sliding token window with 1/8 overlap when the text has no structure (or
// a single paragraph exceeds the window). If the text fits in one chunk,
// v.vecs.length==1 and v.vecs[0] == v.avgVec.
// initEmbed options: split:'auto'|'window', minTokens (default 32,
// -1 disables merging), packParagraphs:true (fewer, window-sized chunks).

//store vector and text somewhere
sql.exec("insert into vecs values (?,?,?,?)", [v.avgVec, docId, Title, Text]);

//unload
emb.destroy();
```

### Reranker:
```
// load module
var llamacpp=require('rampart-llamacpp');

// load model
var rrmodel = process.scriptPath + '/data/models/bge-reranker-v2-m3-Q8_0.gguf';
var rr = llamacpp.initRerank(rrmodel);

// get the score of how well a document/paragraph answers a question
// (sigmoid-squashed to 0..1 by default; initRerank(model,{sigmoid:false}) for raw):
var score = rr.rerank(qestion, mydoc);

// rank many documents: returns [{document, score, index}] sorted by score
// descending (index = the document's position in your array):
var ranked = rr.rerank(qestion, [doc1, doc2, doc3]);

// or just the scores, in DOCUMENT order:
var scores = rr.rerank(qestion, [doc1, doc2, doc3], true);
```

### Text Generation (experimental):

> **Experimental.**  `initGen`/`predict`/`predictAsync` are under active
> development and the API may change.  Vision/image generation is **not**
> supported by this engine (image vectors are handled separately by
> `rampart-clip`).

```
var llamacpp = require('rampart-llamacpp');

// load a text model.  nSeqMax sets how many requests may decode together
// (batched) in the one shared context.
var gen = llamacpp.initGen('/path/to/model.gguf', {
    nCtx:    4096,
    nSeqMax: 4
});

// SYNC: predict() blocks the calling thread and returns the full text.
var text = gen.predict({
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user",   content: "What is the capital of France?" }
    ],
    maxTokens: 128
});
rampart.utils.printf("%s\n", text);

// ASYNC/STREAMING: predictAsync(opts, onToken, onDone) is non-blocking and
// streams tokens as they are produced.  Multiple in-flight calls (across
// threads or from one event loop) batch together through the shared engine.
// It returns a handle whose .cancel() hard-stops that generation (frees the slot).
var h = gen.predictAsync(
    { prompt: "Explain how a combustion engine works.", maxTokens: 256, temp: 0.7 },
    function(res) { if (!res.done && !res.error) rampart.utils.printf("%s", res.token); },
    function(res) { rampart.utils.printf("\n[done]\n"); }   // res.fullText, res.error
);
// h.cancel();   // stop this generation early

// retrieve the full text of the last sync predict()
var fullText = gen.getLast();

// free resources when done
gen.destroy();
```

### initGen Options:

Most options map 1:1 to the matching `llama-server` command-line flag (camelCase
of the flag — e.g. `gpuLayers` = `--gpu-layers`, `flashAttn` = `--flash-attn`,
`cacheTypeK` = `--cache-type-k`). See the
[llama.cpp / llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for what each does.

```
var gen = llamacpp.initGen('/path/to/model.gguf', {
    // model loading
    gpuLayers, mainGpu, splitMode, useMmap, useMlock, checkTensors,

    // context (nCtx: 0 or -1 = the model's trained max, like llama-server)
    nCtx, nSeqMax, nBatch, nUBatch, threads, threadsBatch,
    ropeScaling, ropeFreqBase, ropeFreqScale,
    yarnExtFactor, yarnAttnFactor, yarnBetaFast, yarnBetaSlow, yarnOrigCtx,
    flashAttn,                 // true | false | "auto"
    cacheTypeK, cacheTypeV,    // KV cache type, e.g. "q8_0"
    offloadKqv, opOffload, kvUnified,

    // chat
    jinja,             // apply the chat template via Jinja (default: true)
    chatTemplate,      // custom Jinja template (string)
    chatTemplateFile,  // ... or read the template from a file
});
```

### predict Options:
```
// the same options object is accepted by predict() and predictAsync()
gen.predict({
    // Prompt (use one or the other)
    prompt:   "...",                   // simple text prompt
    messages: [{role, content}, ...],  // chat-style messages

    // Generation params
    maxTokens:     12800,  // maximum tokens to generate
    temp:          0.8,    // temperature
    topP:          0.95,   // top-p sampling
    topK:          40,     // top-k sampling
    minP:          0.05,   // min-p sampling
    repeatPenalty: 1.1,    // repetition penalty
    repeatLastN:   -1,     // tokens to consider for repeat penalty (-1 = auto)
    seed:          -1,     // RNG seed (-1 = random)
    stop:          [],     // array of stop strings
});
// predict() returns the full generated string.  For token-by-token
// streaming use predictAsync(opts, onToken, onDone) instead.
```

### initEmbed / initRerank Options:

`initEmbed` and `initRerank` accept the same model-loading and context options as
`initGen` above (`gpuLayers`, `nCtx`, `threads`, `flashAttn`, `cacheTypeK/V`, the
`rope*`/`yarn*` family, etc. — see the
[llama.cpp docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)),
plus:

```
var emb = llamacpp.initEmbed('model.gguf', {
    pooling,      // "none" | "mean" | "cls" | "last" | "rank" (--pooling)
    attention,    // "causal" | "non-causal"                   (--attention)
    batchChunks,  // EXPERIMENTAL -- see below
    batchTokens,  // EXPERIMENTAL -- see below
});
```

The legacy names `nctx`, `ubatch`, `nthreads`, `nthreads_batch` are still accepted
as aliases for `nCtx`, `nUBatch`, `threads`, `threadsBatch`.

### Chunk batching (EXPERIMENTAL)

A document's chunks are independent sequences, so they can be packed into ONE
`llama_decode` instead of one decode each. This is **experimental**.

```
llamacpp.embedDefaults({
    batchChunks,   // null = auto (on for a GPU backend, off for CPU),
                   // false = one chunk per decode, true = as many as fit, N = cap
    batchTokens,   // soft cap on TOKENS per packed decode (default 512)
    threads,       // n_threads       for embedding contexts
    threadsBatch,  // n_threads_batch for embedding contexts
});
llamacpp.embedDefaults()   // -> the settings in effect, plus gpuInUse
```

`embedDefaults()` sets process-wide defaults for `initEmbed` (an explicit option
on the call still wins) and is the **only** way to configure the `rp_embed_*` C
entry points that rampart-sql drives, since those take no options object. Set it
before models are loaded; a handle captures the settings at load.

Batching never changes chunk boundaries, `k`, or the byte spans — only how many
already-formed chunks share a decode. It does shift vector values slightly
(a larger batch selects different matmul kernels), on the order of 1e-3 with
cosine ~0.9999, scaling with weight quantization.

`batchTokens` matters more than `batchChunks`. Embedding models are typically
no-KV encoders, so attention is computed over the whole packed batch as one NxN
matrix with cross-sequence pairs merely masked — cost grows with batch tokens x
model width while the per-decode overhead saved grows only linearly. Past a few
hundred tokens the quadratic wins and batching starts LOSING. Measured on an
RTX 4070 Ti, bge-m3 gained 1.15x at a 512-token cap but ran **0.65x (slower than
unbatched)** with no cap. 512 is the largest value that did not regress on any
model tested; the optimum falls as model width rises, so re-measure on new
hardware. Wall-clock gains measured there: bge-small 2.2x, nomic 1.7x, bge-m3
1.2x; on CPU, batching measured ~1.0x (no benefit), which is why `auto` leaves
it off there.

### modelInfo (read model metadata without loading weights):

`llamacpp.modelInfo(path)` returns a model's key parameters by reading only its
GGUF metadata and vocabulary — it does **not** load the weight tensors, so there
is no GPU upload and the call is fast (tens to a few hundred ms, mostly vocab
parsing) even for multi-gigabyte models.

```
var info = llamacpp.modelInfo('bge-m3-FP16.gguf');
// {
//   embedDim:   1024,   // size of the vector embed()/embedText* produces
//   hiddenDim:  1024,   // model hidden size
//   nCtxTrain:  8192,   // trained context length
//   nLayer:     24,
//   arch:       "bert", // GGUF general.architecture
//   pooling:    "cls",  // declared pooling: none|mean|cls|last|rank|unspecified
//   nParams:    566703104
// }
```

`embedDim` is the output embedding size (it prefers the GGUF
`embedding_length_out` of a projection head, falling back to `embedding_length`).
This lets a pipeline size its vector storage/index from the model itself rather
than hard-coding a dimension — e.g. `var vecDim = llamacpp.modelInfo(f).embedDim;`.

### Environment variables:

`RAMPART_LLAMA_CUDA_GRAPHS` — On CUDA builds, ggml caches a captured CUDA graph
per compute-graph shape and only evicts entries after they have been idle for 10
seconds. Batched embedding and reranking decode a stream of varying shapes, which
fills that cache faster than it drains and makes GPU memory climb until it runs
out. CUDA graphs only speed up single-stream text generation, so `initEmbed`,
`initRerank`, and the `rampart-sql` embedding path disable them automatically (by
setting `GGML_CUDA_DISABLE_GRAPHS=1` before the first decode). A process that only
calls `initGen` never disables them, so generation performance is unaffected.

Set `RAMPART_LLAMA_CUDA_GRAPHS` (to any value) to opt out and keep CUDA graphs on
even for embedding/reranking — accepting the memory growth above. The setting is
process-global and read once at startup, so it cannot be toggled per call. It has
no effect on CPU or Metal (Apple) builds.

### Logging:
llama.cpp produces log output during model loading and initialization.
This output is captured in an internal buffer rather than printed to
stdout/stderr.

```
var llamacpp=require('rampart-llamacpp');
var emb = llamacpp.initEmbed('all-minilm-l6-v2_f16.gguf');

// retrieve the captured log output as a string
var log = llamacpp.getLog();
console.log(log);

// clear the log buffer
llamacpp.resetLog();
```

Note: The log buffer has a maximum size of 40KB.  If it overflows, the
oldest half of the log is discarded and the first line will read
"WARN: log overflow".  The log callback is process-global, so in
multi-threaded usage all threads write to the same buffer.  The buffer
is protected by a mutex and is safe to use from multiple threads.

## rampart-faiss

### Creating index:
```
rampart.globalize(rampart.utils); // for printf, dateFmt and repl

//example building index for about 30m vectors from a sql table named vecs:
var faiss = require('rampart-faiss');

// see https://github.com/facebookresearch/faiss/wiki/The-index-factory
// and https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
// Highly recommended that IDMap or IDMap2 is used to store artbitrary ids
// associated with each vector.  Otherwise the associated id will be sequentially
// incremented starting with 0.
var idx = faiss.openFactory("IDMap2,OPQ96,IVF262144,PQ48", 384);

// the name we will eventually use for the saved index
var indname = "all-minilm-vec.OPQ96_IVF262144_PQ48_faiss";

//if compiled for CUDA and available:
idx.enableGpu();
printf("GPU Enabled\n");

// if index requires training, idx.trainer will be defined.
if( idx.trainer) {
    // make a new trainer, save train vectors in new file ./tdata
    // or reload vectors in ./tdata and skip/continue to insert

    var trainer = new idx.trainer('tdata');
    printf("%3J\n%s\n", trainer, dateFmt('%c %z'));

    //insert vectors into ./tdata file
    sql.exec("select Id, Vec from vecs", {skipRows:0, maxRows: 10000000}, function(row) {
        trainer.addTrainingfp16(row.Vec); // or addTrainingfp32()
        i++;
        if( ! (i%1000))
        {
            printf("train inserted %d: %.0f\r", i, row.Id);
            fflush(stdout);
        }
    });

    //train from vectors in ./tdata.
    printf("\n%s\nTraining, go get some coffee, read a book or two, don't touch the keyboard ...\n", dateFmt('%c %z'));

    trainer.train();
    console.log(dateFmt('%c %z'));
}

var cpointf = sprintf("%s-trained", indname);
printf("\n%s: Saving training %s\n", dateFmt('%c %z'), cpointf);
idx.save(cpointf); // This is our trained, but empty index

var res = sql.one("select count(Id) tot from vecs");
var tot=res.tot

sql.exec("select Id, Vec from vecs", {maxRows:-1}, function(row,i) {
    // add vector using addFp16() or addFp32()
    idx.addFp16(row.Id, row.Vec);
    if( ! (i%10))
    {
        printf("inserted %d of %d: %llu\r", i, tot, row.Id);
	// save a checkpoint every 2m inserts in case of interrupt
        if( ! (i%2000000) )
        {
            var cpointf = sprintf("%s-%d", indname, i);
            printf("\n%s: Saving checkpoint %s\n", dateFmt('%c %z'), cpointf);
            idx.save(cpointf);
        }
    }
    i++;
});

// done inserting, save with filename
idx.save(indname);

//test it out:
var llamacpp = require('rampart-llamacpp');
var emb = llamacpp.initEmbed('all-minilm-l6-v2_f16.gguf');

printf("\nSemantic Vector Search Test\nEnter Query:\n");

var rl = repl("Query: ");

while ( (l=rl.next()) ) {
    var v = emb.embedTextToFp16Buf(l);
    var res = idx.searchFp16(x.avgVec, /*nres = */10, /* nprobe = */128);
    printf("\nRESULTS:\n");
    var ids = [];
    var idtoscore={};
    res.forEach(function(r){ ids.push(r.id); idtoscore[r.id]=r.distance; });
    //get results from sql table, reorder by actual cosine similarity, print
    sql.exec("select vecdist(Vec, ?, 'dot', 'f16') Dist, Id, Title, Text from vecs where Id in (?) order by 1 DESC", [x.avgVec, ids],
      function(sres,i){
        printf("%as: %as, (%.2f : %.2f)\n%.80s\n", "green", i, "green", sres.Title, idtoscore[sres.Idsec], sres.Dist, sres.Text);
      }
    );
    rl.refresh();
}
```

### Loading existing index:
```
var faiss = require('rampart-faiss');

var indname = "all-minilm-vec.OPQ96_IVF262144_PQ48_faiss";

// load index from file into ram
var idx = faiss.openIndexFromFile(indname);
// or open read only with memmap to serve from disk:
var idx = faiss.openIndexFromFile(indname, true);

// use just like in example above.
var llamacpp = require('rampart-llamacpp');
var emb = llamacpp.initEmbed('all-minilm-l6-v2_f16.gguf');
var v = emb.embedTextToFp16Buf(myquery);
var res = idx.searchFp16(x.avgVec, /*nres = */10, /* nprobe = */128);
// res is an array of Ids inserted into the index
```

## sentencepiece
```
var sp = require('rampart-sentencepiece');

// model from https://huggingface.co/BAAI/bge-m3/blob/main/sentencepiece.bpe.model
var encoder = sp.init('./sentencepiece.bpe.model');

var encoded = encoder.encode('hello there you goat');
// encoded = ["▁hell","o","▁there","▁you","▁go","at"]
var decoded = sp.decode(encoded); // = "hello there you goat"
```

## langtools
All the modules packaged into one.
```
var langtools = require('rampart-langtools');
var faiss = langtools.faiss;
var llamacpp = langtools.llamacpp;
Var sp = langtools.sentencepiece;
```

## dependencies:
* libgfortran.so.5
* libomp.so.5
* cuda libraries for gpu build on linux
