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

### macOS requirement

On Apple Silicon, **macOS 15 (Sequoia) or newer is recommended** and is the only
version verified to work, due to a Metal regression in older macOS. macOS 14 is
untested; verified to fail on macOS 11–13.

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
// also available is embedTextToFp32Buf() and embedTextToNumbers()
var v = emb.embedTextToFp16Buf(mytext);

// v = {vecs[vec1, vec2, ...], avgVec: avgOfVecs}
// If passage is not too large for model, v.vecs.length==1
// and v.vecs[0] == v.avgVec
// Otherwise avgVec will be a renormalized average of vecs[]

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

// get the score of how well a document/paragraph answers a question:
var score = rr.rerank(qestion, mydoc);
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
    pooling,    // "none" | "mean" | "cls" | "last" | "rank"   (--pooling)
    attention,  // "causal" | "non-causal"                     (--attention)
});
```

The legacy names `nctx`, `ubatch`, `nthreads`, `nthreads_batch` are still accepted
as aliases for `nCtx`, `nUBatch`, `threads`, `threadsBatch`.

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
