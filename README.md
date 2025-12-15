# Rampart-langtools

## Build instructions:
```
git clone https://github.com/aflin/rampart-langtools.git
mkdir rampart-langtools/build && cd rampart-langtools/build
cmake ..
## or for CUDA build
cmake LT_ENABLE_GPU=1 ..
# make and copy modules to current rampart install dir
make install
```
## rampart-llamacpp use:
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
var score = rr.rerank = llama_rr.rerank(qestion, mydoc);
```

### Options:
```
// options like nctx, n_threads_batch, batch, ubatch can also be set:
// load module
var llamacpp=require('rampart-llamacpp');

// load model
var rrmodel = process.scriptPath + '/data/models/bge-reranker-v2-m3-Q8_0.gguf';
var rr = langtools.llamacpp.initRerank(rrmodel, {ubatch:256});
```

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
var idx = langtools.faiss.openFactory("IDMap2,OPQ96,IVF262144,PQ48", 384);

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


