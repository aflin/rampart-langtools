#!/usr/bin/env rampart
/* rampart-onnx test suite.
 *
 * Loads plain `rampart-onnx` -- the ONE unified module (static CPU ORT inside;
 * an optional modules/onnx-cuNN/ runtime dir beside it enables GPU -- see
 * onnx.runtimeInfo() for what got selected).
 *
 * Covers: session layer (run/typed outputs/errors/destroy), native tokenizers
 * (WordPiece + SentencePiece), initEmbed self-config with full return shape
 * (vecs/avgVec/coherence/chunks + oversized), batching, prefixes, initRerank
 * self-config (sorted+index+sigmoid / scoresOnly doc order), and rampart.thread:
 * a raw session shared across workers, plus worker-local initEmbed/initRerank
 * (handle objects are closure-based and intentionally NOT thread-copied --
 * workers build their own from the same model file).
 *
 * Each model section needs a model file; this script offers to download each
 * from HuggingFace into the standard, reusable location ~/.rampart/models/
 * (so other tools/demos can find them).  Decline a download and that section
 * prints "- skipped".
 *
 *   all-minilm-l6-v2        embed   WordPiece / BERT        (~90 MB)
 *   multilingual-e5-small   embed   SentencePiece / XLM-R   (~450 MB)
 *   ms-marco-minilm-l6-v2   rerank  WordPiece cross-encoder (~90 MB)
 *
 * add_mul.onnx is a tiny synthetic fixture under ~/.rampart/models/test/
 * (not in any catalog; skipped if absent, never downloaded).
 *
 *   rampart onnx-test.js
 */
rampart.globalize(rampart.utils);

/* locate rampart-onnx.so, in order:
   1) build/  2) build-cpu/  3) build-gpu/  4) build_cpu/  5) build_gpu/
      (local builds, relative to this script)
   6) modules/            (e.g. when run from /usr/local/rampart/run_tests.sh)
   7) bare require        (installed / on rampart's module path)
   require auto-appends the .so extension.  The resolved name is kept in
   ONNX_FROM so worker threads require() the SAME module the main thread did. */
var ONNX_FROM = null;
function loadModule() {
    var sp = process.scriptPath;
    var tries = [
        sp + '/build/rampart-onnx',
        sp + '/build-cpu/rampart-onnx',
        sp + '/build-gpu/rampart-onnx',
        sp + '/build_cpu/rampart-onnx',
        sp + '/build_gpu/rampart-onnx',
        sp + '/modules/rampart-onnx',
        'rampart-onnx'
    ];
    for (var i = 0; i < tries.length; i++) {
        try { var m = require(tries[i]); ONNX_FROM = tries[i]; return m; } catch(e) {}
    }
    printf("ERROR: could not load rampart-onnx.so from any of:\n  %s\n", tries.join('\n  '));
    process.exit(1);
}
var onnx = loadModule();

/* ---- models: located + fetched via rampart-models.js (onnx), cached under
   ~/.rampart/models/<category>/ (standard, reusable location so demos and other
   tools find them).  rampart-models.js is beside this script in the source tree,
   or an installed module (modules/rampart-models.js) from the install tree. --- */
function loadModels() {
    var sp = process.scriptPath;
    try { return require(sp + '/rampart-models.js'); }
    catch (e) { try { return require('rampart-models'); } catch (e2) { return null; } }
}
var models = loadModels();
var MODELS = {
    embed:  { name: 'all-minilm-l6-v2',      what: 'embedding / WordPiece test' },
    spm:    { name: 'multilingual-e5-small', what: 'SentencePiece / prefix test' },
    rerank: { name: 'ms-marco-minilm-l6-v2', what: 'rerank test' }
};

var MODELDIR = homedir() + '/.rampart/models';
var ADDMUL   = MODELDIR + '/test/add_mul.onnx';   // local fixture (never fetched)

/* ================================================================
   test harness (same shape as llamacpp-test.js / faiss-test.js)
   ================================================================ */
var testnum = 0, failed = 0;

function testFeature(name, test) {
    var error = false;
    testnum++;
    if (typeof test == 'function') {
        try { test = test(); }
        catch(e) { error = e; test = false; }
    }
    printf("testing onnx - %3d - %-58s - ", testnum, name);
    fflush(stdout);
    if (test) {
        printf("passed\n");
    } else {
        printf(">>>>> FAILED <<<<<\n");
        failed++;
        if (error) console.log(error);
    }
}

/* ================================================================
   helpers
   ================================================================ */
function has(p) { return !!stat(p); }
function cos(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/* stdin/stdout are bare globals in current rampart; older builds expose them
   on rampart.utils only */
var STDIN  = (typeof stdin  !== 'undefined') ? stdin  : rampart.utils.stdin;
var STDOUT = (typeof stdout !== 'undefined') ? stdout : rampart.utils.stdout;

function ask(question) {
    printf("%s", question);
    fflush(STDOUT);
    var a = fgets(STDIN, 255);
    return a ? a.trim().toLowerCase() : '';
}

/* Locate a model through rampart-models.js, fetching it if absent.  Returns the
   .onnx FILE path (initEmbed/initRerank auto-discover the tokenizer beside it),
   or null if rampart-models.js is unavailable, the user declines a download, or
   the fetch fails -> the section using it is skipped.  The confirm hook fires
   ONLY when a download is actually needed (present models resolve straight
   through).

   fp32 (the reference precision) is requested deliberately: it always exists,
   needs no GPU, and its on-disk name (onnx/model.onnx) matches any pre-existing
   download, so a present model resolves without a surprise re-fetch. */
function obtain(m) {
    if (!models) { printf("  rampart-models.js not found; skipping %s\n", m.name); return null; }
    try {
        return models.onnxGet(m.name, {
            precision: 'fp32',
            confirm: function(info) {
                var r = ask(sprintf("  '%s' (%s) is needed for the %s.\n" +
                                    "  Download to %s? [y/N]: ",
                                    info.name, info.size, m.what, info.dest));
                return r === 'y' || r === 'yes';
            }
        });
    } catch(e) {
        printf("  fetch failed: %s\n", e.message || e);
        return null;
    }
}

/* obtain() returns the .onnx FILE; the tokenizer files (vocab.txt / the SPM
   model) live in the model's base dir -- the file's dir, or its parent when the
   model sits in an onnx/ subdir (the usual HuggingFace layout). */
function modelDir(f) {
    var d = f.replace(/\/[^\/]*$/, '');
    return /\/onnx$/.test(d) ? d.replace(/\/onnx$/, '') : d;
}

var Q   = "how do I bake bread at home";
var REL = "Mix flour, water, salt and yeast, knead, proof, then bake in a hot oven.";
var IRR = "Interest rates rose again on Tuesday.";

var FEEDS = { a: { data: [1,2,3],    shape: [3], type: 'float32' },
              b: { data: [10,20,30], shape: [3], type: 'float32' },
              k: { data: [0,0,0],    shape: [3], type: 'int64'   } };

/* ================================================================
   Test: module loads + API surface
   ================================================================ */
testFeature("module loads", typeof onnx === 'object' && onnx !== null);
testFeature("initSession exists", typeof onnx.initSession === 'function');
testFeature("initEmbed exists",   typeof onnx.initEmbed   === 'function');
testFeature("initRerank exists",  typeof onnx.initRerank  === 'function');

/* ================================================================
   Test: session layer (run / typed outputs / errors / destroy)
   ================================================================ */
function runSessionTest() {
    printf("\n--- session layer (initSession) ---\n");
    if (!has(ADDMUL)) { printf("- skipped: no %s\n", ADDMUL); return; }

    var s = null;
    testFeature("initSession + run", function() {
        s = onnx.initSession(ADDMUL);
        var out = s.run(FEEDS);
        return JSON.stringify(Array.prototype.slice.call(out.sum.array)) === '[11,22,33]';
    });
    if (!s) return;

    testFeature("output raw data buffer present", function() {
        return s.run(FEEDS).sum.data.byteLength === 12;
    });
    testFeature("inputs() lists 3", function() { return s.inputs().length === 3; });
    testFeature("modelInfo matches", function() {
        var mi = onnx.modelInfo(ADDMUL);
        return mi.inputs.length === 3 && mi.outputs.length >= 1;
    });
    testFeature("malformed feed throws cleanly", function() {
        try { s.run({ a: 42 }); return false; }
        catch(e) { return /must be an object/.test(e.message); }
    });
    testFeature("session still works after failed run", function() {
        return s.run(FEEDS).sum.array[2] === 33;
    });
    testFeature("run after destroy throws", function() {
        s.destroy();
        try { s.run(FEEDS); return false; }
        catch(e) { return /destroyed/.test(e.message); }
    });
}

/* ================================================================
   Test: native tokenizers (WordPiece + SentencePiece)
   ================================================================ */
function runTokenizerTest(minilm, e5) {
    printf("\n--- native tokenizers ---\n");
    if (!minilm) printf("- wordPiece skipped: %s unavailable\n", MODELS.embed.name);
    else {
        var wp = null;
        testFeature("wordPieceTokenizer encodes", function() {
            wp = onnx.wordPieceTokenizer(modelDir(minilm) + '/vocab.txt');
            var ids = wp.encodeIds("unbelievable tokenization");
            return ids.length > 2 && typeof ids[0] === 'number';
        });
        testFeature("wordPiece vocabSize sane", function() {
            return wp && wp.vocabSize > 30000;
        });
    }

    if (!e5) printf("- spTokenizer skipped: %s unavailable\n", MODELS.spm.name);
    else testFeature("spTokenizer encodes", function() {
        return onnx.spTokenizer(modelDir(e5)).encodeIds("hello world").length >= 2;
    });
}

/* ================================================================
   Test: embedding (initEmbed self-config)
   ================================================================ */
function runEmbedTest(minilm) {
    printf("\n--- embedding (initEmbed) ---\n");
    if (!minilm) { printf("- skipped\n"); return null; }

    var emb = null, r = null;
    testFeature("initEmbed loads model", function() {
        emb = onnx.initEmbed(minilm);
        return typeof emb === 'object' && emb !== null;
    });
    if (!emb) return null;

    testFeature("full return shape (vecs/avgVec/coherence/chunks)", function() {
        r = emb.embedTextToNumbers("hello world");
        return Array.isArray(r.vecs) && r.avgVec.length === 384 &&
               typeof r.coherence === 'number' && r.chunks[0].text === "hello world";
    });
    testFeature("vector L2-normalized", function() {
        return Math.abs(cos(r.avgVec, r.avgVec) - 1) < 1e-3;
    });
    testFeature("relevant scores above irrelevant", function() {
        var q   = emb.embedTextToNumbers(Q).avgVec,
            rel = emb.embedTextToNumbers(REL).avgVec,
            irr = emb.embedTextToNumbers(IRR).avgVec;
        if (cos(q, rel) <= cos(q, irr))
            throw new Error(sprintf("relevant %.3f <= irrelevant %.3f",
                                    cos(q, rel), cos(q, irr)));
        return true;
    });
    testFeature("empty text -> {vecs:[]}", function() {
        return emb.embedTextToNumbers("").vecs.length === 0;
    });

    /* chunking: one vector per paragraph; monster paragraph sub-windows w/ oversized */
    var mon = [];
    for (var i = 0; i < 900; i++) mon.push("token" + (i % 50) + " word" + i);
    var doc = "First short paragraph about ovens.\n\n" + mon.join(" ") +
              "\n\nClosing short paragraph about bread.";
    var rd = null, over = null, norm = null;

    testFeature("paragraph chunking (multiple vecs, spans line up)", function() {
        rd   = emb.embedTextToNumbers(doc);
        over = rd.chunks.filter(function(c) { return c.oversized; });
        norm = rd.chunks.filter(function(c) { return !c.oversized; });
        return rd.vecs.length >= 4 && rd.chunks.length === rd.vecs.length &&
               doc.slice(rd.chunks[0].start, rd.chunks[0].end) === rd.chunks[0].text;
    });
    testFeature("oversized paragraph flagged on its sub-windows", function() {
        return over.length >= 2 &&
               over.every(function(c) { return c.start === over[0].start; });
    });
    testFeature("normal paragraphs unflagged", function() { return norm.length === 2; });
    testFeature("coherence in [0,1]", function() {
        if (!(rd.coherence >= 0 && rd.coherence <= 1))
            throw new Error("coherence = " + rd.coherence);
        return true;
    });

    /* batch + packed forms */
    var batch = null;
    testFeature("embedTextsToNumbers -> [{avgVec}]", function() {
        batch = emb.embedTextsToNumbers(["hello world", "goodbye moon"]);
        return batch.length === 2 && batch[0].avgVec.length === 384;
    });
    testFeature("batch matches single", function() {
        return cos(batch[0].avgVec, r.avgVec) > 0.9999;
    });
    testFeature("embedTextToFp32Buf returns a vector", function() {
        return emb.embedTextToFp32Buf("hello world").avgVec.byteLength === 384 * 4;
    });
    testFeature("embedTextToFp16Buf returns a vector", function() {
        return emb.embedTextToFp16Buf("hello world").avgVec.byteLength === 384 * 2;
    });
    return emb;
}

/* ================================================================
   Test: SentencePiece family + e5-style query/passage prefixes
   ================================================================ */
function runPrefixTest(e5path) {
    printf("\n--- embedding, SPM + prefixes (initEmbed) ---\n");
    if (!e5path) { printf("- skipped\n"); return; }

    var e5 = null, vq, vp, vi;
    testFeature("e5 (SPM) embeds w/ prefixes, 384-dim", function() {
        e5 = onnx.initEmbed(e5path, { queryPrefix: "query: ", passagePrefix: "passage: " });
        vq = e5.embedTextToNumbers(Q, true).avgVec;    /* isQuery */
        vp = e5.embedTextToNumbers(REL, false).avgVec;
        vi = e5.embedTextToNumbers(IRR, false).avgVec;
        return vq.length === 384;
    });
    if (!e5) return;

    testFeature("e5 relevance sane", function() {
        if (cos(vq, vp) <= cos(vq, vi))
            throw new Error(sprintf("relevant %.3f <= irrelevant %.3f",
                                    cos(vq, vp), cos(vq, vi)));
        return true;
    });
    e5.destroy();
    testFeature("e5 destroy", true);
}

/* ================================================================
   Test: rerank (initRerank self-config)
   ================================================================ */
function runRerankTest(msmarco) {
    printf("\n--- rerank (initRerank) ---\n");
    if (!msmarco) { printf("- skipped\n"); return; }

    var rr = null, DOCS = [IRR, REL, "Cats are excellent nappers."], ranked = null;
    testFeature("sorted desc with document/score/index", function() {
        rr = onnx.initRerank(msmarco);
        ranked = rr.rerank(Q, DOCS);
        return ranked[0].index === 1 && ranked[0].document === REL &&
               ranked[0].score >= ranked[1].score;
    });
    if (!rr) return;

    testFeature("sigmoid default (scores 0..1)", function() {
        return ranked.every(function(x) { return x.score >= 0 && x.score <= 1; });
    });
    testFeature("scoresOnly in DOCUMENT order", function() {
        var so = rr.rerank(Q, DOCS, true);
        return so.length === 3 && so[1] === ranked[0].score;
    });
    testFeature("single doc -> number", function() {
        return typeof rr.rerank(Q, REL) === 'number';
    });
    rr.destroy();
    testFeature("rerank destroy", true);
}

/* ================================================================
   Test: rampart.thread -- a RAW session shared into workers
   ================================================================ */
function runSharedSessionTest() {
    printf("\n--- rampart.thread (shared raw session) ---\n");
    if (!has(ADDMUL)) { printf("- skipped: no %s\n", ADDMUL); return; }

    /* a RAW session created in main is shared into workers (deep-copied handle,
       same native session) -- both must compute correctly, then main again */
    var shared = onnx.initSession(ADDMUL);
    var t1 = new rampart.thread(), t2 = new rampart.thread();
    var worker = function(a) {
        try { rampart.thread.put(a.key, Array.prototype.slice.call(a.s.run(a.f).sum.array)); }
        catch(e) { rampart.thread.put(a.key, "threw: " + e.message); }
    };
    t1.exec(worker, { s: shared, f: FEEDS, key: 't1' });
    t2.exec(worker, { s: shared, f: FEEDS, key: 't2' });

    testFeature("shared session runs in worker 1", function() {
        return JSON.stringify(rampart.thread.get('t1', 60000)) === '[11,22,33]';
    });
    testFeature("shared session runs in worker 2", function() {
        return JSON.stringify(rampart.thread.get('t2', 60000)) === '[11,22,33]';
    });
    testFeature("main still runs the shared session", function() {
        return shared.run(FEEDS).sum.array[0] === 11;
    });

    /* destroy in main -> any thread's reuse throws (no crash) */
    shared.destroy();
    t1.exec(function(a) {
        try { a.s.run(a.f); rampart.thread.put('t1d', "no-throw"); }
        catch(e) { rampart.thread.put('t1d', "threw"); }
    }, { s: shared, f: FEEDS });
    testFeature("destroyed session throws in worker", function() {
        return rampart.thread.get('t1d', 60000) === "threw";
    });

    t1.terminate(); t2.terminate();
}

/* ================================================================
   Test: rampart.thread -- worker-local initEmbed / initRerank
   ================================================================ */
function runWorkerHandleTest(emb, minilm, msmarco) {
    printf("\n--- rampart.thread (worker-local handles) ---\n");
    if (!emb || !minilm || !msmarco) { printf("- skipped\n"); return; }

    /* embed/rerank handles are closure-based: each worker builds its OWN from
       the model file (the supported pattern); results must agree with main's */
    var vmain = emb.embedTextToNumbers("cross thread check").avgVec;
    var t3 = new rampart.thread();
    t3.exec(function(a) {
        var o = require(a.mod);
        try {
            var e = o.initEmbed(a.dir);
            var v = e.embedTextToNumbers("cross thread check").avgVec;
            var rr = o.initRerank(a.rdir);
            var top = rr.rerank(a.q, a.docs)[0].index;
            rr.destroy(); e.destroy();
            rampart.thread.put('t3', { v: v, top: top });
        } catch(err) { rampart.thread.put('t3', { err: err.message }); }
    }, { mod: ONNX_FROM, dir: minilm, rdir: msmarco, q: Q, docs: [IRR, REL] });

    var w = rampart.thread.get('t3', 120000);
    testFeature("worker-local initEmbed matches main", function() {
        if (w && w.err) throw new Error(w.err);
        return w && w.v && cos(w.v, vmain) > 0.9999;
    });
    testFeature("worker-local initRerank ranks correctly", function() {
        return w && w.top === 1;
    });
    testFeature("main embed still works after worker", function() {
        return emb.embedTextToNumbers("still alive").avgVec.length === 384;
    });
    t3.terminate();
}

/* ================================================================
   run
   ================================================================ */
printf("\nmodule: rampart-onnx (ORT %s)\n", onnx.onnxVersion());

var MINILM  = obtain(MODELS.embed);     // WordPiece / BERT
var E5      = obtain(MODELS.spm);       // SentencePiece / XLM-R
var MSMARCO = obtain(MODELS.rerank);    // WordPiece cross-encoder

runSessionTest();
runTokenizerTest(MINILM, E5);
var emb = runEmbedTest(MINILM);
runPrefixTest(E5);
runRerankTest(MSMARCO);
runSharedSessionTest();
runWorkerHandleTest(emb, MINILM, MSMARCO);
if (emb) emb.destroy();

printf("\nonnx: %d tests run, %d failed\n", testnum, failed);
process.exit(failed ? 1 : 0);
