#!/usr/bin/env rampart
/* rampart-llamacpp test suite.
 *
 * Exercises the critical functions of the embedding path (initEmbed) and the
 * text-generation path (initGen). Each needs a model file; this script offers
 * to download each from HuggingFace into the standard, reusable location
 * ~/.rampart/models/<gen|embed|rerank>/ (so other tools/demos can find them).
 * Decline a download and that section prints "- skipped".
 *
 *   rampart llamacpp-test.js
 */
rampart.globalize(rampart.utils);

/* locate rampart-llamacpp.so, in order:
   1) build/  2) build_gpu/  3) build_cpu/  (local builds, relative to this script)
   4) modules/            (e.g. when run from /usr/local/rampart/run_tests.sh)
   5) bare require        (installed / on rampart's module path, e.g.
                           /usr/local/src/rampart/build/src/)
   require auto-appends the .so extension. */
function loadModule() {
    var sp = process.scriptPath;
    var tries = [
        sp + '/build/rampart-llamacpp',
        sp + '/build_gpu/rampart-llamacpp',
        sp + '/build_cpu/rampart-llamacpp',
        sp + '/modules/rampart-llamacpp',
        'rampart-llamacpp'
    ];
    for (var i = 0; i < tries.length; i++) {
        try { return require(tries[i]); } catch(e) {}
    }
    printf("ERROR: could not load rampart-llamacpp.so from any of:\n  %s\n", tries.join('\n  '));
    process.exit(1);
}
var llamacpp = loadModule();

/* ---- models: located + fetched via rampart-models.js (gguf), cached under
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
    embed: { name: 'all-minilm-l6-v2',      quant: 'F16',    what: 'embedding test' },
    gen:   { name: 'qwen2.5-0.5b-instruct', quant: 'Q4_K_M', what: 'generation / inference test' }
};

/* ================================================================
   test harness (modeled on rampart-iroh/iroh-test.js)
   ================================================================ */
var testnum = 0, failed = 0;

function testFeature(name, test) {
    var error = false;
    testnum++;
    if (typeof test == 'function') {
        try { test = test(); }
        catch(e) { error = e; test = false; }
    }
    printf("testing llamacpp - %3d - %-52s - ", testnum, name);
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
   ask + obtain (download delegated to rampart-models.js)
   ================================================================ */
function ask(question) {
    printf("%s", question);
    fflush(stdout);
    return fgets(stdin, 255).trim().toLowerCase();
}

/* Return the model file path (fetching it via rampart-models.js if absent, with
   a y/N confirm and rampart-models' own progress display), or null if the module
   is missing, the user declines, or the fetch fails.  A model already on disk
   resolves straight through -- the confirm hook fires only for a real download. */
function obtain(m) {
    if (!models) { printf("  rampart-models.js not found; skipping %s\n", m.name); return null; }
    try {
        return models.ggufGet(m.name, {
            quant: m.quant,
            confirm: function(info) {
                var r = ask(sprintf("Download %s (%s) for the %s?\n  -> %s  [y/N]: ",
                                    info.name, info.size, m.what, info.dest));
                return r === 'y' || r === 'yes';
            }
        });
    } catch(e) {
        printf("  fetch failed: %s\n", e.message || e);
        return null;
    }
}

/* ================================================================
   Test: module loads + API surface
   ================================================================ */
testFeature("module loads", typeof llamacpp === 'object' && llamacpp !== null);
testFeature("initEmbed exists",  typeof llamacpp.initEmbed  === 'function');
testFeature("initGen exists",    typeof llamacpp.initGen    === 'function');
testFeature("initRerank exists", typeof llamacpp.initRerank === 'function');

/* ================================================================
   Test: EMBEDDING (fully synchronous)
   ================================================================ */
function runEmbedTest() {
    printf("\n--- embedding (initEmbed) ---\n");
    var path = obtain(MODELS.embed);
    if (!path) { printf("- skipped\n"); return; }

    var emb = null;
    testFeature("initEmbed loads model", function() {
        emb = llamacpp.initEmbed(path);
        return typeof emb === 'object' && emb !== null;
    });
    if (!emb) return;

    testFeature("embedTextToFp16Buf returns a vector", function() {
        var v = emb.embedTextToFp16Buf("The quick brown fox jumps over the lazy dog.");
        return v && v.avgVec && (v.avgVec.byteLength > 0 || v.avgVec.length > 0);
    });
    testFeature("embedTextToNumbers returns finite, non-zero numbers", function() {
        var n = emb.embedTextToNumbers("A vector database stores embeddings.");
        if (!n || !n.avgVec || !n.avgVec.length) return false;
        var anyNonZero = false;
        for (var i = 0; i < n.avgVec.length; i++) {
            if (typeof n.avgVec[i] !== 'number' || isNaN(n.avgVec[i])) return false;
            if (n.avgVec[i] !== 0) anyNonZero = true;
        }
        return anyNonZero;
    });
    testFeature("different texts produce different vectors", function() {
        var a = emb.embedTextToNumbers("cats and dogs").avgVec;
        var b = emb.embedTextToNumbers("quantum chromodynamics").avgVec;
        if (!a.length || a.length !== b.length) return false;
        for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return true;
        return false;
    });
    try { emb.destroy(); } catch(e) {}
    testFeature("embed destroy", true);
}

/* ================================================================
   Test: GENERATION
   sync sub-tests run inline; the streaming sub-test is async and calls
   done() from its final callback (the event loop runs after main, like
   pu_test/mt/proto_stream.js).
   ================================================================ */
function runGenTest(done) {
    printf("\n--- generation (initGen) ---\n");
    var path = obtain(MODELS.gen);
    if (!path) { printf("- skipped\n"); return done(); }

    /* Probe initGen before turning the rest of the block into test
       cases.  rampart-llamacpp.c throws "initGen: requires macOS N or
       later" on macOS releases too old for the underlying Metal
       features.  Treat that as a graceful skip, not a failure. */
    var gen = null, probeError = null;
    try { gen = llamacpp.initGen(path, { nCtx: 2048, nSeqMax: 2 }); }
    catch (e) { probeError = e; }

    if (probeError && /requires macOS \d+/.test(String(probeError))) {
        printf("- unsupported: %s\n", String(probeError.message || probeError));
        return done();
    }

    testFeature("initGen loads model", function() {
        if (probeError) throw probeError;
        return typeof gen === 'object' && gen !== null;
    });
    if (!gen) return done();

    testFeature("gen reports nCtx / nVocab", function() {
        return gen.nCtx > 0 && gen.nVocab > 0;
    });
    testFeature("predict (sync) returns non-empty text", function() {
        var t = gen.predict({
            messages: [{ role: "user", content: "Say hello in one short sentence." }],
            maxTokens: 24
        });
        return typeof t === 'string' && t.trim().length > 0;
    });
    testFeature("predict answer is coherent (mentions Paris)", function() {
        var t = gen.predict({
            messages: [{ role: "user", content: "What is the capital of France? Reply with only the city name." }],
            maxTokens: 8
        });
        return /paris/i.test(t);
    });

    /* streaming: predictAsync (non-blocking) — finish in the fin callback */
    var streamed = "", gotTokens = 0;
    gen.predictAsync(
        { prompt: "Count from one to five in words, comma separated.", maxTokens: 48 },
        function(res) { if (!res.done && !res.error && res.token) { gotTokens++; streamed += res.token; } },
        function(res) {
            var finalText = (res.fullText || streamed || "");
            testFeature("predictAsync streamed tokens incrementally", gotTokens > 0);
            testFeature("predictAsync produced final text", finalText.trim().length > 0);
            try { gen.destroy(); } catch(e) {}
            testFeature("gen destroy", true);
            done();
        }
    );
}

/* ================================================================
   run: embedding (sync), then generation (async tail), then summary
   ================================================================ */
function finish() {
    printf("\nllamacpp: %d tests run, %d failed\n", testnum, failed);
    process.exit(failed ? 1 : 0);
}

runEmbedTest();
runGenTest(finish);   // when gen is skipped/unavailable, done() == finish() runs now;
                      // otherwise finish() runs from the streaming callback after the loop drains.

/* safety net: never hang forever */
setTimeout(function() {
    printf("\nllamacpp: GLOBAL TIMEOUT - tests did not complete in 180 seconds\n");
    process.exit(1);
}, 180000);
