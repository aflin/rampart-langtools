#!/usr/bin/env rampart
/* rampart-llamacpp test suite.
 *
 * Exercises the critical functions of the embedding path (initEmbed) and the
 * text-generation path (initGen). Each needs a model file; this script offers
 * to download each from HuggingFace into /tmp (so nothing is left in the
 * project tree). Decline a download and that section prints "- skipped".
 *
 *   rampart llamacpp-test.js
 */
rampart.globalize(rampart.utils);
load.curl;

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

/* ---- models: downloaded to /tmp, source URL on HuggingFace ---------------- */
var MODELS = {
    embed: {
        name: 'all-minilm-l6-v2_f16.gguf',
        file: '/tmp/all-minilm-l6-v2_f16.gguf',
        url:  'https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf',
        size: '~45 MB',
        what: 'embedding test'
    },
    gen: {
        name: 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
        file: '/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf',
        url:  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
        size: '~470 MB',
        what: 'generation / inference test'
    }
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
   ask + download (synchronous curl.fetch; progress per downloadLangDeriv.js)
   ================================================================ */
function ask(question) {
    printf("%s", question);
    fflush(stdout);
    return fgets(stdin, 255).trim().toLowerCase();
}

/* Return the model path if available (already on disk or downloaded), else null
   if the user declined or the download failed. */
function obtain(m) {
    var st = stat(m.file);
    if (st && st.size > 0) {
        printf("  using existing %s\n", m.file);
        return m.file;
    }

    var r = ask(sprintf("Download %s (%s) for the %s?\n  It will be saved to %s  [y/N]: ",
                        m.name, m.size, m.what, m.file));
    if (r !== 'y' && r !== 'yes') return null;

    printf("Downloading %s\n  -> %s\n", m.url, m.file);
    var f = fopen(m.file, 'w+');
    var nchunks = 0, status = -1;
    try {
        /* synchronous (blocking) fetch; chunkCallback streams the body to disk.
           A normal `callback` is required whenever chunkCallback is used — fetch
           still blocks, the callback just delivers the final result. */
        curl.fetch(m.url, {
            location:     true,    // follow HF redirect to the CDN
            returnText:   false,
            skipFinalRes: true,    // don't buffer the whole file in memory
            chunkCallback: function(res) { f.fprintf('%s', res.body); },
            progressCallback: function(res) {
                if (nchunks++ % 30) return;
                var tot = res.progress, rate = tot / (res.totalTime * 1024), unit = "KB/s";
                if (rate > 1024) { rate /= 1024; unit = "MB/s"; }
                if (res.expectedTotal != -1)
                    printf("\r    %.1f%%  %d / %d bytes  (%.2f %s)   ",
                           100 * tot / res.expectedTotal, tot, res.expectedTotal, rate, unit);
                else
                    printf("\r    %d bytes  (%.2f %s)   ", tot, rate, unit);
                fflush(stdout);
            },
            callback: function(res) { status = res.status; }
        });
    } catch(e) {
        f.fclose();
        printf("\n  download error: %s\n", e.message || e);
        return null;
    }
    f.fclose();
    var st2 = stat(m.file);
    if (!st2 || st2.size === 0) {
        printf("\n  download produced no data (status %d)\n", status);
        return null;
    }
    printf("\r    done: %d bytes%20s\n", st2.size, "");
    return m.file;
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

    var gen = null;
    testFeature("initGen loads model", function() {
        gen = llamacpp.initGen(path, { nCtx: 2048, nSeqMax: 2 });
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
