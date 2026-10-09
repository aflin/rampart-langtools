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
    /* LTMOD lets the suite run against an arbitrary build dir (e.g. build-cpu/,
       whose name the list below does not match) without copying the .so around */
    if (process.env.LTMOD) { try { return require(process.env.LTMOD); } catch(e) {} }
    var tries = [
        sp + '/build/rampart-llamacpp',
        sp + '/build-cpu/rampart-llamacpp',
        sp + '/build-gpu/rampart-llamacpp',
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
    gen:   { name: 'qwen2.5-0.5b-instruct', quant: 'Q4_K_M', what: 'generation / inference test' },
    media: { name: 'embeddinggemma-2',      quant: 'Q8_0',   what: 'multimodal (image/audio) embedding test' }
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
   Test: MULTIMODAL EMBEDDING (initEmbed {mmproj} + embedMedia*)
   Media are generated here (solid-colour PPM images, a WAV tone), so the
   only downloads are the model and its projector.  Also covers the handle
   in a worker thread and after fork(): on CPU the child gets the same
   vector; with a GPU backend the child must get the fork refusal, never hang.
   ================================================================ */
function ppm(r, g, b) {                 /* 64x64 solid-colour binary PPM */
    var hdr = "P6\n64 64\n255\n", n = 64 * 64, a = new Uint8Array(hdr.length + n * 3);
    for (var i = 0; i < hdr.length; i++) a[i] = hdr.charCodeAt(i);
    for (i = 0; i < n; i++) { a[hdr.length + i*3] = r; a[hdr.length + i*3 + 1] = g; a[hdr.length + i*3 + 2] = b; }
    return a;
}
function wavTone(hz, secs) {            /* 16 kHz mono 16-bit PCM sine */
    var sr = 16000, n = sr * secs, a = new Uint8Array(44 + n * 2), dv = new DataView(a.buffer);
    function str(o, s) { for (var i = 0; i < s.length; i++) a[o + i] = s.charCodeAt(i); }
    str(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); str(8, "WAVEfmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
    dv.setUint16(34, 16, true); str(36, "data"); dv.setUint32(40, n * 2, true);
    for (var i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.round(12000 * Math.sin(2 * Math.PI * hz * i / sr)), true);
    return a;
}
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

function runMediaTest() {
    printf("\n--- multimodal embedding (initEmbed mmproj + embedMedia) ---\n");
    var path = obtain(MODELS.media);
    if (!path) { printf("- skipped\n"); return; }
    var mmproj = null;
    try {
        mmproj = models.mmprojGet(MODELS.media.name, {
            confirm: function(info) {
                var r = ask(sprintf("Download %s projector %s (%s) for the %s?\n  -> %s  [y/N]: ",
                                    info.name, info.quant, info.size, MODELS.media.what, info.dest));
                return r === 'y' || r === 'yes';
            }
        });
    } catch(e) { printf("  fetch failed: %s\n", e.message || e); }
    if (!mmproj) { printf("- skipped\n"); return; }

    var Q = "task: search result | query: ";
    var red = ppm(220, 20, 20), blue = ppm(20, 40, 220);
    var e = null, vred = null;
    testFeature("initEmbed loads model + projector", function() {
        /* imageTokens 70 (the model's minimum; default 280) keeps the CPU run short --
         * these tests check mechanics and red-vs-blue, not fine detail */
        e = llamacpp.initEmbed(path, {mmproj: mmproj, imageTokens: 70});
        return !!(e && e.media && e.media.vision && e.media.audio && e.media.imageTokens === 70);
    });
    if (!e) return;
    testFeature("image (Buffer) -> 768d unit vector", function() {
        var r = e.embedMediaToNumbers({image: red});
        vred = r.avgVec;
        return vred.length === 768 && Math.abs(dot(vred, vred) - 1) < 1e-3 && r.nTokens > 0;
    });
    testFeature("image by path == same image as Buffer", function() {
        var f = "/tmp/llamacpp-test-red-" + process.getpid() + ".ppm";
        writeFile(f, red);
        var v = e.embedMediaToNumbers({image: f}).avgVec;
        try { rmFile(f); } catch(x) {}
        return dot(v, vred) > 0.9999;
    });
    testFeature("image <-> caption: red and blue each nearest their own caption", function() {
        var vb = e.embedMediaToNumbers({image: blue}).avgVec;
        var cr = e.embedTextToNumbers(Q + "a solid red image").avgVec;
        var cb = e.embedTextToNumbers(Q + "a solid blue image").avgVec;
        return dot(vred, cr) > dot(vred, cb) && dot(vb, cb) > dot(vb, cr);
    });
    testFeature("audio (WAV Buffer) -> 768d vector", function() {
        var r = e.embedMediaToNumbers({audio: wavTone(440, 2)});
        return r.avgVec.length === 768 && r.nTokens > 0;
    });
    testFeature("text with <|image|> placeholder + image", function() {
        var r = e.embedMediaToNumbers({text: "A colour swatch: <|image|>", image: red});
        return r.avgVec.length === 768 && dot(r.avgVec, vred) < 0.9999;
    });
    testFeature("Fp32Buf / Fp16Buf sizes", function() {
        return e.embedMediaToFp32Buf({image: red}).avgVec.byteLength === 3072 &&
               e.embedMediaToFp16Buf({image: red}).avgVec.byteLength === 1536;
    });
    testFeature("error: placeholder count != media count", function() {
        try { e.embedMediaToNumbers({text: "<|image|> <|image|>", image: red}); return false; }
        catch(x) { return /placeholder/.test(x.message); }
    });
    testFeature("error: audio given as an image", function() {
        try { e.embedMediaToNumbers({image: wavTone(440, 1)}); return false; }
        catch(x) { return /decodes as audio/.test(x.message); }
    });
    /* fork: the child may continue on CPU; with a GPU backend it must refuse
     * (both a pre-fork handle and a new load), never hang */
    var gpu = !!llamacpp.embedDefaults().gpuInUse;
    var ff = "/tmp/llamacpp-test-fork-" + process.getpid();
    writeFile(ff, "");
    var pid = -1;
    for (var w = 0; w < 50 && pid < 0; w++) {   /* rampart won't fork while the worker above is still closing */
        try { pid = fork(); } catch(x) { if (!/active threads/.test(x.message)) throw x; sleep(0.1); }
    }
    if (pid < 0) { testFeature("fork child (could not fork: worker thread still active)", false); try { e.destroy(); } catch(x) {} return; }
    if (pid === 0) {
        var out = [];
        try { out.push(Math.abs(e.embedMediaToNumbers({image: red}).avgVec[0] - vred[0]) < 1e-5 ? "same" : "different"); }
        catch(x) { out.push(/fork/.test(x.message) ? "refused" : "threw " + x.message); }
        try { out.push(llamacpp.initEmbed(path, {mmproj: mmproj, imageTokens: 70}).embedMediaToNumbers({image: red}).avgVec.length === 768 ? "loaded" : "bad"); }
        catch(x) { out.push(/fork/.test(x.message) ? "refused" : "threw " + x.message); }
        writeFile(ff, out.join(","));
        process.exit(0);
    }
    var res = "";
    for (var i = 0; i < 60 && !res; i++) { sleep(0.25); try { res = readFile(ff, true); } catch(x) {} }
    if (!res) kill(pid, 9);
    try { rmFile(ff); } catch(x) {}
    testFeature("fork child: " + (gpu ? "GPU -> refused, no hang" : "CPU -> same vector, new load ok"), function() {
        return res === (gpu ? "refused,refused" : "same,loaded") || (printf("    child: %s\n", res || "HUNG"), false);
    });
    /* the worker test runs AFTER the fork: a just-closed worker can still hold
     * rampart's thread lock while fork() already allows forking, and the child
     * would inherit it locked (see rampart/claude-work/fork-thread-lock-race.md) */
    testFeature("handle used in a worker thread: same vector", function() {
        var t = new rampart.thread();
        t.exec(function(a) {
            try { rampart.thread.put("mt", a.h.embedMediaToNumbers({image: a.img}).avgVec[0]); }
            catch(x) { rampart.thread.put("mt", "threw: " + x.message); }
        }, {h: e, img: red});
        var r = rampart.thread.get("mt", 60000);
        t.close();
        return typeof r === 'number' && Math.abs(r - vred[0]) < 1e-5;
    });
    try { e.destroy(); } catch(x) {}
    testFeature("media handle destroy", true);
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
   Test: tool calling (initGen + tools)
   ================================================================ */
var TOOL_MARKUP = /<tool_call|<tool_code|functools|<\|tool/;
var TOOLS = [{
    type: "function",
    function: {
        name: "keep_search",
        description: "Search the document collection for a phrase.",
        parameters: { type: "object",
                      properties: { query: { type: "string" } },
                      required: ["query"] }
    }
}];

function runToolTest(done) {
    printf("\n--- tool calling (initGen + tools) ---\n");
    var path = obtain(MODELS.gen);
    if (!path) { printf("- skipped\n"); return done(); }

    /* No `threads` here on purpose: the module now defaults to libcommon's
       machine heuristic, so this exercises the real default. */
    var gen = null;
    try { gen = llamacpp.initGen(path, { nCtx: 2048, nSeqMax: 2 }); }
    catch (e) { printf("- unsupported: %s\n", String(e.message || e)); return done(); }

    /* Capability reporting: no upstream query exists, so the module probes the
       template at initGen time. */
    testFeature("gen reports supportsTools / chatFormat", function() {
        return typeof gen.supportsTools === 'boolean' && typeof gen.chatFormat === 'string';
    });
    if (!gen.supportsTools) {
        printf("  model's template has no tool support; tool tests skipped\n");
        testFeature("tools on a toolless template throw", function() {
            try { gen.predict({ messages:[{role:"user",content:"hi"}], tools: TOOLS, maxTokens: 8 }); }
            catch (e) { return true; }
            return false;
        });
        try { gen.destroy(); } catch(e) {}
        return done();
    }

    var r = gen.predict({
        messages: [{ role: "user", content: "Search the collection for documents about HTTP/1.1." }],
        tools: TOOLS, maxTokens: 128, temp: 0
    });

    testFeature("predict with tools returns a structured Object", function() {
        return typeof r === 'object' && r !== null && typeof r.fullText === 'string';
    });
    testFeature("a tool call is produced", function() { return !!(r.toolCalls && r.toolCalls.length); });
    testFeature("call names the supplied tool", function() {
        return r.toolCalls && r.toolCalls[0].function.name === 'keep_search';
    });
    testFeature("call arguments parse as JSON", function() {
        if (!r.toolCalls) return false;
        var a = JSON.parse(r.toolCalls[0].function.arguments);
        return a !== null && typeof a === 'object';
    });
    testFeature("call has an id and type", function() {
        return r.toolCalls && typeof r.toolCalls[0].id === 'string' && r.toolCalls[0].id.length > 0
               && r.toolCalls[0].type === 'function';
    });
    testFeature("finishReason is tool_calls", function() { return r.finishReason === 'tool_calls'; });
    /* the regression this whole feature exists to prevent */
    testFeature("no tool markup leaks into fullText", function() {
        return !TOOL_MARKUP.test(r.fullText || '');
    });

    testFeature("tool loop closes (assistant+tool turns round-trip)", function() {
        if (!r.toolCalls) return false;
        var r2 = gen.predict({ messages: [
            { role: "user", content: "Search the collection for documents about HTTP/1.1." },
            { role: "assistant", content: "", tool_calls: r.toolCalls },
            { role: "tool", tool_call_id: r.toolCalls[0].id, name: "keep_search",
              content: "Found 3 documents mentioning HTTP/1.1." }
        ], tools: TOOLS, maxTokens: 48, temp: 0 });
        return !r2.toolCalls && (r2.fullText || '').length > 0;
    });

    /* With toolChoice:"none" upstream renders the tools but disables tool
       PARSING, so the contract we can assert is that no structured call comes
       back.  A small model that parrots the template's own example emits it as
       ordinary content; that is model quality, and per the design notes we do
       not paper over it with a regex in the module. */
    testFeature("toolChoice:'none' returns no structured calls", function() {
        var r3 = gen.predict({
            messages: [{ role: "user", content: "Search the collection for documents about HTTP/1.1." }],
            tools: TOOLS, toolChoice: "none", maxTokens: 32, temp: 0 });
        return !r3.toolCalls && typeof r3.fullText === 'string';
    });

    /* toolChoice:"required" is passed through as COMMON_CHAT_TOOL_CHOICE_REQUIRED
       and upstream does build a non-lazy tool-calls grammar for it -- but as of
       b10446 that grammar is not enforced during sampling (reproducible with no
       rampart code at all).  Assert only that the request is accepted, so this
       test starts telling us something the day upstream enforces it. */
    testFeature("toolChoice:'required' is accepted", function() {
        var r4 = gen.predict({ messages: [{ role: "user", content: "Say hello." }],
                               tools: TOOLS, toolChoice: "required", maxTokens: 32, temp: 0 });
        return typeof r4 === 'object' && r4 !== null;
    });

    testFeature("without tools, predict still returns a String", function() {
        var r5 = gen.predict({ messages: [{ role: "user", content: "Say hi." }], maxTokens: 8 });
        return typeof r5 === 'string';
    });

    /* ---- reasoning separation (no tools involved) --------------------------
       The parser has to run for callers with no tools too: where a format
       carries reasoning as a channel rather than a <think> span, only
       llama.cpp's parser knows where it ends, so a caller that must
       machine-read the reply cannot strip it itself. */
    var REASON_Q = [{ role: "user",
        content: "Which mention email? 1 HTTP spec, 4 SMTP spec, 7 IMAP spec. Only numbers, comma separated." }];

    testFeature("default (no tools, no reasoning) still returns a String", function() {
        var d = gen.predict({ messages: REASON_Q, maxTokens: 24, temp: 0 });
        return typeof d === 'string';
    });

    var rr = gen.predict({ messages: REASON_Q, reasoning: true, maxTokens: 64, temp: 0 });
    testFeature("reasoning:true returns a structured Object", function() {
        return typeof rr === 'object' && rr !== null
               && typeof rr.fullText === 'string' && typeof rr.finishReason === 'string';
    });
    /* Only a thinking model produces any; on a plain model the correct result
       is simply no reasoning and an untouched answer. */
    testFeature("reasoning is separated, not left in fullText", function() {
        if (!rr.reasoning || !rr.reasoning.length) return true;   /* not a thinking model */
        return !/<think|<\|channel|<\|start\|>/i.test(rr.fullText || '');
    });

    testFeature("gen reports supportsThinkingToggle", function() {
        return typeof gen.supportsThinkingToggle === 'boolean';
    });
    testFeature("thinking:false is accepted", function() {
        var nt = gen.predict({ messages: REASON_Q, reasoning: true, thinking: false,
                               maxTokens: 32, temp: 0 });
        return typeof nt === 'object' && typeof nt.fullText === 'string';
    });

    /* streaming must deliver content only -- never the tool-call markup */
    var streamed = "", streamedReasoning = "";
    gen.predictAsync(
        { messages: [{ role: "user", content: "Search the collection for documents about HTTP/1.1." }],
          tools: TOOLS, maxTokens: 64, temp: 0 },
        function(t) { if (t.done || t.error || !t.token) return;
                      if (t.reasoning) streamedReasoning += t.token; else streamed += t.token; },
        function(res) {
            testFeature("streamed tokens contain no tool markup", !TOOL_MARKUP.test(streamed));
            testFeature("streamed text equals fullText", streamed === (res.fullText || ''));
            testFeature("reasoning tokens are flagged, not mixed into content",
                        streamedReasoning === (res.reasoning || ''));
            testFeature("streamed request still reports the call", !!(res.toolCalls && res.toolCalls.length));
            try { gen.destroy(); } catch(e) {}
            done();
        }
    );
}

/* ================================================================
   run: embedding (sync), then generation, then tools (async tails)
   ================================================================ */
function finish() {
    printf("\nllamacpp: %d tests run, %d failed\n", testnum, failed);
    process.exit(failed ? 1 : 0);
}

runEmbedTest();
runMediaTest();
runGenTest(function(){ runToolTest(finish); });
                      // when gen is skipped/unavailable, done() runs now; otherwise
                      // it runs from the streaming callback after the loop drains.

/* safety net: never hang forever */
setTimeout(function() {
    printf("\nllamacpp: GLOBAL TIMEOUT - tests did not complete in 600 seconds\n");
    process.exit(1);
}, 600000);
