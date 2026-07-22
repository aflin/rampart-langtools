#!/usr/bin/env rampart
/* rampart-clip test suite.
 *
 * Exercises CLIP two-tower embedding (image + text in one shared vector space):
 * the mechanics (dimension, the fp16/fp32/Numbers image+text forms, similarity),
 * a rampart.thread cross-thread check -- the model object is SHARED into a worker
 * (thread-copyable handle, refcounted shared weights) and must embed there
 * identically -- and a cross-modal RETRIEVAL check: a text query for horses must
 * rank the horse photos above the flowers, and vice-versa.
 *
 * The model (a CLIP GGUF) is located/fetched via rampart-models.js into the
 * standard ~/.rampart/models/clip/ location; you are asked before it downloads
 * (decline and the whole suite prints "- skipped").  The photos live in
 * test_images/ next to this script (installed to <prefix>/test/test_images/ by
 * `make install` and the docker installs); Public Domain, from free-images.com.
 *
 *   rampart clip-test.js
 */
rampart.globalize(rampart.utils);

/* locate rampart-clip.so: local build dirs first (test a fresh build without
   installing), then the installed module -- same as llamacpp-test / onnx-test.
   The resolved name is reused by worker threads (CLIP_FROM). */
var CLIP_FROM = null;
function loadModule() {
    var sp = process.scriptPath;
    var tries = [
        sp + '/build/rampart-clip',      sp + '/build-cpu/rampart-clip',
        sp + '/build-gpu/rampart-clip',  sp + '/build_cpu/rampart-clip',
        sp + '/build_gpu/rampart-clip',  sp + '/modules/rampart-clip',
        'rampart-clip'
    ];
    for (var i = 0; i < tries.length; i++) {
        try { var m = require(tries[i]); CLIP_FROM = tries[i]; return m; } catch(e) {}
    }
    printf("ERROR: could not load rampart-clip.so from any of:\n  %s\n", tries.join('\n  '));
    process.exit(1);
}
var clip = loadModule();

/* rampart-models.js: beside this script in the source tree, or installed. */
function loadModels() {
    var sp = process.scriptPath;
    try { return require(sp + '/rampart-models.js'); }
    catch (e) { try { return require('rampart-models'); } catch (e2) { return null; } }
}
var models = loadModels();
/* small, fast, cleanest text<->image separation of the catalog's CLIP models */
var MODEL = { name: 'clip-vit-b-32-laion', quant: 'Q4_0', what: 'CLIP image+text test' };

/* photos next to this script: test_images/horses/*.jpg, test_images/flowers/*.jpg
   (source tree, or <prefix>/test/test_images/ once installed) */
var IMGDIR = process.scriptPath + '/test_images';

/* ================================================================
   test harness (same shape as llamacpp-test.js / onnx-test.js)
   ================================================================ */
var testnum = 0, failed = 0;
function testFeature(name, test) {
    var error = false;
    testnum++;
    if (typeof test == 'function') {
        try { test = test(); } catch(e) { error = e; test = false; }
    }
    printf("testing clip - %3d - %-58s - ", testnum, name);
    fflush(stdout);
    if (test) printf("passed\n");
    else { printf(">>>>> FAILED <<<<<\n"); failed++; if (error) console.log(error); }
}

/* stdin/stdout are bare globals in current rampart; older builds expose them
   on rampart.utils only */
var STDIN  = (typeof stdin  !== 'undefined') ? stdin  : rampart.utils.stdin;
var STDOUT = (typeof stdout !== 'undefined') ? stdout : rampart.utils.stdout;
function ask(q) { printf("%s", q); fflush(STDOUT); var a = fgets(STDIN, 255); return a ? a.trim().toLowerCase() : ''; }

/* ask, then fetch the model via rampart-models.js (like llamacpp-test.js) */
function obtain() {
    if (!models) { printf("  rampart-models.js not found; skipping\n"); return null; }
    try {
        return models.ggufGet(MODEL.name, {
            quant: MODEL.quant,
            confirm: function(info) {
                var r = ask(sprintf("Download %s (%s) for the %s?\n  -> %s  [y/N]: ",
                                    info.name, info.size, MODEL.what, info.dest));
                return r === 'y' || r === 'yes';
            }
        });
    } catch(e) { printf("  fetch failed: %s\n", e.message || e); return null; }
}

/* labeled photo set from test_images/<label>/*.jpg */
function loadImages() {
    var out = [];
    ['horses', 'flowers'].forEach(function(label) {
        var d = IMGDIR + '/' + label;
        if (!stat(d)) return;
        readDir(d).filter(function(f){ return /\.jpe?g$/i.test(f); }).sort()
            .forEach(function(f){ out.push({ label: label, path: d + '/' + f }); });
    });
    return out;
}
var IMAGES = loadImages();
function imagesOf(label) { return IMAGES.filter(function(im){ return im.label === label; }); }

/* ================================================================
   Test: module loads + API surface
   ================================================================ */
testFeature("module loads", typeof clip === 'object' && clip !== null);
testFeature("initEmbed exists", typeof clip.initEmbed === 'function');
testFeature("load alias exists", typeof clip.load === 'function' && clip.load === clip.initEmbed);

/* ================================================================
   run
   ================================================================ */
var path = obtain();
var model = null;

function runMechanics() {
    printf("\n--- mechanics (load / embed / similarity) ---\n");
    if (!path)          { printf("- skipped: no model\n");  return; }
    if (!IMAGES.length) { printf("- skipped: no test_images beside the script\n"); return; }
    var img = IMAGES[0].path;

    var dim = 0;
    testFeature("initEmbed returns a model object", function() {
        model = clip.initEmbed(path);
        return typeof model === 'object' && model !== null;
    });
    if (!model) return;
    testFeature("dimension is a positive number", function() { dim = model.dimension; return dim > 0; });

    var iv16, iv32, ivn;
    testFeature("embedImageToFp16Buf -> dim*2 bytes", function() {
        iv16 = model.embedImageToFp16Buf(img); return iv16.byteLength === dim * 2;
    });
    testFeature("embedImageToFp32Buf -> dim*4 bytes", function() {
        iv32 = model.embedImageToFp32Buf(img); return iv32.byteLength === dim * 4;
    });
    testFeature("embedImageToNumbers -> dim finite numbers", function() {
        ivn = model.embedImageToNumbers(img);
        return ivn.length === dim && isFinite(ivn[0]) && ivn[0] !== 0;
    });
    testFeature("image vector is L2-normalized", function() {
        var s = 0; for (var i = 0; i < ivn.length; i++) s += ivn[i]*ivn[i];
        return Math.abs(Math.sqrt(s) - 1) < 1e-2;
    });
    testFeature("embedTextToNumbers -> dim finite numbers", function() {
        var tv = model.embedTextToNumbers("a photo"); return tv.length === dim && isFinite(tv[0]);
    });
    testFeature("similarity(v,v) == 1.0", function() {
        return Math.abs(model.similarity(iv32, iv32) - 1) < 1e-3;
    });
    testFeature("similarity(fp16) matches similarity(fp32)", function() {
        var t16 = model.embedTextToFp16Buf("a photo"), t32 = model.embedTextToFp32Buf("a photo");
        return Math.abs(model.similarity(iv16, t16) - model.similarity(iv32, t32)) < 1e-2;
    });
    testFeature("errMsg is undefined on success", function() { return model.errMsg === undefined; });
}

/* embedImage* also accepts the image BYTES (a Buffer), not just a path -- the
   entry point rampart-sql uses for images stored in a varbyte column.  The single
   most important assertion: the SAME image embedded from a path and from its bytes
   must produce the SAME vector (proves the file and buffer paths share one
   decode-onward code path and haven't drifted). */
function runImageBuffer() {
    printf("\n--- image from a Buffer (bytes) ---\n");
    if (!model || !IMAGES.length) { printf("- skipped\n"); return; }
    var img = IMAGES[0].path;
    var bytes = readFile(img);      /* the raw JPEG bytes, as a Buffer */
    var dim = model.dimension;

    testFeature("embedImageToFp32Buf(buffer) -> dim*4 bytes", function() {
        return model.embedImageToFp32Buf(bytes).byteLength === dim * 4;
    });
    testFeature("path and buffer embed to the SAME vector (<1e-5)", function() {
        var a = new Float32Array(model.embedImageToFp32Buf(img));      // from path
        var b = new Float32Array(model.embedImageToFp32Buf(bytes));    // from bytes
        if (a.length !== b.length) throw new Error("length mismatch");
        var max = 0; for (var i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i]-b[i]));
        if (!(max < 1e-5)) throw new Error("max |a-b| = " + max);
        return true;
    });
    testFeature("embedImageToNumbers(buffer) matches path", function() {
        var a = model.embedImageToNumbers(img), b = model.embedImageToNumbers(bytes);
        return Math.abs(a[0]-b[0]) < 1e-5 && a.length === b.length;
    });
    testFeature("empty buffer throws", function() {
        try { model.embedImageToFp32Buf(new Uint8Array(0)); return false; }
        catch(e) { return /empty|buffer/i.test(e.message); }
    });
    testFeature("non-image buffer throws (not a crash)", function() {
        try { model.embedImageToFp32Buf(new Uint8Array([1,2,3,4,5])); return false; }
        catch(e) { return /decode|format|image/i.test(e.message); }
    });
    testFeature("missing path throws 'cannot open'", function() {
        try { model.embedImageToFp32Buf("/no/such/clip-test-missing.jpg"); return false; }
        catch(e) { return /cannot open|no such/i.test(e.message); }
    });
    testFeature("non-image file throws 'decode' (distinct from missing)", function() {
        var f = ((typeof getenv === 'function' && getenv('TMPDIR')) || '/tmp') + '/clip-test-notimage.txt';
        fwrite(f, "this is not an image");
        try { model.embedImageToFp32Buf(f); return false; }
        catch(e) { return /decode|unrecognized|format/i.test(e.message); }
    });
}

/* the key capability the modern module adds: the handle survives a thread copy
   (shared refcounted weights, per-thread compute ctx) -- the old design threw. */
function runThreadCopy() {
    printf("\n--- rampart.thread (shared handle across a worker) ---\n");
    if (!model || !IMAGES.length) { printf("- skipped\n"); return; }
    var img = IMAGES[0].path;
    var vmain = model.embedImageToNumbers(img);
    var t = new rampart.thread();
    t.exec(function(a) {
        try {
            var v = a.m.embedImageToNumbers(a.img);        // SAME shared handle
            var tv = a.m.embedTextToNumbers("a photo");
            rampart.thread.put('r', { v0: v[0], dim: v.length, tdim: tv.length });
        } catch(e) { rampart.thread.put('r', { err: e.message }); }
    }, { m: model, img: img });
    var r = rampart.thread.get('r', 120000);
    testFeature("worker embeds with the shared handle (no throw)", function() {
        if (r && r.err) throw new Error(r.err);
        return r && r.dim === model.dimension;
    });
    testFeature("worker image vector matches main", function() {
        return r && Math.abs(r.v0 - vmain[0]) < 1e-4;
    });
    testFeature("main still works after worker", function() {
        return model.embedImageToNumbers(img).length === model.dimension;
    });
    t.terminate();
}

/* cross-modal retrieval: a text query for one category must rank ALL of that
   category's photos above every photo of the other category. */
function runSemantic() {
    printf("\n--- cross-modal retrieval (text -> image) ---\n");
    if (!model)         { printf("- skipped: no model\n"); return; }
    var horses = imagesOf('horses'), flowers = imagesOf('flowers');
    if (!horses.length || !flowers.length) { printf("- skipped: need horse + flower photos\n"); return; }

    /* embed each photo once */
    var vecs = {};
    IMAGES.forEach(function(im){ vecs[im.path] = model.embedImageToFp32Buf(im.path); });

    [ { text: "horses running on a beach",   want: "horses"  },
      { text: "pink flowers blooming in spring", want: "flowers" } ].forEach(function(q) {
        var qv = model.embedTextToFp32Buf(q.text);
        var scored = IMAGES.map(function(im){
            return { label: im.label, name: im.path.split('/').pop(), s: model.similarity(vecs[im.path], qv) };
        }).sort(function(a,b){ return b.s - a.s; });
        var wantMin = Math.min.apply(null, scored.filter(function(x){return x.label===q.want;}).map(function(x){return x.s;}));
        var otherMax = Math.max.apply(null, scored.filter(function(x){return x.label!==q.want;}).map(function(x){return x.s;}));
        testFeature(sprintf('query "%s" ranks %s above the rest', q.text, q.want), function() {
            if (!(wantMin > otherMax))
                throw new Error(sprintf("%s min %.3f <= other max %.3f  [%s]", q.want, wantMin, otherMax,
                    scored.map(function(x){return x.name+" "+x.s.toFixed(2);}).join(", ")));
            return true;
        });
    });
}

runMechanics();
runImageBuffer();
runThreadCopy();
runSemantic();
if (model) model.destroy();

printf("\nclip: %d tests run, %d failed\n", testnum, failed);
process.exit(failed ? 1 : 0);
