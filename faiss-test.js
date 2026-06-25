#!/usr/bin/env rampart
/* rampart-faiss test suite.
 *
 * Exercises the core of rampart-faiss on the CPU and, when a CUDA build is
 * loaded and a GPU is actually present, on the GPU as well.  It does NOT try
 * every index the factory can build -- just a few useful in-memory ones:
 *
 *     Flat          (exact, auto-id, fp32 vectors)
 *     IDMap2,Flat   (arbitrary 64-bit ids, fp16 vectors)
 *     IVF<n>,Flat   (requires training -> exercises the trainer)
 *
 * plus a save / openIndexFromFile round-trip and, if available, a clone-to-GPU
 * search.  No model or network access is needed: vectors are synthetic, unit-
 * normalized, and built from a deterministic PRNG so a failure reproduces.
 *
 *   rampart faiss-test.js
 *
 * (Modeled on llamacpp-test.js.)
 */
rampart.globalize(rampart.utils);

/* ----------------------------------------------------------------------------
   Load the module.  Try the bare name first, then the CUDA build, then the CPU
   build -- all as plain module names (no path, no './').  That way require()
   finds a local .so when run from a build directory and finds the installed
   module in <rampart>/modules when installed.
   ---------------------------------------------------------------------------- */
function loadFaiss() {
    var tries = ['rampart-faiss', 'rampart-faiss_cuda', 'rampart-faiss_cpu'];
    for (var i = 0; i < tries.length; i++) {
        try { return require(tries[i]); } catch(e) {}
    }
    printf("ERROR: could not load rampart-faiss from any of:\n  %s\n", tries.join('\n  '));
    process.exit(1);
}
var faiss = loadFaiss();

/* ================================================================
   test harness (same shape as llamacpp-test.js)
   ================================================================ */
var testnum = 0, failed = 0;

function testFeature(name, test) {
    var error = false;
    testnum++;
    if (typeof test == 'function') {
        try { test = test(); }
        catch(e) { error = e; test = false; }
    }
    printf("testing faiss - %3d - %-69s - ", testnum, name);
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
   synthetic data: deterministic, unit-normalized vectors.

   With the inner-product metric the maximum possible score of a query against
   the set is its own self-similarity (1.0), so searching with an exact copy of
   stored vector k must return id k as the top hit -- a robust, exact check that
   does not depend on the dataset's random layout.
   ================================================================ */
var R   = rampart.vector.raw;          // numbersToF32 / numbersToF16
var DIM = 32;
var N   = 1024;                        // > 39*nlist so IVF training is clean

function unit(a) {
    var s = 0, i;
    for (i = 0; i < a.length; i++) s += a[i] * a[i];
    s = Math.sqrt(s) || 1;
    for (i = 0; i < a.length; i++) a[i] /= s;
    return a;
}
/* simple LCG, seeded per-vector so the whole dataset is reproducible */
function makeVec(seed) {
    var a = [], x = (seed * 2654435761) & 0x7fffffff;
    for (var i = 0; i < DIM; i++) {
        x = (1103515245 * x + 12345) & 0x7fffffff;
        a.push((x / 0x7fffffff) * 2 - 1);
    }
    return unit(a);
}
var DATA = [];                         // DATA[i] = number[] (unit vector)
for (var _i = 0; _i < N; _i++) DATA.push(makeVec(_i + 1));

var f32 = function(arr) { return R.numbersToF32(arr); };
var f16 = function(arr) { return R.numbersToF16(arr); };

var PROBE = 7;                         // the vector we self-search for throughout

/* temp files (cleaned up at the end) */
var TMP      = (process.env.TMPDIR || '/tmp');
var PID      = process.getpid();
var TRAINF   = TMP + '/faiss-test-train.' + PID;
var SAVEF    = TMP + '/faiss-test-index.' + PID + '.faiss';
var SAVEIVF  = TMP + '/faiss-test-ivf.'   + PID + '.faiss';
function cleanup() {
    [TRAINF, SAVEF, SAVEIVF].forEach(function(f) { try { rmFile(f); } catch(e) {} });
}

/* ================================================================
   API surface
   ================================================================ */
testFeature("module loads",            typeof faiss === 'object' && faiss !== null);
testFeature("openFactory exists",       typeof faiss.openFactory === 'function');
testFeature("openIndexFromFile exists", typeof faiss.openIndexFromFile === 'function');

/* ================================================================
   Flat (exact, auto-id, fp32)
   ================================================================ */
function runFlatTest() {
    printf("\n--- Flat (exact, auto-id, fp32) ---\n");
    var idx = null;

    testFeature("openFactory('Flat') builds an index", function() {
        idx = faiss.openFactory("Flat", DIM);
        return idx && idx.settings && idx.settings.dimension === DIM
                   && idx.settings.type === "Flat";
    });
    if (!idx) return null;

    testFeature("addFp32 with auto-id (-1) returns sequential ids", function() {
        for (var i = 0; i < N; i++) {
            var id = idx.addFp32(-1, f32(DATA[i]));
            if (id !== i) return false;             // auto-id starts at 0
        }
        return true;
    });
    testFeature("settings.count reflects inserted rows", function() {
        return idx.settings.count === N;
    });
    testFeature("searchFp32 self-query returns its own id (dist ~1.0)", function() {
        var res = idx.searchFp32(f32(DATA[PROBE]), 5);
        return res.length && res[0].id === PROBE && Math.abs(res[0].distance - 1) < 1e-3;
    });
    testFeature("searchFp32 honors nResults", function() {
        return idx.searchFp32(f32(DATA[PROBE]), 3).length === 3;
    });
    testFeature("a different query returns a different top id", function() {
        var top = idx.searchFp32(f32(DATA[PROBE]), 1)[0].id;
        var other = idx.searchFp32(f32(DATA[PROBE + 100]), 1)[0].id;
        return top !== other;
    });
    return idx;
}

/* ================================================================
   IDMap2,Flat (arbitrary 64-bit ids, fp16)
   ================================================================ */
function runIDMapTest() {
    printf("\n--- IDMap2,Flat (custom ids, fp16) ---\n");
    var idx = null;
    var BASE = 1000000;                              // ids well outside 0..N

    testFeature("openFactory('IDMap2,Flat') reports map=IDMap2", function() {
        idx = faiss.openFactory("IDMap2,Flat", DIM);
        return idx && idx.settings && idx.settings.map === "IDMap2";
    });
    if (!idx) return;

    testFeature("addFp16 stores vectors under caller-supplied ids", function() {
        for (var i = 0; i < N; i++) idx.addFp16(BASE + i, f16(DATA[i]));
        return idx.settings.count === N;
    });
    testFeature("searchFp16 returns the custom id (dist ~1.0, fp16 tol)", function() {
        var res = idx.searchFp16(f16(DATA[PROBE]), 5);
        return res.length && res[0].id === BASE + PROBE
                          && Math.abs(res[0].distance - 1) < 5e-2;   // fp16 precision
    });
}

/* ================================================================
   IVF<n>,Flat (requires training -> exercises the trainer)
   ================================================================ */
function runIVFTest() {
    printf("\n--- IVF,Flat (trained, fp32) ---\n");
    var nlist = 16;
    var idx = null;

    testFeature("openFactory('IVF16,Flat') is untrained and exposes a trainer", function() {
        idx = faiss.openFactory("IVF" + nlist + ",Flat", DIM);
        return idx && typeof idx.trainer === 'function';
    });
    if (!idx || typeof idx.trainer !== 'function') return;

    var trainer = null;
    testFeature("new idx.trainer(file) opens a training spool", function() {
        trainer = new idx.trainer(TRAINF);
        return trainer && typeof trainer.addTrainingfp32 === 'function';
    });
    if (!trainer) return;

    testFeature("addTrainingfp32 + train() trains the index", function() {
        for (var i = 0; i < N; i++) trainer.addTrainingfp32(f32(DATA[i]));
        trainer.train();
        return true;                                 // throws on failure
    });
    testFeature("addFp32 after training, count is correct", function() {
        for (var i = 0; i < N; i++) idx.addFp32(i, f32(DATA[i]));
        return idx.settings.count === N;
    });
    testFeature("searchFp32 (nprobe=nlist) finds the self-vector", function() {
        var res = idx.searchFp32(f32(DATA[PROBE]), 5, nlist);   // nprobe=nlist => exact
        return res.length && res[0].id === PROBE;
    });
    return idx;
}

/* ================================================================
   save / openIndexFromFile round-trip
   ================================================================ */
function runPersistTest(flatIdx, ivfIdx) {
    printf("\n--- save / openIndexFromFile round-trip ---\n");
    if (flatIdx) {
        testFeature("Flat: save() writes the index to disk", function() {
            flatIdx.save(SAVEF);
            var st = stat(SAVEF);
            return st && st.size > 0;
        });
        testFeature("Flat: openIndexFromFile() reloads it (count preserved)", function() {
            var re = faiss.openIndexFromFile(SAVEF);
            return re && re.settings.count === N && re.settings.dimension === DIM;
        });
        testFeature("Flat: reloaded index returns the same top id", function() {
            var re = faiss.openIndexFromFile(SAVEF);
            var res = re.searchFp32(f32(DATA[PROBE]), 1);
            return res.length && res[0].id === PROBE;
        });
    } else {
        printf("- Flat round-trip skipped (no Flat index)\n");
    }

    /* IVF round-trip — openIndexFromFile() opens with IO_FLAG_MMAP, which
     * routes IVF invlists through OnDiskInvertedListsIOHook.  That hook
     * is the one that hits the GCC 8 libstdc++ std::string SSO destructor
     * miscompile on armhf (FAISS issues #1071, #2281, #2955, #3292).
     * Without this case the bug would sit latent in rampart-faiss.so on
     * a Buster Pi — the Flat round-trip above doesn't exercise it. */
    if (ivfIdx) {
        testFeature("IVF: save() writes the index to disk", function() {
            ivfIdx.save(SAVEIVF);
            var st = stat(SAVEIVF);
            return st && st.size > 0;
        });
        testFeature("IVF: openIndexFromFile() reloads it (count preserved)", function() {
            var re = faiss.openIndexFromFile(SAVEIVF);
            return re && re.settings.count === N && re.settings.dimension === DIM;
        });
        testFeature("IVF: reloaded index returns the same top id", function() {
            var re = faiss.openIndexFromFile(SAVEIVF);
            var res = re.searchFp32(f32(DATA[PROBE]), 1);
            return res.length && res[0].id === PROBE;
        });
    } else {
        printf("- IVF round-trip skipped (no IVF index)\n");
    }
}

/* ================================================================
   GPU: only when a CUDA build is loaded (idx.enableGpu present) AND a GPU is
   actually usable at runtime.  A missing GPU/driver makes enableGpu() throw --
   treated as a graceful skip, not a failure.
   ================================================================ */
function runGpuTest() {
    printf("\n--- GPU (clone-to-GPU search) ---\n");

    var probe = faiss.openFactory("Flat", DIM);
    if (typeof probe.enableGpu !== 'function') {
        printf("- skipped: CPU-only build (no enableGpu)\n");
        return;
    }

    /* fresh Flat index with data, then clone to GPU */
    var idx = faiss.openFactory("Flat", DIM);
    for (var i = 0; i < N; i++) idx.addFp32(-1, f32(DATA[i]));

    var enabled = false, gpuErr = null;
    try { enabled = idx.enableGpu(); }
    catch (e) { gpuErr = e; }

    if (!enabled) {
        printf("- skipped: no usable GPU at runtime%s\n",
               gpuErr ? (": " + (gpuErr.message || gpuErr)) : "");
        return;
    }

    testFeature("enableGpu() reports the index is on the GPU", function() {
        return idx.settings.onGpu === true;
    });
    testFeature("GPU searchFp32 finds the self-vector", function() {
        var res = idx.searchFp32(f32(DATA[PROBE]), 5);
        return res.length && res[0].id === PROBE;
    });
    testFeature("GPU result matches the CPU result", function() {
        var cpu = faiss.openFactory("Flat", DIM);
        for (var i = 0; i < N; i++) cpu.addFp32(-1, f32(DATA[i]));
        var c = cpu.searchFp32(f32(DATA[PROBE]), 1)[0].id;
        var g = idx.searchFp32(f32(DATA[PROBE]), 1)[0].id;
        return c === g;
    });
}

/* ================================================================
   run
   ================================================================ */
var flatIdx = runFlatTest();
var ivfIdx  = null;
runIDMapTest();
ivfIdx = runIVFTest();
runPersistTest(flatIdx, ivfIdx);
runGpuTest();

cleanup();
printf("\nfaiss: %d tests run, %d failed\n", testnum, failed);
process.exit(failed ? 1 : 0);
