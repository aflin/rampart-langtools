#!/usr/bin/env rampart

/* *** rampart-llamacpp web chat demo server ***
 *
 * A self-contained rampart web server that serves a streaming LLM chat UI backed
 * by rampart-llamacpp running the model IN-PROCESS (no llama-server / ollama).
 *
 * Run it:
 *     rampart llamacpp-chat.js            # start (foreground)
 *     rampart llamacpp-chat.js stop       # stop
 *     rampart llamacpp-chat.js status     # status
 *
 * Then open  http://127.0.0.1:8089/  in a browser.
 *
 * Before running, set the model path in wsapps/llamacpp-models.js (a local .gguf).
 * The default points at the model the langtools test downloads:
 *     rampart /usr/local/rampart/test/llamacpp-test.js   (answer yes to the gen model)
 *
 * SHARED, BATCHED ENGINE: a single initGen() engine is created per model and put
 * on the global object. All server threads share that ONE engine — every chat's
 * predict/predictAsync funnels into it and batches across its nSeqMax slots, so
 * there is just one copy of the model in memory no matter how many threads serve.
 *
 * The engine is created in `init_llm` which is wired as `postForkFunc`: it runs
 * AFTER the server's daemonize fork, because a GPU (CUDA) context cannot be
 * created in a process that then forks. When not daemonizing we call it directly
 * (see the bottom of this file) — same pattern as
 * /usr/local/src/rampart_wikipedia_search/web_server/web_server_conf.js
 */

//set working directory to the location of this script
var working_directory = process.scriptPath;

/* Build the shared engine(s) and publish them on `global` so every server thread
   can use them. MUST run after any fork (hence postForkFunc) so the engine's GPU
   context / owner thread live in the process that actually serves requests. */
function init_llm() {
    rampart.localize(rampart.utils);   // printf/fprintf/stderr in this scope
    var g = global;

    /* resolve rampart-llamacpp: a local build dir first (freshest when running from
       the source tree), else the installed module */
    var llamacpp;
    var tries = [
        working_directory + "/../build/rampart-llamacpp",
        working_directory + "/../build_gpu/rampart-llamacpp",
        working_directory + "/../build_cpu/rampart-llamacpp",
        "rampart-llamacpp"
    ];
    for (var i = 0; i < tries.length; i++) {
        try { llamacpp = require(tries[i]); break; } catch (e) {}
    }
    if (!llamacpp) {
        fprintf(stderr, "rampart-llamacpp.so not found (run 'make install', or build it)\n");
        process.exit(1);
    }

    var models = require(working_directory + "/wsapps/llamacpp-models.js");

    /* ONE transparently-batched engine per model, shared across all threads */
    g.LLMMODELS = models;
    g.LLMGEN    = {};
    for (var name in models) {
        if (name === "defaultModel") continue;
        var m = models[name];
        try {
            g.LLMGEN[name] = llamacpp.initGen(m.model, { nCtx: m.nCtx, nSeqMax: m.nSeqMax });
            printf("[llm] loaded '%s' (%s)  nCtx=%d slots=%d\n",
                   name, m.model, g.LLMGEN[name].nCtx, m.nSeqMax);
        } catch (e) {
            fprintf(stderr, "[llm] failed to load model '%s' (%s): %s\n",
                    name, m.model, e.message || e);
            process.exit(1);
        }
    }
}

var serverConf = {

    /* create the shared engine AFTER the fork (CUDA-safe) */
    postForkFunc:  init_llm,

    /* default is 8088 */
    //port:          8088,

    /* bind to all ips, not just localhost */
    //bindAll:       true,

    htmlRoot:      working_directory + '/html',
    wsappsRoot:    working_directory + '/wsapps',
    dataRoot:      working_directory + '/data',
    logRoot:       working_directory + '/logs',
    accessLog:     working_directory + '/logs/access.log',
    errorLog:      working_directory + '/logs/error.log',
    log:           true,

    /* all of these threads share the ONE engine above, so use as many as you like */
    threads:       4,

    daemon:        true,
    monitor:       false,

    /* show JS stack traces on error while developing the demo */
    developerMode: true,

    serverRoot:    working_directory
};

/* If not daemonizing, postForkFunc is not auto-invoked, so build the engine here
   (foreground process). If daemon:true, init_llm runs as postForkFunc instead. */
if (serverConf.daemon === false)
    init_llm();

require("rampart-webserver").web_server_conf(serverConf);
