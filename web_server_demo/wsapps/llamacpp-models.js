/* Model definitions for the rampart-llamacpp chat demo.
 *
 * Unlike the rampart-llm.js demo (which points at a running llama-server /
 * ollama), here the model runs IN-PROCESS via rampart-llamacpp. Each entry is a
 * path to a local .gguf plus the engine/sampling options.
 *
 * The default points at the gen model downloaded by the langtools test into the
 * standard location ~/.rampart/models/gen/ :
 *     rampart /usr/local/rampart/test/llamacpp-test.js     (answer yes to the gen download)
 * or set `model` to any instruct .gguf you already have.
 *
 * One shared, transparently-batched initGen() engine is created per model (in the
 * server's postForkFunc) and used by every server thread, so there is just ONE
 * copy of the model in memory regardless of thread count. `nSeqMax` is how many
 * generations batch together across all those threads.
 */

var HOME = process.env.HOME || "/tmp";
var MODELDIR = HOME + "/.rampart/models";

var sysPrompt = "You are a concise, helpful assistant.";

var models = {
    qwen: {
        sysPrompt: sysPrompt,
        model:     MODELDIR + "/gen/qwen2.5-0.5b-instruct-q4_k_m.gguf",

        nCtx:      0,      // total context; 0 = the model's full trained max (like
                           //   llama-server). Split across nSeqMax slots, so each
                           //   conversation gets nCtx/nSeqMax (here 32768/4 = 8192).
        nSeqMax:   4,      // concurrent generations batched together in the engine
        maxTokens: 1024,   // max tokens to generate per reply
        temp:      0.2     // sampling temperature
    },

    defaultModel: "qwen"
};

module.exports = models;
