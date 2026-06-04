/* WebSocket chat app backed by rampart-llamacpp (model runs IN-PROCESS).
 *
 * This is the rampart-llamacpp analogue of the rampart-llm.js demo's llmchat.js.
 * Instead of forwarding to a llama-server / ollama over HTTP, it streams tokens
 * straight from a shared, transparently-batched initGen() engine.
 *
 * THE ENGINE IS SHARED: the server's postForkFunc (see ../llamacpp-chat.js)
 * created one initGen() engine per model and put it on the global object as
 * LLMGEN[name]. Every server thread uses that SAME engine, so all concurrent
 * chats — across every thread — batch together across the engine's nSeqMax slots,
 * with just one copy of the model in memory. This wsapp only CONSUMES it.
 *
 * Available at  ws://<host>/wsapps/llamacppchat.txt
 */
rampart.globalize(rampart.utils);

/* have the model produce a short title for this conversation (a second,
   concurrent generation — it batches with the reply on the engine's slots) */
function makeTitle(req) {
    if (req.titled) return;
    req.titled = true;

    var p = req.prompt;
    if (!p || getType(p) !== 'Array' || p.length < 3) return;  // system,user,assistant

    var userQ = sprintf(
        "Below is the beginning of a conversation. Make a four to seven word title " +
        "that best summarizes it.\n%s\n%s\n", p[1].content, p[2].content);

    req.engine.predictAsync(
        {
            messages: [
                { role: 'system', content: "You create short titles for conversations." },
                { role: 'user',   content: userQ }
            ],
            maxTokens: 24,
            temp: 0.3
        },
        null,                                            // no per-token callback needed
        function (res) { req.wsSend({ title: (res.fullText || "").trim() }); }
    );
}

/* generate a reply for req.prompt and stream it back over the websocket */
function answer(req) {
    req.processing = true;
    req.cancel     = false;
    req.ended      = false;
    req.thinking   = true;
    req.wsSend({ srvmsg: "thinking..." });

    /* predictAsync returns a handle with .cancel() (hard cancel: stops generation
       in the engine and frees the slot). We keep it so the cancel command can use it. */
    req.gh = req.engine.predictAsync(
        {
            messages:  req.prompt,
            maxTokens: req.modelCfg.maxTokens || 1024,
            temp:      req.modelCfg.temp
        },

        /* per-token: stream tokens to the client */
        function (res) {
            if (req.ended) return;
            if (res.error) return;
            if (req.thinking) { req.wsSend({ srvmsg: "answering..." }); req.thinking = false; }
            if (res.token) req.wsSend(res.token);
        },

        /* final: close the turn, remember the assistant reply, make a title */
        function (res) {
            req.processing = false;
            if (req.ended) return;                       // already closed by a cancel
            req.ended = true;
            if (res.error) {
                req.wsSend({ srvmsg: "error: " + res.error });
                req.wsSend({ end: "<br><hr><br>" });
                return;
            }
            req.prompt.push({ role: "assistant", content: res.fullText || "" });
            req.wsSend({ end: "<br><hr><br>" });
            req.wsSend({ srvmsg: "" });
            makeTitle(req);
        }
    );
}

/* Called each time new data arrives. `req` persists across calls for a
   connection; only req.body changes. */
function chat(req) {

    /* req.count==0: the ws handshake. Pick the model + grab the shared engine. */
    if (req.count == 0) {
        if (typeof LLMGEN !== 'object' || typeof LLMMODELS !== 'object') {
            req.wsSend({ error: "engine not initialized on the server" });
            req.wsEnd();
            return;
        }
        var settings, modelObj;
        if (req.params.s) {
            try { settings = req.params.s; } catch (e) {}
        }
        var modelName = (settings && settings.model) || LLMMODELS.defaultModel;
        modelObj = LLMMODELS[modelName];
        req.engine = LLMGEN[modelName];
        if (!modelObj || !req.engine) {
            req.wsSend({ error: "model '" + modelName + "' not found" });
            req.wsEnd();
            return;
        }
        req.modelCfg  = modelObj;
        req.sysPrompt = modelObj.sysPrompt;
        return;
    }

    if (!req.body.length) return;

    var intxt = req.body;

    /* control messages / context restore (JSON), else plain user text */
    try {
        var cmdobj = JSON.parse(intxt);

        if (cmdobj.cmd == "cancel") {
            if (req.processing && !req.ended) {
                req.ended  = true;                 // close the turn now (final cb will no-op)
                req.processing = false;
                if (req.gh) req.gh.cancel();        // hard cancel: stops the engine, frees the slot
                req.wsSend({ end: "<br><hr><br>" });
                req.wsSend({ srvmsg: "" });
            }
            return;

        } else if (cmdobj.cmd == "reset") {
            if (req.processing) { req.ended = true; req.processing = false; if (req.gh) req.gh.cancel(); }
            req.prompt = false;
            req.titled = false;
            return;

        } else if (cmdobj.context && cmdobj.q) {
            /* continuing a conversation stored in the browser */
            req.prompt = cmdobj.context;
            req.prompt.unshift({ role: 'system', content: req.sysPrompt });
            intxt = cmdobj.q;
            req.titled = (req.prompt.length >= 3);
            /* fall through to generate */
        }
    } catch (e) { /* plain text */ }

    if (req.processing) return;   // one generation per connection at a time

    if (!req.prompt) {
        req.prompt = [
            { role: 'system', content: req.sysPrompt },
            { role: 'user',   content: sprintf("%s\n", intxt) }
        ];
        req.titled = false;
    } else {
        req.prompt.push({ role: 'user', content: sprintf("%s\n", intxt) });
    }

    answer(req);
}

module.exports = chat;
