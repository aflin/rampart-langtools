# rampart-llamacpp web chat demo

A self-contained rampart web server that serves a streaming LLM chat UI backed by
**rampart-llamacpp running the model in-process** — no `llama-server` and no
`ollama`. It is the rampart-llamacpp analogue of the `rampart-llm.js` chat demo in
`/usr/local/rampart/unsupported_extras/llm-demo/`.

```
web_server_demo/
├── llamacpp-chat.js            # the server (run this)
├── wsapps/
│   ├── llamacppchat.js         # websocket chat handler (in-process gen.predictAsync)
│   └── llamacpp-models.js      # model path + engine/sampling options
└── html/                       # the browser UI (engine-agnostic client + css)
    ├── index.html / chat.html
    ├── js/llmchat-client.js
    └── css/llmchat.css
```

## 1. Get a model

The default model is the small one the langtools test downloads to the standard
location `~/.rampart/models/gen/`:

```
rampart /usr/local/rampart/test/llamacpp-test.js     # answer "y" to the gen download
```

Or edit `wsapps/llamacpp-models.js` and set `model:` to any local instruct `.gguf`.

## 2. Make sure rampart-llamacpp is loadable

Either `make install` the langtools modules, or just run from a checkout that has a
`build/` (the wsapp also looks in `../build`, `../build_gpu`, `../build_cpu`
relative to this server script).

## 3. Run

```
rampart llamacpp-chat.js            # start
rampart llamacpp-chat.js stop       # stop
rampart llamacpp-chat.js status     # status
```

Then open **http://127.0.0.1:8089/** and chat. Tokens stream live; **Cancel** stops
the current generation, **Start New Session** clears history. Conversations are
stored in the browser's localStorage.

## How it works

This demo showcases `initGen()`'s transparent multi-slot batching: **one shared
engine, all threads.**

- **One engine, one model copy.** `llamacpp-chat.js` builds a single
  `initGen(model, {nCtx, nSeqMax})` engine per model and puts it on the `global`
  object (`LLMGEN[name]`). Every one of the server's `threads` worker threads uses
  that same engine — so no matter how many threads serve requests, there is just
  **one** copy of the model in memory.
- **All chats batch together.** Each chat calls `gen.predictAsync(opts, onToken,
  onDone)`; concurrent generations from *any* thread pool into the engine and
  decode together across its `nSeqMax` slots.
- **streaming** — each token is pushed over the websocket as produced.
- **title** — a second short generation runs concurrently to title the chat; it
  batches with the reply on the engine's slots.
- **cancel** — a hard cancel: `predictAsync` returns a handle whose `.cancel()`
  stops the generation in the engine and frees its slot immediately.

### postForkFunc — why the engine is built where it is

The engine is created in `init_llm`, wired as the server's **`postForkFunc`**. A
rampart web server may **fork** (to daemonize), and a GPU (CUDA) context **cannot
survive a fork** — so the engine must be built *after* the fork, in the process
that actually serves requests. `postForkFunc` runs there. When `daemon:false`
there is no fork, so the config calls `init_llm()` directly.
Either way the engine lands in the serving process and is shared by all threads.

Because the model loads once at startup, `threads` can be as large as you like
(set to **4** here) — they all share the one engine.

