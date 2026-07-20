# rampart-langtools docker oven

Builds the **langtools modules** — `rampart-faiss`, `rampart-llamacpp`,
`rampart-sentencepiece`, and `rampart-onnx` (ONNX Runtime) — as **portable,
permissively-licensed `.so`s** inside [manylinux] "oven" containers. Building in
an oven with an old glibc floor but a modern compiler means the result runs on
that glibc and everything newer, without bundling CUDA, the NVIDIA driver, or any
GPL runtime.

> This is the concise operator overview. The deep "why + how + gotchas" companion
> lives outside the committed tree (`no-commit/langtools-docker.md`); onnx
> packaging specifics are in `extern/onnxruntime-vendoring.md`.

```
docker/build.sh <stage> [variant]
```

## Tiers (glibc floors → install prefix)

The install **prefix encodes the tier**; the module filename suffix (`_cpu`,
`_cu12`, …) distinguishes flavors within a tree.

| Prefix | glibc floor | Base image | Contents |
|---|---|---|---|
| `/usr/local/rampart-2_17` | 2.17 | manylinux2014 | CPU modules — max reach; **no onnx** (ORT needs glibc ≥ 2.28) |
| `/usr/local/rampart-2_28` | 2.28 | manylinux_2_28 | CPU **and** GPU modules; newer SIMD; **onnx-capable** |

## Variants

`./build.sh build [variant]` — the token drives base image, toolchain, CUDA,
build dir, module suffix, and default install tier:

| variant | glibc | CUDA | suffix | default prefix | onnx |
|---|---|---|---|---|---|
| *(none)* / `cpu` | 2.17 | — | *(none)* / `_cpu` | `rampart-2_17` | skipped (`LT_ONNX=0`) |
| `cpu_2_28` | 2.28 | — | `_cpu` | `rampart-2_28` | **builds `rampart-onnx.so`** (CPU) |
| `cu11` (x86) | 2.17 | 11.8 | `_cu11` | `rampart-2_17` | skipped (CUDA < 12) |
| `cu11` (arm) | 2.28 | 11.8 | `_cu11` | `rampart-2_28` | skipped (CUDA < 12) |
| `cu11_2_28` | 2.28 | 11.8 | `_cu11` | `rampart-2_28` | skipped (CUDA < 12) |
| `cu12` | 2.28 | 12.8 | `_cu12` | `rampart-2_28` | **builds `modules/onnx-cu12/` runtime dir** |
| `cu13` | 2.28 | 13.0 | `_cu13` | `rampart-2_28` | **builds `modules/onnx-cu13/` runtime dir** |

**rampart-onnx is one unified module.** The single `rampart-onnx.so` (built by the
`cpu_2_28` variant) is CPU-complete on its own; the `cu12`/`cu13` variants build
only a drop-in GPU **runtime directory** (`modules/onnx-cuNN/`) that the module
auto-selects at load time if present. cu11/2_17 tiers omit onnx entirely (ORT
1.27's CUDA EP needs CUDA ≥ 12, and ORT needs glibc ≥ 2.28). See
`extern/onnxruntime-vendoring.md`.

## Stages

| Command | What it does |
|---|---|
| `build.sh build [variant]` | compile into `build/oven[-variant]/` |
| `build.sh install [variant]` | install the modules into `<prefix>/modules` (+ test scripts) |
| `build.sh shell [variant]` | interactive shell in the matching oven |
| `build.sh save-image [variant]` | persist the oven image to `build/<image>.image.tar.gz` |

Typical flow (build and install with the **same token**):
```
docker/build.sh build cu12
docker/build.sh install cu12
```

## Flags

- `--rebuild-image [build [variant]]` — force a fresh oven image first (after a
  Dockerfile edit; cache-aware).
- `-d <dir>` — install into `<dir>` instead of the tier default. If `<dir>` isn't
  the build's home tier, the modules are **grafted** (copied in) rather than
  rebuilt — e.g. put one cu11 build in both trees.

## Environment knobs

All forwarded into the oven by `build.sh`. Prefix the command
(`ONNX_CUDA_PARALLEL=3 ./build.sh build cu12`) or export them.

| Var | Default | Meaning |
|---|---|---|
| `LT_BUILD_PARALLEL` | `nproc` | main compile `-j` |
| `ONNX_CUDA_PARALLEL` | `1` | ORT CUDA-EP build parallelism |
| `ONNX_CPU_PARALLEL` | `8` | ORT CPU-EP build parallelism |
| `ONNX_CUDA_ARCH` | per-variant | override ORT CUDA arch list, e.g. `89-real;89-virtual` |
| `ONNX_CUDA_MINIMAL` | `0` | `1` = cuBLAS-only CUDA EP, no cuDNN (smaller; fewer GPU ops) |
| `LT_TARGET` | — | build ONE cmake target, e.g. `onnxruntime_ep` |

### Tuning `ONNX_CUDA_PARALLEL` (the ORT GPU build is the long pole)

The ORT CUDA execution provider compiles cutlass/flash-attn kernels for **every**
`-real` arch in the variant's list at once, so it's the memory monster of the
build. Empirically:

> **peak RAM ≈ `ONNX_CUDA_PARALLEL` × 4 × ~2 GB**  (nvcc uses 4 threads each)

For the **full arch fleet** (cu12 x86 = `80;86;89;90;100;120`), sized to RAM:

| RAM | safe | max (watch `free -g`) |
|---|---|---|
| 15 GB | 1 | 1 |
| 64 GB | **3** | 4 |

Going higher just spills into swap and gets **slower** — you're memory-bound, not
CPU-bound. The real speed lever is **fewer arches**: for a single-GPU dev build,
`ONNX_CUDA_ARCH="89-real;89-virtual"` cuts per-nvcc RAM ~6× and lets you raise
`ONNX_CUDA_PARALLEL` to saturate the cores. Keep the full list for shippable
artifacts.

## Mounts & images

Nothing host-facing is baked into the image — it's bind-mounted at `docker run`:
the repo at `/lt` (rw; outputs land in `build/oven*`), and the installed rampart
prefix at its **real path** (`:ro` for build/shell, rw for install; the build
reads headers from `<prefix>/include` and installs to `<prefix>/modules`). The
build is handed the prefix directly via `-DRP_PATH` — it does **not** execute the
mounted `rampart` binary, so a host rampart built against a newer glibc than the
oven still works.

Images persist in your local docker store; `build.sh` finds them automatically,
restores from a saved `.image.tar.gz` if present, else builds them. A plain
`docker image prune -f` won't remove them (only `docker rmi` / `prune -a`).

[manylinux]: https://github.com/pypa/manylinux
