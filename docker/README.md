# rampart-langtools docker oven

Builds the **langtools modules** (FAISS, llama.cpp, sentencepiece) for the
portable `centos7-x86_64` target (glibc 2.17 floor) inside a self-contained
[manylinux2014] "oven" container.

There are **two images**, so a CPU build never has to download CUDA:
- **`rampart-langtools-oven`** — the cpu/default base (devtoolset-11 + OpenBLAS, **no CUDA**).
- **`rampart-langtools-oven-cuda`** — `FROM` the base + the CUDA 11.8 toolkit, built **only** when you ask for `cuda`.

The base is self-contained on purpose (not `FROM` the core rampart oven): it
builds its own OpenBLAS **once** at image-build time, then is reused — so an
ordinary langtools change never rebuilds OpenBLAS.

The rampart headers and binary are **not** baked in: `build.sh` bind-mounts the
installed `/usr/local/rampart-ml`, and the build resolves `$RP_PATH` from
`rampart -c "console.log(process.installPath)"`.

```
docker/build.sh <stage> [cpu|cuda]
```

## Commands

| Command | Output | CUDA? |
|---|---|---|
| `docker/build.sh build` | `build/oven/` → `rampart-langtools.so` (unsuffixed) | no |
| `docker/build.sh build cpu` | `build/oven-cpu/` → `rampart-langtools_cpu.so` | no |
| `docker/build.sh build cuda` | `build/oven-cuda/` → `rampart-langtools_cuda.so` | **yes** (builds the cuda image) |
| `docker/build.sh install [cpu\|cuda]` | install matching modules into `/usr/local/rampart-ml/modules` | — |
| `docker/build.sh shell [cpu\|cuda]` | interactive shell in the (base or cuda) oven | — |
| `docker/build.sh save-image [cuda]` | persist the cpu (or cuda) image to a `.tar.gz` | — |
| `docker/build.sh --rebuild-image [build [cpu\|cuda]]` | force a fresh image first | — |

The same applies to `rampart-faiss` and `rampart-llamacpp` (e.g. `build` →
`rampart-faiss.so`, `build cpu` → `rampart-faiss_cpu.so`). `rampart-sentencepiece.so`
is always unsuffixed (it has no GPU variant). Typical flows:

```
docker/build.sh build           # plain modules, no CUDA pulled
docker/build.sh install

docker/build.sh build cuda      # GPU modules; first run downloads the CUDA toolkit
docker/build.sh install cuda
```

### no-suffix vs cpu vs cuda

- **`build`** (no arg) — the canonical CPU modules with **bare names**
  (`rampart-langtools.so`). Uses the base image; **no CUDA download**.
- **`build cpu`** — identical build but **`_cpu`-suffixed** names, for when you
  want cpu and cuda installed side by side. Also no CUDA.
- **`build cuda`** — `_cuda`-suffixed names; uses the cuda image. The CUDA
  libraries are **not bundled**: `libcudart.so.11.0`, `libcublas.so.11`, and
  `libcuda.so.1` (driver) must already be present on the GPU box.

All three keep the glibc 2.17 floor; the CPU build bundles `libgomp`/`libgfortran`
(resolved via `RPATH $ORIGIN/../lib`).

## Mounted directories

Nothing host-facing is baked into the image — it's all bind-mounted at
`docker run` time. `$REPO` is the langtools repo root
(`/usr/local/src/rampart-langtools`).

| Stage | Host path → container path | Mode |
|---|---|---|
| **build** | `/usr/local/src/rampart-langtools` → `/lt` | rw |
| | `/usr/local/rampart-ml` → `/usr/local/rampart-ml` | **ro** |
| | `/etc/passwd` → `/etc/passwd` | ro |
| | `/etc/group` → `/etc/group` | ro |
| **install** | `/usr/local/src/rampart-langtools` → `/lt` | rw |
| | `/usr/local/rampart-ml` → `/usr/local/rampart-ml` | rw |
| **shell** | `/usr/local/src/rampart-langtools` → `/lt` | rw |
| | `/usr/local/rampart-ml` → `/usr/local/rampart-ml` | **ro** |

Why each one:

- **Repo (`/lt`)** — always rw: the build writes outputs to `build/oven[-<variant>]/`.
- **`/usr/local/rampart-ml`** — the *installed* centos7 rampart. Mounted **ro at
  build** (reads headers + runs the binary for `$RP_PATH`) and **rw at install**
  (drops modules into `…/modules`). Only this subdir is mounted, not all of
  `/usr/local`, or it would shadow the oven's own cmake + OpenBLAS.
- **`/etc/passwd` + `/etc/group`** (ro) — only on `build`, which runs as your uid
  (`--user`) so the uid resolves to a name. `install` runs as root (to write the
  system modules dir), so it doesn't mount these.

Everything else (devtoolset-11, OpenBLAS, and — for cuda — the CUDA toolkit)
lives **inside** the image. `install` always uses the base image (it only does
`cmake --install`, which needs no CUDA).

## The oven images

Both images live in your local docker store and persist across reboots and
container runs — `build.sh` finds them automatically.

`save-image` writes the cpu image to `build/rampart-langtools-oven.image.tar.gz`
(or `save-image cuda` → `build/rampart-langtools-oven-cuda.image.tar.gz`). Only
needed to move an image to another machine, back it up before an aggressive
prune, or keep a daemon-independent snapshot. If a tarball exists, `ensure_*`
restores it with `docker load` instead of rebuilding.

After editing a `Dockerfile`, rebuild with `--rebuild-image` (cache-aware — it
reuses the base/devtoolset/OpenBLAS layers; `--rebuild-image build cuda` rebuilds
the cuda image).

> A plain `docker image prune -f` only removes **dangling** (untagged) images and
> won't touch these. Only `docker rmi`, `docker image prune -a`, `docker system
> prune -a`, or a docker reinstall remove them — and even then the `Dockerfile`s
> reproduce them deterministically (needs network).

[manylinux2014]: https://github.com/pypa/manylinux
