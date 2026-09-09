#!/bin/sh
# namespace-tier.sh <tag> <output.o> <input objects...>
#
# Combine one ISA tier's objects and make every symbol it DEFINES its own, so several
# tiers of the same ggml-cpu sources can live in one module.
#
# Renaming (not localizing) is required: ggml-cpu is C++, so its vtables and template
# instantiations sit in COMDAT section groups, and the final link dedups groups by
# signature ACROSS objects -- it would keep one tier's group, discard the others', and
# leave "defined in discarded section" errors (or, worse, point one tier's vtable at
# another tier's differently-compiled methods).  Renaming the defined symbols renames
# the group signature symbols too, so nothing merges.  Undefined symbols are left
# alone so references to ggml-base and libc still resolve normally.
#
# The three entry points get canonical per-tier names; ggml-cpu-dispatch.c calls them.
set -e
tag=$1; out=$2; shift 2

# Work in a private temp dir and rename at the end.  These objects are consumed as
# sources by three targets, and the Makefile generator emits the rule once per
# consuming target -- so `make -j` can run this concurrently for the same tier.  Two
# runs sharing "$out.map" interleaved their symbol lists, and objcopy rejected the
# result ("Multiple redefinition of symbol ggml_backend_cpu_reg").  Per-run temps
# make concurrent invocations harmless; the final mv is atomic.
work=$(mktemp -d "${TMPDIR:-/tmp}/lt-tier-$tag.XXXXXX")
trap 'rm -rf "$work"' EXIT
tmp="$work/all.o"
map="$work/map"

${LD:-ld} -r "$@" -o "$tmp"

${NM:-nm} --defined-only "$tmp" | awk '{print $NF}' | sort -u \
  | grep -vE '^(ggml_backend_cpu_reg|ggml_backend_cpu_init|ggml_backend_score)$' \
  | awk -v p="$tag" '{print $1" "p"__"$1}' > "$map"
cat >> "$map" <<MAP
ggml_backend_cpu_reg ggml_backend_cpu_reg_$tag
ggml_backend_cpu_init ggml_backend_cpu_init_$tag
ggml_backend_score ggml_backend_score_$tag
MAP

${OBJCOPY:-objcopy} --redefine-syms="$map" "$tmp" "$work/out.o"
mv -f "$work/out.o" "$out"
