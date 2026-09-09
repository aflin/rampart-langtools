#!/bin/sh
# check-cpu-tier-symbols.sh <module.so> <baseline libggml-cpu.a>
#
# Guard for the multi-tier CPU backend.  Every ggml-cpu symbol is renamed per tier,
# and ggml-cpu-dispatch.c re-exports only the two the rest of the build actually
# calls (ggml_backend_cpu_reg, ggml_backend_cpu_init).  That set was MEASURED, so if
# a llama.cpp upgrade (or new module code) starts calling something else from
# ggml-cpu -- ggml_graph_compute, ggml_cpu_init, ggml_backend_cpu_set_abort_callback,
# ... -- it would leave an undefined symbol in a `-shared` link, which only shows up
# as a dlopen failure at require() time.  Catch it at build time instead: add a
# forwarder to the dispatcher for whatever this reports.
set -e
mod=$1; cpulib=$2
[ -f "$mod" ] && [ -f "$cpulib" ] || exit 0

nm --undefined-only "$mod" | awk '{print $NF}' | sort -u > "$mod.undef"
nm --defined-only "$cpulib" | awk 'NF==3 {print $3}' | sort -u > "$mod.cpudef"
missing=$(comm -12 "$mod.undef" "$mod.cpudef" || true)
# a tier whose objects never made it into the link leaves ITS namespaced entry
# points undefined -- same silent-until-first-use failure, so flag it here too
missing="$missing
$(grep -E '^(ggml_backend_score|ggml_backend_cpu_reg|ggml_backend_cpu_init)_' "$mod.undef" || true)"
missing=$(echo "$missing" | grep -v '^$' || true)
rm -f "$mod.undef" "$mod.cpudef"

if [ -n "$missing" ]; then
    echo "ERROR: $(basename "$mod") references ggml-cpu symbols no tier exports:" >&2
    echo "$missing" | sed 's/^/    /' >&2
    echo "  -> add a forwarder for each in cmake/ggml-cpu-dispatch.c" >&2
    exit 1
fi
