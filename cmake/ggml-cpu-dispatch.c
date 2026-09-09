/* ggml-cpu-dispatch.c -- ONE module, several ggml-cpu ISA tiers, chosen at runtime.
 *
 * ggml compiles its whole CPU backend per instruction-set tier; upstream ships each
 * tier as a separate dlopen'd .so (GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL).  We want
 * one self-contained module instead, so each tier is compiled here, its entry points
 * renamed to <name>_<tier> and every other symbol localized, and this file supplies
 * the two names the rest of ggml actually calls.  Tier code is reached ONLY through
 * these functions -- ggml-cpu has no global constructors -- so the AVX-512 copy never
 * executes a single instruction on a CPU that lacks AVX-512.
 *
 * Selection uses ggml's OWN scorer (arch/x86/cpu-feats.cpp): it CPUIDs and returns 0
 * when any feature the tier was built with is missing, else a bit-weighted score, so
 * the richest runnable tier wins and x64 (score 1) is always a valid floor. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml-cpu.h"

#define GGML_CPU_TIERS  X(skylakex) X(haswell) X(sse42) X(x64)

#define X(t) \
    extern ggml_backend_reg_t ggml_backend_cpu_reg_##t (void); \
    extern ggml_backend_t     ggml_backend_cpu_init_##t(void); \
    extern int                ggml_backend_score_##t   (void);
GGML_CPU_TIERS
#undef X

enum { 
#define X(t) TIER_##t,
    GGML_CPU_TIERS
#undef X
    TIER_COUNT
};

static const char * const tier_name[TIER_COUNT] = {
#define X(t) #t,
    GGML_CPU_TIERS
#undef X
};

static int tier_score(int i) {
    switch (i) {
#define X(t) case TIER_##t: return ggml_backend_score_##t();
        GGML_CPU_TIERS
#undef X
    }
    return 0;
}

/* Highest-scoring runnable tier.  RAMPART_GGML_CPU_TIER=<name> forces one (for
 * testing a lower tier on capable hardware); RAMPART_GGML_CPU_DEBUG=1 reports the
 * choice on stderr.  Computed once; the scorers are pure CPUID reads, so a race
 * only ever recomputes the same answer. */
static int cpu_tier(void) {
    static int chosen = -1;
    if (chosen >= 0) return chosen;

    int best = TIER_x64, best_score = 0;
    const char *force = getenv("RAMPART_GGML_CPU_TIER");

    for (int i = 0; i < TIER_COUNT; i++) {
        int s = tier_score(i);
        if (force && !strcmp(force, tier_name[i])) {
            if (s > 0) { best = i; best_score = s; break; }
            fprintf(stderr, "rampart: RAMPART_GGML_CPU_TIER=%s is not supported by this CPU\n",
                    force);
            force = NULL;                       /* fall back to auto-detect */
        }
        if (s > best_score) { best_score = s; best = i; }
    }
    if (getenv("RAMPART_GGML_CPU_DEBUG"))
        fprintf(stderr, "rampart: ggml cpu tier = %s (score %d)\n", tier_name[best], best_score);
    chosen = best;
    return chosen;
}

/* the two symbols the rest of ggml links against */
ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    switch (cpu_tier()) {
#define X(t) case TIER_##t: return ggml_backend_cpu_reg_##t();
        GGML_CPU_TIERS
#undef X
    }
    return ggml_backend_cpu_reg_x64();
}

ggml_backend_t ggml_backend_cpu_init(void) {
    switch (cpu_tier()) {
#define X(t) case TIER_##t: return ggml_backend_cpu_init_##t();
        GGML_CPU_TIERS
#undef X
    }
    return ggml_backend_cpu_init_x64();
}

/* which tier was picked -- for tests and for clip.getLog()-style reporting */
const char * rampart_ggml_cpu_tier(void) { return tier_name[cpu_tier()]; }
