/* llama_gen_shim.cc — C++ implementation of the multi-session, slot-based
 * continuous-batching generation engine, exposed via the C ABI in
 * llama_gen_shim.h. Built as its own object (llama_gen_shim_obj), mirroring
 * extern/sentencepiece/wrapper/spm_c_wrapper.cc.
 *
 * P0 (scaffold): C-ABI stubs only. This file's job right now is to prove the
 * C++ translation unit compiles against libcommon and links into the module
 * alongside the C embedding/rerank code. Real engine/slot logic lands in P1.
 */

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"

#include <string>
#include <string_view>
#include <cstring>

#include "llama_gen_shim.h"

/* ---- opaque types (filled in during P1) ---- */
struct lgen_engine  { int placeholder; };
struct lgen_session { int placeholder; };

static void set_err(char *errbuf, size_t errlen, const char *msg) {
    if (errbuf && errlen) {
        std::strncpy(errbuf, msg, errlen - 1);
        errbuf[errlen - 1] = '\0';
    }
}

extern "C" {

lgen_engine *lgen_engine_create(const lgen_engine_params *p, char *errbuf, size_t errlen) {
    (void)p;
    /* Touch a real libcommon symbol so P0 validates the C++/libcommon link. */
    (void)string_find_partial_stop(std::string_view(""), std::string_view(""));
    set_err(errbuf, errlen, "lgen_engine_create: not implemented (P0 scaffold)");
    return nullptr;
}

void lgen_engine_free(lgen_engine *e) { (void)e; }

int lgen_engine_has_active(lgen_engine *e) { (void)e; return 0; }

int lgen_engine_rebind(lgen_engine *e, char *errbuf, size_t errlen) {
    (void)e; set_err(errbuf, errlen, "lgen_engine_rebind: not implemented (P0 scaffold)");
    return -1;
}

int lgen_engine_step(lgen_engine *e) { (void)e; return 0; }

uint32_t lgen_engine_n_ctx(lgen_engine *e)   { (void)e; return 0; }
int32_t  lgen_engine_n_vocab(lgen_engine *e) { (void)e; return 0; }

lgen_session *lgen_session_create(lgen_engine *e) { (void)e; return nullptr; }
void          lgen_session_free(lgen_session *s)  { (void)s; }

uint64_t lgen_session_submit(lgen_session *s, const lgen_request *req,
                             lgen_on_piece on_piece, lgen_on_done on_done,
                             void *ud, char *errbuf, size_t errlen) {
    (void)s; (void)req; (void)on_piece; (void)on_done; (void)ud;
    set_err(errbuf, errlen, "lgen_session_submit: not implemented (P0 scaffold)");
    return 0;
}

void lgen_session_cancel(lgen_session *s) { (void)s; }

} /* extern "C" */
