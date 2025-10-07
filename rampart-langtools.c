#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <llama.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "llama.h"
#include "rampart-langtools.h"
#include "rampart.h"
#include "spm_c_wrapper.h"
#include <c_api/AutoTune_c.h>
#include <c_api/Clustering_c.h>
#include <c_api/IndexBinaryIVF_c.h>
#include <c_api/IndexBinary_c.h>
#include <c_api/IndexFlat_c.h>
#include <c_api/IndexIVFFlat_c.h>
#include <c_api/IndexIVF_c.h>
#include <c_api/IndexLSH_c.h>
#include <c_api/IndexPreTransform_c.h>
#include <c_api/IndexReplicas_c.h>
#include <c_api/IndexScalarQuantizer_c.h>
#include <c_api/IndexShards_c.h>
#include <c_api/Index_c.h>
#include <c_api/MetaIndexes_c.h>
#include <c_api/VectorTransform_c.h>
#include <c_api/clone_index_c.h>
#include <c_api/error_c.h>
#include <c_api/index_factory_c.h>
#include <c_api/index_io_c.h>

#include "convert_vec.c"

#define LANGTOOLS_MAIN_INCLUDE

/* *******************************************************
                LLAMACPP INCLUDE
   ******************************************************* */

#include "rampart-llamacpp.c"

/* *******************************************************
                SENTENCEPIECE INCLUDE
   ******************************************************* */

#include "rampart-sentencepiece.c"

/* **************************************************
                FAISS INCLUDE
   ************************************************** */

#include "rampart-faiss.c"


/* ***********************************************************
                    UTILS
   *********************************************************** */

#include "rampart-vecutils.c"

/* **************************************************
   Initialize module
   ************************************************** */

duk_ret_t duk_open_module(duk_context *ctx)
{
    duk_push_object(ctx);
    open_llama(ctx);
    duk_put_prop_string(ctx, -2, "llamacpp");

    open_sentencepiece(ctx);
    duk_put_prop_string(ctx, -2, "sentencepiece");

    open_faiss(ctx);
    duk_put_prop_string(ctx, -2, "faiss");

    open_utils(ctx);
    duk_put_prop_string(ctx, -2, "vecutils");

    return 1;
}
#undef LANGTOOLS_MAIN_INCLUDE
