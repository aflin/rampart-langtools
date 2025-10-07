#ifndef LANGTOOLS_MAIN_INCLUDE

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

#include "rampart-langtools.h"
#include "rampart.h"
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

#endif

/* **************************************************
           FAISS
   ************************************************** */

static idx_t _add_one(FaissIndex *idx, idx_t id, float *v, size_t sz, const char **err)
{
    int rc;

    if (!idx || !v)
    {
        if (err)
            *err = "Null index or vector pointer";
        return -1;
    }

    int size = faiss_Index_d(idx) * sizeof(float);
    if (size != sz)
    {
        if (err)
            *err = "Vector dimension does not match index dimension";
        return -1;
    }

    if(id < 0 )
    {
        rc = faiss_Index_add(idx, 1, v);
        id = faiss_Index_ntotal(idx);
    }
    else
        rc = faiss_Index_add_with_ids(idx, 1, v, &id);

    if (rc != 0)
    {
        if (err)
        {
            *err = faiss_get_last_error();
            if (!*err)
                *err = "faiss_Index_add_with_ids failed";
        }
        return -1;
    }

    if (err)
        *err = NULL;

    return id;
}

static duk_ret_t add_fp32(duk_context *ctx)
{
    duk_size_t sz;
    idx_t id;
    const char *err = NULL;
    double count = 0;
    FaissIndex *idx = NULL;
    float *v=NULL;
    
    if(duk_is_string(ctx, 0))
    {
        char *endptr;
        const char *str = duk_get_string(ctx, 0);

        errno = 0;
        unsigned long long val = strtoull(str, &endptr, 10);

        if (errno == ERANGE) {
            RP_THROW(ctx, "addFp16 - First argument is out of range for a 64 bit number");
        }
        if (*endptr != '\0') {
            RP_THROW(ctx, "addFp16 - First argument (id) must be a Number or String representation of a 64 bit value");
        }

        id = (idx_t) val;

    }
    else
    {
        // idx_t is int64_t, but let's do it right:
        double d = REQUIRE_NUMBER(ctx, 0, "addFp16 requires an integer (id|-1) as its first argument");
        id = (idx_t) d;
    }

    v = REQUIRE_BUFFER_DATA(ctx, 1, &sz, "addFp32 requires a buffer as its second argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!idx)
        RP_THROW(ctx, "faiss.addFp32 - Internal error getting index handle");


    err = NULL;
    id = _add_one(idx, id, v, sz, &err);

    if (id == -1)
        RP_THROW(ctx, "addFp32 - %s", err);

    count = (double)faiss_Index_ntotal(idx);

    duk_get_prop_string(ctx, -1, "settings");
    duk_push_number(ctx, count);
    duk_rp_put_prop_string_ro(ctx, -2, "count");

    duk_push_number(ctx, (double)id);

    return 1;

}

static duk_ret_t add_fp16(duk_context *ctx)
{
    duk_size_t sz;
    idx_t id;
    const char *err = NULL;
    double count = 0;
    FaissIndex *idx = NULL;
    const uint16_t *v16;
    float *v=NULL;

    if(duk_is_string(ctx, 0))
    {
        char *endptr;
        const char *str = duk_get_string(ctx, 0);
        errno = 0;
        unsigned long long val = strtoull(str, &endptr, 10);

        if (errno == ERANGE) {
            RP_THROW(ctx, "addFp16 - First argument is out of range for a 64 bit number");
        }
        if (*endptr != '\0') {
            RP_THROW(ctx, "addFp16 - First argument (id) must be a Number or String representation of a 64 bit value");
        }

        id = (idx_t) val;

    }
    else
    {
        double d = REQUIRE_NUMBER(ctx, 0, "addFp16 requires a positive integer (id) as its first argument");
        id = (idx_t) d;
    }

    v16 = REQUIRE_BUFFER_DATA(ctx, 1, &sz, "addFp16 requires a buffer as its second argument");

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!idx)
        RP_THROW(ctx, "faiss.addFp16 - Internal error getting index handle");

    v = vec_fp16_to_fp32(v16, (size_t)sz/2);

    id = _add_one(idx, id, v, sz*2, &err);
    if (id == -1)
    {
        free(v);
        RP_THROW(ctx, "faiss.addFp16 - %s", err);
    }

    count = (double)faiss_Index_ntotal(idx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_push_number(ctx, count);
    duk_rp_put_prop_string_ro(ctx, -2, "count");

    duk_push_number(ctx, (double)id);

    free(v);

    return 1;
}

/* Search top-k with optional IVF/HNSW parameters.
 * - If nprobe > 0, uses IVF search params.
 * - Else if efSearch > 0, uses HNSW search params.
 * - Else uses default index settings.
 *
 * Returns malloc'ed array of k labels (idx_t) on success; caller must free().
 * On error returns NULL and sets *err to faiss_get_last_error() or a static message.
 */
static idx_t *faiss_search_topk_ids_params(FaissIndex *idx, const float *q, /* query vector */
                                           idx_t k,                         /* desired number of res */
                                           int nprobe,                      /* set >0 for IVF */
                                           float **Dret,                    /* if not null, set distances to this */
                                           const char **err)
{
    if (err)
        *err = NULL;
    if (!idx || !q || k <= 0)
    {
        if (err)
            *err = "Invalid args: idx/q NULL or k <= 0";
        return NULL;
    }

    float *D = (float *)malloc((size_t)k * sizeof(float));
    idx_t *I = (idx_t *)malloc((size_t)k * sizeof(idx_t));

    if (!D || !I)
    {
        free(D);
        free(I);
        if (err)
            *err = "Out of memory";
        return NULL;
    }

    /* Build one search-params object if requested */
    const FaissSearchParameters *sp = NULL;
    FaissSearchParametersIVF *sp_ivf = NULL;

    int rc = 0;

    if (nprobe > 0)
    {
        if (faiss_SearchParametersIVF_new(&sp_ivf) == 0)
        {
            faiss_SearchParametersIVF_set_nprobe(sp_ivf, nprobe);
            sp = (const FaissSearchParameters *)sp_ivf;
        }
        /* if allocation failed, we just fall back to defaults below */
    }

    if (sp)
    {
        rc = faiss_Index_search_with_params(idx, 1, q, k, sp, D, I);
        /* If the params are incompatible with the index type, try plain search as a fallback */
        if (rc != 0)
        {
            rc = faiss_Index_search(idx, 1, q, k, D, I);
        }
    }
    else
    {
        rc = faiss_Index_search(idx, 1, q, k, D, I);
    }

    /* free params objects (they are independent of the search results) */
    if (sp_ivf)
        faiss_SearchParametersIVF_free(sp_ivf);

    if (rc != 0)
    {
        if (err)
            *err = faiss_get_last_error();
        free(D);
        free(I);
        return NULL;
    }

    if (Dret)
        *Dret = D;
    else
        free(D);

    return I;
}

static void do_search_(duk_context *ctx, const float *v, size_t dim, int is16)
{
    duk_uarridx_t aidx = 0;
    idx_t i = 0, k = 10, np=0;
    const char *fname = "searchFp32";

    if (is16)
        fname = "searchFp16";

    if (!duk_is_undefined(ctx, 1))
        k = (idx_t)REQUIRE_UINT(ctx, 1, "%s requires a positive integer (nResults) as its second argument", fname);

    if (!duk_is_undefined(ctx, 2))
        np = (idx_t)REQUIRE_UINT(ctx, 2, "%s requires a positive integer (nProbe for IVF) as its third argument", fname);

    // this is still at -1
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);

    const char *err = NULL;
    float *distances = NULL;

    idx_t *ids = faiss_search_topk_ids_params(idx, v, k, np, &distances, &err);

    if (err)
        RP_THROW(ctx, "%s - Error: %s", fname, err);
    else if (!ids)
        RP_THROW(ctx, "%s - unknown error while searching index", fname);

    duk_push_array(ctx);
    for (; i < k; i++)
    {
        if (ids[i] == -1)
            break;

        duk_push_object(ctx);
        duk_push_number(ctx, (double)ids[i]);
        duk_put_prop_string(ctx, -2, "id");
        duk_push_number(ctx, (double)distances[i]);
        duk_put_prop_string(ctx, -2, "distance");

        duk_put_prop_index(ctx, -2, aidx++);
    }
    free(ids);
    if (distances)
        free(distances);
}

static duk_ret_t do_search_fp16(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const uint16_t *v16 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addFp16 requires a buffer as its first argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "dimension");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 2 != dim)
        RP_THROW(ctx, "searchFp16 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    float *v = vec_fp16_to_fp32(v16, (size_t)dim);

    do_search_(ctx, v, dim, 1);

    free(v);

    return 1;
}

static duk_ret_t do_search_fp32(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const float *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "searchFp32 requires a buffer as its first argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "dimension");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 4 != dim)
        RP_THROW(ctx, "searchFp32 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    do_search_(ctx, v, dim, 1);

    return 1;
}

static duk_ret_t save_index(duk_context *ctx)
{
    const char *fname = REQUIRE_STRING(ctx, 0, "faiss.save - argument must be a filename");

    duk_push_this(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    int rc = faiss_write_index_fname(idx, fname);

    if (rc)
        RP_THROW(ctx, "Failed to save file - %s", faiss_get_last_error());
    return 0;
}

/* training */
int write_vector_to_file(FILE *fh, const float *vec, int dim)
{
    if (!fh || !vec || dim <= 0)
    {
        errno = EINVAL;
        return -1;
    }

    size_t nw = fwrite(vec, sizeof(float), (size_t)dim, fh);
    if (nw != (size_t)dim)
    {
        return -1; /* errno is set by fwrite */
    }
    return 0;
}

static duk_ret_t add_trainvec_fp32(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    FILE *fh = NULL;
    float *v = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addTrainingFp32 requires a Buffer as an argument");

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "dimension");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 4 != dim)
        RP_THROW(ctx, "addFp32 - buffer is %lu long, should be %lu (4 * %lu dimensions)", sz, dim * 4, dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "addTrainingFp32 - Internal error getting file handle");

    if (write_vector_to_file(fh, v, dim) == -1)
        RP_THROW(ctx, "addTrainingFp32 - error writing to tmpfile - %s", strerror(errno));

    return 0;
}

static duk_ret_t add_trainvec_fp16(duk_context *ctx)
{
    duk_size_t sz;
    duk_size_t dim;
    const uint16_t *v16 = REQUIRE_BUFFER_DATA(ctx, 0, &sz, "addTrainingFp16 requires a Buffer as an argument");
    FILE *fh;

    duk_push_this(ctx);
    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "dimension");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    if (sz / 2 != dim)
        RP_THROW(ctx, "addTrainingFp16 - buffer is %lu long, should be %lu (2 * %lu dimensions)", sz, dim * 2, dim);

    float *v = vec_fp16_to_fp32(v16, (size_t)dim);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "addTrainingFp16 - Internal error getting file handle");

    if (write_vector_to_file(fh, v, dim) == -1)
        RP_THROW(ctx, "addTrainingFp16 - error writing to file - %s", strerror(errno));

    free(v);

    return 0;
}

int close_and_unlink(FILE *fh, const char *path, const char **err)
{
    if (err)
        *err = NULL;

    if (fh)
    {
        if (fclose(fh) != 0)
        {
            if (err)
                *err = strerror(errno);
            return -1;
        }
    }
    if (path)
    {
        if (unlink(path) != 0)
        {
            if (err)
                *err = strerror(errno);
            return -1;
        }
    }
    return 0;
}
#define RP_THROW_AND_CLOSE(ctx, fh, path, ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        const char *__err;                                                                                             \
        if (close_and_unlink(fh, path, &__err))                                                                        \
        {                                                                                                              \
            char tothrow[2048];                                                                                        \
            snprintf(tothrow, 2048, __VA_ARGS__);                                                                      \
            duk_push_error_object(ctx, DUK_ERR_ERROR, "%s - Error closing file: %s", tothrow, __err);             \
        }                                                                                                              \
        else                                                                                                           \
            duk_push_error_object(ctx, DUK_ERR_ERROR, __VA_ARGS__);                                                    \
        (void)duk_throw(ctx);                                                                                          \
    } while (0)

static duk_ret_t dotrain(duk_context *ctx)
{
    duk_size_t dim;
    int nrows, fd;
    FILE *fh;

    //const char *filename;

    duk_push_this(ctx);
    /*
    duk_get_prop_string(ctx, -1, "trainFile");
    filename = duk_get_string(ctx, -1);
    duk_pop(ctx);
    */

    duk_get_prop_string(ctx, -1, "settings");
    duk_get_prop_string(ctx, -1, "dimension");
    dim = (duk_size_t)duk_get_int(ctx, -1);
    duk_pop_2(ctx);

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("fh"));
    fh = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!fh)
        RP_THROW(ctx, "faiss.train - Internal error getting file handle");

    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("faissIdx"));
    FaissIndex *idx = duk_get_pointer(ctx, -1);
    duk_pop(ctx);

    if (!idx)
        RP_THROW(ctx, "faiss.train - Internal error getting index handle");

    fflush(fh);

    fd = fileno(fh);
    if (fd < 0)
    {
        RP_THROW(ctx, "faiss.train - Error :%s", strerror(errno));
    }

    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        RP_THROW(ctx, "faiss.train - Internal error - cannot stat training file");
    }

    if( st.st_size % ((off_t)dim * sizeof(float)) )
    {
        RP_THROW(ctx, "faiss.train - Training file is not expected size ( size(%lu) % (dim * 4) != 0)", st.st_size);
    }
    nrows = (int)(st.st_size / ((off_t)dim * sizeof(float)));

    if (!nrows)
    {
        RP_THROW(ctx, "faiss.train - no vectors have been added yet");
    }

    void *addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
    {
        RP_THROW(ctx, "faiss.train - Internal error - cannot load training file - %s",
                           strerror(errno));
    }

    /* optional: advise the kernel about our access pattern */
    // posix_madvise(addr, need_bytes, POSIX_MADV_SEQUENTIAL);

    int rc = faiss_Index_train(idx, nrows, (const float *)addr);

    if (rc != 0)
    {
        const char *err = faiss_get_last_error();
        munmap(addr, st.st_size);
        RP_THROW(ctx, "faiss.train - training failed: %s", err);
    }

    if (munmap(addr, st.st_size) != 0)
    {
        fprintf(stderr, "faiss.train - training completed, but internal error in munmap: %s\n", strerror(errno));
        return 0;
    }

    return 0;
}

// TODO: close filehandle in a finalizer

static duk_ret_t new_trainer(duk_context *ctx)
{
    const char *trainpath = "/tmp";
    char trainfile[PATH_MAX];
    static int counter = 0; /* monotonically increasing */
    FILE *fh = NULL;
    struct stat st;

    if (!duk_is_undefined(ctx, 0))
        trainpath = REQUIRE_STRING(ctx, 0, "faiss.trainer - argument must be a String (directory path)");

    errno=0;
    if(stat(trainpath, &st) != 0)
    {
        fh = fopen(trainpath, "a+");
        if(!fh)
            RP_THROW(ctx, "faiss.trainer - can't open path '%s': %s", trainpath, strerror(errno));
        if(stat(trainpath, &st) != 0)
            RP_THROW(ctx, "faiss.trainer - error: stat path '%s': %s", trainpath, strerror(errno));
        strncpy(trainfile, trainpath, PATH_MAX-1);
    }
    else
    {
        if (S_ISDIR(st.st_mode))
        {
            /* strip trailing slashes (except for root "/") */
            size_t tlen = strlen(trainpath);
            while (tlen > 1 && trainpath[tlen - 1] == '/')
                tlen--;

            /* build filename: <trainpath>/faisstrainingdata.<counter>.<pid> */
            pid_t pid = getpid();
            int n = snprintf(trainfile, PATH_MAX, "%.*s/faisstrainingdata.%d.%ld", (int)tlen, trainpath, counter++, (long)pid);

            if (n < 0 || n >= PATH_MAX)
                RP_THROW(ctx, "faiss.trainer - training file path too long");
        }
        else if (S_ISREG(st.st_mode))
        {
            strncpy(trainfile, trainpath, PATH_MAX-1);
        }
        else
            RP_THROW(ctx, "faiss.trainer - path '%s' is something not a file or directory", trainpath);

        fh = fopen(trainfile, "a+");
        if (!fh)
            RP_THROW(ctx, "faiss.trainer - failed to open training file '%s' - %s", trainfile, strerror(errno));
    }

    duk_push_current_function(ctx);
    duk_get_prop_string(ctx, -1, DUK_HIDDEN_SYMBOL("idxthis"));

    duk_push_object(ctx);

    duk_get_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));

    duk_get_prop_string(ctx, -2, "settings");
    if(S_ISREG(st.st_mode))
    {
        duk_size_t vsize=0;
        duk_get_prop_string(ctx, -1, "dimension");
        vsize = 4 * (duk_size_t)duk_get_int(ctx, -1);
        duk_pop(ctx);
        //check file against dim and set rows
        if( (duk_size_t)st.st_size % vsize)
        {
            fclose(fh);
            RP_THROW(ctx, "faiss.trainer - invalid file size (%lu) for dim==%lu", st.st_size, vsize/4);
        }

        duk_push_int(ctx, (int)(st.st_size/vsize));
        duk_rp_put_prop_string_ro(ctx, -2, "loadedRows");
    }
    duk_put_prop_string(ctx, -2, "settings");


    duk_push_pointer(ctx, fh);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("fh"));

    duk_push_c_function(ctx, add_trainvec_fp16, 1);
    duk_put_prop_string(ctx, -2, "addTrainingfp16");

    duk_push_c_function(ctx, add_trainvec_fp32, 1);
    duk_put_prop_string(ctx, -2, "addTrainingfp32");

    duk_push_c_function(ctx, dotrain, 1);
    duk_put_prop_string(ctx, -2, "train");

    duk_push_string(ctx, trainfile);
    duk_rp_put_prop_string_ro(ctx, -2, "trainFile");

    return 1;
}

/* end training */

static void load_index(const char *fname, FaissIndex **out, const char **err, int ro)
{
    int flags = ro ? FAISS_IO_FLAG_MMAP | FAISS_IO_FLAG_READ_ONLY : 0;
    if (access(fname, R_OK) != 0)
    {
        *out = NULL;
        *err = "Could not access file";
    }
    if (faiss_read_index_fname(fname, flags, out) != 0)
    {
        *out = NULL;
        *err = faiss_get_last_error();
    };
}

FaissIndexType faiss_detect_type(FaissIndex* idx, int* pqM, int* pqBits, int *mapped);

static void push_faiss_obj(duk_context *ctx, FaissIndex *idx, FaissMetricType mtype, int dim, double rows)
{
    const char *mtype_str = "innerProduct", *type_str="Unknown";
    int pqm, pqb, mapped;
    FaissIndexType type = faiss_detect_type(idx, &pqm, &pqb, &mapped);

    switch (type)
    {
    case FAISS_INDEX_FLAT:
        type_str = "Flat";
        break;
    case FAISS_INDEX_PQ:
        type_str = "PQ";
        break;
    case FAISS_INDEX_IVFFLAT:
        type_str = "IVFFlat";
        break;
    case FAISS_INDEX_IVFPQ:
        type_str = "IVFPQ";
        break;
    case FAISS_INDEX_IVFSCALAR:
        type_str = "IVFScalar";
        break;
    /* seems to be missing from this build?
    case FAISS_INDEX_IVFRSQ:
        type_str =  "IVFRSQ";
        break;
    */
    case FAISS_INDEX_HNSWFLAT:
        type_str = "HNSWFlat";
        break;
    case FAISS_INDEX_HNSWPQ:
        type_str = "HNSWPQ";
        break;
    case FAISS_INDEX_HNSWSQ:
        type_str = "HNSWSQ";
        break;
    case FAISS_INDEX_LSH:
        type_str = "LSH";
        break;
    case FAISS_INDEX_BIN_FLAT:
        type_str = "BinFlat";
        break;
    case FAISS_INDEX_BIN_IVFFLAT:
        type_str = "BinIVFFlat";
        break;
    /*
    case FAISS_INDEX_SHARD:
        type_str =  "Shard";
        break;
    case FAISS_INDEX_PROXY:
        type_str =  "Proxy";
        break;
    */
    default:
        type_str = "Unknown";
        break;
    }

    switch (mtype)
    {
    case METRIC_INNER_PRODUCT:
        mtype_str = "innerProduct";
        break;
    case METRIC_L2:
        mtype_str = "l2";
        break;
    case METRIC_L1:
        mtype_str = "l1";
        break;
    case METRIC_Linf:
        mtype_str = "infinity";
        break;
    case METRIC_Lp:
        mtype_str = "lp";
        break;
    case METRIC_Canberra:
        mtype_str = "canberra";
        break;
    case METRIC_BrayCurtis:
        mtype_str = "brayCurtis";
        break;
    case METRIC_JensenShannon:
        mtype_str = "jensenShannon";
    }

    duk_push_object(ctx);

    duk_push_pointer(ctx, idx);
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("faissIdx"));

    duk_push_int(ctx, get_thread_num());
    duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("thread_num"));

    //duk_push_int(ctx, (int)type);
    //duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("type"));

    duk_push_object(ctx); // settings

    duk_push_int(ctx, dim);
    duk_rp_put_prop_string_ro(ctx, -2, "dimension");

    duk_push_number(ctx, rows);
    duk_rp_put_prop_string_ro(ctx, -2, "count");

    //duk_push_string(ctx, type_str);
    //duk_rp_put_prop_string_ro(ctx, -2, "type");

    duk_push_string(ctx, mtype_str);
    duk_rp_put_prop_string_ro(ctx, -2, "metricType");

    duk_push_string(ctx, type_str);
    duk_rp_put_prop_string_ro(ctx, -2, "type");

    if(pqm)
    {
        duk_push_int(ctx, pqm);
        duk_rp_put_prop_string_ro(ctx, -2, "PQm");
    }

    if(pqb)
    {
        duk_push_int(ctx, pqb);
        duk_rp_put_prop_string_ro(ctx, -2, "PQbits");
    }

    if(mapped)
    {
        if(mapped==2)
            duk_push_string(ctx, "IDMap2");
        else
            duk_push_string(ctx, "IDMap");
        duk_rp_put_prop_string_ro(ctx, -2, "map");
    }

    duk_rp_put_prop_string_ro(ctx, -2, "settings");

    duk_push_c_function(ctx, add_fp16, 2);
    duk_put_prop_string(ctx, -2, "addFp16");

    duk_push_c_function(ctx, add_fp32, 2);
    duk_put_prop_string(ctx, -2, "addFp32");

    duk_push_c_function(ctx, save_index, 1);
    duk_put_prop_string(ctx, -2, "save");

    duk_push_c_function(ctx, do_search_fp16, 3);
    duk_put_prop_string(ctx, -2, "searchFp16");

    duk_push_c_function(ctx, do_search_fp32, 3);
    duk_put_prop_string(ctx, -2, "searchFp32");

    if (!faiss_Index_is_trained(idx))
    {
        duk_push_c_function(ctx, new_trainer, 1);
        duk_dup(ctx, -2);
        duk_put_prop_string(ctx, -2, DUK_HIDDEN_SYMBOL("idxthis"));
        duk_put_prop_string(ctx, -2, "trainer");
    }
}

// https://github.com/facebookresearch/faiss/wiki/The-index-factory
static duk_ret_t faiss_open_factory(duk_context *ctx)
{
    const char *desc =
        REQUIRE_STRING(ctx, 0, "faiss.openFactory - first argument must be a String (factory index description)");
    int dim =
        REQUIRE_UINT(ctx, 1, "faiss.openFactory - second argument must be a Positive Integer (vector dimensions)");
    const char *mtype_str = NULL;
    FaissIndex *idx;
    FaissMetricType mtype = METRIC_INNER_PRODUCT;

    if (!duk_is_undefined(ctx, 2))
        mtype_str =
            REQUIRE_STRING(ctx, 2, "faiss.openFactory - third argument, if provided, must be a String (metric type)");

    if (mtype_str)
    {
        if (strcasecmp(mtype_str, "innerProduct") == 0 || strcasecmp(mtype_str, "ip") == 0)
        {
            mtype = METRIC_INNER_PRODUCT;
        }
        else if (strcasecmp(mtype_str, "l2") == 0)
        {
            mtype = METRIC_L2;
        }
        else if (strcasecmp(mtype_str, "l1") == 0 || strcasecmp(mtype_str, "manhattan") == 0 ||
                 strcasecmp(mtype_str, "cityBlock") == 0)
        {
            mtype = METRIC_L1;
        }
        else if (strcasecmp(mtype_str, "linf") == 0 || strcasecmp(mtype_str, "infinity") == 0)
        {
            mtype = METRIC_Linf;
        }
        else if (strcasecmp(mtype_str, "lp") == 0)
        {
            mtype = METRIC_Lp;
        }
        else if (strcasecmp(mtype_str, "canberra") == 0)
        {
            mtype = METRIC_Canberra;
        }
        else if (strcasecmp(mtype_str, "braycurtis") == 0)
        {
            mtype = METRIC_BrayCurtis;
        }
        else if (strcasecmp(mtype_str, "jensenshannon") == 0)
        {
            mtype = METRIC_JensenShannon;
        }
        else
            RP_THROW(ctx, "faiss.openFactory - unknown metric type '%s'", mtype_str);
    }

    int err = faiss_index_factory(&idx, dim, desc, mtype);
    if (err)
    {
        const char *errmsg = faiss_get_last_error();
        RP_THROW(ctx, "Index creation failed for desc='%s': %s", desc, errmsg ? errmsg : "unknown error");
    }

    push_faiss_obj(ctx, idx, mtype, dim, 0.0);

    return 1;
}

static duk_ret_t faiss_openidx_fromfile(duk_context *ctx)
{
    const char *fname = REQUIRE_STRING(ctx, 0, "faiss.openIndexFromFile - argument must be a String (filename)");
    int ro = 0;
    int dim = 0;
    double rows = 0.0;
    FaissMetricType mtype = METRIC_INNER_PRODUCT;
    FaissIndex *idx = NULL;
    const char *err = NULL;
    
    if(!duk_is_undefined(ctx,1))
        ro = REQUIRE_BOOL(ctx, 1, "faiss.openIndexFromFile - Second argument, if defined, must be a Boolean (open readonly w memmap)");

    load_index(fname, &idx, &err, ro);

    if (!idx)
        RP_THROW(ctx, "faiss.openIndexFromFile - %s", err);

    dim = faiss_Index_d(idx);
    mtype = faiss_Index_metric_type(idx);
    rows = (double)faiss_Index_ntotal(idx);

    push_faiss_obj(ctx, idx, mtype, dim, rows);

    return 1;
}

#ifdef LANGTOOLS_MAIN_INCLUDE
static duk_ret_t open_faiss(duk_context *ctx)
#else
duk_ret_t duk_open_module(duk_context *ctx)
#endif
{
    duk_push_object(ctx);

    duk_push_c_function(ctx, faiss_open_factory, 3);
    duk_put_prop_string(ctx, -2, "openFactory");

    duk_push_c_function(ctx, faiss_openidx_fromfile, 2);
    duk_put_prop_string(ctx, -2, "openIndexFromFile");
    return 1;
}
