// tools/iq_fixture_gen.c - generate the iq_parity fixtures from ggml itself (the ground truth
// the kernels are checked against): quantize random rows with ggml's own quantizers, dequantize
// with ggml's own reference row decoders, and write both.
//
//   cc -O2 -I <llama.cpp>/ggml/src -I <llama.cpp>/ggml/include tools/iq_fixture_gen.c \
//        -L build-hip/ggml/src -lggml-base -lm -o build-hip/iq_fixture_gen
//   ./build-hip/iq_fixture_gen logs/iq_fixture
#include "ggml.h"
#include "ggml-quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char* name; enum ggml_type type; } Spec;
static const Spec specs[] = {
    {"IQ2_XXS", GGML_TYPE_IQ2_XXS}, {"IQ2_XS", GGML_TYPE_IQ2_XS}, {"IQ2_S", GGML_TYPE_IQ2_S},
    {"IQ3_XXS", GGML_TYPE_IQ3_XXS}, {"IQ3_S", GGML_TYPE_IQ3_S},   {"IQ1_M", GGML_TYPE_IQ1_M},
    {"IQ4_NL", GGML_TYPE_IQ4_NL},   {"IQ4_XS", GGML_TYPE_IQ4_XS}, {"Q2_0", GGML_TYPE_Q2_0},
    {"Q3_K", GGML_TYPE_Q3_K},
};

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "logs/iq_fixture";
    const int rows = 8, cols = 512;              // rows*cols % 256 == 0, cols % 32 == 0
    float* src = malloc(sizeof(float) * rows * cols);
    float* ref = malloc(sizeof(float) * rows * cols);
    float* imatrix = malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; ++i) imatrix[i] = 1.0f;   // uniform importance: valid, quality is irrelevant for parity
    uint8_t* q = malloc((size_t) rows * cols * 4);
    for (size_t i = 0; i < (size_t) rows * cols; ++i)
        src[i] = (float) (((double) rand() / RAND_MAX) * 2.0 - 1.0) * (0.5f + (float) (i % 7) / 7.0f);

    for (unsigned s = 0; s < sizeof specs / sizeof specs[0]; ++s) {
        const size_t row_bytes = ggml_row_size(specs[s].type, cols);
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", dir, specs[s].name);
        FILE* f = fopen(path, "wb");
        if (!f) { perror(path); return 1; }
        const int32_t hdr[3] = { (int32_t) specs[s].type, rows, cols };
        fwrite(hdr, 4, 3, f);
        int bad = 0;
        for (int r = 0; r < rows; ++r) {
            memset(q, 0, row_bytes);
            const int64_t nq = ggml_quantize_chunk(specs[s].type, src + (size_t) r * cols, q, 0, 1, cols, imatrix);
            if (nq != (int64_t) row_bytes) { fprintf(stderr, "%s: quantize_chunk wrote %lld, row %zu\n", specs[s].name, (long long) nq, row_bytes); ++bad; break; }
            fwrite(q, 1, row_bytes, f);
            ggml_get_type_traits(specs[s].type)->to_float(q, ref + (size_t) r * cols, cols);   // ggml's reference row decoder
        }
        fclose(f);
        snprintf(path, sizeof path, "%s/%s.f32", dir, specs[s].name);
        f = fopen(path, "wb");
        fwrite(ref, 4, (size_t) rows * cols, f);
        fclose(f);
        printf("%-8s type %2d  %d x %d  row %zu B  %s\n", specs[s].name, (int) specs[s].type, rows, cols,
               row_bytes, bad ? "QUANT-ERROR" : "ok");
    }
    free(src); free(ref); free(q);
    return 0;
}
