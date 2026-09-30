// .rocm-eval/probe/wmma_gemm_test.cu - Self-test probe for RDNA3 WMMA FP16 GEMM vs rocBLAS
#include "src/prefill/wmma_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

struct ShapeTest {
    int64_t n;
    int64_t t;
    int64_t k;
    float beta;
    const char* label;
};

int main() {
    printf("===============================================================================================================\n");
    printf(" RDNA3 WMMA FP16 GEMM vs rocBLAS self-test probe (AMD gfx1100 / ROCm)\n");
    printf("===============================================================================================================\n");

    rocblas_handle handle;
    rocblas_status rst = rocblas_create_handle(&handle);
    if (rst != rocblas_status_success) {
        std::fprintf(stderr, "Failed to create rocblas handle: %d\n", (int)rst);
        return 1;
    }

    ShapeTest shapes[] = {
        // Engine shapes
        {12288, 1023, 2560,  0.0f, "12288x1023x2560 (q_proj-like)"},
        {1023,  2560, 12288, 0.0f, "1023x2560x12288 (o_proj-like)"},
        {2560,  1023, 12288, 0.0f, "2560x1023x12288 (down_proj-like)"},
        {5120,  1023, 2560,  0.0f, "5120x1023x2560  (ssm/gate-like)"},
        {4096,  4096, 4096,  0.0f, "4096x4096x4096  (square 4K)"},
        // Awkward shapes
        {1023,  257,  2560,  0.0f, "1023x257x2560   (awkward T=257)"},
        {1023,  33,   2560,  0.0f, "1023x33x2560    (awkward T=33)"},
        {1023,  7,    2560,  0.0f, "1023x7x2560     (awkward T=7)"},
        {1023,  1,    2560,  0.0f, "1023x1x2560     (awkward T=1)"},
        // Beta == 1 test
        {4096,  1023, 2560,  1.0f, "4096x1023x2560  (beta=1.0 accumulation)"},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    printf("%-35s | %8s %8s | %8s %8s | %10s %10s\n",
           "Shape", "WMMA ms", "WMMA TF", "rocB ms", "rocB TF", "MaxAbsErr", "MaxRelErr");
    printf("---------------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        const int64_t N = s.n;
        const int64_t T = s.t;
        const int64_t K = s.k;
        const int64_t ldy = N;
        const float beta = s.beta;

        const size_t x_bytes = (size_t)T * K * sizeof(uint16_t);
        const size_t w_bytes = (size_t)N * K * sizeof(uint16_t);
        const size_t y_bytes = (size_t)T * ldy * sizeof(float);

        std::vector<uint16_t> h_X(T * K);
        std::vector<uint16_t> h_W(N * K);
        std::vector<float> h_Y_init(T * ldy, 0.0f);

        for (size_t i = 0; i < h_X.size(); ++i) h_X[i] = (uint16_t)__float2half_rn(dist(rng));
        for (size_t i = 0; i < h_W.size(); ++i) h_W[i] = (uint16_t)__float2half_rn(dist(rng));
        if (beta != 0.0f) {
            for (size_t i = 0; i < h_Y_init.size(); ++i) h_Y_init[i] = dist(rng);
        }

        uint16_t *d_X = nullptr, *d_W = nullptr;
        float *d_Y_wmma = nullptr, *d_Y_rocb = nullptr;

        (void)hipMalloc(&d_X, x_bytes);
        (void)hipMalloc(&d_W, w_bytes);
        (void)hipMalloc(&d_Y_wmma, y_bytes);
        (void)hipMalloc(&d_Y_rocb, y_bytes);

        (void)hipMemcpy(d_X, h_X.data(), x_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_W, h_W.data(), w_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_Y_wmma, h_Y_init.data(), y_bytes, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_Y_rocb, h_Y_init.data(), y_bytes, hipMemcpyHostToDevice);

        const float alpha = 1.0f;

        // Run both once for correctness check
        bool wmma_ok = strata_wmma_gemm_f16(d_X, d_W, d_Y_wmma, T, N, K, ldy, beta);
        if (!wmma_ok) {
            printf("%-35s | strata_wmma_gemm_f16 returned false (fallback)\n", s.label);
            (void)hipFree(d_X); (void)hipFree(d_W); (void)hipFree(d_Y_wmma); (void)hipFree(d_Y_rocb);
            continue;
        }

        rst = rocblas_gemm_ex(handle, rocblas_operation_transpose, rocblas_operation_none,
                              (rocblas_int) N, (rocblas_int) T, (rocblas_int) K,
                              &alpha, d_W, rocblas_datatype_f16_r, (rocblas_int) K,
                              d_X, rocblas_datatype_f16_r, (rocblas_int) K,
                              &beta, d_Y_rocb, rocblas_datatype_f32_r, (rocblas_int) ldy,
                              d_Y_rocb, rocblas_datatype_f32_r, (rocblas_int) ldy,
                              rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        if (rst != rocblas_status_success) {
            printf("%-35s | rocblas_gemm_ex returned status %d\n", s.label, (int)rst);
            (void)hipFree(d_X); (void)hipFree(d_W); (void)hipFree(d_Y_wmma); (void)hipFree(d_Y_rocb);
            continue;
        }

        (void)hipDeviceSynchronize();

        // Compare results
        std::vector<float> h_Y_wmma(T * ldy), h_Y_rocb(T * ldy);
        (void)hipMemcpy(h_Y_wmma.data(), d_Y_wmma, y_bytes, hipMemcpyDeviceToHost);
        (void)hipMemcpy(h_Y_rocb.data(), d_Y_rocb, y_bytes, hipMemcpyDeviceToHost);

        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t n = 0; n < N; ++n) {
                float vw = h_Y_wmma[t * ldy + n];
                float vr = h_Y_rocb[n + t * ldy]; // rocblas is col-major
                float diff = std::abs(vw - vr);
                float rel = diff / (std::abs(vr) + 1e-6f);
                if (diff > max_abs_err) max_abs_err = diff;
                if (rel > max_rel_err) max_rel_err = rel;
            }
        }

        // Benchmark timing
        hipEvent_t start, stop;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);
        const int reps = 20;

        // 1. WMMA timing
        (void)hipEventRecord(start);
        for (int i = 0; i < reps; ++i) {
            strata_wmma_gemm_f16(d_X, d_W, d_Y_wmma, T, N, K, ldy, beta);
        }
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);
        float ms_wmma = 0.0f;
        (void)hipEventElapsedTime(&ms_wmma, start, stop);
        ms_wmma /= reps;
        double tflops_wmma = (2.0 * N * T * K) / (ms_wmma * 1e-3) / 1e12;

        // 2. rocBLAS timing
        (void)hipEventRecord(start);
        for (int i = 0; i < reps; ++i) {
            rocblas_gemm_ex(handle, rocblas_operation_transpose, rocblas_operation_none,
                            (rocblas_int) N, (rocblas_int) T, (rocblas_int) K,
                            &alpha, d_W, rocblas_datatype_f16_r, (rocblas_int) K,
                            d_X, rocblas_datatype_f16_r, (rocblas_int) K,
                            &beta, d_Y_rocb, rocblas_datatype_f32_r, (rocblas_int) ldy,
                            d_Y_rocb, rocblas_datatype_f32_r, (rocblas_int) ldy,
                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        }
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);
        float ms_rocb = 0.0f;
        (void)hipEventElapsedTime(&ms_rocb, start, stop);
        ms_rocb /= reps;
        double tflops_rocb = (2.0 * N * T * K) / (ms_rocb * 1e-3) / 1e12;

        printf("%-35s | %8.3f %8.1f | %8.3f %8.1f | %10.2e %10.2e\n",
               s.label, ms_wmma, tflops_wmma, ms_rocb, tflops_rocb, max_abs_err, max_rel_err);

        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
        (void)hipFree(d_X);
        (void)hipFree(d_W);
        (void)hipFree(d_Y_wmma);
        (void)hipFree(d_Y_rocb);
    }

    printf("===============================================================================================================\n");
    rocblas_destroy_handle(handle);
    return 0;
}
