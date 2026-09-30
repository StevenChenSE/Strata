// What does rocBLAS actually achieve for FP16 GEMM with F32 compute on gfx1100?  The engine's dense
// projections (Gemm::f16 -> cublasGemmEx/rocblas) are 43% of the 1K prefill; if rocBLAS does not use
// RDNA3's WMMA units, that is the pp gap.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#include <chrono>
int main() {
    rocblas_handle h; rocblas_create_handle(&h);
    struct S { int64_t m, n, k; const char* label; };
    S shapes[] = {
        {4096, 4096, 4096, "square 4096^3"},
        {12288, 1023, 2560, "q_proj-like (M=12288 N=1023 K=2560)"},
        {1023, 2560, 12288, "o_proj-like (M=1023 N=2560 K=12288)"},
        {5120, 1023, 2560, "ssm/gate-like (M=5120 N=1023 K=2560)"},
    };
    bool warned = false;
    for (auto s : shapes) {
        __half *A, *B; float* C;
        hipMalloc(&A, s.m * s.k * 2); hipMalloc(&B, s.k * s.n * 2); hipMalloc(&C, s.m * s.n * 4);
        hipMemset(A, 0, s.m * s.k * 2); hipMemset(B, 0, s.k * s.n * 2); hipMemset(C, 0, s.m * s.n * 4);
        const float alpha = 1.f, beta = 0.f;
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        auto run = [&] {
            // rocBLAS is column-major; this computes C(m,n) = A(m,k)^T * B(k,n) with the row-major buffers above
            rocblas_status stt = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                            (rocblas_int) s.m, (rocblas_int) s.n, (rocblas_int) s.k,
                            &alpha, A, rocblas_datatype_f16_r, (rocblas_int) s.k,
                            B, rocblas_datatype_f16_r, (rocblas_int) s.k,
                            &beta, C, rocblas_datatype_f32_r, (rocblas_int) s.m,
                            C, rocblas_datatype_f32_r, (rocblas_int) s.m,
                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, (uint32_t) 0);
            if (stt != rocblas_status_success && !warned) { printf("   rocblas status %d for %s\n", (int) stt, s.label); warned = true; }
        };
        run(); hipDeviceSynchronize();
        const int reps = 20;
        hipEventRecord(e0);
        for (int i = 0; i < reps; ++i) run();
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= reps;
        double tflops = 2.0 * s.m * s.n * s.k / (ms * 1e-3) / 1e12;
        printf("%-44s %8.3f ms  %6.1f TFLOPS\n", s.label, ms, tflops);
        hipFree(A); hipFree(B); hipFree(C);
    }
    rocblas_destroy_handle(h);
    return 0;
}
