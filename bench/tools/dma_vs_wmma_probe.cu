// Does a real WMMA GEMM (the load the engine actually runs) stall 2 MB H2D expert DMA, where a plain
// streaming-copy kernel did not?
#include "src/prefill/wmma_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)
int main() {
    const int64_t T=1023, N=12288, K=2560;                 // the engine's q_proj shape
    const size_t C=2046400;                                // the engine's expert blob size
    uint16_t *X,*W; float* Y;
    CK(hipMalloc(&X,(size_t)T*K*2)); CK(hipMalloc(&W,(size_t)N*K*2)); CK(hipMalloc(&Y,(size_t)T*N*4));
    CK(hipMemset(X,0x11,(size_t)T*K*2)); CK(hipMemset(W,0x11,(size_t)N*K*2));
    char* h; char* d; CK(hipHostMalloc(&h,C,hipHostAllocDefault)); CK(hipMalloc(&d,C));
    for (size_t i=0;i<C;i+=4096) h[i]=(char)i;
    hipStream_t copy, load; CK(hipStreamCreate(&copy)); CK(hipStreamCreate(&load));
    hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int NC=512;
    auto copies=[&](const char* label){
        CK(hipEventRecord(e0,copy));
        for (int i=0;i<NC;++i) CK(hipMemcpyAsync(d,h,C,hipMemcpyHostToDevice,copy));
        CK(hipEventRecord(e1,copy)); CK(hipEventSynchronize(e1));
        float ms=0; CK(hipEventElapsedTime(&ms,e0,e1));
        printf("%-34s %7.1f ms  %5.1f GB/s\n", label, ms, (double)NC*C/(ms*1e-3)/1e9);
    };
    copies("idle");
    // the real load: the engine's WMMA GEMM, launched repeatedly on another stream
    for (int i=0;i<20;++i) strata_wmma_gemm_f16(X,W,Y,T,N,K,N,0.0f,(void*)load);
    copies("during gemm_wmma 12288x1023x2560");
    CK(hipStreamSynchronize(load));
    copies("idle again");
    return 0;
}
