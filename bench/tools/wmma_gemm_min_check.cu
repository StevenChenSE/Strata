// Minimal, conversion-free correctness check: X = 1.0h (0x3C00), W = 1.0h -> Y must be exactly K.
#include "src/prefill/wmma_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdint>
int main() {
    struct S { int64_t T,N,K; uint16_t xv, wv; double want; const char* label; };
    const uint16_t ONE=0x3C00, TWO=0x4000;   // fp16 1.0 and 2.0
    S shapes[] = {
        {16,16,16, ONE, ONE, 16.0, "16x16x16 single K tile"},
        {16,16,32, ONE, ONE, 32.0, "16x16x32 two K tiles"},
        {16,16,16, ONE, TWO, 32.0, "16x16x16 W=2 -> 32"},
        {33,48,64, ONE, ONE, 64.0, "33x48x64 ragged T,N"},
        {64,64,96, ONE, ONE, 96.0, "64x64x96 (64x64_4w path)"},
        {7,16,32,  ONE, ONE, 32.0, "7x16x32 small T"},
    };
    for (auto s : shapes) {
        std::vector<uint16_t> X((size_t)s.T*s.K, s.xv), W((size_t)s.N*s.K, s.wv);
        uint16_t *dX,*dW; float* dY;
        hipMalloc(&dX,X.size()*2); hipMalloc(&dW,W.size()*2); hipMalloc(&dY,(size_t)s.T*s.N*4);
        hipMemset(dY,0,(size_t)s.T*s.N*4);
        hipMemcpy(dX,X.data(),X.size()*2,hipMemcpyHostToDevice); hipMemcpy(dW,W.data(),W.size()*2,hipMemcpyHostToDevice);
        bool ok = strata_wmma_gemm_f16(dX,dW,dY,s.T,s.N,s.K,s.N,0.0f);
        std::vector<float> Y((size_t)s.T*s.N);
        hipMemcpy(Y.data(),dY,Y.size()*4,hipMemcpyDeviceToHost);
        double bad=0, maxd=0; int nan=0;
        for (float v : Y) { if (v!=v) { ++nan; continue; } double d=((double)v - s.want); if (d<0) d=-d; if (d>maxd) maxd=d; if (d>1e-3) ++bad; }
        printf("%-28s ran=%d  Y[0]=%10.3f want=%7.1f  maxdev=%.3e  wrong=%lld  nan=%d/%zu\n",
               s.label, (int)ok, Y[0], s.want, maxd, (long long)bad, nan, Y.size());
        hipFree(dX); hipFree(dW); hipFree(dY);
    }
    return 0;
}
