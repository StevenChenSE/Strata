// Varied but exactly-representable fp16/bf16 inputs + a host fp64 reference: separates a kernel bug from a
// probe input-generation artifact.  Patterns: +-1, +-0.5, +-2, +-0.25 (all exact in both formats).
#include "src/prefill/wmma_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
static uint16_t h_from_float(float v) {   // exact for our pattern values, in fp16 AND bf16
    uint16_t s = (v < 0) ? 0x8000 : 0;
    float a = std::fabs(v);
    int e = 0; while (a >= 2.0f) { a /= 2.0f; ++e; } while (a < 1.0f) { a *= 2.0f; --e; }
    uint16_t mant = (uint16_t) ((a - 1.0f) * 1024.0f);   // 10 explicit mantissa bits
    uint16_t exp = (uint16_t) (e + 15);
    return (uint16_t) (s | (exp << 10) | mant);
}
int main() {
    const float vals[4] = {1.0f, 0.5f, 2.0f, 0.25f};
    struct S { int64_t T,N,K; const char* label; };
    S shapes[] = {{32,32,64,"32x32x64"}, {12288,1023,2560,"12288x1023x2560 (engine, 4w path)"},
                  {1023,2560,12288,"1023x2560x12288"}};
    for (auto s : shapes) {
        std::vector<uint16_t> X((size_t)s.T*s.K), W((size_t)s.N*s.K);
        std::vector<uint16_t> Xb(X.size()), Wb(W.size());
        for (size_t i=0;i<X.size();++i) { float v=vals[i%4]*((i%8<4)?1.0f:-1.0f); X[i]=h_from_float(v); uint32_t b; memcpy(&b,&v,4); Xb[i]=(uint16_t)(b>>16); }
        for (size_t i=0;i<W.size();++i) { float v=vals[(i/4)%4]*((i%3<2)?1.0f:-1.0f); W[i]=h_from_float(v); uint32_t b; memcpy(&b,&v,4); Wb[i]=(uint16_t)(b>>16); }
        for (const char* kind : {"f16","bf16"}) {
            const std::vector<uint16_t>& XX = (kind[0]=='f') ? X : Xb;
            const std::vector<uint16_t>& WW = (kind[0]=='f') ? W : Wb;
            uint16_t *dX,*dW; float* dY;
            hipMalloc(&dX,X.size()*2); hipMalloc(&dW,W.size()*2); hipMalloc(&dY,(size_t)s.T*s.N*4);
            hipMemcpy(dX,XX.data(),XX.size()*2,hipMemcpyHostToDevice); hipMemcpy(dW,WW.data(),WW.size()*2,hipMemcpyHostToDevice);
            bool ok = (kind[0]=='f') ? strata_wmma_gemm_f16(dX,dW,dY,s.T,s.N,s.K,s.N,0.0f)
                                     : strata_wmma_gemm_bf16(dX,dW,dY,s.T,s.N,s.K,s.N,0.0f);
            std::vector<float> Y((size_t)s.T*s.N);
            hipMemcpy(Y.data(),dY,Y.size()*4,hipMemcpyDeviceToHost);
            hipFree(dX); hipFree(dW); hipFree(dY);
            // host fp64 reference - the pattern values decode exactly in both formats, so one decoder suffices
            int nan=0, bad=0; double maxd=0;
            for (int64_t t=0;t<s.T;++t) for (int64_t n=0;n<s.N;++n) {
                float v = Y[t*s.N+n];
                if (v!=v) { ++nan; continue; }
                double acc=0;
                for (int64_t k=0;k<s.K;++k) {
                    const bool bf = (kind[0] != 'f');
                    auto dec=[bf](uint16_t h){
                        if (bf) { uint16_t m=(uint16_t)(h&0x7F); int e=(h>>7)&0xFF; float f=1.0f+m/128.0f;
                            for (int i=0;i<e-127;++i) f*=2.0f; for (int i=0;i<127-e;++i) f/=2.0f; return (h&0x8000)?-f:f; }
                        uint16_t m=(uint16_t)(h&0x3FF); int e=(h>>10)&0x1F; float f=1.0f+m/1024.0f;
                        for (int i=0;i<e-15;++i) f*=2.0f; for (int i=0;i<15-e;++i) f/=2.0f; return (h&0x8000)?-f:f; };
                    acc += (double)dec(XX[t*s.K+k]) * (double)dec(WW[n*s.K+k]);
                }
                double d=std::fabs((double)v-acc); if (d>maxd) maxd=d; if (d>1e-2) ++bad;
            }
            printf("%-32s %-5s ran=%d  maxdev(vs host fp64)=%.3e  wrong=%d  nan=%d/%zu\n",
                   s.label, kind, (int)ok, maxd, bad, nan, Y.size());
        }
    }
    return 0;
}
