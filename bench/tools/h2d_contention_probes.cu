// bench/tools/h2d_contention_probes.cu - three probes that together bound why the expert stream runs at
// 11.3 GB/s in the engine while the link measures 13.5 GB/s standalone (2 MB blobs):
//   1) H2D under a saturating SM-side VRAM load      -> no effect (13.2/13.5/13.6 GB/s)
//   2) scattered vs cached source regions in an arena -> no effect (13.5 vs 13.1)
//   3) 8 rotating destination slots, +D2D traffic     -> 13.5, and 12.7 with concurrent D2D (-6%)
// Build each section separately: hipcc -O3 --offload-arch=gfx1100 <file> -o /tmp/probe
// (they are concatenated here for the record; compile by extracting one main() at a time).

// Does an SM-side VRAM-bandwidth load (what the prefill's GEMMs create) slow down 2 MB H2D expert DMA?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)
__global__ void bw_load(const float4* __restrict__ s, float4* __restrict__ d, size_t n4) {
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n4; i += gridDim.x*(size_t)blockDim.x)
        d[i] = s[i];
}
int main() {
    const size_t C = 2u<<20, N = 1024, V = 256u<<20;
    char *h=nullptr, *d=nullptr; float4 *a=nullptr, *b=nullptr;
    CK(hipHostMalloc(&h, C, hipHostAllocDefault)); CK(hipMalloc(&d, C));
    CK(hipMalloc(&a, V)); CK(hipMalloc(&b, V));
    for (size_t i=0;i<C;i+=4096) h[i]=(char)i;
    hipStream_t copy, load; CK(hipStreamCreate(&copy)); CK(hipStreamCreate(&load));
    hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto run=[&](const char* label){
        CK(hipEventRecord(e0, copy));
        for (size_t i=0;i<N;++i) CK(hipMemcpyAsync(d, h, C, hipMemcpyHostToDevice, copy));
        CK(hipEventRecord(e1, copy)); CK(hipEventSynchronize(e1));
        float ms=0; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-26s %7.1f ms  %5.1f GB/s   (%zu x 2 MB)\n", label, ms, (double)N*C/(ms*1e-3)/1e9, N);
    };
    run("idle");
    for (int r=0;r<60;++r) bw_load<<<8192,256,0,load>>>(a, b, V/16);   // ~60 x 512 MB moved = SM/VRAM load
    run("under SM VRAM load");
    CK(hipStreamSynchronize(load));
    run("idle again");
    return 0;
}
// Is the engine's 11.3 GB/s (vs the 13.5 GB/s probe peak) explained by *scattered* reads from a large pinned
// arena instead of the same 2 MB repeatedly (host L3/TLB behaviour)?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)
int main() {
    const size_t C = 2046400, N = 1024, ARENA = 6ull<<30;     // the engine's blob size, and a 6 GiB arena
    char *h=nullptr, *d=nullptr;
    CK(hipHostMalloc(&h, ARENA, hipHostAllocDefault)); CK(hipMalloc(&d, C));
    for (size_t i=0;i<ARENA;i+=4096) h[i]=(char)i;            // fault the pages in
    hipStream_t s; CK(hipStreamCreate(&s)); hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto run=[&](const char* label, bool scatter, size_t fixed_bytes){
        std::mt19937_64 rng(1234);
        std::uniform_int_distribution<size_t> off(0, (ARENA - C - fixed_bytes)/C);
        CK(hipEventRecord(e0, s));
        for (size_t i=0;i<N;++i) {
            const size_t o = scatter ? off(rng)*C : fixed_bytes;
            CK(hipMemcpyAsync(d, h + o, C, hipMemcpyHostToDevice, s));
        }
        CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
        float ms=0; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-40s %7.1f ms  %5.1f GB/s\n", label, ms, (double)N*C/(ms*1e-3)/1e9);
    };
    run("same 2 MB region repeatedly (probe)", false, 0);
    run("scattered 2 MB regions in a 6 GiB arena", true, 0);
    run("same region again", false, 0);
    return 0;
}
// Two remaining differences from the engine's DMA: rotating destination slots, and other traffic sharing the
// copy engine.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)
int main() {
    const size_t C = 2046400, N = 1024; const int SLOTS = 8;
    char *h=nullptr; char **dst=nullptr; char *src2=nullptr, *dst2=nullptr;
    CK(hipHostMalloc(&h, 4ull<<30, hipHostAllocDefault)); for (size_t i=0;i<(4ull<<30);i+=4096) h[i]=(char)i;
    dst=(char**)malloc(sizeof(char*)*SLOTS);
    for (int i=0;i<SLOTS;++i) CK(hipMalloc(&dst[i], C));
    CK(hipMalloc(&src2, 1u<<20)); CK(hipMalloc(&dst2, 1u<<20));
    hipStream_t s, s2; CK(hipStreamCreate(&s)); CK(hipStreamCreate(&s2));
    hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto run=[&](const char* label, bool rotate, bool extra){
        int reps = extra ? 200 : 0;
        if (extra) for (int i=0;i<reps;++i) CK(hipMemcpyAsync(dst2, src2, 1u<<20, hipMemcpyDeviceToDevice, s2));
        CK(hipEventRecord(e0, s));
        for (size_t i=0;i<N;++i) CK(hipMemcpyAsync(rotate ? dst[i%SLOTS] : dst[0], h + (i%64)*C, C, hipMemcpyHostToDevice, s));
        CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
        float ms=0; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("%-46s %7.1f ms  %5.1f GB/s\n", label, ms, (double)N*C/(ms*1e-3)/1e9);
        if (extra) CK(hipStreamSynchronize(s2));
    };
    run("single destination slot", false, false);
    run("8 rotating destination slots", true, false);
    run("8 rotating slots + concurrent D2D traffic", true, true);
    run("single destination again", false, false);
    return 0;
}
