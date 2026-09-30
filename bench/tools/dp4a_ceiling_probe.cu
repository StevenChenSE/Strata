// How much of a 2-bit expert dot product is the dot, and how much is the unpacking that no kernel
// design can avoid?  Measures pure dp4a throughput against a Q2_0-style block dot (16 bytes -> 64 values).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("%s: %s\n",#x,hipGetErrorString(e));exit(2);} } while(0)
// pure dp4a: 4 independent accumulators for ILP
__global__ void k_dp4a(const uint32_t* a, const uint32_t* b, uint32_t* out, long long iters) {
    uint32_t x=a[threadIdx.x&31], y=b[threadIdx.x&31];
    uint32_t c0=0,c1=0,c2=0,c3=0;
    for (long long i=0;i<iters;++i) { c0=__builtin_amdgcn_sudot4(true,x,true,y,c0,false); c1=__builtin_amdgcn_sudot4(true,y,true,x,c1,false); c2=__builtin_amdgcn_sudot4(true,x,true,y,c2,false); c3=__builtin_amdgcn_sudot4(true,y,true,x,c3,false); }
    out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}
// Q2_0-style: each 16-byte block holds 64 two-bit codes; unpack to int8, then dp4a against int8 activations.
__global__ void k_q2_0(const uint8_t* qs, const int8_t* x, uint32_t* out, long long blocks) {
    uint32_t acc=0;
    for (long long bl=0; bl<blocks; ++bl) {
        const uint8_t* q = qs + (bl & 1023)*16;
        const int8_t*  a = x  + (bl & 1023)*64;
        #pragma unroll
        for (int w=0; w<4; ++w) {                       // 4 bytes -> 16 values
            uint32_t u = q[w];
            // 4 two-bit codes per byte, sign-extended to int8
            uint32_t p0 = (uint32_t)(int32_t)(int8_t)((u & 3) - 0) & 0xff;
            uint32_t p1 = ((u >> 2) & 3) << 8, p2 = ((u >> 4) & 3) << 16, p3 = ((u >> 6) & 3) << 24;
            uint32_t packed = p0 | p1 | p2 | p3;
            int32_t xa; __builtin_memcpy(&xa, a + w*4, 4);
            acc = __builtin_amdgcn_sudot4(true,(int)packed,true,(int)xa,acc,false);
        }
    }
    out[blockIdx.x*blockDim.x+threadIdx.x]=acc;
}
int main() {
    uint32_t *da,*db,*dout; uint8_t* dqs; int8_t* dx;
    CK(hipMalloc(&da,1024)); CK(hipMalloc(&db,1024)); CK(hipMalloc(&dout,1<<22)); CK(hipMalloc(&dqs,1<<16)); CK(hipMalloc(&dx,1<<18));
    CK(hipMemset(da,0x11,1024)); CK(hipMemset(db,0x22,1024)); CK(hipMemset(dqs,0x5a,1<<16)); CK(hipMemset(dx,1,1<<18));
    hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int BLK=2048, THR=256; const long long iters=200000;
    k_dp4a<<<BLK,THR>>>(da,db,dout,100); CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0)); k_dp4a<<<BLK,THR>>>(da,db,dout,iters); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms=0; CK(hipEventElapsedTime(&ms,e0,e1));
    double dp4a_ops = (double)BLK*THR*iters*4;
    printf("pure dp4a     : %8.1f ms  %6.0f G dp4a/s  = %6.1f TOPS (4 MACs each)\n", ms, dp4a_ops/ms/1e6, dp4a_ops*4/ms/1e9);
    const long long blocks=1<<20;   // 1M blocks of 64 values = 64M values
    k_q2_0<<<BLK,THR>>>(dqs,dx,dout,1000); CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0)); k_q2_0<<<BLK,THR>>>(dqs,dx,dout,blocks); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms2=0; CK(hipEventElapsedTime(&ms2,e0,e1));
    double vals=(double)BLK*THR*blocks*64;          // values processed (each with its dp4a share)
    double dp=(double)BLK*THR*blocks*16;            // dp4a instructions issued
    printf("Q2_0 block dot: %8.1f ms  %6.1f G dp4a/s  = %6.1f TOPS equivalent\n", ms2, dp/ms2/1e6, dp*4/ms2/1e9);
    printf("  -> the dot runs at %.0f%% of its pure rate once unpacking is included\n", 100.0*(dp/ms2)/(dp4a_ops/ms));
    return 0;
}
