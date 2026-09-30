// compat/hip/cuda_runtime.h - CUDA-spelling shim for the ROCm/HIP build (gfx1100).
//
// The engine's sources keep their CUDA spellings; this header, force-placed first on the include
// path when STRATA_ENABLE_HIP is on, maps the ~90 runtime/cublas symbols the engine uses onto
// their HIP twins.  Anything that is NOT a mechanical rename lives at the bottom as an inline
// function, with the reason written next to it.
//
// Port evidence for every mapping: docs in ROCM-GFX1100-FEASIBILITY.md §3, raw probe logs in
// .rocm-eval/.  Mirrors of the same mappings exist in llama.cpp's ggml-cuda/vendors/hip.h (MIT).
#pragma once

#define STRATA_USE_HIP 1

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstddef>

// ---- types ------------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaDeviceProp hipDeviceProp_t
#define cudaGraphNodeType hipGraphNodeType
#define cudaMemcpyKind hipMemcpyKind

// ---- error / enum values ----------------------------------------------------
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
// ROCm 7.14 has no "optin" attribute; on RDNA3 the per-block max IS the cap (64 KB), which is the
// semantics the engine needs (feasibility report §5: gfx1100 has no >64 KB opt-in).
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphNodeTypeMemcpy hipGraphNodeTypeMemcpy
#define cudaGraphNodeTypeMemset hipGraphNodeTypeMemset

// ---- memory -----------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipMallocHost
#define cudaFreeHost hipFreeHost
#define cudaHostAlloc hipHostAlloc
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo

// ---- device / context -------------------------------------------------------
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
// inline template below: CUDA's overload accepts function pointers directly, hipFuncSetAttribute
// wants const void* and the engine passes template instantiations (e.g. gr_down_multi_kernel).

// ---- streams / events -------------------------------------------------------
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// ---- graphs (launch/upload/instantiate differences are inline functions below) ----
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaLaunchHostFunc hipLaunchHostFunc

// ---- cuBLAS -> hipBLAS ------------------------------------------------------
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B   // hipBLAS spells bf16 R_16B (matches ggml's hip.h)
#define CUDA_R_32F HIPBLAS_R_32F
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSetMathMode hipblasSetMathMode
#define cublasSetWorkspace hipblasSetWorkspace
#define cublasGemmEx hipblasGemmEx

// ---- device intrinsics ------------------------------------------------------
// HIP's __shfl*_sync templates static_assert a 64-bit mask on ROCm >= 7.2 and Strata passes
// 0xffffffffu; the mask itself carries no meaning on AMD (wave-uniform execute), so drop it.
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#define __ballot_sync(mask, ...) __ballot(__VA_ARGS__)

// CUDA's signed-int8 4-byte dot product. RDNA3 has no dot1-insts feature for the signed builtin;
// sudot4 with both operand flags true is the signed x signed dot (same choice as llama.cpp
// common.cuh:712-716).  Every call site must be re-audited for signedness (feasibility report §5).
#ifndef __dp4a
#define __dp4a(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)
#endif

// compiled-out MMA paths use this as a tripwire; same semantics, no header dependency
#ifndef __trap
#define __trap() __builtin_trap()
#endif

// ---- explicitly-rounded float arithmetic ------------------------------------
// CUDA guarantees __fmul_rn/__fadd_rn/__fsub_rn are single IEEE ops with NO FMA contraction.
// HIP defines them as plain a*b / a+b and -ffp-contract=fast fuses them into v_fmac_f32
// (verified: the embedding-gather parity showed 1-2 ulp diffs; ISA showed v_fmac).  A pragma
// float_control(precise) does not survive inlining, so these go through inline asm: one exact
// instruction each, opaque to the optimizer - which is exactly what CUDA's _rn spellings cost.
#if defined(__HIP_DEVICE_COMPILE__)
__device__ inline float strata_fmul_rn(float a, float b) {
    float r; asm("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
__device__ inline float strata_fadd_rn(float a, float b) {
    float r; asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
__device__ inline float strata_fsub_rn(float a, float b) {
    float r; asm("v_sub_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
#define __fmul_rn strata_fmul_rn
#define __fadd_rn strata_fadd_rn
#define __fsub_rn strata_fsub_rn
// __fmaf_rn maps to __builtin_fmaf (correctly rounded, no re-rounding issue) and __fdiv_rn to
// OCML's correctly-rounded division - both already honour their contracts on gfx1100.
#endif

// gfx1100 has no nanosleep facility; these calls back off a doorbell spin-wait, where pacing (not
// precision) is what matters.  s_sleep takes an immediate operand, hence the constant-sleep loop.
// NOTE: the loop must count DOWN from a quotient, not subtract from `ns` until it is zero - with
// ns = 100 the old form ran 100, 68, 36, 4, 4294967268, ... and never hit 0 (the value is invariant
// mod 32 and 2^32 is a multiple of 32), so the FIRST unsatisfied poll spun here forever and never
// re-read its flag.  Every engine wait that was not already satisfied on its first load hung in this
// loop (rocgdb: `strata_hip_nanosleep (ns=2583458180)`, and 2583458180 % 32 == 4).
#if defined(__HIP_DEVICE_COMPILE__)
__device__ inline void strata_hip_nanosleep(unsigned ns) {
    for (unsigned k = (ns + 31u) / 32u; k; --k) __builtin_amdgcn_s_sleep(8);
}
#define __nanosleep(ns) strata_hip_nanosleep((unsigned) (ns))

// Per-byte SIMD-video intrinsics with no gfx1100 instruction.  Branch-free bit-trick emulations
// transcribed from llama.cpp's ggml-cuda/vendors/hip.h (MIT), which uses them for the same
// purpose (IQ sign-grid decode in the i-quant dot products).
typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
__device__ inline int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int&>(c);
}
__device__ inline int __vsub4(const int a, const int b) {
    // avoid per-byte underflow by subtracting in 7 bits, then repair the sign bits
    const unsigned int a_large = (unsigned int) a | 0x80808080u;
    const unsigned int b_small = (unsigned int) b & 0x7f7f7f7fu;
    const unsigned int low7 = a_large - b_small;
    const unsigned int flip = ((unsigned int) a ^ ~(unsigned int) b) & 0x80808080u;
    return (int) (low7 ^ flip);
}
__device__ inline unsigned int __vcmpne4(const unsigned int a, const unsigned int b) {
    const unsigned int x = a ^ b;
    const unsigned int low7 = ((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) & 0x80808080u;
    const unsigned int any = low7 | (x & 0x80808080u);
    return (any >> 7) * 0xffu;
}
#endif

// ---- non-mechanical renames (signature differences) -------------------------

// CUDA 12 dropped the error-node/log-buffer out-params; HIP keeps the 5-arg form and exposes the
// flags form separately.  Overloads cover both call shapes used by the engine.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                       unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                       hipGraphNode_t* error_node, char* log_buffer,
                                       size_t log_size) {
    (void) error_node; (void) log_buffer; (void) log_size;
    return hipGraphInstantiateWithFlags(exec, graph, 0);
}

// No HIP twin.  Only used to label a stalled kernel node in a diagnostic.
// CUDA's signature is (const char** name, const void* func).
inline hipError_t cudaFuncGetName(const char** name, const void* func) {
    (void) func;
    *name = "(hip:unnamed)";
    return hipSuccess;
}

// Accept function pointers / template instantiations like CUDA's overload does.
template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr, value);
}

// No HIP twin: ROCm has no device-flag init call; the flags are honoured (or ignored) via
// hipSetDeviceFlags before the context is created, which is the only effect on ROCm anyway
// (hipDeviceMapHost is documented as ignored there).
inline hipError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int) {
    hipError_t e = hipSetDeviceFlags(deviceFlags);
    if (e != hipSuccess) return e;
    return hipSetDevice(device);
}
