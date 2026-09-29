// compat/hip/cuda_fp16.h - route CUDA fp16 spellings to HIP's.
// Strata's kernels use: __half, __half2, __float2half, __float2half_rn, __half2float,
// __half22float2, __halves2half2, __half_as_ushort, __ushort_as_half, __hsub2.
// All exist in HIP's hip_fp16.h with the same semantics (round-to-nearest-even conversions);
// the bitwise round-trips (__half_as_ushort) are the same bit patterns.
#pragma once
#include <hip/hip_fp16.h>
