// compat/hip/math_constants.h - CUDA's float constants; HIP ships the same values as HIP_*.
// The engine only uses CUDART_INF_F.
#pragma once
#include <hip/hip_math_constants.h>
#ifndef CUDART_INF_F
#define CUDART_INF_F HIP_INF_F
#endif
