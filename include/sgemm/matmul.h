#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matmul1x1(size_t m, size_t n, size_t k, const float* __restrict__ a,
               const float* __restrict__ b, float* __restrict__ c);
void matmul8x8(size_t m, size_t n, size_t k, const float* __restrict__ a,
               const float* __restrict__ b, float* __restrict__ c);
void matmul16x8(size_t m, size_t n, size_t k, const float* __restrict__ a,
                const float* __restrict__ b, float* __restrict__ c);
void matmul16x16(size_t m, size_t n, size_t k, const float* __restrict__ a,
                 const float* __restrict__ b, float* __restrict__ c);

#ifdef __cplusplus
}
#endif
