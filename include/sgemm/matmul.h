#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matmulNaive(size_t m, size_t n, size_t k, const float* __restrict__ a,
                 const float* __restrict__ b, float* __restrict__ c);
void matmulAvx2(size_t m, size_t n, size_t k, const float* __restrict__ a,
                const float* __restrict__ b, float* __restrict__ c);
void matmulAvx2Parallel(size_t m, size_t n, size_t k,
                        const float* __restrict__ a,
                        const float* __restrict__ b, float* __restrict__ c);
void matmulAvx512(size_t m, size_t n, size_t k, const float* __restrict__ a,
                  const float* __restrict__ b, float* __restrict__ c);
void matmulAvx512Parallel(size_t m, size_t n, size_t k,
                          const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ c);

#ifdef __cplusplus
}
#endif
