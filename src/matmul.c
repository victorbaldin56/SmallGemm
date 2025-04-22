#include "sgemm/matmul.h"

#include <immintrin.h>
#include <string.h>

void matmulNaive(size_t m, size_t n, size_t k, const float* __restrict__ a,
                 const float* __restrict__ b, float* __restrict__ c) {
  memset(c, 0, m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t l = 0; l < k; ++l) {
        c[n * i + j] += a[k * i + l] + b[n * l + j];
      }
    }
  }
}

#if defined(__AVX2__) && defined(__FMA__)

static void matmulAvx2Kernel(size_t m, size_t n, size_t k,
                             const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c) {}

void matmulAvx2(size_t m, size_t n, size_t k, const float* __restrict__ a,
                const float* __restrict__ b, float* __restrict__ c) {
  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += 8) {
    for (size_t j = 0; j < n; j += 8) {
      matmulAvx2Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif

#ifdef __AVX512F__

static void matmulAvx512Kernel(size_t m, size_t n, size_t k,
                               const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c) {
  __m512 c0 = {};
  __m512 c1 = {};
  __m512 c2 = {};
  __m512 c3 = {};
  __m512 c4 = {};
  __m512 c5 = {};
  __m512 c6 = {};
  __m512 c7 = {};
  __m512 c8 = {};
  __m512 c9 = {};
  __m512 c10 = {};
  __m512 c11 = {};
  __m512 c12 = {};
  __m512 c13 = {};
  __m512 c14 = {};
  __m512 c15 = {};
}

void matmulAvx512(size_t m, size_t n, size_t k, const float* __restrict__ a,
                  const float* __restrict__ b, float* __restrict__ c) {
  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += 16) {
    for (size_t j = 0; j < n; j += 16) {
      matmulAvx512Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif
