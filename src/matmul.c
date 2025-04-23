#include "sgemm/matmul.h"

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "omp.h"

void matmulNaive(size_t m, size_t n, size_t k, const float* __restrict__ a,
                 const float* __restrict__ b, float* __restrict__ c) {
  memset(c, 0, m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t l = 0; l < k; ++l) {
        c[n * i + j] += a[k * i + l] * b[n * l + j];
      }
    }
  }
}

#if defined(__AVX2__) && defined(__FMA__)

#define BLOCK_STRIDE_AVX2 2

static void matmulAvx2Kernel(size_t m, size_t n, size_t k,
                             const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c) {
  __m256 c0 = {};
  __m256 c1 = {};

  for (size_t l = 0; l < k; ++l) {
    __m256 b_vector = _mm256_load_ps(&b[n * l]);

    c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 0 + l]), b_vector, c0);
    c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 1 + l]), b_vector, c1);
  }

  _mm256_store_ps(&c[n * 0], c0);
  _mm256_store_ps(&c[n * 1], c1);
}

void matmulAvx2(size_t m, size_t n, size_t k, const float* __restrict__ a,
                const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0x7) == 0);
  assert((n & 0x7) == 0);
  assert((k & 0x7) == 0);
  assert(((uintptr_t)a & 0x1f) == 0);
  assert(((uintptr_t)b & 0x1f) == 0);
  assert(((uintptr_t)c & 0x1f) == 0);

  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += BLOCK_STRIDE_AVX2) {
    for (size_t j = 0; j < n; j += 8) {
      matmulAvx2Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

void matmulAvx2Parallel(size_t m, size_t n, size_t k,
                        const float* __restrict__ a,
                        const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0x7) == 0);
  assert((n & 0x7) == 0);
  assert((k & 0x7) == 0);
  assert(((uintptr_t)a & 0x1f) == 0);
  assert(((uintptr_t)b & 0x1f) == 0);
  assert(((uintptr_t)c & 0x1f) == 0);

  memset(c, 0, m * n);

#pragma omp parallel for
  for (size_t i = 0; i < m; i += BLOCK_STRIDE_AVX2) {
    for (size_t j = 0; j < n; j += 8) {
      matmulAvx2Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif

#ifdef __AVX512F__

#define BLOCK_STRIDE_AVX512 2

static void matmulAvx512Kernel(size_t m, size_t n, size_t k,
                               const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c) {
  __m512 c0 = {};
  __m512 c1 = {};

  for (size_t l = 0; l < k; ++l) {
    __m512 b_vector = _mm512_load_ps(&b[n * l]);

    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 0 + l]), b_vector, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 1 + l]), b_vector, c1);
  }

  _mm512_store_ps(&c[n * 0], c0);
  _mm512_store_ps(&c[n * 1], c1);
}

void matmulAvx512(size_t m, size_t n, size_t k, const float* __restrict__ a,
                  const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0xf) == 0);
  assert((n & 0xf) == 0);
  assert((k & 0xf) == 0);
  assert(((uintptr_t)a & 0x3f) == 0);
  assert(((uintptr_t)b & 0x3f) == 0);
  assert(((uintptr_t)c & 0x3f) == 0);

  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += BLOCK_STRIDE_AVX512) {
    for (size_t j = 0; j < n; j += 16) {
      matmulAvx512Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

void matmulAvx512Parallel(size_t m, size_t n, size_t k,
                          const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0xf) == 0);
  assert((n & 0xf) == 0);
  assert((k & 0xf) == 0);
  assert(((uintptr_t)a & 0x3f) == 0);
  assert(((uintptr_t)b & 0x3f) == 0);
  assert(((uintptr_t)c & 0x3f) == 0);

  memset(c, 0, m * n);

#pragma omp parallel for
  for (size_t i = 0; i < m; i += BLOCK_STRIDE_AVX512) {
    for (size_t j = 0; j < n; j += 16) {
      matmulAvx512Kernel(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif
