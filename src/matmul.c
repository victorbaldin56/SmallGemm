#include "sgemm/matmul.h"

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

void matmul1x1(size_t m, size_t n, size_t k, const float* __restrict__ a,
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

static void matmulKernel8x8(size_t m, size_t n, size_t k,
                            const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c) {
  __m256 c0 = {};
  __m256 c1 = {};
  __m256 c2 = {};
  __m256 c3 = {};
  __m256 c4 = {};
  __m256 c5 = {};
  __m256 c6 = {};
  __m256 c7 = {};

  for (size_t l = 0; l < k; ++l) {
    __m256 b_vector = _mm256_load_ps(&b[n * l]);

    c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 0 + l]), b_vector, c0);
    c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 1 + l]), b_vector, c1);
    c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 2 + l]), b_vector, c2);
    c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 3 + l]), b_vector, c3);
    c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 4 + l]), b_vector, c4);
    c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 5 + l]), b_vector, c5);
    c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 6 + l]), b_vector, c6);
    c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[k * 7 + l]), b_vector, c7);
  }

  _mm256_store_ps(&c[n * 0], c0);
  _mm256_store_ps(&c[n * 1], c1);
  _mm256_store_ps(&c[n * 2], c2);
  _mm256_store_ps(&c[n * 3], c3);
  _mm256_store_ps(&c[n * 4], c4);
  _mm256_store_ps(&c[n * 5], c5);
  _mm256_store_ps(&c[n * 6], c6);
  _mm256_store_ps(&c[n * 7], c7);
}

void matmul8x8(size_t m, size_t n, size_t k, const float* __restrict__ a,
               const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0x7) == 0);
  assert((n & 0x7) == 0);
  assert((k & 0x7) == 0);
  assert(((uintptr_t)a & 0x1f) == 0);
  assert(((uintptr_t)b & 0x1f) == 0);
  assert(((uintptr_t)c & 0x1f) == 0);

  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += 8) {
    for (size_t j = 0; j < n; j += 8) {
      matmulKernel8x8(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif

#ifdef __AVX512F__

static void matmulKernel16x8(size_t m, size_t n, size_t k,
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

  for (size_t l = 0; l < k; ++l) {
    __m512 b_vector = _mm512_load_ps(&b[n * l]);

    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 0 + l]), b_vector, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 1 + l]), b_vector, c1);
    c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 2 + l]), b_vector, c2);
    c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 3 + l]), b_vector, c3);
    c4 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 4 + l]), b_vector, c4);
    c5 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 5 + l]), b_vector, c5);
    c6 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 6 + l]), b_vector, c6);
    c7 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 7 + l]), b_vector, c7);
  }

  _mm512_store_ps(&c[n * 0], c0);
  _mm512_store_ps(&c[n * 1], c1);
  _mm512_store_ps(&c[n * 2], c2);
  _mm512_store_ps(&c[n * 3], c3);
  _mm512_store_ps(&c[n * 4], c4);
  _mm512_store_ps(&c[n * 5], c5);
  _mm512_store_ps(&c[n * 6], c6);
  _mm512_store_ps(&c[n * 7], c7);
}

void matmul16x8(size_t m, size_t n, size_t k, const float* __restrict__ a,
                const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0x7) == 0);
  assert((n & 0xf) == 0);
  assert((k & 0xf) == 0);
  assert(((uintptr_t)a & 0x3f) == 0);
  assert(((uintptr_t)b & 0x3f) == 0);
  assert(((uintptr_t)c & 0x3f) == 0);

  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += 8) {
    for (size_t j = 0; j < n; j += 16) {
      matmulKernel16x8(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

static void matmulKernel16x16(size_t m, size_t n, size_t k,
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

  for (size_t l = 0; l < k; ++l) {
    __m512 b_vector = _mm512_load_ps(&b[n * l]);

    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 0 + l]), b_vector, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 1 + l]), b_vector, c1);
    c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 2 + l]), b_vector, c2);
    c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 3 + l]), b_vector, c3);
    c4 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 4 + l]), b_vector, c4);
    c5 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 5 + l]), b_vector, c5);
    c6 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 6 + l]), b_vector, c6);
    c7 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 7 + l]), b_vector, c7);
    c8 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 8 + l]), b_vector, c8);
    c9 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 9 + l]), b_vector, c9);
    c10 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 10 + l]), b_vector, c10);
    c11 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 11 + l]), b_vector, c11);
    c12 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 12 + l]), b_vector, c12);
    c13 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 13 + l]), b_vector, c13);
    c14 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 14 + l]), b_vector, c14);
    c15 = _mm512_fmadd_ps(_mm512_set1_ps(a[k * 15 + l]), b_vector, c15);
  }

  _mm512_store_ps(&c[n * 0], c0);
  _mm512_store_ps(&c[n * 1], c1);
  _mm512_store_ps(&c[n * 2], c2);
  _mm512_store_ps(&c[n * 3], c3);
  _mm512_store_ps(&c[n * 4], c4);
  _mm512_store_ps(&c[n * 5], c5);
  _mm512_store_ps(&c[n * 6], c6);
  _mm512_store_ps(&c[n * 7], c7);
  _mm512_store_ps(&c[n * 8], c8);
  _mm512_store_ps(&c[n * 9], c9);
  _mm512_store_ps(&c[n * 10], c10);
  _mm512_store_ps(&c[n * 11], c11);
  _mm512_store_ps(&c[n * 12], c12);
  _mm512_store_ps(&c[n * 13], c13);
  _mm512_store_ps(&c[n * 14], c14);
  _mm512_store_ps(&c[n * 15], c15);
}

void matmul16x16(size_t m, size_t n, size_t k, const float* __restrict__ a,
                 const float* __restrict__ b, float* __restrict__ c) {
  assert((m & 0x7) == 0);
  assert((m & 0xf) == 0);
  assert((n & 0xf) == 0);
  assert((k & 0xf) == 0);
  assert(((uintptr_t)a & 0x3f) == 0);
  assert(((uintptr_t)b & 0x3f) == 0);
  assert(((uintptr_t)c & 0x3f) == 0);

  memset(c, 0, m * n);
  for (size_t i = 0; i < m; i += 16) {
    for (size_t j = 0; j < n; j += 16) {
      matmulKernel16x16(m, n, k, &a[k * i], &b[j], &c[n * i + j]);
    }
  }
}

#endif
