#include <immintrin.h>

#include <random>

#include "benchmark/benchmark.h"
#include "openblas/cblas.h"
#include "sgemm/matmul.h"

namespace {

constexpr std::size_t kM = 0x20;
constexpr std::size_t kN = 0x20;
constexpr std::size_t kK = 0x10;

constexpr unsigned kSeed = 0xdeadbeef;

class MatmulBenchmark : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) noexcept override {
    std::mt19937_64 rng(kSeed);
    std::uniform_real_distribution<float> dist(0, 1);
    std::generate(std::begin(a_), std::end(a_), [&]() { return dist(rng); });
    std::generate(std::begin(b_), std::end(b_), [&]() { return dist(rng); });
  }

  template <void (*matmul_func)(std::size_t, std::size_t, std::size_t,
                                const float*, const float*, float*)>
  auto measureMatmul(benchmark::State& state) {
    state.PauseTiming();
    auto start = __rdtsc();
    for (auto i = 0; i < 100000; ++i) {
      matmul_func(kM, kN, kK, a_, b_, c_);
    }
    state.counters["cpu cycles"] = __rdtsc() - start;
  }

  static void matmulOpenblas(std::size_t m, std::size_t n, std::size_t k,
                             const float* a, const float* b,
                             float* c) noexcept {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k,
                b, n, 0.0f, c, n);
  }

 protected:
  alignas(64) float a_[kK * kM];
  alignas(64) float b_[kN * kK];
  alignas(64) float c_[kM * kN];
};
}  // namespace

BENCHMARK_F(MatmulBenchmark, Naive)(benchmark::State& state) {
  measureMatmul<matmulNaive>(state);
}

BENCHMARK_F(MatmulBenchmark, AVX2)(benchmark::State& state) {
  measureMatmul<matmulAvx2>(state);
}

BENCHMARK_F(MatmulBenchmark, AVX2Parallel)(benchmark::State& state) {
  measureMatmul<matmulAvx2Parallel>(state);
}

BENCHMARK_F(MatmulBenchmark, AVX512)(benchmark::State& state) {
  measureMatmul<matmulAvx512>(state);
}

BENCHMARK_F(MatmulBenchmark, AVX512Parallel)(benchmark::State& state) {
  measureMatmul<matmulAvx512Parallel>(state);
}

BENCHMARK_F(MatmulBenchmark, OpenBLAS)(benchmark::State& state) {
  measureMatmul<matmulOpenblas>(state);
}

BENCHMARK_MAIN();
