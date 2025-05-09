#include <immintrin.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "openblas/cblas.h"
#include "sgemm/matmul.h"

namespace {

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t n) {
    auto ptr = std::aligned_alloc(Alignment, n * sizeof(T));
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept { std::free(p); }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  using is_always_equal = std::true_type;
};

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&,
                const AlignedAllocator<T2, A2>&) noexcept {
  return A1 == A2;
}

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>& a1,
                const AlignedAllocator<T2, A2>& a2) noexcept {
  return !(a1 == a2);
}

class MatmulBenchmark : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) noexcept override {
    m_ = state.range(0);
    n_ = state.range(1);
    k_ = state.range(2);

    a_.resize(k_ * m_);
    b_.resize(n_ * k_);
    c_.resize(m_ * n_);

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
      matmul_func(m_, n_, k_, a_.data(), b_.data(), c_.data());
    }
    state.counters["cpu cycles"] = __rdtsc() - start;
  }

  static void matmulOpenblas(std::size_t m, std::size_t n, std::size_t k,
                             const float* a, const float* b,
                             float* c) noexcept {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k,
                b, n, 0.0f, c, n);
  }

 private:
  static constexpr unsigned kSeed = 0xdeadbeef;
  using AlignedMatrix = std::vector<float, AlignedAllocator<float, 64>>;

 protected:
  std::size_t m_;
  std::size_t n_;
  std::size_t k_;
  AlignedMatrix a_;
  AlignedMatrix b_;
  AlignedMatrix c_;
};
}  // namespace

#define REGISTER_MATMUL_BENCHMARK(name, func)                           \
  BENCHMARK_DEFINE_F(MatmulBenchmark, name)(benchmark::State & state) { \
    measureMatmul<func>(state);                                         \
  }                                                                     \
  BENCHMARK_REGISTER_F(MatmulBenchmark, name)                           \
      ->Args({0x10, 0x10, 0x10})                                        \
      ->Args({0x20, 0x20, 0x10})                                        \
      ->Args({0x40, 0x40, 0x20})                                        \
      ->Args({0x40, 0x40, 0x40});

REGISTER_MATMUL_BENCHMARK(1x1, matmul1x1);
REGISTER_MATMUL_BENCHMARK(8x8, matmul8x8);
REGISTER_MATMUL_BENCHMARK(16x8, matmul16x8);
REGISTER_MATMUL_BENCHMARK(16x16, matmul16x16);
REGISTER_MATMUL_BENCHMARK(OpenBLAS, matmulOpenblas);

BENCHMARK_MAIN();
