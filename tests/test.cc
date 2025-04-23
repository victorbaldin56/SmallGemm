#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

#include "gtest/gtest.h"
#include "openblas/cblas.h"
#include "sgemm/matmul.h"

namespace {

constexpr std::size_t kM = 0x10;
constexpr std::size_t kN = 0x20;
constexpr std::size_t kK = 0x100;

constexpr unsigned kSeed = 0xdeadbeef;

constexpr float kEpsilon = 1e-5;

void matmulOpenblas(std::size_t m, std::size_t n, std::size_t k, const float* a,
                    const float* b, float* c) noexcept {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b,
              n, 0.0f, c, n);
}
}  // namespace

TEST(sgemm, matmulNaive) {
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<float> dist(0, 1);

  float a[kK * kM];
  float b[kN * kK];
  float c[kM * kN];
  float c_ref[kM * kN];
  std::generate(std::begin(a), std::end(a), [&](){ return dist(rng); });
  std::generate(std::begin(b), std::end(b), [&](){ return dist(rng); });

  matmulNaive(kM, kN, kK, a, b, c);
  matmulOpenblas(kM, kN, kK, a, b, c_ref);
  ASSERT_TRUE(std::equal(
      std::begin(c), std::end(c), std::begin(c_ref),
      [](auto&& x, auto&& y) { return std::fabs(x - y) <= kEpsilon; }));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
