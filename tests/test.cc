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

class MatmulTest : public testing::Test {
 protected:
  void SetUp() noexcept override {
    std::mt19937_64 rng(kSeed);
    std::uniform_real_distribution<float> dist(0, 1);
    std::generate(std::begin(a_), std::end(a_), [&]() { return dist(rng); });
    std::generate(std::begin(b_), std::end(b_), [&]() { return dist(rng); });
    matmulOpenblas(kM, kN, kK, a_, b_, c_ref_);
  }

  auto isResultsEqual() const noexcept {
    return std::equal(
        std::begin(c_), std::end(c_), std::begin(c_ref_),
        [](auto&& x, auto&& y) { return std::fabs(x - y) <= kEpsilon; });
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
  float c_ref_[kM * kN];
};

}  // namespace

TEST_F(MatmulTest, matmulNaive) {
  matmulNaive(kM, kN, kK, a_, b_, c_);
  ASSERT_TRUE(isResultsEqual());
}

TEST_F(MatmulTest, matmulAvx2) {
  matmulAvx2(kM, kN, kK, a_, b_, c_);
  ASSERT_TRUE(isResultsEqual());
}

TEST_F(MatmulTest, matmulAvx2Parallel) {
  matmulAvx2Parallel(kM, kN, kK, a_, b_, c_);
  ASSERT_TRUE(isResultsEqual());
}

TEST_F(MatmulTest, matmulAvx512) {
  matmulAvx512(kM, kN, kK, a_, b_, c_);
  ASSERT_TRUE(isResultsEqual());
}

TEST_F(MatmulTest, matmulAvx512Parallel) {
  matmulAvx512Parallel(kM, kN, kK, a_, b_, c_);
  ASSERT_TRUE(isResultsEqual());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
