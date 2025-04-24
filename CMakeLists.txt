cmake_minimum_required(VERSION 3.14)
project(SmallGemm)

add_library(sgemm src/matmul.c)
target_include_directories(sgemm PUBLIC include)

find_package(benchmark REQUIRED)
add_subdirectory(bench)

option(ENABLE_TESTS "Enable testing" ON)
if (ENABLE_TESTS)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
  add_subdirectory(tests)
endif()
