cmake_minimum_required(VERSION 3.14)
project(SmallGemm)

add_library(sgemm src/matmul.c)
target_include_directories(sgemm PUBLIC include)
