add_executable(bench bench.cc)
target_link_libraries(bench sgemm openblas benchmark::benchmark_main)
