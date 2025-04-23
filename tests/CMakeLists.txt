add_executable(sgemm_test test.cc)
target_link_libraries(sgemm_test GTest::gtest_main sgemm openblas)
gtest_discover_tests(sgemm_test)
