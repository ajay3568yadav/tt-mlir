set(TESTSUITE AllocationTests)

add_mlir_unittest(${TESTSUITE}
    TestAllocation.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRTestUtils
    MLIRTTIRAnalysis
    coverage_config
)
