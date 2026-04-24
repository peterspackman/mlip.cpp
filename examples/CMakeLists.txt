# =============================================================================
# Example programs
# =============================================================================

# Legacy example kept behind an off-by-default flag. It pokes at private
# model internals from the pre-graph-interpreter era and isn't something
# downstream users should build against — prefer cpp_api_test / graph_inference.
option(MLIPCPP_BUILD_LEGACY_EXAMPLES "Build deprecated pre-graph-interpreter examples (simple_inference)" OFF)
if(MLIPCPP_BUILD_LEGACY_EXAMPLES)
    add_executable(simple_inference simple_inference.cpp)
    target_link_libraries(simple_inference PRIVATE mlipcpp ggml fmt::fmt)
    target_include_directories(simple_inference PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

# Internal examples (use private headers, need ggml and fmt)
add_executable(backend_benchmark backend_benchmark.cpp)
target_link_libraries(backend_benchmark PRIVATE mlipcpp ggml fmt::fmt)
target_include_directories(backend_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Graph-based inference using auto-exported models
add_executable(graph_inference ${PROJECT_SOURCE_DIR}/src/bin/graph_inference.cpp)
target_link_libraries(graph_inference PRIVATE mlipcpp ggml fmt::fmt nlohmann_json::nlohmann_json)
target_include_directories(graph_inference PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Public API examples (only use public headers)
add_executable(c_api_test c_api_test.c)
target_link_libraries(c_api_test PRIVATE mlipcpp)
set_target_properties(c_api_test PROPERTIES LINKER_LANGUAGE C)

add_executable(cpp_api_test cpp_api_test.cpp)
target_link_libraries(cpp_api_test PRIVATE mlipcpp)

# Fortran API test (only when Fortran bindings are enabled)
if(MLIPCPP_BUILD_FORTRAN)
    add_executable(fortran_api_test fortran_api_test.f90)
    target_link_libraries(fortran_api_test PRIVATE mlipcpp_fortran)
endif()

# =============================================================================
# Register public-API examples as ctest cases when tests are enabled.
# Each test is pointed at a canonical GGUF fixture (fetched by CI into
# build/tests/gguf/pet-auto.gguf); when the file is absent the test binaries
# exit 77 so CTest reports "skipped" instead of failing.
# =============================================================================
if(MLIPCPP_BUILD_TESTS)
    set(MLIPCPP_TEST_MODEL "${CMAKE_BINARY_DIR}/tests/gguf/pet-auto.gguf"
        CACHE FILEPATH "GGUF model used as a fixture for public-API example tests")

    add_test(NAME c_api_test COMMAND c_api_test "${MLIPCPP_TEST_MODEL}")
    set_tests_properties(c_api_test PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME cpp_api_test COMMAND cpp_api_test "${MLIPCPP_TEST_MODEL}")
    set_tests_properties(cpp_api_test PROPERTIES SKIP_RETURN_CODE 77)

    if(MLIPCPP_BUILD_FORTRAN)
        add_test(NAME fortran_api_test
            COMMAND fortran_api_test "${MLIPCPP_TEST_MODEL}")
        set_tests_properties(fortran_api_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
