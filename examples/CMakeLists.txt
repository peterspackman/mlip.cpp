# =============================================================================
# Example programs
# =============================================================================

# Internal examples (use private headers, need ggml and fmt)
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE mlipcpp ggml fmt::fmt)
target_include_directories(simple_inference PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_executable(backend_benchmark backend_benchmark.cpp)
target_link_libraries(backend_benchmark PRIVATE mlipcpp ggml fmt::fmt)
target_include_directories(backend_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)

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
