# Test executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# PET comprehensive test
add_executable(test_pet
    test_pet.cpp
)

target_link_libraries(test_pet
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_pet
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# GGML layout test (standalone, no Catch2)
add_executable(test_ggml_layout
    test_ggml_layout.cpp
)

target_link_libraries(test_ggml_layout
    PRIVATE
        ggml
        fmt::fmt
)

target_include_directories(test_ggml_layout
    PRIVATE
        ${ggml_SOURCE_DIR}/include
)

# PET gradient test
add_executable(test_pet_gradients
    test_pet_gradients.cpp
)

target_link_libraries(test_pet_gradients
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_pet_gradients
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# IO tests (no model file needed)
add_executable(test_io
    test_io.cpp
)

target_link_libraries(test_io
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        fmt::fmt
)

# Neighbor list tests (no model file needed)
add_executable(test_neighbor_list
    test_neighbor_list.cpp
)

target_link_libraries(test_neighbor_list
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        fmt::fmt
)

# System/Cell tests (no model file needed)
add_executable(test_system
    test_system.cpp
)

target_link_libraries(test_system
    PRIVATE
        mlipcpp
        fmt::fmt
        Catch2::Catch2WithMain
)

# Graph interpreter tests
add_executable(test_graph_interpreter
    test_graph_interpreter.cpp
)

target_link_libraries(test_graph_interpreter
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_graph_interpreter
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# Auto-export vs manual comparison tests
add_executable(test_auto_vs_manual
    test_auto_vs_manual.cpp
)

target_link_libraries(test_auto_vs_manual
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_auto_vs_manual
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
)

# Register with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_pet)
catch_discover_tests(test_pet_gradients)
catch_discover_tests(test_io)
catch_discover_tests(test_neighbor_list)
catch_discover_tests(test_system)
catch_discover_tests(test_graph_interpreter)
catch_discover_tests(test_auto_vs_manual)

# Full export integration test
add_executable(test_full_export
    test_full_export.cpp
)

target_link_libraries(test_full_export
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_full_export
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

catch_discover_tests(test_full_export)

# GraphModel tests
add_executable(test_graph_model
    test_graph_model.cpp
)

target_link_libraries(test_graph_model
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_graph_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

catch_discover_tests(test_graph_model)

# Backend-selection unit tests (use only the public C API)
add_executable(test_backend_selection
    test_backend_selection.cpp
)

target_link_libraries(test_backend_selection
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
)

target_compile_definitions(test_backend_selection
    PRIVATE
        MLIPCPP_TEST_MODEL_DEFAULT="${CMAKE_BINARY_DIR}/tests/gguf/pet-auto.gguf"
)

catch_discover_tests(test_backend_selection)
