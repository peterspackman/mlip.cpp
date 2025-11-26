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

# Create symlink to model file for CTest
add_custom_command(TARGET test_pet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/../pet-mad.gguf
        ${CMAKE_CURRENT_BINARY_DIR}/pet-mad.gguf
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

add_custom_command(TARGET test_pet_gradients POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/../pet-mad.gguf
        ${CMAKE_CURRENT_BINARY_DIR}/pet-mad.gguf
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

# Register with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_pet)
catch_discover_tests(test_pet_gradients)
catch_discover_tests(test_io)
catch_discover_tests(test_neighbor_list)
catch_discover_tests(test_system)
