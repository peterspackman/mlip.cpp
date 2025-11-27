# =============================================================================
# mlipcpp library sources
# =============================================================================

# Core components
set(MLIPCPP_CORE_SOURCES
    core/system.cpp
    core/neighbor_list.cpp
    core/gguf_loader.cpp
    core/backend.cpp
    core/ggml_utils.cpp
    core/timer.cpp
)

# Model implementations
set(MLIPCPP_MODEL_SOURCES
    models/model.cpp
    models/pet/pet.cpp
    models/pet/pet_batch.cpp
    models/pet/pet_layers.cpp
)

# I/O and utilities
set(MLIPCPP_IO_SOURCES
    io/xyz.cpp
)

# Utilities
set(MLIPCPP_UTIL_SOURCES
    ggml_attention.cpp
)

# API bindings
set(MLIPCPP_API_SOURCES
    api/c/mlipcpp_api.cpp
    api/cpp/mlipcpp_cpp.cpp
)

# Combine all sources (prepend src/ directory)
set(MLIPCPP_SOURCES)
foreach(src IN LISTS
    MLIPCPP_CORE_SOURCES
    MLIPCPP_MODEL_SOURCES
    MLIPCPP_IO_SOURCES
    MLIPCPP_UTIL_SOURCES
    MLIPCPP_API_SOURCES)
    list(APPEND MLIPCPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()

# Create library target
add_library(mlipcpp ${MLIPCPP_SOURCES})
add_library(mlipcpp::mlipcpp ALIAS mlipcpp)

target_include_directories(mlipcpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ggml_SOURCE_DIR}/include
)

target_link_libraries(mlipcpp
    PRIVATE
        ggml
        $<$<NOT:$<BOOL:${EMSCRIPTEN}>>:fmt::fmt>
)

target_compile_options(mlipcpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

set_target_properties(mlipcpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME mlipcpp
)

# =============================================================================
# Fortran bindings (optional)
# =============================================================================
if(MLIPCPP_BUILD_FORTRAN)
    add_library(mlipcpp_fortran
        api/fortran/mlipcpp.f90
    )
    add_library(mlipcpp::fortran ALIAS mlipcpp_fortran)

    target_link_libraries(mlipcpp_fortran PUBLIC mlipcpp)

    set_target_properties(mlipcpp_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME fortran
    )

    target_include_directories(mlipcpp_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/fortran_modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mlipcpp/fortran>
    )
endif()
