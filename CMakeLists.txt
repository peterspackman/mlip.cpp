cmake_minimum_required(VERSION 3.18)
project(mlipcpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set output directories (can be overridden by user)
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# =============================================================================
# Options
# =============================================================================
option(MLIPCPP_BUILD_EXAMPLES "Build examples" ON)
option(MLIPCPP_BUILD_TESTS "Build tests" OFF)
option(MLIPCPP_BUILD_FORTRAN "Build Fortran bindings" OFF)
option(MLIPCPP_BUILD_PYTHON "Build Python bindings" OFF)
option(MLIPCPP_INSTALL "Generate install target" OFF)
option(MLIPCPP_USE_CUDA "Enable CUDA backend via ggml" OFF)
option(MLIPCPP_USE_HIP "Enable HIP/ROCm backend via ggml (AMD)" OFF)
option(MLIPCPP_USE_METAL "Enable Metal backend via ggml" OFF)
option(MLIPCPP_USE_VULKAN "Enable Vulkan backend via ggml" OFF)
option(MLIPCPP_USE_SYCL "Enable SYCL backend via ggml (Intel)" OFF)
option(MLIPCPP_USE_CANN "Enable CANN backend via ggml (Huawei Ascend)" OFF)
option(MLIPCPP_USE_BLAS "Enable BLAS acceleration via ggml" OFF)
option(MLIPCPP_USE_SYSTEM_FMT "Use system-installed fmtlib instead of bundled" OFF)

# Enable Fortran if requested
if(MLIPCPP_BUILD_FORTRAN)
    enable_language(Fortran)
endif()

# Enable Python bindings via scikit-build
if(SKBUILD)
    set(MLIPCPP_BUILD_PYTHON ON)
endif()

if(MLIPCPP_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    message(STATUS "Python bindings enabled, setting CMAKE_POSITION_INDEPENDENT_CODE")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# =============================================================================
# Dependencies (via CPM)
# =============================================================================
set(CPM_DOWNLOAD_VERSION 0.40.2)
if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# ggml (forked version with gradient support)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Configure ggml backend options based on mlipcpp options
if(MLIPCPP_USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
else()
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_SYCL)
    set(GGML_SYCL ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_CANN)
    set(GGML_CANN ON CACHE BOOL "" FORCE)
endif()
if(MLIPCPP_USE_BLAS)
    set(GGML_BLAS ON CACHE BOOL "" FORCE)
endif()

CPMAddPackage(
    NAME ggml
    GITHUB_REPOSITORY peterspackman/ggml
    GIT_TAG d850966f  # Fix out_prod backward pass for non-contiguous tensors
    EXCLUDE_FROM_ALL YES
)

# fmt
if(MLIPCPP_USE_SYSTEM_FMT)
    find_package(fmt REQUIRED)
else()
    CPMAddPackage(
        NAME fmt
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.0.2
        OPTIONS
            "FMT_INSTALL OFF"
        EXCLUDE_FROM_ALL YES
    )
endif()

# =============================================================================
# Library
# =============================================================================
add_subdirectory(src)

# =============================================================================
# Examples
# =============================================================================
if(MLIPCPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Tests
# =============================================================================
if(MLIPCPP_BUILD_TESTS)
    CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.7.1
        OPTIONS
            "CATCH_BUILD_TESTING OFF"
            "CATCH_INSTALL_DOCS OFF"
            "CATCH_INSTALL_EXTRAS OFF"
        EXCLUDE_FROM_ALL YES
    )

    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Python Bindings
# =============================================================================
if(MLIPCPP_BUILD_PYTHON)
    CPMAddPackage(
        NAME nanobind
        GITHUB_REPOSITORY wjakob/nanobind
        GIT_TAG v2.4.0
        EXCLUDE_FROM_ALL YES
    )

    find_package(nanobind CONFIG REQUIRED)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PythonBindings.cmake)

    nanobind_add_module(_mlipcpp
        NB_STATIC
        LTO
        STABLE_ABI
        ${CMAKE_CURRENT_SOURCE_DIR}/src/api/python/mlipcpp_bindings.cpp
    )

    target_link_libraries(_mlipcpp PRIVATE mlipcpp fmt::fmt)
    target_include_directories(_mlipcpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(_mlipcpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

    install(TARGETS _mlipcpp LIBRARY DESTINATION mlipcpp)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/mlipcpp/
        DESTINATION mlipcpp
        FILES_MATCHING PATTERN "*.py"
    )

    add_nanobind_stubs(_mlipcpp _mlipcpp ${CMAKE_CURRENT_BINARY_DIR}/stubs)
endif()

# =============================================================================
# Installation
# =============================================================================
if(MLIPCPP_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install headers
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )

    # Targets to export
    set(MLIPCPP_EXPORT_TARGETS mlipcpp)
    if(MLIPCPP_BUILD_FORTRAN)
        list(APPEND MLIPCPP_EXPORT_TARGETS mlipcpp_fortran)

        # Install Fortran module files
        install(DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlipcpp/fortran
            FILES_MATCHING PATTERN "*.mod"
        )
    endif()

    # Install library targets
    install(TARGETS ${MLIPCPP_EXPORT_TARGETS}
        EXPORT mlipcppTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export targets
    install(EXPORT mlipcppTargets
        FILE mlipcppTargets.cmake
        NAMESPACE mlipcpp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlipcpp
    )

    # Package config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlipcppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/mlipcppConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlipcpp
    )

    # Version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/mlipcppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Install config files
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mlipcppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mlipcppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlipcpp
    )

    # Export from build tree (for use without installing)
    export(EXPORT mlipcppTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/mlipcppTargets.cmake
        NAMESPACE mlipcpp::
    )
endif()
