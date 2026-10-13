# mlip.cpp as a metatomic plugin: one shared module for mta_load_plugin.
#
# It links libmetatomic (and through it metatensor) dynamically -- the host
# process's copy -- and carries mlipcpp and ggml inside, their symbols kept to
# itself so a host with its own ggml does not see them.
#
# metatomic and metatensor are found with find_package; their Python wheels
# ship the CMake configs:
#   -DCMAKE_PREFIX_PATH="$(python -c 'import metatomic; print(metatomic.utils.cmake_prefix_path)');$(python -c 'import metatensor; print(metatensor.utils.cmake_prefix_path)')"

find_package(metatomic CONFIG REQUIRED)

add_library(mlip_metatomic MODULE mlip_metatomic.cpp)
# The plugin builds on mlipcpp's internals (GraphModel, the GGUF loader).
target_include_directories(mlip_metatomic PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${ggml_SOURCE_DIR}/include
)
target_link_libraries(mlip_metatomic PRIVATE mlipcpp ggml metatomic)
target_compile_features(mlip_metatomic PRIVATE cxx_std_17)

# The same file name everywhere, so engines can be pointed at it plainly.
set_target_properties(mlip_metatomic PROPERTIES
    OUTPUT_NAME mlipcpp-metatomic
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Only mta_plugin_init leaves the module; the static mlipcpp and ggml inside
# stay private.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(mlip_metatomic PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Find libmetatomic where it was found at build time.
get_target_property(_mlip_mta_location metatomic::shared IMPORTED_LOCATION)
if(_mlip_mta_location)
    get_filename_component(_mlip_mta_dir "${_mlip_mta_location}" DIRECTORY)
    set_target_properties(mlip_metatomic PROPERTIES
        BUILD_RPATH "${_mlip_mta_dir}"
        INSTALL_RPATH "${_mlip_mta_dir}")
endif()

if(MLIPCPP_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS mlip_metatomic LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
