# PythonBindings.cmake - CMake module for nanobind Python bindings

function(add_nanobind_stubs TARGET MODULE OUTPUT_DIR)
    if(SKBUILD)
        nanobind_add_stub(
            ${TARGET}_stub
            INSTALL_TIME
            MODULE ${MODULE}
            OUTPUT_PATH mlipcpp
            PYTHON_PATH "\${CMAKE_INSTALL_PREFIX}/mlipcpp"
            MARKER_FILE py.typed
        )
        message(STATUS "Python stub generation enabled for ${MODULE} (install-time)")
    else()
        add_custom_target(${TARGET}_stubs
            COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen
                    -m ${MODULE}
                    -o ${OUTPUT_DIR}
                    -M py.typed
            DEPENDS ${TARGET}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TARGET}>
            COMMENT "Generating Python stub files for ${MODULE}"
            VERBATIM
        )
        message(STATUS "Python stub generation available via '${TARGET}_stubs' target")
    endif()
endfunction()
