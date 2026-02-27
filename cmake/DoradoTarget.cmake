# Helper to create a cmake library target.
#
# Example usage:
#
#   dorado_add_library(
#       NAME
#           dorado_example
#       PUBLIC_DIR
#           example
#       SOURCES_PUBLIC
#           blah.h # lives at include/example/blah.h
#       SOURCES_PRIVATE
#           blah.cpp
#           internal.h # implementation detail, not intended for dependents to include
#       SOURCES_TEST
#           test.cpp # testing code has access to private sources too
#       DEPENDS_PUBLIC
#           dorado_torch_utils # blah.h includes torch and hence depends publicly on it
#       DEPENDS_PRIVATE
#           toml11::toml11 # implementation uses toml11 but dependents don't need to know that
#       NO_TORCH # optionally can be added to say that this target doesn't need the torch pch
#   )
#
function(dorado_add_library)
    # Parse the args.
    set(options NO_TORCH)
    set(oneValueArgs NAME PUBLIC_DIR)
    set(multiValueArgs SOURCES_PUBLIC SOURCES_PRIVATE SOURCES_TEST DEPENDS_PUBLIC DEPENDS_PRIVATE)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create the library.
    add_library(${arg_NAME} STATIC)
    foreach (src ${arg_SOURCES_PUBLIC})
        # All public sources must exist in the public include dir.
        target_sources(${arg_NAME} PUBLIC include/${arg_PUBLIC_DIR}/${src})
    endforeach()
    target_sources(${arg_NAME} PRIVATE ${arg_SOURCES_PRIVATE})
    set(targets ${arg_NAME})

    # Create a test executable if tests have been provided.
    if (DEFINED arg_SOURCES_TEST AND NOT DORADO_DISABLE_TESTS)
        set(test_name test_${arg_NAME})
        add_executable(${test_name})
        foreach (src ${arg_SOURCES_TEST})
            # All test files must exist in the test dir.
            target_sources(${test_name} PUBLIC tests/${src})
        endforeach()
        target_link_libraries(${test_name} PRIVATE dorado_tests_common)
        list(APPEND targets ${test_name})
    endif()

    foreach (target IN LISTS targets)
        # Add our dependencies.
        target_link_libraries(${target}
            PUBLIC ${arg_DEPENDS_PUBLIC}
            PRIVATE ${arg_DEPENDS_PRIVATE}
        )

        # Anything linking to us can use our public includes only.
        target_include_directories(${target} PUBLIC include)

        # Anything internal can see what's inside.
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

        # All of our code should compile with warnings enabled.
        enable_warnings_as_errors(${target})

        # Reuse the PCH if it makes use of torch.
        if (arg_NO_TORCH)
            # Validate that this target really doesn't link to torch.
            # We defer this so that links not using this helper function aren't missed.
            cmake_language(EVAL CODE "
                cmake_language(DEFER
                    DIRECTORY ${CMAKE_SOURCE_DIR}
                    CALL check_no_dependency_on_torch [[${target}]]
                )
            ")
        elseif (DORADO_ENABLE_PCH)
            target_link_libraries(${target} PRIVATE dorado_pch)
            target_precompile_headers(${target} REUSE_FROM dorado_pch)
        endif()

        # Enable coverage if told to do so.
        if (GENERATE_TEST_COVERAGE)
            append_coverage_compiler_flags_to_target(${target})
        endif()
    endforeach()
endfunction()
