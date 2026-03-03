add_library(dorado_pch STATIC
    precompiled.cpp
    precompiled.h
)

# We need an exe too since -fPIC is used for libraries and -fPIE is used for
# exes and GCC will complain about the flags being different if we reuse the
# library's PCH with an exe.
add_executable(dorado_pch_exe
    main.cpp
)

foreach(target dorado_pch dorado_pch_exe)
    target_precompile_headers(${target}
        PUBLIC
            precompiled.h
    )
    target_link_libraries(${target}
        PUBLIC
            torch_lib
    )
    enable_warnings_as_errors(${target})
endforeach()
