#
# Helper for downloading and setting up prebuilt flash-attention targets.
#
# Manual builds can be used by specifying the install path via DORADO_FLASHATTENTION_PATH.
#

set(FLASHATTENTION_VERSION "fa4-v4.0.0.beta10")



# Work out what we need to download.
unset(FLASHATTENTION_PATCH_SUFFIX)
unset(FLASHATTENTION_HASH)
if (LINUX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
            set(FLASHATTENTION_PATCH_SUFFIX "0-cu13.0-Linux-aarch64")
            set(FLASHATTENTION_HASH "aaa433668c65476d60143e80ef4303b55e7866cad13fc39e38f278e960ebb43a")
        endif()
    elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu13.0-Linux-x86_64")
        set(FLASHATTENTION_HASH "45a16c86fc24c61d5fc45e542cc2c6708cbce2875afae361d7726e8f49cfcac2")
    elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu12.8-Linux-x86_64")
        set(FLASHATTENTION_HASH "d931a7457b1951b6c83f2a9f04e33654266eec437201cbfc369cf8bc2496d1fa")
    endif()
elseif (WIN32)
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu13.0-Windows-x86_64")
        set(FLASHATTENTION_HASH "9097855ecbafff81b9589e375f81f179d1c53fdde6d81cd8d15c64dafc3e6d03")
    else()
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu12.8-Windows-x86_64")
        set(FLASHATTENTION_HASH "22c47360e2aa28988d17ea14a891645cf984d9c1e785a3b48dc71b4f3a6601c6")
    endif()
endif()



# Check for manual build, otherwise we'll download it.
unset(FLASHATTENTION_PATH)
if (DEFINED DORADO_FLASHATTENTION_PATH)
    message(STATUS "Using local flashattention at: ${DORADO_FLASHATTENTION_PATH}")
    set(FLASHATTENTION_PATH "${DORADO_FLASHATTENTION_PATH}")

elseif (DEFINED FLASHATTENTION_PATCH_SUFFIX)
    # Download the library.
    set(dir_name "flashattention-${FLASHATTENTION_VERSION}-${FLASHATTENTION_PATCH_SUFFIX}")
    set(url "${DORADO_CDN_URL}/FLASHATTENTION-${FLASHATTENTION_VERSION}-${FLASHATTENTION_PATCH_SUFFIX}.zip")
    download_and_extract("${url}" "${dir_name}" ${FLASHATTENTION_HASH})
    set(FLASHATTENTION_PATH "${DORADO_3RD_PARTY_DOWNLOAD}/${dir_name}")

endif()



if (DEFINED FLASHATTENTION_PATH)
    # Check that the torch builds match.
    file(READ "${FLASHATTENTION_PATH}/share/torch-hash" FLASHATTENTION_TORCH_HASH)
    if (NOT TORCH_HASH STREQUAL FLASHATTENTION_TORCH_HASH)
        message(FATAL_ERROR "Mismatch between torch builds: flashattention expects ${FLASHATTENTION_TORCH_HASH} but we're using ${TORCH_HASH}")
    endif()

    # Handle platform differences.
    if (WIN32)
        set(lib_prefix "")
        set(lib_suffix "lib")
    else()
        set(lib_prefix "lib")
        set(lib_suffix "a")
    endif()

    # Make the target.
    message(STATUS "Using flashattention: ${FLASHATTENTION_PATH}")
    add_library(dorado_flashattention3 STATIC IMPORTED)
    set_target_properties(dorado_flashattention3
        PROPERTIES
            IMPORTED_LOCATION ${FLASHATTENTION_PATH}/lib/${lib_prefix}flashattention3.${lib_suffix}
            INTERFACE_INCLUDE_DIRECTORIES ${FLASHATTENTION_PATH}/include
    )
    target_link_libraries(dorado_flashattention3
        INTERFACE
            torch_lib
    )
    if (NOT WIN32)
        target_link_libraries(dorado_flashattention3
            INTERFACE
                CUDA::cudart
        )
    endif()
    target_compile_definitions(dorado_flashattention3
        INTERFACE
            DORADO_HAS_FLASHATTENTION3=1
    )

else()
    # Dummy target if we don't support it on this build.
    message(STATUS "No flashattention support")
    add_library(dorado_flashattention3 INTERFACE)
    target_compile_definitions(dorado_flashattention3
        INTERFACE
            DORADO_HAS_FLASHATTENTION3=0
    )

endif()
