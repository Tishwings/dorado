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
            set(FLASHATTENTION_HASH "f7a3bc80a6027f0452f539ce2be6cb8b90531b6e01a07da2a2b648822de268e3")
        endif()
    elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu13.0-Linux-x86_64")
        set(FLASHATTENTION_HASH "cf29e3d70e633d6ad985203771ecf5382d6539211947bb000add5bd1ae498841")
    elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu12.8-Linux-x86_64")
        set(FLASHATTENTION_HASH "efcf6a44fdfd50f60fa00b9f082193591cc3072b8fed9804d748362c922eb69e")
    endif()
elseif (WIN32)
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu13.0-Windows-x86_64")
        set(FLASHATTENTION_HASH "abefe62d7952fdb29ee9e65b3240915a435200823898e2cd72403878381de498")
    else()
        set(FLASHATTENTION_PATCH_SUFFIX "0-cu12.8-Windows-x86_64")
        set(FLASHATTENTION_HASH "8374e9d98d6a4781848ba2ee37401464135d750fb24c73c389ea1418339ae841")
    endif()
endif()



# Check for manual build, otherwise we'll download it.
unset(FLASHATTENTION_PATH)
if (DEFINED DORADO_FLASHATTENTION_PATH)
    message(STATUS "Using local flashattention at: ${DORADO_FLASHATTENTION_PATH}")
    set(FLASHATTENTION_PATH "${DORADO_FLASHATTENTION_PATH}")

elseif (DEFINED FLASHATTENTION_PATCH_SUFFIX)
    # Download the library.
    set(filename "flashattention-${FLASHATTENTION_VERSION}-${FLASHATTENTION_PATCH_SUFFIX}")
    set(url "${DORADO_CDN_URL}/${filename}.zip")
    download_and_extract("${url}" "${filename}" ${FLASHATTENTION_HASH})
    set(FLASHATTENTION_PATH "${DORADO_3RD_PARTY_DOWNLOAD}/${filename}")

endif()



unset(DORADO_HAS_FLASHATTENTION3)
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

    message(STATUS "Using flashattention: ${FLASHATTENTION_PATH}")
    set(DORADO_HAS_FLASHATTENTION3 TRUE)

    # Make the target.
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

else()
    message(STATUS "No flashattention support")
    set(DORADO_HAS_FLASHATTENTION3 FALSE)

    # Dummy target if we don't support it on this build.
    add_library(dorado_flashattention3 INTERFACE)

endif()

target_compile_definitions(dorado_flashattention3
    INTERFACE
        DORADO_HAS_FLASHATTENTION3=$<BOOL:${DORADO_HAS_FLASHATTENTION3}>
)
