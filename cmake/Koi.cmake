OPTION(BUILD_KOI_FROM_SOURCE OFF)

function(get_best_compatible_koi_version KOI_CUDA)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        # Koi provides binaries for these cuda versions when targeting aarch64
        set(SUPPORTED_VERSIONS 13.0 12.6)
    else()
        set(SUPPORTED_VERSIONS 13.0 12.8 11.8)
    endif()

    list(SORT SUPPORTED_VERSIONS COMPARE NATURAL ORDER DESCENDING)
    foreach(SUPPORTED_VERSION IN LISTS SUPPORTED_VERSIONS)
        if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL ${SUPPORTED_VERSION})
            set(${KOI_CUDA} ${SUPPORTED_VERSION} PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Unsupported CUDA toolkit version: ${CUDAToolkit_VERSION}")
endfunction()

function(get_koi_download_hash KOI_DIR KOI_HASH)
    # List of valid hashes.
    set(hash__libkoi__0_6_9__Linux__aarch64__cuda__12_6 "d4a5f2b0656d1c7a1f03eba1dd7a857ba1bb00db3fce23c24df1b4b694fbf1ba")
    set(hash__libkoi__0_6_9__Linux__aarch64__cuda__13_0 "b72eb34317240362d579655045a2e17eb9edb93dd7ed18b89fc98c86fbe054c7")
    set(hash__libkoi__0_6_9__Linux__x86_64__cuda__11_8 "141c894ac821a9981d542b6799ed5d2e7a2c70e6966db02e2078c65b5ba2f66c")
    set(hash__libkoi__0_6_9__Linux__x86_64__cuda__12_8 "d64b0bb823d660e59e1db36e30463f99f2524ad998606745144bfd88c79ace4c")
    set(hash__libkoi__0_6_9__Linux__x86_64__cuda__13_0 "eb2e368f523f64288684101a77a315e444cf543b3d8c5e2ee586371cbb1e4559")
    set(hash__libkoi__0_6_9__Windows__AMD64__cuda__12_8 "7b306ac5ea508050ab16ac8b82acf8017a5e30bf979497c2ea9efbee9a402099")
    set(hash__libkoi__0_6_9__Windows__AMD64__cuda__13_0 "36fdd1481e3a72243c796d23ae196c47cb45fc3da67ca83a173d37e282e8ca59")

    # Do the lookup.
    string(REPLACE "." "_" hash_key ${KOI_DIR})
    string(REPLACE "-" "__" hash_key ${hash_key})
    set(hash_key "hash__${hash_key}")
    if (NOT DEFINED ${hash_key})
        message(FATAL_ERROR "Missing hash for ${KOI_DIR}")
    endif()
    set(${KOI_HASH} ${${hash_key}} PARENT_SCOPE)
endfunction()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)

    set(KOI_VERSION 0.6.9)
    if(BUILD_KOI_FROM_SOURCE)
        set(KOI_DIR "${DORADO_3RD_PARTY_SOURCE}/koi")
        if(NOT EXISTS ${KOI_DIR})
            set(KOI_DIR "${DORADO_3RD_PARTY_DOWNLOAD}/koi")
        endif()
        message(STATUS "Building Koi from source: ${KOI_DIR}")

        if(NOT EXISTS ${KOI_DIR})
            if(DEFINED GITLAB_CI_TOKEN)
                message("Cloning Koi using CI token")
                set(KOI_REPO https://gitlab-ci-token:${GITLAB_CI_TOKEN}@git.oxfordnanolabs.local/machine-learning/koi.git)
            else()
                message("Cloning Koi using ssh")
                set(KOI_REPO git@git.oxfordnanolabs.local:machine-learning/koi.git)
            endif()
            execute_process(
                COMMAND
                    git clone
                        -b v${KOI_VERSION}
                        # TODO: once we drop centos support we can use these instead of a separate submodule update
                        #--depth 1
                        #--recurse-submodules
                        #--shallow-submodules
                        ${KOI_REPO}
                        ${KOI_DIR}
                COMMAND_ERROR_IS_FATAL ANY
            )
            execute_process(
                COMMAND git submodule update --init --checkout
                WORKING_DIRECTORY ${KOI_DIR}
                COMMAND_ERROR_IS_FATAL ANY
            )
        endif()
        add_subdirectory(${KOI_DIR})

    else()
        find_package(CUDAToolkit REQUIRED)
        get_best_compatible_koi_version(KOI_CUDA)
        set(KOI_DIR libkoi-${KOI_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-cuda-${KOI_CUDA})
        get_koi_download_hash(${KOI_DIR} KOI_HASH)

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            download_and_extract(${DORADO_CDN_URL}/${KOI_DIR}.tar.gz ${KOI_DIR} ${KOI_HASH})
            set(KOI_LIBRARY ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/lib/libkoi.a)
        elseif(WIN32)
            download_and_extract(${DORADO_CDN_URL}/${KOI_DIR}.zip ${KOI_DIR} ${KOI_HASH})
            set(KOI_LIBRARY ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/lib/koi.lib)
        endif()
        set(KOI_INCLUDE ${DORADO_3RD_PARTY_DOWNLOAD}/${KOI_DIR}/${KOI_DIR}/include)

        add_library(koi STATIC IMPORTED)
        set_target_properties(koi
            PROPERTIES
                IMPORTED_LOCATION ${KOI_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${KOI_INCLUDE}
        )

    endif()
endif()
