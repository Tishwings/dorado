if(TARGET OpenSSL::SSL)
  return()
endif()

if(ECM_ENABLE_SANITIZERS)
  set(OPENSSL_USE_STATIC_LIBS FALSE)
else()
  set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

if(NOT DEFINED OPENSSL_ROOT_DIR)
    set(OPENSSL_VERSION 3.5.5)
    if(APPLE)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-macos-aarch64.zip
            openssl-${OPENSSL_VERSION}-macos-aarch64
            "afba0cc43232833343db607d2ed999cfd9af43df1e889b8edb2ebd6766844ae2"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-macos-aarch64)
    elseif(WIN32)
        download_and_extract(
            ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-win.zip
            openssl-${OPENSSL_VERSION}-win
            "200bd6d518c58b74d30319a382df32673ce8b3f28aa527f4dccb46bf3bfd2392"
        )
        set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-win)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-x86_64.zip
                openssl-${OPENSSL_VERSION}-Linux-x86_64
                "20d7e3caf86368e03f61071c67d152cc3e18865d8b38e369df74ae4c57d1421b"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-x86_64)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
            download_and_extract(
                ${DORADO_CDN_URL}/openssl-${OPENSSL_VERSION}-linux-aarch64.zip
                openssl-${OPENSSL_VERSION}-Linux-aarch64
                "5d31d284e6f5734d390e7c36584e001f764613109111520420a9f84c4bd6fd2d"
            )
            set(OPENSSL_ROOT_DIR ${DORADO_3RD_PARTY_DOWNLOAD}/openssl-${OPENSSL_VERSION}-Linux-aarch64)
        endif()
    endif()
else()
    message(STATUS "Using existing OpenSSL at ${OPENSSL_ROOT_DIR}")
endif()

set(CMAKE_PREFIX_PATH ${OPENSSL_ROOT_DIR} ${CMAKE_PREFIX_PATH}) # put the selected openssl path before any older imported one.

find_package(OpenSSL REQUIRED QUIET)
