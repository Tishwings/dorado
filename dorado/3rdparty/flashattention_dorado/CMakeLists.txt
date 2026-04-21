set(_DORADO_FLASHATTN3_ROOT ${DORADO_3RD_PARTY_SOURCE}/flashattention)
set(_DORADO_FLASHATTN3_HOPPER_DIR ${_DORADO_FLASHATTN3_ROOT}/hopper)
set(_DORADO_FLASHATTN3_CUTLASS_INCLUDE ${_DORADO_FLASHATTN3_ROOT}/csrc/cutlass/include)
set(_DORADO_FLASHATTN3_MIN_CUDA_VERSION 12.3)

set(_DORADO_FLASHATTN3_CAN_BUILD OFF)
if (NOT APPLE
    AND CUDAToolkit_FOUND
    AND EXISTS "${_DORADO_FLASHATTN3_HOPPER_DIR}/flash_api.cpp"
    AND EXISTS "${_DORADO_FLASHATTN3_CUTLASS_INCLUDE}/cutlass/cutlass.h")
    # Keep this aligned with the vendored FlashAttention3 Hopper package, whose
    # setup.py and README require CUDA 12.3+.
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL ${_DORADO_FLASHATTN3_MIN_CUDA_VERSION})
        include(CheckLanguage)
        check_language(CUDA)
        if (CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(_DORADO_FLASHATTN3_CAN_BUILD ON)
        endif()
    else()
        message(STATUS
            "Disabling dorado_flashattention3: CUDA ${CUDAToolkit_VERSION} is older than "
            "the required ${_DORADO_FLASHATTN3_MIN_CUDA_VERSION}")
    endif()
endif()

if (_DORADO_FLASHATTN3_CAN_BUILD)
    # FlashAttention3 manages its own supported architectures via explicit
    # per-file -gencode flags below. Disable CMake's CUDA arch propagation in
    # this directory so we don't inherit stale defaults such as compute_50.
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    if (WIN32)
        # Match the rest of the Windows build, which already links shared cudart
        # through LibTorch. Disable nvcc's implicit runtime selection here so the
        # wrapper does not inject cudart_static into downstream link lines.
        set(CMAKE_CUDA_RUNTIME_LIBRARY None)

        # FlashAttention's CUDA sources are expensive template instantiations and
        # the Windows debug jobs do not rely on stepping through these kernels.
        # Strip CUDA-specific debug flags for this wrapper to reduce compile
        # time, but do not force full release flags because those conflict with
        # Visual Studio's normal Debug host settings (e.g. /RTC1).
        set(_DORADO_FLASHATTN3_SAVED_CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
        set(CMAKE_CUDA_FLAGS_DEBUG "")

        # Let nvcc parallelize its own internal compilation work on Windows.
        set(_DORADO_FLASHATTN3_NVCC_THREADS 4)
        if (DEFINED WIN_THREADS)
            set(_DORADO_FLASHATTN3_NVCC_THREADS "${WIN_THREADS}")
        endif()
    endif()

    # Torch's bundled Caffe2/FindCUDA logic can also inject its own -gencode flags
    # via CMAKE_CUDA_FLAGS. Strip any inherited architecture selection in this
    # directory and rely on the explicit per-file gencode flags below instead.
    function(_dorado_flashattention3_strip_inherited_gencode_flags out_var in_var)
        set(_DORADO_FLASHATTN3_FILTERED_FLAGS "${${in_var}}")
        string(REGEX REPLACE "(^| )(-gencode|--generate-code)=[^ ]+" "\\1"
               _DORADO_FLASHATTN3_FILTERED_FLAGS "${_DORADO_FLASHATTN3_FILTERED_FLAGS}")
        string(REGEX REPLACE "(^| )(-gencode|--generate-code) +[^ ]+" "\\1"
               _DORADO_FLASHATTN3_FILTERED_FLAGS "${_DORADO_FLASHATTN3_FILTERED_FLAGS}")
        string(REGEX REPLACE "  +" " "
               _DORADO_FLASHATTN3_FILTERED_FLAGS "${_DORADO_FLASHATTN3_FILTERED_FLAGS}")
        string(STRIP "${_DORADO_FLASHATTN3_FILTERED_FLAGS}" _DORADO_FLASHATTN3_FILTERED_FLAGS)
        set(${out_var} "${_DORADO_FLASHATTN3_FILTERED_FLAGS}" PARENT_SCOPE)
    endfunction()

    foreach (_DORADO_FLASHATTN3_CUDA_FLAG_VAR
             CMAKE_CUDA_FLAGS
             CMAKE_CUDA_FLAGS_DEBUG
             CMAKE_CUDA_FLAGS_RELEASE
             CMAKE_CUDA_FLAGS_RELWITHDEBINFO
             CMAKE_CUDA_FLAGS_MINSIZEREL)
        if (DEFINED ${_DORADO_FLASHATTN3_CUDA_FLAG_VAR})
            _dorado_flashattention3_strip_inherited_gencode_flags(
                _DORADO_FLASHATTN3_SANITIZED_FLAGS
                ${_DORADO_FLASHATTN3_CUDA_FLAG_VAR})
            set(${_DORADO_FLASHATTN3_CUDA_FLAG_VAR} "${_DORADO_FLASHATTN3_SANITIZED_FLAGS}")
        endif()
    endforeach()

    set(_DORADO_FLASHATTN3_SRCS_COMMON
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/flash_api.cpp
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/flash_prepare_scheduler.cu
    )

    set(_DORADO_FLASHATTN3_SRCS_SM8X
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim64_bf16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim64_fp16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim96_bf16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim96_fp16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim128_bf16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim128_fp16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim192_bf16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim192_fp16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim256_bf16_sm80.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim256_fp16_sm80.cu
    )

    set(_DORADO_FLASHATTN3_SRCS_SM90
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim64_bf16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim64_fp16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim96_bf16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim96_fp16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim128_bf16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim128_fp16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim192_bf16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim192_fp16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim256_bf16_sm90.cu
        ${_DORADO_FLASHATTN3_HOPPER_DIR}/instantiations/flash_fwd_hdim256_fp16_sm90.cu
    )

    set(_DORADO_FLASHATTN3_SRCS
        ${_DORADO_FLASHATTN3_SRCS_COMMON}
        ${_DORADO_FLASHATTN3_SRCS_SM8X}
        ${_DORADO_FLASHATTN3_SRCS_SM90}
    )

    set(_DORADO_FLASHATTN3_PRIVATE_INCLUDE_DIRS
        ${_DORADO_FLASHATTN3_HOPPER_DIR}
        ${_DORADO_FLASHATTN3_CUTLASS_INCLUDE}
    )

    set(_DORADO_FLASHATTN3_PRIVATE_DEFINITIONS
        FLASHATTENTION_DISABLE_APPENDKV
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_HDIMDIFF192
        FLASHATTENTION_DISABLE_HDIMDIFF64
        FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_PACKGQA
        FLASHATTENTION_DISABLE_PAGEDKV
        FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_SPLIT
        FLASHATTENTION_DISABLE_FP8
        CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
        CUTLASS_ENABLE_GDC_FOR_SM90
        CUTLASS_DEBUG_TRACE_LEVEL=0
        # CUTLASS prepends "cutlass_" to this suffix, yielding namespace
        # `cutlass_flashattention` and avoiding collisions with other CUTLASS users.
        CUTLASS_NAMESPACE=flashattention
        NDEBUG
    )

    set(_DORADO_FLASHATTN3_PRIVATE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--ftemplate-backtrace-limit=0>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set(_DORADO_FLASHATTN3_GENCODE_SM8X -gencode=arch=compute_80,code=sm_80)
    set(_DORADO_FLASHATTN3_GENCODE_SM90 -gencode=arch=compute_90a,code=sm_90a)
    set_source_files_properties(${_DORADO_FLASHATTN3_SRCS_SM8X}
        PROPERTIES COMPILE_OPTIONS "${_DORADO_FLASHATTN3_GENCODE_SM8X}")
    set_source_files_properties(${_DORADO_FLASHATTN3_SRCS_SM90}
        PROPERTIES COMPILE_OPTIONS "${_DORADO_FLASHATTN3_GENCODE_SM90}")
    set_source_files_properties(${_DORADO_FLASHATTN3_HOPPER_DIR}/flash_prepare_scheduler.cu
        PROPERTIES COMPILE_OPTIONS
                   "${_DORADO_FLASHATTN3_GENCODE_SM8X};${_DORADO_FLASHATTN3_GENCODE_SM90}")

    function(_dorado_flashattention3_configure_build_target target_name)
        target_include_directories(${target_name}
            PRIVATE
                ${_DORADO_FLASHATTN3_PRIVATE_INCLUDE_DIRS}
        )

        target_compile_definitions(${target_name}
            PRIVATE
                ${_DORADO_FLASHATTN3_PRIVATE_DEFINITIONS}
        )
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
            target_compile_definitions(${target_name}
                PRIVATE
                    __NV_NO_VECTOR_DEPRECATION_DIAG
            )
        endif()
        if (WIN32)
            target_compile_definitions(${target_name} PRIVATE _USE_MATH_DEFINES)
        endif()

        target_compile_options(${target_name}
            PRIVATE
                ${_DORADO_FLASHATTN3_PRIVATE_COMPILE_OPTIONS}
        )
        if (WIN32)
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:/W0>
                    $<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>
                    $<$<COMPILE_LANGUAGE:CUDA>:--forward-slash-prefix-opts>
                    $<$<COMPILE_LANGUAGE:CUDA>:--threads=${_DORADO_FLASHATTN3_NVCC_THREADS}>
                    $<$<COMPILE_LANGUAGE:CUDA>:-cudart=none>
                    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=177>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
            )
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-narrowing>
            )
        endif()

        set_target_properties(${target_name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_ARCHITECTURES OFF
        )
        if (WIN32)
            set_target_properties(${target_name} PROPERTIES
                CUDA_RUNTIME_LIBRARY None
                MSVC_DEBUG_INFORMATION_FORMAT ""
            )
        endif()

        target_link_libraries(${target_name}
            PRIVATE
                torch_lib
        )
        if (NOT WIN32)
            target_link_libraries(${target_name}
                PRIVATE
                    CUDA::cudart
            )
        endif()
    endfunction()

    add_library(dorado_flashattention3 STATIC ${_DORADO_FLASHATTN3_SRCS})
    target_include_directories(dorado_flashattention3
        INTERFACE
            ${_DORADO_FLASHATTN3_ROOT}
    )
    target_compile_definitions(dorado_flashattention3
        PUBLIC
            DORADO_HAS_FLASHATTENTION3=1
    )
    _dorado_flashattention3_configure_build_target(dorado_flashattention3)
    if (WIN32)
        set(CMAKE_CUDA_FLAGS_DEBUG "${_DORADO_FLASHATTN3_SAVED_CMAKE_CUDA_FLAGS_DEBUG}")
        unset(_DORADO_FLASHATTN3_SAVED_CMAKE_CUDA_FLAGS_DEBUG)
    endif()
else()
    add_library(dorado_flashattention3 INTERFACE)
    target_compile_definitions(dorado_flashattention3
        INTERFACE
            DORADO_HAS_FLASHATTENTION3=0
    )
endif()
