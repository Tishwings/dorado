option(DORADO_DISABLE_TESTS "Disable building the tests" OFF)
option(DORADO_RUN_TESTS "Run the tests. Certain platforms don't have the infrastructure to do this atm" ON)
option(DORADO_ENABLE_BENCHMARK_TESTS "Enable benchmarks to run as part of the tests" OFF)

if (DORADO_DISABLE_TESTS)
    return()
endif()

# dorado_tests
add_executable(dorado_tests
    AdapterDetectorTest.cpp
    AlignerTest.cpp
    arg_parse_ext_test.cpp
    async_task_executor_test.cpp
    AsyncQueueTest.cpp
    BamReaderTest.cpp
    BamUtilsTest.cpp
    BamWriterTest.cpp
    BarcodeClassifierSelectorTest.cpp
    BarcodeClassifierTest.cpp
    BasecallModelConfigTest.cpp
    BatchParamsTest.cpp
    bed_file_test.cpp
    ChunkTest.cpp
    CigarTest.cpp
    CliUtilsTest.cpp
    context_container_test.cpp
    CorrectionWindowTest.cpp
    CustomBarcodeParserTest.cpp
    DuplexReadTaggingNodeTest.cpp
    DuplexSplitTest.cpp
    FastqTagsTest.cpp
    FastxRandomReaderTest.cpp
    FastxSequentialReaderTest.cpp
    FileInfoTest.cpp
    FixedSizeQueueTest.cpp
    gpu_monitor_test.cpp
    HeaderMapperTest.cpp
    HtsFileTest.cpp
    IndexFileAccessTest.cpp
    KadayashiTest.cpp
    MathUtilsTest.cpp
    MergeHeadersTest.cpp
    Minimap2IndexTest.cpp
    ModBaseChunkTest.cpp
    ModBaseConfigTest.cpp
    ModBaseContextTest.cpp
    ModBaseEncoderTest.cpp
    ModelComplexTest.cpp
    ModelKitsTest.cpp
    ModelMetadataTest.cpp
    ModelUtilsTest.cpp
    MotifMatcherTest.cpp
    multi_queue_thread_pool_test.cpp
    myers_test.cpp
    PafUtilsTest.cpp
    PairingNodeTest.cpp
    PipelineTest.cpp
    Pod5DataLoaderTest.cpp
    PolishImplTest.cpp
    PolyACalculatorTest.cpp
    PostConditionTest.cpp
    priority_task_queue_test.cpp
    ReadFilterNodeTest.cpp
    ReadForwarderNodeTest.cpp
    ReadTest.cpp
    RealignMovesTest.cpp
    ResourceLimiterTest.cpp
    ResumeLoaderTest.cpp
    RleTest.cpp
    RNASplitTest.cpp
    SampleSheetTests.cpp
    SamUtilsTest.cpp
    ScaledDotProductAttention.cpp
    SecondaryDecodeVariantsTest.cpp
    SecondaryEncoderReadAlignmentTest.cpp
    SecondaryEncoderUtilsTest.cpp
    SecondaryKadayashiUtils.cpp
    SecondaryMergeVCSamplesTest.cpp
    SecondaryModelFactory.cpp
    SecondaryNormalizeVariantsTest.cpp
    SecondarySampleCollateUtilsTest.cpp
    SecondarySampleTest.cpp
    SecondaryTrimTest.cpp
    SecondaryWindowTest.cpp
    SeparatedStreamTest.cpp
    SequenceUtilsTest.cpp
    StereoDuplexTest.cpp
    StitchTest.cpp
    StringUtilsTest.cpp
    SummaryFileWriterTest.cpp
    synchronisation_test.cpp
    TensorUtilsTest.cpp
    TimeUtilsTest.cpp
    TrimTest.cpp
    WriterNodeTest.cpp
)

if(APPLE)
    target_sources(dorado_tests PRIVATE MetalLinearTest.cpp)
else()
    target_sources(dorado_tests PRIVATE cuda_utils_test.cpp)
endif()

# dorado_smoke_tests
add_executable(dorado_smoke_tests
    NodeSmokeTest.cpp
)


# dorado_tests_common
add_library(dorado_tests_common STATIC
    main.cpp
    TestUtils.cpp
)
enable_warnings_as_errors(dorado_tests_common)
target_link_libraries(dorado_tests_common
    PUBLIC
        dorado_torch_utils
        dorado_utils
        dorado_compat
        dorado_secondary
        kadayashi::haplotag_lib
        Catch2::Catch2
)
target_compile_definitions(dorado_tests_common
    PUBLIC
        DORADO_ENABLE_BENCHMARK_TESTS=$<BOOL:${DORADO_ENABLE_BENCHMARK_TESTS}>
        # Workaround torch defining a CATCH_CHECK() macro that conflicts with catch2
        CATCH_CONFIG_PREFIX_ALL=1
)

# Finish setting up each target and add them as tests.
foreach(TEST_BIN dorado_tests dorado_smoke_tests)
    if (DORADO_ENABLE_PCH)
        target_precompile_headers(${TEST_BIN}
            PUBLIC
                "<torch/torch.h>"
        )
    endif()

    target_link_libraries(${TEST_BIN}
        PRIVATE
            dorado_basecall
            dorado_cli
            dorado_file_info
            dorado_io_lib
            dorado_lib
            dorado_modbase
            dorado_model_resolver_lib
            dorado_model_downloader_lib
            dorado_models_lib
            dorado_nn
            dorado_polish
            dorado_secondary
            dorado_tests_common
            intervaltree
            minimap2
            toml11::toml11
            ${ZLIB_LIBRARIES}
    )

    enable_warnings_as_errors(${TEST_BIN})
    if (MSVC)
        set_property(TARGET ${TEST_BIN} APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    else()
        # Add -Wno-trigraphs because one of the tests uses a qstring that has a trigraph sequence.
        target_compile_options(${TEST_BIN} PRIVATE "-Wno-trigraphs")
    endif()

    # Don't add the test if we can't run it
    if (NOT DORADO_RUN_TESTS)
        continue()
    endif()

    # Add the test.
    dorado_add_test(${TEST_BIN})
endforeach()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()

# We don't build as PIC by default on Linux, so we fail to create a dynamic library there.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Create a dynamic library that checks that all symbols are available.
    # If this fails to link then we're not providing all of the libs for the symbols
    # that we use.
    add_library(dorado_symbol_test SHARED symbol_test.cpp)
    enable_warnings_as_errors(dorado_symbol_test)
    target_link_libraries(dorado_symbol_test PUBLIC dorado_lib dorado_models_lib)
    if(MSVC)
        set_property(TARGET dorado_symbol_test APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    endif()
endif()
