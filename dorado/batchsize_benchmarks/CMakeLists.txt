dorado_add_library(dorado_benchmarks
    NAME
        dorado_benchmarks
    PUBLIC_DIR
        batchsize_benchmarks
    SOURCES_PUBLIC
        batchsize_benchmarks.h
    SOURCES_PRIVATE
        batchsize_benchmarks.cpp
        BenchmarkCache.cpp
        BenchmarkCache.h
        BenchmarkSink.cpp
        BenchmarkSink.h
        compiled_timings.h
        csv_helpers.h
        SpeedEntry.h
    DEPENDS_PUBLIC
        # nothing
    DEPENDS_PRIVATE
        dorado_api
        dorado_basecall
        dorado_config
        dorado_io_lib
        dorado_read_pipeline
        dorado_utils
        spdlog::spdlog
)

# Bake the benchmark timings.
include(compile_benchmarks.cmake)
set(compiled_timings_cpp "${CMAKE_CURRENT_BINARY_DIR}/compiled_timings.cpp")
dorado_compile_benchmarks(
    FILES
        "timings/test.csv"
    OUTPUT
        ${compiled_timings_cpp}
)
target_sources(dorado_benchmarks PRIVATE ${compiled_timings_cpp})
