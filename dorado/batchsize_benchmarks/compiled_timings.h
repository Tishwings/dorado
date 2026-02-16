#pragma once

#include "BenchmarkCache.h"

#include <algorithm>
#include <span>
#include <string_view>

namespace dorado::batchsize_benchmarks::compiled_cache {

struct ModelTimings {
    std::string_view model_name;
    std::span<const SpeedEntry> entries;
};

struct GPUModelTimings {
    std::string_view gpu_name;
    std::span<const ModelTimings> models;
};

std::span<const GPUModelTimings> get();

inline constexpr bool is_sorted(std::span<const SpeedEntry> entries) {
    return std::is_sorted(entries.begin(), entries.end(),
                          [](const SpeedEntry& lhs, const SpeedEntry& rhs) {
                              return lhs.batch_size < rhs.batch_size;
                          });
}

}  // namespace dorado::batchsize_benchmarks::compiled_cache
