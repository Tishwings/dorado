#pragma once

#include <cstdint>
#include <span>

namespace dorado::batchsize_benchmarks {

struct SpeedEntry;

int pick_best_batch_size(std::span<const SpeedEntry> speeds,
                         uint64_t memory_limit,
                         float batch_size_time_penalty);

}  // namespace dorado::batchsize_benchmarks
