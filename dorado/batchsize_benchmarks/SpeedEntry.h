#pragma once

#include <cstdint>

namespace dorado::batchsize_benchmarks {

struct SpeedEntry {
    std::uint32_t batch_size;
    double basecall_speed;
    std::uint64_t memory_used;
};

}  // namespace dorado::batchsize_benchmarks
