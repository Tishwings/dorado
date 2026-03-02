#pragma once
#include "SpeedEntry.h"

#include <algorithm>
#include <span>

namespace dorado::batchsize_benchmarks::tests {

// SpeedEntry shouldn't need an equality check outside of the tests, so implement that here.
inline bool entries_equal(std::span<const SpeedEntry> lhs, std::span<const SpeedEntry> rhs) {
    const auto compare = [](const SpeedEntry& a, const SpeedEntry& b) {
        return a.batch_size == b.batch_size && a.basecall_speed == b.basecall_speed &&
               a.memory_used == b.memory_used;
    };
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), compare);
}

inline void entries_sort(std::span<SpeedEntry> entries) {
    std::sort(entries.begin(), entries.end(), [](const SpeedEntry& lhs, const SpeedEntry& rhs) {
        return lhs.batch_size < rhs.batch_size;
    });
}

}  // namespace dorado::batchsize_benchmarks::tests
