#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace dorado::secondary {

// clang-format off
struct Window {
    int32_t seq_id = -1;            // ID of the sequence from which the window was sampled.
    int64_t seq_length = 0;         // Length of the sequence where the window was sampled from.
    int64_t start = 0;              // Window start, possible overlap with neighboring windows.
    int64_t end = 0;                // Window end, possible overlap with neighboring windows.
    int64_t start_no_overlap = 0;   // Start coordinate of the unique portion of this window (no overlaps with neighbors).
    int64_t end_no_overlap = 0;     // End coordinate of the unique portion of this window (no overlaps with neighbors).
    int32_t source_region_id = -1;
};
// clang-format on

std::ostream& operator<<(std::ostream& os, const Window& w);

std::string window_to_string(const Window& w);

bool operator==(const Window& lhs, const Window& rhs);

}  // namespace dorado::secondary
