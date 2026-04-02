#include "secondary/consensus/window_utils.h"

#include "secondary/common/batching.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace dorado::secondary {

std::vector<Window> create_windows_from_regions(
        const std::vector<Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap) {
    std::vector<Window> windows;

    for (int64_t i = 0; i < std::ssize(regions); ++i) {
        Region region = regions[i];

        spdlog::debug("Creating windows for region: '{}'.", region_to_string(region));

        const auto it = draft_lookup.find(region.name);
        if (it == std::end(draft_lookup)) {
            throw std::runtime_error(
                    "Sequence specified by custom region not found in input! Sequence name: " +
                    region.name);
        }
        const auto [seq_id, seq_length] = it->second;

        region.start = std::max<int64_t>(0, region.start);
        region.end = (region.end < 0) ? seq_length : std::min(seq_length, region.end);

        if (region.start >= region.end) {
            throw std::runtime_error{"Region coordinates not valid. Given: region.name = '" +
                                     region.name +
                                     "', region.start = " + std::to_string(region.start) +
                                     ", region.end = " + std::to_string(region.end)};
        }

        std::vector<Window> new_windows =
                create_windows(static_cast<int32_t>(seq_id), region.start, region.end, seq_length,
                               bam_chunk_len, window_overlap, static_cast<int32_t>(i));

        spdlog::debug("Generated {} windows for region: '{}'.", std::size(new_windows),
                      region_to_string(region));
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    return windows;
}

}  // namespace dorado::secondary
