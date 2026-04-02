#pragma once

#include "secondary/common/region.h"
#include "secondary/consensus/window.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Splits each input region into BAM windows, clamping the coordinates to the
 *        corresponding draft sequence bounds.
 */
std::vector<Window> create_windows_from_regions(
        const std::vector<Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        int32_t bam_chunk_len,
        int32_t window_overlap);

}  // namespace dorado::secondary
