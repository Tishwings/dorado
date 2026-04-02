#pragma once

#include "secondary/common/region.h"
#include "secondary/common/window.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Linearly splits sequence lengths into windows. It also returns the backward mapping of which
 *          windows correspond to which sequences, needed for stitching.
 */
std::vector<Window> create_windows(int32_t seq_id,
                                   int64_t seq_start,
                                   int64_t seq_end,
                                   int64_t seq_len,
                                   int32_t window_len,
                                   int32_t window_overlap,
                                   int32_t source_region_id);

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
