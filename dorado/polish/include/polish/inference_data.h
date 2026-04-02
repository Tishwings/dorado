#pragma once

#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"

#include <vector>

namespace dorado::polisher {

/**
 * \brief Struct which holds data prepared for inference. In practice,
 *          vectors here hold one batch for inference. Both vectors should
 *          have identical length.
 */
struct InferenceData {
    std::vector<secondary::Sample> samples;
    std::vector<secondary::TrimInfo> trims;
};

}  // namespace dorado::polisher
