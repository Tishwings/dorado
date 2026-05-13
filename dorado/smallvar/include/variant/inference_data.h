#pragma once

#include "secondary/consensus/sample.h"

#include <vector>

namespace dorado::smallvar {

/**
 * \brief Struct which holds samples prepared for inference.
 *          In practice, the vector here holds one batch for inference.
 */
struct InferenceData {
    std::vector<secondary::Sample> samples;
};

}  // namespace dorado::smallvar
