#pragma once

#include "secondary/consensus/sample.h"

#include <vector>

namespace dorado::variant {

/**
 * \brief Struct which holds samples prepared for inference.
 *          In practice, the vector here holds one batch for inference.
 */
struct InferenceData {
    std::vector<secondary::Sample> samples;
};

/**
 * \brief Struct which holds batched data prepared for inference. This is
 *          the  collation of the contents of InferenceData.
 */
struct BatchedData {
    at::Tensor features;
    std::optional<at::Tensor> refseqs;
};

}  // namespace dorado::variant
