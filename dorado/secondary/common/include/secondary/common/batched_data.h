#pragma once

#include <ATen/core/TensorBody.h>

#include <optional>

namespace dorado::secondary {

/**
 * \brief Struct which holds batched data prepared for inference. This is
 *          the  collation of the contents of InferenceData.
 */
struct BatchedData {
    at::Tensor features;
    std::optional<at::Tensor> refseqs;
};

}  // namespace dorado::secondary