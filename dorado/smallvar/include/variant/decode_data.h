#pragma once

#include "secondary/consensus/sample.h"

#include <ATen/core/TensorBody.h>

#include <vector>

namespace dorado::smallvar {

/**
 * \brief Struct which holds output of inference, passed into the decoding thread.
 */
struct DecodeData {
    std::vector<secondary::Sample> samples;
    at::Tensor logits;
};

}  // namespace dorado::smallvar
