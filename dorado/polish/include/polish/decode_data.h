#pragma once

#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"

#include <ATen/ATen.h>

#include <vector>

namespace dorado::polisher {

/**
 * \brief Struct which holds output of inference, passed into the decoding thread.
 */
struct DecodeData {
    std::vector<secondary::Sample> samples;
    at::Tensor logits;
    std::vector<secondary::TrimInfo> trims;
};

}  // namespace dorado::polisher
