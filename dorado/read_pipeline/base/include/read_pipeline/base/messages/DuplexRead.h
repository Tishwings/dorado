#pragma once

#include "ReadCommon.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace dorado {

// Class representing a duplex read, including stereo-encoded raw data
class DuplexRead {
public:
    // Data used to generate the stereo features in read_common.raw_data.
    class StereoFeatureInputs {
    public:
        std::vector<unsigned char> alignment;
        uint64_t template_seq_start = std::numeric_limits<uint64_t>::max();
        uint64_t complement_seq_start = std::numeric_limits<uint64_t>::max();
        std::string template_seq;
        std::string complement_seq;
        std::string template_qstring;
        std::string complement_qstring;
        std::vector<uint8_t> template_moves;
        std::vector<uint8_t> complement_moves;
        at::Tensor template_signal;
        at::Tensor complement_signal;
        int signal_stride = -1;
    };
    StereoFeatureInputs stereo_feature_inputs;

    ReadCommon read_common;
};

using DuplexReadPtr = std::unique_ptr<DuplexRead>;

}  // namespace dorado
