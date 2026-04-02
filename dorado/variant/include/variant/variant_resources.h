#pragma once

#include "secondary/architectures/model_torch_base.h"
#include "secondary/common/device_info.h"
#include "secondary/features/decoder_factory.h"
#include "secondary/features/encoder_factory.h"

#include <c10/core/Stream.h>

#include <memory>
#include <vector>

namespace dorado::variant {

struct VariantResources {
    std::vector<std::unique_ptr<secondary::EncoderBase>> encoders;
    std::unique_ptr<secondary::DecoderBase> decoder;
    std::vector<secondary::DeviceInfo> devices;
    std::vector<std::shared_ptr<secondary::ModelTorchBase>> models;
    std::vector<c10::optional<c10::Stream>> streams;
};

}  // namespace dorado::variant
