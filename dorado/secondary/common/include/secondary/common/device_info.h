#pragma once

#include <c10/core/Device.h>

#include <string>

namespace dorado::secondary {

struct DeviceInfo {
    std::string name;
    torch::Device device;
    double available_memory_GB = 0.0;
};

}  // namespace dorado::secondary
