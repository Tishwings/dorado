#pragma once

#include <c10/core/Device.h>

#include <string>

namespace dorado::secondary {

enum class DeviceType { CPU, CUDA, METAL, UNKNOWN };

struct DeviceInfo {
    std::string name;
    DeviceType type;
    torch::Device device;
    double available_memory_GB = 0.0;
};

}  // namespace dorado::secondary
