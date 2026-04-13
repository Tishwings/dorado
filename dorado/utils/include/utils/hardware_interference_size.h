#pragma once

#include <cstddef>
#include <new>

namespace dorado::utils {

// Some stdlibs don't have this C++17 feature yet.
#if defined(__cpp_lib_hardware_interference_size)
constexpr std::size_t hardware_destructive_interference_size =
        std::hardware_destructive_interference_size;
constexpr std::size_t hardware_constructive_interference_size =
        std::hardware_constructive_interference_size;
#elif defined(__aarch64__)
// https://developer.arm.com/documentation/ddi0601/2025-06/AArch64-Registers/CCSIDR-EL1--Current-Cache-Size-ID-Register
constexpr std::size_t hardware_destructive_interference_size = 256;
constexpr std::size_t hardware_constructive_interference_size = 64;
#else
// Assuming x86_64.
constexpr std::size_t hardware_destructive_interference_size = 64;
constexpr std::size_t hardware_constructive_interference_size = 64;
#endif

}  // namespace dorado::utils
