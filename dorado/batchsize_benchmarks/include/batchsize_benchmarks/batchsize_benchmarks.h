#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace dorado::config {
struct BasecallModelConfig;
}  // namespace dorado::config

namespace dorado::data_loader {
class InputFiles;
}  // namespace dorado::data_loader

namespace dorado::batchsize_benchmarks {

// Retrieve the best batch size for a given config, if it exists in the cache.
std::optional<int> get(const std::string &device,
                       float memory_limit_fraction,
                       const config::BasecallModelConfig &config,
                       float batch_size_time_penalty);

// Generate full benchmarks for a given config on a device.
// This will overwrite existing benchmarks if they exist.
void generate(const std::string &device,
              const config::BasecallModelConfig &config,
              const data_loader::InputFiles &input_files);

// Load additional benchmarks into the cache.
bool load_cache(const std::filesystem::path &file);

// Export the current cache to a file.
bool export_cache(const std::filesystem::path &file);

}  // namespace dorado::batchsize_benchmarks
