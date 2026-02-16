#include "BenchmarkCache.h"

#include "compiled_timings.h"
#include "utils/SeparatedStream.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <optional>
#include <string>

namespace dorado::batchsize_benchmarks {

namespace {

std::span<const SpeedEntry> get_from_compiled_cache(std::string_view gpu_name,
                                                    std::string_view model_name) {
    auto all_gpus = compiled_cache::get();

    // Find the GPU.
    auto is_gpu = [gpu_name](const compiled_cache::GPUModelTimings &gpu) {
        return gpu.gpu_name == gpu_name;
    };
    auto gpu_it = std::find_if(all_gpus.begin(), all_gpus.end(), is_gpu);
    if (gpu_it == all_gpus.end()) {
        return {};
    }

    // Find the model.
    auto all_models = gpu_it->models;
    auto is_model = [model_name](const compiled_cache::ModelTimings &model) {
        return model.model_name == model_name;
    };
    auto model_it = std::find_if(all_models.begin(), all_models.end(), is_model);
    if (model_it == all_models.end()) {
        return {};
    }

    return model_it->entries;
}

void csv_write_entries(std::ofstream &out,
                       std::string_view gpu_name,
                       std::string_view model_name,
                       std::span<const SpeedEntry> entries) {
    for (const auto &entry : entries) {
        out << gpu_name << ',';
        out << model_name << ',';
        out << entry.batch_size << ',';
        out << entry.basecall_speed;
    }
}

template <typename Callback>
void csv_read_lines(std::ifstream &in, Callback &&callback) {
    std::string line;
    while (std::getline(in, line)) {
        // Ignore empty lines.
        if (line.empty()) {
            continue;
        }

        // Split up this line.
        utils::SeparatedStream<','> stream(line);
        const auto gpu_name = stream.getline();
        const auto model_name = stream.getline();
        const auto batch_size_str = stream.getline();
        const auto basecall_speed_str = stream.getline();
        if (stream.eof()) {
            // Ignore this line if any of the components are missing.
            spdlog::warn("Line doesn't have enough entries: {}", line);
            continue;
        }

        // Parse numbers.
        const auto batch_size = utils::from_chars<int>(batch_size_str.value());
        const auto basecall_speed = utils::from_chars<double>(basecall_speed_str.value());
        if (!batch_size.has_value() || !basecall_speed.has_value()) {
            // Ignore this line if the values failed to parse.
            spdlog::warn("Failed to parse line: {}", line);
            continue;
        }

        // Hand off processing to the caller.
        const SpeedEntry entry{
                .batch_size = batch_size.value(),
                .basecall_speed = basecall_speed.value(),
        };
        callback(gpu_name.value(), model_name.value(), entry);
    }
}

}  // namespace

void BenchmarkCache::CacheProxy::add_timings(std::string gpu_name,
                                             std::string model_name,
                                             std::vector<SpeedEntry> entries) {
    auto key = Key(std::move(gpu_name), std::move(model_name));
    auto &speeds = m_cache.m_runtime_cache[std::move(key)];
    speeds.swap(entries);
}

std::span<const SpeedEntry> BenchmarkCache::CacheProxy::get_timings(
        const std::string_view gpu_name,
        const std::string_view model_name) const {
    const auto key = std::make_pair(gpu_name, model_name);

    // Check the runtime cache first.
    auto runtime = m_cache.m_runtime_cache.find(key);
    if (runtime != m_cache.m_runtime_cache.end()) {
        return runtime->second;
    }

    // Fall back to the compiled cache.
    return get_from_compiled_cache(gpu_name, model_name);
}

bool BenchmarkCache::CacheProxy::load_from_file(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) {
        spdlog::warn("Failed to open {}", path.string());
        return false;
    }

    // Read the new entries in.
    std::map<Key, std::vector<SpeedEntry>, std::less<>> new_entries;
    auto process_line = [&](std::string_view gpu_name, std::string_view model_name,
                            const SpeedEntry &entry) {
        const auto key = Key(gpu_name, model_name);
        new_entries[key].push_back(entry);
    };
    csv_read_lines(file, process_line);

    // Update the cache.
    for (auto &[key, entries] : new_entries) {
        const auto &[gpu_name, model_name] = key;
        add_timings(gpu_name, model_name, std::move(entries));
    }

    return true;
}

bool BenchmarkCache::CacheProxy::export_to_file(const std::filesystem::path &path) const {
    std::ofstream file(path);
    if (!file) {
        spdlog::warn("Failed to open {}", path.string());
        return false;
    }

    // Write the runtime cache out as a CSV.
    // We assume that the compiled cache has been stored somewhere so doesn't need writing out again.
    for (const auto &[key, entries] : m_cache.m_runtime_cache) {
        const auto &[gpu_name, model_name] = key;
        csv_write_entries(file, gpu_name, model_name, entries);
    }

    return true;
}

BenchmarkCache::BenchmarkCache() = default;

BenchmarkCache::~BenchmarkCache() = default;

BenchmarkCache BenchmarkCache::s_cache;

}  // namespace dorado::batchsize_benchmarks
