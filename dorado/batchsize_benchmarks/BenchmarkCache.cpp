#include "BenchmarkCache.h"

#include "compiled_timings.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <functional>
#include <unordered_map>

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

template <typename T>
void json_write(std::ofstream &file, std::span<T> object) {
    file << '[';
    bool first = true;
    for (const auto &[key, value] : object) {
        if (!first) {
            file << ',';
        }
        first = false;

        file << "{\"" << key << "\":\"" << value << "\"}";
    }
    file << ']';
}

template <typename Key, typename Value>
void json_write(std::ofstream &file, const std::unordered_map<Key, Value> &object) {
    file << '{';
    bool first = true;
    for (const auto &[key, value] : object) {
        if (!first) {
            file << ',';
        }
        first = false;

        file << '"' << key << "\":";
        json_write(file, value);
    }
    file << '}';
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
        const std::string &gpu_name,
        const std::string &model_name) const {
    const auto key = Key(gpu_name, model_name);

    // Check the runtime cache first.
    auto runtime = m_cache.m_runtime_cache.find(key);
    if (runtime != m_cache.m_runtime_cache.end()) {
        return runtime->second;
    }

    // Fall back to the compiled cache.
    return get_from_compiled_cache(gpu_name, model_name);
}

bool BenchmarkCache::CacheProxy::load_from_file(const std::filesystem::path &) {
    // TODO: need a JSON parser
    return false;
}

bool BenchmarkCache::CacheProxy::export_to_file(const std::filesystem::path &path) const {
    std::ofstream file(path);
    if (!file) {
        spdlog::warn("Failed to open {}", path.string());
        return false;
    }

    // Build up a friendlier JSON-like version of the data.
    std::unordered_map<GPUName, std::unordered_map<ModelName, std::span<const SpeedEntry>>> json;
    for (const auto &[key, entries] : m_cache.m_runtime_cache) {
        const auto &[gpu_name, model_name] = key;
        json[gpu_name][model_name] = entries;
    }

    // Write it out.
    // TODO: replace this with a proper JSON library
    json_write(file, json);
    return true;
}

BenchmarkCache::BenchmarkCache() = default;

BenchmarkCache::~BenchmarkCache() = default;

BenchmarkCache BenchmarkCache::s_cache;

}  // namespace dorado::batchsize_benchmarks
