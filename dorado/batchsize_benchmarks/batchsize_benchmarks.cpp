#include "batchsize_benchmarks/batchsize_benchmarks.h"

#include "BenchmarkCache.h"
#include "BenchmarkSink.h"
#include "api/runner_creation.h"
#include "config/BasecallModelConfig.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/BasecallerNode.h"
#include "spdlog/spdlog.h"
#include "utils/jthread.h"
#include "utils/parameters.h"
#include "utils/string_utils.h"
#include "utils/sys_utils.h"

#include <c10/core/CachingDeviceAllocator.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <thread>

#if DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#elif DORADO_METAL_BUILD
#include "basecall/MetalCaller.h"
#include "torch_utils/metal_utils.h"
#endif

namespace dorado::batchsize_benchmarks {

namespace {

// Floating point version of chrono::seconds.
using Seconds = std::chrono::duration<double>;

Seconds min_benchmark_duration(const config::BasecallModelConfig &config) {
    const bool is_fast = utils::contains(config.model_name(), "fast");
    return is_fast ? Seconds(5) : Seconds(10);
}

SpeedEntry calculate_one(const std::string &device,
                         const config::BasecallModelConfig &config,
                         const data_loader::InputFiles &input_files) {
    const auto batch_size = config.basecaller.batch_size();
#if DORADO_CUDA_BUILD
    // Assume that we'll want VCS enabled.
    const c10::Device torch_device{device};
    const int device_ids[]{torch_device.index()};
    const bool enable_vcs = api::check_variable_chunk_sizes_supported(config, device_ids);
#else
    const bool enable_vcs = false;
#endif

    // Create runners.
    auto [runners, num_devices] = api::create_basecall_runners(
            {
                    .model_config = config,
                    .device = device,
                    .memory_limit_fraction = 1.f,
                    .pipeline_type = api::PipelineType::simplex,
                    .batch_size_time_penalty = 0.f,
                    .variable_chunk_sizes = enable_vcs,
            },
            utils::default_parameters.num_runners, 0);

    // Reset peak allocator counters so that we can measure how much was used during processing.
#if DORADO_CUDA_BUILD
    auto *const allocator = c10::getDeviceAllocator(torch_device.type());
    allocator->resetPeakStats(torch_device.index());
#elif DORADO_METAL_BUILD
    // We replace the allocator on the metal path and don't implement the DeviceAllocator
    // interface, so we can't get peak stats.
    // TODO: implement the interface and reuse the cuda path
#endif

    // Build a benchmarking pipeline.
    PipelineDescriptor descriptor;
    const NodeHandle sink_handle = descriptor.add_node<BenchmarkSink>({});
    descriptor.add_node<BasecallerNode>({sink_handle}, std::move(runners),
                                        config.basecaller.overlap(), config.model_name(), 1000,
                                        "BasecallerNode", config.mean_qscore_start_pos);
    auto pipeline = Pipeline::create(std::move(descriptor), nullptr);

    // We need to feed in the data on a separate thread since it'll block.
    std::atomic<bool> finished_data{false};
    auto source = utils::jthread([&] {
        data_loader::DataLoader loader(*pipeline, device, 1, 0, std::nullopt, {});
        loader.load_reads(input_files, ReadOrder::UNRESTRICTED);
        finished_data.store(true, std::memory_order_relaxed);
    });

    // Wait for reads to start coming through.
    auto &sink = pipeline->get_node_ref<BenchmarkSink>(sink_handle);
    const Seconds first_read_latency = sink.wait_for_reads();
    spdlog::debug("First read latency for batchsize={}: {}s", batch_size,
                  first_read_latency.count());

    // Make sure we let at least a few batches through before the benchmark ends.
    const int min_num_batches = 5;
    const auto benchmark_duration =
            std::max(first_read_latency * min_num_batches, min_benchmark_duration(config));

    // Let the benchmark run and grab the speed.
    std::this_thread::sleep_for(benchmark_duration);
    if (finished_data.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Ran out of data while benchmarking. Need a bigger input file");
    }
    const double speed = sink.samples_per_second();
    spdlog::debug("[{}] {} @ {}", device, speed, batch_size);

    // Teardown the pipeline. This will teardown the source thread too.
    pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::Yes});
    source.join();

#if DORADO_CUDA_BUILD
    const auto aggregate_idx = static_cast<uint64_t>(c10::CachingAllocator::StatType::AGGREGATE);
    const auto memory_used =
            allocator->getDeviceStats(torch_device.index()).allocated_bytes.at(aggregate_idx).peak;
#elif DORADO_METAL_BUILD
    // This slightly underestimates since it's not the peak like the cuda path.
    const auto memory_used = dorado::utils::get_mtl_device()->currentAllocatedSize();
#endif

    return SpeedEntry{
            .batch_size = static_cast<uint32_t>(batch_size),
            .basecall_speed = sink.samples_per_second(),
            .memory_used = static_cast<uint64_t>(memory_used),
    };
}

std::string get_gpu_name(const std::string &device) {
    if (device == "cpu") {
        throw std::logic_error("Trying to calculate batchsize on CPU");
    }

#if DORADO_CUDA_BUILD
    c10::cuda::CUDAGuard device_guard(device);
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return prop->name;

#elif DORADO_METAL_BUILD
    return dorado::utils::get_mtl_device()->name()->utf8String();

#else
#error "Invalid build"
#endif
}

int get_batch_size_granularity(const config::BasecallModelConfig &config) {
#if DORADO_CUDA_BUILD
    return basecall::CudaCaller::get_batch_size_granularity(config);
#elif DORADO_METAL_BUILD
    return config.is_lstm_model() ? basecall::MetalLSTMCaller::get_batch_size_granularity()
                                  : basecall::MetalTxCaller::get_batch_size_granularity();
#else
#error "Invalid build"
#endif
}

int get_max_safe_batch_size(const std::string &device, const config::BasecallModelConfig &config) {
#if DORADO_CUDA_BUILD
    return basecall::CudaCaller::get_max_safe_batch_size(device, 1, config);
#elif DORADO_METAL_BUILD
    (void)device;
    return config.is_lstm_model() ? basecall::MetalLSTMCaller::get_max_safe_batch_size(1, config)
                                  : basecall::MetalTxCaller::get_max_safe_batch_size(1, config);
#else
#error "Invalid build"
#endif
}

uint64_t get_gpu_mem_limit(const std::string &device, float memory_limit_fraction) {
#if DORADO_CUDA_BUILD
    return basecall::CudaCaller::get_gpu_mem_limit(device, memory_limit_fraction);
#elif DORADO_METAL_BUILD
    (void)device;
    // TODO: this assumes that we're the only thing running on the machine.
    return utils::get_apple_physical_memory_bytes() * memory_limit_fraction;
#else
#error "Invalid build"
#endif
}

}  // namespace

std::optional<int> get(const std::string &device,
                       float memory_limit_fraction,
                       const config::BasecallModelConfig &config,
                       float batch_size_time_penalty) {
    const std::string gpu_name = get_gpu_name(device);
    const auto model_name = config.model_name();
    const auto memory_limit = get_gpu_mem_limit(device, memory_limit_fraction);

    return BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) -> std::optional<int> {
        auto speeds = proxy.get_timings(gpu_name, model_name);
        if (speeds.empty()) {
            spdlog::debug("No batch size timings for {}/{}", gpu_name, model_name);
            return std::nullopt;
        }

        auto is_slower = [](const SpeedEntry &lhs, const SpeedEntry &rhs) {
            return lhs.basecall_speed < rhs.basecall_speed;
        };

        // Look for the fastest batch size.
        const auto fastest_iter = std::max_element(speeds.begin(), speeds.end(), is_slower);
        if (fastest_iter == speeds.end()) {
            throw std::logic_error("Failed to find fastest batch size in non-empty span");
        }
        spdlog::debug("Fastest batch size is {}", fastest_iter->batch_size);

        // Cap to the largest batch size under our memory limit.
        const auto first_batchsize_over_max_iter = std::find_if(
                speeds.begin(), std::next(fastest_iter), [memory_limit](const SpeedEntry &entry) {
                    return entry.memory_used > memory_limit;
                });
        if (first_batchsize_over_max_iter == speeds.begin()) {
            throw std::runtime_error(
                    fmt::format("No entries remaining after applying memory_limit_fraction ({})",
                                memory_limit));
        }

        // Limit based on the time penalty.
        const double threshold_speed =
                fastest_iter->basecall_speed / (1.0 + batch_size_time_penalty);
        auto fastest_under_threshold_iter =
                std::find_if(speeds.begin(), first_batchsize_over_max_iter,
                             [threshold_speed](const SpeedEntry &entry) {
                                 return entry.basecall_speed >= threshold_speed;
                             });
        if (fastest_under_threshold_iter == speeds.end()) {
            spdlog::debug(
                    "No entries are faster than threshold of {}. Using absolute fastest instead",
                    threshold_speed);
            fastest_under_threshold_iter =
                    std::max_element(speeds.begin(), first_batchsize_over_max_iter, is_slower);
        }
        if (fastest_under_threshold_iter == speeds.end()) {
            throw std::logic_error("Error in batch size selection algorithm");
        }
        spdlog::debug("Fastest capped+limited batch size is {} @ {}",
                      fastest_under_threshold_iter->batch_size,
                      fastest_under_threshold_iter->basecall_speed);

        return fastest_under_threshold_iter->batch_size;
    });
}

void generate(const std::string &device,
              const config::BasecallModelConfig &orig_config,
              const data_loader::InputFiles &input_files,
              const std::function<void(float)> &progress_callback) {
    if (utils::running_in_docker()) {
        spdlog::warn(
                "Generating benchmarks inside of a container may not be representitive of the real "
                "hardware.");
    }

    const std::string gpu_name = get_gpu_name(device);
    const auto batch_size_granularity = get_batch_size_granularity(orig_config);
    const auto max_safe_batch_size = get_max_safe_batch_size(device, orig_config);

    // Do the benchmarking.
    spdlog::info("Benchmarking batch sizes in steps of {} for {} ({})", batch_size_granularity,
                 device, gpu_name);
    std::vector<SpeedEntry> speeds;
    speeds.reserve(max_safe_batch_size / batch_size_granularity);
    for (int batch_size = batch_size_granularity; batch_size <= max_safe_batch_size;
         batch_size += batch_size_granularity) {
        // Make a copy so that we can change the batch size.
        auto config = orig_config;
        config.basecaller.set_batch_size(batch_size);
        config.normalise_basecaller_params();
        const auto entry = calculate_one(device, config, input_files);
        speeds.emplace_back(entry);

        if (progress_callback) {
            progress_callback(batch_size / static_cast<float>(max_safe_batch_size));
        }
    }

    // Add them to the cache.
    BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
        proxy.add_timings(gpu_name, orig_config.model_name(), std::move(speeds));
    });
}

bool load_cache(const std::filesystem::path &file) {
    return BenchmarkCache::with_lock(
            [&](BenchmarkCache::CacheProxy proxy) { return proxy.load_from_file(file); });
}

bool export_cache(const std::filesystem::path &file) {
    return BenchmarkCache::with_lock(
            [&](BenchmarkCache::CacheProxy proxy) { return proxy.export_to_file(file); });
}

}  // namespace dorado::batchsize_benchmarks
