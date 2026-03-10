#include "batchsize_helpers.h"

#include "SpeedEntry.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace dorado::batchsize_benchmarks {

int pick_best_batch_size(std::span<const SpeedEntry> speeds,
                         uint64_t memory_limit,
                         float batch_size_time_penalty) {
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
            speeds.begin(), std::next(fastest_iter),
            [memory_limit](const SpeedEntry &entry) { return entry.memory_used > memory_limit; });
    if (first_batchsize_over_max_iter == speeds.begin()) {
        throw std::runtime_error(fmt::format(
                "No entries remaining after applying memory_limit_fraction ({})", memory_limit));
    }

    // Limit based on the time penalty.
    const double threshold_speed = fastest_iter->basecall_speed / (1.0 + batch_size_time_penalty);
    auto fastest_under_threshold_iter =
            std::find_if(speeds.begin(), first_batchsize_over_max_iter,
                         [threshold_speed](const SpeedEntry &entry) {
                             return entry.basecall_speed >= threshold_speed;
                         });
    if (fastest_under_threshold_iter == speeds.end()) {
        spdlog::debug("No entries are faster than threshold of {}. Using absolute fastest instead",
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
}

}  // namespace dorado::batchsize_benchmarks
