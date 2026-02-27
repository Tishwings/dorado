#pragma once

#include "BenchmarkCache.h"
#include "utils/SeparatedStream.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <istream>
#include <ostream>
#include <span>

namespace dorado::batchsize_benchmarks {

inline void csv_write_entries(std::ostream &out,
                              std::string_view gpu_name,
                              std::string_view model_name,
                              std::span<const SpeedEntry> entries) {
    for (const auto &entry : entries) {
        out << gpu_name << ',';
        out << model_name << ',';
        out << entry.batch_size << ',';
        out << entry.basecall_speed << ',';
        out << entry.memory_used;
    }
}

template <typename Callback>
inline void csv_read_lines(std::istream &in, Callback &&callback) {
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
        const auto memory_used_str = stream.getline();
        if (stream.eof()) {
            // Ignore this line if any of the components are missing.
            spdlog::warn("Line doesn't have enough entries: {}", line);
            continue;
        }

        // Parse numbers.
        const auto batch_size = utils::from_chars<std::uint32_t>(batch_size_str.value());
        const auto basecall_speed = utils::from_chars<double>(basecall_speed_str.value());
        const auto memory_used = utils::from_chars<std::uint64_t>(memory_used_str.value());
        if (!batch_size.has_value() || !basecall_speed.has_value() || !memory_used.has_value()) {
            // Ignore this line if the values failed to parse.
            spdlog::warn("Failed to parse line: {}", line);
            continue;
        }

        // Hand off processing to the caller.
        const SpeedEntry entry{
                .batch_size = batch_size.value(),
                .basecall_speed = basecall_speed.value(),
                .memory_used = memory_used.value(),
        };
        callback(gpu_name.value(), model_name.value(), entry);
    }
}

}  // namespace dorado::batchsize_benchmarks
