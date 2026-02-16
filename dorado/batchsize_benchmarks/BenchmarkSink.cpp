#include "BenchmarkSink.h"

#include "read_pipeline/base/messages/ReadCommon.h"

#include <mutex>

namespace dorado::batchsize_benchmarks {

BenchmarkSink::BenchmarkSink() : MessageSink(1000, 4) {}

BenchmarkSink::~BenchmarkSink() { terminate({.fast = utils::AsyncQueueTerminateFast::Yes}); }

std::string BenchmarkSink::get_name() const { return "BenchmarkSink"; }

void BenchmarkSink::terminate(const TerminateOptions &terminate_options) {
    stop_input_processing(terminate_options.fast);
}

void BenchmarkSink::restart() {
    m_time_of_first_read.reset();
    start_input_processing([this] { input_thread_fn(); }, "bench_sink");
}

BenchmarkSink::Clock::duration BenchmarkSink::wait_for_reads() {
    const auto start_time = Clock::now();

    std::unique_lock lock(m_first_read_mutex);
    m_first_read_cv.wait(lock, [&] { return m_time_of_first_read.has_value(); });
    return *m_time_of_first_read - start_time;
}

double BenchmarkSink::samples_per_second() {
    std::lock_guard lock(m_first_read_mutex);
    if (!m_time_of_first_read) {
        return 0;
    }
    const auto ds = m_samples_processed.load(std::memory_order_relaxed);
    const auto dt = std::chrono::duration<double>(Clock::now() - *m_time_of_first_read).count();
    return ds / dt;
}

void BenchmarkSink::input_thread_fn() {
    // Pull reads out of the queue.
    Message message;
    while (get_input_message(message)) {
        // Increment counters.
        const auto &data = get_read_common_data(message);
        const std::size_t num_samples = data.get_raw_data_samples();
        const std::size_t prev_count =
                m_samples_processed.fetch_add(num_samples, std::memory_order_relaxed);

        // If this is the first read we've seen then set the current time.
        if (prev_count == 0) {
            {
                std::lock_guard lock(m_first_read_mutex);
                m_time_of_first_read = Clock::now();
            }
            m_first_read_cv.notify_one();
        }
    }
}

}  // namespace dorado::batchsize_benchmarks
