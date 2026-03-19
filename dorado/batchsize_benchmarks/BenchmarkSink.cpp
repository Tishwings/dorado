#include "BenchmarkSink.h"

#include <mutex>

namespace dorado::batchsize_benchmarks {

BenchmarkSink::BenchmarkSink() : MessageSink(1000, 4) {}

BenchmarkSink::~BenchmarkSink() { terminate({.fast = utils::AsyncQueueTerminateFast::Yes}); }

std::string BenchmarkSink::get_name() const { return "BenchmarkSink"; }

void BenchmarkSink::terminate(const TerminateOptions &terminate_options) {
    stop_input_processing(terminate_options.fast);
}

void BenchmarkSink::restart() {
    m_pipeline_start_time = Clock::now();
    m_first_read_batch_time.reset();
    start_input_processing([this] { input_thread_fn(); }, "bench_sink");
}

BenchmarkSink::Clock::duration BenchmarkSink::wait_for_reads() {
    std::unique_lock lock(m_first_read_mutex);
    m_first_read_cv.wait(lock, [&] { return m_first_read_batch_time.has_value(); });
    return *m_first_read_batch_time;
}

BenchmarkSink::Clock::duration BenchmarkSink::elapsed() {
    return Clock::now() - m_pipeline_start_time;
}

void BenchmarkSink::input_thread_fn() {
    // Pull reads out of the queue.
    Message message;
    bool first_message = true;
    while (get_input_message(message)) {
        // If this is the first read we've seen then set the current time.
        if (std::exchange(first_message, false)) {
            {
                std::lock_guard lock(m_first_read_mutex);
                m_first_read_batch_time = elapsed();
            }
            m_first_read_cv.notify_one();
        }
    }
}

}  // namespace dorado::batchsize_benchmarks
