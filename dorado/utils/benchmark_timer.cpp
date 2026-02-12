#include "utils/benchmark_timer.h"

#include "utils/thread_utils.h"

namespace dorado {

BenchmarkTimer::BenchmarkTimer(std::chrono::seconds benchmarking_period, ShutdownCallback callback)
        : m_shutdown_callback(std::move(callback)),
          m_benchmarking_thread(
                  [this, benchmarking_period] { benchmarking_thread_fn(benchmarking_period); }) {}

BenchmarkTimer::~BenchmarkTimer() {
    if (m_benchmarking_thread.joinable()) {
        terminate();
    }
}

void BenchmarkTimer::terminate() {
    m_should_terminate = true;
    m_benchmarking_thread.join();
}

void BenchmarkTimer::benchmarking_thread_fn(std::chrono::seconds benchmarking_period) {
    utils::set_thread_name("benchmarking_timer");

    using Clock = std::chrono::system_clock;
    const auto end_time = Clock::now() + benchmarking_period;
    const auto sleep_time = std::chrono::seconds(1);

    while (!m_should_terminate) {
        std::this_thread::sleep_for(sleep_time);
        const auto now = Clock::now();
        if (now > end_time) {
            m_shutdown_callback();
            return;
        }
    }
}

}  // namespace dorado
