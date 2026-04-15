#pragma once

#include "demux/BarcodeClassifierSelector.h"
#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/async_task_executor.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

namespace demux {
struct BarcodingInfo;
}

namespace utils::concurrency {
class MultiQueueThreadPool;
}  // namespace utils::concurrency

class BarcodeClassifierNode : public MessageSink {
public:
    static inline constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
    static inline constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};

    BarcodeClassifierNode(utils::concurrency::MultiQueueThreadPool& thread_pool,
                          utils::concurrency::TaskPriority pipeline_priority);
    BarcodeClassifierNode(utils::concurrency::MultiQueueThreadPool& thread_pool);

    ~BarcodeClassifierNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    utils::concurrency::AsyncTaskExecutor m_task_executor;

    std::atomic<int> m_num_records{0};
    demux::BarcodeClassifierSelector m_barcoder_selector{};

    void input_thread_fn();
    void barcode(BamMessage& read, const demux::BarcodingInfo* barcoding_info);
    void barcode(SimplexRead& read);

    // Track how many reads were classified as each barcode for debugging
    // purposes.
    std::atomic<size_t> m_mid_strand_count{0};
    std::map<std::string, size_t> m_barcode_count;
    mutable std::mutex m_barcode_count_mutex;
};

}  // namespace dorado
