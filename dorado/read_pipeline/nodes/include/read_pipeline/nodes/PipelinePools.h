#pragma once

#include "utils/concurrency/multi_queue_thread_pool.h"
#include "utils/parameters.h"

namespace dorado {

struct PipelineWorkers {
    explicit PipelineWorkers(const utils::ThreadAllocations& thread_allocations)
            : aligner_pool(thread_allocations.aligner_threads, "align_node_pool"),
              barcode_pool(thread_allocations.barcoder_threads, "barcode_pool"),
              polya_pool(std::thread::hardware_concurrency(), "polya_pool") {}

    utils::concurrency::MultiQueueThreadPool aligner_pool;
    utils::concurrency::MultiQueueThreadPool barcode_pool;
    utils::concurrency::MultiQueueThreadPool polya_pool;
};

}  // namespace dorado
