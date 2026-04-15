#pragma once

#include "utils/concurrency/multi_queue_thread_pool.h"
#include "utils/parameters.h"

namespace dorado {

struct PipelineWorkers {
    explicit PipelineWorkers(const utils::ThreadAllocations& thread_allocations)
            : aligner_pool(thread_allocations.aligner_threads, "align_node_pool") {}

    utils::concurrency::MultiQueueThreadPool aligner_pool;
};

}  // namespace dorado
