#pragma once

#include "utils/concurrency/AsyncExecutor.h"
#include "utils/concurrency/TaskPool.h"
#include "utils/concurrency/WorkerPool.h"

namespace dorado {

// Simple wrapper around a set of workers and tasks.
template <typename Node>
class SimpleExecutors {
    utils::concurrency::WorkerPool workers;
    utils::concurrency::TaskPool tasks;
    utils::concurrency::WorkerPool::BindTasks binder;

public:
    // Create |num_threads| workers, with space for |num_executors| executors/nodes.
    explicit SimpleExecutors(std::size_t num_threads, std::size_t num_executors)
            : workers(num_threads, Node::QUEUE_THREAD_NAME),
              tasks(num_executors, Node::MAX_PROCESSING_QUEUE_SIZE),
              binder{workers, tasks} {}

    auto get(std::size_t idx) & { return utils::concurrency::AsyncExecutor{tasks, idx}; }
};

// dorado standalone is single-pipeline everywhere, so this simplifies that setup.
template <typename Node>
class SimpleExecutor {
    SimpleExecutors<Node> executors;

public:
    explicit SimpleExecutor(std::size_t num_threads) : executors(num_threads, 1) {}
    auto get() & { return executors.get(0); }
};

}  // namespace dorado
