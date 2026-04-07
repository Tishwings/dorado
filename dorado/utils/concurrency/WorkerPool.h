#pragma once

#include <cstddef>
#include <memory>

namespace dorado::utils::concurrency {

class TaskPool;

// These are the threads that will execute the tasks.
class WorkerPool {
    struct WorkerState;
    const std::unique_ptr<WorkerState[]> m_states;
    const std::size_t m_num_workers;

private:
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;

private:
    void worker_thread(size_t worker_idx);

public:
    explicit WorkerPool(size_t num_workers);
    ~WorkerPool();

    // Bind the given task pool to this worker pool.
    // Tasks will begin being popped and executed immediately.
    void set_task_pool(TaskPool& pool);

    // Wait for all existing work to complete.
    void flush();
};

}  // namespace dorado::utils::concurrency
