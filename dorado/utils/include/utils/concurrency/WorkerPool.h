#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

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
    TaskPool* get_task_pool() const;
    void set_task_pool(TaskPool* pool);

    void bind_task_pool(TaskPool& pool);
    void unbind_task_pool();

public:
    explicit WorkerPool(size_t num_workers, std::string_view name);
    ~WorkerPool();

    // Bind the given task pool to this worker pool.
    // Tasks will begin being popped and executed immediately after binding.
    class BindTasks {
        WorkerPool& m_workers;

        BindTasks(const BindTasks&) = delete;
        BindTasks& operator=(const BindTasks&) = delete;
        BindTasks(BindTasks&&) = delete;
        BindTasks& operator=(BindTasks&&) = delete;

    public:
        explicit BindTasks(WorkerPool& workers, TaskPool& tasks);
        ~BindTasks();
    };

    // Wait for all existing work to complete.
    void flush();
};

}  // namespace dorado::utils::concurrency
