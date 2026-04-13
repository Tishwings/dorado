#pragma once

#include "utils/AsyncQueue.h"
#include "utils/concurrency/Task.h"

namespace dorado::utils::concurrency {

// A pool of tasks that the workers will pull work from.
//
// The pool consists of |num_queues| queues, each of size |q_capacity|.
class TaskPool {
    using TaskQueue = utils::AsyncQueue<Task>;
    std::vector<TaskQueue> m_task_qs;

private:
    static std::vector<TaskQueue> make_queues(std::size_t num_queues, std::size_t q_capacity);

    bool pop_task(Task &task, size_t worker_idx);

private:
    TaskPool(const TaskPool &) = delete;
    TaskPool &operator=(const TaskPool &) = delete;
    TaskPool(TaskPool &&) = delete;
    TaskPool &operator=(TaskPool &&) = delete;

public:
    explicit TaskPool(std::size_t num_queues, std::size_t q_capacity);
    ~TaskPool();

    std::size_t num_queues() const { return m_task_qs.size(); }
    std::size_t queue_size(std::size_t q_idx) const { return m_task_qs.at(q_idx).size(); }

    // Push a task into the pool.
    void send(Task &&task, std::size_t q_idx) {
        auto &q = m_task_qs.at(q_idx);
        q.try_push(std::move(task));
    }

    // Try and run a task from the pool.
    void run_task(size_t worker_idx);
};

}  // namespace dorado::utils::concurrency
