#pragma once

#include "utils/AsyncQueue.h"
#include "utils/concurrency/Task.h"
#include "utils/hardware_interference_size.h"

#include <atomic>

namespace dorado::utils::concurrency {

// A pool of tasks that the workers will pull work from.
//
// The pool consists of |num_queues| queues, each of size |q_capacity|.
class TaskPool {
    using TaskQueue = utils::AsyncQueue<Task>;
    std::vector<TaskQueue> m_task_qs;
    struct alignas(hardware_destructive_interference_size) QueueCounter {
        std::atomic_size_t value = 0;
    };
    std::vector<QueueCounter> m_q_counters;

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
    std::size_t tasks_in_flight(std::size_t q_idx) const {
        return m_q_counters.at(q_idx).value.load(std::memory_order_relaxed);
    }

    // Push a task into the pool.
    template <typename Func>
    void send(Func &&func, std::size_t q_idx) {
        auto &q = m_task_qs.at(q_idx);
        auto &counter = m_q_counters.at(q_idx).value;

        // Keep track of the number of tasks in flight so that we can wait on them during a flush.
        counter.fetch_add(1, std::memory_order_relaxed);
        q.try_push([task = std::forward<Func>(func), &counter] {
            task();

            // We only need to wake waiters when this value hits 0, so avoid unnecessary notifies.
            const std::size_t old_value = counter.fetch_sub(1, std::memory_order_release);
            if (old_value == 1) {
                counter.notify_one();
            }
        });
    }

    // Try and run a task from the pool.
    void run_task(size_t worker_idx);

    // Wait for all tasks from this queue to be completed.
    void wait_for_queue_to_complete(std::size_t q_idx);
};

}  // namespace dorado::utils::concurrency
