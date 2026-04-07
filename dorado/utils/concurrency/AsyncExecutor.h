#pragma once

#include "Task.h"

#include <cassert>
#include <cstddef>
#include <utility>

namespace dorado::utils::concurrency {

class TaskPool;

// Executor that you can push tasks to.
class AsyncExecutor {
    TaskPool* m_tasks = nullptr;
    std::size_t m_q_idx = 0;

private:
    AsyncExecutor(const AsyncExecutor&) = delete;
    AsyncExecutor& operator=(const AsyncExecutor&) = delete;

public:
    explicit AsyncExecutor(TaskPool& tasks, std::size_t q_idx) noexcept
            : m_tasks(&tasks), m_q_idx(q_idx) {}

    explicit AsyncExecutor() noexcept = default;
    AsyncExecutor(AsyncExecutor&& o) noexcept : AsyncExecutor() {
        std::swap(m_tasks, o.m_tasks);
        std::swap(m_q_idx, o.m_q_idx);
    }
    AsyncExecutor& operator=(AsyncExecutor&&) = delete;

    // Push a new task to the pool.
    void send(Task&& func);

    // Wait for all existing tasks to finish.
    void flush();
};

}  // namespace dorado::utils::concurrency
