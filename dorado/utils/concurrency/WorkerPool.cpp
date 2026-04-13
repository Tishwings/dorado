#include "utils/concurrency/WorkerPool.h"

#include "utils/concurrency/TaskPool.h"
#include "utils/hardware_interference_size.h"

#include <cassert>
#include <stdexcept>
#include <thread>

namespace dorado::utils::concurrency {

struct alignas(hardware_destructive_interference_size) WorkerPool::WorkerState {
    std::thread worker;

    // Technically all workers will have the same TaskPool, but putting it
    // in here doesn't increase the size of WorkerState and keeps it local
    // to the worker.
    TaskPool* task_pool = nullptr;

    // All state changes are controlled by the pool, except for Pausing->Paused which is done by the worker.
    enum State { Running, Pausing, Paused, Stopped };
    std::atomic<State> state{Paused};
};

void WorkerPool::worker_thread(size_t worker_idx) {
    auto& worker_state = m_states[worker_idx];

    // Wait for us to become unpaused.
    worker_state.state.wait(WorkerState::Paused, std::memory_order_acquire);

    while (true) {
        // Handle state change.
        const WorkerState::State state = worker_state.state.load(std::memory_order_relaxed);
        if (state == WorkerState::Stopped) {
            break;
        } else if (state == WorkerState::Pausing) {
            // Inform the pool that we're paused and then wait for it to be changed.
            worker_state.state.exchange(WorkerState::Paused, std::memory_order_release);
            worker_state.state.notify_one();
            worker_state.state.wait(WorkerState::Paused, std::memory_order_acquire);
            continue;
        }

        // If we get here then we should be running.
        assert(state == WorkerState::Running);

        // If we don't have an active task pool then wait for one to be bound.
        if (worker_state.task_pool == nullptr) {
            worker_state.state.wait(WorkerState::Running, std::memory_order_relaxed);
            continue;
        }

        worker_state.task_pool->run_task(worker_idx);
    }
}

void WorkerPool::set_task_pool(TaskPool* pool) {
    flush();

    // Tell the workers to pause.
    for (size_t idx = 0; idx < m_num_workers; idx++) {
        m_states[idx].state.exchange(WorkerState::Pausing, std::memory_order_relaxed);
        m_states[idx].state.notify_one();
    }
    // Wait for all of them to become paused.
    for (size_t idx = 0; idx < m_num_workers; idx++) {
        m_states[idx].state.wait(WorkerState::Pausing, std::memory_order_acquire);
    }
    // Assign the new task pool and start them off again.
    for (size_t idx = 0; idx < m_num_workers; idx++) {
        m_states[idx].task_pool = pool;
        m_states[idx].state.exchange(WorkerState::Running, std::memory_order_release);
        m_states[idx].state.notify_one();
    }
}

WorkerPool::WorkerPool(size_t num_workers)
        : m_states(std::make_unique<WorkerState[]>(num_workers)), m_num_workers(num_workers) {
    for (size_t idx = 0; idx < m_num_workers; idx++) {
        m_states[idx].worker = std::thread([this, idx] { worker_thread(idx); });
    }
}

WorkerPool::~WorkerPool() {
    // TODO: fast shutdown should discard
    flush();

    // Tell the workers to stop, then join them.
    for (size_t idx = 0; idx < m_num_workers; idx++) {
        m_states[idx].state.exchange(WorkerState::Stopped);
        m_states[idx].state.notify_one();
        m_states[idx].worker.join();
    }
}

void WorkerPool::bind_task_pool(TaskPool& pool) {
    if (m_states[0].task_pool != nullptr) {
        throw std::logic_error("WorkerPool already has a TaskPool bound");
    }
    set_task_pool(&pool);
}

void WorkerPool::unbind_task_pool() { set_task_pool(nullptr); }

void WorkerPool::flush() {
    // If we haven't been assigned a pool yet then bail.
    TaskPool* task_pool = m_states[0].task_pool;
    if (task_pool == nullptr) {
        return;
    }

    // Wait for all the queues to finish.
    const std::size_t num_queues = task_pool->num_queues();
    for (size_t idx = 0; idx < num_queues; idx++) {
        task_pool->wait_for_queue_to_complete(idx);
    }
}

WorkerPool::BindTasks::BindTasks(WorkerPool& workers, TaskPool& tasks) : m_workers(workers) {
    m_workers.bind_task_pool(tasks);
}

WorkerPool::BindTasks::~BindTasks() { m_workers.unbind_task_pool(); }

}  // namespace dorado::utils::concurrency
