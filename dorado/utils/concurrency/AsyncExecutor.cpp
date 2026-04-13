#include "utils/concurrency/AsyncExecutor.h"

#include "utils/concurrency/TaskPool.h"

namespace dorado::utils::concurrency {

void AsyncExecutor::send(Task&& task) {
    assert(m_tasks != nullptr);
    m_tasks->send(std::move(task), m_q_idx);
}

void AsyncExecutor::flush() {
    assert(m_tasks != nullptr);
    m_tasks->wait_for_queue_to_complete(m_q_idx);
}

}  // namespace dorado::utils::concurrency
