#include "utils/concurrency/AsyncExecutor.h"

namespace dorado::utils::concurrency {

void AsyncExecutor::flush() {
    assert(m_tasks != nullptr);
    m_tasks->wait_for_queue_to_complete(m_q_idx);
}

std::size_t AsyncExecutor::tasks_in_flight() const {
    assert(m_tasks != nullptr);
    return m_tasks->tasks_in_flight(m_q_idx);
}

}  // namespace dorado::utils::concurrency
