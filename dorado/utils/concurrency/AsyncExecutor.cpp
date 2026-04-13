#include "utils/concurrency/AsyncExecutor.h"

namespace dorado::utils::concurrency {

void AsyncExecutor::flush() {
    assert(m_tasks != nullptr);
    m_tasks->wait_for_queue_to_complete(m_q_idx);
}

}  // namespace dorado::utils::concurrency
