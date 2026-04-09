#include "utils/concurrency/AsyncExecutor.h"

#include "utils/concurrency/TaskPool.h"

#include <latch>
#include <memory>

namespace dorado::utils::concurrency {

void AsyncExecutor::send(Task&& task) {
    assert(m_tasks != nullptr);
    m_tasks->send(std::move(task), m_q_idx);
}

void AsyncExecutor::flush() {
    assert(m_tasks != nullptr);

    // Each producer has a dedicated queue, so we only need to push a blocking task to ours.
    auto blocker = std::make_shared<std::latch>(2);
    m_tasks->send([blocker] { blocker->arrive_and_wait(); }, m_q_idx);
    blocker->arrive_and_wait();
}

}  // namespace dorado::utils::concurrency
