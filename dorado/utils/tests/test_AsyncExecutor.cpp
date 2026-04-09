#include "utils/concurrency/AsyncExecutor.h"
#include "utils/concurrency/TaskPool.h"
#include "utils/concurrency/WorkerPool.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#define CUT_TAG "[AsyncExecutor]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::utils::concurrency;

namespace {

DEFINE_TEST("Smoke test") {
    const std::size_t num_workers = GENERATE(1, 2);
    const std::size_t num_producers = GENERATE(1, 2);
    const std::size_t queue_capacity = GENERATE(1, 10);
    const std::size_t num_tasks_per_producer = GENERATE(1, 2, 10);

    // Create the pools.
    WorkerPool workers(num_workers);
    TaskPool tasks(num_producers, queue_capacity);

    // Bind the tasks to the workers.
    WorkerPool::BindTasks binder(workers, tasks);

    // Setup executors.
    std::vector<AsyncExecutor> executors;
    executors.reserve(num_producers);
    for (std::size_t producer_idx = 0; producer_idx < num_producers; producer_idx++) {
        executors.emplace_back(tasks, producer_idx);
    }

    // Push some tasks.
    for (std::size_t task_idx = 0; task_idx < num_tasks_per_producer; task_idx++) {
        for (auto &producer : executors) {
            producer.send([] {});
        }
    }

    // Wait for them to complete.
    //for (auto &producer : executors) {
    //    producer.flush();
    //}
    //workers.flush();
}

}  // namespace
