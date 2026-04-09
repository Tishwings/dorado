#pragma once

#include "utils/MoveOnlyFunction.h"

namespace dorado::utils::concurrency {

//
// Rough overview:
//   * Fixed number of persistent workers, held by WorkerPool.
//   * TaskPools are (re)created and attached to the WorkerPool when pipelines are (re)created.
//   * Fixed number of producers, one per node type (dorado) and/or pipeline (ont_core).
//   * Each producer has a fixed size queue, all of the same size.
//   * Every worker round-robins the queues trying to grab a new task.
//
// Visually:
//
// +------------+   +------------------+     +----------------+      +----------------+
// | Pipeline A |-->| AsyncExecutor[0] |--\  |  TaskPool (3)  |      | WorkerPool (5) |
// +------------+   +------------------+  |  +----------------+      +----------------+
//                                        \--|-> TaskQueue[0] |<=====|   Worker [0]   |
// +------------+   +------------------+     |                |<=====|   Worker [1]   |
// | Pipeline B |-->| AsyncExecutor[1] |-----|-> TaskQueue[1] |<=====|   Worker [2]   |
// +------------+   +------------------+     |                |<=====|   Worker [3]   |
//                                        /--|-> TaskQueue[2] |<=====|   Worker [4]   |
// +------------+   +------------------+  |  +----------------+      +----------------+
// | Pipeline C |-->| AsyncExecutor[2] |--/
// +------------+   +------------------+
//

using Task = utils::MoveOnlyFunction<void()>;

}  // namespace dorado::utils::concurrency
