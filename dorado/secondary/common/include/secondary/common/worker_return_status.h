#pragma once

#include <string>

namespace dorado::secondary {

struct WorkerReturnStatus {
    bool exception_thrown{false};
    std::string message;
};

}  // namespace dorado::secondary
