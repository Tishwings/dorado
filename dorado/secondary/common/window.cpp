#include "secondary/common/window.h"

#include <ostream>
#include <sstream>
#include <tuple>

namespace dorado::secondary {

std::ostream& operator<<(std::ostream& os, const Window& w) {
    os << "seq_id = " << w.seq_id << ", seq_length = " << w.seq_length << ", start = " << w.start
       << ", end = " << w.end << ", start_no_overlap = " << w.start_no_overlap
       << ", end_no_overlap = " << w.end_no_overlap;
    return os;
}

bool operator==(const Window& lhs, const Window& rhs) {
    return std::tie(lhs.seq_id, lhs.seq_length, lhs.start, lhs.end, lhs.start_no_overlap,
                    lhs.end_no_overlap) == std::tie(rhs.seq_id, rhs.seq_length, rhs.start, rhs.end,
                                                    rhs.start_no_overlap, rhs.end_no_overlap);
}

std::string window_to_string(const Window& w) {
    std::ostringstream oss;
    oss << w;
    return oss.str();
}

}  // namespace dorado::secondary
