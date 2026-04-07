#pragma once

#include <IntervalTree.h>

#include <cstdint>
#include <unordered_map>

namespace dorado::secondary {

using IntervalInt64 = interval_tree::Interval<int64_t, int64_t>;
using IntervalTreeInt64 = interval_tree::IntervalTree<int64_t, int64_t>;
using IntervalTreesInt64Map = std::unordered_map<int32_t, IntervalTreeInt64>;

}  // namespace dorado::secondary