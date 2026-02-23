#pragma once

#include "ReadCommon.h"

#include <cstdint>
#include <memory>

namespace dorado {

class SimplexRead;

// A pair of reads for Duplex calling
struct ReadPair {
    struct ReadData {
        ReadCommon read_common;
        uint64_t seq_start;
        uint64_t seq_end;
        static ReadData from_read(const SimplexRead& read, uint64_t seq_start, uint64_t seq_end);
    };
    ReadData template_read;
    ReadData complement_read;
};

using ReadPairPtr = std::unique_ptr<ReadPair>;

}  // namespace dorado
