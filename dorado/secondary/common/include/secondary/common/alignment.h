#pragma once

#include "hts_utils/hts_types.h"
#include "utils/cigar.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::secondary {

struct Alignment {
    std::string qname;
    int64_t qlen{0};
    int64_t qstart{0};
    int64_t qend{0};
    StrandOrientation strand{StrandOrientation::UNKNOWN};
    std::string rname;
    int64_t rlen{0};
    int64_t rstart{0};
    int64_t rend{0};
    int32_t mapq{0};
    uint32_t flag{4};
    std::vector<CigarOp> cigar;
    std::string seq;
    std::vector<uint8_t> qual;
    int32_t dwell_stride{0};
    std::vector<uint8_t> dwells;
};

void serialize_alignment(std::ostream& os,
                         const Alignment& aln,
                         const std::string_view separator,
                         const bool write_cigar);

std::string serialize_alignment_to_string(const Alignment& aln,
                                          const std::string_view separator,
                                          const bool write_cigar);

void serialize_to_paf(std::ostream& os, const Alignment& aln);

}  // namespace dorado::secondary
