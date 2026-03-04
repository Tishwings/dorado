#include "secondary/common/alignment.h"

#include <ostream>
#include <sstream>

namespace dorado::secondary {

void serialize_alignment(std::ostream& os,
                         const Alignment& aln,
                         const std::string_view separator,
                         const bool write_cigar) {
    os << aln.qname << separator << aln.qlen << separator << aln.qstart << separator << aln.qend
       << separator << ((aln.strand == StrandOrientation::FORWARD) ? "" : "-") << separator
       << aln.rname << separator << aln.rlen << separator << aln.rstart << separator << aln.rend
       << separator << aln.mapq << separator << "fl:i:" << aln.flag;
    if (write_cigar) {
        os << separator << "cg:Z:" << aln.cigar;
    }
}

std::string serialize_alignment_to_string(const Alignment& aln,
                                          const std::string_view separator,
                                          const bool write_cigar) {
    std::ostringstream oss;
    serialize_alignment(oss, aln, separator, write_cigar);
    return oss.str();
}

void serialize_to_paf(std::ostream& os, const Alignment& aln) {
    serialize_alignment(os, aln, "\t", true);
}

}  // namespace dorado::secondary