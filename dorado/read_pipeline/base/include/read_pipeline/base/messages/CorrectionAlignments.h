#pragma once

#include "utils/cigar.h"
#include "utils/overlap.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

// Overlaps for error correction
struct CorrectionAlignments {
    // Populated in CorrectionMapperNode::extract_alignments
    std::string read_name;
    std::vector<std::string> qnames;
    std::vector<std::vector<CigarOp>> cigars;
    std::vector<utils::Overlap> overlaps;

    // Populated in CorrectionInferenceNode::populate_alignments if the alignment is useful
    std::string read_seq;
    std::vector<uint8_t> read_qual;
    std::vector<std::string> seqs;
    std::vector<std::vector<uint8_t>> quals;

    // This is mostly to workaround an issue where sometimes
    // the tend of an overlap is much bigger than the
    // tlen of the read. This is unexpected and happens
    // intermittently. Again this is was observed before the
    // split index loading in mm2 was fixed. But keeping check around
    // for now to catch any lingering issues.
    // TODO: Remove this function if the error is not observed again.
    bool check_consistent_overlaps() {
        for (size_t i = 0; i < overlaps.size(); i++) {
            auto& ovlp = overlaps[i];
            if (ovlp.tlen < ovlp.tstart || ovlp.tlen < ovlp.tend) {
                return false;
            }
        }
        return true;
    }

    size_t size() {
        size_t si = read_name.length() + read_seq.length() + read_qual.size();
        for (auto& o : overlaps) {
            si += sizeof(o);
        }
        for (auto& v : cigars) {
            si += v.size() * sizeof(CigarOp);
        }
        for (auto& s : seqs) {
            si += s.length();
        }
        for (auto& v : quals) {
            si += v.size();
        }
        for (auto& s : qnames) {
            si += s.length();
        }

        return si;
    }
};

using CorrectionAlignmentsPtr = std::unique_ptr<CorrectionAlignments>;

}  // namespace dorado
