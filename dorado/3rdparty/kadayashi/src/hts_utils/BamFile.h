#pragma once

#include "bam_file_view.h"
#include "hts_types.h"

#include <filesystem>

namespace kadayashi::hts_utils {

class BamFile {
public:
    BamFile(const std::filesystem::path& in_fn, int n_threads);

    // Getters.
    htsFile* fp() const { return m_fp.get(); }
    hts_idx_t* idx() const { return m_idx.get(); }
    sam_hdr_t* hdr() const { return m_hdr.get(); }

    htsFile* fp() { return m_fp.get(); }
    hts_idx_t* idx() { return m_idx.get(); }
    sam_hdr_t* hdr() { return m_hdr.get(); }

    BamPtr get_next();
    BamFileView get_view();

private:
    HtsFilePtr m_fp;
    HtsIdxPtr m_idx;
    SamHdrPtr m_hdr;
    int m_n_threads;
};

}  // namespace kadayashi::hts_utils