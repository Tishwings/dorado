#include "BamFile.h"

#include <htslib/sam.h>

#include <stdexcept>

namespace kadayashi::hts_utils {
BamFile::BamFile(const std::filesystem::path &in_fn, int n_threads)
        : m_fp{nullptr}, m_idx{nullptr}, m_hdr{nullptr}, m_n_threads{1} {
    m_fp = HtsFilePtr{hts_open(in_fn.string().c_str(), "rb"), HtsFileDestructor()};
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    m_idx = HtsIdxPtr{sam_index_load(m_fp.get(), in_fn.string().c_str()), HtsIdxDestructor()};
    if (!m_idx) {
        throw std::runtime_error{"Could not open index for BAM file: '" + in_fn.string() + "'!"};
    }

    m_hdr = SamHdrPtr{sam_hdr_read(m_fp.get()), SamHdrDestructor()};
    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }

    if (n_threads > 1) {
        hts_set_threads(m_fp.get(), n_threads);
        m_n_threads = n_threads;
    }
}

BamPtr BamFile::get_next() {
    BamPtr record(bam_init1(), BamDestructor());

    if (record == nullptr) {
        throw std::runtime_error{"Failed to initialize BAM record"};
        return BamPtr(nullptr, BamDestructor());
    }

    if (sam_read1(m_fp.get(), m_hdr.get(), record.get()) >= 0) {
        return record;
    }

    return BamPtr(nullptr, BamDestructor());
}

BamFileView BamFile::get_view() {
    return BamFileView{.fp = m_fp.get(), .idx = m_idx.get(), .hdr = m_hdr.get()};
}

}  // namespace kadayashi::hts_utils