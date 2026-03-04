#include "secondary/common/bam_file.h"

#include <htslib/hts.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

void HtsIdxDestructor::operator()(hts_idx_t* bam) { hts_idx_destroy(bam); }

void HtsItrDestructor::operator()(hts_itr_t* itr) const noexcept { hts_itr_destroy(itr); }

namespace dorado::secondary {
BamIterator::BamIterator(hts_itr_t* itr, htsFile* fp) : m_itr(itr), m_fp(fp) {}

BamPtr BamIterator::get_next() {
    BamPtr rec(bam_init1(), BamDestructor());

    if (!rec) {
        throw std::runtime_error{"Failed to initialize BAM record"};
    }

    const int32_t ret = sam_itr_next(m_fp, m_itr.get(), rec.get());

    if (ret >= 0) {
        return rec;
    }

    // Iteration finished.
    return BamPtr(nullptr, BamDestructor());
}

BamFile::BamFile(const std::filesystem::path& in_fn, int n_threads)
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

BamIterator BamFile::fetch(const std::string& chrom, const int64_t start, const int64_t end) {
    const int32_t tid = sam_hdr_name2tid(m_hdr.get(), chrom.c_str());

    if (tid < 0) {
        throw std::runtime_error{"Chromosome not found in BAM header: '" + chrom + "'"};
    }

    hts_itr_t* raw_itr = sam_itr_queryi(m_idx.get(), tid, start, end);

    if (!raw_itr) {
        throw std::runtime_error{"Failed to create iterator for '" + chrom + ":" +
                                 std::to_string(start) + "-" + std::to_string(end) + "'"};
    }

    return BamIterator(raw_itr, m_fp.get());
}

}  // namespace dorado::secondary
