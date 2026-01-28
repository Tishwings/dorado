#include "FastxRandomReader.h"

#include "faidx_utils.h"

#include <htslib/faidx.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace kadayashi::hts_utils {

namespace {

enum class SequenceFormatType {
    BAM,
    FASTA,
    FASTQ,
    SAM,
    UNKNOWN,
};

SequenceFormatType parse_sequence_format(const std::filesystem::path& in_path) {
    SequenceFormatType fmt = SequenceFormatType::UNKNOWN;

    // Convert the string to lowercase.
    std::string path_str = in_path.string();
    std::transform(std::begin(path_str), std::end(path_str), std::begin(path_str),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (path_str.ends_with(".bam")) {
        fmt = SequenceFormatType::BAM;

    } else if (path_str.ends_with(".sam")) {
        fmt = SequenceFormatType::SAM;

    } else if (path_str.ends_with(".fasta") || path_str.ends_with(".fa") ||
               path_str.ends_with(".fna") || path_str.ends_with(".fasta.gz") ||
               path_str.ends_with(".fa.gz") || path_str.ends_with(".fna.gz")) {
        fmt = SequenceFormatType::FASTA;

    } else if (path_str.ends_with(".fastq") || path_str.ends_with(".fq") ||
               path_str.ends_with(".fastq.gz") || path_str.ends_with(".fq.gz")) {
        fmt = SequenceFormatType::FASTQ;
    }

    return fmt;
}

FaidxPtr open_fai(const std::filesystem::path& fastx_path, const fai_format_options fmt) {
    FaidxPtr faidx_ptr(fai_load_format(fastx_path.string().c_str(), fmt));

    if (!faidx_ptr) {
        throw std::runtime_error{"Could not open .fai index file from path: '" +
                                 fastx_path.string() + "'."};
    }

    return faidx_ptr;
}
}  // namespace

FastxRandomReader::FastxRandomReader(const std::filesystem::path& fastx_path) {
    // Convert the string to lowercase.
    std::string path_str = fastx_path.string();
    std::transform(std::begin(path_str), std::end(path_str), std::begin(path_str),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const SequenceFormatType fmt = parse_sequence_format(fastx_path);

    const fai_format_options fai_fmt = (fmt == SequenceFormatType::FASTA)   ? FAI_FASTA
                                       : (fmt == SequenceFormatType::FASTQ) ? FAI_FASTQ
                                                                            : FAI_NONE;

    m_faidx = open_fai(fastx_path, fai_fmt);

    // Both attempts failed.
    if (!m_faidx) {
        throw std::runtime_error{"Could not create/load index for FASTx file " +
                                 fastx_path.string()};
    }
}

std::string FastxRandomReader::fetch_seq(const std::string& reg) const {
    return kadayashi::hts_utils::fetch_seq(m_faidx.get(), reg);
}

int FastxRandomReader::fetch_seq_len(const std::string& tn) const {
    return kadayashi::hts_utils::fetch_seq_len(m_faidx.get(), tn);
}

std::vector<uint8_t> FastxRandomReader::fetch_qual(const std::string& read_id) const {
    return kadayashi::hts_utils::fetch_qual(m_faidx.get(), read_id);
}

faidx_t* FastxRandomReader::get_raw_faidx_ptr() { return m_faidx.get(); }

int FastxRandomReader::num_entries() const { return faidx_nseq(m_faidx.get()); }

}  // namespace kadayashi::hts_utils
