#include "bam_tagging.h"

#include "BamFile.h"
#include "kadayashi_utils.h"
#include "kthread.h"
#include "local_haplotagging.h"
#include "pipeline.h"

#include <htslib/bgzf.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdlib>
#include <sstream>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace kadayashi {

int write_haptagged_bam_given_hashtable_and_itvl(
        const std::filesystem::path &fn_bam,
        const std::string_view itvl,
        const std::filesystem::path &fn_out,
        const std::unordered_map<std::string, int> &qname2hp,
        const int n_threads) {
    assert(!fn_bam.native().empty());
    assert(!itvl.empty());
    int ret = 0;

    hts_utils::BamFile hf{fn_bam, n_threads / 2};
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), itvl.data()), HtsItrDestructor());

    BGZF *fp_out = bgzf_open(fn_out.string().c_str(), "w");
    if (!fp_out) {
        spdlog::error("[{}] failed to open output file: {}\n", __func__, fn_out.string());
        return 1;
    }
    int stat = bam_hdr_write(fp_out, hf.hdr());
    if (stat != 0) {
        spdlog::error("[{}] output bam header write failed\n", __func__);
        bgzf_close(fp_out);
        return 1;
    }

    BamPtr aln = BamPtr(bam_init1(), BamDestructor());
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        const char *refname = hf.hdr()->target_name[aln->core.tid];
        const int start_pos = static_cast<int>(aln->core.pos);
        const std::string qn = bam_get_qname(aln.get());

        const auto it = qname2hp.find(qn);
        const int haptag = it == qname2hp.cend() ? HAPTAG_UNPHASED : it->second;

        bam_aux_update_int(aln.get(), "HP", haptag + 1);
        stat = bam_write1(fp_out, aln.get());
        if (stat < 0) {
            spdlog::error("[{}] failed to write bam entry (ref={} pos={} qn={} newhp={})\n",
                          __func__, refname, start_pos, qn, haptag);
        }
    }
    aln = {};

    // index output file
    bgzf_close(fp_out);
    const std::string fn_bai_out = fn_out.string() + ".bai";
    stat = sam_index_build3(fn_out.string().c_str(), fn_bai_out.c_str(), 0, 1);
    if (stat != 0) {
        spdlog::error("[{}] failed to index output (stat={})\n", __func__, stat);
        ret = 1;
    }

    return ret;
}

int write_haptagged_bam_given_bin_and_itvl(const std::filesystem::path &fn_bam,
                                           const std::filesystem::path &fn_bin,
                                           const std::string_view itvl,
                                           const std::filesystem::path &fn_out,
                                           const int n_threads,
                                           const int n_bam_threads) {
    assert(!fn_bam.native().empty());
    assert(!fn_bin.native().empty());
    assert(!itvl.empty());
    int ret = 0;

    const region_string_t region = parse_region_string2(itvl);
    if (!region.is_parse_success || region.is_whole_chrom) {
        spdlog::error(
                "[{}] failed to parse region string (string={} stat={}) or range not fully "
                "specified "
                "(is_whole_chrom={} (need to provide start&end))\n",
                __func__, itvl.data(), region.is_parse_success ? "true" : "false",
                region.is_whole_chrom ? "true" : "false");
        return 1;
    }

    hts_utils::BamFile hf{fn_bam, n_bam_threads};
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), itvl.data()), HtsItrDestructor());

    const std::unordered_map<std::string, int> qname2hp =
            query_bin_file_get_qname2hp(fn_bin, region.chrom, region.start, region.end);

    spdlog::info("[{}] write to {}\n", __func__, fn_out.string());
    BGZF *fp_out = bgzf_open(fn_out.string().c_str(), "w");
    if (!fp_out) {
        spdlog::error("[{}] failed to open output file: {}\n", __func__, fn_out.string());
        return 1;
    }
    if (n_threads > 1) {
        bgzf_mt(fp_out, n_threads, 0 /*unused*/);
    }
    int stat = bam_hdr_write(fp_out, hf.hdr());
    if (stat != 0) {
        spdlog::error("[{}] output bam header write failed\n", __func__);
        bgzf_close(fp_out);
        return 1;
    }

    BamPtr aln = BamPtr(bam_init1(), BamDestructor());
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        const std::string qn = bam_get_qname(aln.get());

        const auto it = qname2hp.find(qn);
        const int haptag = it == qname2hp.cend() ? HAPTAG_UNPHASED : it->second;

        bam_aux_update_int(aln.get(), "HP", haptag + 1);  // need to use 1-index
        stat = bam_write1(fp_out, aln.get());
        if (stat < 0) {
            const char *refname = hf.hdr()->target_name[aln->core.tid];
            const int start_pos = static_cast<int>(aln->core.pos);
            spdlog::error(
                    "[{}] failed to write bam entry (ref={} pos={} qn={} newhp={} (0-index))\n",
                    __func__, refname, start_pos, qn, haptag);
        }
    }
    aln = {};

    // index output file
    bgzf_close(fp_out);
    const std::string fn_bai_out = std::string(fn_out) + ".bai";
    stat = sam_index_build3(fn_out.string().c_str(), fn_bai_out.c_str(), 0, n_threads);
    if (stat != 0) {
        spdlog::error("[{}] failed to index output (stat={})\n", __func__, stat);
        ret = 1;
    }

    return ret;
}

int write_haptagged_bam_given_hashtable_and_multiple_itvls(
        const std::filesystem::path &fn_bam,
        const std::vector<std::string> &query_regions,
        const std::filesystem::path &fn_out,
        const std::unordered_map<std::string, int> &qname2hp,
        const int n_threads) {
    int ret = 0;
    if (query_regions.size() == 0) {
        write_haptagged_bam_given_hashtable_and_itvl(fn_bam, ".", fn_out, qname2hp, n_threads);
        return ret;
    }

    // prep input
    assert(!fn_bam.native().empty());
    hts_utils::BamFile hf{fn_bam, n_threads};

    // prep output
    BGZF *fp_out = bgzf_open(fn_out.string().c_str(), "w");
    if (!fp_out) {
        spdlog::error("[{}] failed to open output file: {}\n", __func__, fn_out.string());
        return 1;
    }
    if (n_threads > 1) {
        bgzf_mt(fp_out, n_threads / 2, 0 /*unused*/);
    }

    // output header
    int stat = bam_hdr_write(fp_out, hf.hdr());
    if (stat != 0) {
        spdlog::error("[{}] output bam header write failed\n", __func__);
        bgzf_close(fp_out);
        return 1;
    }

    // calcualte sorted, non-overlapping intervals
    intervals_t query_interavls = region_strings_to_intervals(hf, 0, query_regions);

    for (int i_ref = 0; i_ref < hf.hdr()->n_targets; i_ref++) {
        char *chrom = hf.hdr()->target_name[i_ref];

        if (query_interavls.size() != 0 && query_interavls.find(chrom) == query_interavls.end()) {
            continue;
        }

        if (query_interavls[chrom].size() == 0) {
            query_interavls[chrom].push_back({0, hf.hdr()->target_len[i_ref]});
        }

        BamPtr aln = BamPtr(bam_init1(), BamDestructor());
        for (auto &_ : query_interavls[chrom]) {
            std::string itvl_s = chrom;
            itvl_s.append(":");
            itvl_s.append(std::to_string(_.first));
            itvl_s.append("-");
            itvl_s.append(std::to_string(_.second));
            LOG_TRACE("[{}] writing bam: {}\n", __func__, itvl_s);

            HtsItrPtr bamitr = HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), itvl_s.c_str()),
                                         HtsItrDestructor());
            int haptag;
            while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
                char *refname = hf.hdr()->target_name[aln->core.tid];
                int start_pos = static_cast<uint32_t>(aln->core.pos);
                std::string qn = bam_get_qname(aln);

                auto it = qname2hp.find(qn);
                if (it == qname2hp.cend()) {
                    haptag = HAPTAG_UNPHASED;
                } else {
                    haptag = it->second;
                }

                bam_aux_update_int(aln.get(), "HP", haptag + 1);
                stat = bam_write1(fp_out, aln.get());
                if (stat < 0) {
                    spdlog::error(
                            "[{}] failed to write bam entry (ref={} pos={} qn={} newhp={} "
                            "(0-index))\n",
                            __func__, refname, start_pos, qn, haptag);
                }
            }
        }
        aln = {};
    }

    // index output file
    bgzf_close(fp_out);
    std::string fn_bai_out = fn_out.string() + ".bai";
    stat = sam_index_build3(fn_out.string().c_str(), fn_bai_out.c_str(), 0, 1);
    if (stat != 0) {
        spdlog::error("[{}] failed to index output (stat={})\n", __func__, stat);
        ret = 1;
    }

    return ret;
}

}  // namespace kadayashi
