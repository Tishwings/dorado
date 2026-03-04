#include "bam_tagging.h"
#include "cli.h"
#include "kadayashi_utils.h"
#include "local_haplotagging.h"
#include "pipeline.h"
#include "resources.h"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

void print_help_main() {
    // clang-format off
    fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
    fprintf(stdout, "Usage:\n");
    fprintf(stdout, "  kadayashi phase -o prefix --ref ref.fa fn_bam 2>log\n");
    fprintf(stdout, "  kadayashi modifytag --ref hg38.fa --readtags fn_tsv_0index -o out.bam [-i chrom:s-e] [-t threads] fn_bam #note: use 2-col tsv mapping qname to haptag (0-index), not kadayashi tsv\n");
    fprintf(stdout, "  kadayashi tsv2bin -o out.bin in.tsv\n");
    fprintf(stdout, "  kadayashi phaseglobal -r fn_ref [-o output.vcf] [-t threads] fn_bam fn_vcf\n");
    fprintf(stdout, "  kadayashi varcall [--varcall-use-dvr] -o prefix [-t threads -b -r REGION -w window_size] fn_ref fn_bam 2>log\n");
    fprintf(stdout, "  kadayashi dbg #runs whatever is in the current main_debug()\n");
    // clang-format on
}

void print_end_summary(int argc, char *argv[], double T) {
    fprintf(stdout, "\n[M::%s] ELAPSED: %.1fs , PeakRSS: %.1f GiB\n", __func__,
            kadayashi::get_timestamp() - T, kadayashi::get_peakrss());
    fprintf(stdout, "[M::%s] kadayashi %s\n", __func__, KADAYASHI_VERSION);
    fprintf(stdout, "[M::%s] CMD: ", __func__);
    for (int i = 0; i < argc; i++) {
        fprintf(stdout, "%s ", argv[i]);
    }
    fprintf(stdout, "\n");
}

int main_phase(cliopt_t &clio) {
    int ret = kadayashi::local_haplotagging(clio.fn_bam, clio.fn_bed, clio.slice_in_bed,
                                            clio.fn_ref, clio.one_region_str, clio.one_chrom_str,
                                            clio.fn_vcf, clio.chunk_l, clio.chunk_stride, clio.pp,
                                            clio.n_threads,  // threads
                                            BAM_THREAD_MULT,
                                            clio.n_threads * 2,  // chunk per batch
                                            clio.fn_tsv, clio.use_simple);
    return ret;
}

int main_debug(int argc, char *argv[]) {
    // clang-format off
    //kadayashi::dorado::secondary::BamFile hf{argv[1], 1};
    //faidx_t *fai = fai_load_format(argv[2], FAI_FASTA);
    //const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
    //            hf.fp(), hf.idx(), hf.hdr(),
    //            fai, "chr20", 1, 1000000, false, 5,
    //            5, 0.2f, 200, 0.1f, false);
    //for (auto &var : result.variants){
    //    if (var.is_valid)
    //        fprintf(stdout, "%d %s %s\n", var.pos0, var.ref_allele_seq0.c_str(), var.alt_allele_seq0.c_str());
    //}
    return 0;
    // clang-format on
}

int main(int argc, char *argv[]) {
    int ret = 0;
    double T = kadayashi::get_timestamp();

    spdlog::set_level(spdlog::level::trace);

    if (argc < 2) {
        print_help_main();
        return 1;
    }

    const std::string_view subcommand{argv[1]};

    if (subcommand == "phase") {
        cliopt_t clio = parse_cli(argc - 1, argv + 1);
        if ((!clio.is_valid) || clio.is_print_help) {
            return 1;
        }

        ret = main_phase(clio);
        if (ret != 0) {
            spdlog::error("[kdys::{}] phasing failed; not generating bin file from tsv output",
                          __func__);
            print_end_summary(argc, argv, T);
            return ret;
        }

        std::string fn_out_binary = clio.output_prefix.string() + ".bin";
        kadayashi::write_binary_given_tsv(clio.fn_tsv, fn_out_binary);
        if (clio.write_dbg_bam) {
            std::string fn_out_bam = fn_out_binary + ".bam";
            const int writebam_err = kadayashi::write_haptagged_bam_given_bin_and_itvl(
                    clio.fn_bam, fn_out_binary, clio.one_region_str, fn_out_bam, clio.n_threads,
                    BAM_THREAD_MULT);
            if (writebam_err) {
                spdlog::error("[kdys::{}] failed to write debug haptagged bam, check code!",
                              __func__);
            } else {
                spdlog::info("[kdys::{}] wrote debug haptagged bam", __func__);
            }
        }
    } else if (subcommand == "modifytag") {
        cliopt_t clio = parse_cli(argc - 1, argv + 1);
        if ((!clio.is_valid) || clio.is_print_help) {
            return 1;
        }
        if (clio.fn_tsv.empty()) {
            spdlog::error("[kdys::{}] must provide read-haptag mapping file (2-col tsv, 0-index)",
                          __func__);
            return 1;
        }
        ret = kadayashi::util_modify_tag_given_tsv(clio.fn_bam, clio.fn_tsv, clio.interval_str,
                                                   clio.n_threads, BAM_THREAD_MULT,
                                                   clio.output_prefix);
    } else if (subcommand == "tsv2bin") {
        cliopt_t clio = parse_cli_tsv2bin(argc - 1, argv + 1);
        if (!clio.is_valid || clio.is_print_help) {
            return 1;
        }

        if (clio.fn_tsv.empty()) {
            spdlog::error("[kdys::{}] must provide tsv file name", __func__);
        }

        kadayashi::write_binary_given_tsv(clio.fn_tsv, clio.output_name);
    } else if (subcommand == "bin2bam") {
        // given bam, bin and an interval, output bam with modified haplotag. Debug util.
        cliopt_t clio = parse_cli_bin2bam(argc - 1, argv + 1);
        if (!clio.is_valid || clio.is_print_help) {
            return 1;
        }
        ret = kadayashi::write_haptagged_bam_given_bin_and_itvl(
                clio.fn_bam, clio.fn_bin, clio.one_region_str, clio.output_name, clio.n_threads,
                BAM_THREAD_MULT);
    } else if (subcommand == "phaseglobal") {
        cliopt_phaseglobal_t clio = parse_cli_phaseglobal(argc - 1, argv + 1);
        if (!clio.is_valid || clio.is_print_help) {
            return 1;
        }

        spdlog::info("[kdys::{}] start phasing with {} threads...", __func__, clio.n_threads);
        auto qname2hp = kadayashi::kadayashi_global_phasing_simple_modify_vcf(
                clio.fn_ref, clio.fn_bam, clio.fn_vcf, clio.output_name, clio.n_threads);

        if (!clio.fn_out_bam.empty()) {
            spdlog::info("[kdys::{}] To write haptagged bam...", __func__);
            double T = kadayashi::get_timestamp();
            kadayashi::write_haptagged_bam_given_hashtable_and_itvl(
                    clio.fn_bam, ".", clio.fn_out_bam, qname2hp, clio.n_threads);
            spdlog::info("[kdys::{}] haptagged bam written, used %.1fs", __func__,
                         kadayashi::get_timestamp() - T);
        }
        ret = 0;
    } else if (subcommand == "varcall") {
        cliopt_varcall_t clio = parse_cli_varcall(argc - 1, argv + 1);
        if (!clio.is_valid || clio.is_print_help) {
            return 1;
        }

        double T = kadayashi::get_timestamp();
        std::string fn_out_vcf = clio.output_prefix.string() + ".vcf";
        kadayashi::str2int_t qname2hp = kadayashi::kadayashi_phased_variant_calling_threaded(
                clio.fn_ref, clio.fn_bam, clio.n_threads, fn_out_vcf, clio.output_prefix,
                clio.varcall_w, clio.varcall_regions, clio.pp.min_base_quality,
                clio.pp.min_varcall_coverage, clio.pp.min_varcall_fraction, clio.pp.max_clipping,
                clio.pp.min_strand_cov, clio.pp.min_strand_cov_frac,
                clio.pp.max_gapcompressed_seqdiv, clio.vcf_write_allow_refbase_N,
                clio.pp.disable_region_expansion, clio.varcall_use_dvr, clio.bed_flanking);
        spdlog::info("[kdys::{}] varcall main routine done, used %.1fs", __func__,
                     kadayashi::get_timestamp() - T);

        if (clio.write_dbg_bam) {
            T = kadayashi::get_timestamp();
            std::string fn_out_bam = clio.output_prefix.string() + ".kadayashi.bam";
            kadayashi::write_haptagged_bam_given_hashtable_and_multiple_itvls(
                    clio.fn_bam, clio.varcall_regions, fn_out_bam, qname2hp, clio.n_threads);
            spdlog::info("[kdys::{}] haptagged bam written, used %.1fs", __func__,
                         kadayashi::get_timestamp() - T);
        }
        ret = 0;
    } else if (subcommand == "--version") {
        fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
        ret = 0;
    } else if (subcommand == "dbg") {
        ret = main_debug(argc - 1, argv + 1);
    } else {
        fprintf(stdout, "[E::%s] unknown subcommand: %s\n", __func__, argv[1]);
        print_help_main();
        return 1;
    }

    print_end_summary(argc, argv, T);
    return ret;
}
