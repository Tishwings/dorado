#include "TestUtils.h"
#include "secondary/architectures/model_config.h"
#include "secondary/architectures/model_config_validation.h"

#include <catch2/catch_test_macros.hpp>
#include <toml.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary::tests {
namespace {

#define TEST_GROUP "[SecondaryModelConfigValidation]"

const std::string GRU_COUNTS_V1 = R"toml(
config_version = 1
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"

[model]
type = "GRUModel"

[feature_encoder]
type = "CountsFeatureEncoder"

[label_scheme]
type = "HaploidLabelScheme"

[model.kwargs]
num_features = 10
num_classes = 5
gru_size = 128
n_layers = 2
bidirectional = true

[feature_encoder.kwargs]
normalise = "total"
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
sym_indels = false
)toml";

const std::string GRU_COUNTS_V2 = R"toml(
config_version = 2
basecaller_model = "unknown"
supported_basecallers = ["dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"]

[model]
type = "GRUModel"

[feature_encoder]
type = "CountsFeatureEncoder"

[label_scheme]
type = "HaploidLabelScheme"

[model.kwargs]
num_features = 10
num_classes = 5
gru_size = 128
n_layers = 2
bidirectional = true

[feature_encoder.kwargs]
normalise = "total"
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
sym_indels = false
)toml";

const std::string LSTM_READ_ALIGNMENT_V1 = R"toml(
config_version = 1
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"

[model]
type = "LatentSpaceLSTM"

[feature_encoder]
type = "ReadAlignmentFeatureEncoder"

[label_scheme]
type = "HaploidLabelScheme"

[model.kwargs]
num_classes = 5
lstm_size = 384
cnn_size = 64
kernel_sizes = [ 1, 17,]
pooler_type = "mean"
use_dwells = false
bases_alphabet_size = 6
bases_embedding_size = 6
bidirectional = false

[feature_encoder.kwargs]
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
max_reads = 100
row_per_read = false
include_dwells = false
include_haplotype = false

[model.kwargs.pooler_args]
)toml";

const std::string LSTM_READ_ALIGNMENT_V3 = R"toml(
config_version = 3
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
supported_basecallers = [ "dna_r10.4.1_e8.2_400bps_hac@v5.2.0",]

[model]
type = "LatentSpaceLSTM"

[feature_encoder]
type = "ReadAlignmentFeatureEncoder"

[label_scheme]
type = "HaploidLabelScheme"

[model.kwargs]
num_classes = 5
lstm_size = 384
cnn_size = 64
kernel_sizes = [ 1, 17,]
pooler_type = "mean"
use_dwells = false
bases_alphabet_size = 6
bases_embedding_size = 6
bidirectional = false

[feature_encoder.kwargs]
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
max_reads = 100
row_per_read = false
include_dwells = false
include_haplotype = false

[model.kwargs.pooler_args]
)toml";

const std::string SLOT_ATTENTION_VARIANT_V3 = R"toml(
config_version = 3
supported_basecallers = [ "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",]
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"

[model]
type = "SlotAttentionConsensus"

[feature_encoder]
type = "ReadAlignmentFeatureEncoder"

[label_scheme]
type = "DiploidLabelScheme"

[model.kwargs]
num_slots = 2
classes_per_slot = 5
read_embedding_size = 192
cnn_size = 64
kernel_sizes = [ 1, 17,]
pooler_type = "mean"
use_mapqc = true
use_dwells = true
use_haplotags = true
bases_alphabet_size = 6
bases_embedding_size = 6
add_lstm = true
use_reference = false

[feature_encoder.kwargs]
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
max_reads = 100
row_per_read = false
include_dwells = true
include_haplotype = true
right_align_insertions = true
region_split = 100000

[label_scheme.kwargs]
ordered = true
right_align_insertions = true

[model.kwargs.pooler_args]
)toml";

const std::string VARIANT_PERCEIVER_V4 = R"toml(
config_version = 4
supported_basecallers = [ "dna_r10.4.1_e8.2_400bps_hac@v5.2.0",]
chunk_size = 300
chunk_overlap = 100

[model]
type = "VariantPerceiver"

[feature_encoder]
type = "ReadAlignmentFeatureEncoder"

[label_scheme]
type = "DiploidLabelScheme"

[model.kwargs]
read_max_depth = 100
ploidy = 2
num_classes = 5
cnn_size = 64
kernel_sizes = [ 1, 17,]
dimension = 256
num_blocks = 2
num_heads = 8
self_attn_layers_per_block = 8
use_mapqc = true
use_dwells = true
use_haplotags = true
use_snp_qv = true
bases_alphabet_size = 6
bases_embedding_size = 6
use_decoder_lstm = false
use_per_read_embedding = false
embedding_type = "learned"
shuffle_embeddings = false
update_read_embeddings = false
latent_init_method = "ref_seq"
mask_partial_rows = true
add_null_tokens = false

[feature_encoder.kwargs]
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
max_reads = 100
row_per_read = false
include_dwells = true
include_haplotype = true
include_snp_qv = true
right_align_insertions = true
region_split = 100000

[label_scheme.kwargs]
ploidy = 2
)toml";

std::string replace_first(std::string config, const std::string& from, const std::string& to) {
    const std::size_t pos = config.find(from);
    CATCH_REQUIRE(pos != std::string::npos);
    config.replace(pos, std::size(from), to);
    return config;
}

std::string erase_first(std::string config, const std::string& text) {
    return replace_first(std::move(config), text, "");
}

std::string insert_after(std::string config, const std::string& marker, const std::string& text) {
    const std::size_t pos = config.find(marker);
    CATCH_REQUIRE(pos != std::string::npos);
    config.insert(pos + std::size(marker), text);
    return config;
}

void check_valid_config(const std::string& config) {
    CATCH_CAPTURE(config);
    CATCH_CHECK_NOTHROW(validate_model_config_toml(toml::parse_str(config)));
}

void check_invalid_config(const std::string& config) {
    CATCH_CAPTURE(config);
    CATCH_CHECK_THROWS(validate_model_config_toml(toml::parse_str(config)));
}

ModelConfig parse_config_string(const std::string& config) {
    const TempDir temp_dir = make_temp_dir("secondary_model_config");
    const std::filesystem::path config_path = temp_dir.m_path / "config.toml";
    std::ofstream output(config_path);
    output << config;
    output.close();
    return parse_model_config(config_path, "weights.pt");
}

ModelConfig make_lstm_config(const int32_t version) {
    ModelConfig config;
    config.version = version;
    config.model_type = "LatentSpaceLSTM";
    config.model_kwargs = {
            {"num_classes", "5"},         {"lstm_size", "384"},
            {"cnn_size", "64"},           {"pooler_type", "mean"},
            {"bases_alphabet_size", "6"}, {"bases_embedding_size", "6"},
            {"kernel_sizes", "[1, 17]"},  {"use_dwells", "false"},
    };
    return config;
}

ModelConfig make_read_alignment_config(const int32_t version) {
    ModelConfig config;
    config.version = version;
    config.feature_encoder_type = "ReadAlignmentFeatureEncoder";
    config.feature_encoder_kwargs = {
            {"dtypes", "[\"\"]"},
            {"tag_keep_missing", "false"},
            {"min_mapq", "1"},
            {"max_reads", "100"},
            {"row_per_read", "false"},
            {"include_dwells", "false"},
            {"include_haplotype", "false"},
            {"right_align_insertions", "true"},
    };
    return config;
}

}  // namespace

CATCH_TEST_CASE("Versioned value lookup respects support windows and defaults", TEST_GROUP) {
    const VersionedParamSpec windowed{
            .name = "example",
            .type = ModelConfigValueType::INTEGER,
            .min_version = 3,
            .max_version = 5,
            .below_min_version = ValueUnavailableHandling::USE_DEFAULT,
            .default_before_min_version = "11",
            .above_max_version = ValueUnavailableHandling::USE_DEFAULT,
            .default_after_max_version = "17",
    };
    const VersionedParamSpec strict{
            .name = "example",
            .type = ModelConfigValueType::INTEGER,
            .min_version = 3,
            .max_version = 5,
    };
    const VersionedParamSpec missing_default{
            .name = "missing",
            .type = ModelConfigValueType::STRING,
            .missing_value = ValueUnavailableHandling::USE_DEFAULT,
            .default_value = "fallback",
    };
    const VersionedParamSpec missing_parse_or_use_default{
            .name = "missing",
            .type = ModelConfigValueType::STRING,
            .missing_value = ValueUnavailableHandling::PARSE_OR_USE_DEFAULT,
            .default_value = "fallback-if-missing",
    };
    const VersionedParamSpec early_parse_or_use_default{
            .name = "example",
            .type = ModelConfigValueType::INTEGER,
            .min_version = 3,
            .below_min_version = ValueUnavailableHandling::PARSE_OR_USE_DEFAULT,
            .default_before_min_version = "11",
    };

    const std::unordered_map<std::string, std::string> values{
            {"example", "13"},
            {"quoted", "\"abc\""},
    };
    const VersionedParamSpec quoted{.name = "quoted", .type = ModelConfigValueType::STRING};

    CATCH_CHECK(get_versioned_value(values, windowed, 2, "test") == "11");
    CATCH_CHECK(get_versioned_value(values, windowed, 3, "test") == "13");
    CATCH_CHECK(get_versioned_value(values, windowed, 4, "test") == "13");
    CATCH_CHECK(get_versioned_value(values, windowed, 5, "test") == "13");
    CATCH_CHECK(get_versioned_value(values, windowed, 6, "test") == "17");
    CATCH_CHECK(get_versioned_value(values, early_parse_or_use_default, 2, "test") == "13");
    CATCH_CHECK(get_versioned_value({}, early_parse_or_use_default, 2, "test") == "11");
    CATCH_CHECK(get_versioned_value(values, early_parse_or_use_default, 3, "test") == "13");
    CATCH_CHECK_THROWS(get_versioned_value(values, strict, 2, "test"));
    CATCH_CHECK(get_versioned_value(values, strict, 5, "test") == "13");
    CATCH_CHECK_THROWS(get_versioned_value(values, strict, 6, "test"));
    CATCH_CHECK(get_versioned_value({}, windowed, 2, "test") == "11");
    CATCH_CHECK(get_versioned_value(values, missing_default, 1, "test") == "fallback");
    CATCH_CHECK(get_versioned_value(values, missing_parse_or_use_default, 1, "test") ==
                "fallback-if-missing");
    CATCH_CHECK(get_versioned_value(values, quoted, 1, "test") == "abc");
}

CATCH_TEST_CASE("Current shipped secondary model configs validate", TEST_GROUP) {
    const std::vector<std::pair<std::string, std::string>> configs{
            {"GRUModel CountsFeatureEncoder v1", GRU_COUNTS_V1},
            {"GRUModel CountsFeatureEncoder v2", GRU_COUNTS_V2},
            {"LatentSpaceLSTM ReadAlignmentFeatureEncoder v1", LSTM_READ_ALIGNMENT_V1},
            {"LatentSpaceLSTM ReadAlignmentFeatureEncoder v3", LSTM_READ_ALIGNMENT_V3},
            {"SlotAttentionConsensus variant v3", SLOT_ATTENTION_VARIANT_V3},
            {"VariantPerceiver variant v4", VARIANT_PERCEIVER_V4},
    };

    for (const std::pair<std::string, std::string>& config : configs) {
        CATCH_CAPTURE(config.first);
        check_valid_config(config.second);
    }
}

CATCH_TEST_CASE("ModelConfig parser applies top-level chunk defaults and config values",
                TEST_GROUP) {
    const ModelConfig v1_config = parse_config_string(LSTM_READ_ALIGNMENT_V1);
    CATCH_CHECK(v1_config.chunk_size == 10000);
    CATCH_CHECK(v1_config.chunk_overlap == 1000);

    const ModelConfig v4_config = parse_config_string(VARIANT_PERCEIVER_V4);
    CATCH_CHECK(v4_config.chunk_size == 300);
    CATCH_CHECK(v4_config.chunk_overlap == 100);
}

CATCH_TEST_CASE("Top-level validation catches version boundaries and unknown keys", TEST_GROUP) {
    check_invalid_config(erase_first(GRU_COUNTS_V1, "config_version = 1\n"));
    check_invalid_config(erase_first(GRU_COUNTS_V1,
                                     "basecaller_model = "
                                     "\"dna_r10.4.1_e8.2_400bps_hac@v4.2.0\"\n"));
    check_invalid_config(erase_first(VARIANT_PERCEIVER_V4, "chunk_size = 300\n"));
    check_invalid_config(erase_first(VARIANT_PERCEIVER_V4, "chunk_overlap = 100\n"));
    check_invalid_config(
            insert_after(GRU_COUNTS_V1, "config_version = 1\n", "undocumented = true\n"));
    check_invalid_config(VARIANT_PERCEIVER_V4 +
                         "\n[inference_parameters]\nchunk_size = 300\nchunk_overlap = 100\n");

    check_valid_config(LSTM_READ_ALIGNMENT_V3);
}

CATCH_TEST_CASE("Validation rejects missing required sections and parameters", TEST_GROUP) {
    check_invalid_config(R"toml(
config_version = 1
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"

[feature_encoder]
type = "CountsFeatureEncoder"

[feature_encoder.kwargs]
normalise = "total"
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
sym_indels = false

[label_scheme]
type = "HaploidLabelScheme"
)toml");

    check_invalid_config(R"toml(
config_version = 1
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"

[model]
type = "GRUModel"

[feature_encoder]
type = "CountsFeatureEncoder"

[feature_encoder.kwargs]
normalise = "total"
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
sym_indels = false

[label_scheme]
type = "HaploidLabelScheme"
)toml");

    check_invalid_config(erase_first(GRU_COUNTS_V1, "num_classes = 5\n"));
    check_invalid_config(erase_first(GRU_COUNTS_V1, "dtypes = [ \"\",]\n"));
    check_invalid_config(erase_first(LSTM_READ_ALIGNMENT_V3, "max_reads = 100\n"));
    check_invalid_config(erase_first(VARIANT_PERCEIVER_V4, "right_align_insertions = true\n"));
}

CATCH_TEST_CASE("Validation rejects extra keys in known tables", TEST_GROUP) {
    check_invalid_config(insert_after(GRU_COUNTS_V1, "[model.kwargs]\n", "extra_model_key = 1\n"));
    check_invalid_config(insert_after(GRU_COUNTS_V1, "[feature_encoder.kwargs]\n",
                                      "extra_encoder_key = true\n"));
    check_invalid_config(insert_after(VARIANT_PERCEIVER_V4, "[label_scheme.kwargs]\n",
                                      "extra_label_key = true\n"));
    check_invalid_config(GRU_COUNTS_V1 + "\n[model.kwargs.extra]\nvalue = 1\n");
}

CATCH_TEST_CASE("Validation rejects TOML type mismatches", TEST_GROUP) {
    check_invalid_config(
            replace_first(GRU_COUNTS_V1, "config_version = 1", "config_version = \"1\""));
    check_invalid_config(R"toml(
config_version = 1
basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"
model = "not a table"

[feature_encoder]
type = "CountsFeatureEncoder"

[feature_encoder.kwargs]
normalise = "total"
dtypes = [ "",]
tag_keep_missing = false
min_mapq = 1
sym_indels = false

[label_scheme]
type = "HaploidLabelScheme"
)toml");
    check_invalid_config(replace_first(GRU_COUNTS_V1, "num_classes = 5", "num_classes = \"5\""));
    check_invalid_config(replace_first(LSTM_READ_ALIGNMENT_V3, "kernel_sizes = [ 1, 17,]",
                                       "kernel_sizes = \"1,17\""));
    check_invalid_config(replace_first(VARIANT_PERCEIVER_V4, "include_dwells = true",
                                       "include_dwells = \"true\""));
}

CATCH_TEST_CASE("Validation rejects unknown and unsupported section types", TEST_GROUP) {
    check_invalid_config(
            replace_first(GRU_COUNTS_V1, "type = \"GRUModel\"", "type = \"UnknownModel\""));
    check_invalid_config(replace_first(GRU_COUNTS_V1, "type = \"CountsFeatureEncoder\"",
                                       "type = \"UnknownFeatureEncoder\""));
    check_invalid_config(replace_first(GRU_COUNTS_V1, "type = \"HaploidLabelScheme\"",
                                       "type = \"UnknownLabelScheme\""));
    check_invalid_config(
            replace_first(SLOT_ATTENTION_VARIANT_V3, "config_version = 3", "config_version = 2"));
    check_invalid_config(replace_first(GRU_COUNTS_V2, "type = \"HaploidLabelScheme\"",
                                       "type = \"DiploidLabelScheme\""));
    check_invalid_config(replace_first(VARIANT_PERCEIVER_V4, "config_version = 4\n",
                                       "config_version = 3\nbasecaller_model = "
                                       "\"dna_r10.4.1_e8.2_400bps_hac@v5.2.0\"\n"));
}

CATCH_TEST_CASE("ModelConfig getter helpers apply versioned defaults and throw at boundaries",
                TEST_GROUP) {
    ModelConfig lstm_v2 = make_lstm_config(2);
    ModelConfig lstm_v3 = make_lstm_config(3);
    ModelConfig read_alignment_v3 = make_read_alignment_config(3);
    ModelConfig read_alignment_v4 = make_read_alignment_config(4);

    lstm_v2.model_kwargs.emplace("bidirectional", "false");
    CATCH_CHECK(get_model_config_model_value(lstm_v2, "bidirectional") == "false");
    lstm_v2.model_kwargs.erase("bidirectional");
    CATCH_CHECK_THROWS(get_model_config_model_value(lstm_v2, "bidirectional"));
    CATCH_CHECK_THROWS(get_model_config_model_value(lstm_v3, "bidirectional"));
    CATCH_CHECK(get_model_config_feature_encoder_value(read_alignment_v3, "include_snp_qv") ==
                "false");
    CATCH_CHECK(get_model_config_feature_encoder_value(read_alignment_v3,
                                                       "right_align_insertions") == "true");
    read_alignment_v3.feature_encoder_kwargs.erase("right_align_insertions");
    CATCH_CHECK(get_model_config_feature_encoder_value(read_alignment_v3,
                                                       "right_align_insertions") == "false");
    CATCH_CHECK_THROWS(get_model_config_feature_encoder_value(read_alignment_v4, "include_snp_qv"));
    read_alignment_v4.feature_encoder_kwargs.erase("right_align_insertions");
    CATCH_CHECK_THROWS(
            get_model_config_feature_encoder_value(read_alignment_v4, "right_align_insertions"));
    CATCH_CHECK_THROWS(get_model_config_model_value(lstm_v2, "unknown"));
    CATCH_CHECK_THROWS(get_model_config_feature_encoder_value(read_alignment_v3, "unknown"));
}

}  // namespace dorado::secondary::tests
