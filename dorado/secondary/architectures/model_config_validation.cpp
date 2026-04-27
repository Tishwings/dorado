#include "secondary/architectures/model_config_validation.h"

#include <toml.hpp>

#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {
namespace {

using ParamSpecs = std::vector<VersionedParamSpec>;
using SpecMap = std::unordered_map<std::string, ParamSpecs>;

VersionedParamSpec required(std::string name, ModelConfigValueType type) {
    return {.name = std::move(name), .type = type};
}

VersionedParamSpec optional_default(std::string name,
                                    ModelConfigValueType type,
                                    std::string default_value) {
    return {.name = std::move(name),
            .type = type,
            .missing_value = ValueUnavailableHandling::USE_DEFAULT,
            .default_value = std::move(default_value)};
}

VersionedParamSpec required_since(std::string name,
                                  ModelConfigValueType type,
                                  int32_t min_version,
                                  std::string default_before_min_version) {
    return {.name = std::move(name),
            .type = type,
            .min_version = min_version,
            .below_min_version = ValueUnavailableHandling::USE_DEFAULT,
            .default_before_min_version = std::move(default_before_min_version)};
}

VersionedParamSpec required_since_parse_or_use_default(std::string name,
                                                       ModelConfigValueType type,
                                                       int32_t min_version,
                                                       std::string default_before_min_version) {
    return {.name = std::move(name),
            .type = type,
            .min_version = min_version,
            .below_min_version = ValueUnavailableHandling::PARSE_OR_USE_DEFAULT,
            .default_before_min_version = std::move(default_before_min_version)};
}

VersionedParamSpec required_until(std::string name,
                                  ModelConfigValueType type,
                                  int32_t max_version,
                                  std::string default_after_max_version) {
    return {.name = std::move(name),
            .type = type,
            .max_version = max_version,
            .above_max_version = ValueUnavailableHandling::USE_DEFAULT,
            .default_after_max_version = std::move(default_after_max_version)};
}

VersionedParamSpec type_since(const int32_t min_version) {
    return {.name = "type", .type = ModelConfigValueType::STRING, .min_version = min_version};
}

namespace model_config::specs {

const ParamSpecs required_typed_section{
        required("type", ModelConfigValueType::STRING),
        required("kwargs", ModelConfigValueType::TABLE),
};

const ParamSpecs optional_typed_section{
        required("type", ModelConfigValueType::STRING),
        optional_default("kwargs", ModelConfigValueType::TABLE, ""),
};

const ParamSpecs top_level{
        required("config_version", ModelConfigValueType::INTEGER),
        required_until("basecaller_model", ModelConfigValueType::STRING, 3, ""),
        required_since("supported_basecallers", ModelConfigValueType::ARRAY, 2, ""),
        required_since("chunk_size", ModelConfigValueType::INTEGER, 4, "10000"),
        required_since("chunk_overlap", ModelConfigValueType::INTEGER, 4, "1000"),
        required("model", ModelConfigValueType::TABLE),
        required("feature_encoder", ModelConfigValueType::TABLE),
        required("label_scheme", ModelConfigValueType::TABLE),
};

const ParamSpecs model_gru{
        required("num_features", ModelConfigValueType::INTEGER),
        required("num_classes", ModelConfigValueType::INTEGER),
        required("gru_size", ModelConfigValueType::INTEGER),
        required("n_layers", ModelConfigValueType::INTEGER),
        required("bidirectional", ModelConfigValueType::BOOLEAN),
};

const ParamSpecs model_latent_space_lstm{
        required("num_classes", ModelConfigValueType::INTEGER),
        required("lstm_size", ModelConfigValueType::INTEGER),
        required("cnn_size", ModelConfigValueType::INTEGER),
        required("pooler_type", ModelConfigValueType::STRING),
        required("bases_alphabet_size", ModelConfigValueType::INTEGER),
        required("bases_embedding_size", ModelConfigValueType::INTEGER),
        required("kernel_sizes", ModelConfigValueType::ARRAY),
        required("use_dwells", ModelConfigValueType::BOOLEAN),
        required("bidirectional", ModelConfigValueType::BOOLEAN),
        optional_default("pooler_args", ModelConfigValueType::TABLE, ""),
};

const ParamSpecs model_slot_attention_consensus{
        required("num_slots", ModelConfigValueType::INTEGER),
        required("classes_per_slot", ModelConfigValueType::INTEGER),
        required("read_embedding_size", ModelConfigValueType::INTEGER),
        required("cnn_size", ModelConfigValueType::INTEGER),
        required("kernel_sizes", ModelConfigValueType::ARRAY),
        required("pooler_type", ModelConfigValueType::STRING),
        required("use_mapqc", ModelConfigValueType::BOOLEAN),
        required("use_dwells", ModelConfigValueType::BOOLEAN),
        required("use_haplotags", ModelConfigValueType::BOOLEAN),
        optional_default("use_snp_qv", ModelConfigValueType::BOOLEAN, "false"),
        required("bases_alphabet_size", ModelConfigValueType::INTEGER),
        required("bases_embedding_size", ModelConfigValueType::INTEGER),
        required("add_lstm", ModelConfigValueType::BOOLEAN),
        required("use_reference", ModelConfigValueType::BOOLEAN),
        optional_default("pooler_args", ModelConfigValueType::TABLE, ""),
};

const ParamSpecs model_variant_perceiver{
        required("read_max_depth", ModelConfigValueType::INTEGER),
        required("ploidy", ModelConfigValueType::INTEGER),
        required("num_classes", ModelConfigValueType::INTEGER),
        required("cnn_size", ModelConfigValueType::INTEGER),
        required("kernel_sizes", ModelConfigValueType::ARRAY),
        required("dimension", ModelConfigValueType::INTEGER),
        required("num_blocks", ModelConfigValueType::INTEGER),
        required("num_heads", ModelConfigValueType::INTEGER),
        required("self_attn_layers_per_block", ModelConfigValueType::INTEGER),
        required("use_mapqc", ModelConfigValueType::BOOLEAN),
        required("use_dwells", ModelConfigValueType::BOOLEAN),
        required("use_haplotags", ModelConfigValueType::BOOLEAN),
        required("use_snp_qv", ModelConfigValueType::BOOLEAN),
        required("bases_alphabet_size", ModelConfigValueType::INTEGER),
        required("bases_embedding_size", ModelConfigValueType::INTEGER),
        required("use_decoder_lstm", ModelConfigValueType::BOOLEAN),
        required("use_per_read_embedding", ModelConfigValueType::BOOLEAN),
        required("embedding_type", ModelConfigValueType::STRING),
        required("update_read_embeddings", ModelConfigValueType::BOOLEAN),
        required("latent_init_method", ModelConfigValueType::STRING),
        required("shuffle_embeddings", ModelConfigValueType::BOOLEAN),
        required("mask_partial_rows", ModelConfigValueType::BOOLEAN),
        required("add_null_tokens", ModelConfigValueType::BOOLEAN),
};

const ParamSpecs feature_encoder_counts{
        required("normalise", ModelConfigValueType::STRING),
        required("tag_keep_missing", ModelConfigValueType::BOOLEAN),
        required("min_mapq", ModelConfigValueType::INTEGER),
        required("sym_indels", ModelConfigValueType::BOOLEAN),
        required("dtypes", ModelConfigValueType::ARRAY),
};

/**
 * \brief Kwargs for the feature encoder. The right_align_insertions was
 *          added inconsistently in version 3 of the config (some configs have it,
 *          some don't), so it is accepted before version 4 but only required from
 *          version 4 onward.
 */
const ParamSpecs feature_encoder_read_alignment{
        required("dtypes", ModelConfigValueType::ARRAY),
        required("tag_keep_missing", ModelConfigValueType::BOOLEAN),
        required("min_mapq", ModelConfigValueType::INTEGER),
        required("max_reads", ModelConfigValueType::INTEGER),
        required("row_per_read", ModelConfigValueType::BOOLEAN),
        required("include_dwells", ModelConfigValueType::BOOLEAN),
        required("include_haplotype", ModelConfigValueType::BOOLEAN),
        required_since("include_snp_qv", ModelConfigValueType::BOOLEAN, 4, "false"),
        required_since_parse_or_use_default("right_align_insertions",
                                            ModelConfigValueType::BOOLEAN,
                                            4,
                                            "false"),
        optional_default("region_split", ModelConfigValueType::INTEGER, ""),
};

/**
 * \brief The label_scheme kwargs are not actually used, but some config files contain
 *          values. To prevent the validation from failing, we list them here as optional.
 */
const ParamSpecs label_scheme_common{
        optional_default("ploidy", ModelConfigValueType::INTEGER, ""),
        optional_default("ordered", ModelConfigValueType::BOOLEAN, ""),
        optional_default("right_align_insertions", ModelConfigValueType::BOOLEAN, ""),
};

const SpecMap params{
        {"", top_level},
        {"model", required_typed_section},
        {"model::GRUModel", {type_since(1)}},
        {"model::LatentSpaceLSTM", {type_since(1)}},
        {"model::SlotAttentionConsensus", {type_since(3)}},
        {"model::VariantPerceiver", {type_since(4)}},
        {"model.kwargs::GRUModel", model_gru},
        {"model.kwargs::LatentSpaceLSTM", model_latent_space_lstm},
        {"model.kwargs::SlotAttentionConsensus", model_slot_attention_consensus},
        {"model.kwargs::VariantPerceiver", model_variant_perceiver},
        {"feature_encoder", required_typed_section},
        {"feature_encoder::CountsFeatureEncoder", {type_since(1)}},
        {"feature_encoder::ReadAlignmentFeatureEncoder", {type_since(1)}},
        {"feature_encoder.kwargs::CountsFeatureEncoder", feature_encoder_counts},
        {"feature_encoder.kwargs::ReadAlignmentFeatureEncoder", feature_encoder_read_alignment},
        {"label_scheme", optional_typed_section},
        {"label_scheme::HaploidLabelScheme", {type_since(1)}},
        {"label_scheme::DiploidLabelScheme", {type_since(3)}},
        {"label_scheme.kwargs::HaploidLabelScheme", label_scheme_common},
        {"label_scheme.kwargs::DiploidLabelScheme", label_scheme_common},
};

}  // namespace model_config::specs

std::string section_key(const std::string& section, const std::string& key) {
    return std::empty(section) ? key : section + "." + key;
}

std::string typed_section_key(const std::string& section, const std::string& type) {
    return section + "::" + type;
}

std::string context_name(const std::string& section) {
    return std::empty(section) ? "top-level" : section;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

const ParamSpecs* find_specs(const std::string& section) {
    const auto it = model_config::specs::params.find(section);
    return it == std::cend(model_config::specs::params) ? nullptr : &it->second;
}

const ParamSpecs& require_specs(const std::string& section) {
    if (const ParamSpecs* specs = find_specs(section)) {
        return *specs;
    }
    throw std::runtime_error("No model config validation spec for section '" + section + "'.");
}

const VersionedParamSpec* find_spec(const ParamSpecs& specs, const std::string& name) {
    for (const VersionedParamSpec& spec : specs) {
        if (spec.name == name) {
            return &spec;
        }
    }
    return nullptr;
}

bool has_type_specs(const std::string& section) {
    const std::string prefix = typed_section_key(section, "");
    for (const auto& [key, specs] : model_config::specs::params) {
        (void)specs;
        if (starts_with(key, prefix)) {
            return true;
        }
    }
    return false;
}

ModelConfigValueType parse_toml_value(const toml::value& value) {
    if (value.is_array()) {
        return ModelConfigValueType::ARRAY;
    }
    if (value.is_boolean()) {
        return ModelConfigValueType::BOOLEAN;
    }
    if (value.is_integer()) {
        return ModelConfigValueType::INTEGER;
    }
    if (value.is_string()) {
        return ModelConfigValueType::STRING;
    }
    if (value.is_table()) {
        return ModelConfigValueType::TABLE;
    }
    throw std::runtime_error{"Unsupported TOML value type."};
}

std::string type_name(const ModelConfigValueType type) {
    switch (type) {
    case ModelConfigValueType::ANY:
        return "value";
    case ModelConfigValueType::ARRAY:
        return "array";
    case ModelConfigValueType::BOOLEAN:
        return "boolean";
    case ModelConfigValueType::INTEGER:
        return "integer";
    case ModelConfigValueType::STRING:
        return "string";
    case ModelConfigValueType::TABLE:
        return "table";
    }
    return "value";
}

bool is_in_version_range(const VersionedParamSpec& spec, const int32_t version) {
    return (version >= spec.min_version) && (version <= spec.max_version);
}

void validate_table(const toml::value& table,
                    const ParamSpecs& specs,
                    const int32_t version,
                    const std::string& section) {
    if (!table.is_table()) {
        throw std::runtime_error("Model config section '" + context_name(section) +
                                 "' must be a TOML table.");
    }

    for (const auto& [key, value] : table.as_table()) {
        const VersionedParamSpec* spec = find_spec(specs, key);
        if (spec == nullptr) {
            throw std::runtime_error("Unexpected model config key '" + section_key(section, key) +
                                     "'.");
        }
        if ((spec->type != ModelConfigValueType::ANY) && (parse_toml_value(value) != spec->type)) {
            throw std::runtime_error("Model config key '" + section_key(section, key) +
                                     "' must be a TOML " + type_name(spec->type) + ".");
        }
    }

    for (const VersionedParamSpec& spec : specs) {
        std::unordered_map<std::string, std::string> value;
        if (table.contains(spec.name)) {
            value.emplace(spec.name, toml::format(toml::find(table, spec.name)));
        } else if ((spec.missing_value != ValueUnavailableHandling::THROW) &&
                   !is_in_version_range(spec, version)) {
            continue;
        }
        (void)get_versioned_value(value, spec, version, context_name(section));
    }
}

std::string validate_section_type(const toml::value& table,
                                  const std::string& section,
                                  const int32_t version) {
    if (!table.is_table() || !table.contains("type") || !has_type_specs(section)) {
        return {};
    }

    const std::string type = toml::find<std::string>(table, "type");
    const std::string type_spec_key = typed_section_key(section, type);
    const ParamSpecs* specs = find_specs(type_spec_key);
    if (specs == nullptr) {
        throw std::runtime_error("Unknown model config type '" + type + "' for section '" +
                                 context_name(section) + "'.");
    }

    if (const VersionedParamSpec* type_spec = find_spec(*specs, "type")) {
        const std::unordered_map<std::string, std::string> value{{"type", type}};
        (void)get_versioned_value(value, *type_spec, version, context_name(section));
    }

    return type;
}

void validate_config_table(const toml::value& table,
                           const std::string& section,
                           const std::string& inherited_type,
                           const int32_t version) {
    if (const ParamSpecs* specs = find_specs(section)) {
        validate_table(table, *specs, version, section);
    }

    if (!std::empty(inherited_type)) {
        if (const ParamSpecs* specs = find_specs(typed_section_key(section, inherited_type))) {
            validate_table(table, *specs, version, section);
        }
    }

    const std::string local_type = validate_section_type(table, section, version);
    const std::string child_type = std::empty(local_type) ? inherited_type : local_type;

    if (!table.is_table()) {
        return;
    }
    for (const auto& [key, value] : table.as_table()) {
        if (value.is_table()) {
            validate_config_table(value, section_key(section, key), child_type, version);
        }
    }
}

std::string strip_toml_string_quotes(const std::string& value) {
    if ((value.size() >= 2) && (value.front() == '"') && (value.back() == '"')) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

}  // namespace

std::string get_versioned_value(const std::unordered_map<std::string, std::string>& values,
                                const VersionedParamSpec& spec,
                                const int32_t version,
                                const std::string& context) {
    const std::string key = context + "." + spec.name;
    const auto it = values.find(spec.name);
    const bool present = it != std::cend(values);

    if (version < spec.min_version) {
        if (spec.below_min_version == ValueUnavailableHandling::PARSE_OR_USE_DEFAULT) {
            return present ? strip_toml_string_quotes(it->second) : spec.default_before_min_version;
        } else if (spec.below_min_version == ValueUnavailableHandling::USE_DEFAULT) {
            return spec.default_before_min_version;
        }
        throw std::runtime_error("Model config key '" + key +
                                 "' is not supported for config version " +
                                 std::to_string(version) + ". Minimum supported version is " +
                                 std::to_string(spec.min_version) + ".");
    }

    if (version > spec.max_version) {
        if (spec.above_max_version == ValueUnavailableHandling::PARSE_OR_USE_DEFAULT) {
            return present ? strip_toml_string_quotes(it->second) : spec.default_after_max_version;
        } else if (spec.above_max_version == ValueUnavailableHandling::USE_DEFAULT) {
            return spec.default_after_max_version;
        }
        throw std::runtime_error("Model config key '" + key +
                                 "' is not supported for config version " +
                                 std::to_string(version) + ". Maximum supported version is " +
                                 std::to_string(spec.max_version) + ".");
    }

    if (present) {
        return strip_toml_string_quotes(it->second);
    }

    if (spec.missing_value != ValueUnavailableHandling::THROW) {
        return spec.default_value;
    }

    throw std::runtime_error("Model config is missing required key '" + key + "'.");
}

std::string get_model_config_model_value(const ModelConfig& config, const std::string& key) {
    const std::string section = typed_section_key("model.kwargs", config.model_type);
    const ParamSpecs& specs = require_specs(section);
    const VersionedParamSpec* spec = find_spec(specs, key);
    if (spec == nullptr) {
        throw std::runtime_error("Unexpected model config key 'model.kwargs." + key +
                                 "' for model type '" + config.model_type + "'.");
    }
    return get_versioned_value(config.model_kwargs, *spec, config.version, "model.kwargs");
}

std::string get_model_config_feature_encoder_value(const ModelConfig& config,
                                                   const std::string& key) {
    const std::string section =
            typed_section_key("feature_encoder.kwargs", config.feature_encoder_type);
    const ParamSpecs& specs = require_specs(section);
    const VersionedParamSpec* spec = find_spec(specs, key);
    if (spec == nullptr) {
        throw std::runtime_error("Unexpected model config key 'feature_encoder.kwargs." + key +
                                 "' for feature encoder type '" + config.feature_encoder_type +
                                 "'.");
    }
    return get_versioned_value(config.feature_encoder_kwargs, *spec, config.version,
                               "feature_encoder.kwargs");
}

void validate_model_config_toml(const toml::value& config_toml) {
    if (!config_toml.is_table()) {
        throw std::runtime_error("Model config must be a TOML table.");
    }
    if (!config_toml.contains("config_version")) {
        throw std::runtime_error("Model config must contain 'config_version' attribute.");
    }

    const int32_t version = toml::find<int32_t>(config_toml, "config_version");
    validate_config_table(config_toml, "", "", version);
}

}  // namespace dorado::secondary
