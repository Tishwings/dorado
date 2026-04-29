#pragma once

#include "secondary/architectures/model_config.h"

#include <toml.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>

namespace dorado::secondary {

constexpr int32_t MODEL_CONFIG_MAX_VERSION = std::numeric_limits<int32_t>::max();

enum class ModelConfigValueType {
    ANY,
    ARRAY,
    BOOLEAN,
    INTEGER,
    STRING,
    TABLE,
};

enum class ValueUnavailableHandling {
    THROW,
    USE_DEFAULT,
    PARSE_OR_USE_DEFAULT,
};

struct VersionedParamSpec {
    std::string name{};
    ModelConfigValueType type{ModelConfigValueType::ANY};
    int32_t min_version{0};
    int32_t max_version{MODEL_CONFIG_MAX_VERSION};  // Inclusive.
    ValueUnavailableHandling missing_value{ValueUnavailableHandling::THROW};
    std::string default_value{};
    ValueUnavailableHandling below_min_version{ValueUnavailableHandling::THROW};
    std::string default_before_min_version{};
    ValueUnavailableHandling above_max_version{ValueUnavailableHandling::THROW};
    std::string default_after_max_version{};
};

std::string get_versioned_value(const std::unordered_map<std::string, std::string>& values,
                                const VersionedParamSpec& spec,
                                int32_t version,
                                const std::string& context);

std::string get_model_config_model_value(const ModelConfig& config, const std::string& key);

std::string get_model_config_feature_encoder_value(const ModelConfig& config,
                                                   const std::string& key);

void validate_model_config_toml(const toml::value& config_toml);

}  // namespace dorado::secondary
