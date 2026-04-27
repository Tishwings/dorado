
#include "secondary/architectures/model_factory.h"

#include "model_gru.h"
#include "model_latent_space_lstm.h"
#include "model_slot_attention_consensus.h"
#include "model_torch_script.h"
#include "model_variant_perceiver.h"
#include "secondary/architectures/model_config_validation.h"
#include "secondary/features/encoder_base.h"
#include "secondary/features/encoder_factory.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>
#include <torch/autograd.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/script.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::secondary {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    } else if (type == "LatentSpaceLSTM") {
        return ModelType::LATENT_SPACE_LSTM;
    } else if (type == "SlotAttentionConsensus") {
        return ModelType::SLOT_ATTENTION_CONSENSUS;
    } else if (type == "VariantPerceiver") {
        return ModelType::VARIANT_PERCEIVER;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

namespace {

/**
 * \brief This function is a workaround around missing features in torchlib. There
 *          is currently no way to load only the state dict without the model either
 *          being traced or scripted.
 *          Issue: "PytorchStreamReader failed locating file constants.pkl: file not found"
 *          Source: https://github.com/pytorch/pytorch/issues/36577#issuecomment-1279666295
 */
void load_parameters(ModelTorchBase& model, const std::filesystem::path& in_pt) {
    const auto get_bytes = [](const std::filesystem::path& filename) {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                (std::istreambuf_iterator<char>()));
        return bytes;
    };

    const std::unordered_set<std::string> non_persistent_buffers =
            model.get_non_persistent_buffers();

    if (spdlog::default_logger()->should_log(spdlog::level::debug)) {
        const torch::OrderedDict<std::string, at::Tensor> model_params = model.named_parameters();
        for (const auto& w : model_params) {
            spdlog::debug("[model_params] w.key() = {}", w.key());
        }
        for (const auto& buffer : model.named_buffers()) {
            spdlog::debug("[model_params] Buffer key: {}, shape: {}, persistent: {}", buffer.key(),
                          (buffer.value().defined() ? utils::tensor_shape_as_string(buffer.value())
                                                    : "undefined"),
                          ((non_persistent_buffers.count(buffer.key())) ? "no" : "yes"));
        }
    }

    at::InferenceMode infer_guard;

    const std::vector<char> bytes = get_bytes(in_pt);

    try {
        const c10::Dict<c10::IValue, c10::IValue> weights =
                torch::jit::pickle_load(bytes).toGenericDict();

        if (spdlog::default_logger()->should_log(spdlog::level::debug)) {
            for (const auto& w : weights) {
                spdlog::debug("[loaded pt_param] w.key() = {}", w.key().toStringRef());
            }
        }

        auto params = model.named_parameters(true /*recurse*/);
        auto buffers = model.named_buffers(true /*recurse*/);

        // Create a set of model parameters.
        std::unordered_set<std::string> set_model_params;
        for (const auto& param : params) {
            set_model_params.emplace(param.key());
        }
        for (const auto& param : buffers) {
            // Skip non-persistent buffers.
            if (non_persistent_buffers.count(param.key())) {
                continue;
            }
            set_model_params.emplace(param.key());
        }

        // Create a set of loaded parameters.
        std::unordered_set<std::string> set_loaded_params;
        for (const auto& w : weights) {
            const std::string& name = w.key().toStringRef();
            set_loaded_params.emplace(name);
        }

        // Validate that all parameters exist.
        {
            std::vector<std::string> missing;
            for (const auto& name : set_model_params) {
                if (set_loaded_params.count(name) == 0) {
                    missing.emplace_back(name);
                }
            }
            if (!std::empty(missing)) {
                throw std::runtime_error(
                        "Cannot load weights into the model: model contains parameters which are "
                        "not present in the weights file. Missing parameters: " +
                        utils::print_container_as_string(missing, ", ", true));
            }
        }
        {
            std::vector<std::string> missing;
            for (const auto& name : set_loaded_params) {
                if (set_model_params.count(name) == 0) {
                    missing.emplace_back(name);
                }
            }
            if (!std::empty(missing)) {
                throw std::runtime_error(
                        "Cannot load weights into the model: weights file contains parameters "
                        "which are not present in the model. Missing parameters: " +
                        utils::print_container_as_string(missing, ", ", true));
            }
        }

        // Set the model weights.
        for (const auto& w : weights) {
            const std::string& name = w.key().toStringRef();
            const at::Tensor& value = w.value().toTensor();

            if (params.contains(name)) {
                params[name].copy_(value);

            } else if (buffers.contains(name)) {
                buffers[name].copy_(value);

            } else {
                throw std::runtime_error(
                        "Some loaded parameters cannot be found in the libtorch model! name = " +
                        name);
            }
        }

    } catch (const c10::Error& e) {
        throw std::runtime_error{std::string("Error: ") + e.what()};
    }
}

}  // namespace

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config,
                                              const ParameterLoadingStrategy param_strategy) {
    const auto get_value = [&config](const std::string& key) -> std::string {
        return get_model_config_model_value(config, key);
    };

    const auto get_bool_value = [&config](const std::string& key) -> bool {
        return get_model_config_model_value(config, key) == "true";
    };

    const ModelType model_type = parse_model_type(config.model_type);

    const FeatureColumnMap feature_column_map = feature_column_map_factory(config);

    std::shared_ptr<ModelTorchBase> model;

    if ((config.model_file != "model.pt") && (config.model_file != "weights.pt")) {
        throw std::runtime_error{"Unexpected weights/model file name! model_file = '" +
                                 config.model_file.string() +
                                 "', expected either 'model.pt' or 'weights.pt'."};
    }

    if (config.model_file == "model.pt") {
        // Load a TorchScript model. Parameters are not important here.
        spdlog::debug("Loading a TorchScript model.");

        if (param_strategy != ParameterLoadingStrategy::LOAD_WEIGHTS) {
            throw std::runtime_error(
                    "TorchScript model cannot be loaded without loading the model.pt file.");
        }

        model = ModelTorchScript::make<ModelTorchScript>(config.model_dir / config.model_file);
        model->set_normalise(false);

        return model;
    }

    if (model_type == ModelType::GRU) {
        spdlog::debug("Constructing a GRU model.");

        const int32_t num_features = std::stoi(get_value("num_features"));
        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t gru_size = std::stoi(get_value("gru_size"));
        const int32_t n_layers = std::stoi(get_value("n_layers"));
        const bool bidirectional = get_bool_value("bidirectional");

        model = ModelGRU::make<ModelGRU>(num_features, num_classes, gru_size, n_layers,
                                         bidirectional);

    } else if (model_type == ModelType::LATENT_SPACE_LSTM) {
        spdlog::debug("Constructing a LATENT_SPACE_LSTM model.");

        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t lstm_size = std::stoi(get_value("lstm_size"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::string pooler_type = get_value("pooler_type");
        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const bool use_dwells = get_bool_value("use_dwells");
        const bool bidirectional = get_bool_value("bidirectional");

        model = ModelLatentSpaceLSTM::make<ModelLatentSpaceLSTM>(
                num_classes, lstm_size, cnn_size, kernel_sizes, pooler_type, use_dwells,
                bases_alphabet_size, bases_embedding_size, bidirectional, feature_column_map);

    } else if (model_type == ModelType::SLOT_ATTENTION_CONSENSUS) {
        spdlog::debug("Constructing a SLOT_ATTENTION_CONSENSUS model.");

        const int32_t num_slots = std::stoi(get_value("num_slots"));
        const int32_t classes_per_slot = std::stoi(get_value("classes_per_slot"));
        const int32_t read_embedding_size = std::stoi(get_value("read_embedding_size"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const std::string pooler_type = get_value("pooler_type");
        const bool use_mapqc = get_bool_value("use_mapqc");
        const bool use_dwells = get_bool_value("use_dwells");
        const bool use_haplotags = get_bool_value("use_haplotags");
        const bool use_snp_qv = get_bool_value("use_snp_qv");
        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));
        const bool add_lstm = get_bool_value("add_lstm");
        const bool use_reference = get_bool_value("use_reference");

        const std::unordered_map<std::string, std::string> pooler_args;

        model = ModelSlotAttentionConsensus::make<ModelSlotAttentionConsensus>(
                num_slots, classes_per_slot, read_embedding_size, cnn_size, kernel_sizes,
                pooler_type, pooler_args, use_mapqc, use_dwells, use_haplotags, use_snp_qv,
                bases_alphabet_size, bases_embedding_size, add_lstm, use_reference,
                feature_column_map);

        // The SlotAttentionConsensus model normalizes internally because of phasing, so
        // deactivate normalization after phasing.
        model->set_normalise(false);

    } else if (model_type == ModelType::VARIANT_PERCEIVER) {
        spdlog::debug("Constructing a VARIANT_PERCEIVER model.");

        const int32_t read_max_depth = std::stoi(get_value("read_max_depth"));
        const int32_t ploidy = std::stoi(get_value("ploidy"));
        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const int32_t dimension = std::stoi(get_value("dimension"));
        const int32_t num_blocks = std::stoi(get_value("num_blocks"));
        const int32_t num_heads = std::stoi(get_value("num_heads"));
        const int32_t self_attn_layers_per_block =
                std::stoi(get_value("self_attn_layers_per_block"));

        const bool use_mapqc = get_bool_value("use_mapqc");
        const bool use_dwells = get_bool_value("use_dwells");
        const bool use_haplotags = get_bool_value("use_haplotags");
        const bool use_snp_qv = get_bool_value("use_snp_qv");

        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));

        const bool use_decoder_lstm = get_bool_value("use_decoder_lstm");
        const bool update_read_embeddings = get_bool_value("update_read_embeddings");
        const bool use_per_read_embedding = get_bool_value("use_per_read_embedding");
        const EmbeddingType embedding_type = parse_embedding_type(get_value("embedding_type"));
        const bool latent_ref_init = parse_latent_init_from_ref(get_value("latent_init_method"));

        model = ModelVariantPerceiver::make<ModelVariantPerceiver>(
                read_max_depth, ploidy, num_classes, cnn_size, kernel_sizes, dimension, num_blocks,
                num_heads, self_attn_layers_per_block, use_mapqc, use_dwells, use_haplotags,
                use_snp_qv, bases_alphabet_size, bases_embedding_size, use_decoder_lstm,
                use_per_read_embedding, embedding_type, update_read_embeddings, latent_ref_init,
                feature_column_map);

    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    // Set the weights of the internally constructed model. This is optional for testing purposes.
    if (param_strategy == ParameterLoadingStrategy::LOAD_WEIGHTS) {
        load_parameters(*model, config.model_dir / config.model_file);
    }

    return model;
}

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config) {
    return model_factory(config, ParameterLoadingStrategy::LOAD_WEIGHTS);
}

}  // namespace dorado::secondary
