#pragma once

#include "model_latent_space_lstm.h"
#include "nn/RMSNorm.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/features/encoder_base.h"

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/rnn.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Rotary embedding implementation.
 *
 * NOTE: There is a very similar implementation in TxModules.cpp, but that one differs in the
 *       registered buffers and the way that it handles computing the cos/sin values
 *       (it precomputes them once in the constructor for the max seq length, then reuses them).
 */
class RotaryEmbeddingImpl : public torch::nn::Module {
public:
    RotaryEmbeddingImpl(int64_t dim,
                        float theta,
                        const int64_t max_seq_len,
                        const at::TensorOptions& options);

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor q, at::Tensor k);

private:
    int64_t m_dim{0};
    float m_theta{0};
    at::Tensor m_cos_freqs{nullptr};
    at::Tensor m_sin_freqs{nullptr};

    at::Tensor rotate_half(const at::Tensor& x) const;
};
TORCH_MODULE(RotaryEmbedding);

/**
 * \brief SwiGLU implementation.
 *          There is an existing, almost compatible SwiGLU implementation from the basecaller (dorado::nn::GatedMLP),
 *          but it generates tensors of incompatible shape for our use case (higher dimensionality). TODO.
 */
class SwiGLUImpl : public torch::nn::Module {
public:
    SwiGLUImpl(int32_t in_features, int32_t hidden_features, bool bias);

    at::Tensor forward(const at::Tensor& x);

private:
    torch::nn::Linear m_fc1{nullptr};
    torch::nn::Linear m_fc2{nullptr};
};
TORCH_MODULE(SwiGLU);

class MultiHeadCrossAttentionImpl : public torch::nn::Module {
public:
    MultiHeadCrossAttentionImpl(int64_t d_model,
                                int64_t q_max_depth,   // currently not used
                                int64_t kv_max_depth,  // currently not used
                                int64_t nhead,
                                bool embed_features,  // currently not used
                                // std::string& embedding_type,  // currently not used
                                bool qkv_bias,
                                bool out_bias,
                                const std::optional<int64_t>& rotary_dim,
                                const std::optional<int64_t>& attn_window);

    /**
     * \brief Update the `update_seq` tensor by attending to the `cross_attn_seqs` tensor.
     * \param x Tensor of shape (batch_size, num_positions, num_sequences_q, input_dim).
     * \param y Tensor of shape (batch_size, num_positions, num_sequences_kv, input_dim).
     * \param pos_mask Tensor of shape (batch_size, num_heads, num_positions * num_sequences_q, num_positions * num_sequences_kv).
     * \returns out Tensor of shape (batch_size, num_positions, num_sequences, output_dim).
     */
    at::Tensor forward(at::Tensor x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& pos_mask);

private:
    int64_t m_d_model{0};
    int64_t m_nhead{0};
    int64_t m_head_dim{0};
    bool m_embed_features{false};  // placeholder for possible future implementation
    // std::string m_embedding_type{std:nullptr};  // placeholder for possible future implementation
    std::optional<int64_t> m_rotary_dim{std::nullopt};
    std::optional<int64_t> m_attn_window{std::nullopt};

    torch::nn::Linear m_kv_proj{nullptr};
    torch::nn::Linear m_q_proj{nullptr};
    torch::nn::Linear m_out_proj{nullptr};
    torch::nn::Identity m_q_embedding{nullptr};  // placeholder for possible future implementation
    torch::nn::Identity m_k_embedding{nullptr};  // placeholder for possible future implementation
    RotaryEmbedding m_positional_embeddings{nullptr};

    /**
     * \brief Implements the following masking logic:
     *          abs((query_pos % T) - (key_pos % T)) <= self.attn_window
     *
     * TODO: Cache results of the mask to avoid recomputation.
     */
    at::Tensor local_attention_mask(const int64_t T,
                                    const int64_t num_q_seqs,
                                    const int64_t num_kv_seqs,
                                    const int64_t attn_window) const;

    at::Tensor attn_fn(const at::Tensor& q,
                       const at::Tensor& k,
                       const at::Tensor& v,
                       const std::optional<at::Tensor>& pos_mask) const;
};
TORCH_MODULE(MultiHeadCrossAttention);

class MultiSequenceCrossAttentionBlockImpl : public torch::nn::Module {
public:
    MultiSequenceCrossAttentionBlockImpl(int64_t d_model,
                                         int64_t q_max_depth,   // currently not used
                                         int64_t kv_max_depth,  // currently not used
                                         int64_t nhead,
                                         bool embed_features,  // currently not used
                                         // std::string embedding_type;  // currently not used
                                         bool qkv_bias,
                                         bool out_bias,
                                         const std::optional<int64_t>& rotary_dim,
                                         const std::optional<int64_t>& attn_window,
                                         int64_t dim_feedforward,
                                         const float deepnorm_alpha);

    at::Tensor forward(at::Tensor& x,
                       const at::Tensor& y,
                       const std::optional<at::Tensor>& pos_mask);

private:
    at::Tensor m_deepnorm_alpha{torch::empty({})};

    MultiHeadCrossAttention m_attention{nullptr};
    SwiGLU m_ff{nullptr};
    nn::RMSNorm m_norm1{nullptr};
    nn::RMSNorm m_norm2{nullptr};
};
TORCH_MODULE(MultiSequenceCrossAttentionBlock);

class SelfAttentionBlockImpl : public MultiSequenceCrossAttentionBlockImpl {
public:
    SelfAttentionBlockImpl(int64_t d_model,
                           int64_t max_depth,  // currently not used
                           int64_t nhead,
                           bool embed_features,  // currently not used
                           // std::string embedding_type;  // currently not used
                           bool qkv_bias,
                           bool out_bias,
                           const std::optional<int64_t>& rotary_dim,
                           const std::optional<int64_t>& attn_window,
                           int64_t dim_feedforward,
                           const float deepnorm_alpha);

    at::Tensor forward(at::Tensor& x);
};
TORCH_MODULE(SelfAttentionBlock);

class MessagePassingBlockImpl : public torch::nn::Module {
public:
    MessagePassingBlockImpl(int64_t dim,
                            int64_t read_max_dim,
                            int64_t num_heads,
                            int64_t self_attn_layers_per_block,
                            bool embed_features,
                            // std::string embedding_type,
                            bool update_read_embeddings,
                            bool cross_attend_read_embeddings,
                            const std::optional<int64_t>& attn_window);

    /**
     * \brief Forward function of the MessagePassingBlock module.
     * \param read_seqs Tensor of shape (batch_size, num_positions, num_sequences, dim).
     * \param hap_seqs Tensor of shape (batch_size, num_positions, num_sequences, dim).
     * \param mask Tensor of shape (batch_size, num_sequences (read), num_positions).
     * \return out Tensor of shape (batch_size, num_positions, num_sequences, dim).
     */
    std::pair<at::Tensor, at::Tensor> forward(at::Tensor read_seqs,
                                              at::Tensor hap_seqs,
                                              const at::Tensor& mask);

private:
    bool m_update_read_embeddings{false};
    bool m_cross_attend_read_embeddings{false};
    MultiSequenceCrossAttentionBlock m_reads_to_haplotypes{nullptr};
    torch::nn::ModuleList m_haplotype_self_attention{nullptr};
    MultiSequenceCrossAttentionBlock m_haplotypes_to_reads{nullptr};
};
TORCH_MODULE(MessagePassingBlock);

class ModelVariantPerceiver : public ModelTorchBase {
public:
    ModelVariantPerceiver(const MustConstructWithFactory& ctor_tag,
                          int32_t read_max_depth,
                          int32_t ploidy,
                          int32_t num_classes,
                          int32_t cnn_size,
                          const std::vector<int32_t>& kernel_sizes,
                          int32_t dimension,
                          int32_t num_blocks,
                          int32_t num_heads,
                          int32_t self_attn_layers_per_block,
                          bool use_mapqc,
                          bool use_dwells,
                          bool use_haplotags,
                          bool use_snp_qv,
                          int32_t bases_alphabet_size,
                          int32_t bases_embedding_size,
                          // bool time_steps,
                          bool use_decoder_lstm,
                          bool use_per_read_embedding,
                          // std::string& embedding_type,
                          bool update_read_embeddings,
                          // std::optional<int32_t> attn_window,
                          const FeatureColumnMap& feature_column_map);

    /**
     * \brief Forward pass.
     * \param x Read level feature matrix, shape
     *          (num_batch, num_positions, num_reads (padded), num_features).
     * \param ref_seq The integer encoded haploid reference.
     *                  Can be None if the model doesn't require it, else has shape
     *                  (num_batch, num_positions).
     * \return Logits for positionwise predictions (num_positions, num_slots, num_classes).
     */
    at::Tensor forward(at::Tensor x) override;

    double estimate_batch_memory(const std::vector<int64_t>& batch_tensor_shape) const override;

private:
    static constexpr int32_t MAX_HAPLOTAGS{16};

    int32_t m_ploidy{2};
    int32_t m_num_classes{5};
    int32_t m_cnn_size{128};
    std::vector<int32_t> m_kernel_sizes{1, 17};
    int32_t m_dimension{256};
    int32_t m_num_blocks{4};
    int32_t m_num_heads{8};
    bool m_use_mapqc{false};
    bool m_use_dwells{false};
    bool m_use_haplotags{false};
    bool m_use_snp_qv{false};
    int32_t m_bases_alphabet_size{6};
    int32_t m_bases_embedding_size{6};
    bool m_use_decoder_lstm{false};
    bool m_update_read_embeddings{false};
    FeatureColumnMap m_feature_column_map{};

    torch::nn::Embedding m_base_embedder{nullptr};
    torch::nn::Embedding m_haplotag_embedder{nullptr};
    torch::nn::Embedding m_strand_embedder{nullptr};
    ReadLevelConv m_read_level_conv{nullptr};
    torch::nn::Linear m_expansion_layer{nullptr};
    at::Tensor m_latent_init{nullptr};
    torch::nn::ModuleList m_blocks{nullptr};
    torch::nn::LSTM m_decoder_lstm{nullptr};
    torch::nn::Identity m_decoder_identity{nullptr};
    torch::nn::Linear m_output{nullptr};

    int32_t m_column_base{-1};
    int32_t m_column_qual{-1};
    int32_t m_column_strand{-1};
    int32_t m_column_mapq{-1};
    int32_t m_column_dwell{-1};
    int32_t m_column_haplotag{-1};
    int32_t m_column_snp_qv{-1};

    void validate_feature_tensor(const at::Tensor& x) const;

    /**
     * \brief Preprocessing of input tensor.
     * \param in_x Tensor of shape (batch_size, num_positions, num_sequences, num_features).
     * \return embedding tensor of shape (batch_size, num_positions, num_sequences, dim),
               mask tensor of shape (batch_size, num_sequences, num_positions).
     */
    std::pair<at::Tensor, const at::Tensor> create_embedded_features(const at::Tensor& in_x);

    at::Tensor forward_impl(const at::Tensor& x);
};

}  // namespace dorado::secondary
