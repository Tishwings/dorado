#pragma once

#include <ATen/ATen.h>
#include <torch/nn/module.h>

#include <optional>

namespace dorado::secondary {

/**
 * \brief Attention helper which dispatches to FlashAttention3 when available and falls back to
 *        PyTorch SDPA otherwise.
 */
class FlashAttentionModuleImpl : public torch::nn::Module {
public:
    /**
     * \brief Construct a stateless attention helper module.
     */
    FlashAttentionModuleImpl() = default;

    /**
     * \brief Run attention on batched query, key, and value tensors.
     * \param q Query tensor of shape (batch_size, query_tokens, num_heads, head_dim).
     * \param k Key tensor of shape (batch_size, key_tokens, num_heads, head_dim).
     * \param v Value tensor of shape (batch_size, key_tokens, num_heads, head_dim).
     * \returns Attention output tensor of shape (batch_size, query_tokens, num_heads, head_dim).
     */
    at::Tensor forward_batched(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) const;

    /**
     * \brief Run attention on packed variable-length query, key, and value tensors.
     * \param q Packed query tensor of shape (total_query_tokens, num_heads, head_dim).
     * \param k Packed key tensor of shape (total_key_tokens, num_heads, head_dim).
     * \param v Packed value tensor of shape (total_key_tokens, num_heads, head_dim).
     * \param cumsum_seqlens_q Prefix-sum sequence boundaries for \p q with shape (batch_size + 1).
     * \param cumsum_seqlens_k Prefix-sum sequence boundaries for \p k and \p v with shape
     *        (batch_size + 1).
     * \returns Attention output tensor with the same packed layout and shape as \p q.
     */
    at::Tensor forward_varlen(const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const at::Tensor& cumsum_seqlens_q,
                              const at::Tensor& cumsum_seqlens_k) const;

private:
    /**
     * \brief Shared implementation for batched and variable-length attention.
     * \param q Query tensor in batched or packed variable-length layout.
     * \param k Key tensor in batched or packed variable-length layout.
     * \param v Value tensor matching the layout of \p k.
     * \param cumsum_seqlens_q Optional packed query sequence boundaries. Leave unset for batched
     *        attention.
     * \param cumsum_seqlens_k Optional packed key/value sequence boundaries. Leave unset for batched
     *        attention.
     * \returns Attention output with the same layout as \p q.
     */
    at::Tensor forward(const at::Tensor& q,
                       const at::Tensor& k,
                       const at::Tensor& v,
                       const std::optional<at::Tensor>& cumsum_seqlens_q,
                       const std::optional<at::Tensor>& cumsum_seqlens_k) const;
};
TORCH_MODULE(FlashAttentionModule);

}  // namespace dorado::secondary
