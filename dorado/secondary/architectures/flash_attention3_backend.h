#pragma once

#include <ATen/ATen.h>

#include <optional>

namespace dorado::secondary {

/**
 * \brief Check whether FlashAttention3 can be used for the batched attention pathway.
 *          IMPORTANT: This tests against the currently bound device using at::cuda::getCurrentDeviceProperties()
 *                      to verify that the FlashAttention implementation supports it.
 * \param q Query tensor of shape (batch_size, query_tokens, num_heads, head_dim).
 * \param k Key tensor of shape (batch_size, key_tokens, num_heads, head_dim).
 * \param v Value tensor of shape (batch_size, key_tokens, num_heads, head_dim).
 * \returns True when the tensors satisfy the FlashAttention3 runtime requirements for the
 *          batched kernel in this build.
 */
bool flash_attention3_available_for(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);

/**
 * \brief Check whether FlashAttention3 can be used for the variable-length attention pathway.
 *          IMPORTANT: This tests against the currently bound device using at::cuda::getCurrentDeviceProperties()
 *                      to verify that the FlashAttention implementation supports it.
 * \param q Packed query tensor of shape (total_query_tokens, num_heads, head_dim).
 * \param k Packed key tensor of shape (total_key_tokens, num_heads, head_dim).
 * \param v Packed value tensor of shape (total_key_tokens, num_heads, head_dim).
 * \param cumsum_seqlens_q Prefix-sum sequence boundaries for \p q with shape (batch_size + 1).
 * \param cumsum_seqlens_k Prefix-sum sequence boundaries for \p k and \p v with shape
 *        (batch_size + 1).
 * \returns True when the tensors satisfy the FlashAttention3 runtime requirements for the
 *          variable-length kernel in this build.
 */
bool flash_attention3_varlen_available_for(const at::Tensor& q,
                                           const at::Tensor& k,
                                           const at::Tensor& v,
                                           const at::Tensor& cumsum_seqlens_q,
                                           const at::Tensor& cumsum_seqlens_k);

/**
 * \brief Run FlashAttention3 for either the batched or variable-length pathway.
 * \param q Query tensor in batched form `(batch_size, query_tokens, num_heads, head_dim)` or
 *        packed variable-length form `(total_query_tokens, num_heads, head_dim)`.
 * \param k Key tensor in batched form `(batch_size, key_tokens, num_heads, head_dim)` or packed
 *        variable-length form `(total_key_tokens, num_heads, head_dim)`.
 * \param v Value tensor matching the layout of \p k.
 * \param cumsum_seqlens_q Optional prefix-sum sequence boundaries for packed queries. Leave unset for
 *        batched attention.
 * \param cumsum_seqlens_k Optional prefix-sum sequence boundaries for packed keys and values. Leave
 *        unset for batched attention.
 * \returns The FlashAttention3 output tensor with the same leading layout as \p q.
 */
at::Tensor flash_attention3_forward(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& v,
                                    const std::optional<at::Tensor>& cumsum_seqlens_q,
                                    const std::optional<at::Tensor>& cumsum_seqlens_k);

}  // namespace dorado::secondary
