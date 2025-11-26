#pragma once

#include <ggml.h>

/**
 * Scaled dot product attention using basic GGML operations.
 *
 * Implements: softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
 *
 * Input tensors (GGML layout):
 *   Q: [head_dim, seq_len, n_batch, num_heads]
 *   K: [head_dim, seq_len, n_batch, num_heads]
 *   V: [head_dim, seq_len, n_batch, num_heads]
 *   mask: [seq_len, seq_len, n_batch, num_heads] or nullptr
 *
 * Output:
 *   [head_dim, seq_len, n_batch, num_heads]
 *
 * Parameters:
 *   ctx: GGML context for tensor operations
 *   Q, K, V: Query, Key, Value tensors
 *   mask: Optional attention mask (can be nullptr)
 *   n_batch: Batch size (number of independent sequences)
 *   num_heads: Number of attention heads
 *   seq_len: Sequence length
 *   head_dim: Dimension of each attention head
 *
 * Note: The last two dimensions should have batch varying slower than heads
 * for proper batching behavior (i.e., [batch0_head0, batch0_head1,
 * batch1_head0, ...])
 */
ggml_tensor *ggml_scaled_dot_product_attention(
    ggml_context *ctx, ggml_tensor *Q, ggml_tensor *K, ggml_tensor *V,
    ggml_tensor *mask, int n_batch, int num_heads, int seq_len, int head_dim);
