#include "ggml_attention.h"
#include <cmath>

ggml_tensor *ggml_scaled_dot_product_attention(
    ggml_context *ctx, ggml_tensor *Q, ggml_tensor *K, ggml_tensor *V,
    ggml_tensor *mask, int n_batch, int num_heads, int seq_len, int head_dim) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Batch: merge n_batch and num_heads into single dimension
  // [head_dim, seq_len, n_batch * num_heads]
  ggml_tensor *Q_batched =
      ggml_reshape_3d(ctx, Q, head_dim, seq_len, n_batch * num_heads);
  ggml_tensor *K_batched =
      ggml_reshape_3d(ctx, K, head_dim, seq_len, n_batch * num_heads);

  // Compute Q @ K^T
  // ggml_mul_mat(K, Q) computes Q @ K^T
  // Result: [seq_len, seq_len, n_batch * num_heads]
  ggml_tensor *KQ = ggml_mul_mat(ctx, K_batched, Q_batched);

  // Scale by 1/sqrt(head_dim)
  KQ = ggml_scale(ctx, KQ, scale);

  // Apply mask if provided
  if (mask != nullptr) {
    // Mask is 3D [seq_len, seq_len, n_batch] and needs to be broadcast across
    // heads Reshape to 4D with singleton head dimension
    ggml_tensor *mask_4d =
        ggml_reshape_4d(ctx, mask, seq_len, seq_len, n_batch, 1);
    // Repeat across heads using ggml_repeat
    ggml_tensor *mask_shape = ggml_new_tensor_4d(ctx, mask->type, seq_len,
                                                 seq_len, n_batch, num_heads);
    mask_4d = ggml_repeat(ctx, mask_4d, mask_shape);
    // Reshape to match KQ: [seq_len, seq_len, n_batch * num_heads]
    mask = ggml_reshape_3d(ctx, mask_4d, seq_len, seq_len, n_batch * num_heads);
    KQ = ggml_add(ctx, KQ, mask);
  }

  // Softmax over keys (last dimension)
  ggml_tensor *attn_weights = ggml_soft_max(ctx, KQ);

  // Multiply by V
  ggml_tensor *V_batched =
      ggml_reshape_3d(ctx, V, head_dim, seq_len, n_batch * num_heads);

  // Permute V: swap first two dimensions for mul_mat
  // This is required by GGML's mul_mat semantics
  V_batched = ggml_permute(ctx, V_batched, 1, 0, 2, 3);
  V_batched = ggml_cont(ctx, V_batched);

  // Compute attn_weights @ V
  // Result: [head_dim, seq_len, n_batch * num_heads]
  ggml_tensor *attn_out = ggml_mul_mat(ctx, V_batched, attn_weights);

  // Unbatch: reshape back to [head_dim, seq_len, n_batch, num_heads]
  attn_out = ggml_cont(ctx, attn_out);
  attn_out =
      ggml_reshape_4d(ctx, attn_out, head_dim, seq_len, n_batch, num_heads);

  return attn_out;
}
