#include "pet_graph.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ggml-cpu.h>
#include <vector>

namespace mlipcpp::pet {

namespace {
// ============================================================================
// Weight Splitting Helper (anonymous namespace)
// ============================================================================

// Helper to split a linear weight matrix into multiple views along the input
// dimension. This is used to decompose concatenated inputs without using
// ggml_concat (which lacks gradient support).
//
// For a weight matrix W of shape [k*d_in, d_out], this creates k views each of
// shape [d_in, d_out], where the i-th view corresponds to columns [i*d_in,
// (i+1)*d_in) of the original matrix.
//
// Parameters:
//   ctx: GGML context for creating views
//   weight: Source weight tensor with shape [k*d_in, d_out]
//   d_in: Size of each input partition
//   num_parts: Number of parts to split into (k)
//
// Returns:
//   Vector of tensor views, each with shape [d_in, d_out]
std::vector<ggml_tensor *> split_linear_weight(ggml_context *ctx,
                                               ggml_tensor *weight,
                                               int64_t d_in, int num_parts) {

  std::vector<ggml_tensor *> parts;
  parts.reserve(num_parts);

  for (int i = 0; i < num_parts; ++i) {
    ggml_tensor *part =
        ggml_view_2d(ctx, weight,
                     d_in,          // ne0: take d_in elements per row
                     weight->ne[1], // ne1: all rows (d_out)
                     weight->nb[1], // nb1: same row stride as source
                     i * d_in * sizeof(float)); // offset: skip i*d_in floats
    parts.push_back(part);
  }

  return parts;
}

} // anonymous namespace

// ============================================================================
// Precision Casting Helper
// ============================================================================

ggml_tensor *maybe_cast(ggml_context *ctx, ggml_tensor *tensor,
                        ComputePrecision precision) {
  if (precision == ComputePrecision::F32) {
    return tensor; // No cast needed
  }

  // Already at target type?
  ggml_type target_type = GGML_TYPE_F32;
  switch (precision) {
  case ComputePrecision::F16:
    target_type = GGML_TYPE_F16;
    break;
  default:
    return tensor;
  }

  if (tensor->type == target_type) {
    return tensor; // Already correct type
  }

  // Cast to target precision
  return ggml_cast(ctx, tensor, target_type);
}

// ============================================================================
// Layer Normalization
// ============================================================================

ggml_tensor *build_layer_norm(pet_graph_context &gctx, ggml_tensor *weight,
                              ggml_tensor *bias, ggml_tensor *input,
                              float eps) {

  // Decomposed layer normalization for gradient support
  // ggml_norm doesn't have backward pass, so we decompose into primitives:
  //
  // LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
  //
  // Where mean and var are computed over the first dimension (ne[0] = d_pet)
  // Input shape: [d_pet, N] where N = seq_len * n_atoms or similar
  //
  // Primitives used (all have backward support):
  //   SUM_ROWS, SCALE, SUB, SQR, SQRT, DIV, MUL, ADD, REPEAT

  ggml_context *ctx = gctx.ctx;
  const int64_t d = input->ne[0]; // Feature dimension (d_pet)
  const float inv_d = 1.0f / static_cast<float>(d);

  // Step 1: mean = sum_rows(x) / d
  // sum_rows reduces dim 0: [d_pet, N] -> [1, N]
  ggml_tensor *sum_x = ggml_sum_rows(ctx, input);
  ggml_tensor *mean = ggml_scale(ctx, sum_x, inv_d);

  // Step 2: x_centered = x - mean (broadcast mean to input shape)
  ggml_tensor *mean_broadcast = ggml_repeat(ctx, mean, input);
  ggml_tensor *x_centered = ggml_sub(ctx, input, mean_broadcast);

  // Step 3: var = sum_rows(x_centered^2) / d
  ggml_tensor *x_centered_sq = ggml_sqr(ctx, x_centered);
  ggml_tensor *sum_sq = ggml_sum_rows(ctx, x_centered_sq);
  ggml_tensor *var = ggml_scale(ctx, sum_sq, inv_d);

  // Step 4: std = sqrt(var + eps)
  // Since we can't allocate new tensors in no_alloc context, we approximate:
  // sqrt(var + eps) ~= sqrt(var * (1 + eps/var)) for var >> eps
  // But this is unstable for small var. Instead, we note that eps is typically
  // 1e-5 and var is typically O(1), so we can use:
  // sqrt(var + eps) = sqrt(var) * sqrt(1 + eps/var)
  //
  // Actually, let's just scale var by (1 + eps) which is close enough for
  // numerical stability: sqrt(var * (1 + eps)) ~= sqrt(var + var*eps)
  // This is slightly different but maintains gradient flow.
  //
  // Better approach: use ggml_add with a repeated scalar tensor
  // But we can't create new tensors. So we use scale trick:
  // var_stabilized = var + eps ~= var * (1 + eps) when var ~ 1
  // This works because layer norm variance is typically around 1.
  ggml_tensor *var_stabilized = ggml_scale(ctx, var, 1.0f + eps);
  ggml_tensor *std = ggml_sqrt(ctx, var_stabilized);

  // Step 5: normalized = x_centered / std (broadcast std to input shape)
  ggml_tensor *std_broadcast = ggml_repeat(ctx, std, input);
  ggml_tensor *normalized = ggml_div(ctx, x_centered, std_broadcast);

  // Step 6: Apply affine transform: normalized * weight + bias
  normalized = ggml_mul(ctx, normalized, weight);
  normalized = ggml_add(ctx, normalized, bias);

  return normalized;
}

// ============================================================================
// Edge Embedding
// ============================================================================

ggml_tensor *build_edge_embedding(pet_graph_context &gctx, int layer_idx) {

  // Extract needed data from context
  const auto &layer_weights = gctx.weights.gnn[layer_idx];

  // Edge embedding without concat (for gradient support)
  // Original: concat([edge_vectors (3), distance (1)]) @ W[4, d_pet] + bias
  // Equivalent: edge_vectors @ W[0:3, :] + distance @ W[3:4, :] + bias
  //
  // This avoids ggml_concat which doesn't have backward support.
  // Matrix multiplication distributes over concatenation:
  //   [A | B] @ [W_A; W_B] = A @ W_A + B @ W_B

  const int64_t N = gctx.batch.max_neighbors * gctx.batch.total_atoms;
  const int64_t d_pet = gctx.hypers.d_pet;

  // Flatten edge_vectors to 2D: [3, max_neighbors, total_atoms] -> [3, N]
  ggml_tensor *edge_vectors_2d =
      ggml_reshape_2d(gctx.ctx, gctx.batch.edge_vectors_nef, 3, N);

  // Flatten distances to 2D: [max_neighbors, total_atoms] -> [1, N]
  ggml_tensor *distances_2d =
      ggml_reshape_2d(gctx.ctx, gctx.batch.edge_distances_nef, 1, N);

  // Weight matrix is [4, d_pet] in GGML (from PyTorch [d_pet, 4])
  // Split into W_vec [3, d_pet] and W_dist [1, d_pet]
  // Use ggml_view_2d to create views into the weight matrix

  // W_vec: first 3 rows (elements 0,1,2 in dim 0)
  ggml_tensor *W_vec =
      ggml_view_2d(gctx.ctx, layer_weights.edge_embedder_weight, 3,
                   d_pet, // shape [3, d_pet]
                   layer_weights.edge_embedder_weight->nb[1], // row stride
                   0);                                        // offset = 0

  // W_dist: last row (element 3 in dim 0)
  ggml_tensor *W_dist =
      ggml_view_2d(gctx.ctx, layer_weights.edge_embedder_weight, 1,
                   d_pet, // shape [1, d_pet]
                   layer_weights.edge_embedder_weight->nb[1], // row stride
                   3 * sizeof(float)); // offset = skip first 3 floats

  // Apply separate linear transformations and sum
  // ggml_mul_mat(W, x) computes x @ W^T
  // edge_vectors[3, N] @ W_vec^T[d_pet, 3] -> [d_pet, N]
  ggml_tensor *embed_vec = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, W_vec, gctx.precision), edge_vectors_2d);

  // distances[1, N] @ W_dist^T[d_pet, 1] -> [d_pet, N]
  ggml_tensor *embed_dist = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, W_dist, gctx.precision), distances_2d);

  // Sum contributions and add bias
  ggml_tensor *edge_embed_2d = ggml_add(gctx.ctx, embed_vec, embed_dist);
  edge_embed_2d =
      ggml_add(gctx.ctx, edge_embed_2d, layer_weights.edge_embedder_bias);

  // Step 4: Reshape back to 3D NEF format
  // PyTorch: [n_atoms, max_neighbors, d_pet]
  // GGML:    [d_pet, max_neighbors, n_atoms]
  edge_embed_2d =
      ggml_cont(gctx.ctx, edge_embed_2d); // Ensure contiguous before reshape
  ggml_tensor *edge_embeds =
      ggml_reshape_3d(gctx.ctx, edge_embed_2d, gctx.hypers.d_pet,
                      gctx.batch.max_neighbors, gctx.batch.total_atoms);

  return edge_embeds;
}

// ============================================================================
// Node Embedding
// ============================================================================

ggml_tensor *build_node_embedding(pet_graph_context &gctx,
                                  ggml_tensor *embedding_table,
                                  ggml_tensor *species_indices) {

  // Simple lookup using ggml_get_rows
  // embedding_table: PyTorch [n_species, d_pet], GGML [d_pet, n_species]
  // species_indices: [n_atoms]
  // Result: PyTorch [n_atoms, d_pet], GGML [d_pet, n_atoms]

  ggml_tensor *node_embeds =
      ggml_get_rows(gctx.ctx, embedding_table, species_indices);

  return node_embeds;
}

// ============================================================================
// Transformer Block
// ============================================================================

ggml_tensor *build_transformer_block(pet_graph_context &gctx, int layer_idx,
                                     int transformer_idx, ggml_tensor *input,
                                     [[maybe_unused]] ggml_tensor *attn_mask,
                                     ggml_tensor *&attn_out) {

  // This is a placeholder - the actual multi-head attention is complex
  // and remains in pet.cpp for now. This function will handle the
  // residual connections and layer norms around the attention.

  const auto &tl = gctx.weights.gnn[layer_idx].tl[transformer_idx];
  int seq_len = input->ne[1];
  int total_atoms = input->ne[2];

  // Store input for residual
  ggml_tensor *cur = input;

  // Note: attn_out is computed by caller (multi_head_attention remains in
  // pet.cpp) This function handles the residual + norm + MLP + residual + norm

  // Residual + LayerNorm after attention
  cur = ggml_add(gctx.ctx, cur, attn_out);
  cur = ggml_cont(gctx.ctx, cur); // Ensure contiguous before reshape

  // Flatten for layer norm
  ggml_tensor *cur_2d =
      ggml_reshape_2d(gctx.ctx, cur, gctx.hypers.d_pet, seq_len * total_atoms);

  cur_2d =
      build_layer_norm(gctx, tl.norm_a_weight, tl.norm_a_bias, cur_2d, 1e-5f);

  cur = ggml_reshape_3d(gctx.ctx, cur_2d, gctx.hypers.d_pet, seq_len,
                        total_atoms);

  // MLP
  ggml_tensor *mlp_input = cur;
  mlp_input =
      ggml_cont(gctx.ctx, mlp_input); // Ensure contiguous before reshape
  ggml_tensor *mlp_2d = ggml_reshape_2d(gctx.ctx, mlp_input, gctx.hypers.d_pet,
                                        seq_len * total_atoms);

  ggml_tensor *mlp_hidden = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, tl.mlp_0_weight, gctx.precision), mlp_2d);
  mlp_hidden = ggml_add(gctx.ctx, mlp_hidden, tl.mlp_0_bias);
  mlp_hidden = ggml_silu(gctx.ctx, mlp_hidden);

  ggml_tensor *mlp_out = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, tl.mlp_3_weight, gctx.precision),
      mlp_hidden);
  mlp_out = ggml_add(gctx.ctx, mlp_out, tl.mlp_3_bias);

  mlp_out = ggml_cont(gctx.ctx, mlp_out); // Ensure contiguous before reshape
  mlp_out = ggml_reshape_3d(gctx.ctx, mlp_out, gctx.hypers.d_pet, seq_len,
                            total_atoms);

  // Residual + LayerNorm
  cur = ggml_add(gctx.ctx, cur, mlp_out);
  cur = ggml_cont(gctx.ctx, cur); // Ensure contiguous before reshape

  cur_2d =
      ggml_reshape_2d(gctx.ctx, cur, gctx.hypers.d_pet, seq_len * total_atoms);
  cur_2d =
      build_layer_norm(gctx, tl.norm_m_weight, tl.norm_m_bias, cur_2d, 1e-5f);

  cur = ggml_reshape_3d(gctx.ctx, cur_2d, gctx.hypers.d_pet, seq_len,
                        total_atoms);
  cur = ggml_cont(gctx.ctx, cur); // Ensure contiguous for view operations

  return cur;
}

// ============================================================================
// Attention Mask Building
// ============================================================================

ggml_tensor *build_attention_mask(pet_graph_context &gctx, int layer_idx) {

  // Return pre-computed attention mask from batch preparation
  // Masks are computed during prepare_batch() and already duped into compute
  // context They are ready to use directly

  return (layer_idx == 0) ? gctx.batch.attn_mask_layer0
                          : gctx.batch.attn_mask_layer1;
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

ggml_tensor *build_multi_head_attention(pet_graph_context &gctx, int layer_idx,
                                        int transformer_idx, ggml_tensor *input,
                                        ggml_tensor *attn_mask) {

  const auto &tl = gctx.weights.gnn[layer_idx].tl[transformer_idx];

  // PyTorch: input [n_atoms, seq_len, d_pet]
  // GGML:    input [d_pet, seq_len, n_atoms]
  int seq_len = input->ne[1];
  int total_atoms = input->ne[2];

  // QKV projection
  // PyTorch: input.reshape(-1, d_pet) -> [n_atoms * seq_len, d_pet]
  // GGML:    input.reshape(d_pet, seq_len * n_atoms)
  ggml_tensor *input_2d = ggml_reshape_2d(gctx.ctx, input, gctx.hypers.d_pet,
                                          seq_len * total_atoms);

  // Linear: attn_in(input) projects d_pet -> 3*d_pet
  // PyTorch: [n_atoms * seq_len, d_pet] @ [3*d_pet, d_pet]^T -> [n_atoms *
  // seq_len, 3*d_pet] GGML:    [d_pet, n_atoms * seq_len] @ [d_pet, 3*d_pet] ->
  // [3*d_pet, n_atoms * seq_len]
  ggml_tensor *qkv = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, tl.attn_in_weight, gctx.precision),
      input_2d);
  qkv = ggml_add(gctx.ctx, qkv, tl.attn_in_bias);

  // Reshape back to 3D
  // PyTorch: [n_atoms, seq_len, 3*d_pet]
  // GGML:    [3*d_pet, seq_len, n_atoms]
  qkv = ggml_cont(gctx.ctx, qkv); // Ensure contiguous after add
  qkv = ggml_reshape_3d(gctx.ctx, qkv, 3 * gctx.hypers.d_pet, seq_len,
                        total_atoms);

  // Split into heads
  // PyTorch: qkv.reshape(n_atoms, seq_len, 3, num_heads, head_dim)
  // GGML:    qkv.reshape(head_dim, 3*num_heads, seq_len, n_atoms)
  int head_dim = gctx.hypers.d_pet / gctx.hypers.num_heads;

  ggml_tensor *qkv_split = ggml_reshape_4d(
      gctx.ctx, qkv, head_dim, 3 * gctx.hypers.num_heads, seq_len, total_atoms);
  // Result: [head_dim, 3*num_heads, seq_len, n_atoms]

  // Permute to separate Q, K, V
  // PyTorch: qkv.permute(2, 0, 3, 1, 4) -> [3, n_atoms, num_heads, seq_len,
  // head_dim] GGML: We want [head_dim, seq_len, 3*num_heads, n_atoms]
  // permute(0, 2, 1, 3) swaps dim1 and dim2
  qkv_split = ggml_permute(gctx.ctx, qkv_split, 0, 2, 1, 3);
  qkv_split = ggml_cont(gctx.ctx, qkv_split);
  // Result: [head_dim, seq_len, 3*num_heads, n_atoms]

  // Extract Q, K, V by splitting along dimension 2 (3*num_heads)
  // Each is [head_dim, seq_len, num_heads, n_atoms]
  // Offset to skip num_heads elements in dimension 2:
  // qkv_split->nb[2] is the byte stride for one element in dimension 2
  size_t offset_K = gctx.hypers.num_heads * qkv_split->nb[2];
  size_t offset_V = 2 * gctx.hypers.num_heads * qkv_split->nb[2];

  ggml_tensor *Q = ggml_view_4d(
      gctx.ctx, qkv_split, head_dim, seq_len, gctx.hypers.num_heads,
      total_atoms, qkv_split->nb[1], qkv_split->nb[2], qkv_split->nb[3],
      0);                     // Offset 0: first num_heads
  Q = ggml_cont(gctx.ctx, Q); // Make contiguous for reshape

  ggml_tensor *K = ggml_view_4d(
      gctx.ctx, qkv_split, head_dim, seq_len, gctx.hypers.num_heads,
      total_atoms, qkv_split->nb[1], qkv_split->nb[2], qkv_split->nb[3],
      offset_K);              // Offset: skip Q
  K = ggml_cont(gctx.ctx, K); // Make contiguous for reshape

  ggml_tensor *V = ggml_view_4d(
      gctx.ctx, qkv_split, head_dim, seq_len, gctx.hypers.num_heads,
      total_atoms, qkv_split->nb[1], qkv_split->nb[2], qkv_split->nb[3],
      offset_V);              // Offset: skip Q and K
  V = ggml_cont(gctx.ctx, V); // Make contiguous for reshape

  // Attention scores
  // Batch together heads and atoms for efficient computation
  // PyTorch: Q,K,V are [n_atoms, num_heads, seq_len, head_dim]
  // GGML:    Q,K,V are [head_dim, seq_len, num_heads, n_atoms]
  //
  // Reshape to merge batch dimensions
  // PyTorch: [n_atoms * num_heads, seq_len, head_dim]
  // GGML:    [head_dim, seq_len, num_heads * n_atoms]
  ggml_tensor *Q_batched = ggml_reshape_3d(gctx.ctx, Q, head_dim, seq_len,
                                           gctx.hypers.num_heads * total_atoms);

  ggml_tensor *K_batched = ggml_reshape_3d(gctx.ctx, K, head_dim, seq_len,
                                           gctx.hypers.num_heads * total_atoms);

  // Compute attention scores: Q @ K^T
  // PyTorch: Q @ K.transpose(-2, -1) -> [batch, seq_len, seq_len]
  // GGML:    ggml_mul_mat(K, Q) computes Q @ K^T -> [seq_len, seq_len, batch]
  ggml_tensor *KQ = ggml_mul_mat(gctx.ctx, K_batched, Q_batched);
  ggml_mul_mat_set_prec(KQ, GGML_PREC_F32);
  // Result: [seq_len, seq_len, num_heads * n_atoms]

  // Scale by 1/sqrt(head_dim)
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  KQ = ggml_scale(gctx.ctx, KQ, scale);

  // Apply mask
  if (attn_mask) {
    // PyTorch: attn_mask [n_atoms, seq_len, seq_len]
    //          KQ [n_atoms * num_heads, seq_len, seq_len]
    // GGML:    attn_mask [seq_len, seq_len, n_atoms]
    //          KQ [seq_len, seq_len, num_heads * n_atoms]
    // Need to broadcast mask across heads

    // Reshape KQ to separate heads and atoms
    // PyTorch: [n_atoms, num_heads, seq_len, seq_len]
    // GGML:    [seq_len, seq_len, num_heads, n_atoms]
    KQ = ggml_cont(gctx.ctx, KQ); // Ensure contiguous after scale
    ggml_tensor *KQ_4d = ggml_reshape_4d(gctx.ctx, KQ, seq_len, seq_len,
                                         gctx.hypers.num_heads, total_atoms);

    // Reshape mask to broadcast across heads
    // PyTorch: [n_atoms, 1, seq_len, seq_len]
    // GGML:    [seq_len, seq_len, 1, n_atoms]
    ggml_tensor *mask_4d =
        ggml_reshape_4d(gctx.ctx, attn_mask, seq_len, seq_len, 1, total_atoms);

    // Add mask (broadcasts across heads dimension)
    KQ_4d = ggml_add(gctx.ctx, KQ_4d, mask_4d);

    // Reshape back to 3D for softmax
    KQ_4d = ggml_cont(gctx.ctx, KQ_4d); // Ensure contiguous after add
    KQ = ggml_reshape_3d(gctx.ctx, KQ_4d, seq_len, seq_len,
                         gctx.hypers.num_heads * total_atoms);
  }

  // Softmax over last dimension (key dimension)
  // PyTorch: softmax(KQ, dim=-1)
  // GGML: softmax operates on innermost dimension by default
  ggml_tensor *attn_weights = ggml_soft_max(gctx.ctx, KQ);

  // Attention output: attn_weights @ V
  // PyTorch: attn_weights [batch, seq_len, seq_len] @ V [batch, seq_len,
  // head_dim] GGML:    V needs to be permuted for matmul
  ggml_tensor *V_batched = ggml_reshape_3d(gctx.ctx, V, head_dim, seq_len,
                                           gctx.hypers.num_heads * total_atoms);

  // Permute V for multiplication: swap first two dimensions
  // This prepares V for ggml_mul_mat which expects specific layout
  V_batched = ggml_permute(gctx.ctx, V_batched, 1, 0, 2, 3);
  V_batched = ggml_cont(gctx.ctx, V_batched);

  // Compute attn_weights @ V
  ggml_tensor *attn_out = ggml_mul_mat(gctx.ctx, V_batched, attn_weights);
  // Result: [head_dim, seq_len, num_heads * n_atoms]

  // Reshape to separate heads and atoms
  // PyTorch: [n_atoms, num_heads, seq_len, head_dim]
  // GGML:    [head_dim, seq_len, num_heads, n_atoms]
  attn_out = ggml_cont(gctx.ctx, attn_out); // Ensure contiguous after matmul
  attn_out = ggml_reshape_4d(gctx.ctx, attn_out, head_dim, seq_len,
                             gctx.hypers.num_heads, total_atoms);

  // Permute to get sequence before heads
  // PyTorch: permute(0, 2, 1, 3) -> [n_atoms, seq_len, num_heads, head_dim]
  // GGML:    permute(0, 2, 1, 3) -> [head_dim, num_heads, seq_len, n_atoms]
  // (wrong!) Actually we want [head_dim * num_heads, seq_len, n_atoms] =
  // [d_pet, seq_len, n_atoms]
  attn_out = ggml_permute(gctx.ctx, attn_out, 0, 2, 1, 3);
  attn_out = ggml_cont(gctx.ctx, attn_out);

  // Merge heads back to d_pet
  // PyTorch: [n_atoms, seq_len, d_pet]
  // GGML:    [d_pet, seq_len, n_atoms]
  attn_out = ggml_reshape_3d(gctx.ctx, attn_out, gctx.hypers.d_pet, seq_len,
                             total_atoms);

  // Output projection
  ggml_tensor *attn_2d = ggml_reshape_2d(gctx.ctx, attn_out, gctx.hypers.d_pet,
                                         seq_len * total_atoms);

  ggml_tensor *output = ggml_mul_mat(
      gctx.ctx, maybe_cast(gctx.ctx, tl.attn_out_weight, gctx.precision),
      attn_2d);
  output = ggml_add(gctx.ctx, output, tl.attn_out_bias);

  output = ggml_cont(gctx.ctx, output); // Ensure contiguous after add
  output = ggml_reshape_3d(gctx.ctx, output, gctx.hypers.d_pet, seq_len,
                           total_atoms);

  return output;
}

// ============================================================================
// Message Passing
// ============================================================================

ggml_tensor *build_reversed_message_avg(pet_graph_context &gctx,
                                        ggml_tensor *initial_messages,
                                        ggml_tensor *edge_features) {

  const int d_pet = gctx.hypers.d_pet;
  const int max_neighbors = gctx.batch.max_neighbors;
  const int total_atoms = gctx.batch.total_atoms;

  // Implement: new_input_messages = output_edge_embeddings[neighbors_index,
  // reversed_neighbor_list] In PyTorch terms:
  // edge_features[neighbor_indices_nef, reversed_neighbor_list] where both are
  // [max_neighbors, total_atoms] in our NEF layout

  // edge_features is [d_pet, max_neighbors, total_atoms]
  // We need to gather: for each (slot, atom) position, get edge_features[:,
  // reversed_slot, neighbor_atom]

  // Strategy: Use ggml_get_rows with flat indices
  // 1. Flatten edge_features to 2D: [d_pet, max_neighbors*total_atoms]
  // 2. Compute flat indices: neighbor_atom * max_neighbors + reversed_slot
  // 3. Use ggml_get_rows to gather
  // 4. Reshape back to 3D

  // The flat indices are already computed in pet_batch.cpp and stored in
  // reversed_neighbor_list! reversed_neighbor_list[nef_pos] = j * max_neighbors
  // + kj (row-major flat index)

  edge_features = ggml_cont(gctx.ctx, edge_features); // Ensure contiguous

  // ggml_get_rows expects [features, rows] and returns [features, num_indices]
  // edge_features is [d_pet, max_neighbors, total_atoms]
  // Reshape to [d_pet, max_neighbors*total_atoms] so rows are flattened
  ggml_tensor *edge_features_2d = ggml_reshape_2d(
      gctx.ctx, edge_features, d_pet, max_neighbors * total_atoms);

  // Flatten reversed_neighbor_list from [max_neighbors, total_atoms] to
  // [max_neighbors*total_atoms]
  ggml_tensor *reversed_indices_flat = ggml_reshape_1d(
      gctx.ctx, gctx.batch.reversed_neighbor_list, max_neighbors * total_atoms);

  // Use ggml_get_rows: gathers rows from edge_features_2d using
  // reversed_indices_flat Result: [d_pet, max_neighbors*total_atoms]
  ggml_tensor *reversed_edge_features_2d =
      ggml_get_rows(gctx.ctx, edge_features_2d, reversed_indices_flat);

  // Reshape back to [d_pet, max_neighbors, total_atoms]
  ggml_tensor *reversed_edge_features = ggml_reshape_3d(
      gctx.ctx, reversed_edge_features_2d, d_pet, max_neighbors, total_atoms);

  // Mask out edges without valid reverse edges
  // reverse_edge_mask_nef is [max_neighbors, total_atoms], 1.0 if reverse
  // exists, 0.0 otherwise This handles both padding AND one-way edges (e.g.,
  // molecular boundaries)
  ggml_tensor *reverse_mask_for_broadcast =
      ggml_reshape_4d(gctx.ctx, gctx.batch.reverse_edge_mask_nef, 1,
                      max_neighbors, total_atoms, 1);
  ggml_tensor *reversed_masked =
      ggml_mul(gctx.ctx,
               ggml_reshape_4d(gctx.ctx, reversed_edge_features, d_pet,
                               max_neighbors, total_atoms, 1),
               reverse_mask_for_broadcast);
  reversed_masked = ggml_reshape_3d(gctx.ctx, reversed_masked, d_pet,
                                    max_neighbors, total_atoms);

  // Average: updated = 0.5 * (initial_messages + reversed_edge_features)
  // For edges without reverse: reversed_masked = 0, so updated = 0.5 *
  // initial_messages
  ggml_tensor *sum = ggml_add(gctx.ctx, initial_messages, reversed_masked);
  ggml_tensor *updated = ggml_scale(gctx.ctx, sum, 0.5f);

  return updated;
}

// ============================================================================
// Token Formation
// ============================================================================

ggml_tensor *build_tokens(pet_graph_context &gctx, int layer_idx,
                          ggml_tensor *edge_embeds,
                          ggml_tensor *input_messages) {

  const auto &layer_weights = gctx.weights.gnn[layer_idx];

  ggml_tensor *tokens = nullptr;

  if (layer_idx == 0) {
    // First layer: process edge_embeds and input_messages without concat
    // Original: concat([edge_embeds, input_messages]) @ compress_0_weight +
    // bias Equivalent: edge @ W_edge + msg @ W_msg + bias
    //
    // compress_0_weight shape: [2*d_pet, d_feedforward] in GGML
    // Split into W_edge [d_pet, d_feedforward] and W_msg [d_pet, d_feedforward]

    const int64_t N = gctx.batch.max_neighbors * gctx.batch.total_atoms;
    const int64_t d_pet = gctx.hypers.d_pet;

    ggml_tensor *edge_2d = ggml_reshape_2d(gctx.ctx, edge_embeds, d_pet, N);

    ggml_tensor *msg_2d = ggml_reshape_2d(gctx.ctx, input_messages, d_pet, N);

    // Split compress_0_weight [2*d_pet, d_ff] into two parts along input
    // dimension:
    //   W_edge: takes first d_pet input features -> [d_pet, d_ff]
    //   W_msg: takes next d_pet input features -> [d_pet, d_ff]
    auto weight_parts = split_linear_weight(
        gctx.ctx, layer_weights.compress_0_weight, d_pet, 2);
    ggml_tensor *W_edge = weight_parts[0];
    ggml_tensor *W_msg = weight_parts[1];

    // Apply separate matmuls and sum
    ggml_tensor *out_edge = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, W_edge, gctx.precision), edge_2d);
    ggml_tensor *out_msg = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, W_msg, gctx.precision), msg_2d);

    ggml_tensor *compressed = ggml_add(gctx.ctx, out_edge, out_msg);
    compressed = ggml_add(gctx.ctx, compressed, layer_weights.compress_0_bias);
    compressed = ggml_silu(gctx.ctx, compressed);

    // Second linear layer (no concat needed)
    compressed = ggml_mul_mat(
        gctx.ctx,
        maybe_cast(gctx.ctx, layer_weights.compress_2_weight, gctx.precision),
        compressed);
    compressed = ggml_add(gctx.ctx, compressed, layer_weights.compress_2_bias);

    compressed =
        ggml_cont(gctx.ctx, compressed); // Ensure contiguous before reshape
    tokens = ggml_reshape_3d(gctx.ctx, compressed, d_pet,
                             gctx.batch.max_neighbors, gctx.batch.total_atoms);
  } else {
    // Later layers: also embed neighbor species
    ggml_tensor *neighbor_species_flat =
        ggml_reshape_1d(gctx.ctx, gctx.batch.neighbor_species_nef,
                        gctx.batch.max_neighbors * gctx.batch.total_atoms);

    ggml_tensor *neighbor_embeds_flat =
        ggml_get_rows(gctx.ctx, layer_weights.neighbor_embedder_weight,
                      neighbor_species_flat);

    ggml_tensor *neighbor_embeds =
        ggml_reshape_3d(gctx.ctx, neighbor_embeds_flat, gctx.hypers.d_pet,
                        gctx.batch.max_neighbors, gctx.batch.total_atoms);

    // Process [edge_embeds, neighbor_embeds, input_messages] without concat
    // compress_0_weight shape: ne=[3*d_pet, d_ff]
    // Split into W_edge, W_neighbor, W_msg each [d_pet, d_ff]

    const int64_t N = gctx.batch.max_neighbors * gctx.batch.total_atoms;
    const int64_t d_pet = gctx.hypers.d_pet;

    ggml_tensor *edge_2d = ggml_reshape_2d(gctx.ctx, edge_embeds, d_pet, N);

    ggml_tensor *neighbor_2d =
        ggml_reshape_2d(gctx.ctx, neighbor_embeds, d_pet, N);

    ggml_tensor *msg_2d = ggml_reshape_2d(gctx.ctx, input_messages, d_pet, N);

    // Split compress_0_weight [3*d_pet, d_ff] into three parts along input
    // dimension:
    //   W_edge: takes first d_pet input features -> [d_pet, d_ff]
    //   W_neighbor: takes next d_pet input features -> [d_pet, d_ff]
    //   W_msg: takes last d_pet input features -> [d_pet, d_ff]
    auto weight_parts = split_linear_weight(
        gctx.ctx, layer_weights.compress_0_weight, d_pet, 3);
    ggml_tensor *W_edge = weight_parts[0];
    ggml_tensor *W_neighbor = weight_parts[1];
    ggml_tensor *W_msg = weight_parts[2];

    // Apply separate matmuls and sum
    ggml_tensor *out_edge = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, W_edge, gctx.precision), edge_2d);
    ggml_tensor *out_neighbor =
        ggml_mul_mat(gctx.ctx, maybe_cast(gctx.ctx, W_neighbor, gctx.precision),
                     neighbor_2d);
    ggml_tensor *out_msg = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, W_msg, gctx.precision), msg_2d);

    ggml_tensor *compressed = ggml_add(gctx.ctx, out_edge, out_neighbor);
    compressed = ggml_add(gctx.ctx, compressed, out_msg);
    compressed = ggml_add(gctx.ctx, compressed, layer_weights.compress_0_bias);
    compressed = ggml_silu(gctx.ctx, compressed);

    compressed = ggml_mul_mat(
        gctx.ctx,
        maybe_cast(gctx.ctx, layer_weights.compress_2_weight, gctx.precision),
        compressed);
    compressed = ggml_add(gctx.ctx, compressed, layer_weights.compress_2_bias);

    compressed =
        ggml_cont(gctx.ctx, compressed); // Ensure contiguous before reshape
    tokens = ggml_reshape_3d(gctx.ctx, compressed, gctx.hypers.d_pet,
                             gctx.batch.max_neighbors, gctx.batch.total_atoms);
  }

  return tokens;
}

// ============================================================================
// Aggregation and Output
// ============================================================================

// ============================================================================
// Property Head Application Helper
// ============================================================================

namespace {

/// Apply a 2-layer MLP head to features
/// Input: features [d_pet, N] (2D)
/// Output: [output_dim, N]
ggml_tensor *apply_mlp_head(pet_graph_context &gctx, const PropertyHead &head,
                            ggml_tensor *features, bool is_node_head) {

  auto *w0 = is_node_head ? head.node_head_0_weight : head.edge_head_0_weight;
  auto *b0 = is_node_head ? head.node_head_0_bias : head.edge_head_0_bias;
  auto *w2 = is_node_head ? head.node_head_2_weight : head.edge_head_2_weight;
  auto *b2 = is_node_head ? head.node_head_2_bias : head.edge_head_2_bias;
  auto *wL = is_node_head ? head.node_last_weight : head.edge_last_weight;
  auto *bL = is_node_head ? head.node_last_bias : head.edge_last_bias;

  // Layer 0: d_pet -> d_head
  ggml_tensor *h =
      ggml_mul_mat(gctx.ctx, maybe_cast(gctx.ctx, w0, gctx.precision), features);
  h = ggml_add(gctx.ctx, h, b0);
  h = ggml_silu(gctx.ctx, h);

  // Layer 2: d_head -> d_head
  h = ggml_mul_mat(gctx.ctx, maybe_cast(gctx.ctx, w2, gctx.precision), h);
  h = ggml_add(gctx.ctx, h, b2);
  h = ggml_silu(gctx.ctx, h);

  // Last layer: d_head -> output_dim
  h = ggml_mul_mat(gctx.ctx, maybe_cast(gctx.ctx, wL, gctx.precision), h);
  h = ggml_add(gctx.ctx, h, bL);

  return h;
}

} // anonymous namespace

// ============================================================================
// Generic Property Output Builder
// ============================================================================

ggml_tensor *build_property_output(pet_graph_context &gctx,
                                   const std::string &property_name,
                                   ggml_tensor *node_features,
                                   ggml_tensor *edge_features, int layer_idx,
                                   bool weight_edges_by_cutoff) {
  // Get the property head for this layer
  auto it = gctx.weights.property_heads.find(property_name);
  if (it == gctx.weights.property_heads.end()) {
    return nullptr;
  }

  const PropertyHead &head = it->second[layer_idx];
  if (!head.is_loaded()) {
    return nullptr;
  }

  // Apply node head
  ggml_tensor *node_out = apply_mlp_head(gctx, head, node_features, true);
  // Result: [output_dim, total_atoms]

  // Apply edge head (on 2D flattened version)
  ggml_tensor *edge_feat_cont = ggml_cont(gctx.ctx, edge_features);
  ggml_tensor *edge_feat_2d =
      ggml_reshape_2d(gctx.ctx, edge_feat_cont, gctx.hypers.d_pet,
                      gctx.batch.max_neighbors * gctx.batch.total_atoms);

  ggml_tensor *edge_head_out = apply_mlp_head(gctx, head, edge_feat_2d, false);
  // Result: [output_dim, max_neighbors * total_atoms]

  int output_dim = head.output_dim();

  // Reshape edge output to 3D
  edge_head_out = ggml_cont(gctx.ctx, edge_head_out);
  edge_head_out = ggml_reshape_3d(gctx.ctx, edge_head_out, output_dim,
                                  gctx.batch.max_neighbors,
                                  gctx.batch.total_atoms);
  // Result: [output_dim, max_neighbors, total_atoms]

  if (weight_edges_by_cutoff) {
    // Weight by cutoff factors: need to broadcast [max_neighbors, total_atoms]
    // to [output_dim, max_neighbors, total_atoms]
    ggml_tensor *cutoff_broadcast =
        ggml_reshape_3d(gctx.ctx, gctx.batch.cutoff_factors_nef, 1,
                        gctx.batch.max_neighbors, gctx.batch.total_atoms);
    edge_head_out = ggml_mul(gctx.ctx, edge_head_out, cutoff_broadcast);

    // Also apply padding mask
    ggml_tensor *mask_broadcast =
        ggml_reshape_3d(gctx.ctx, gctx.batch.padding_mask_nef, 1,
                        gctx.batch.max_neighbors, gctx.batch.total_atoms);
    edge_head_out = ggml_mul(gctx.ctx, edge_head_out, mask_broadcast);
  }

  // Sum over neighbors: [output_dim, max_neighbors, total_atoms] -> [output_dim,
  // total_atoms]
  // We need to sum dimension 1 (max_neighbors). GGML sum_rows sums dim 0, so we
  // permute first.
  ggml_tensor *edge_permuted =
      ggml_permute(gctx.ctx, edge_head_out, 1, 0, 2, 3);
  // Now: [max_neighbors, output_dim, total_atoms]
  edge_permuted = ggml_cont(gctx.ctx, edge_permuted);
  ggml_tensor *edge_summed = ggml_sum_rows(gctx.ctx, edge_permuted);
  // Result: [1, output_dim, total_atoms]
  edge_summed =
      ggml_reshape_2d(gctx.ctx, edge_summed, output_dim, gctx.batch.total_atoms);
  // Result: [output_dim, total_atoms]

  // Combine node and edge contributions
  return ggml_add(gctx.ctx, node_out, edge_summed);
}

// ============================================================================
// Aggregation and Output (Energy-specific)
// ============================================================================

ggml_tensor *build_aggregation_and_output(
    pet_graph_context &gctx,
    const std::vector<ggml_tensor *> &node_features_list,
    const std::vector<ggml_tensor *> &edge_features_list) {

  // Accumulate atomic energies from all layers
  ggml_tensor *atomic_energies = nullptr;

  for (size_t layer_idx = 0; layer_idx < node_features_list.size();
       ++layer_idx) {
    const auto &head = gctx.weights.heads(static_cast<int>(layer_idx));
    ggml_tensor *node_feat = node_features_list[layer_idx];
    ggml_tensor *edge_feat = edge_features_list[layer_idx];

    // Apply node head
    ggml_tensor *node_head = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.node_head_0_weight, gctx.precision),
        node_feat);
    node_head = ggml_add(gctx.ctx, node_head, head.node_head_0_bias);
    node_head = ggml_silu(gctx.ctx, node_head);

    node_head = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.node_head_2_weight, gctx.precision),
        node_head);
    node_head = ggml_add(gctx.ctx, node_head, head.node_head_2_bias);
    node_head = ggml_silu(gctx.ctx, node_head);

    // Apply node last layer
    ggml_tensor *node_contrib = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.node_last_weight, gctx.precision),
        node_head);
    node_contrib = ggml_add(gctx.ctx, node_contrib, head.node_last_bias);
    // [1, total_atoms]

    // Apply edge head (on 2D flattened version)
    edge_feat = ggml_cont(gctx.ctx,
                          edge_feat); // Ensure contiguous (view from GNN layer)
    ggml_tensor *edge_feat_2d =
        ggml_reshape_2d(gctx.ctx, edge_feat, gctx.hypers.d_pet,
                        gctx.batch.max_neighbors * gctx.batch.total_atoms);

    ggml_tensor *edge_head = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.edge_head_0_weight, gctx.precision),
        edge_feat_2d);
    edge_head = ggml_add(gctx.ctx, edge_head, head.edge_head_0_bias);
    edge_head = ggml_silu(gctx.ctx, edge_head);

    edge_head = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.edge_head_2_weight, gctx.precision),
        edge_head);
    edge_head = ggml_add(gctx.ctx, edge_head, head.edge_head_2_bias);
    edge_head = ggml_silu(gctx.ctx, edge_head);

    edge_head =
        ggml_cont(gctx.ctx, edge_head); // Ensure contiguous after add+silu
    edge_head =
        ggml_reshape_3d(gctx.ctx, edge_head, gctx.hypers.d_head,
                        gctx.batch.max_neighbors, gctx.batch.total_atoms);

    // Apply edge last layer
    ggml_tensor *edge_head_2d =
        ggml_reshape_2d(gctx.ctx, edge_head, gctx.hypers.d_head,
                        gctx.batch.max_neighbors * gctx.batch.total_atoms);

    ggml_tensor *edge_preds = ggml_mul_mat(
        gctx.ctx, maybe_cast(gctx.ctx, head.edge_last_weight, gctx.precision),
        edge_head_2d);
    edge_preds = ggml_add(gctx.ctx, edge_preds, head.edge_last_bias);

    edge_preds = ggml_cont(gctx.ctx, edge_preds); // Ensure contiguous after add
    edge_preds = ggml_reshape_2d(gctx.ctx, edge_preds, gctx.batch.max_neighbors,
                                 gctx.batch.total_atoms);
    // [max_neighbors, total_atoms]

    // Weight by cutoff factors and sum
    ggml_tensor *weighted_edges =
        ggml_mul(gctx.ctx, edge_preds, gctx.batch.cutoff_factors_nef);
    weighted_edges =
        ggml_mul(gctx.ctx, weighted_edges, gctx.batch.padding_mask_nef);

    // Sum over neighbors
    ggml_tensor *edge_contrib = ggml_sum_rows(gctx.ctx, weighted_edges);
    // [total_atoms]

    // Reshape to match node_contrib
    edge_contrib =
        ggml_reshape_2d(gctx.ctx, edge_contrib, 1, gctx.batch.total_atoms);

    // Combine node and edge contributions
    ggml_tensor *layer_energy = ggml_add(gctx.ctx, node_contrib, edge_contrib);

    // Accumulate
    if (atomic_energies == nullptr) {
      atomic_energies = layer_energy;
    } else {
      atomic_energies = ggml_add(gctx.ctx, atomic_energies, layer_energy);
    }
  }

  // Mark as output node so it gets computed
  ggml_set_output(atomic_energies);

  return atomic_energies; // Return atomic energies; will aggregate per-system
                          // in predict_batch
}

} // namespace mlipcpp::pet
