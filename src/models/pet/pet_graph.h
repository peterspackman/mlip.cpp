#pragma once

#include "pet_types.h"

struct ggml_context;
struct ggml_tensor;

namespace mlipcpp::pet {

/**
 * Graph building context for PET model
 *
 * Following llama.cpp pattern for graph construction helpers.
 * This struct bundles all the context needed by helper functions.
 */
struct pet_graph_context {
  ggml_context *ctx;          // Compute context for allocating tensors
  const PETHypers &hypers;    // Model hyperparameters
  const BatchedInput &batch;  // Batched input data
  const Weights &weights;     // Model weights
  ComputePrecision precision; // Compute precision for matmuls
  bool use_flash_attention;   // Use flash attention (disabled for backward pass)
};

/**
 * Cast tensor to specified precision if needed
 *
 * Used to optionally cast weight matrices to lower precision (F16)
 * for faster GPU computation. Returns original tensor if already
 * at target precision or if precision is F32.
 *
 * @param ctx GGML context for tensor allocation
 * @param tensor Tensor to potentially cast
 * @param precision Target precision
 * @return Original tensor (if F32 or already correct type) or cast tensor
 */
ggml_tensor *maybe_cast(ggml_context *ctx, ggml_tensor *tensor,
                        ComputePrecision precision);

// ============================================================================
// Layer Building Helpers (Phase 1 Extraction)
// ============================================================================

/**
 * Apply layer normalization
 *
 * PyTorch: LayerNorm with affine parameters
 * GGML: norm + scale + bias
 *
 * @param gctx Graph context
 * @param weight Scale parameter [d_pet]
 * @param bias Bias parameter [d_pet]
 * @param input Input tensor (2D: [d_pet, N])
 * @param eps Epsilon for numerical stability
 * @return Normalized output (same shape as input)
 */
ggml_tensor *build_layer_norm(pet_graph_context &gctx, ggml_tensor *weight,
                              ggml_tensor *bias, ggml_tensor *input,
                              float eps = 1e-5f);

/**
 * Build edge embedding from vectors and distances
 *
 * Steps:
 * 1. Concatenate edge_vectors [3, max_neighbors, total_atoms]
 *    with edge_distances [1, max_neighbors, total_atoms]
 * 2. Flatten to 2D
 * 3. Apply linear layer: [d_pet, 4] @ [4, N] -> [d_pet, N]
 * 4. Reshape back to 3D
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index (0 or 1)
 * @return Edge embeddings [d_pet, max_neighbors, total_atoms]
 */
ggml_tensor *build_edge_embedding(pet_graph_context &gctx, int layer_idx);

/**
 * Build node embedding from species indices
 *
 * Simple lookup in embedding table using ggml_get_rows.
 *
 * @param gctx Graph context
 * @param embedding_table Embedding weights [d_pet, num_species+1]
 * @param species_indices Species for each atom [total_atoms]
 * @return Node embeddings [d_pet, total_atoms]
 */
ggml_tensor *build_node_embedding(pet_graph_context &gctx,
                                  ggml_tensor *embedding_table,
                                  ggml_tensor *species_indices);

/**
 * Build transformer block (single attention + MLP layer)
 *
 * Architecture:
 * - Multi-head attention with residual
 * - Layer norm
 * - MLP (Linear + SiLU + Linear) with residual
 * - Layer norm
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index
 * @param transformer_idx Transformer block index (0 or 1)
 * @param input Input tokens [d_pet, seq_len, total_atoms]
 * @param attn_mask Attention mask [seq_len, seq_len, total_atoms]
 * @param attn_out Output of attention computation
 * @return Transformer output [d_pet, seq_len, total_atoms]
 */
ggml_tensor *build_transformer_block(pet_graph_context &gctx, int layer_idx,
                                     int transformer_idx, ggml_tensor *input,
                                     ggml_tensor *attn_mask,
                                     ggml_tensor *&attn_out);

// ============================================================================
// Additional Layer Building Helpers (Phase 2 Extraction)
// ============================================================================

/**
 * Build attention mask for transformer layers
 *
 * Returns pre-computed attention mask from batch preparation.
 * Masks are computed during prepare_batch() when actual data is available.
 *
 * Mask semantics:
 * - Node token (position 0) can attend to all valid neighbors
 * - Edge token i can attend to node token and all valid neighbors
 * - Mask value = log(cutoff_factor) for valid positions, -inf for padding
 * - Masks out cross-system attention in batched scenarios
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index (0 or 1)
 * @return Attention mask [seq_len, seq_len, total_atoms]
 */
ggml_tensor *build_attention_mask(pet_graph_context &gctx, int layer_idx);

/**
 * Multi-head self-attention
 *
 * Standard transformer attention:
 * - QKV projection
 * - Split into heads
 * - Scaled dot-product attention
 * - Output projection
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index
 * @param transformer_idx Transformer block index
 * @param input Input tokens [d_pet, seq_len, total_atoms]
 * @param attn_mask Attention mask [seq_len, seq_len, total_atoms]
 * @return Attention output [d_pet, seq_len, total_atoms]
 */
ggml_tensor *build_multi_head_attention(pet_graph_context &gctx, int layer_idx,
                                        int transformer_idx, ggml_tensor *input,
                                        ggml_tensor *attn_mask);

/**
 * Multi-head self-attention using GGML flash attention
 *
 * Uses ggml_flash_attn_ext for fused, optimized attention computation.
 * Requires mask in F16 format with shape [seq_len, seq_len_padded, 1, n_atoms].
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index
 * @param transformer_idx Transformer block index
 * @param input Input tokens [d_pet, seq_len, total_atoms]
 * @param attn_mask Flash attention mask [seq_len, seq_len_padded, 1, total_atoms] (F16)
 * @return Attention output [d_pet, seq_len, total_atoms]
 */
ggml_tensor *build_multi_head_attention_flash(pet_graph_context &gctx,
                                               int layer_idx,
                                               int transformer_idx,
                                               ggml_tensor *input,
                                               ggml_tensor *attn_mask);

/**
 * Compute reversed message averaging for message passing between layers
 *
 * For each edge (i->j), averages the forward and reverse edge features:
 * new_message[i,j] = 0.5 * (initial[i,j] + edge_features[j,i])
 *
 * Handles missing reverse edges and cross-system boundaries.
 *
 * @param gctx Graph context
 * @param initial_messages Initial messages [d_pet, max_neighbors, total_atoms]
 * @param edge_features Edge features from GNN layer [d_pet, max_neighbors,
 * total_atoms]
 * @return Updated messages [d_pet, max_neighbors, total_atoms]
 */
ggml_tensor *build_reversed_message_avg(pet_graph_context &gctx,
                                        ggml_tensor *initial_messages,
                                        ggml_tensor *edge_features);

/**
 * Form tokens from edge and node embeddings
 *
 * Layer 0: concat [edge_embeds, input_messages] then compress
 * Layer 1+: concat [edge_embeds, neighbor_embeds, input_messages] then compress
 *
 * @param gctx Graph context
 * @param layer_idx GNN layer index
 * @param edge_embeds Edge embeddings [d_pet, max_neighbors, total_atoms]
 * @param input_messages Input messages [d_pet, max_neighbors, total_atoms]
 * @return Compressed tokens [d_pet, max_neighbors, total_atoms]
 */
ggml_tensor *build_tokens(pet_graph_context &gctx, int layer_idx,
                          ggml_tensor *edge_embeds,
                          ggml_tensor *input_messages);

/**
 * Apply output heads (node and edge) and aggregate per atom
 *
 * For each GNN layer:
 * - Apply node head: MLP -> energy contribution
 * - Apply edge head: MLP -> per-edge energy
 * - Weight edges by cutoff factors
 * - Sum edge contributions
 * - Combine node + edge contributions
 *
 * @param gctx Graph context
 * @param node_features_list Node features from all layers
 * @param edge_features_list Edge features from all layers
 * @return Atomic energies [1, total_atoms]
 */
ggml_tensor *build_aggregation_and_output(
    pet_graph_context &gctx,
    const std::vector<ggml_tensor *> &node_features_list,
    const std::vector<ggml_tensor *> &edge_features_list);

/**
 * Generic property output builder
 *
 * Applies a property head (node + edge MLPs) to features from a single GNN
 * layer. This is the building block for all property predictions (energy,
 * forces, stress).
 *
 * @param gctx Graph context
 * @param property_name Name of the property ("energy", "nc_forces", "nc_stress")
 * @param node_features Node features [d_pet, total_atoms]
 * @param edge_features Edge features [d_pet, max_neighbors, total_atoms]
 * @param layer_idx GNN layer index (0 or 1)
 * @param weight_edges_by_cutoff If true, weight edge contributions by cutoff
 * @return Property values [output_dim, total_atoms], or nullptr if not available
 */
ggml_tensor *build_property_output(pet_graph_context &gctx,
                                   const std::string &property_name,
                                   ggml_tensor *node_features,
                                   ggml_tensor *edge_features, int layer_idx,
                                   bool weight_edges_by_cutoff = true);

} // namespace mlipcpp::pet
