#pragma once

#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include <array>
#include <cstdint>
#include <ggml.h>
#include <map>
#include <string>
#include <vector>

/**
 * @file pet_types.h
 * @brief Common types for PET (Polarizable Embedding Transfer) model
 *
 * This header defines all shared data structures used throughout the PET
 * implementation. Separated from implementation files to avoid circular
 * dependencies and improve compilation times.
 */

namespace mlipcpp::pet {

/**
 * @brief Compute precision for matrix operations
 *
 * Controls the precision of weight matrices during computation.
 * F32 is the default and most accurate. F16 can provide 2-4x speedup
 * on GPUs with native f16 support (e.g., Apple Metal).
 *
 * Note: Activations are always computed in F32 for numerical stability.
 * Only weight matrices are cast to the specified precision.
 */
enum class ComputePrecision {
  F32, ///< Full precision (default)
  F16, ///< Half precision (faster on GPU, slight accuracy loss)
       // Future: BF16, Q8_0, Q4_0, etc.
};

/**
 * @brief PET model hyperparameters
 *
 * Configuration parameters that define the model architecture and behavior.
 * Default values are provided but should always be overridden by loading
 * from a GGUF file to ensure consistency with trained weights.
 *
 * @example
 * ```cpp
 * PETHypers hypers;
 * hypers.cutoff = 4.5f;
 * hypers.d_pet = 256;
 * hypers.num_heads = 8;
 * ```
 */
struct PETHypers {
  int d_pet = 256;              ///< Hidden dimension (feature size)
  int num_heads = 8;            ///< Number of attention heads
  int num_gnn_layers = 2;       ///< Number of GNN layers (typically 2)
  int d_feedforward = 512;      ///< Feedforward dimension in transformer MLP
  int num_attention_layers = 2; ///< Number of transformer blocks per GNN layer
  int d_head = 128;             ///< Output dimension of aggregation heads
  float cutoff = 4.5f;          ///< Cutoff radius in Angstroms
  float cutoff_width = 0.5f;    ///< Cutoff smoothing width in Angstroms
};

// ============================================================================
// Property Head System (unified, extensible)
// ============================================================================

/**
 * @brief Unified property head weights
 *
 * A single structure that handles all property types (energy, forces, stress,
 * etc.). Each property head consists of:
 * - Node head: 2-layer MLP applied to per-atom features
 * - Edge head: 2-layer MLP applied to per-edge features
 * Both output to an output_dim determined by the last layer weight shape.
 *
 * Output dimensions by property:
 * - Energy: 1 (scalar per atom)
 * - Forces: 3 (vector per atom)
 * - Stress: 9 (3x3 tensor per atom, then aggregated)
 */
struct PropertyHead {
  // Node head MLP: d_pet -> d_head -> d_head -> output_dim
  ggml_tensor *node_head_0_weight = nullptr; ///< [d_head, d_pet]
  ggml_tensor *node_head_0_bias = nullptr;   ///< [d_head]
  ggml_tensor *node_head_2_weight = nullptr; ///< [d_head, d_head]
  ggml_tensor *node_head_2_bias = nullptr;   ///< [d_head]
  ggml_tensor *node_last_weight = nullptr;   ///< [output_dim, d_head]
  ggml_tensor *node_last_bias = nullptr;     ///< [output_dim]

  // Edge head MLP: d_pet -> d_head -> d_head -> output_dim
  ggml_tensor *edge_head_0_weight = nullptr; ///< [d_head, d_pet]
  ggml_tensor *edge_head_0_bias = nullptr;   ///< [d_head]
  ggml_tensor *edge_head_2_weight = nullptr; ///< [d_head, d_head]
  ggml_tensor *edge_head_2_bias = nullptr;   ///< [d_head]
  ggml_tensor *edge_last_weight = nullptr;   ///< [output_dim, d_head]
  ggml_tensor *edge_last_bias = nullptr;     ///< [output_dim]

  /// Check if this head has been loaded
  bool is_loaded() const { return node_head_0_weight != nullptr; }

  /// Get output dimension from last layer weight shape
  int output_dim() const {
    return node_last_weight ? static_cast<int>(node_last_weight->ne[1]) : 0;
  }
};

/**
 * @brief Configuration for loading a property head from GGUF
 *
 * Defines the tensor naming patterns for a property type.
 * The "{}" placeholder is replaced with the layer index.
 */
struct PropertyConfig {
  std::string name;          ///< Property identifier (e.g., "energy")
  std::string node_prefix;   ///< Prefix for node head tensors
  std::string edge_prefix;   ///< Prefix for edge head tensors
  std::string node_last_fmt; ///< Format for node last layer (with {} for layer)
  std::string edge_last_fmt; ///< Format for edge last layer (with {} for layer)
};

/// Standard property configurations
namespace property_configs {
inline const PropertyConfig ENERGY = {
    "energy", "model.node_heads.energy", "model.edge_heads.energy",
    "model.node_last_layers.energy.{}.energy___0",
    "model.edge_last_layers.energy.{}.energy___0"};

inline const PropertyConfig NC_FORCES = {
    "nc_forces", "model.node_heads.nc_forces", "model.edge_heads.nc_forces",
    "model.node_last_layers.nc_forces.{}.nc_f",
    "model.edge_last_layers.nc_forces.{}.nc_f"};

inline const PropertyConfig NC_STRESS = {
    "nc_stress", "model.node_heads.nc_stress", "model.edge_heads.nc_stress",
    "model.node_last_layers.nc_stress.{}.nc_s",
    "model.edge_last_layers.nc_stress.{}.nc_s"};
} // namespace property_configs

// ============================================================================

/**
 * @brief Hierarchical weight structure for PET model
 *
 * Organizes all model weights in a nested structure that mirrors the
 * PyTorch architecture. All tensors are stored in backend buffers
 * (CPU or GPU) and accessed via GGML tensor pointers.
 *
 * Memory layout:
 * - All weights in single backend buffer (weight_buffer_)
 * - Tensors are views into this buffer (no separate allocations)
 * - Backend handles memory lifetime and transfers
 *
 * @note Tensor shapes are documented in GGML format: [ne[0], ne[1], ne[2],
 * ne[3]] which corresponds to PyTorch [batch, channel, height, width] in
 * reverse
 */
struct Weights {
  /**
   * @brief Species embedding table (shared across all layers)
   *
   * Maps species indices to d_pet dimensional embeddings.
   * Shape: [d_pet, num_species+1]
   * - num_species+1 includes padding index at position num_species
   */
  ggml_tensor *embedding = nullptr;

  /**
   * @brief Single GNN layer weights
   *
   * Each GNN layer contains:
   * - Edge embedder: MLP for edge feature embedding
   * - Node embedder: Lookup table for center atom species
   * - Neighbor embedder: Lookup table for neighbor species (layer 1+ only)
   * - Token compression: MLP to compress concatenated features
   * - Transformer blocks: 2 attention + MLP blocks
   */
  struct GNNLayer {
    // Edge embedding MLP: R^4 -> R^d_pet (vectors + distance -> features)
    ggml_tensor *edge_embedder_weight = nullptr; ///< [d_pet, 4]
    ggml_tensor *edge_embedder_bias = nullptr;   ///< [d_pet]

    // Node embedder: species -> features
    ggml_tensor *node_embedder_weight = nullptr; ///< [d_pet, num_species+1]

    // Neighbor embedder: species -> features (only in layer 1+)
    ggml_tensor *neighbor_embedder_weight = nullptr; ///< [d_pet, num_species+1]

    // Token compression MLP: R^(2*d_pet or 3*d_pet) -> R^d_pet
    ggml_tensor *compress_0_weight = nullptr; ///< [d_pet, 2*d_pet or 3*d_pet]
    ggml_tensor *compress_0_bias = nullptr;   ///< [d_pet]
    ggml_tensor *compress_2_weight = nullptr; ///< [d_pet, d_pet]
    ggml_tensor *compress_2_bias = nullptr;   ///< [d_pet]

    /**
     * @brief Single transformer block weights
     *
     * Standard transformer architecture with multi-head attention
     * and MLP, plus layer normalization.
     */
    struct TransformerLayer {
      // Multi-head attention (QKV combined projection)
      ggml_tensor *attn_in_weight = nullptr;  ///< [3*d_pet, d_pet]
      ggml_tensor *attn_in_bias = nullptr;    ///< [3*d_pet]
      ggml_tensor *attn_out_weight = nullptr; ///< [d_pet, d_pet]
      ggml_tensor *attn_out_bias = nullptr;   ///< [d_pet]

      // Layer normalization (after attention, after MLP)
      ggml_tensor *norm_a_weight = nullptr; ///< [d_pet]
      ggml_tensor *norm_a_bias = nullptr;   ///< [d_pet]
      ggml_tensor *norm_m_weight = nullptr; ///< [d_pet]
      ggml_tensor *norm_m_bias = nullptr;   ///< [d_pet]

      // MLP: d_pet -> d_feedforward -> d_pet
      ggml_tensor *mlp_0_weight = nullptr; ///< [d_feedforward, d_pet]
      ggml_tensor *mlp_0_bias = nullptr;   ///< [d_feedforward]
      ggml_tensor *mlp_3_weight = nullptr; ///< [d_pet, d_feedforward]
      ggml_tensor *mlp_3_bias = nullptr;   ///< [d_pet]
    } tl[2]; ///< 2 transformer blocks per GNN layer
  } gnn[2];  ///< 2 GNN layers

  /**
   * @brief Property heads organized by property name and layer
   *
   * Maps property name -> array of heads (one per GNN layer).
   * Common properties:
   * - "energy": scalar energy prediction (output_dim=1)
   * - "nc_forces": non-conservative forces (output_dim=3)
   * - "nc_stress": non-conservative stress (output_dim=9)
   *
   * Example access:
   *   weights.property_heads["energy"][0]  // Layer 0 energy head
   *   weights.property_heads["nc_forces"][1] // Layer 1 force head
   */
  std::map<std::string, std::array<PropertyHead, 2>> property_heads;

  /// Convenience accessor for energy heads (legacy compatibility)
  PropertyHead &heads(int layer) { return property_heads["energy"][layer]; }
  const PropertyHead &heads(int layer) const {
    static const PropertyHead empty;
    auto it = property_heads.find("energy");
    return it != property_heads.end() ? it->second[layer] : empty;
  }

  /// Check if a property is available
  bool has_property(const std::string &name) const {
    auto it = property_heads.find(name);
    return it != property_heads.end() && it->second[0].is_loaded();
  }

  /// Convenience flags for common properties
  bool has_force_heads() const { return has_property("nc_forces"); }
  bool has_stress_heads() const { return has_property("nc_stress"); }
};

/**
 * @brief Batched input structure for multi-system inference
 *
 * Converts multiple atomic systems into a single batched representation
 * using NEF (Node-Edge-Feature) format. All systems are concatenated
 * along the atom dimension, with edges padded to max_neighbors.
 *
 * Design principles:
 * - Global atom indexing: atoms numbered 0 to total_atoms-1 across all systems
 * - NEF format: edges organized as [max_neighbors, total_atoms] for efficient
 * gather
 * - Pre-computed indices: all gather operations use pre-built index tensors
 * - Pre-computed masks: attention masks built during batch preparation
 *
 * @example
 * For 2 systems with 3 and 2 atoms:
 * - total_atoms = 5
 * - system_indices = [0, 0, 0, 1, 1]
 * - atoms_per_system = [3, 2]
 * - system_atom_offsets = [0, 3]
 */
struct BatchedInput {
  // Batch metadata
  int total_atoms = 0;   ///< Total atoms across all systems
  int n_systems = 0;     ///< Number of systems in batch
  int max_neighbors = 0; ///< Maximum neighbors per atom (for padding)

  // Atom-level tensors (flat format)
  ggml_tensor *positions = nullptr; ///< [3, total_atoms] Cartesian positions
  ggml_tensor *species = nullptr;   ///< [total_atoms] (I32) species indices
  ggml_tensor *system_indices =
      nullptr; ///< [total_atoms] (I32) system membership

  // Edge-level data (compact format, no padding)
  int total_edges = 0;                   ///< Total number of edges
  ggml_tensor *edge_vectors = nullptr;   ///< [3, total_edges] Cartesian vectors
  ggml_tensor *edge_distances = nullptr; ///< [total_edges] distances
  ggml_tensor *cutoff_factors = nullptr; ///< [total_edges] smooth cutoff values
  ggml_tensor *edge_center_indices =
      nullptr; ///< [total_edges] (I32) center atom
  ggml_tensor *edge_neighbor_indices =
      nullptr; ///< [total_edges] (I32) neighbor atom

  // NEF (Node-Edge-Feature) format - padded structure
  ggml_tensor *edge_vectors_nef = nullptr; ///< [3, max_neighbors, total_atoms]
  ggml_tensor *neighbor_species_nef =
      nullptr; ///< [max_neighbors, total_atoms] (I32)
  ggml_tensor *neighbor_species_transposed =
      nullptr; ///< [total_atoms, max_neighbors] (I32) - pre-transposed for GPU
  ggml_tensor *edge_distances_nef = nullptr; ///< [max_neighbors, total_atoms]
  ggml_tensor *cutoff_factors_nef = nullptr; ///< [max_neighbors, total_atoms]
  ggml_tensor *padding_mask_nef =
      nullptr; ///< [max_neighbors, total_atoms] 1=real, 0=pad
  ggml_tensor *neighbor_indices_nef =
      nullptr; ///< [max_neighbors, total_atoms] (I32)

  // Reverse neighbor mapping for message passing
  ggml_tensor *reversed_neighbor_list =
      nullptr; ///< [max_neighbors, total_atoms] (I32)
  ggml_tensor *reverse_edge_mask_nef =
      nullptr; ///< [max_neighbors, total_atoms] mask

  // Pre-computed attention masks (one per GNN layer)
  ggml_tensor *attn_mask_layer0 = nullptr; ///< [seq_len, seq_len, total_atoms]
  ggml_tensor *attn_mask_layer1 = nullptr; ///< [seq_len, seq_len, total_atoms]

  // Scalar constant tensors for cutoff computation (used in gradient mode)
  ggml_tensor *scalar_offset = nullptr; ///< [1] offset for cutoff computation
  ggml_tensor *scalar_half = nullptr;   ///< [1] constant 0.5
  float offset_val = 0.0f;              ///< Value to set in scalar_offset
  float half_val = 0.5f;                ///< Value to set in scalar_half

  // CPU-side metadata for post-processing
  std::vector<int> atoms_per_system;    ///< Atoms in each system
  std::vector<int> system_atom_offsets; ///< Cumulative offsets
};

/**
 * @brief Loaded weights with composition data
 *
 * Complete package returned by weight loader, including both
 * neural network weights and atomic reference energies.
 */
struct LoadedWeights {
  Weights weights;                           ///< Neural network weights
  PETHypers hypers;                          ///< Model hyperparameters
  std::map<int, float> composition_energies; ///< Atomic reference energies
  std::map<int, int> species_to_index;       ///< Atomic number -> species index
};

} // namespace mlipcpp::pet
