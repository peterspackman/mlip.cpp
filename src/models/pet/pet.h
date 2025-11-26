#pragma once

#include "core/backend.h"
#include "mlipcpp/model.h"
#include "mlipcpp/neighbor_list.h"
#include "pet_batch.h"
#include "pet_types.h"
#include <map>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_sched;

// Type aliases for backend types
typedef struct ggml_backend_buffer *ggml_backend_buffer_t;
typedef struct ggml_backend_sched *ggml_backend_sched_t;

namespace mlipcpp::pet {

// Re-export BackendPreference from core for convenience
using BackendPreference = mlipcpp::BackendPreference;

// PETHypers and Weights are now in pet_types.h

/**
 * Clean PET model implementation
 *
 * This is a from-scratch implementation following the clean design spec.
 * It uses fully batched tensor operations with no loops over atoms or edges
 * in the forward pass.
 *
 * Key Features:
 * - Batched inference on multiple systems simultaneously
 * - NEF (Node-Edge-Feature) format for efficient edge operations
 * - Pre-computed indices for gather operations
 * - Minimal graph depth
 * - Numerical equivalence with PyTorch reference
 *
 * Architecture:
 * 1. Data Preparation: Convert systems to batched NEF format
 * 2. Initial Embeddings: Embed neighbor species
 * 3. GNN Layers (x2): Edge/node embedding + transformer + message passing
 * 4. Aggregation: Head networks + sum to system energy
 */
class PETModel : public Model {
public:
  explicit PETModel(const PETHypers &hypers);
  ~PETModel() override;

  // Model interface implementation
  ModelResult predict(const AtomicSystem &system) override;
  ModelResult predict(const AtomicSystem &system, bool compute_forces) override;
  std::string model_type() const override { return "pet"; }
  float cutoff() const override { return hypers_.cutoff; }

  /**
   * Override the model's cutoff radius
   *
   * This updates both the hyperparameters and rebuilds the neighbor list
   * builder. Useful for testing or when the GGUF file has a different cutoff
   * than desired.
   *
   * @param new_cutoff New cutoff radius in Angstroms
   */
  void set_cutoff(float new_cutoff) {
    hypers_.cutoff = new_cutoff;
    neighbor_builder_ =
        NeighborListBuilder(NeighborListOptions{new_cutoff, true, false});
  }

  /**
   * Set backend preference for computation
   *
   * Must be called BEFORE load_from_gguf(). The backend is created during
   * model loading based on this preference.
   *
   * @param pref Backend preference (Auto, CPU, Metal, etc.)
   */
  void set_backend_preference(BackendPreference pref) {
    backend_preference_ = pref;
  }

  /**
   * Set a custom backend provider
   *
   * Allows sharing a backend across multiple models. Must be called BEFORE
   * load_from_gguf().
   *
   * @param provider Shared backend provider
   */
  void set_backend(std::shared_ptr<BackendProvider> provider) {
    backend_provider_ = std::move(provider);
  }

  /**
   * Get current backend preference
   */
  BackendPreference backend_preference() const { return backend_preference_; }

  /**
   * Get backend name (after model is loaded)
   */
  const std::string &backend_name() const {
    static const std::string empty;
    return backend_provider_ ? backend_provider_->name() : empty;
  }

  /**
   * Enable/disable op-level profiling
   * When enabled, prints timing for each operation type after compute
   */
  void set_profiling(bool enabled) { profiling_enabled_ = enabled; }
  bool profiling_enabled() const { return profiling_enabled_; }

  /**
   * Set compute precision for matrix operations
   *
   * F16 can provide 2-4x speedup on GPUs with native f16 support.
   * Default is F32 for maximum accuracy.
   *
   * @param prec Compute precision (F32 or F16)
   */
  void set_precision(ComputePrecision prec) { compute_precision_ = prec; }
  ComputePrecision precision() const { return compute_precision_; }

  /**
   * Batched prediction on multiple systems
   *
   * More efficient than calling predict() multiple times.
   *
   * @param systems Vector of atomic systems
   * @param compute_forces If true, compute forces via backpropagation
   * @return Vector of model results (same order as input)
   */
  std::vector<ModelResult>
  predict_batch(const std::vector<AtomicSystem> &systems,
                bool compute_forces = false);

  /**
   * Load weights from GGUF file
   *
   * Uses same weight format as original PET implementation for compatibility.
   *
   * @param path Path to GGUF file
   * @return true if successful, false otherwise
   */
  bool load_from_gguf(const std::string &path);

private:
  PETHypers hypers_;
  NeighborListBuilder neighbor_builder_;
  BackendPreference backend_preference_ = BackendPreference::Auto;
  ComputePrecision compute_precision_ = ComputePrecision::F32;
  bool profiling_enabled_ = false;

  // GGML contexts
  ggml_context *ctx_weights_ =
      nullptr; // Persistent weight storage (no_alloc=true)
  ggml_context *ctx_compute_ =
      nullptr; // Per-inference computation tensors (no_alloc=true)

  // Backend system - uses shared BackendProvider
  std::shared_ptr<BackendProvider> backend_provider_;
  ggml_backend_buffer_t weight_buffer_ = nullptr; // Backend buffer for weights
  ggml_backend_sched_t sched_ = nullptr;          // Scheduler for graph execution

  // Model weights
  Weights weights_;

  // Composition energies (atomic reference energies, indexed by atomic number)
  std::map<int, float> composition_energies_;

  // Species-to-index mapping: atomic_number -> species_index
  // Used to convert atomic numbers (Z=1, 6, 7, ...) to embedding indices (0, 5,
  // 6, ...)
  std::map<int, int> species_to_index_;

  /**
   * Build computation graph for batched forward pass
   *
   * Implements the 4-phase PET architecture:
   * - Phase 1: Data preparation (done in prepare_batch)
   * - Phase 2: Initial embeddings
   * - Phase 3: GNN layers
   * - Phase 4: Aggregation and output
   *
   * @param batch Batched input structure
   * @return System energies tensor [n_systems]
   */
  ggml_tensor *build_forward_graph(const BatchedInput &batch);

  // Phase implementations (see PET_CLEAN_DESIGN.md for details)

  /**
   * Phase 2: Embed neighbor species to initialize edge messages
   *
   * Uses ggml_get_rows to look up embeddings for all neighbors.
   *
   * @param batch Batched input
   * @return input_messages [d_pet, max_neighbors, total_atoms]
   */
  ggml_tensor *initial_embeddings(const BatchedInput &batch);

  /**
   * Phase 3: Apply single GNN layer
   *
   * Steps:
   * 1. Embed center atom species
   * 2. Embed edge features (vectors + distances)
   * 3. Form tokens (edge + neighbor + message)
   * 4. Apply transformer blocks (x2)
   * 5. Extract node and edge features
   * 6. Message passing (prepare for next layer)
   *
   * @param layer_idx Layer index (0 or 1)
   * @param batch Batched input
   * @param input_messages Messages from previous layer [d_pet, max_neighbors,
   * total_atoms]
   * @param node_features Output: node features [d_pet, total_atoms]
   * @param edge_features Output: edge features [d_pet, max_neighbors,
   * total_atoms]
   * @return Updated messages for next layer [d_pet, max_neighbors, total_atoms]
   */
  ggml_tensor *apply_gnn_layer(int layer_idx, const BatchedInput &batch,
                               ggml_tensor *input_messages,
                               ggml_tensor *&node_features,
                               ggml_tensor *&edge_features);

  /**
   * Extract forces from gradient tensor
   *
   * @param batch Batched input structure
   * @param batch_src Source batch with CPU-accessible data
   * @param gf Computation graph
   * @param results Output: results vector to populate with forces
   */
  void extract_forces(const BatchedInput &batch, const BatchedInput &batch_src,
                      ggml_cgraph *gf, std::vector<ModelResult> &results);

  /**
   * Extract stress tensor from gradient data
   *
   * @param batch Batched input structure
   * @param batch_src Source batch with CPU-accessible data
   * @param gf Computation graph
   * @param systems Input systems (for cell information)
   * @param results Output: results vector to populate with stress
   */
  void extract_stress(const BatchedInput &batch, const BatchedInput &batch_src,
                      ggml_cgraph *gf, const std::vector<AtomicSystem> &systems,
                      std::vector<ModelResult> &results);
};

} // namespace mlipcpp::pet
