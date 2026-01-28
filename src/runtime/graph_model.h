#pragma once

#include "core/backend.h"
#include "graph_interpreter.h"
#include "mlipcpp/model.h"
#include "mlipcpp/neighbor_list.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend_sched;

typedef struct ggml_backend_buffer *ggml_backend_buffer_t;
typedef struct ggml_backend_sched *ggml_backend_sched_t;

// Forward declaration for batch input structure
namespace mlipcpp::pet {
struct BatchedInput;
}

namespace mlipcpp::runtime {

/**
 * Model implementation using auto-exported computation graphs.
 *
 * This class wraps GraphInterpreter to provide the standard Model interface,
 * enabling automatic PyTorch -> GGML model conversion without manual C++ code.
 *
 * Key features:
 * - Loads graph JSON and weights from a single GGUF file
 * - Uses NEF (Node-Edge-Feature) format for efficient batched operations
 * - Supports energy prediction (forces via backprop coming later)
 *
 * Usage:
 *   GraphModel model;
 *   model.load_from_gguf("model.gguf");
 *   ModelResult result = model.predict(system);
 */
class GraphModel : public Model {
public:
  GraphModel();
  ~GraphModel() override;

  // Model interface
  ModelResult predict(const AtomicSystem &system) override;
  ModelResult predict(const AtomicSystem &system, bool compute_forces) override;
  std::string model_type() const override { return "graph"; }
  float cutoff() const override { return cutoff_; }

  /**
   * Load model from GGUF file.
   *
   * The GGUF file must contain:
   * - Weights as tensors
   * - Graph JSON in metadata field "graph.json"
   * - Model hyperparameters (cutoff, etc.)
   *
   * @param path Path to GGUF file
   * @return true if successful
   */
  bool load_from_gguf(const std::string &path);

  /**
   * Load graph from separate JSON file (for testing).
   *
   * @param path Path to graph JSON file
   */
  void load_graph_file(const std::string &path);

  /**
   * Set a weight tensor manually (for testing).
   */
  void set_weight(const std::string &name, ggml_tensor *tensor);

  /**
   * Set backend preference.
   */
  void set_backend_preference(BackendPreference pref) {
    backend_preference_ = pref;
  }

  /**
   * Get the underlying graph interpreter for inspection.
   */
  const GraphInterpreter &interpreter() const { return interp_; }

  /**
   * Batched prediction on multiple systems.
   */
  std::vector<ModelResult>
  predict_batch(const std::vector<AtomicSystem> &systems,
                bool compute_forces = false);

  /**
   * Get the graph's expected input dimensions.
   * Returns (n_atoms, max_neighbors) or (-1, -1) if not set.
   */
  std::pair<int, int> expected_dimensions() const {
    return {expected_n_atoms_, expected_max_neighbors_};
  }

  /**
   * Set expected input dimensions (extracted from graph metadata).
   */
  void set_expected_dimensions(int n_atoms, int max_neighbors) {
    expected_n_atoms_ = n_atoms;
    expected_max_neighbors_ = max_neighbors;
  }

private:
  GraphInterpreter interp_;
  float cutoff_ = 4.5f;
  float cutoff_width_ = 0.5f;
  BackendPreference backend_preference_ = BackendPreference::Auto;

  // GGML contexts
  ggml_context *ctx_weights_ = nullptr;

  // Backend system
  std::shared_ptr<BackendProvider> backend_provider_;
  ggml_backend_buffer_t weight_buffer_ = nullptr;

  // Species mapping (atomic number -> index)
  std::map<int, int> species_to_index_;

  // Composition energies (atomic reference energies)
  std::map<int, float> composition_energies_;

  // Neighbor list builder
  NeighborListBuilder neighbor_builder_;

  // Expected graph dimensions (from export metadata)
  int expected_n_atoms_ = -1;
  int expected_max_neighbors_ = -1;

  // Whether graph uses direct inputs (species, neighbor_species, edge_vectors, edge_distances)
  // vs NEF format inputs
  bool uses_direct_inputs_ = false;

  // Input tensor mapping (graph input name -> BatchedInput field)
  struct InputMapping {
    std::string graph_name;
    std::string batch_field;
  };
  std::vector<InputMapping> input_mappings_;

  // Build input mappings from graph specification
  void build_input_mappings();

  // Detect expected dimensions from graph input shapes
  void detect_dimensions_from_graph();

  // Register BatchedInput tensors with the interpreter
  void register_batch_inputs(ggml_context *ctx,
                             const struct pet::BatchedInput &batch);

  // Prepare simple inputs for direct-format graphs (single system only)
  // Creates tensors in PyTorch format: species[n_atoms], edge_vectors[n_atoms, max_neighbors, 3], etc.
  void prepare_direct_inputs(ggml_context *ctx, const AtomicSystem &system,
                             const NeighborList &nlist);
};

} // namespace mlipcpp::runtime
