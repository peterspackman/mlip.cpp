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
struct ggml_backend;
struct ggml_backend_buffer;

typedef struct ggml_backend *ggml_backend_t;
typedef struct ggml_backend_buffer *ggml_backend_buffer_t;

namespace mlipcpp::runtime {

/**
 * Model implementation using auto-exported computation graphs.
 *
 * Wraps GraphInterpreter to provide the standard Model interface,
 * enabling automatic PyTorch -> GGML model conversion.
 *
 * Supports dynamic system sizes: the graph is exported with symbolic
 * dimensions (n_atoms, max_neighbors) that are resolved at runtime.
 *
 * Usage:
 *   GraphModel model;
 *   model.load_from_gguf("model.gguf");
 *   ModelResult result = model.predict(system);
 *   ModelResult result_f = model.predict(system, true); // with forces
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
   * - Model hyperparameters (cutoff, species map, etc.)
   */
  bool load_from_gguf(const std::string &path);

  /**
   * Load graph from separate JSON file (for testing).
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

private:
  GraphInterpreter interp_;

  // Model hyperparameters
  float cutoff_ = 4.5f;
  float cutoff_width_ = 0.2f;
  float energy_scale_ = 1.0f;
  std::string cutoff_function_ = "cosine";
  bool forces_mode_ = false;
  float num_neighbors_adaptive_ = 0.0f;

  BackendPreference backend_preference_ = BackendPreference::Auto;

  // GGML contexts and backend
  ggml_context *ctx_weights_ = nullptr;
  std::shared_ptr<BackendProvider> backend_provider_;
  ggml_backend_buffer_t weight_buffer_ = nullptr;
  ggml_backend_t cpu_backend_ = nullptr;

  // Species mapping (atomic number -> index)
  std::map<int, int> species_to_index_;

  // Composition energies (atomic reference energies)
  std::map<int, float> composition_energies_;

  // Neighbor list builder
  NeighborListBuilder neighbor_builder_;

  // Predict a single system (all logic lives here)
  ModelResult predict_single(const AtomicSystem &system, bool compute_forces);
};

} // namespace mlipcpp::runtime
