#pragma once

#include "core/backend.h"
#include "graph_interpreter.h"
#include "mlipcpp/model.h"
#include "mlipcpp/neighbor_list.h"

#include <functional>
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
   * Run inference over a neighbor list built by the caller (e.g. a simulation
   * engine): a full list of the pairs within cutoff(), with edge vectors and
   * distances. With `selected`, only those atoms' energies count; the forces
   * and stress are still with respect to every atom (ghost atoms in a domain
   * decomposition, say).
   */
  ModelResult predict(const AtomicSystem &system, const NeighborList &neighbors,
                      bool compute_forces,
                      const std::vector<int32_t> *selected = nullptr);

  /// Atomic numbers the model knows.
  std::vector<int32_t> atomic_types() const;

  /// How far an atom's energy reaches: one cutoff per message-passing layer,
  /// and one more for the adaptive cutoff.
  float interaction_range() const;

  /// Threads for the CPU backend (after loading).
  void set_cpu_threads(int n);

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

  /**
   * Run inference and invoke `cb` after the graph compute completes, while
   * the compute buffer is still alive. The callback typically calls
   * interp_.capture_all_outputs(...) to read intermediate tensors. Forces
   * are not computed in this path (the proposed visualisation contract is
   * energy-only; see ACTIVATION_CAPTURE.md).
   */
  ModelResult predict_with_capture(
      const AtomicSystem &system,
      const std::function<void(GraphInterpreter &)> &cb);

private:
  GraphInterpreter interp_;

  // Model hyperparameters
  float cutoff_ = 4.5f;
  float cutoff_width_ = 0.2f;
  float energy_scale_ = 1.0f;
  std::string cutoff_function_ = "cosine";
  bool forces_mode_ = false;
  float num_neighbors_adaptive_ = 0.0f;
  int num_gnn_layers_ = 0;

  BackendPreference backend_preference_ = BackendPreference::Auto;

  // GGML contexts and backend
  ggml_context *ctx_weights_ = nullptr;
  std::shared_ptr<BackendProvider> backend_provider_;
  ggml_backend_buffer_t weight_buffer_ = nullptr;
  ggml_backend_t compute_backend_ = nullptr;

  // Species mapping (atomic number -> index)
  std::map<int, int> species_to_index_;

  // Composition energies (atomic reference energies)
  std::map<int, float> composition_energies_;

  // Neighbor list builder
  NeighborListBuilder neighbor_builder_;

  // Predict a single system (all logic lives here).
  // post_compute_hook (if set) runs after a successful graph compute and
  // before the compute buffer is freed — this is the window during which
  // intermediate tensor data is live on the backend buffer.
  // `neighbors` (if set) is used instead of building a list; `selected` (if
  // set) restricts the energy to those atoms.
  ModelResult predict_single(
      const AtomicSystem &system, bool compute_forces,
      const std::function<void()> &post_compute_hook = nullptr,
      const NeighborList *neighbors = nullptr,
      const std::vector<int32_t> *selected = nullptr);
};

} // namespace mlipcpp::runtime
