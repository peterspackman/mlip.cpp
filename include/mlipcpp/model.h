#pragma once

#include "system.h"
#include <memory>
#include <string>
#include <vector>

namespace mlipcpp {

/// Result from model inference
struct ModelResult {
  float energy = 0.0f;
  std::vector<float> forces; // [n_atoms * 3], flattened
  std::vector<float> stress; // [6] in Voigt notation (xx, yy, zz, yz, xz, xy)

  bool has_forces = false;
  bool has_stress = false;
};

/// Base interface for all MLIP models
class Model {
public:
  virtual ~Model() = default;

  /// Run inference on a single system (energy only)
  virtual ModelResult predict(const AtomicSystem &system) = 0;

  /// Run inference with optional force computation
  virtual ModelResult predict(const AtomicSystem &system, bool compute_forces) {
    (void)compute_forces; // Default ignores this flag
    return predict(system);
  }

  /// Get model metadata
  virtual std::string model_type() const = 0;
  virtual float cutoff() const = 0;
};

/// Load a model from GGUF file
std::unique_ptr<Model> load_model(const std::string &path);

} // namespace mlipcpp
