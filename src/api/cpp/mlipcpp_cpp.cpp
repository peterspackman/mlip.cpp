/**
 * @file mlipcpp_cpp.cpp
 * @brief C++ API implementation for mlipcpp
 */

#include "core/backend.h"
#include "core/gguf_loader.h"
#include "core/log.h"
#include "mlipcpp/mlipcpp.hpp"
#include "mlipcpp/model.h"
#include "mlipcpp/system.h"
#include "models/pet/pet.h"
#include "runtime/graph_model.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstring>
#include <mutex>

namespace mlipcpp {

const char *version() { return "0.1.2"; }

// Global backend provider - shared across all models
static std::shared_ptr<BackendProvider> g_backend_provider;
static std::mutex g_backend_mutex;
static BackendPreference g_backend_preference = BackendPreference::Auto;

// Get or create the global backend provider
static std::shared_ptr<BackendProvider> get_global_backend() {
  std::lock_guard<std::mutex> lock(g_backend_mutex);
  if (!g_backend_provider) {
    g_backend_provider = BackendProvider::create(g_backend_preference);
  }
  return g_backend_provider;
}

// Convert public Backend enum to internal BackendPreference
static BackendPreference to_internal(Backend b) {
  switch (b) {
  case Backend::CPU:
    return BackendPreference::CPU;
  case Backend::CUDA:
    return BackendPreference::CUDA;
  case Backend::HIP:
    return BackendPreference::HIP;
  case Backend::Metal:
    return BackendPreference::Metal;
  case Backend::Vulkan:
    return BackendPreference::Vulkan;
  case Backend::WebGPU:
    return BackendPreference::WebGPU;
  case Backend::SYCL:
    return BackendPreference::SYCL;
  case Backend::CANN:
    return BackendPreference::CANN;
  case Backend::Auto:
  default:
    return BackendPreference::Auto;
  }
}

// Set the global backend preference. Creation is deferred until the first
// Predictor is built, so this is nothrow even if the requested backend
// turns out to be unavailable — construction of the Predictor will surface
// a BackendUnavailableError from BackendProvider::create().
void set_backend(Backend backend) {
  std::lock_guard<std::mutex> lock(g_backend_mutex);
  auto pref = to_internal(backend);
  if (g_backend_preference != pref) {
    g_backend_provider.reset();
  }
  g_backend_preference = pref;
}

// Get the current backend name
const char *get_backend_name() {
  auto backend = get_global_backend();
  return backend->name().c_str();
}

bool backend_is_gpu() {
  try {
    auto backend = get_global_backend();
    return backend->is_gpu();
  } catch (...) {
    return false;
  }
}

bool is_backend_available(Backend backend) {
  return mlipcpp::is_backend_available(to_internal(backend));
}

// Suppress verbose logging
void suppress_logging() {
  log::suppress_ggml_logging();
  log::set_level(log::Level::Off);
}

struct Predictor::Impl {
  std::unique_ptr<Model> model; // Internal Model from model.h
  std::string model_type_str;

  Impl(const std::string &path, const ModelOptions &options) {
    // Load metadata to determine model type
    GGUFLoader loader(path);
    std::string arch = loader.get_string("general.architecture", "");

    if (arch == "pet") {
      auto pet_model = std::make_unique<pet::PETModel>(pet::PETHypers{});

      // Set backend BEFORE loading (uses global shared backend)
      auto backend_pref = to_internal(options.backend);
      if (backend_pref != BackendPreference::Auto) {
        // User specified a backend in options - update global preference
        set_backend(options.backend);
      }
      pet_model->set_backend(get_global_backend());

      // Now load the model
      if (!pet_model->load_from_gguf(path)) {
        throw std::runtime_error("Failed to load PET model from: " + path);
      }

      // Apply cutoff override if specified
      if (options.cutoff_override > 0.0f) {
        pet_model->set_cutoff(options.cutoff_override);
      }

      model_type_str = "PET";
      model = std::move(pet_model);
    } else if (arch == "pet-graph") {
      auto graph_model = std::make_unique<runtime::GraphModel>();
      // options.backend wins; when it's Auto we fall through to the
      // global preference so a prior mlipcpp::set_backend() call isn't
      // silently ignored.
      auto graph_pref = to_internal(options.backend);
      if (graph_pref == BackendPreference::Auto) {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        graph_pref = g_backend_preference;
      }
      graph_model->set_backend_preference(graph_pref);

      if (!graph_model->load_from_gguf(path)) {
        throw std::runtime_error("Failed to load graph model from: " + path);
      }

      model_type_str = "PET-Graph";
      model = std::move(graph_model);
    } else {
      throw std::runtime_error("Unsupported model architecture: " + arch);
    }
  }

  Result predict_impl(const AtomicSystem &system, bool compute_forces) {
    auto internal_result = model->predict(system, compute_forces);

    Result result;
    result.energy = internal_result.energy;
    if (internal_result.has_forces) {
      result.forces = std::move(internal_result.forces);
    }
    if (internal_result.has_stress) {
      result.stress = std::move(internal_result.stress);
    }
    return result;
  }

  /**
   * Symmetric Voigt strain → cell @ F.T, positions @ F.T (rows are lattice
   * vectors). Used by the FD stress path.
   */
  static AtomicSystem strain(const AtomicSystem &src, const float eta[6]) {
    AtomicSystem out = src;
    // Strain matrix from Voigt (engineering shear → factor of 1/2 off-diag)
    double e[3][3] = {
        {1.0 + eta[0], 0.5 * eta[5], 0.5 * eta[4]},
        {0.5 * eta[5], 1.0 + eta[1], 0.5 * eta[3]},
        {0.5 * eta[4], 0.5 * eta[3], 1.0 + eta[2]},
    };
    // Transform cell rows: each lattice vector a_i' = e * a_i
    if (auto *cell_ptr = src.cell(); cell_ptr) {
      Cell new_cell = *cell_ptr;
      for (int i = 0; i < 3; ++i) {
        double a[3] = {cell_ptr->matrix[i][0], cell_ptr->matrix[i][1],
                       cell_ptr->matrix[i][2]};
        for (int alpha = 0; alpha < 3; ++alpha) {
          new_cell.matrix[i][alpha] = static_cast<float>(
              e[alpha][0] * a[0] + e[alpha][1] * a[1] + e[alpha][2] * a[2]);
        }
      }
      // Recompute inverse via the public Cell ctor
      Cell rebuilt(new_cell.matrix, new_cell.periodic[0],
                   new_cell.periodic[1], new_cell.periodic[2]);
      out.set_cell(rebuilt);
    }
    // Transform positions: r' = e * r
    auto &pos = out.positions_mut();
    for (int i = 0; i < src.num_atoms(); ++i) {
      double r[3] = {pos[3 * i + 0], pos[3 * i + 1], pos[3 * i + 2]};
      for (int alpha = 0; alpha < 3; ++alpha) {
        pos[3 * i + alpha] = static_cast<float>(
            e[alpha][0] * r[0] + e[alpha][1] * r[1] + e[alpha][2] * r[2]);
      }
    }
    return out;
  }

  /**
   * Stress via 4th-order central difference in symmetric Voigt strain.
   * 12 forward passes; bypasses autograd entirely.
   *
   * σ_voigt[k] = [8(E(+δ) - E(-δ)) - (E(+2δ) - E(-2δ))] / (12 V δ)
   *
   * Voigt[k] convention: (xx, yy, zz, yz, xz, xy). Off-diagonals use
   * symmetric strain ε_αβ = ε_βα = η_k/2 (engineering shear) so the same
   * formula applies to all six components.
   */
  std::vector<float> compute_fd_stress(const AtomicSystem &system,
                                       float delta = 5e-3f) {
    const auto *cell = system.cell();
    if (!cell) {
      return {};
    }
    const auto &m = cell->matrix;
    double vol = static_cast<double>(m[0][0]) *
                     (static_cast<double>(m[1][1]) * m[2][2] -
                      static_cast<double>(m[1][2]) * m[2][1]) -
                 static_cast<double>(m[0][1]) *
                     (static_cast<double>(m[1][0]) * m[2][2] -
                      static_cast<double>(m[1][2]) * m[2][0]) +
                 static_cast<double>(m[0][2]) *
                     (static_cast<double>(m[1][0]) * m[2][1] -
                      static_cast<double>(m[1][1]) * m[2][0]);
    vol = std::abs(vol);
    if (vol < 1e-10) {
      return {};
    }

    auto E_at = [&](int k, double scale) {
      float eta[6] = {0, 0, 0, 0, 0, 0};
      eta[k] = static_cast<float>(delta * scale);
      auto strained = strain(system, eta);
      return static_cast<double>(model->predict(strained, /*forces=*/false).energy);
    };

    std::vector<float> stress(6, 0.0f);
    for (int k = 0; k < 6; ++k) {
      double ep1 = E_at(k, +1.0);
      double em1 = E_at(k, -1.0);
      double ep2 = E_at(k, +2.0);
      double em2 = E_at(k, -2.0);
      stress[k] = static_cast<float>(
          (8.0 * (ep1 - em1) - (ep2 - em2)) / (12.0 * vol * delta));
    }
    return stress;
  }

  Result predict_impl(const AtomicSystem &system, const PredictOptions &options) {
    // Use predict_batch for NC forces support
    auto *pet_model = dynamic_cast<pet::PETModel *>(model.get());
    if (pet_model) {
      const bool compute_grad =
          (options.compute_forces || options.compute_stress) &&
          !options.use_nc_forces;
      auto internal_results = pet_model->predict_batch(
          {system},
          compute_grad, // gradient-based outputs
          options.use_nc_forces // NC outputs from forward pass
      );
      auto &internal_result = internal_results[0];

      Result result;
      result.energy = internal_result.energy;
      if (options.compute_forces && internal_result.has_forces) {
        result.forces = std::move(internal_result.forces);
      }
      if (options.compute_stress && internal_result.has_stress) {
        result.stress = std::move(internal_result.stress);
      }
      if (options.compute_stress && options.fd_stress) {
        result.stress = compute_fd_stress(system);
      }
      return result;
    } else {
      // Fallback for non-PET models
      auto result = predict_impl(system, options.compute_forces || options.compute_stress);
      if (!options.compute_forces) {
        result.forces.clear();
      }
      if (!options.compute_stress) {
        result.stress.clear();
      }
      if (options.compute_stress && options.fd_stress) {
        result.stress = compute_fd_stress(system);
      }
      return result;
    }
  }
};

Predictor::Predictor(const std::string &path, const ModelOptions &options)
    : impl_(std::make_unique<Impl>(path, options)) {}

Predictor::~Predictor() = default;

Predictor::Predictor(Predictor &&other) noexcept = default;
Predictor &Predictor::operator=(Predictor &&other) noexcept = default;

float Predictor::cutoff() const { return impl_->model->cutoff(); }

std::string_view Predictor::model_type() const { return impl_->model_type_str; }

Result Predictor::predict(std::span<const float> positions,
                          std::span<const int32_t> atomic_numbers,
                          bool compute_forces) {
  if (positions.size() != atomic_numbers.size() * 3) {
    throw std::invalid_argument(
        "positions.size() must be 3 * atomic_numbers.size()");
  }

  AtomicSystem system(static_cast<int>(atomic_numbers.size()), positions.data(),
                      atomic_numbers.data(),
                      nullptr // non-periodic
  );

  return impl_->predict_impl(system, compute_forces);
}

Result Predictor::predict(std::span<const float> positions,
                          std::span<const int32_t> atomic_numbers,
                          std::span<const float, 9> cell,
                          std::array<bool, 3> pbc, bool compute_forces) {
  if (positions.size() != atomic_numbers.size() * 3) {
    throw std::invalid_argument(
        "positions.size() must be 3 * atomic_numbers.size()");
  }

  // Convert cell span to Cell object
  float lattice[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      lattice[i][j] = cell[i * 3 + j];
    }
  }
  Cell periodic_cell(lattice, pbc[0], pbc[1], pbc[2]);

  AtomicSystem system(static_cast<int>(atomic_numbers.size()), positions.data(),
                      atomic_numbers.data(), &periodic_cell);

  return impl_->predict_impl(system, compute_forces);
}

Result Predictor::predict(int32_t n_atoms, const float *positions,
                          const int32_t *atomic_numbers, const float *cell,
                          const bool *pbc, bool compute_forces) {
  if (!positions || !atomic_numbers) {
    throw std::invalid_argument(
        "positions and atomic_numbers must not be null");
  }

  if (cell) {
    // Periodic system
    float lattice[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lattice[i][j] = cell[i * 3 + j];
      }
    }
    bool pbc_flags[3] = {true, true, true};
    if (pbc) {
      pbc_flags[0] = pbc[0];
      pbc_flags[1] = pbc[1];
      pbc_flags[2] = pbc[2];
    }
    Cell periodic_cell(lattice, pbc_flags[0], pbc_flags[1], pbc_flags[2]);

    AtomicSystem system(n_atoms, positions, atomic_numbers, &periodic_cell);
    return impl_->predict_impl(system, compute_forces);
  } else {
    // Non-periodic system
    AtomicSystem system(n_atoms, positions, atomic_numbers, nullptr);
    return impl_->predict_impl(system, compute_forces);
  }
}

namespace {

// Map a ggml dtype to the canonical short name used in NodeActivation.dtype.
// f16 is reported as "f32" because we widen the data on capture.
const char *activation_dtype_name(ggml_type t) {
  switch (t) {
  case GGML_TYPE_F32: return "f32";
  case GGML_TYPE_F16: return "f32";  // converted on read
  case GGML_TYPE_I32: return "i32";
  case GGML_TYPE_I8:  return "i8";   // bool/byte
  default:            return "unsupported";
  }
}

// Read a tensor's data off the active backend and emit a NodeActivation.
// Returns std::nullopt only on truly empty tensors. Unsupported dtypes
// produce a NodeActivation with empty `data` and dtype "unsupported" — the
// shape and name are still useful for visualisation.
NodeActivation tensor_to_activation(int id, const std::string &name,
                                    ggml_tensor *t) {
  NodeActivation a;
  a.node_id = id;
  a.name = name;
  a.dtype = activation_dtype_name(t->type);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t->ne[i] <= 0) break;
    a.shape.push_back(t->ne[i]);
  }
  const size_t n = ggml_nelements(t);
  if (n == 0) return a;

  switch (t->type) {
  case GGML_TYPE_F32: {
    a.data.resize(n * sizeof(float));
    ggml_backend_tensor_get(t, a.data.data(), 0, n * sizeof(float));
    break;
  }
  case GGML_TYPE_F16: {
    // Read as f16 then widen to f32 in the output buffer.
    std::vector<ggml_fp16_t> half(n);
    ggml_backend_tensor_get(t, half.data(), 0, n * sizeof(ggml_fp16_t));
    a.data.resize(n * sizeof(float));
    auto *out = reinterpret_cast<float *>(a.data.data());
    for (size_t i = 0; i < n; ++i) {
      out[i] = ggml_fp16_to_fp32(half[i]);
    }
    break;
  }
  case GGML_TYPE_I32: {
    a.data.resize(n * sizeof(int32_t));
    ggml_backend_tensor_get(t, a.data.data(), 0, n * sizeof(int32_t));
    break;
  }
  case GGML_TYPE_I8: {
    a.data.resize(n * sizeof(int8_t));
    ggml_backend_tensor_get(t, a.data.data(), 0, n * sizeof(int8_t));
    break;
  }
  default:
    break;  // leave data empty for unsupported types
  }
  return a;
}

}  // namespace

Result Predictor::predict(int32_t n_atoms, const float *positions,
                          const int32_t *atomic_numbers, const float *cell,
                          const bool *pbc, const PredictOptions &options) {
  if (!positions || !atomic_numbers) {
    throw std::invalid_argument(
        "positions and atomic_numbers must not be null");
  }

  if (cell) {
    // Periodic system
    float lattice[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lattice[i][j] = cell[i * 3 + j];
      }
    }
    bool pbc_flags[3] = {true, true, true};
    if (pbc) {
      pbc_flags[0] = pbc[0];
      pbc_flags[1] = pbc[1];
      pbc_flags[2] = pbc[2];
    }
    Cell periodic_cell(lattice, pbc_flags[0], pbc_flags[1], pbc_flags[2]);

    AtomicSystem system(n_atoms, positions, atomic_numbers, &periodic_cell);
    return impl_->predict_impl(system, options);
  } else {
    // Non-periodic system
    AtomicSystem system(n_atoms, positions, atomic_numbers, nullptr);
    return impl_->predict_impl(system, options);
  }
}

ActivationResult Predictor::predict_with_activations(
    int32_t n_atoms, const float *positions, const int32_t *atomic_numbers,
    const float *cell, const bool *pbc) {
  if (!positions || !atomic_numbers) {
    throw std::invalid_argument(
        "positions and atomic_numbers must not be null");
  }

  // Only the auto-export graph path exposes intermediate tensors today.
  auto *graph_model =
      dynamic_cast<runtime::GraphModel *>(impl_->model.get());
  if (!graph_model) {
    throw std::runtime_error(
        "predict_with_activations is only supported for graph-based models "
        "(architecture 'pet-graph'); the loaded model is '" +
        impl_->model_type_str + "'.");
  }

  std::optional<AtomicSystem> system;
  if (cell) {
    float lattice[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lattice[i][j] = cell[i * 3 + j];
      }
    }
    bool pbc_flags[3] = {true, true, true};
    if (pbc) {
      pbc_flags[0] = pbc[0];
      pbc_flags[1] = pbc[1];
      pbc_flags[2] = pbc[2];
    }
    Cell periodic_cell(lattice, pbc_flags[0], pbc_flags[1], pbc_flags[2]);
    system.emplace(n_atoms, positions, atomic_numbers, &periodic_cell);
  } else {
    system.emplace(n_atoms, positions, atomic_numbers, nullptr);
  }

  ActivationResult out;
  auto internal_result = graph_model->predict_with_capture(
      *system, [&out](runtime::GraphInterpreter &interp) {
        interp.capture_all_outputs(
            [&out](int id, const std::string &name, ggml_tensor *t) {
              out.activations.push_back(tensor_to_activation(id, name, t));
            });
      });

  out.prediction.energy = internal_result.energy;
  if (internal_result.has_forces) {
    out.prediction.forces = std::move(internal_result.forces);
  }
  if (internal_result.has_stress) {
    out.prediction.stress = std::move(internal_result.stress);
  }
  return out;
}

} // namespace mlipcpp
