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
#include <mutex>

namespace mlipcpp {

const char *version() { return "0.1.0"; }

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
  case Backend::SYCL:
    return BackendPreference::SYCL;
  case Backend::CANN:
    return BackendPreference::CANN;
  case Backend::Auto:
  default:
    return BackendPreference::Auto;
  }
}

// Set the global backend preference (must be called before loading any models)
void set_backend(Backend backend) {
  std::lock_guard<std::mutex> lock(g_backend_mutex);
  auto pref = to_internal(backend);
  if (g_backend_provider && g_backend_provider->preference() != pref) {
    // Recreate backend with new preference
    g_backend_provider = BackendProvider::create(pref);
  }
  g_backend_preference = pref;
}

// Get the current backend name
const char *get_backend_name() {
  auto backend = get_global_backend();
  return backend->name().c_str();
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

  Result predict_impl(const AtomicSystem &system, const PredictOptions &options) {
    // Use predict_batch for NC forces support
    auto *pet_model = dynamic_cast<pet::PETModel *>(model.get());
    if (pet_model) {
      auto internal_results = pet_model->predict_batch(
          {system},
          options.compute_forces && !options.use_nc_forces,  // gradient-based forces
          options.use_nc_forces  // NC forces from forward pass
      );
      auto &internal_result = internal_results[0];

      Result result;
      result.energy = internal_result.energy;
      if (internal_result.has_forces) {
        result.forces = std::move(internal_result.forces);
      }
      if (internal_result.has_stress) {
        result.stress = std::move(internal_result.stress);
      }
      return result;
    } else {
      // Fallback for non-PET models
      return predict_impl(system, options.compute_forces);
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

} // namespace mlipcpp
