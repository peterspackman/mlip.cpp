/**
 * @file mlipcpp_api.cpp
 * @brief C API implementation for mlipcpp
 *
 * This file implements the C interface declared in include/mlipcpp/mlipcpp.h
 * It wraps the C++ PETModel class to provide a pure C API for easy integration
 * with other languages and build systems.
 *
 * Thread safety: Each model instance is independent, but not thread-safe.
 * Error handling: Uses thread-local error storage for error messages.
 */

#include "mlipcpp/mlipcpp.h"
#include "core/backend.h"
#include "core/log.h"
#include "mlipcpp/model.h"
#include "mlipcpp/system.h"
#include "models/pet/pet.h"
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

/**
 * @brief Internal model implementation
 *
 * Wraps the C++ PETModel and stores the last result for zero-copy access.
 */
struct mlipcpp_model_impl {
  std::unique_ptr<mlipcpp::pet::PETModel> model;
  mlipcpp::ModelResult last_result;
  bool weights_loaded = false;
  int32_t last_n_atoms = 0; // Track atoms in last result
};

/**
 * @brief Internal result handle
 *
 * Points to the model's last_result to avoid copying large arrays.
 * The result remains valid until the next prediction or model destruction.
 */
struct mlipcpp_result_impl {
  mlipcpp::ModelResult *result; // Points to model's last_result
  int32_t n_atoms;
};

// Global backend provider - shared across all models
static std::shared_ptr<mlipcpp::BackendProvider> g_backend_provider;
static std::mutex g_backend_mutex;
static mlipcpp::BackendPreference g_backend_preference =
    mlipcpp::BackendPreference::Auto;

// Get or create the global backend provider
static std::shared_ptr<mlipcpp::BackendProvider> get_global_backend() {
  std::lock_guard<std::mutex> lock(g_backend_mutex);
  if (!g_backend_provider) {
    g_backend_provider = mlipcpp::BackendProvider::create(g_backend_preference);
  }
  return g_backend_provider;
}

// Anonymous namespace for internal helper functions
namespace {

/**
 * @brief Thread-local error storage
 *
 * Each thread maintains its own error message to support concurrent use
 * of different model instances in different threads.
 */
thread_local std::string g_last_error;

/**
 * @brief Set the last error message
 *
 * @param msg Error message to store
 */
void set_error(const std::string &msg) { g_last_error = msg; }

/**
 * @brief Clear the last error message
 */
void clear_error() { g_last_error.clear(); }

/**
 * @brief Convert C API BackendPreference enum to internal enum
 */
mlipcpp::BackendPreference to_backend_preference(mlipcpp_backend_t backend) {
  switch (backend) {
  case MLIPCPP_BACKEND_CPU:
    return mlipcpp::BackendPreference::CPU;
  case MLIPCPP_BACKEND_CUDA:
    return mlipcpp::BackendPreference::CUDA;
  case MLIPCPP_BACKEND_HIP:
    return mlipcpp::BackendPreference::HIP;
  case MLIPCPP_BACKEND_METAL:
    return mlipcpp::BackendPreference::Metal;
  case MLIPCPP_BACKEND_VULKAN:
    return mlipcpp::BackendPreference::Vulkan;
  case MLIPCPP_BACKEND_SYCL:
    return mlipcpp::BackendPreference::SYCL;
  case MLIPCPP_BACKEND_CANN:
    return mlipcpp::BackendPreference::CANN;
  case MLIPCPP_BACKEND_AUTO:
  default:
    return mlipcpp::BackendPreference::Auto;
  }
}

/**
 * @brief Convert ComputePrecision enum
 */
mlipcpp::pet::ComputePrecision
to_compute_precision(mlipcpp_precision_t precision) {
  switch (precision) {
  case MLIPCPP_PRECISION_F16:
    return mlipcpp::pet::ComputePrecision::F16;
  case MLIPCPP_PRECISION_F32:
  default:
    return mlipcpp::pet::ComputePrecision::F32;
  }
}

} // anonymous namespace

// ============================================================================
// Version Information
// ============================================================================

const char *mlipcpp_version(void) { return "0.1.0"; }

// ============================================================================
// Backend and Logging Control
// ============================================================================

void mlipcpp_set_backend(mlipcpp_backend_t backend) {
  std::lock_guard<std::mutex> lock(g_backend_mutex);
  auto pref = to_backend_preference(backend);
  if (g_backend_provider && g_backend_provider->preference() != pref) {
    // Recreate backend with new preference
    g_backend_provider = mlipcpp::BackendProvider::create(pref);
  }
  g_backend_preference = pref;
}

const char *mlipcpp_get_backend_name(void) {
  auto backend = get_global_backend();
  return backend->name().c_str();
}

void mlipcpp_suppress_logging(void) {
  mlipcpp::log::suppress_ggml_logging();
  mlipcpp::log::set_level(mlipcpp::log::Level::Off);
}

// ============================================================================
// Error Handling
// ============================================================================

const char *mlipcpp_error_string(mlipcpp_error_t error) {
  switch (error) {
  case MLIPCPP_OK:
    return "Success";
  case MLIPCPP_ERROR_INVALID_HANDLE:
    return "Invalid model or result handle";
  case MLIPCPP_ERROR_NULL_POINTER:
    return "NULL pointer passed to function";
  case MLIPCPP_ERROR_MODEL_NOT_LOADED:
    return "Model weights not loaded";
  case MLIPCPP_ERROR_OUT_OF_MEMORY:
    return "Memory allocation failed";
  case MLIPCPP_ERROR_IO:
    return "File I/O error";
  case MLIPCPP_ERROR_COMPUTATION:
    return "Computation graph execution failed";
  case MLIPCPP_ERROR_INVALID_PARAMETER:
    return "Invalid parameter value";
  case MLIPCPP_ERROR_BACKEND:
    return "Backend initialization failed";
  case MLIPCPP_ERROR_UNSUPPORTED:
    return "Operation not supported";
  case MLIPCPP_ERROR_INTERNAL:
    return "Internal error (bug in mlipcpp)";
  default:
    return "Unknown error";
  }
}

const char *mlipcpp_get_last_error(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

// ============================================================================
// Model Configuration
// ============================================================================

mlipcpp_error_t
mlipcpp_model_options_default(mlipcpp_model_options_t *options) {
  if (!options) {
    set_error("NULL pointer for options");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  options->backend = MLIPCPP_BACKEND_AUTO;
  options->precision = MLIPCPP_PRECISION_F32;
  options->cutoff_override = 0.0f;

  clear_error();
  return MLIPCPP_OK;
}

// ============================================================================
// Model Lifecycle
// ============================================================================

mlipcpp_model_t mlipcpp_model_create(const mlipcpp_model_options_t *options) {
  try {
    // Use default options if not provided
    mlipcpp_model_options_t default_opts;
    if (!options) {
      mlipcpp_model_options_default(&default_opts);
      options = &default_opts;
    }

    // Create internal model structure
    auto impl = std::make_unique<mlipcpp_model_impl>();

    // Create PETModel with default hypers (will be overridden by GGUF)
    mlipcpp::pet::PETHypers hypers;
    impl->model = std::make_unique<mlipcpp::pet::PETModel>(hypers);

    // Update global backend preference if specified in options
    auto backend_pref = to_backend_preference(options->backend);
    if (backend_pref != mlipcpp::BackendPreference::Auto) {
      mlipcpp_set_backend(options->backend);
    }

    // Set backend provider (uses global shared backend)
    impl->model->set_backend(get_global_backend());

    // Set compute precision
    impl->model->set_precision(to_compute_precision(options->precision));

    // Override cutoff if requested
    if (options->cutoff_override > 0.0f) {
      impl->model->set_cutoff(options->cutoff_override);
    }

    clear_error();
    return impl.release();
  } catch (const std::exception &e) {
    set_error(std::string("Failed to create model: ") + e.what());
    return nullptr;
  } catch (...) {
    set_error("Failed to create model: unknown error");
    return nullptr;
  }
}

mlipcpp_error_t mlipcpp_model_load(mlipcpp_model_t model, const char *path) {
  if (!model) {
    set_error("Invalid model handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!path) {
    set_error("NULL path");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    bool success = model->model->load_from_gguf(path);
    if (!success) {
      set_error(std::string("Failed to load model from: ") + path);
      return MLIPCPP_ERROR_IO;
    }

    model->weights_loaded = true;
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error loading model: ") + e.what());
    return MLIPCPP_ERROR_IO;
  } catch (...) {
    set_error("Error loading model: unknown error");
    return MLIPCPP_ERROR_INTERNAL;
  }
}

void mlipcpp_model_free(mlipcpp_model_t model) {
  if (model) {
    delete model;
  }
}

mlipcpp_error_t mlipcpp_model_get_cutoff(mlipcpp_model_t model, float *cutoff) {
  if (!model) {
    set_error("Invalid model handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!cutoff) {
    set_error("NULL cutoff pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (!model->weights_loaded) {
    set_error("Model weights not loaded");
    return MLIPCPP_ERROR_MODEL_NOT_LOADED;
  }

  try {
    *cutoff = model->model->cutoff();
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error getting cutoff: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

// ============================================================================
// Prediction Options
// ============================================================================

mlipcpp_error_t
mlipcpp_predict_options_default(mlipcpp_predict_options_t *options) {
  if (!options) {
    set_error("NULL pointer for options");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  options->compute_forces = true;
  options->compute_stress = false;
  options->use_nc_forces = false;

  clear_error();
  return MLIPCPP_OK;
}

// ============================================================================
// Prediction
// ============================================================================

mlipcpp_error_t mlipcpp_predict(mlipcpp_model_t model,
                                const mlipcpp_system_t *system,
                                bool compute_forces, mlipcpp_result_t *result) {
  if (!model) {
    set_error("Invalid model handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!system) {
    set_error("NULL system pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (!result) {
    set_error("NULL result pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (!model->weights_loaded) {
    set_error("Model weights not loaded");
    return MLIPCPP_ERROR_MODEL_NOT_LOADED;
  }

  // Validate system data
  if (system->n_atoms <= 0) {
    set_error("Invalid number of atoms");
    return MLIPCPP_ERROR_INVALID_PARAMETER;
  }
  if (!system->positions || !system->atomic_numbers) {
    set_error("NULL positions or atomic_numbers");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    // Build AtomicSystem from C data
    // Use the constructor that properly sets n_atoms
    const mlipcpp::Cell *cell_ptr = nullptr;
    mlipcpp::Cell cell;

    if (system->cell) {
      bool pbc[3] = {true, true, true};
      if (system->pbc) {
        pbc[0] = system->pbc[0];
        pbc[1] = system->pbc[1];
        pbc[2] = system->pbc[2];
      }

      // Unpack flattened cell matrix (row-major)
      float lattice[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          lattice[i][j] = system->cell[i * 3 + j];
        }
      }

      cell = mlipcpp::Cell(lattice, pbc[0], pbc[1], pbc[2]);
      cell_ptr = &cell;
    }

    // Create AtomicSystem using the constructor
    mlipcpp::AtomicSystem cpp_system(system->n_atoms, system->positions,
                                     system->atomic_numbers, cell_ptr);

    // Run prediction
    model->last_result = model->model->predict(cpp_system, compute_forces);
    model->last_n_atoms = system->n_atoms;

    // Create result handle pointing to the stored result
    auto result_impl = std::make_unique<mlipcpp_result_impl>();
    result_impl->result = &model->last_result;
    result_impl->n_atoms = system->n_atoms;

    *result = result_impl.release();

    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Prediction failed: ") + e.what());
    return MLIPCPP_ERROR_COMPUTATION;
  } catch (...) {
    set_error("Prediction failed: unknown error");
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_predict_ptr(mlipcpp_model_t model, int32_t n_atoms,
                                    const float *positions,
                                    const int32_t *atomic_numbers,
                                    const float *cell, const bool *pbc,
                                    bool compute_forces,
                                    mlipcpp_result_t *result) {
  // Construct mlipcpp_system_t and delegate to mlipcpp_predict
  mlipcpp_system_t system;
  system.n_atoms = n_atoms;
  system.positions = positions;
  system.atomic_numbers = atomic_numbers;
  system.cell = cell;
  system.pbc = pbc;

  return mlipcpp_predict(model, &system, compute_forces, result);
}

mlipcpp_error_t mlipcpp_predict_with_options(mlipcpp_model_t model,
                                              const mlipcpp_system_t *system,
                                              const mlipcpp_predict_options_t *options,
                                              mlipcpp_result_t *result) {
  if (!model) {
    set_error("Invalid model handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!system) {
    set_error("NULL system pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (!result) {
    set_error("NULL result pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (!model->weights_loaded) {
    set_error("Model weights not loaded");
    return MLIPCPP_ERROR_MODEL_NOT_LOADED;
  }

  // Use default options if not provided
  mlipcpp_predict_options_t default_opts;
  if (!options) {
    mlipcpp_predict_options_default(&default_opts);
    options = &default_opts;
  }

  // Validate system data
  if (system->n_atoms <= 0) {
    set_error("Invalid number of atoms");
    return MLIPCPP_ERROR_INVALID_PARAMETER;
  }
  if (!system->positions || !system->atomic_numbers) {
    set_error("NULL positions or atomic_numbers");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    // Build AtomicSystem from C data
    const mlipcpp::Cell *cell_ptr = nullptr;
    mlipcpp::Cell cell;

    if (system->cell) {
      bool pbc[3] = {true, true, true};
      if (system->pbc) {
        pbc[0] = system->pbc[0];
        pbc[1] = system->pbc[1];
        pbc[2] = system->pbc[2];
      }

      // Unpack flattened cell matrix (row-major)
      float lattice[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          lattice[i][j] = system->cell[i * 3 + j];
        }
      }

      cell = mlipcpp::Cell(lattice, pbc[0], pbc[1], pbc[2]);
      cell_ptr = &cell;
    }

    // Create AtomicSystem using the constructor
    mlipcpp::AtomicSystem cpp_system(system->n_atoms, system->positions,
                                     system->atomic_numbers, cell_ptr);

    // Run prediction using predict_batch for NC forces support
    auto *pet_model = dynamic_cast<mlipcpp::pet::PETModel *>(model->model.get());
    if (pet_model) {
      auto results = pet_model->predict_batch(
          {cpp_system},
          options->compute_forces && !options->use_nc_forces,  // gradient-based forces
          options->use_nc_forces  // NC forces from forward pass
      );
      model->last_result = std::move(results[0]);
    } else {
      // Fallback for non-PET models
      model->last_result = model->model->predict(cpp_system, options->compute_forces);
    }
    model->last_n_atoms = system->n_atoms;

    // Create result handle pointing to the stored result
    auto result_impl = std::make_unique<mlipcpp_result_impl>();
    result_impl->result = &model->last_result;
    result_impl->n_atoms = system->n_atoms;

    *result = result_impl.release();

    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Prediction failed: ") + e.what());
    return MLIPCPP_ERROR_COMPUTATION;
  } catch (...) {
    set_error("Prediction failed: unknown error");
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_predict_ptr_with_options(mlipcpp_model_t model, int32_t n_atoms,
                                                  const float *positions,
                                                  const int32_t *atomic_numbers,
                                                  const float *cell, const bool *pbc,
                                                  const mlipcpp_predict_options_t *options,
                                                  mlipcpp_result_t *result) {
  // Construct mlipcpp_system_t and delegate to mlipcpp_predict_with_options
  mlipcpp_system_t system;
  system.n_atoms = n_atoms;
  system.positions = positions;
  system.atomic_numbers = atomic_numbers;
  system.cell = cell;
  system.pbc = pbc;

  return mlipcpp_predict_with_options(model, &system, options, result);
}

// ============================================================================
// Result Access
// ============================================================================

mlipcpp_error_t mlipcpp_result_get_energy(mlipcpp_result_t result,
                                          float *energy) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!energy) {
    set_error("NULL energy pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    *energy = result->result->energy;
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error getting energy: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_result_get_forces(mlipcpp_result_t result,
                                          float *forces, int32_t n_atoms) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!forces) {
    set_error("NULL forces pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }
  if (n_atoms != result->n_atoms) {
    set_error("n_atoms mismatch");
    return MLIPCPP_ERROR_INVALID_PARAMETER;
  }

  try {
    if (!result->result->has_forces) {
      set_error("Forces not available (compute_forces was false)");
      return MLIPCPP_ERROR_UNSUPPORTED;
    }

    // Copy forces to output buffer
    std::memcpy(forces, result->result->forces.data(),
                n_atoms * 3 * sizeof(float));

    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error getting forces: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_result_get_stress(mlipcpp_result_t result,
                                          float stress[6]) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!stress) {
    set_error("NULL stress pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    if (!result->result->has_stress) {
      set_error("Stress not available");
      return MLIPCPP_ERROR_UNSUPPORTED;
    }

    // Copy stress tensor to output buffer
    std::memcpy(stress, result->result->stress.data(), 6 * sizeof(float));

    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error getting stress: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_result_n_atoms(mlipcpp_result_t result,
                                       int32_t *n_atoms) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!n_atoms) {
    set_error("NULL n_atoms pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    *n_atoms = result->n_atoms;
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error getting n_atoms: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_result_has_forces(mlipcpp_result_t result,
                                          bool *has_forces) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!has_forces) {
    set_error("NULL has_forces pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    *has_forces = result->result->has_forces;
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error checking forces: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

mlipcpp_error_t mlipcpp_result_has_stress(mlipcpp_result_t result,
                                          bool *has_stress) {
  if (!result) {
    set_error("Invalid result handle");
    return MLIPCPP_ERROR_INVALID_HANDLE;
  }
  if (!has_stress) {
    set_error("NULL has_stress pointer");
    return MLIPCPP_ERROR_NULL_POINTER;
  }

  try {
    *has_stress = result->result->has_stress;
    clear_error();
    return MLIPCPP_OK;
  } catch (const std::exception &e) {
    set_error(std::string("Error checking stress: ") + e.what());
    return MLIPCPP_ERROR_INTERNAL;
  }
}

void mlipcpp_result_free(mlipcpp_result_t result) {
  if (result) {
    delete result;
  }
}
