#ifndef MLIPCPP_H
#define MLIPCPP_H

/**
 * @file mlipcpp.h
 * @brief C API for mlipcpp - Machine Learning Interatomic Potentials in C++
 *
 * This header provides a C-compatible interface for loading and running
 * machine learning interatomic potentials (MLIPs) using GGUF model files.
 *
 * Thread safety: Model instances are NOT thread-safe. Each thread should
 * create its own model instance, or external synchronization must be used.
 *
 * Memory management: The C API uses explicit creation/free functions for
 * all opaque handles. Users must call the corresponding free function for
 * each created object.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version Information
 * ============================================================================
 */

/**
 * @brief Get the version string of the mlipcpp library
 * @return Version string in the format "MAJOR.MINOR.PATCH"
 */
const char *mlipcpp_version(void);

/* ============================================================================
 * Error Codes
 * ============================================================================
 */

/**
 * @brief Error codes returned by mlipcpp functions
 */
typedef enum {
  MLIPCPP_OK = 0,                     /**< Operation completed successfully */
  MLIPCPP_ERROR_INVALID_HANDLE = 1,   /**< Invalid model or result handle */
  MLIPCPP_ERROR_NULL_POINTER = 2,     /**< NULL pointer passed to function */
  MLIPCPP_ERROR_MODEL_NOT_LOADED = 3, /**< Model weights not loaded */
  MLIPCPP_ERROR_OUT_OF_MEMORY = 4,    /**< Memory allocation failed */
  MLIPCPP_ERROR_IO = 5,               /**< File I/O error */
  MLIPCPP_ERROR_COMPUTATION = 6,      /**< Computation graph execution failed */
  MLIPCPP_ERROR_INVALID_PARAMETER = 7, /**< Invalid parameter value */
  MLIPCPP_ERROR_BACKEND = 8,           /**< Backend initialization failed */
  MLIPCPP_ERROR_UNSUPPORTED = 9,       /**< Operation not supported */
  MLIPCPP_ERROR_INTERNAL = 255         /**< Internal error (bug in mlipcpp) */
} mlipcpp_error_t;

/**
 * @brief Convert error code to human-readable string
 * @param error Error code to convert
 * @return Static string describing the error (never NULL)
 */
const char *mlipcpp_error_string(mlipcpp_error_t error);

/**
 * @brief Get the last error message from any failed operation
 * @return Error message string, or NULL if no error occurred
 *
 * Note: This is thread-local. Each thread has its own error message.
 * The returned string is valid until the next mlipcpp function call.
 */
const char *mlipcpp_get_last_error(void);

/* ============================================================================
 * Backend and Precision Configuration
 * ============================================================================
 */

/**
 * @brief Backend selection for computation
 */
typedef enum {
  MLIPCPP_BACKEND_AUTO = 0,   /**< Automatically select best available backend */
  MLIPCPP_BACKEND_CPU = 1,    /**< Use CPU backend */
  MLIPCPP_BACKEND_CUDA = 2,   /**< NVIDIA CUDA GPU */
  MLIPCPP_BACKEND_HIP = 3,    /**< AMD HIP/ROCm GPU */
  MLIPCPP_BACKEND_METAL = 4,  /**< Use Metal GPU backend (macOS only) */
  MLIPCPP_BACKEND_VULKAN = 5, /**< Vulkan GPU (cross-platform) */
  MLIPCPP_BACKEND_SYCL = 6,   /**< Intel SYCL (oneAPI) */
  MLIPCPP_BACKEND_CANN = 7    /**< Huawei Ascend NPU */
} mlipcpp_backend_t;

/**
 * @brief Set the global backend for all subsequently loaded models
 * @param backend Backend to use
 *
 * Call this before loading any models to ensure they use the desired backend.
 * Existing models are not affected.
 */
void mlipcpp_set_backend(mlipcpp_backend_t backend);

/**
 * @brief Get the name of the current backend
 * @return Backend name string (e.g., "Metal", "CPU", "CUDA")
 */
const char *mlipcpp_get_backend_name(void);

/**
 * @brief Suppress verbose logging from mlipcpp and GGML
 *
 * Call this at program startup to disable informational log messages.
 * This is useful for embedding mlipcpp in applications that have their
 * own logging infrastructure. After calling this, only errors will be logged.
 */
void mlipcpp_suppress_logging(void);

/**
 * @brief Precision for computation
 */
typedef enum {
  MLIPCPP_PRECISION_F32 = 0, /**< 32-bit floating point */
  MLIPCPP_PRECISION_F16 =
      1 /**< 16-bit floating point (if supported by backend) */
} mlipcpp_precision_t;

/* ============================================================================
 * Opaque Handles
 * ============================================================================
 */

/**
 * @brief Opaque handle to a model instance
 */
typedef struct mlipcpp_model_impl *mlipcpp_model_t;

/**
 * @brief Opaque handle to prediction results
 *
 * Results are typically stack-allocated and do not need explicit freeing
 * unless created with _new variants (future extension).
 */
typedef struct mlipcpp_result_impl *mlipcpp_result_t;

/* ============================================================================
 * Model Configuration
 * ============================================================================
 */

/**
 * @brief Configuration options for model creation
 */
typedef struct {
  mlipcpp_backend_t backend;     /**< Backend to use for computation */
  mlipcpp_precision_t precision; /**< Precision for computation */
  float cutoff_override; /**< Override model cutoff (0.0 = use model default) */
} mlipcpp_model_options_t;

/**
 * @brief Get default model options
 * @param options Pointer to options struct to initialize
 * @return MLIPCPP_OK on success
 *
 * Default values:
 * - backend: MLIPCPP_BACKEND_AUTO
 * - precision: MLIPCPP_PRECISION_F32
 * - cutoff_override: 0.0 (use model default)
 */
mlipcpp_error_t mlipcpp_model_options_default(mlipcpp_model_options_t *options);

/* ============================================================================
 * Atomic System Definition
 * ============================================================================
 */

/**
 * @brief Atomic system data for prediction
 *
 * Memory layout:
 * - positions: [n_atoms * 3] flattened array of (x,y,z) coordinates
 * - atomic_numbers: [n_atoms] array of atomic numbers (Z values)
 * - cell: [9] flattened row-major 3x3 lattice matrix (can be NULL for
 * non-periodic)
 * - pbc: [3] periodicity flags for x, y, z directions (ignored if cell is NULL)
 *
 * Example for 2 atoms:
 *   positions = {x1, y1, z1, x2, y2, z2}
 *   atomic_numbers = {14, 14}  // Two silicon atoms
 *   cell = {a1, a2, a3, b1, b2, b3, c1, c2, c3}  // Lattice vectors as rows
 *   pbc = {true, true, true}  // Periodic in all directions
 */
typedef struct {
  int32_t n_atoms;               /**< Number of atoms in the system */
  const float *positions;        /**< Atom positions [n_atoms * 3] */
  const int32_t *atomic_numbers; /**< Atomic numbers [n_atoms] */
  const float *cell; /**< Lattice matrix [9] or NULL for non-periodic */
  const bool *pbc;   /**< Periodicity flags [3] or NULL */
} mlipcpp_system_t;

/* ============================================================================
 * Model Lifecycle
 * ============================================================================
 */

/**
 * @brief Create a new model instance with specified options
 * @param options Model configuration options (NULL for defaults)
 * @return Model handle on success, NULL on failure
 *
 * The returned model must be freed with mlipcpp_model_free().
 * Call mlipcpp_get_last_error() to retrieve error message on failure.
 */
mlipcpp_model_t mlipcpp_model_create(const mlipcpp_model_options_t *options);

/**
 * @brief Load model weights from GGUF file
 * @param model Model handle returned by mlipcpp_model_create()
 * @param path Path to GGUF model file
 * @return MLIPCPP_OK on success, error code on failure
 *
 * The model must be created before loading weights.
 * After successful loading, the model is ready for prediction.
 */
mlipcpp_error_t mlipcpp_model_load(mlipcpp_model_t model, const char *path);

/**
 * @brief Free a model instance and all associated resources
 * @param model Model handle to free
 *
 * After this call, the model handle is invalid and must not be used.
 * It is safe to pass NULL (no operation will be performed).
 */
void mlipcpp_model_free(mlipcpp_model_t model);

/**
 * @brief Get the cutoff radius of the loaded model
 * @param model Model handle
 * @param cutoff Pointer to store cutoff value (in Angstroms)
 * @return MLIPCPP_OK on success, error code on failure
 *
 * The cutoff defines the maximum interaction distance between atoms.
 * Only available after model weights are loaded.
 */
mlipcpp_error_t mlipcpp_model_get_cutoff(mlipcpp_model_t model, float *cutoff);

/* ============================================================================
 * Prediction
 * ============================================================================
 */

/**
 * @brief Run prediction on an atomic system
 * @param model Model handle
 * @param system Atomic system to predict on
 * @param compute_forces Whether to compute forces (true) or energy only (false)
 * @param result Pointer to result handle to store output
 * @return MLIPCPP_OK on success, error code on failure
 *
 * The result handle is internally managed and valid until the next prediction
 * call on the same model, or until the model is freed. No need to free.
 *
 * If compute_forces is false, only energy will be computed.
 * If compute_forces is true, both energy and forces will be computed.
 * Stress tensor computation depends on the model capabilities.
 */
mlipcpp_error_t mlipcpp_predict(mlipcpp_model_t model,
                                const mlipcpp_system_t *system,
                                bool compute_forces, mlipcpp_result_t *result);

/**
 * @brief Run prediction using raw pointers (alternative interface)
 * @param model Model handle
 * @param n_atoms Number of atoms
 * @param positions Atom positions [n_atoms * 3]
 * @param atomic_numbers Atomic numbers [n_atoms]
 * @param cell Lattice matrix [9] or NULL
 * @param pbc Periodicity flags [3] or NULL
 * @param compute_forces Whether to compute forces
 * @param result Pointer to result handle to store output
 * @return MLIPCPP_OK on success, error code on failure
 *
 * Convenience function that constructs mlipcpp_system_t internally.
 * Same lifetime and semantics as mlipcpp_predict().
 */
mlipcpp_error_t mlipcpp_predict_ptr(mlipcpp_model_t model, int32_t n_atoms,
                                    const float *positions,
                                    const int32_t *atomic_numbers,
                                    const float *cell, const bool *pbc,
                                    bool compute_forces,
                                    mlipcpp_result_t *result);

/* ============================================================================
 * Result Access
 * ============================================================================
 */

/**
 * @brief Get the energy from prediction results
 * @param result Result handle from mlipcpp_predict()
 * @param energy Pointer to store energy value (in eV)
 * @return MLIPCPP_OK on success, error code on failure
 */
mlipcpp_error_t mlipcpp_result_get_energy(mlipcpp_result_t result,
                                          float *energy);

/**
 * @brief Get the forces from prediction results
 * @param result Result handle from mlipcpp_predict()
 * @param forces Buffer to copy forces into [n_atoms * 3] (in eV/Angstrom)
 * @param n_atoms Number of atoms (must match prediction input)
 * @return MLIPCPP_OK on success, error code on failure
 *
 * The forces array must be pre-allocated with size [n_atoms * 3].
 * Forces are returned as flattened (fx, fy, fz) per atom.
 * Returns error if forces were not computed (compute_forces was false).
 */
mlipcpp_error_t mlipcpp_result_get_forces(mlipcpp_result_t result,
                                          float *forces, int32_t n_atoms);

/**
 * @brief Get the stress tensor from prediction results
 * @param result Result handle from mlipcpp_predict()
 * @param stress Buffer to copy stress into [6] in Voigt notation (in
 * eV/Angstrom^3)
 * @return MLIPCPP_OK on success, error code on failure
 *
 * The stress array must be pre-allocated with size [6].
 * Stress is returned in Voigt notation: [xx, yy, zz, yz, xz, xy]
 * Returns error if stress is not available.
 */
mlipcpp_error_t mlipcpp_result_get_stress(mlipcpp_result_t result,
                                          float stress[6]);

/**
 * @brief Get the number of atoms in the result
 * @param result Result handle from mlipcpp_predict()
 * @param n_atoms Pointer to store number of atoms
 * @return MLIPCPP_OK on success, error code on failure
 */
mlipcpp_error_t mlipcpp_result_n_atoms(mlipcpp_result_t result,
                                       int32_t *n_atoms);

/**
 * @brief Check if forces are available in the result
 * @param result Result handle from mlipcpp_predict()
 * @param has_forces Pointer to store availability flag
 * @return MLIPCPP_OK on success, error code on failure
 */
mlipcpp_error_t mlipcpp_result_has_forces(mlipcpp_result_t result,
                                          bool *has_forces);

/**
 * @brief Check if stress tensor is available in the result
 * @param result Result handle from mlipcpp_predict()
 * @param has_stress Pointer to store availability flag
 * @return MLIPCPP_OK on success, error code on failure
 */
mlipcpp_error_t mlipcpp_result_has_stress(mlipcpp_result_t result,
                                          bool *has_stress);

/**
 * @brief Free a result handle (only for results from _new variants)
 * @param result Result handle to free
 *
 * Currently not needed for standard mlipcpp_predict() results,
 * which are managed internally. Reserved for future API extensions.
 *
 * It is safe to pass NULL (no operation will be performed).
 */
void mlipcpp_result_free(mlipcpp_result_t result);

#ifdef __cplusplus
}
#endif

#endif /* MLIPCPP_H */
