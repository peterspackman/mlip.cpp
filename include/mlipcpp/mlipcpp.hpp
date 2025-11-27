/**
 * @file mlipcpp.hpp
 * @brief C++ API for mlipcpp - Machine Learning Interatomic Potentials
 *
 * This header provides a modern C++ interface with RAII semantics.
 */

#ifndef MLIPCPP_HPP
#define MLIPCPP_HPP

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlipcpp {

// Forward declarations
class Predictor;

/**
 * @brief Backend selection for computation
 *
 * Backends are initialized if the corresponding ggml backend is available.
 * Use Auto for automatic selection of the best available backend.
 */
enum class Backend {
  Auto,   ///< Automatically select best available GPU, fallback to CPU
  CPU,    ///< CPU only
  CUDA,   ///< NVIDIA CUDA GPU
  HIP,    ///< AMD HIP/ROCm GPU
  Metal,  ///< Apple Metal GPU (macOS/iOS)
  Vulkan, ///< Vulkan GPU (cross-platform)
  SYCL,   ///< Intel SYCL (oneAPI)
  CANN,   ///< Huawei Ascend NPU
};

/**
 * @brief Prediction results from a model
 */
struct Result {
  float energy = 0.0f;
  std::vector<float> forces; ///< [n_atoms * 3] flattened (fx, fy, fz)
  std::vector<float> stress; ///< [6] Voigt notation (xx, yy, zz, yz, xz, xy)

  bool has_forces() const { return !forces.empty(); }
  bool has_stress() const { return !stress.empty(); }

  /// Get force on atom i as array reference
  std::span<const float, 3> force(size_t i) const {
    return std::span<const float, 3>(forces.data() + i * 3, 3);
  }
};

/**
 * @brief Model configuration options
 */
struct ModelOptions {
  Backend backend = Backend::Auto;
  float cutoff_override = 0.0f; ///< 0 = use model default
};

/**
 * @brief RAII wrapper for MLIP models
 *
 * Example usage:
 * @code
 * mlipcpp::Predictor model("pet-mad.gguf");
 *
 * // Non-periodic system
 * auto result = model.predict(positions, atomic_numbers);
 *
 * // Periodic system
 * auto result = model.predict(positions, atomic_numbers, cell, {true, true,
 * true});
 * @endcode
 */
class Predictor {
public:
  /**
   * @brief Load a model from a GGUF file
   * @param path Path to the GGUF model file
   * @param options Optional configuration
   * @throws std::runtime_error if loading fails
   */
  explicit Predictor(const std::string &path, const ModelOptions &options = {});

  /**
   * @brief Load a model from a memory buffer
   * @param data Pointer to GGUF data in memory
   * @param size Size of the buffer in bytes
   * @param options Optional configuration
   * @throws std::runtime_error if loading fails
   */
  Predictor(const uint8_t *data, size_t size, const ModelOptions &options = {});

  ~Predictor();

  // Move-only semantics
  Predictor(Predictor &&other) noexcept;
  Predictor &operator=(Predictor &&other) noexcept;
  Predictor(const Predictor &) = delete;
  Predictor &operator=(const Predictor &) = delete;

  /**
   * @brief Get the model's cutoff radius in Angstroms
   */
  float cutoff() const;

  /**
   * @brief Get the model type string (e.g., "PET")
   */
  std::string_view model_type() const;

  /**
   * @brief Predict energy and optionally forces for a non-periodic system
   *
   * @param positions Flattened positions [n_atoms * 3]
   * @param atomic_numbers Atomic numbers [n_atoms]
   * @param compute_forces Whether to compute forces (default: true)
   * @return Prediction results
   */
  Result predict(std::span<const float> positions,
                 std::span<const int32_t> atomic_numbers,
                 bool compute_forces = true);

  /**
   * @brief Predict energy and optionally forces for a periodic system
   *
   * @param positions Flattened positions [n_atoms * 3]
   * @param atomic_numbers Atomic numbers [n_atoms]
   * @param cell Lattice vectors as row-major 3x3 matrix [9]
   * @param pbc Periodic boundary conditions for x, y, z
   * @param compute_forces Whether to compute forces (default: true)
   * @return Prediction results
   */
  Result predict(std::span<const float> positions,
                 std::span<const int32_t> atomic_numbers,
                 std::span<const float, 9> cell, std::array<bool, 3> pbc,
                 bool compute_forces = true);

  /**
   * @brief Predict with raw pointers (for interfacing with other libraries)
   *
   * @param n_atoms Number of atoms
   * @param positions Positions array [n_atoms * 3]
   * @param atomic_numbers Atomic numbers array [n_atoms]
   * @param cell Lattice matrix [9] or nullptr for non-periodic
   * @param pbc PBC flags [3] or nullptr
   * @param compute_forces Whether to compute forces
   * @return Prediction results
   */
  Result predict(int32_t n_atoms, const float *positions,
                 const int32_t *atomic_numbers, const float *cell = nullptr,
                 const bool *pbc = nullptr, bool compute_forces = true);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Get the library version string
 */
const char *version();

/**
 * @brief Set the global backend for all models
 *
 * This affects all subsequently loaded models. Existing models are not affected.
 * Call this before loading any models to ensure they use the desired backend.
 *
 * @param backend Backend to use
 */
void set_backend(Backend backend);

/**
 * @brief Get the name of the current backend
 *
 * @return Backend name (e.g., "Metal", "CPU", "CUDA")
 */
const char *get_backend_name();

/**
 * @brief Suppress verbose logging from mlipcpp and GGML
 *
 * Call this at program startup to disable informational log messages.
 * This is useful for embedding mlipcpp in applications that have their
 * own logging infrastructure. After calling this, only errors will be logged.
 */
void suppress_logging();

} // namespace mlipcpp

#endif // MLIPCPP_HPP
