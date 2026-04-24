#pragma once

#include <ggml-backend.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlipcpp {

// Thrown when a specific backend is requested but the corresponding ggml
// backend is either not compiled in or has no available devices on this
// host. C/C++/Python APIs translate this into MLIPCPP_ERROR_BACKEND /
// a Python RuntimeError with a clear message, rather than a generic IO
// error, so callers can distinguish "missing GPU" from "bad GGUF file".
class BackendUnavailableError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Backend preference for computation
enum class BackendPreference {
  Auto,   // Use best available GPU, fall back to CPU
  CPU,    // Force CPU only
  CUDA,   // NVIDIA CUDA GPU
  HIP,    // AMD HIP/ROCm GPU
  Metal,  // Apple Metal GPU (macOS/iOS)
  Vulkan, // Vulkan GPU (cross-platform)
  WebGPU, // WebGPU (Dawn native or browser)
  SYCL,   // Intel SYCL (oneAPI)
  CANN,   // Huawei Ascend NPU
};

/**
 * Backend provider - manages ggml backend lifecycle
 *
 * This class owns the backend instances and can be shared across multiple
 * models. The scheduler requires CPU as a fallback, so we always maintain
 * both a primary backend and CPU backend.
 *
 * Usage:
 *   auto backend = BackendProvider::create(BackendPreference::Metal);
 *   model.set_backend(backend);
 */
class BackendProvider {
public:
  ~BackendProvider();

  // Non-copyable, movable
  BackendProvider(const BackendProvider &) = delete;
  BackendProvider &operator=(const BackendProvider &) = delete;
  BackendProvider(BackendProvider &&) noexcept;
  BackendProvider &operator=(BackendProvider &&) noexcept;

  /**
   * Create a backend provider with the specified preference
   *
   * @param pref Backend preference (Auto, CPU, Metal, CUDA, etc.)
   * @return Backend provider instance
   * @throws std::runtime_error if requested backend is not available
   */
  static std::shared_ptr<BackendProvider>
  create(BackendPreference pref = BackendPreference::Auto);

  // Accessors
  ggml_backend_t primary() const { return primary_; }
  ggml_backend_t cpu() const { return cpu_; }
  const std::string &name() const { return name_; }
  BackendPreference preference() const { return preference_; }

  // Check if primary backend is GPU (different from CPU)
  bool is_gpu() const { return primary_ != cpu_; }

  // Get buffer type for the primary backend
  ggml_backend_buffer_type_t buffer_type() const {
    return ggml_backend_get_default_buffer_type(primary_);
  }

  // Get buffer type for CPU backend
  ggml_backend_buffer_type_t cpu_buffer_type() const {
    return ggml_backend_get_default_buffer_type(cpu_);
  }

private:
  BackendProvider() = default;

  ggml_backend_t primary_ = nullptr; // Primary backend (CPU or GPU)
  ggml_backend_t cpu_ = nullptr;     // CPU backend (always available)
  std::string name_;
  BackendPreference preference_ = BackendPreference::Auto;
};

// Convenience function to get preference name
inline const char *backend_preference_name(BackendPreference pref) {
  static constexpr const char *names[] = {"auto",   "cpu",    "cuda", "hip",
                                          "metal",  "vulkan", "webgpu",
                                          "sycl",   "cann"};
  return names[static_cast<size_t>(pref)];
}

// Parse backend preference from string. Accepts common aliases
// (e.g. "mtl" → Metal, "rocm" → HIP) so the same names work for the CLI,
// Python, and JS entry points.
BackendPreference parse_backend_preference(std::string_view name);

// Check whether the ggml backend corresponding to `pref` is both compiled in
// and has at least one device available on this host. For Auto and CPU this
// is always true (CPU is always present). For specific GPU preferences this
// enumerates ggml_backend_dev_* and checks the device names.
bool is_backend_available(BackendPreference pref);

} // namespace mlipcpp
