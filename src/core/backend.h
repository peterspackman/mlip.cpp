#pragma once

#include <ggml-backend.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlipcpp {

// Backend preference for computation
enum class BackendPreference {
  Auto,   // Use best available GPU, fall back to CPU
  CPU,    // Force CPU only
  CUDA,   // NVIDIA CUDA GPU
  HIP,    // AMD HIP/ROCm GPU
  Metal,  // Apple Metal GPU (macOS/iOS)
  Vulkan, // Vulkan GPU (cross-platform)
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
  static constexpr const char *names[] = {"auto",  "cpu",    "cuda", "hip",
                                          "metal", "vulkan", "sycl", "cann"};
  return names[static_cast<size_t>(pref)];
}

// Parse backend preference from string
inline BackendPreference parse_backend_preference(std::string_view name) {
  if (name == "auto")
    return BackendPreference::Auto;
  if (name == "cpu")
    return BackendPreference::CPU;
  if (name == "cuda")
    return BackendPreference::CUDA;
  if (name == "hip")
    return BackendPreference::HIP;
  if (name == "metal")
    return BackendPreference::Metal;
  if (name == "vulkan")
    return BackendPreference::Vulkan;
  if (name == "sycl")
    return BackendPreference::SYCL;
  if (name == "cann")
    return BackendPreference::CANN;
  throw std::runtime_error("Unknown backend: " + std::string(name));
}

} // namespace mlipcpp
