#include "backend.h"
#include "log.h"
#include <array>
#include <ggml.h>
#include <ggml-cpu.h>

namespace mlipcpp {

#ifndef __EMSCRIPTEN__
namespace log {

// No-op callback for suppressing ggml logging
static void ggml_log_noop(enum ggml_log_level /*level*/, const char * /*text*/,
                          void * /*user_data*/) {
  // Suppress all ggml logging
}

void suppress_ggml_logging() { ggml_log_set(ggml_log_noop, nullptr); }

} // namespace log
#endif

BackendProvider::~BackendProvider() {
  if (primary_ && primary_ != cpu_) {
    ggml_backend_free(primary_);
  }
  if (cpu_) {
    ggml_backend_free(cpu_);
  }
}

BackendProvider::BackendProvider(BackendProvider &&other) noexcept
    : primary_(other.primary_), cpu_(other.cpu_), name_(std::move(other.name_)),
      preference_(other.preference_) {
  other.primary_ = nullptr;
  other.cpu_ = nullptr;
}

BackendProvider &BackendProvider::operator=(BackendProvider &&other) noexcept {
  if (this != &other) {
    if (primary_ && primary_ != cpu_) {
      ggml_backend_free(primary_);
    }
    if (cpu_) {
      ggml_backend_free(cpu_);
    }
    primary_ = other.primary_;
    cpu_ = other.cpu_;
    name_ = std::move(other.name_);
    preference_ = other.preference_;
    other.primary_ = nullptr;
    other.cpu_ = nullptr;
  }
  return *this;
}

std::shared_ptr<BackendProvider>
BackendProvider::create(BackendPreference pref) {
  auto provider = std::shared_ptr<BackendProvider>(new BackendProvider());
  provider->preference_ = pref;

  // Always initialize CPU backend (required by scheduler as fallback)
  ggml_backend_dev_t cpu_dev =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (!cpu_dev) {
    throw std::runtime_error("Failed to get CPU backend device");
  }
  provider->cpu_ = ggml_backend_dev_init(cpu_dev, nullptr);
  if (!provider->cpu_) {
    throw std::runtime_error("Failed to initialize CPU backend");
  }

  // Helper to check if GPU backend name matches preference
  auto gpu_matches_preference = [pref](ggml_backend_t gpu) -> bool {
    if (!gpu)
      return false;
    std::string_view name = ggml_backend_name(gpu);
    switch (pref) {
    case BackendPreference::CUDA:
      return name.find("CUDA") != std::string_view::npos;
    case BackendPreference::HIP:
      return name.find("ROCm") != std::string_view::npos ||
             name.find("HIP") != std::string_view::npos;
    case BackendPreference::Metal:
      return name.find("Metal") != std::string_view::npos;
    case BackendPreference::Vulkan:
      return name.find("Vulkan") != std::string_view::npos;
    case BackendPreference::SYCL:
      return name.find("SYCL") != std::string_view::npos;
    case BackendPreference::CANN:
      return name.find("CANN") != std::string_view::npos;
    case BackendPreference::Auto:
      return true; // Any GPU is fine for Auto
    default:
      return false;
    }
  };

  // For WASM/Emscripten, set threads to 1 (no pthread support)
#ifdef __EMSCRIPTEN__
  ggml_backend_cpu_set_n_threads(provider->cpu_, 1);
#endif

  // CPU-only mode
  if (pref == BackendPreference::CPU) {
    provider->primary_ = provider->cpu_;
    provider->name_ = ggml_backend_name(provider->primary_);
    log::info("Backend: {}", provider->name_);
    return provider;
  }

  // Try GPU (discrete first, then integrated)
  ggml_backend_t gpu = nullptr;
  ggml_backend_dev_t gpu_dev =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
  if (gpu_dev) {
    gpu = ggml_backend_dev_init(gpu_dev, nullptr);
  }
  if (!gpu) {
    gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (gpu_dev) {
      gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    }
  }

  // Check if GPU matches preference
  if (gpu && gpu_matches_preference(gpu)) {
    provider->primary_ = gpu;
    provider->name_ = ggml_backend_name(provider->primary_);
    log::info("Backend: {}", provider->name_);
    return provider;
  }

  // GPU didn't match or not available
  if (gpu) {
    ggml_backend_free(gpu);
  }

  // For specific GPU preferences, fail if not available
  if (pref != BackendPreference::Auto) {
    throw std::runtime_error(std::string(backend_preference_name(pref)) +
                             " backend requested but not available");
  }

  // Auto mode: fall back to CPU
  provider->primary_ = provider->cpu_;
  provider->name_ = ggml_backend_name(provider->primary_);
  log::info("Backend: {} (GPU not available)", provider->name_);
  return provider;
}

} // namespace mlipcpp
