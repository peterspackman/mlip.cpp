#include "backend.h"
#include "log.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <ggml.h>
#include <ggml-cpu.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlipcpp {

bool is_backend_available(BackendPreference pref) {
  if (pref == BackendPreference::Auto || pref == BackendPreference::CPU) {
    return true;
  }

  const char *needle = nullptr;
  const char *needle2 = nullptr;
  switch (pref) {
  case BackendPreference::CUDA:   needle = "CUDA"; break;
  case BackendPreference::HIP:    needle = "ROCm"; needle2 = "HIP"; break;
  case BackendPreference::Metal:  needle = "Metal"; needle2 = "MTL"; break;
  case BackendPreference::Vulkan: needle = "Vulkan"; break;
  case BackendPreference::WebGPU: needle = "WebGPU"; break;
  case BackendPreference::SYCL:   needle = "SYCL"; break;
  case BackendPreference::CANN:   needle = "CANN"; break;
  default: return false;
  }

  size_t n = ggml_backend_dev_count();
  for (size_t i = 0; i < n; ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    auto t = ggml_backend_dev_type(dev);
    if (t != GGML_BACKEND_DEVICE_TYPE_GPU &&
        t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
      continue;
    }
    std::string_view name = ggml_backend_dev_name(dev);
    if (name.find(needle) != std::string_view::npos) return true;
    if (needle2 && name.find(needle2) != std::string_view::npos) return true;
  }
  return false;
}

BackendPreference parse_backend_preference(std::string_view name_in) {
  std::string name(name_in);
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, BackendPreference> table = {
      {"auto",   BackendPreference::Auto},
      {"cpu",    BackendPreference::CPU},
      {"cuda",   BackendPreference::CUDA},
      {"nvidia", BackendPreference::CUDA},
      {"hip",    BackendPreference::HIP},
      {"rocm",   BackendPreference::HIP},
      {"metal",  BackendPreference::Metal},
      {"mtl",    BackendPreference::Metal},
      {"vulkan", BackendPreference::Vulkan},
      {"vk",     BackendPreference::Vulkan},
      {"webgpu", BackendPreference::WebGPU},
      {"wgpu",   BackendPreference::WebGPU},
      {"sycl",   BackendPreference::SYCL},
      {"cann",   BackendPreference::CANN},
  };
  auto it = table.find(name);
  if (it == table.end()) {
    throw std::runtime_error("Unknown backend: " + std::string(name_in));
  }
  return it->second;
}

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
      // Upstream renamed the Metal backend to "MTL" (with device suffixes).
      return name.find("Metal") != std::string_view::npos ||
             name.find("MTL")   != std::string_view::npos;
    case BackendPreference::Vulkan:
      return name.find("Vulkan") != std::string_view::npos;
    case BackendPreference::WebGPU:
      return name.find("WebGPU") != std::string_view::npos;
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

  // Enumerate all GPU devices and try to match the preference. For specific
  // preferences (Metal, WebGPU, ...) we scan every GPU and pick one whose
  // name matches. For Auto we pick the first GPU, except on Emscripten where
  // we prefer WebGPU.
  std::vector<ggml_backend_dev_t> gpu_devs;
  {
    size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
      ggml_backend_dev_t d = ggml_backend_dev_get(i);
      auto t = ggml_backend_dev_type(d);
      if (t == GGML_BACKEND_DEVICE_TYPE_GPU ||
          t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        gpu_devs.push_back(d);
      }
    }
  }

  ggml_backend_t gpu = nullptr;
  if (pref == BackendPreference::Auto) {
#ifdef __EMSCRIPTEN__
    for (auto d : gpu_devs) {
      std::string_view n = ggml_backend_dev_name(d);
      if (n.find("WebGPU") != std::string_view::npos ||
          n.find("webgpu") != std::string_view::npos) {
        gpu = ggml_backend_dev_init(d, nullptr);
        if (gpu) break;
      }
    }
#endif
    if (!gpu && !gpu_devs.empty()) {
      gpu = ggml_backend_dev_init(gpu_devs[0], nullptr);
    }
  } else {
    // Specific GPU preference: scan devices until one matches.
    for (auto d : gpu_devs) {
      ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
      if (!b) continue;
      if (gpu_matches_preference(b)) {
        gpu = b;
        break;
      }
      ggml_backend_free(b);
    }
  }

  // Check if GPU matches preference (Auto accepts any)
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
    throw BackendUnavailableError(
        std::string(backend_preference_name(pref)) +
        " backend requested but not available (not compiled in or no "
        "compatible device detected)");
  }

  // Auto mode: fall back to CPU. Log at warn level so it survives callers
  // that set the log level to Warn or Error to suppress info-level chatter
  // — users expecting GPU acceleration should see this.
  provider->primary_ = provider->cpu_;
  provider->name_ = ggml_backend_name(provider->primary_);
  log::warn("Backend: {} (GPU not available; Auto fell back to CPU)",
            provider->name_);
  return provider;
}

} // namespace mlipcpp
