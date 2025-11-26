#pragma once

#include <memory>
#include <string>

struct ggml_backend;
struct ggml_gallocr;
struct ggml_backend_buffer_type;

namespace mlipcpp {

/**
 * Backend manager for GGML computations
 *
 * Provides a centralized interface for selecting and managing compute backends
 * (CPU, Metal, CUDA, etc.) across all models.
 */
class BackendManager {
public:
  /**
   * Backend types supported by the system
   */
  enum class Type {
    AUTO,  // Auto-detect best available backend
    CPU,   // CPU backend (always available)
    METAL, // Metal backend (macOS GPU)
    CUDA   // CUDA backend (NVIDIA GPU)
  };

  /**
   * Initialize backend with specified type
   *
   * @param type Backend type to initialize
   * @return true if successful, false otherwise
   */
  static bool initialize(Type type = Type::AUTO);

  /**
   * Get the current backend instance
   *
   * @return Pointer to backend, or nullptr if not initialized
   */
  static ggml_backend *backend();

  /**
   * Get the backend buffer type for tensor allocation
   *
   * @return Backend buffer type for use with ggml_gallocr
   */
  static ggml_backend_buffer_type *buffer_type();

  /**
   * Get the name of the current backend
   *
   * @return Backend name (e.g., "CPU", "Metal", "CUDA")
   */
  static std::string backend_name();

  /**
   * Check if a specific backend type is available
   *
   * @param type Backend type to check
   * @return true if available, false otherwise
   */
  static bool is_available(Type type);

  /**
   * Free backend resources
   */
  static void shutdown();

  /**
   * Parse backend type from string
   *
   * @param str String representation ("auto", "cpu", "metal", "cuda")
   * @return Backend type
   */
  static Type parse_type(const std::string &str);

private:
  BackendManager() = default;

  static ggml_backend *backend_;
  static Type current_type_;
};

} // namespace mlipcpp
