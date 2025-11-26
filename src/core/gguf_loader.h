#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace mlipcpp {

/// GGUF file loader using ggml's built-in GGUF support
class GGUFLoader {
public:
  // Load into a new context (we own it)
  explicit GGUFLoader(const std::string &path);

  // Load into an existing context (caller owns it)
  GGUFLoader(const std::string &path, ggml_context *existing_ctx);

  ~GGUFLoader();

  // Prevent copying
  GGUFLoader(const GGUFLoader &) = delete;
  GGUFLoader &operator=(const GGUFLoader &) = delete;

  /// Get string metadata
  std::string get_string(const std::string &key,
                         const std::string &default_val = "") const;

  /// Get float32 metadata
  float get_float32(const std::string &key, float default_val = 0.0f) const;

  /// Get int32 metadata
  int32_t get_int32(const std::string &key, int32_t default_val = 0) const;

  /// Get int32 array metadata
  std::vector<int32_t> get_array_int32(const std::string &key) const;

  /// Get float32 array metadata
  std::vector<float> get_array_float32(const std::string &key) const;

  /// Get tensor by name
  ggml_tensor *get_tensor(const std::string &name) const;

  /// Get all tensor names
  std::vector<std::string> get_tensor_names() const;

  /// Get GGML context (owned by loader)
  ggml_context *context() const { return ctx_; }

  /// Get GGUF context
  gguf_context *gguf_ctx() const { return gguf_ctx_; }

private:
  ggml_context *ctx_ = nullptr;
  gguf_context *gguf_ctx_ = nullptr;
};

} // namespace mlipcpp
