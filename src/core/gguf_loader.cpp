#include "gguf_loader.h"
#include <cstring>
#include <ggml.h>
#include <gguf.h>
#include <stdexcept>

namespace mlipcpp {

GGUFLoader::GGUFLoader(const std::string &path) {
  // Initialize GGUF context from file
  struct gguf_init_params params = {
      .no_alloc = false,
      .ctx = &ctx_, // Pass address of our context pointer
  };

  gguf_ctx_ = gguf_init_from_file(path.c_str(), params);
  if (!gguf_ctx_) {
    throw std::runtime_error("Failed to load GGUF file: " + path);
  }

  if (!ctx_) {
    gguf_free(gguf_ctx_);
    throw std::runtime_error("Failed to create GGML context from GGUF");
  }
}

GGUFLoader::GGUFLoader(const std::string &path, ggml_context *existing_ctx)
    : ctx_(existing_ctx) {

  // Initialize GGUF context from file with existing context
  struct gguf_init_params params = {
      .no_alloc = false,
      .ctx = &ctx_, // Use existing context
  };

  gguf_ctx_ = gguf_init_from_file(path.c_str(), params);
  if (!gguf_ctx_) {
    throw std::runtime_error("Failed to load GGUF file: " + path);
  }
}

GGUFLoader::~GGUFLoader() {
  if (gguf_ctx_) {
    gguf_free(gguf_ctx_);
  }
  // Only free context if we own it
  // If owns_context_ is false, caller manages the context lifetime
}

std::string GGUFLoader::get_string(const std::string &key,
                                   const std::string &default_val) const {
  int64_t key_id = gguf_find_key(gguf_ctx_, key.c_str());
  if (key_id < 0) {
    return default_val;
  }

  const char *val = gguf_get_val_str(gguf_ctx_, key_id);
  return val ? std::string(val) : default_val;
}

float GGUFLoader::get_float32(const std::string &key, float default_val) const {
  int64_t key_id = gguf_find_key(gguf_ctx_, key.c_str());
  if (key_id < 0) {
    return default_val;
  }

  return gguf_get_val_f32(gguf_ctx_, key_id);
}

int32_t GGUFLoader::get_int32(const std::string &key,
                              int32_t default_val) const {
  int64_t key_id = gguf_find_key(gguf_ctx_, key.c_str());
  if (key_id < 0) {
    return default_val;
  }

  return gguf_get_val_i32(gguf_ctx_, key_id);
}

std::vector<int32_t> GGUFLoader::get_array_int32(const std::string &key) const {
  int64_t key_id = gguf_find_key(gguf_ctx_, key.c_str());
  if (key_id < 0) {
    return {};
  }

  enum gguf_type type = gguf_get_kv_type(gguf_ctx_, key_id);
  if (type != GGUF_TYPE_ARRAY) {
    return {};
  }

  enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx_, key_id);
  if (arr_type != GGUF_TYPE_INT32) {
    return {};
  }

  int64_t n = gguf_get_arr_n(gguf_ctx_, key_id);
  std::vector<int32_t> result(n);

  const int32_t *data =
      static_cast<const int32_t *>(gguf_get_arr_data(gguf_ctx_, key_id));
  std::memcpy(result.data(), data, n * sizeof(int32_t));

  return result;
}

std::vector<float> GGUFLoader::get_array_float32(const std::string &key) const {
  int64_t key_id = gguf_find_key(gguf_ctx_, key.c_str());
  if (key_id < 0) {
    return {};
  }

  enum gguf_type type = gguf_get_kv_type(gguf_ctx_, key_id);
  if (type != GGUF_TYPE_ARRAY) {
    return {};
  }

  enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx_, key_id);
  if (arr_type != GGUF_TYPE_FLOAT32) {
    return {};
  }

  int64_t n = gguf_get_arr_n(gguf_ctx_, key_id);
  std::vector<float> result(n);

  const float *data =
      static_cast<const float *>(gguf_get_arr_data(gguf_ctx_, key_id));
  std::memcpy(result.data(), data, n * sizeof(float));

  return result;
}

ggml_tensor *GGUFLoader::get_tensor(const std::string &name) const {
  return ggml_get_tensor(ctx_, name.c_str());
}

std::vector<std::string> GGUFLoader::get_tensor_names() const {
  std::vector<std::string> names;
  int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);

  for (int64_t i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(gguf_ctx_, i);
    if (name) {
      names.push_back(name);
    }
  }

  return names;
}

} // namespace mlipcpp
