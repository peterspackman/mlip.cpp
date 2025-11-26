#include "ggml_utils.h"
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mlipcpp {
namespace ggml {

// ============================================================================
// Context implementation
// ============================================================================

Context::Context(size_t mem_size, bool no_alloc) {
  ggml_init_params params = {
      mem_size, // mem_size
      nullptr,  // mem_buffer (nullptr = allocate internally)
      no_alloc  // no_alloc
  };

  ctx_ = ggml_init(params);
  if (!ctx_) {
    throw std::runtime_error("Failed to create GGML context");
  }
}

Context::~Context() {
  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
  }
}

Context::Context(Context &&other) noexcept : ctx_(other.ctx_) {
  other.ctx_ = nullptr;
}

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    // Free existing resource
    if (ctx_) {
      ggml_free(ctx_);
    }
    // Transfer ownership
    ctx_ = other.ctx_;
    other.ctx_ = nullptr;
  }
  return *this;
}

// ============================================================================
// Backend implementation
// ============================================================================

Backend::Backend(enum ggml_backend_dev_type device_type) {
  ggml_backend_dev_t dev = ggml_backend_dev_by_type(device_type);
  if (!dev) {
    throw std::runtime_error("Failed to get backend device");
  }

  backend_ = ggml_backend_dev_init(dev, nullptr);
  if (!backend_) {
    throw std::runtime_error("Failed to initialize backend");
  }
}

Backend::~Backend() {
  if (backend_) {
    ggml_backend_free(backend_);
    backend_ = nullptr;
  }
}

Backend::Backend(Backend &&other) noexcept : backend_(other.backend_) {
  other.backend_ = nullptr;
}

Backend &Backend::operator=(Backend &&other) noexcept {
  if (this != &other) {
    if (backend_) {
      ggml_backend_free(backend_);
    }
    backend_ = other.backend_;
    other.backend_ = nullptr;
  }
  return *this;
}

ggml_backend_buffer_type_t Backend::default_buffer_type() const {
  if (!backend_) {
    throw std::runtime_error("Cannot get buffer type from null backend");
  }
  return ggml_backend_get_default_buffer_type(backend_);
}

// ============================================================================
// BackendBuffer implementation
// ============================================================================

BackendBuffer::BackendBuffer(ggml_context *ctx,
                             ggml_backend_buffer_type_t buf_type) {
  if (!ctx) {
    throw std::runtime_error("Cannot create buffer from null context");
  }
  if (!buf_type) {
    throw std::runtime_error("Cannot create buffer with null buffer type");
  }

  buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buf_type);
  if (!buffer_) {
    throw std::runtime_error("Failed to allocate backend buffer");
  }
}

BackendBuffer::~BackendBuffer() {
  if (buffer_) {
    ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
  }
}

BackendBuffer::BackendBuffer(BackendBuffer &&other) noexcept
    : buffer_(other.buffer_) {
  other.buffer_ = nullptr;
}

BackendBuffer &BackendBuffer::operator=(BackendBuffer &&other) noexcept {
  if (this != &other) {
    if (buffer_) {
      ggml_backend_buffer_free(buffer_);
    }
    buffer_ = other.buffer_;
    other.buffer_ = nullptr;
  }
  return *this;
}

size_t BackendBuffer::size() const {
  if (!buffer_) {
    return 0;
  }
  return ggml_backend_buffer_get_size(buffer_);
}

void BackendBuffer::set_usage(ggml_backend_buffer_usage usage) {
  if (!buffer_) {
    throw std::runtime_error("Cannot set usage on null buffer");
  }
  ggml_backend_buffer_set_usage(buffer_, usage);
}

// ============================================================================
// Scheduler implementation
// ============================================================================

Scheduler::Scheduler(
    const std::vector<ggml_backend_t> &backends,
    const std::vector<ggml_backend_buffer_type_t> &buffer_types,
    size_t max_nodes, bool parallel) {

  if (backends.empty()) {
    throw std::runtime_error("Cannot create scheduler with no backends");
  }
  if (backends.size() != buffer_types.size()) {
    throw std::runtime_error(
        "Number of backends must match number of buffer types");
  }

  // ggml_backend_sched_new requires non-const pointers (it modifies them
  // internally) Create mutable copies
  std::vector<ggml_backend_t> backends_copy = backends;
  std::vector<ggml_backend_buffer_type_t> buffer_types_copy = buffer_types;

  sched_ =
      ggml_backend_sched_new(backends_copy.data(), buffer_types_copy.data(),
                             backends_copy.size(), max_nodes, parallel,
                             false // skip_backends (op_offload parameter)
      );

  if (!sched_) {
    throw std::runtime_error("Failed to create backend scheduler");
  }
}

Scheduler::~Scheduler() {
  if (sched_) {
    ggml_backend_sched_free(sched_);
    sched_ = nullptr;
  }
}

Scheduler::Scheduler(Scheduler &&other) noexcept : sched_(other.sched_) {
  other.sched_ = nullptr;
}

Scheduler &Scheduler::operator=(Scheduler &&other) noexcept {
  if (this != &other) {
    if (sched_) {
      ggml_backend_sched_free(sched_);
    }
    sched_ = other.sched_;
    other.sched_ = nullptr;
  }
  return *this;
}

bool Scheduler::alloc_graph(ggml_cgraph *graph) {
  if (!sched_) {
    throw std::runtime_error("Cannot allocate graph with null scheduler");
  }
  if (!graph) {
    throw std::runtime_error("Cannot allocate null graph");
  }
  return ggml_backend_sched_alloc_graph(sched_, graph);
}

ggml_status Scheduler::compute_graph(ggml_cgraph *graph) {
  if (!sched_) {
    throw std::runtime_error("Cannot compute graph with null scheduler");
  }
  if (!graph) {
    throw std::runtime_error("Cannot compute null graph");
  }
  return ggml_backend_sched_graph_compute(sched_, graph);
}

// ============================================================================
// Tensor operations
// ============================================================================

namespace ops {

ggml_tensor *dup(ggml_context *ctx, ggml_tensor *src) {
  if (!ctx) {
    throw std::runtime_error("Cannot dup into null context");
  }
  if (!src) {
    throw std::runtime_error("Cannot dup null tensor");
  }
  return ggml_dup(ctx, src);
}

template <typename T> std::vector<T> read(ggml_tensor *tensor) {
  if (!tensor) {
    throw std::runtime_error("Cannot read from null tensor");
  }

  size_t n_elements = ggml_nelements(tensor);
  std::vector<T> result(n_elements);

  // Read from backend buffer
  ggml_backend_tensor_get(tensor, result.data(), 0, n_elements * sizeof(T));

  return result;
}

// Explicit template instantiations for common types
template std::vector<float> read<float>(ggml_tensor *tensor);
template std::vector<int32_t> read<int32_t>(ggml_tensor *tensor);
template std::vector<int64_t> read<int64_t>(ggml_tensor *tensor);
template std::vector<uint32_t> read<uint32_t>(ggml_tensor *tensor);

template <typename T>
void write(ggml_tensor *tensor, const std::vector<T> &data) {
  if (!tensor) {
    throw std::runtime_error("Cannot write to null tensor");
  }

  size_t n_elements = ggml_nelements(tensor);
  if (data.size() != n_elements) {
    throw std::runtime_error("Data size mismatch: expected " +
                             std::to_string(n_elements) + " elements, got " +
                             std::to_string(data.size()));
  }

  // Write to backend buffer
  ggml_backend_tensor_set(tensor, data.data(), 0, n_elements * sizeof(T));
}

// Explicit template instantiations for common types
template void write<float>(ggml_tensor *tensor, const std::vector<float> &data);
template void write<int32_t>(ggml_tensor *tensor,
                             const std::vector<int32_t> &data);
template void write<int64_t>(ggml_tensor *tensor,
                             const std::vector<int64_t> &data);
template void write<uint32_t>(ggml_tensor *tensor,
                              const std::vector<uint32_t> &data);

bool same_shape(const ggml_tensor *a, const ggml_tensor *b) {
  if (!a || !b) {
    return false;
  }

  int n_dims_a = ggml_n_dims(a);
  int n_dims_b = ggml_n_dims(b);

  if (n_dims_a != n_dims_b) {
    return false;
  }

  for (int i = 0; i < n_dims_a; ++i) {
    if (a->ne[i] != b->ne[i]) {
      return false;
    }
  }

  return true;
}

std::string shape_string(const ggml_tensor *tensor) {
  if (!tensor) {
    return "null";
  }

  std::ostringstream oss;
  oss << "[";

  int n_dims = ggml_n_dims(tensor);
  for (int i = 0; i < n_dims; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << tensor->ne[i];
  }

  oss << "]";
  return oss.str();
}

} // namespace ops

} // namespace ggml
} // namespace mlipcpp
