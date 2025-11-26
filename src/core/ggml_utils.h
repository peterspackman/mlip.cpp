#pragma once

#include <cstddef>
#include <ggml-backend.h>
#include <ggml.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mlipcpp {
namespace ggml {

/**
 * @brief RAII wrapper for ggml_context
 *
 * Manages the lifetime of a GGML context, automatically freeing resources
 * when the wrapper goes out of scope. This eliminates manual cleanup and
 * prevents resource leaks.
 *
 * The context can be created with or without memory allocation:
 * - no_alloc=false: Context manages memory for tensor data (for input data)
 * - no_alloc=true: Context only stores tensor metadata (for graph building)
 *
 * @example
 * ```cpp
 * // Create context for input data (allocates memory)
 * Context input_ctx(128 * 1024 * 1024);
 *
 * // Create context for graph building (metadata only)
 * Context graph_ctx(1024 * 1024, true);
 * ```
 */
class Context {
public:
  /**
   * @brief Construct a new GGML context
   *
   * @param mem_size Size of memory to allocate (in bytes)
   * @param no_alloc If true, context only stores metadata (no tensor data
   * allocation)
   * @throws std::runtime_error if context creation fails
   */
  explicit Context(size_t mem_size, bool no_alloc = false);

  /**
   * @brief Destructor - automatically frees the context
   */
  ~Context();

  // No copy (context is unique resource)
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  /**
   * @brief Move constructor
   */
  Context(Context &&other) noexcept;

  /**
   * @brief Move assignment operator
   */
  Context &operator=(Context &&other) noexcept;

  /**
   * @brief Get the underlying ggml_context pointer
   * @return Raw context pointer (never null if object is valid)
   */
  ggml_context *get() const noexcept { return ctx_; }

  /**
   * @brief Implicit conversion to ggml_context* for convenience
   */
  operator ggml_context *() const noexcept { return ctx_; }

  /**
   * @brief Check if context is valid (non-null)
   */
  explicit operator bool() const noexcept { return ctx_ != nullptr; }

private:
  ggml_context *ctx_ = nullptr;
};

/**
 * @brief RAII wrapper for ggml_backend
 *
 * Manages the lifetime of a GGML backend (CPU, Metal, CUDA, etc.),
 * automatically freeing resources when the wrapper goes out of scope.
 *
 * Backends are initialized from device types (CPU, GPU) and handle
 * the execution of computation graphs on specific hardware.
 *
 * @example
 * ```cpp
 * // Create CPU backend
 * Backend cpu_backend(GGML_BACKEND_DEVICE_TYPE_CPU);
 *
 * // Create GPU backend (Metal on macOS)
 * Backend gpu_backend(GGML_BACKEND_DEVICE_TYPE_GPU);
 * ```
 */
class Backend {
public:
  /**
   * @brief Construct a backend from device type
   *
   * @param device_type Type of device (CPU, GPU, etc.)
   * @throws std::runtime_error if backend creation fails
   */
  explicit Backend(enum ggml_backend_dev_type device_type);

  /**
   * @brief Destructor - automatically frees the backend
   */
  ~Backend();

  // No copy (backend is unique resource)
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;

  /**
   * @brief Move constructor
   */
  Backend(Backend &&other) noexcept;

  /**
   * @brief Move assignment operator
   */
  Backend &operator=(Backend &&other) noexcept;

  /**
   * @brief Get the underlying ggml_backend_t pointer
   */
  ggml_backend_t get() const noexcept { return backend_; }

  /**
   * @brief Implicit conversion to ggml_backend_t for convenience
   */
  operator ggml_backend_t() const noexcept { return backend_; }

  /**
   * @brief Get the default buffer type for this backend
   * @return Buffer type suitable for tensor allocation
   */
  ggml_backend_buffer_type_t default_buffer_type() const;

  /**
   * @brief Check if backend is valid (non-null)
   */
  explicit operator bool() const noexcept { return backend_ != nullptr; }

private:
  ggml_backend_t backend_ = nullptr;
};

/**
 * @brief RAII wrapper for ggml_backend_buffer
 *
 * Manages the lifetime of a backend buffer, which holds tensor data
 * on a specific backend (CPU, GPU, etc.). Automatically frees the
 * buffer when the wrapper goes out of scope.
 *
 * Backend buffers are allocated from a context's tensors using a
 * specific buffer type (e.g., CPU, Metal, CUDA).
 *
 * @example
 * ```cpp
 * Backend cpu_backend(GGML_BACKEND_DEVICE_TYPE_CPU);
 * Context weight_ctx(64 * 1024 * 1024, true);
 *
 * // Allocate buffer for weight tensors
 * BackendBuffer weight_buffer(weight_ctx.get(),
 * cpu_backend.default_buffer_type());
 * ```
 */
class BackendBuffer {
public:
  /**
   * @brief Construct a backend buffer from context and buffer type
   *
   * Allocates memory on the backend for all tensors in the context.
   * The context must be created with no_alloc=true.
   *
   * @param ctx GGML context containing tensor metadata
   * @param buf_type Backend buffer type (CPU, Metal, CUDA, etc.)
   * @throws std::runtime_error if buffer allocation fails
   */
  BackendBuffer(ggml_context *ctx, ggml_backend_buffer_type_t buf_type);

  /**
   * @brief Destructor - automatically frees the buffer
   */
  ~BackendBuffer();

  // No copy (buffer is unique resource)
  BackendBuffer(const BackendBuffer &) = delete;
  BackendBuffer &operator=(const BackendBuffer &) = delete;

  /**
   * @brief Move constructor
   */
  BackendBuffer(BackendBuffer &&other) noexcept;

  /**
   * @brief Move assignment operator
   */
  BackendBuffer &operator=(BackendBuffer &&other) noexcept;

  /**
   * @brief Get the underlying ggml_backend_buffer_t pointer
   */
  ggml_backend_buffer_t get() const noexcept { return buffer_; }

  /**
   * @brief Implicit conversion to ggml_backend_buffer_t for convenience
   */
  operator ggml_backend_buffer_t() const noexcept { return buffer_; }

  /**
   * @brief Get the size of the buffer in bytes
   */
  size_t size() const;

  /**
   * @brief Set the usage hint for the buffer
   *
   * @param usage Usage type (e.g., GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
   */
  void set_usage(ggml_backend_buffer_usage usage);

  /**
   * @brief Check if buffer is valid (non-null)
   */
  explicit operator bool() const noexcept { return buffer_ != nullptr; }

private:
  ggml_backend_buffer_t buffer_ = nullptr;
};

/**
 * @brief RAII wrapper for ggml_backend_sched (backend scheduler)
 *
 * Manages the lifetime of a backend scheduler, which orchestrates
 * computation graph execution across multiple backends (CPU, GPU, etc.).
 *
 * The scheduler allocates tensors, assigns operations to backends,
 * and handles data transfers between backends automatically.
 *
 * @example
 * ```cpp
 * std::vector<Backend> backends;
 * backends.emplace_back(GGML_BACKEND_DEVICE_TYPE_GPU);
 * backends.emplace_back(GGML_BACKEND_DEVICE_TYPE_CPU);
 *
 * std::vector<ggml_backend_t> backend_ptrs;
 * std::vector<ggml_backend_buffer_type_t> buffer_types;
 * for (auto& backend : backends) {
 *     backend_ptrs.push_back(backend.get());
 *     buffer_types.push_back(backend.default_buffer_type());
 * }
 *
 * Scheduler sched(backend_ptrs, buffer_types, 4096);
 * ```
 */
class Scheduler {
public:
  /**
   * @brief Construct a backend scheduler
   *
   * @param backends Vector of backend pointers
   * @param buffer_types Vector of buffer types (one per backend)
   * @param max_nodes Maximum number of nodes in computation graph
   * @param parallel Enable parallel execution (experimental)
   * @throws std::runtime_error if scheduler creation fails
   */
  Scheduler(const std::vector<ggml_backend_t> &backends,
            const std::vector<ggml_backend_buffer_type_t> &buffer_types,
            size_t max_nodes, bool parallel = false);

  /**
   * @brief Destructor - automatically frees the scheduler
   */
  ~Scheduler();

  // No copy (scheduler is unique resource)
  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;

  /**
   * @brief Move constructor
   */
  Scheduler(Scheduler &&other) noexcept;

  /**
   * @brief Move assignment operator
   */
  Scheduler &operator=(Scheduler &&other) noexcept;

  /**
   * @brief Get the underlying ggml_backend_sched_t pointer
   */
  ggml_backend_sched_t get() const noexcept { return sched_; }

  /**
   * @brief Implicit conversion to ggml_backend_sched_t for convenience
   */
  operator ggml_backend_sched_t() const noexcept { return sched_; }

  /**
   * @brief Allocate tensors for a computation graph
   *
   * @param graph Computation graph to allocate
   * @return true if allocation succeeded
   */
  bool alloc_graph(ggml_cgraph *graph);

  /**
   * @brief Compute a graph using the scheduler
   *
   * @param graph Computation graph to execute
   * @return GGML status code
   */
  ggml_status compute_graph(ggml_cgraph *graph);

  /**
   * @brief Check if scheduler is valid (non-null)
   */
  explicit operator bool() const noexcept { return sched_ != nullptr; }

private:
  ggml_backend_sched_t sched_ = nullptr;
};

/**
 * @brief Common tensor operations
 *
 * Provides convenient wrappers for frequently-used GGML tensor operations
 * with better error handling and cleaner APIs.
 */
namespace ops {

/**
 * @brief Duplicate tensor into target context
 *
 * Creates a copy operation in the computation graph. The actual data
 * copy happens during graph execution.
 *
 * @param ctx Target context for the duplicated tensor
 * @param src Source tensor to duplicate
 * @return Tensor in target context (operation node, not actual copy)
 */
ggml_tensor *dup(ggml_context *ctx, ggml_tensor *src);

/**
 * @brief Read tensor data from backend buffer
 *
 * Reads tensor data from backend memory (GPU, CPU, etc.) into a
 * host-side vector. Handles type conversion automatically.
 *
 * @tparam T Element type (float, int32_t, etc.)
 * @param tensor Tensor to read from
 * @return Vector containing tensor data in row-major order
 * @throws std::runtime_error if read fails or type mismatch
 *
 * @example
 * ```cpp
 * auto energies = ops::read<float>(energy_tensor);
 * auto indices = ops::read<int32_t>(index_tensor);
 * ```
 */
template <typename T> std::vector<T> read(ggml_tensor *tensor);

/**
 * @brief Write tensor data to backend buffer
 *
 * Writes data from a host-side vector to backend memory (GPU, CPU, etc.).
 * Handles type conversion automatically.
 *
 * @tparam T Element type (float, int32_t, etc.)
 * @param tensor Tensor to write to
 * @param data Vector containing data to write (row-major order)
 * @throws std::runtime_error if write fails or type mismatch
 *
 * @example
 * ```cpp
 * std::vector<float> weights = {...};
 * ops::write(weight_tensor, weights);
 * ```
 */
template <typename T>
void write(ggml_tensor *tensor, const std::vector<T> &data);

/**
 * @brief Get number of elements in a tensor
 *
 * @param tensor Tensor to query
 * @return Total number of elements (product of all dimensions)
 */
inline size_t nelements(const ggml_tensor *tensor) {
  return ggml_nelements(tensor);
}

/**
 * @brief Get size of a tensor in bytes
 *
 * @param tensor Tensor to query
 * @return Size in bytes
 */
inline size_t nbytes(const ggml_tensor *tensor) { return ggml_nbytes(tensor); }

/**
 * @brief Check if two tensors have the same shape
 *
 * @param a First tensor
 * @param b Second tensor
 * @return true if all dimensions match
 */
bool same_shape(const ggml_tensor *a, const ggml_tensor *b);

/**
 * @brief Get human-readable shape string
 *
 * @param tensor Tensor to describe
 * @return String like "[256, 128, 8]"
 */
std::string shape_string(const ggml_tensor *tensor);

} // namespace ops

} // namespace ggml
} // namespace mlipcpp
