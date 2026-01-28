#pragma once

#include "../core/ggml_utils.h"
#include "graph_ir.h"

#include <ggml.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlipcpp::runtime {

// The graph interpreter builds and executes GGML graphs from GIR
class GraphInterpreter {
public:
  GraphInterpreter() = default;
  ~GraphInterpreter() = default;

  // Load a graph from JSON
  void load_graph(const std::string &json_str);
  void load_graph_file(const std::string &path);

  // Set a runtime dimension value (e.g., "n_atoms" -> 3, "max_neighbors" -> 20)
  // Must be called before build() for graphs with symbolic dimensions
  void set_dimension(const std::string &name, int64_t value);

  // Set a weight tensor (must be called before build)
  void set_weight(const std::string &name, ggml_tensor *tensor);

  // Set an input tensor (must be called before compute)
  void set_input(const std::string &name, ggml_tensor *tensor);

  // Build the GGML computation graph
  // This uses the provided context for allocations
  ggml_tensor *build(ggml_context *ctx);

  // Get the output tensor after build
  ggml_tensor *get_output() const { return output_; }

  // Initialize pending constants (call after graph allocation)
  void init_constants();

  // Get summary of the loaded graph
  std::string summary() const;

  // Check if a graph is loaded
  bool has_graph() const { return !graph_.nodes.empty(); }

  // Get the GIR graph for inspection
  const GIRGraph &graph() const { return graph_; }

  // Debug mode: set output directory for dumping intermediate tensors
  void set_debug_output_dir(const std::string &dir);

  // Callback for tensor inspection during graph building
  using TensorCallback = std::function<void(ggml_tensor *, const char *, int)>;
  void set_tensor_callback(TensorCallback cb) { tensor_cb_ = std::move(cb); }

  // Dump a tensor to the debug directory (after compute)
  void dump_tensor(ggml_tensor *t, const std::string &name, int node_id);

  // Dump all node outputs after compute (call after backend_graph_compute)
  void dump_all_tensors();

private:
  // The IR graph
  GIRGraph graph_;

  // Runtime dimension values (for symbolic dimensions like "n_atoms")
  std::map<std::string, int64_t> dimensions_;

  // Tensor references
  std::map<std::string, ggml_tensor *> weights_;
  std::map<std::string, ggml_tensor *> inputs_;
  std::map<int, ggml_tensor *> node_outputs_; // node_id -> tensor

  // Output tensor
  ggml_tensor *output_ = nullptr;

  // Pending constants to initialize after allocation
  struct PendingConstant {
    ggml_tensor *tensor;
    float value;
  };
  std::vector<PendingConstant> pending_constants_;

  // Debug mode
  bool debug_mode_ = false;
  std::string debug_dir_;
  TensorCallback tensor_cb_ = [](ggml_tensor *, const char *, int) {};

  // Resolve symbolic dimensions in a shape to actual values
  std::vector<int64_t> resolve_shape(const std::vector<int64_t> &shape) const;

  // Build helpers
  ggml_tensor *resolve_input(ggml_context *ctx, const std::string &ref);
  ggml_tensor *build_node(ggml_context *ctx, const GIRNode &node);

  // Operation builders
  ggml_tensor *build_add(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_sub(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_mul(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_div(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_mul_mat(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_reshape(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_view(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_select(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_permute(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_transpose(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_cont(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_scale(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_sqr(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_sqrt(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_log(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_cos(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_sin(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_sum_rows(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_repeat(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_clamp(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_softmax(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_flash_attn(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_unary(ggml_context *ctx, const GIRNode &node,
                           ggml_unary_op op);
  ggml_tensor *build_decompose(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_layer_norm(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_concat(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_get_rows(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_new_zeros(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_new_ones(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_linear(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_slice(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_split(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_bitwise_not(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_index(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_index_put(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_where(ggml_context *ctx, const GIRNode &node);
  ggml_tensor *build_pow(ggml_context *ctx, const GIRNode &node);
};

} // namespace mlipcpp::runtime
