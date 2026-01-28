#include "graph_interpreter.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mlipcpp::runtime {

namespace {

// Parse a reference string like "node:5", "input:name", "weight:name", "const:value"
struct RefParsed {
  std::string type; // "node", "input", "weight", "const"
  std::string value;
};

RefParsed parse_ref(const std::string &ref) {
  auto colon_pos = ref.find(':');
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("Invalid reference format: " + ref);
  }
  return {ref.substr(0, colon_pos), ref.substr(colon_pos + 1)};
}

// Check if a parameter exists in a node's params map
bool has_param(const GIRNode &node, const std::string &key) {
  return node.params.find(key) != node.params.end();
}

// Get a parameter from a node's params map
template <typename T>
T get_param(const GIRNode &node, const std::string &key, T default_val) {
  auto it = node.params.find(key);
  if (it == node.params.end()) {
    return default_val;
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    if (std::holds_alternative<int64_t>(it->second)) {
      return std::get<int64_t>(it->second);
    }
    if (std::holds_alternative<double>(it->second)) {
      return static_cast<int64_t>(std::get<double>(it->second));
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if (std::holds_alternative<double>(it->second)) {
      return std::get<double>(it->second);
    }
    if (std::holds_alternative<int64_t>(it->second)) {
      return static_cast<double>(std::get<int64_t>(it->second));
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    if (std::holds_alternative<bool>(it->second)) {
      return std::get<bool>(it->second);
    }
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (std::holds_alternative<std::string>(it->second)) {
      return std::get<std::string>(it->second);
    }
  }
  return default_val;
}

// Get int array parameter
std::vector<int64_t> get_int_array_param(const GIRNode &node,
                                         const std::string &key) {
  auto it = node.params.find(key);
  if (it == node.params.end()) {
    return {};
  }
  if (std::holds_alternative<std::vector<int64_t>>(it->second)) {
    return std::get<std::vector<int64_t>>(it->second);
  }
  return {};
}

} // namespace

void GraphInterpreter::load_graph(const std::string &json_str) {
  graph_ = parse_gir_json(json_str);
  node_outputs_.clear();
  pending_constants_.clear();
  output_ = nullptr;
}

void GraphInterpreter::load_graph_file(const std::string &path) {
  graph_ = load_gir_file(path);
  node_outputs_.clear();
  pending_constants_.clear();
  output_ = nullptr;
}

void GraphInterpreter::set_dimension(const std::string &name, int64_t value) {
  dimensions_[name] = value;
}

std::vector<int64_t> GraphInterpreter::resolve_shape(
    const std::vector<int64_t> &shape) const {
  std::vector<int64_t> resolved;
  resolved.reserve(shape.size());

  for (int64_t dim : shape) {
    if (dim == DIM_N_ATOMS) {
      auto it = dimensions_.find("n_atoms");
      if (it == dimensions_.end()) {
        throw std::runtime_error(
            "Symbolic dimension 'n_atoms' used but not set. "
            "Call set_dimension(\"n_atoms\", value) before build().");
      }
      resolved.push_back(it->second);
    } else if (dim == DIM_MAX_NEIGHBORS) {
      auto it = dimensions_.find("max_neighbors");
      if (it == dimensions_.end()) {
        throw std::runtime_error(
            "Symbolic dimension 'max_neighbors' used but not set. "
            "Call set_dimension(\"max_neighbors\", value) before build().");
      }
      resolved.push_back(it->second);
    } else if (dim == DIM_SEQ_LEN) {
      // seq_len = n_atoms * (max_neighbors + 1)
      auto it_n = dimensions_.find("n_atoms");
      auto it_m = dimensions_.find("max_neighbors");
      if (it_n == dimensions_.end() || it_m == dimensions_.end()) {
        throw std::runtime_error(
            "Symbolic dimension 'seq_len' requires both 'n_atoms' and "
            "'max_neighbors' to be set.");
      }
      resolved.push_back(it_n->second * (it_m->second + 1));
    } else if (dim == DIM_N_EDGES) {
      // n_edges = n_atoms * max_neighbors
      auto it_n = dimensions_.find("n_atoms");
      auto it_m = dimensions_.find("max_neighbors");
      if (it_n == dimensions_.end() || it_m == dimensions_.end()) {
        throw std::runtime_error(
            "Symbolic dimension 'n_edges' requires both 'n_atoms' and "
            "'max_neighbors' to be set.");
      }
      resolved.push_back(it_n->second * it_m->second);
    } else if (dim == DIM_MN_PLUS_ONE) {
      // max_neighbors_plus_one = max_neighbors + 1
      auto it_m = dimensions_.find("max_neighbors");
      if (it_m == dimensions_.end()) {
        throw std::runtime_error(
            "Symbolic dimension 'max_neighbors_plus_one' requires "
            "'max_neighbors' to be set.");
      }
      resolved.push_back(it_m->second + 1);
    } else {
      // Regular concrete dimension
      resolved.push_back(dim);
    }
  }

  return resolved;
}

void GraphInterpreter::set_weight(const std::string &name,
                                  ggml_tensor *tensor) {
  weights_[name] = tensor;
}

void GraphInterpreter::set_input(const std::string &name, ggml_tensor *tensor) {
  inputs_[name] = tensor;
}

void GraphInterpreter::init_constants() {
  // Set constant values after graph allocation
  for (const auto &pc : pending_constants_) {
    if (pc.tensor && pc.tensor->data) {
      // Fill ALL elements of the tensor with the constant value
      float *data = static_cast<float *>(pc.tensor->data);
      size_t n_elements = ggml_nelements(pc.tensor);
      for (size_t i = 0; i < n_elements; i++) {
        data[i] = pc.value;
      }
    }
  }
}

ggml_tensor *GraphInterpreter::resolve_input(ggml_context *ctx,
                                             const std::string &ref) {
  auto parsed = parse_ref(ref);

  if (parsed.type == "node") {
    int node_id = std::stoi(parsed.value);
    auto it = node_outputs_.find(node_id);
    if (it == node_outputs_.end()) {
      throw std::runtime_error("Node " + parsed.value + " not yet computed");
    }
    return it->second;
  } else if (parsed.type == "input") {
    auto it = inputs_.find(parsed.value);
    if (it == inputs_.end()) {
      throw std::runtime_error("Input not found: " + parsed.value);
    }
    return it->second;
  } else if (parsed.type == "weight") {
    auto it = weights_.find(parsed.value);
    if (it == weights_.end()) {
      throw std::runtime_error("Weight not found: " + parsed.value);
    }
    return it->second;
  } else if (parsed.type == "const") {
    // Constants are typically parameters, not tensors
    // For now, create a scalar constant tensor and mark as input
    // The value will need to be set after allocation
    float value = std::stof(parsed.value);
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t);
    // Store for later initialization
    pending_constants_.push_back({t, value});
    return t;
  } else {
    throw std::runtime_error("Unknown reference type: " + parsed.type);
  }
}

ggml_tensor *GraphInterpreter::build(ggml_context *ctx) {
  if (graph_.nodes.empty()) {
    throw std::runtime_error("No graph loaded");
  }

  node_outputs_.clear();

  // Build nodes in order (they should already be topologically sorted)
  for (const auto &node : graph_.nodes) {
    ggml_tensor *output = nullptr;
    try {
      output = build_node(ctx, node);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed building node " +
                               std::to_string(node.id) + " (" + node.op +
                               " \"" + node.name + "\"): " + e.what());
    }
    if (output) {
      node_outputs_[node.id] = output;
      if (!node.name.empty()) {
        ggml_set_name(output, node.name.c_str());
      }
    }
  }

  // Find the output tensor
  if (!graph_.outputs.empty()) {
    auto parsed = parse_ref(graph_.outputs[0].node_ref);
    if (parsed.type == "node") {
      int node_id = std::stoi(parsed.value);
      auto it = node_outputs_.find(node_id);
      if (it != node_outputs_.end()) {
        output_ = it->second;
      }
    }
  }

  // If no explicit output, use the last node
  if (!output_ && !graph_.nodes.empty()) {
    output_ = node_outputs_[graph_.nodes.back().id];
  }

  return output_;
}

ggml_tensor *GraphInterpreter::build_node(ggml_context *ctx,
                                          const GIRNode &node) {
  // Dispatch based on operation type
  if (node.op == "ADD") {
    return build_add(ctx, node);
  } else if (node.op == "SUB") {
    return build_sub(ctx, node);
  } else if (node.op == "MUL") {
    return build_mul(ctx, node);
  } else if (node.op == "DIV") {
    return build_div(ctx, node);
  } else if (node.op == "MUL_MAT") {
    return build_mul_mat(ctx, node);
  } else if (node.op == "RESHAPE") {
    return build_reshape(ctx, node);
  } else if (node.op == "VIEW") {
    return build_view(ctx, node);
  } else if (node.op == "SELECT") {
    return build_select(ctx, node);
  } else if (node.op == "PERMUTE") {
    return build_permute(ctx, node);
  } else if (node.op == "TRANSPOSE") {
    return build_transpose(ctx, node);
  } else if (node.op == "CONT") {
    return build_cont(ctx, node);
  } else if (node.op == "SCALE") {
    return build_scale(ctx, node);
  } else if (node.op == "SQR") {
    return build_sqr(ctx, node);
  } else if (node.op == "SQRT") {
    return build_sqrt(ctx, node);
  } else if (node.op == "LOG") {
    return build_log(ctx, node);
  } else if (node.op == "SUM_ROWS") {
    return build_sum_rows(ctx, node);
  } else if (node.op == "REPEAT") {
    return build_repeat(ctx, node);
  } else if (node.op == "CLAMP") {
    return build_clamp(ctx, node);
  } else if (node.op == "SOFT_MAX") {
    return build_softmax(ctx, node);
  } else if (node.op == "FLASH_ATTN_EXT") {
    return build_flash_attn(ctx, node);
  } else if (node.op == "UNARY_SILU") {
    return build_unary(ctx, node, GGML_UNARY_OP_SILU);
  } else if (node.op == "UNARY_RELU") {
    return build_unary(ctx, node, GGML_UNARY_OP_RELU);
  } else if (node.op == "UNARY_GELU") {
    return build_unary(ctx, node, GGML_UNARY_OP_GELU);
  } else if (node.op == "UNARY_TANH") {
    return build_unary(ctx, node, GGML_UNARY_OP_TANH);
  } else if (node.op == "UNARY_EXP") {
    return build_unary(ctx, node, GGML_UNARY_OP_EXP);
  } else if (node.op == "UNARY_NEG") {
    return build_unary(ctx, node, GGML_UNARY_OP_NEG);
  } else if (node.op == "DECOMPOSE") {
    return build_decompose(ctx, node);
  } else if (node.op == "LAYER_NORM") {
    return build_layer_norm(ctx, node);
  } else if (node.op == "CONCAT") {
    return build_concat(ctx, node);
  } else if (node.op == "GET_ROWS") {
    return build_get_rows(ctx, node);
  } else if (node.op == "NEW_ZEROS") {
    return build_new_zeros(ctx, node);
  } else if (node.op == "NEW_ONES") {
    return build_new_ones(ctx, node);
  } else if (node.op == "LINEAR") {
    return build_linear(ctx, node);
  } else if (node.op == "SLICE") {
    return build_slice(ctx, node);
  } else if (node.op == "SPLIT") {
    return build_split(ctx, node);
  } else if (node.op == "BITWISE_NOT") {
    return build_bitwise_not(ctx, node);
  } else if (node.op == "INDEX") {
    return build_index(ctx, node);
  } else if (node.op == "INDEX_PUT") {
    return build_index_put(ctx, node);
  } else if (node.op == "WHERE") {
    return build_where(ctx, node);
  } else {
    throw std::runtime_error("Unknown operation: " + node.op);
  }
}

// ===================== Binary Operations =====================

ggml_tensor *GraphInterpreter::build_add(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("ADD requires at least 1 input at node: " + node.name);
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  if (node.inputs.size() == 1) {
    // Single input ADD is identity (e.g., from torch.zeros() + x optimization)
    return a;
  }
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_add(ctx, a, b);
}

ggml_tensor *GraphInterpreter::build_sub(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.size() < 2) {
    throw std::runtime_error("SUB requires 2 inputs");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_sub(ctx, a, b);
}

ggml_tensor *GraphInterpreter::build_mul(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("MUL requires at least 1 input");
  }

  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // Check for scalar multiplication (tensor * scalar)
  if (node.inputs.size() == 1 && has_param(node, "scalar")) {
    float scalar = static_cast<float>(get_param<double>(node, "scalar", 1.0));
    return ggml_scale(ctx, a, scalar);
  }

  // Standard element-wise multiplication
  if (node.inputs.size() < 2) {
    throw std::runtime_error("MUL requires 2 inputs (or 1 input with scalar param)");
  }
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_mul(ctx, a, b);
}

ggml_tensor *GraphInterpreter::build_div(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.size() < 2) {
    throw std::runtime_error("DIV requires 2 inputs");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_div(ctx, a, b);
}

// ===================== Matrix Operations =====================

ggml_tensor *GraphInterpreter::build_mul_mat(ggml_context *ctx,
                                             const GIRNode &node) {
  if (node.inputs.size() < 2) {
    throw std::runtime_error("MUL_MAT requires 2 inputs");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_mul_mat(ctx, a, b);
}

// ===================== Shape Operations =====================

ggml_tensor *GraphInterpreter::build_reshape(ggml_context *ctx,
                                             const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("RESHAPE requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // GGML reshape requires contiguous input - make contiguous if needed
  if (!ggml_is_contiguous(a)) {
    a = ggml_cont(ctx, a);
  }

  auto shape = get_int_array_param(node, "shape");

  // Prefer output_shape if it's more complete (FX export may have partial
  // params)
  if (shape.empty() || (!node.output_shape.empty() &&
                        node.output_shape.size() > shape.size())) {
    shape = node.output_shape;
  }

  if (shape.empty()) {
    throw std::runtime_error("RESHAPE: no shape available");
  }

  // Resolve symbolic dimensions (e.g., DIM_N_ATOMS -> actual n_atoms value)
  shape = resolve_shape(shape);

  // Reverse shape (Python→GGML dimension order)
  std::reverse(shape.begin(), shape.end());

  // Verify element count matches
  int64_t target_nelements = 1;
  for (auto d : shape) {
    target_nelements *= d;
  }
  int64_t actual_nelements = ggml_nelements(a);
  if (actual_nelements != target_nelements) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0) shape_str += ", ";
      shape_str += std::to_string(shape[i]);
    }
    shape_str += "]";
    throw std::runtime_error(
        "RESHAPE: element count mismatch for node '" + node.name +
        "': input has " + std::to_string(actual_nelements) +
        " elements, target shape " + shape_str + " needs " +
        std::to_string(target_nelements) + " elements");
  }

  switch (shape.size()) {
  case 1:
    return ggml_reshape_1d(ctx, a, shape[0]);
  case 2:
    return ggml_reshape_2d(ctx, a, shape[0], shape[1]);
  case 3:
    return ggml_reshape_3d(ctx, a, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_reshape_4d(ctx, a, shape[0], shape[1], shape[2], shape[3]);
  default:
    throw std::runtime_error("RESHAPE: unsupported number of dimensions: " +
                             std::to_string(shape.size()));
  }
}

ggml_tensor *GraphInterpreter::build_view(ggml_context *ctx,
                                          const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("VIEW requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  auto shape = get_int_array_param(node, "shape");
  auto index = get_param<int64_t>(node, "index", -1);

  // Prefer output_shape if it has fully resolved dimensions (no -1)
  // But use params.shape if we need to know the original intent
  bool has_negative = false;
  for (auto d : shape) {
    if (d < 0) has_negative = true;
  }

  // Use output_shape which has resolved dimensions
  if (has_negative || shape.empty()) {
    if (!node.output_shape.empty()) {
      shape = node.output_shape;
    }
  }

  // If still empty, this might be an indexing operation - pass through input
  if (shape.empty()) {
    // VIEW with no shape is used for getitem[index] - just pass through
    return a;
  }

  // Resolve symbolic dimensions (e.g., DIM_N_ATOMS -> actual n_atoms value)
  shape = resolve_shape(shape);

  // Reverse shape (Python→GGML dimension order)
  std::reverse(shape.begin(), shape.end());

  // If this is a chunk extraction (getitem with index), use view with offset
  if (index >= 0) {
    // Chunk offset: index * chunk_size * element_size
    size_t byte_offset = static_cast<size_t>(index) * shape[0] * ggml_element_size(a);

    ggml_tensor *view = nullptr;
    switch (shape.size()) {
    case 1:
      view = ggml_view_1d(ctx, a, shape[0], byte_offset);
      break;
    case 2:
      view = ggml_view_2d(ctx, a, shape[0], shape[1], a->nb[1], byte_offset);
      break;
    case 3:
      view = ggml_view_3d(ctx, a, shape[0], shape[1], shape[2],
                          a->nb[1], a->nb[2], byte_offset);
      break;
    case 4:
      view = ggml_view_4d(ctx, a, shape[0], shape[1], shape[2], shape[3],
                          a->nb[1], a->nb[2], a->nb[3], byte_offset);
      break;
    default:
      throw std::runtime_error("VIEW: unsupported number of dimensions");
    }
    // Make contiguous so subsequent reshapes work correctly
    return ggml_cont(ctx, view);
  }

  // For regular view/reshape (no chunk extraction), use reshape which is safer
  // when changing dimensionality. GGML reshape just reinterprets the same memory.
  switch (shape.size()) {
  case 1:
    return ggml_reshape_1d(ctx, a, shape[0]);
  case 2:
    return ggml_reshape_2d(ctx, a, shape[0], shape[1]);
  case 3:
    return ggml_reshape_3d(ctx, a, shape[0], shape[1], shape[2]);
  case 4:
    return ggml_reshape_4d(ctx, a, shape[0], shape[1], shape[2], shape[3]);
  default:
    throw std::runtime_error("VIEW: unsupported number of dimensions: " +
                             std::to_string(shape.size()));
  }
}

ggml_tensor *GraphInterpreter::build_select(ggml_context *ctx,
                                            const GIRNode &node) {
  // SELECT: extract one slice from a dimension, reducing dimensionality by 1
  // PyTorch: x[:, idx, :] on [N, S, D] -> [N, D]
  // PyTorch: x[:, :, idx, :] on [B, S, 3, D] -> [B, S, D]
  if (node.inputs.empty()) {
    throw std::runtime_error("SELECT requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // Get select dimension (PyTorch convention) and index
  int py_dim = static_cast<int>(get_param<int64_t>(node, "dim", 1));
  int64_t idx = get_param<int64_t>(node, "index", 0);

  // Use output_shape to determine expected output dimensionality
  // This is more reliable than ggml_n_dims which may compress dimensions
  int n_dims_output = static_cast<int>(node.output_shape.size());
  int n_dims_input = n_dims_output + 1;  // SELECT removes one dimension

  // Convert PyTorch dim to GGML dim (reversed order)
  // Use expected input dims, not GGML's compressed dims
  int ggml_dim = n_dims_input - 1 - py_dim;

  if (ggml_dim < 0 || ggml_dim >= n_dims_input) {
    throw std::runtime_error("SELECT: invalid dimension: py_dim=" +
                             std::to_string(py_dim) + " for " +
                             std::to_string(n_dims_input) + "D input");
  }

  // Calculate byte offset to the selected slice
  size_t offset = static_cast<size_t>(idx) * a->nb[ggml_dim];

  // Handle each case based on input dimensions and which dim we're selecting
  if (n_dims_input == 4) {
    // 4D tensor [ne0, ne1, ne2, ne3] in GGML order
    // PyTorch shape is [ne3, ne2, ne1, ne0]
    if (ggml_dim == 1) {
      // PyTorch dim=2: select from ne1, result is [ne0, ne2, ne3]
      // View with stride that skips over ne1
      return ggml_view_3d(ctx, a, a->ne[0], a->ne[2], a->ne[3],
                          a->nb[2], a->nb[3], offset);
    } else if (ggml_dim == 2) {
      // PyTorch dim=1: select from ne2, result is [ne0, ne1, ne3]
      return ggml_view_3d(ctx, a, a->ne[0], a->ne[1], a->ne[3],
                          a->nb[1], a->nb[3], offset);
    } else if (ggml_dim == 3) {
      // PyTorch dim=0: select from ne3, result is [ne0, ne1, ne2]
      return ggml_view_3d(ctx, a, a->ne[0], a->ne[1], a->ne[2],
                          a->nb[1], a->nb[2], offset);
    } else if (ggml_dim == 0) {
      // PyTorch dim=3: select from ne0, result is [ne1, ne2, ne3]
      // This is selecting a single element from the innermost dimension
      return ggml_view_3d(ctx, a, a->ne[1], a->ne[2], a->ne[3],
                          a->nb[2], a->nb[3], idx * a->nb[0]);
    }
  } else if (n_dims_input == 3) {
    if (ggml_dim == 1) {
      // Selecting from middle dimension of 3D: [ne0, ne1, ne2] -> [ne0, ne2]
      return ggml_view_2d(ctx, a, a->ne[0], a->ne[2], a->nb[2], offset);
    } else if (ggml_dim == 2) {
      // Selecting from last dimension (PyTorch first): [ne0, ne1, ne2] -> [ne0, ne1]
      return ggml_view_2d(ctx, a, a->ne[0], a->ne[1], a->nb[1], offset);
    } else if (ggml_dim == 0) {
      // Selecting from first dimension (PyTorch last): [ne0, ne1, ne2] -> [ne1, ne2]
      return ggml_view_2d(ctx, a, a->ne[1], a->ne[2], a->nb[2], idx * a->nb[0]);
    }
  } else if (n_dims_input == 2) {
    if (ggml_dim == 1) {
      // 2D selecting from PyTorch dim 0: [ne0, ne1] -> [ne0]
      return ggml_view_1d(ctx, a, a->ne[0], offset);
    } else if (ggml_dim == 0) {
      // 2D selecting from PyTorch dim 1: [ne0, ne1] -> [ne1]
      return ggml_view_1d(ctx, a, a->ne[1], idx * a->nb[0]);
    }
  }

  throw std::runtime_error("SELECT: unsupported dimension configuration: " +
                           std::to_string(n_dims_input) + "D input, ggml_dim=" +
                           std::to_string(ggml_dim));
}

ggml_tensor *GraphInterpreter::build_permute(ggml_context *ctx,
                                             const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("PERMUTE requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  auto axes = get_int_array_param(node, "axes");

  if (axes.size() != 4) {
    // Pad to 4 dimensions
    while (axes.size() < 4) {
      axes.push_back(axes.size());
    }
  }

  // Convert from Python axis order to GGML
  // In Python: [0,1,2,3] means [batch, channel, height, width]
  // In GGML: [0,1,2,3] means [width, height, channel, batch]
  // So we need to reverse the axis mapping
  int n_dims = static_cast<int>(axes.size());
  std::vector<int> ggml_axes(4);
  for (int i = 0; i < n_dims; i++) {
    ggml_axes[n_dims - 1 - i] = n_dims - 1 - static_cast<int>(axes[i]);
  }

  return ggml_permute(ctx, a, ggml_axes[0], ggml_axes[1], ggml_axes[2],
                      ggml_axes[3]);
}

ggml_tensor *GraphInterpreter::build_transpose(ggml_context *ctx,
                                               const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("TRANSPOSE requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // Get PyTorch dimensions to transpose (defaults to [0, 1] for simple transpose)
  auto dims = get_int_array_param(node, "dims");
  if (dims.empty() || dims.size() != 2) {
    // Default to swapping dims 0 and 1
    return ggml_transpose(ctx, a);
  }

  int64_t py_dim0 = dims[0];
  int64_t py_dim1 = dims[1];
  int n_dims = ggml_n_dims(a);

  // Convert PyTorch dims to GGML dims (reversed order)
  // PyTorch dim i -> GGML dim (n_dims - 1 - i)
  int ggml_dim0 = n_dims - 1 - static_cast<int>(py_dim0);
  int ggml_dim1 = n_dims - 1 - static_cast<int>(py_dim1);

  // Build permutation array - start with identity
  int perm[4] = {0, 1, 2, 3};
  // Swap the two dimensions
  std::swap(perm[ggml_dim0], perm[ggml_dim1]);

  return ggml_permute(ctx, a, perm[0], perm[1], perm[2], perm[3]);
}

ggml_tensor *GraphInterpreter::build_cont(ggml_context *ctx,
                                          const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("CONT requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  return ggml_cont(ctx, a);
}

// ===================== Unary Operations =====================

ggml_tensor *GraphInterpreter::build_scale(ggml_context *ctx,
                                           const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("SCALE requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  float scale = static_cast<float>(get_param<double>(node, "scale", 1.0));
  return ggml_scale(ctx, a, scale);
}

ggml_tensor *GraphInterpreter::build_sqr(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("SQR requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  return ggml_sqr(ctx, a);
}

ggml_tensor *GraphInterpreter::build_sqrt(ggml_context *ctx,
                                          const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("SQRT requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  return ggml_sqrt(ctx, a);
}

ggml_tensor *GraphInterpreter::build_log(ggml_context *ctx,
                                         const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("LOG requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  return ggml_log(ctx, a);
}

// ===================== Reduction Operations =====================

ggml_tensor *GraphInterpreter::build_sum_rows(ggml_context *ctx,
                                              const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("SUM_ROWS requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *result = ggml_sum_rows(ctx, a);

  // ggml_sum_rows reduces ne[0] to 1, producing shape [1, ne[1], ...].
  // If the expected output has fewer dimensions (e.g., [n_atoms] instead of
  // [1, n_atoms]), reshape to squeeze the leading dimension.
  if (!node.output_shape.empty()) {
    auto expected = resolve_shape(node.output_shape);
    std::reverse(expected.begin(), expected.end()); // PyTorch -> GGML order

    // Check if we need to reshape
    bool needs_reshape = (expected.size() < static_cast<size_t>(ggml_n_dims(result)));
    if (!needs_reshape) {
      // Also check if shapes differ (e.g., [1, n] vs [n])
      for (size_t i = 0; i < expected.size(); i++) {
        if (expected[i] != result->ne[i]) {
          needs_reshape = true;
          break;
        }
      }
    }

    if (needs_reshape) {
      switch (expected.size()) {
      case 1:
        result = ggml_reshape_1d(ctx, result, expected[0]);
        break;
      case 2:
        result = ggml_reshape_2d(ctx, result, expected[0], expected[1]);
        break;
      case 3:
        result = ggml_reshape_3d(ctx, result, expected[0], expected[1],
                                  expected[2]);
        break;
      default:
        break;
      }
    }
  }

  return result;
}

// ===================== Other Operations =====================

ggml_tensor *GraphInterpreter::build_repeat(ggml_context *ctx,
                                            const GIRNode &node) {
  if (node.inputs.size() < 2) {
    // Need a template tensor for repeat
    // If not provided, create one from output_shape
    ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
    auto shape = resolve_shape(node.output_shape);
    std::reverse(shape.begin(), shape.end());

    // Create a dummy tensor with target shape
    ggml_tensor *b = nullptr;
    switch (shape.size()) {
    case 1:
      b = ggml_new_tensor_1d(ctx, a->type, shape[0]);
      break;
    case 2:
      b = ggml_new_tensor_2d(ctx, a->type, shape[0], shape[1]);
      break;
    case 3:
      b = ggml_new_tensor_3d(ctx, a->type, shape[0], shape[1], shape[2]);
      break;
    case 4:
      b = ggml_new_tensor_4d(ctx, a->type, shape[0], shape[1], shape[2],
                             shape[3]);
      break;
    default:
      throw std::runtime_error("REPEAT: unsupported number of dimensions");
    }
    return ggml_repeat(ctx, a, b);
  }

  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *b = resolve_input(ctx, node.inputs[1]);
  return ggml_repeat(ctx, a, b);
}

ggml_tensor *GraphInterpreter::build_clamp(ggml_context *ctx,
                                           const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("CLAMP requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  float min_val = static_cast<float>(get_param<double>(node, "min", -INFINITY));
  float max_val = static_cast<float>(get_param<double>(node, "max", INFINITY));
  return ggml_clamp(ctx, a, min_val, max_val);
}

ggml_tensor *GraphInterpreter::build_softmax(ggml_context *ctx,
                                             const GIRNode &node) {
  if (node.inputs.empty()) {
    throw std::runtime_error("SOFT_MAX requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // Check for mask input
  ggml_tensor *mask = nullptr;
  if (node.inputs.size() > 1 && node.inputs[1] != "null") {
    mask = resolve_input(ctx, node.inputs[1]);
  }

  float scale = static_cast<float>(get_param<double>(node, "scale", 1.0));
  return ggml_soft_max_ext(ctx, a, mask, scale, 0.0f);
}

ggml_tensor *GraphInterpreter::build_flash_attn(ggml_context *ctx,
                                                const GIRNode &node) {
  if (node.inputs.size() < 3) {
    throw std::runtime_error("FLASH_ATTN_EXT requires at least 3 inputs (Q, K, V)");
  }

  ggml_tensor *q = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *k = resolve_input(ctx, node.inputs[1]);
  ggml_tensor *v = resolve_input(ctx, node.inputs[2]);

  // Flash attention requires contiguous Q, K, V tensors
  // Add ggml_cont if tensors are not contiguous (e.g., after transpose)
  if (!ggml_is_contiguous(q)) {
    q = ggml_cont(ctx, q);
  }
  if (!ggml_is_contiguous(k)) {
    k = ggml_cont(ctx, k);
  }
  if (!ggml_is_contiguous(v)) {
    v = ggml_cont(ctx, v);
  }

  // Optional mask
  ggml_tensor *mask = nullptr;
  if (node.inputs.size() > 3 && node.inputs[3] != "null") {
    mask = resolve_input(ctx, node.inputs[3]);

    // Ensure mask is contiguous for ggml_add
    if (!ggml_is_contiguous(mask)) {
      mask = ggml_cont(ctx, mask);
    }
  }

  // Get scale parameter, or compute from head dimension (GGML Q shape is [head_dim, ...])
  float scale;
  if (has_param(node, "scale")) {
    scale = static_cast<float>(get_param<double>(node, "scale", 1.0));
  } else {
    // PyTorch SDPA default: 1/sqrt(head_dim)
    int64_t head_dim = q->ne[0];  // head_dim is first GGML dimension
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }
  // Use ggml_flash_attn_ext.
  // Q, K, V are all [head_dim, seq, heads, batch] in GGML order.
  //
  // flash_attn_ext requires:
  // 1. mask ne[1] padded to GGML_KQ_MASK_PAD (64)
  // 2. mask in F16 format (the kernel reads mask data as ggml_fp16_t)
  if (mask) {
    int64_t seq_q = q->ne[1];
    int64_t seq_q_pad = GGML_PAD(seq_q, GGML_KQ_MASK_PAD);

    if (seq_q_pad != mask->ne[1]) {
      mask = ggml_pad(ctx, mask, 0, static_cast<int>(seq_q_pad - mask->ne[1]), 0, 0);
    }
    if (mask->type != GGML_TYPE_F16) {
      mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
    }
  }

  ggml_tensor *result = ggml_flash_attn_ext(ctx, q, k, v, mask,
                                             scale, 0.0f, 0.0f);
  ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);

  // flash_attn_ext output is [head_dim, heads, seq, batch] (permuted).
  // The graph expects [head_dim, seq, heads, batch], so swap dims 1 and 2.
  result = ggml_permute(ctx, result, 0, 2, 1, 3);

  return result;
}

ggml_tensor *GraphInterpreter::build_unary(ggml_context *ctx,
                                           const GIRNode &node,
                                           ggml_unary_op op) {
  if (node.inputs.empty()) {
    throw std::runtime_error("Unary operation requires at least 1 input");
  }
  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  return ggml_unary(ctx, a, op);
}

// ===================== Decomposition Operations =====================

ggml_tensor *GraphInterpreter::build_decompose(ggml_context *ctx,
                                               const GIRNode &node) {
  // Check the node name to determine what decomposition to apply
  // For now, we handle layer_norm (norm_attention, norm_mlp)
  if (node.name.find("norm") != std::string::npos) {
    return build_layer_norm(ctx, node);
  }

  // For unknown decompositions, just pass through the first input
  if (node.inputs.empty()) {
    throw std::runtime_error("DECOMPOSE requires at least 1 input");
  }
  return resolve_input(ctx, node.inputs[0]);
}

ggml_tensor *GraphInterpreter::build_layer_norm(ggml_context *ctx,
                                                const GIRNode &node) {
  // Layer norm inputs:
  // FX style (3 inputs): [input, weight, bias]
  // TS style (4 inputs): [input, normalized_shape, weight, bias]
  // params.eps = epsilon
  if (node.inputs.size() < 3) {
    throw std::runtime_error(
        "LAYER_NORM requires at least 3 inputs (input, weight, bias)");
  }

  ggml_tensor *input = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *weight = nullptr;
  ggml_tensor *bias = nullptr;

  if (node.inputs.size() == 3) {
    // FX style: [input, weight, bias]
    weight = resolve_input(ctx, node.inputs[1]);
    bias = resolve_input(ctx, node.inputs[2]);
  } else {
    // TS style: [input, shape, weight, bias]
    weight = resolve_input(ctx, node.inputs[2]);
    bias = resolve_input(ctx, node.inputs[3]);
  }

  float eps = static_cast<float>(get_param<double>(node, "eps", 1e-5));

  // Use GGML's norm operation (normalizes over the last dimension)
  ggml_tensor *normalized = ggml_norm(ctx, input, eps);

  // Apply affine transformation: output = normalized * weight + bias
  ggml_tensor *scaled = ggml_mul(ctx, normalized, weight);
  return ggml_add(ctx, scaled, bias);
}

ggml_tensor *GraphInterpreter::build_concat(ggml_context *ctx,
                                            const GIRNode &node) {
  // CONCAT: concatenate tensors along a dimension
  // inputs: [tensor1, tensor2, ...] (at least 2)
  // params.dim: dimension to concatenate along (PyTorch convention)
  if (node.inputs.size() < 2) {
    throw std::runtime_error("CONCAT requires at least 2 inputs");
  }

  // Get PyTorch dimension (defaults to 0)
  int py_dim = static_cast<int>(get_param<int64_t>(node, "dim", 0));

  // Resolve all input tensors
  std::vector<ggml_tensor *> tensors;
  for (const auto &input_ref : node.inputs) {
    tensors.push_back(resolve_input(ctx, input_ref));
  }

  // GGML ggml_concat concatenates along a GGML dimension
  // Convert PyTorch dim to GGML dim (reversed order)
  int n_dims = ggml_n_dims(tensors[0]);
  int ggml_dim = n_dims - 1 - py_dim;

  // Handle negative dimension
  if (py_dim < 0) {
    py_dim = n_dims + py_dim;
    ggml_dim = n_dims - 1 - py_dim;
  }

  // Concatenate iteratively: result = concat(a, b), then concat(result, c), etc.
  ggml_tensor *result = tensors[0];
  for (size_t i = 1; i < tensors.size(); i++) {
    result = ggml_concat(ctx, result, tensors[i], ggml_dim);
  }

  return result;
}

ggml_tensor *GraphInterpreter::build_get_rows(ggml_context *ctx,
                                              const GIRNode &node) {
  // GET_ROWS: embedding lookup / row selection
  // inputs: [weight_table, indices]
  // weight_table: [embedding_dim, num_embeddings] in GGML order
  // indices: [n_indices] or [n1, n2, ...] integer tensor
  // output: [embedding_dim, n_indices] or [embedding_dim, n1, n2, ...]
  if (node.inputs.size() < 2) {
    throw std::runtime_error("GET_ROWS requires 2 inputs (weight, indices)");
  }

  ggml_tensor *weight_table = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *indices = resolve_input(ctx, node.inputs[1]);

  // Get original indices shape for later reshape
  int64_t idx_ne0 = indices->ne[0];
  int64_t idx_ne1 = indices->ne[1];
  int64_t idx_ne2 = indices->ne[2];
  int64_t idx_ne3 = indices->ne[3];
  int64_t n_indices = ggml_nelements(indices);

  // If indices is multi-dimensional, flatten to 1D first
  bool need_reshape = (idx_ne1 > 1 || idx_ne2 > 1 || idx_ne3 > 1);
  if (need_reshape) {
    indices = ggml_cont(ctx, ggml_reshape_1d(ctx, indices, n_indices));
  }

  // Perform the get_rows operation
  ggml_tensor *result = ggml_get_rows(ctx, weight_table, indices);

  // If we flattened, reshape output to match original index dimensions
  // output shape: [embedding_dim, idx_ne0, idx_ne1, ...]
  if (need_reshape) {
    int64_t embed_dim = weight_table->ne[0];
    if (idx_ne2 > 1) {
      result = ggml_reshape_4d(ctx, result, embed_dim, idx_ne0, idx_ne1, idx_ne2);
    } else if (idx_ne1 > 1) {
      result = ggml_reshape_3d(ctx, result, embed_dim, idx_ne0, idx_ne1);
    }
  }

  return result;
}

ggml_tensor *GraphInterpreter::build_new_zeros(ggml_context *ctx,
                                               const GIRNode &node) {
  // NEW_ZEROS: create a tensor filled with zeros
  // params.shape: the shape of the tensor to create
  auto shape = get_int_array_param(node, "shape");

  // Use output_shape if params.shape is empty
  if (shape.empty()) {
    shape = node.output_shape;
  }

  if (shape.empty()) {
    throw std::runtime_error("NEW_ZEROS: no shape available");
  }

  // Resolve symbolic dimensions (e.g., DIM_N_ATOMS -> actual n_atoms value)
  shape = resolve_shape(shape);

  // Reverse shape (Python→GGML dimension order)
  std::reverse(shape.begin(), shape.end());

  // Create zero-initialized tensor
  ggml_tensor *result = nullptr;
  switch (shape.size()) {
  case 1:
    result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
    break;
  case 2:
    result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
    break;
  case 3:
    result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2]);
    break;
  case 4:
    result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
    break;
  default:
    throw std::runtime_error("NEW_ZEROS: unsupported number of dimensions: " +
                             std::to_string(shape.size()));
  }

  // Mark as input so it will be initialized
  ggml_set_input(result);
  // Store for later initialization to zero
  pending_constants_.push_back({result, 0.0f});

  return result;
}

ggml_tensor *GraphInterpreter::build_new_ones(ggml_context *ctx,
                                              const GIRNode &node) {
  // NEW_ONES: create a tensor filled with ones
  auto shape = get_int_array_param(node, "shape");
  if (shape.empty()) {
    shape = node.output_shape;
  }
  if (shape.empty()) {
    throw std::runtime_error("NEW_ONES: no shape available");
  }

  // Resolve symbolic dimensions (e.g., DIM_N_ATOMS -> actual n_atoms value)
  shape = resolve_shape(shape);

  // Reverse shape (Python→GGML dimension order)
  std::reverse(shape.begin(), shape.end());

  ggml_tensor *result = nullptr;
  switch (shape.size()) {
  case 1:
    result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
    break;
  case 2:
    result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
    break;
  case 3:
    result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2]);
    break;
  case 4:
    result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
    break;
  default:
    throw std::runtime_error("NEW_ONES: unsupported number of dimensions");
  }

  ggml_set_input(result);
  pending_constants_.push_back({result, 1.0f});
  return result;
}

ggml_tensor *GraphInterpreter::build_linear(ggml_context *ctx,
                                            const GIRNode &node) {
  // LINEAR: y = x @ W.T + b
  // inputs: [input, weight] or [input, weight, bias]
  if (node.inputs.size() < 2) {
    throw std::runtime_error("LINEAR requires at least 2 inputs (input, weight)");
  }

  ggml_tensor *input = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *weight = resolve_input(ctx, node.inputs[1]);

  // GGML mul_mat: (weight @ input.T).T = input @ weight.T
  ggml_tensor *result = ggml_mul_mat(ctx, weight, input);

  // Add bias if present
  if (node.inputs.size() > 2) {
    ggml_tensor *bias = resolve_input(ctx, node.inputs[2]);
    result = ggml_add(ctx, result, bias);
  }

  return result;
}

ggml_tensor *GraphInterpreter::build_slice(ggml_context *ctx,
                                           const GIRNode &node) {
  // SLICE: extract a slice from a tensor
  // This is a simplified version - full slicing is complex
  if (node.inputs.empty()) {
    throw std::runtime_error("SLICE requires at least 1 input");
  }

  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // For now, if output_shape matches input, just pass through
  // This handles the common case of x[..., :, :]
  auto output_shape = node.output_shape;
  if (output_shape.empty()) {
    return a;
  }

  // Resolve symbolic dimensions (e.g., DIM_N_ATOMS -> actual n_atoms value)
  output_shape = resolve_shape(output_shape);

  // Reverse for GGML
  std::reverse(output_shape.begin(), output_shape.end());

  // Check if shapes match
  bool shapes_match = true;
  for (size_t i = 0; i < output_shape.size() && i < 4; i++) {
    if (output_shape[i] != static_cast<int64_t>(a->ne[i])) {
      shapes_match = false;
      break;
    }
  }

  if (shapes_match) {
    return a;
  }

  // Use view for actual slicing
  switch (output_shape.size()) {
  case 1:
    return ggml_view_1d(ctx, a, output_shape[0], 0);
  case 2:
    return ggml_view_2d(ctx, a, output_shape[0], output_shape[1], a->nb[1], 0);
  case 3:
    return ggml_view_3d(ctx, a, output_shape[0], output_shape[1], output_shape[2],
                        a->nb[1], a->nb[2], 0);
  case 4:
    return ggml_view_4d(ctx, a, output_shape[0], output_shape[1], output_shape[2],
                        output_shape[3], a->nb[1], a->nb[2], a->nb[3], 0);
  default:
    return a;
  }
}

ggml_tensor *GraphInterpreter::build_split(ggml_context *ctx,
                                           const GIRNode &node) {
  // SPLIT: split a tensor into chunks
  // The actual extraction is done by subsequent getitem/VIEW nodes
  // Just pass through the input tensor
  if (node.inputs.empty()) {
    throw std::runtime_error("SPLIT requires at least 1 input");
  }
  return resolve_input(ctx, node.inputs[0]);
}

ggml_tensor *GraphInterpreter::build_bitwise_not(ggml_context *ctx,
                                                 const GIRNode &node) {
  // BITWISE_NOT: invert boolean tensor
  // For float representation of bool: not(x) = 1 - x
  if (node.inputs.empty()) {
    throw std::runtime_error("BITWISE_NOT requires at least 1 input");
  }

  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);

  // Create a ones tensor to subtract from
  ggml_tensor *ones = nullptr;
  switch (ggml_n_dims(a)) {
  case 1:
    ones = ggml_new_tensor_1d(ctx, a->type, a->ne[0]);
    break;
  case 2:
    ones = ggml_new_tensor_2d(ctx, a->type, a->ne[0], a->ne[1]);
    break;
  case 3:
    ones = ggml_new_tensor_3d(ctx, a->type, a->ne[0], a->ne[1], a->ne[2]);
    break;
  case 4:
    ones = ggml_new_tensor_4d(ctx, a->type, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
    break;
  default:
    throw std::runtime_error("BITWISE_NOT: unsupported dimensions");
  }

  ggml_set_input(ones);
  pending_constants_.push_back({ones, 1.0f});

  return ggml_sub(ctx, ones, a);
}

ggml_tensor *GraphInterpreter::build_index(ggml_context *ctx,
                                           const GIRNode &node) {
  // INDEX: advanced indexing with tensor indices
  // This is a complex operation - for now, handle simple cases
  if (node.inputs.size() < 2) {
    throw std::runtime_error("INDEX requires at least 2 inputs");
  }

  ggml_tensor *a = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *indices = resolve_input(ctx, node.inputs[1]);

  // Use get_rows for simple 1D index case
  return ggml_get_rows(ctx, a, indices);
}

ggml_tensor *GraphInterpreter::build_index_put(ggml_context *ctx,
                                               const GIRNode &node) {
  // INDEX_PUT: scatter values into tensor at indices
  // In PET, this is used for masking: tensor[boolean_mask] = scalar_value
  // When the value is 0 and mask indicates invalid positions:
  //   result = source * (1 - mask)  (zeros out masked positions)
  if (node.inputs.size() < 3) {
    throw std::runtime_error("INDEX_PUT requires 3 inputs (source, mask, values)");
  }

  ggml_tensor *source = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *mask = resolve_input(ctx, node.inputs[1]);
  ggml_tensor *values = resolve_input(ctx, node.inputs[2]);

  // For scalar values (typically 0.0 for masking), we can simplify:
  // result = source * (1 - mask) + values * mask
  // When values = 0: result = source * (1 - mask)

  // Create a ones tensor for computing (1 - mask)
  ggml_tensor *ones = nullptr;
  switch (ggml_n_dims(mask)) {
  case 1:
    ones = ggml_new_tensor_1d(ctx, mask->type, mask->ne[0]);
    break;
  case 2:
    ones = ggml_new_tensor_2d(ctx, mask->type, mask->ne[0], mask->ne[1]);
    break;
  case 3:
    ones = ggml_new_tensor_3d(ctx, mask->type, mask->ne[0], mask->ne[1], mask->ne[2]);
    break;
  case 4:
    ones = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3]);
    break;
  default:
    throw std::runtime_error("INDEX_PUT: unsupported mask dimensions");
  }
  ggml_set_input(ones);
  pending_constants_.push_back({ones, 1.0f});

  // (1 - mask): where mask=1 (to replace), this gives 0; where mask=0 (to keep), this gives 1
  ggml_tensor *inv_mask = ggml_sub(ctx, ones, mask);

  // source * inv_mask: keeps only non-masked positions
  ggml_tensor *kept = ggml_mul(ctx, source, inv_mask);

  // mask * values: the values to insert at masked positions
  ggml_tensor *inserted = ggml_mul(ctx, mask, values);

  // Combine: kept + inserted
  return ggml_add(ctx, kept, inserted);
}

ggml_tensor *GraphInterpreter::build_where(ggml_context *ctx,
                                            const GIRNode &node) {
  // WHERE(condition, x, y): returns x where condition is true, y otherwise
  // Implemented as: x * condition_f32 + y * (1 - condition_f32)
  if (node.inputs.size() < 3) {
    throw std::runtime_error("WHERE requires 3 inputs (condition, x, y)");
  }

  ggml_tensor *condition = resolve_input(ctx, node.inputs[0]);
  ggml_tensor *x = resolve_input(ctx, node.inputs[1]);
  ggml_tensor *y = resolve_input(ctx, node.inputs[2]);

  // condition is a float tensor where 1.0 = true, 0.0 = false
  // result = x * condition + y * (1 - condition)
  ggml_tensor *x_masked = ggml_mul(ctx, x, condition);
  ggml_tensor *ones = nullptr;
  switch (ggml_n_dims(condition)) {
  case 1:
    ones = ggml_new_tensor_1d(ctx, condition->type, condition->ne[0]);
    break;
  case 2:
    ones = ggml_new_tensor_2d(ctx, condition->type, condition->ne[0],
                               condition->ne[1]);
    break;
  case 3:
    ones = ggml_new_tensor_3d(ctx, condition->type, condition->ne[0],
                               condition->ne[1], condition->ne[2]);
    break;
  case 4:
    ones = ggml_new_tensor_4d(ctx, condition->type, condition->ne[0],
                               condition->ne[1], condition->ne[2],
                               condition->ne[3]);
    break;
  default:
    throw std::runtime_error("WHERE: unsupported condition dimensions");
  }
  ggml_set_input(ones);
  pending_constants_.push_back({ones, 1.0f});

  ggml_tensor *inv_condition = ggml_sub(ctx, ones, condition);
  ggml_tensor *y_masked = ggml_mul(ctx, y, inv_condition);

  return ggml_add(ctx, x_masked, y_masked);
}

std::string GraphInterpreter::summary() const {
  if (graph_.nodes.empty()) {
    return "No graph loaded";
  }

  std::stringstream ss;
  ss << "Graph: " << graph_.model_type << " v" << graph_.version << "\n";
  ss << "Inputs: " << graph_.inputs.size() << "\n";
  for (const auto &input : graph_.inputs) {
    ss << "  - " << input.name << ": [";
    for (size_t i = 0; i < input.shape.size(); i++) {
      if (i > 0)
        ss << ", ";
      ss << input.shape[i];
    }
    ss << "]\n";
  }
  ss << "Nodes: " << graph_.nodes.size() << "\n";

  // Count operations
  std::map<std::string, int> op_counts;
  for (const auto &node : graph_.nodes) {
    op_counts[node.op]++;
  }
  ss << "Operations:\n";
  for (const auto &[op, count] : op_counts) {
    ss << "  " << op << ": " << count << "\n";
  }

  ss << "Outputs: " << graph_.outputs.size() << "\n";
  for (const auto &output : graph_.outputs) {
    ss << "  - " << output.name << " -> " << output.node_ref << "\n";
  }

  return ss.str();
}

void GraphInterpreter::set_debug_output_dir(const std::string &dir) {
  debug_dir_ = dir;
  debug_mode_ = !dir.empty();
  if (debug_mode_) {
    std::filesystem::create_directories(dir);
  }
}

void GraphInterpreter::dump_tensor(ggml_tensor *t, const std::string &name,
                                   int node_id) {
  if (!debug_mode_ || !t || !t->data) {
    return;
  }

  // Format filename: node_XXXX_name.bin
  std::stringstream filename;
  filename << debug_dir_ << "/node_" << std::setfill('0') << std::setw(4)
           << node_id << "_" << name << ".bin";

  // Get tensor data size
  size_t n_elements = ggml_nelements(t);
  size_t data_size = n_elements * ggml_element_size(t);

  // Write binary data
  std::ofstream file(filename.str(), std::ios::binary);
  if (file.is_open()) {
    file.write(static_cast<const char *>(t->data), data_size);
    file.close();
  }

  // Write metadata JSON
  std::stringstream meta_filename;
  meta_filename << debug_dir_ << "/node_" << std::setfill('0') << std::setw(4)
                << node_id << "_" << name << ".json";

  std::ofstream meta_file(meta_filename.str());
  if (meta_file.is_open()) {
    meta_file << "{\n";
    meta_file << "  \"node_id\": " << node_id << ",\n";
    meta_file << "  \"name\": \"" << name << "\",\n";
    meta_file << "  \"shape\": [" << t->ne[0] << ", " << t->ne[1] << ", "
              << t->ne[2] << ", " << t->ne[3] << "],\n";
    meta_file << "  \"n_dims\": " << ggml_n_dims(t) << ",\n";
    meta_file << "  \"type\": " << static_cast<int>(t->type) << ",\n";
    meta_file << "  \"n_elements\": " << n_elements << ",\n";

    // Compute basic statistics if F32
    if (t->type == GGML_TYPE_F32) {
      const float *data = static_cast<const float *>(t->data);
      float min_val = data[0], max_val = data[0], sum = 0.0f;
      for (size_t i = 0; i < n_elements; i++) {
        if (data[i] < min_val)
          min_val = data[i];
        if (data[i] > max_val)
          max_val = data[i];
        sum += data[i];
      }
      float mean = sum / static_cast<float>(n_elements);
      meta_file << "  \"min\": " << min_val << ",\n";
      meta_file << "  \"max\": " << max_val << ",\n";
      meta_file << "  \"mean\": " << mean << ",\n";

      // First few values
      meta_file << "  \"first_values\": [";
      for (size_t i = 0; i < std::min(n_elements, size_t(10)); i++) {
        if (i > 0)
          meta_file << ", ";
        meta_file << data[i];
      }
      meta_file << "]\n";
    } else {
      meta_file << "  \"stats\": \"non-f32 tensor\"\n";
    }

    meta_file << "}\n";
    meta_file.close();
  }
}

void GraphInterpreter::dump_all_tensors() {
  if (!debug_mode_) {
    return;
  }

  // Dump all node outputs
  for (const auto &[node_id, tensor] : node_outputs_) {
    // Find the node name
    std::string name = "unknown";
    for (const auto &node : graph_.nodes) {
      if (node.id == node_id) {
        name = node.name.empty() ? node.op : node.name;
        break;
      }
    }
    dump_tensor(tensor, name, node_id);
  }

  // Dump inputs
  int input_id = -1000;
  for (const auto &[name, tensor] : inputs_) {
    dump_tensor(tensor, "input_" + name, input_id--);
  }

  // Dump output
  if (output_) {
    dump_tensor(output_, "final_output", 9999);
  }
}

} // namespace mlipcpp::runtime
