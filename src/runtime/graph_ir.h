#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlipcpp::runtime {

// Special values for symbolic dimensions in shapes
// These are used when a dimension depends on runtime parameters
constexpr int64_t DIM_N_ATOMS = -1000001;
constexpr int64_t DIM_MAX_NEIGHBORS = -1000002;
constexpr int64_t DIM_SEQ_LEN = -1000003;          // n_atoms * (max_neighbors + 1)
constexpr int64_t DIM_N_EDGES = -1000004;          // n_atoms * max_neighbors
constexpr int64_t DIM_MN_PLUS_ONE = -1000005;      // max_neighbors + 1

// Data types matching the Python GGMLDtype
enum class GIRDtype { F32, F16, I32, I16, I8, BOOL };

// Input specification
struct GIRInput {
  std::string name;
  GIRDtype dtype;
  std::vector<int64_t> shape; // -1 for dynamic dimensions
  std::vector<int> dynamic_dims;
};

// Output specification
struct GIROutput {
  std::string name;
  std::string node_ref; // "node:N"
  GIRDtype dtype;
  std::vector<int64_t> shape;
};

// Node parameters - can hold various types
using GIRParam = std::variant<int64_t, double, bool, std::string,
                              std::vector<int64_t>, std::vector<double>>;

// A computation node
struct GIRNode {
  int id;
  std::string op;
  std::string name;
  std::vector<std::string> inputs; // "node:N", "input:name", "weight:name",
                                   // "const:value"
  std::vector<int64_t> output_shape;
  GIRDtype output_dtype;
  std::map<std::string, GIRParam> params;
};

// The complete graph
struct GIRGraph {
  std::string version;
  std::string model_type;
  std::vector<GIRInput> inputs;
  std::vector<GIROutput> outputs;
  std::vector<GIRNode> nodes;
  std::map<std::string, GIRParam> constants;
  std::map<std::string, std::string> metadata;

  // Helper to get a node by id
  const GIRNode *get_node(int id) const {
    for (const auto &node : nodes) {
      if (node.id == id) {
        return &node;
      }
    }
    return nullptr;
  }

  // Helper to find input by name
  const GIRInput *get_input(const std::string &name) const {
    for (const auto &input : inputs) {
      if (input.name == name) {
        return &input;
      }
    }
    return nullptr;
  }
};

// Parse GIR graph from JSON string
GIRGraph parse_gir_json(const std::string &json_str);

// Parse GIR graph from file
GIRGraph load_gir_file(const std::string &path);

} // namespace mlipcpp::runtime
