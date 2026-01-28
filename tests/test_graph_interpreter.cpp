#include <catch2/catch_test_macros.hpp>

#include "../src/runtime/graph_interpreter.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace mlipcpp::runtime;

TEST_CASE("GIR JSON parsing", "[runtime]") {
  // Create a simple test graph JSON
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [10, 20]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:1"}
    ],
    "nodes": [
      {"id": 0, "op": "MUL_MAT", "name": "matmul0", "inputs": ["weight:W", "input:x"], "output_shape": [30, 20], "output_dtype": "f32"},
      {"id": 1, "op": "UNARY_SILU", "name": "silu0", "inputs": ["node:0"], "output_shape": [30, 20], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  REQUIRE(interp.has_graph());

  const auto &graph = interp.graph();
  REQUIRE(graph.version == "1.0.0");
  REQUIRE(graph.model_type == "test");
  REQUIRE(graph.inputs.size() == 1);
  REQUIRE(graph.inputs[0].name == "x");
  REQUIRE(graph.inputs[0].dtype == GIRDtype::F32);
  REQUIRE(graph.inputs[0].shape.size() == 2);
  REQUIRE(graph.inputs[0].shape[0] == 10);
  REQUIRE(graph.inputs[0].shape[1] == 20);

  REQUIRE(graph.outputs.size() == 1);
  REQUIRE(graph.outputs[0].name == "y");
  REQUIRE(graph.outputs[0].node_ref == "node:1");

  REQUIRE(graph.nodes.size() == 2);
  REQUIRE(graph.nodes[0].id == 0);
  REQUIRE(graph.nodes[0].op == "MUL_MAT");
  REQUIRE(graph.nodes[0].inputs.size() == 2);
  REQUIRE(graph.nodes[0].inputs[0] == "weight:W");
  REQUIRE(graph.nodes[0].inputs[1] == "input:x");

  REQUIRE(graph.nodes[1].id == 1);
  REQUIRE(graph.nodes[1].op == "UNARY_SILU");
  REQUIRE(graph.nodes[1].inputs[0] == "node:0");
}

TEST_CASE("Graph summary", "[runtime]") {
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [10]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "UNARY_RELU", "name": "relu", "inputs": ["input:x"], "output_shape": [10], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  std::string summary = interp.summary();
  REQUIRE(summary.find("test") != std::string::npos);
  REQUIRE(summary.find("UNARY_RELU") != std::string::npos);
  REQUIRE(summary.find("Nodes: 1") != std::string::npos);
}

TEST_CASE("Load exported PET transformer graph", "[runtime][pet]") {
  // Load the exported PET transformer graph if it exists
  std::ifstream file("/tmp/pet_transformer.json");
  if (!file.is_open()) {
    SKIP("PET transformer graph not found at /tmp/pet_transformer.json");
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  GraphInterpreter interp;
  REQUIRE_NOTHROW(interp.load_graph(json));
  REQUIRE(interp.has_graph());

  const auto &graph = interp.graph();
  INFO("Loaded graph with " << graph.nodes.size() << " nodes");

  // TorchScript export produces ~40 nodes for transformer
  REQUIRE(graph.nodes.size() >= 30);

  // Check for expected operations
  std::map<std::string, int> op_counts;
  for (const auto &node : graph.nodes) {
    op_counts[node.op]++;
  }

  // Should have flash attention
  REQUIRE(op_counts["FLASH_ATTN_EXT"] >= 1);

  // Should have matrix multiplications
  REQUIRE(op_counts["MUL_MAT"] >= 1);

  // Print summary
  INFO("Summary:\n" << interp.summary());
}

TEST_CASE("Build simple addition graph", "[runtime][graph]") {
  // Create a graph that does: output = input + input (element-wise doubling)
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [4]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "ADD", "name": "add", "inputs": ["input:x", "input:x"], "output_shape": [4], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  // Create GGML context for graph building
  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create input tensor
  ggml_tensor *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
  ggml_set_input(input);

  // Set input and build graph
  interp.set_input("x", input);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->ne[0] == 4);
  REQUIRE(output->op == GGML_OP_ADD);

  // Verify graph structure - the ADD operation should reference the same tensor twice
  REQUIRE(output->src[0] == input);
  REQUIRE(output->src[1] == input);

  ggml_free(ctx);
}

TEST_CASE("Build matrix multiplication graph", "[runtime][graph]") {
  // Test MUL_MAT: output = W @ x
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [2]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "MUL_MAT", "name": "matmul", "inputs": ["weight:W", "input:x"], "output_shape": [3], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create weight tensor W: [2, 3] - 2 input features, 3 output features
  ggml_tensor *W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);

  // Input x = [2]
  ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  ggml_set_input(x);

  interp.set_weight("W", W);
  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->op == GGML_OP_MUL_MAT);
  REQUIRE(output->src[0] == W);
  REQUIRE(output->src[1] == x);

  ggml_free(ctx);
}

TEST_CASE("Build unary operations chain", "[runtime][graph]") {
  // Test SQR and SQRT: output = sqrt(sqr(x))
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [3]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:1"}
    ],
    "nodes": [
      {"id": 0, "op": "SQR", "name": "sqr", "inputs": ["input:x"], "output_shape": [3], "output_dtype": "f32"},
      {"id": 1, "op": "SQRT", "name": "sqrt", "inputs": ["node:0"], "output_shape": [3], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_input(x);

  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->op == GGML_OP_SQRT);

  // Check that the chain is correctly built
  ggml_tensor *sqr_result = output->src[0];
  REQUIRE(sqr_result != nullptr);
  REQUIRE(sqr_result->op == GGML_OP_SQR);
  REQUIRE(sqr_result->src[0] == x);

  ggml_free(ctx);
}

TEST_CASE("Build scale operation", "[runtime][graph]") {
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [4]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "SCALE", "name": "scale", "inputs": ["input:x"], "output_shape": [4], "output_dtype": "f32", "params": {"scale": 2.5}}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
  ggml_set_input(x);

  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->op == GGML_OP_SCALE);
  REQUIRE(output->src[0] == x);

  ggml_free(ctx);
}

// Helper to load a binary float array
static std::vector<float> load_binary_floats(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<float> data(size / sizeof(float));
  file.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

TEST_CASE("Execute simple MLP and compare to PyTorch", "[runtime][mlp][numerical]") {
  // This test requires running the Python export first
  std::ifstream file("/tmp/simple_mlp.json");
  if (!file.is_open()) {
    SKIP("Simple MLP graph not found at /tmp/simple_mlp.json");
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  // Load binary data files
  auto fc1_weight_data = load_binary_floats("/tmp/mlp_fc1_weight.bin");
  auto fc1_bias_data = load_binary_floats("/tmp/mlp_fc1_bias.bin");
  auto fc2_weight_data = load_binary_floats("/tmp/mlp_fc2_weight.bin");
  auto fc2_bias_data = load_binary_floats("/tmp/mlp_fc2_bias.bin");
  auto input_data = load_binary_floats("/tmp/mlp_input.bin");
  auto expected_output = load_binary_floats("/tmp/mlp_output.bin");

  if (fc1_weight_data.empty() || input_data.empty()) {
    SKIP("Binary data files not found - run Python export first");
    return;
  }

  GraphInterpreter interp;
  REQUIRE_NOTHROW(interp.load_graph(json));

  // Create GGML context with no_alloc=true for backend allocation
  struct ggml_init_params params = {
      .mem_size = 64 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // fc1: [128, 64] in PyTorch -> [64, 128] in GGML (transposed)
  // fc2: [64, 128] in PyTorch -> [128, 64] in GGML (transposed)
  ggml_tensor *fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 128);
  ggml_tensor *fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
  ggml_tensor *fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
  ggml_tensor *fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);

  // Input: [4, 64] in PyTorch -> [64, 4] in GGML
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
  ggml_set_input(x);

  interp.set_weight("fc1_weight", fc1_weight);
  interp.set_weight("fc1_bias", fc1_bias);
  interp.set_weight("fc2_weight", fc2_weight);
  interp.set_weight("fc2_bias", fc2_bias);
  interp.set_input("x", x);

  // Build the graph
  ggml_tensor *output = interp.build(ctx);
  REQUIRE(output != nullptr);
  REQUIRE(output->ne[0] == 64);
  REQUIRE(output->ne[1] == 4);
  ggml_set_output(output);

  // Create compute graph
  ggml_cgraph *cgraph = ggml_new_graph(ctx);
  ggml_build_forward_expand(cgraph, output);

  // Allocate using CPU backend
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
  REQUIRE(buf != nullptr);

  // Copy data to tensors
  ggml_backend_tensor_set(fc1_weight, fc1_weight_data.data(), 0,
                          fc1_weight_data.size() * sizeof(float));
  ggml_backend_tensor_set(fc1_bias, fc1_bias_data.data(), 0,
                          fc1_bias_data.size() * sizeof(float));
  ggml_backend_tensor_set(fc2_weight, fc2_weight_data.data(), 0,
                          fc2_weight_data.size() * sizeof(float));
  ggml_backend_tensor_set(fc2_bias, fc2_bias_data.data(), 0,
                          fc2_bias_data.size() * sizeof(float));
  ggml_backend_tensor_set(x, input_data.data(), 0,
                          input_data.size() * sizeof(float));

  // Compute
  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  // Get output data
  std::vector<float> out_data(expected_output.size());
  ggml_backend_tensor_get(output, out_data.data(), 0,
                          out_data.size() * sizeof(float));

  ggml_backend_buffer_free(buf);
  ggml_backend_free(cpu_backend);

  // Compare output to expected
  float max_diff = 0.0f;
  float sum_diff = 0.0f;
  for (size_t i = 0; i < expected_output.size(); i++) {
    float diff = std::abs(out_data[i] - expected_output[i]);
    max_diff = std::max(max_diff, diff);
    sum_diff += diff;
  }

  INFO("Max difference: " << max_diff);
  INFO("Mean difference: " << sum_diff / expected_output.size());
  INFO("Expected[0:4]: " << expected_output[0] << ", " << expected_output[1]
                         << ", " << expected_output[2] << ", "
                         << expected_output[3]);
  INFO("Got[0:4]: " << out_data[0] << ", " << out_data[1] << ", "
                    << out_data[2] << ", " << out_data[3]);

  // Should match within floating point tolerance
  REQUIRE(max_diff < 1e-4f);

  ggml_free(ctx);
}

TEST_CASE("Load and build PET transformer graph", "[runtime][transformer]") {
  // This test loads the exported PET transformer graph and verifies it can be built
  std::ifstream file("/tmp/transformer_validation/transformer.json");
  if (!file.is_open()) {
    SKIP("PET transformer graph not found - run export_transformer_validation.py first");
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  GraphInterpreter interp;
  REQUIRE_NOTHROW(interp.load_graph(json));

  // Verify graph structure
  const auto &graph = interp.graph();
  INFO("Graph has " << graph.nodes.size() << " nodes");
  REQUIRE(graph.nodes.size() == 52);  // 4D-compatible wrapper, no mask

  // Check inputs
  REQUIRE(graph.inputs.size() == 2);
  REQUIRE(graph.inputs[0].name == "tokens");
  REQUIRE(graph.inputs[1].name == "cutoff_factors");

  // Create context with no_alloc for backend allocation
  struct ggml_init_params params = {
      .mem_size = 256 * 1024 * 1024,  // 256 MB for transformer
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create input tensors - GGML shape [256, 9, 2] = PyTorch [2, 9, 256]
  ggml_tensor *tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 9, 2);
  ggml_tensor *cutoff = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 9, 2);
  ggml_set_input(tokens);
  ggml_set_input(cutoff);

  interp.set_input("tokens", tokens);
  interp.set_input("cutoff_factors", cutoff);

  // Create weight tensors
  // Layer 0 weights (GGML shapes = transposed PyTorch shapes)
  std::map<std::string, std::pair<int, int>> weight_shapes_2d = {
      {"layers_0_attention_input_linear_weight", {256, 768}},
      {"layers_0_attention_output_linear_weight", {256, 256}},
      {"layers_0_mlp_0_weight", {256, 512}},
      {"layers_0_mlp_3_weight", {512, 256}},
      {"layers_1_attention_input_linear_weight", {256, 768}},
      {"layers_1_attention_output_linear_weight", {256, 256}},
      {"layers_1_mlp_0_weight", {256, 512}},
      {"layers_1_mlp_3_weight", {512, 256}},
  };

  std::map<std::string, int> weight_shapes_1d = {
      {"layers_0_attention_input_linear_bias", 768},
      {"layers_0_attention_output_linear_bias", 256},
      {"layers_0_mlp_0_bias", 512},
      {"layers_0_mlp_3_bias", 256},
      {"layers_0_norm_attention_weight", 256},
      {"layers_0_norm_attention_bias", 256},
      {"layers_0_norm_mlp_weight", 256},
      {"layers_0_norm_mlp_bias", 256},
      {"layers_1_attention_input_linear_bias", 768},
      {"layers_1_attention_output_linear_bias", 256},
      {"layers_1_mlp_0_bias", 512},
      {"layers_1_mlp_3_bias", 256},
      {"layers_1_norm_attention_weight", 256},
      {"layers_1_norm_attention_bias", 256},
      {"layers_1_norm_mlp_weight", 256},
      {"layers_1_norm_mlp_bias", 256},
  };

  for (const auto &[name, shape] : weight_shapes_2d) {
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.first, shape.second);
    interp.set_weight(name, w);
  }

  for (const auto &[name, size] : weight_shapes_1d) {
    auto w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    interp.set_weight(name, w);
  }

  // Try to build the graph
  ggml_tensor *output = nullptr;
  REQUIRE_NOTHROW(output = interp.build(ctx));
  REQUIRE(output != nullptr);

  // Check output shape - GGML [256, 9, 2] = PyTorch [2, 9, 256]
  INFO("Output shape: [" << output->ne[0] << ", " << output->ne[1] << ", "
                         << output->ne[2] << "]");
  REQUIRE(output->ne[0] == 256);
  REQUIRE(output->ne[1] == 9);
  REQUIRE(output->ne[2] == 2);

  ggml_free(ctx);
}

TEST_CASE("Load and build PET energy graph", "[runtime][pet_energy]") {
  // This test loads the exported PET energy computation graph
  std::ifstream file("/tmp/pet_energy_validation/pet_energy.json");
  if (!file.is_open()) {
    SKIP("PET energy graph not found - run export_pet_energy.py first");
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();

  GraphInterpreter interp;
  REQUIRE_NOTHROW(interp.load_graph(json));

  // Verify graph structure
  const auto &graph = interp.graph();
  INFO("Graph has " << graph.nodes.size() << " nodes");
  REQUIRE(graph.nodes.size() == 126);  // Full PET energy path (includes 4 SILU activations)

  // Check inputs
  REQUIRE(graph.inputs.size() == 1);
  REQUIRE(graph.inputs[0].name == "tokens");

  // Create context with no_alloc for backend allocation
  struct ggml_init_params params = {
      .mem_size = 512 * 1024 * 1024,  // 512 MB for full model
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create input tensors - GGML shape [256, 9, 2] = PyTorch [2, 9, 256]
  ggml_tensor *tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 9, 2);
  ggml_set_input(tokens);
  interp.set_input("tokens", tokens);

  // Load metadata to get weight shapes
  std::ifstream meta_file("/tmp/pet_energy_validation/metadata.json");
  if (!meta_file.is_open()) {
    SKIP("Metadata file not found");
    return;
  }

  // Parse metadata JSON to get weight shapes
  // Simple manual parsing for "weights": {"name": [dim0, dim1], ...}
  std::string meta_content((std::istreambuf_iterator<char>(meta_file)),
                           std::istreambuf_iterator<char>());
  meta_file.close();

  // Create weight tensors based on the graph's weight references
  std::set<std::string> weight_names;
  for (const auto &node : graph.nodes) {
    for (const auto &input : node.inputs) {
      if (input.rfind("weight:", 0) == 0) {
        weight_names.insert(input.substr(7));
      }
    }
  }

  INFO("Found " << weight_names.size() << " unique weights");

  // Create weight tensors using shapes from metadata
  for (const auto &name : weight_names) {
    ggml_tensor *w = nullptr;

    // Find shape in metadata: "name": [dim0, dim1]
    std::string pattern = "\"" + name + "\": [";
    size_t pos = meta_content.find(pattern);
    if (pos != std::string::npos) {
      pos += pattern.length();
      size_t end = meta_content.find("]", pos);
      std::string shape_str = meta_content.substr(pos, end - pos);

      // Parse shape array
      std::vector<int64_t> shape;
      std::stringstream ss(shape_str);
      std::string item;
      while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoll(item));
      }

      // The export already transposes 2D weights for GGML.
      // Metadata has PyTorch shape [out, in]. After export transpose,
      // the file has [in, out] which is correct for GGML MUL_MAT.
      // We just need to reverse for GGML dimension order.
      std::reverse(shape.begin(), shape.end());

      // Create tensor with appropriate dimensions
      if (shape.size() == 1) {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
      } else if (shape.size() == 2) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
      } else if (shape.size() == 3) {
        w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2]);
      }
    }

    if (w) {
      interp.set_weight(name, w);
    }
  }

  // Try to build the graph
  ggml_tensor *output = nullptr;
  REQUIRE_NOTHROW(output = interp.build(ctx));
  REQUIRE(output != nullptr);

  // Check output shape - should be [2] for 2 atoms
  INFO("Output shape: [" << output->ne[0] << ", " << output->ne[1] << ", "
                         << output->ne[2] << ", " << output->ne[3] << "]");
  REQUIRE(output->ne[0] == 2);  // 2 atoms

  ggml_free(ctx);
}

TEST_CASE("Execute PET energy graph with numerical validation",
          "[runtime][pet_energy][numerical]") {
  // Load PET energy graph
  std::ifstream file("/tmp/pet_energy_validation/pet_energy.json");
  if (!file.is_open()) {
    SKIP("PET energy graph not found");
    return;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json = buffer.str();
  file.close();

  GraphInterpreter interp;
  REQUIRE_NOTHROW(interp.load_graph(json));

  // Enable debug output
  interp.set_debug_output_dir("/tmp/pet_debug/cpp");

  // Create GGML context
  struct ggml_init_params params = {
      .mem_size = 512 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Load metadata for weight shapes
  std::ifstream meta_file("/tmp/pet_energy_validation/metadata.json");
  REQUIRE(meta_file.is_open());
  std::string meta_content((std::istreambuf_iterator<char>(meta_file)),
                           std::istreambuf_iterator<char>());
  meta_file.close();

  // Create input tensor - GGML shape [256, 9, 2]
  ggml_tensor *tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 9, 2);
  ggml_set_input(tokens);
  interp.set_input("tokens", tokens);

  // Create weight tensors from metadata
  const auto &graph = interp.graph();
  std::set<std::string> weight_names;
  for (const auto &node : graph.nodes) {
    for (const auto &input : node.inputs) {
      if (input.rfind("weight:", 0) == 0) {
        weight_names.insert(input.substr(7));
      }
    }
  }

  std::map<std::string, ggml_tensor *> weight_tensors;
  for (const auto &name : weight_names) {
    std::string pattern = "\"" + name + "\": [";
    size_t pos = meta_content.find(pattern);
    if (pos != std::string::npos) {
      pos += pattern.length();
      size_t end = meta_content.find("]", pos);
      std::string shape_str = meta_content.substr(pos, end - pos);

      std::vector<int64_t> shape;
      std::stringstream ss(shape_str);
      std::string item;
      while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoll(item));
      }

      // Reverse shape for GGML dimension ordering
      // PyTorch [768, 256] -> GGML [256, 768] (same memory, reversed indices)
      std::reverse(shape.begin(), shape.end());

      ggml_tensor *w = nullptr;
      if (shape.size() == 1) {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
      } else if (shape.size() == 2) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
      }

      if (w) {
        weight_tensors[name] = w;
        interp.set_weight(name, w);
      }
    }
  }

  // Build graph
  ggml_tensor *output = interp.build(ctx);
  REQUIRE(output != nullptr);
  REQUIRE(output->ne[0] == 2);
  ggml_set_output(output);

  // Create compute graph
  ggml_cgraph *cgraph = ggml_new_graph(ctx);
  ggml_build_forward_expand(cgraph, output);

  // Allocate using CPU backend
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
  REQUIRE(buf != nullptr);

  // Load and set input data
  auto input_data = load_binary_floats("/tmp/pet_energy_validation/input_tokens.bin");
  REQUIRE(!input_data.empty());
  INFO("Input data size: " << input_data.size() << " floats");
  INFO("Input[0:4]: " << input_data[0] << ", " << input_data[1] << ", "
                      << input_data[2] << ", " << input_data[3]);
  ggml_backend_tensor_set(tokens, input_data.data(), 0,
                          input_data.size() * sizeof(float));

  // Load and set weight data
  int weights_loaded = 0;
  for (const auto &[name, tensor] : weight_tensors) {
    std::string path = "/tmp/pet_energy_validation/" + name + ".bin";
    auto data = load_binary_floats(path);
    if (!data.empty()) {
      ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
      weights_loaded++;
    }
  }
  INFO("Loaded " << weights_loaded << " / " << weight_tensors.size() << " weights");

  // Compute
  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  // Dump all intermediate tensors for debugging
  interp.dump_all_tensors();
  INFO("Debug tensors dumped to /tmp/pet_debug/cpp/");

  // Get output data
  auto expected_output = load_binary_floats("/tmp/pet_energy_validation/expected_output.bin");
  REQUIRE(expected_output.size() == 2);

  std::vector<float> out_data(2);
  ggml_backend_tensor_get(output, out_data.data(), 0, 2 * sizeof(float));

  ggml_backend_buffer_free(buf);
  ggml_backend_free(cpu_backend);

  // Compare output
  INFO("Expected: [" << expected_output[0] << ", " << expected_output[1] << "]");
  INFO("Got: [" << out_data[0] << ", " << out_data[1] << "]");
  INFO("Expected total: " << expected_output[0] + expected_output[1]);
  INFO("Got total: " << out_data[0] + out_data[1]);

  float max_diff = 0.0f;
  for (size_t i = 0; i < 2; i++) {
    float diff = std::abs(out_data[i] - expected_output[i]);
    max_diff = std::max(max_diff, diff);
  }

  INFO("Max difference: " << max_diff);
  REQUIRE(max_diff < 1e-3f);  // Allow 0.1% error for complex graph

  ggml_free(ctx);
}

TEST_CASE("Build layer norm graph", "[runtime][graph]") {
  // Test layer norm decomposition
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [4, 256]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "DECOMPOSE", "name": "norm", "inputs": ["input:x", "const:0", "weight:w", "weight:b"], "output_shape": [4, 256], "output_dtype": "f32", "params": {"eps": 1e-5}}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Input shape [4, 256] in PyTorch = [256, 4] in GGML
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
  ggml_set_input(x);

  // Weight and bias: [256] in PyTorch = [256] in GGML
  ggml_tensor *w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
  ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);

  interp.set_input("x", x);
  interp.set_weight("w", w);
  interp.set_weight("b", b);

  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  // Layer norm decomposition produces: add(mul(norm(x), w), b)
  REQUIRE(output->op == GGML_OP_ADD);

  ggml_free(ctx);
}

// ============================================================================
// Isolated operation tests for debugging PET numerical accuracy
// ============================================================================

TEST_CASE("VIEW chunk extraction from 3D tensor", "[runtime][view][chunk]") {
  // Test chunk extraction via VIEW:
  // PyTorch: qkv = [2, 9, 768], chunk(3, dim=-1) -> 3 x [2, 9, 256]
  // GGML:    qkv = [768, 9, 2], VIEW with offset -> [256, 9, 2] each

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create source tensor [768, 9, 2] in GGML = [2, 9, 768] in PyTorch
  ggml_tensor *qkv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 9, 2);

  // Fill with test data: value = ne0_idx + 1000*ne1_idx + 100000*ne2_idx
  float *data = static_cast<float *>(qkv->data);
  for (int i2 = 0; i2 < 2; i2++) {
    for (int i1 = 0; i1 < 9; i1++) {
      for (int i0 = 0; i0 < 768; i0++) {
        int idx = i0 + 768 * i1 + 768 * 9 * i2;
        data[idx] = static_cast<float>(i0 + 1000 * i1 + 100000 * i2);
      }
    }
  }

  // Create 3 views for Q, K, V chunks
  // Each chunk has shape [256, 9, 2] starting at offset 0, 256, 512 in ne[0]

  // Chunk 0 (Q): offset 0, size 256
  ggml_tensor *q =
      ggml_view_3d(ctx, qkv, 256, 9, 2,
                   qkv->nb[1], // row stride (768 * sizeof(float))
                   qkv->nb[2], // slice stride
                   0);         // byte offset

  // Chunk 1 (K): offset 256
  ggml_tensor *k = ggml_view_3d(ctx, qkv, 256, 9, 2, qkv->nb[1], qkv->nb[2],
                                256 * sizeof(float));

  // Chunk 2 (V): offset 512
  ggml_tensor *v = ggml_view_3d(ctx, qkv, 256, 9, 2, qkv->nb[1], qkv->nb[2],
                                512 * sizeof(float));

  // Verify shapes
  REQUIRE(q->ne[0] == 256);
  REQUIRE(q->ne[1] == 9);
  REQUIRE(q->ne[2] == 2);

  REQUIRE(k->ne[0] == 256);
  REQUIRE(v->ne[0] == 256);

  // Verify values
  // Q should start at index 0: value = 0 + 1000*0 + 100000*0 = 0
  // K should start at index 256: value = 256 + 1000*0 + 100000*0 = 256
  // V should start at index 512: value = 512 + 1000*0 + 100000*0 = 512

  float *q_data = static_cast<float *>(q->data);
  float *k_data = static_cast<float *>(k->data);
  float *v_data = static_cast<float *>(v->data);

  // Check first element of each chunk
  REQUIRE(q_data[0] == 0.0f);   // Index 0 in original
  REQUIRE(k_data[0] == 256.0f); // Index 256 in original
  REQUIRE(v_data[0] == 512.0f); // Index 512 in original

  // Check element in second row (ne1=1)
  // Q[0, 1, 0] should be at original index 768 (next row)
  // Value = 0 + 1000*1 + 100000*0 = 1000
  int row_stride = 768; // Elements per row
  REQUIRE(q_data[row_stride] == 1000.0f); // Actually this is wrong indexing

  // Correct: need to use strides properly
  // For view tensors, data pointer points to start, but strides may differ
  // Check using ggml's internal view offset mechanism
  // q->data should point to qkv->data + 0
  // k->data should point to qkv->data + 256*4 bytes
  // v->data should point to qkv->data + 512*4 bytes
  REQUIRE(q->data == qkv->data);
  REQUIRE(k->data == static_cast<char *>(qkv->data) + 256 * sizeof(float));
  REQUIRE(v->data == static_cast<char *>(qkv->data) + 512 * sizeof(float));

  ggml_free(ctx);
}

TEST_CASE("TRANSPOSE 3D tensor axis mapping", "[runtime][transpose]") {
  // Test transpose in GGML vs PyTorch
  // PyTorch: transpose(1, 2) on [2, 9, 4, 64] -> [2, 4, 9, 64]
  // GGML: transpose on [64, 4, 9, 2] (reversed) should produce [64, 9, 4, 2]

  // For 3D case:
  // PyTorch: [2, 9, 256] with transpose(1, 2) -> not valid (only 3 dims)
  // Let's test 4D:
  // PyTorch: [2, 9, 4, 64] with transpose(1, 2) -> [2, 4, 9, 64]
  // GGML:    [64, 4, 9, 2] with permute(0, 2, 1, 3) -> [64, 9, 4, 2]

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Create 4D tensor [64, 4, 9, 2] in GGML = [2, 9, 4, 64] in PyTorch
  ggml_tensor *src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 9, 2);

  // Fill with test data: value = ne0 + 100*ne1 + 10000*ne2 + 1000000*ne3
  float *data = static_cast<float *>(src->data);
  for (int i3 = 0; i3 < 2; i3++) {
    for (int i2 = 0; i2 < 9; i2++) {
      for (int i1 = 0; i1 < 4; i1++) {
        for (int i0 = 0; i0 < 64; i0++) {
          int idx = i0 + 64 * (i1 + 4 * (i2 + 9 * i3));
          data[idx] = static_cast<float>(i0 + 100 * i1 + 10000 * i2 +
                                         1000000 * i3);
        }
      }
    }
  }

  // PyTorch transpose(1, 2) swaps dims 1 and 2 (0-indexed from left)
  // In PyTorch order: [2, 9, 4, 64] -> [2, 4, 9, 64]
  // In GGML order:    [64, 4, 9, 2] -> [64, 9, 4, 2]
  // This is ggml_permute(src, 0, 2, 1, 3)

  ggml_tensor *dst = ggml_permute(ctx, src, 0, 2, 1, 3);

  REQUIRE(dst->ne[0] == 64);
  REQUIRE(dst->ne[1] == 9); // Was 4
  REQUIRE(dst->ne[2] == 4); // Was 9
  REQUIRE(dst->ne[3] == 2);

  // Verify strides changed but data didn't move
  // Original strides: [sizeof(float), 64*4, 64*4*4, 64*4*9*4]
  // Permuted strides: [sizeof(float), 64*4*4, 64*4, 64*4*9*4]
  REQUIRE(dst->nb[0] == src->nb[0]); // Element stride unchanged
  REQUIRE(dst->nb[1] == src->nb[2]); // Swapped
  REQUIRE(dst->nb[2] == src->nb[1]); // Swapped
  REQUIRE(dst->nb[3] == src->nb[3]); // Batch stride unchanged

  ggml_free(ctx);
}

TEST_CASE("SELECT operation for node feature extraction",
          "[runtime][select]") {
  // Test SELECT: [:, 0, :] on [2, 9, 256] -> [2, 256]
  // This extracts the first position from sequence dimension

  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [2, 9, 256]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "SELECT", "name": "select", "inputs": ["input:x"], "output_shape": [2, 256], "output_dtype": "f32", "params": {"dim": 1, "index": 0}}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // Input [2, 9, 256] in PyTorch = [256, 9, 2] in GGML
  ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 9, 2);
  ggml_set_input(x);

  // Fill with identifiable values
  float *data = static_cast<float *>(x->data);
  for (int i2 = 0; i2 < 2; i2++) {
    for (int i1 = 0; i1 < 9; i1++) {
      for (int i0 = 0; i0 < 256; i0++) {
        int idx = i0 + 256 * i1 + 256 * 9 * i2;
        // Encode position in value: seq_pos * 1000 + feature_idx
        data[idx] = static_cast<float>(i1 * 1000 + i0);
      }
    }
  }

  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);

  // Output should be [256, 2] in GGML = [2, 256] in PyTorch
  REQUIRE(output->ne[0] == 256);
  REQUIRE(output->ne[1] == 2);

  // Verify we got seq_pos=0 data (values should be 0*1000 + feature_idx = feature_idx)
  // The output is a view, so we need to check the data pointer offset
  // For SELECT dim=1 index=0, we should get the slice at ne[1]=0

  // The output should have data at offset 0 (index 0 of dim 1)
  float *out_data = static_cast<float *>(output->data);
  REQUIRE(out_data[0] == 0.0f);    // seq_pos=0, feature=0: 0*1000 + 0 = 0
  REQUIRE(out_data[1] == 1.0f);    // seq_pos=0, feature=1: 0*1000 + 1 = 1
  REQUIRE(out_data[255] == 255.0f); // seq_pos=0, feature=255

  ggml_free(ctx);
}

TEST_CASE("FLASH_ATTN_EXT basic graph build", "[runtime][flash_attn]") {
  // Test flash attention graph building (not execution)
  // GGML flash attention shape requirements are complex,
  // just verify the graph builds correctly

  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "q", "dtype": "f32", "shape": [2, 4, 9, 64]},
      {"name": "k", "dtype": "f32", "shape": [2, 4, 9, 64]},
      {"name": "v", "dtype": "f32", "shape": [2, 4, 9, 64]}
    ],
    "outputs": [
      {"name": "attn", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "FLASH_ATTN_EXT", "name": "attn", "inputs": ["input:q", "input:k", "input:v"], "output_shape": [2, 4, 9, 64], "output_dtype": "f32", "params": {"scale": 0.125}}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 64 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // GGML flash attention expects:
  // Q: [head_dim, seq_len, n_heads, batch]
  // K: [head_dim, kv_seq_len, n_heads, batch]
  // V: [head_dim, kv_seq_len, n_heads, batch]
  // PyTorch: [batch, n_heads, seq_len, head_dim]

  // [2, 4, 9, 64] PyTorch = [64, 9, 4, 2] GGML
  ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 9, 4, 2);
  ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 9, 4, 2);
  ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 9, 4, 2);

  ggml_set_input(q);
  ggml_set_input(k);
  ggml_set_input(v);

  interp.set_input("q", q);
  interp.set_input("k", k);
  interp.set_input("v", v);

  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  // Flash attention output is permuted to match PyTorch SDPA format
  // So the output op is PERMUTE wrapping FLASH_ATTN_EXT
  REQUIRE(output->op == GGML_OP_PERMUTE);
  REQUIRE(output->src[0]->op == GGML_OP_FLASH_ATTN_EXT);

  // Verify output has expected number of elements
  // GGML flash attention output shape depends on internal logic
  REQUIRE(ggml_nelements(output) > 0);
  REQUIRE(output->ne[0] == 64); // head_dim is preserved

  ggml_free(ctx);
}

TEST_CASE("Debug tensor dumping", "[runtime][debug]") {
  // Test the debug tensor dumping functionality

  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [4]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "SCALE", "name": "scale_test", "inputs": ["input:x"], "output_shape": [4], "output_dtype": "f32", "params": {"scale": 2.0}}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  // Enable debug mode
  std::string debug_dir = "/tmp/test_debug_dump";
  interp.set_debug_output_dir(debug_dir);

  // Use no_alloc = true for backend allocation
  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
  ggml_set_input(x);

  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->op == GGML_OP_SCALE);
  ggml_set_output(output);

  // Compute the graph
  ggml_cgraph *cgraph = ggml_new_graph(ctx);
  ggml_build_forward_expand(cgraph, output);

  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
  REQUIRE(buf != nullptr);

  // Set input data after allocation
  float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ggml_backend_tensor_set(x, input_data, 0, 4 * sizeof(float));

  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  // Dump all tensors
  interp.dump_all_tensors();

  // Verify output values
  float out_data[4];
  ggml_backend_tensor_get(output, out_data, 0, 4 * sizeof(float));
  REQUIRE(out_data[0] == 2.0f);
  REQUIRE(out_data[1] == 4.0f);
  REQUIRE(out_data[2] == 6.0f);
  REQUIRE(out_data[3] == 8.0f);

  ggml_backend_buffer_free(buf);
  ggml_backend_free(cpu_backend);
  ggml_free(ctx);

  // Note: We don't verify the files exist here to keep the test simple
  // In practice, you would check /tmp/test_debug_dump/ for the dumped files
}
