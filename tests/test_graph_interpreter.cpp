#include <catch2/catch_test_macros.hpp>

#include "../src/runtime/graph_interpreter.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstring>
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

TEST_CASE("Execute MATMUL with non-square matrices", "[runtime][matmul][numerical]") {
  // Test MUL_MAT: output = W @ x  with non-square dimensions
  // W: [4, 3] (PyTorch) -> [3, 4] (GGML) — 3 input features, 4 output features
  // x: [3]   (PyTorch) -> [3]   (GGML) — 3 input features
  // output: [4]
  std::string json = R"({
    "version": "1.0.0",
    "model_type": "test",
    "inputs": [
      {"name": "x", "dtype": "f32", "shape": [3]}
    ],
    "outputs": [
      {"name": "y", "node_ref": "node:0"}
    ],
    "nodes": [
      {"id": 0, "op": "MUL_MAT", "name": "matmul", "inputs": ["weight:W", "input:x"], "output_shape": [4], "output_dtype": "f32"}
    ]
  })";

  GraphInterpreter interp;
  interp.load_graph(json);

  struct ggml_init_params params = {
      .mem_size = 16 * 1024 * 1024,
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(params);
  REQUIRE(ctx != nullptr);

  // GGML W: [3, 4] (ne[0]=3 input_features, ne[1]=4 output_features)
  ggml_tensor *W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
  ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_input(x);

  interp.set_weight("W", W);
  interp.set_input("x", x);
  ggml_tensor *output = interp.build(ctx);

  REQUIRE(output != nullptr);
  REQUIRE(output->ne[0] == 4);
  ggml_set_output(output);

  ggml_cgraph *cgraph = ggml_new_graph(ctx);
  ggml_build_forward_expand(cgraph, output);

  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend);
  REQUIRE(buf != nullptr);

  // W (stored in GGML column-major layout):
  // Row 0: [1, 2, 3]   -> output[0] = 1*1 + 2*2 + 3*3 = 14
  // Row 1: [4, 5, 6]   -> output[1] = 4*1 + 5*2 + 6*3 = 32
  // Row 2: [7, 8, 9]   -> output[2] = 7*1 + 8*2 + 9*3 = 50
  // Row 3: [10, 11, 12] -> output[3] = 10*1 + 11*2 + 12*3 = 68
  float W_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float x_data[] = {1, 2, 3};
  ggml_backend_tensor_set(W, W_data, 0, sizeof(W_data));
  ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));

  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  float out_data[4];
  ggml_backend_tensor_get(output, out_data, 0, sizeof(out_data));

  REQUIRE(out_data[0] == 14.0f);
  REQUIRE(out_data[1] == 32.0f);
  REQUIRE(out_data[2] == 50.0f);
  REQUIRE(out_data[3] == 68.0f);

  ggml_backend_buffer_free(buf);
  ggml_backend_free(cpu_backend);
  ggml_free(ctx);
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
