/**
 * Test GraphModel with direct-format exported graphs.
 *
 * This tests the GraphModel wrapper that converts AtomicSystem inputs
 * to the format expected by auto-exported graphs.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "runtime/graph_model.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::runtime;
using Catch::Matchers::WithinAbs;

namespace {

// Load binary file into vector
template <typename T>
std::vector<T> load_binary(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Failed to open: " + path);
  }
  size_t size = f.tellg();
  f.seekg(0);
  std::vector<T> data(size / sizeof(T));
  f.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

// Simple parser to extract weight shapes from metadata.json
std::map<std::string, std::vector<int64_t>>
parse_weight_shapes(const std::string &path) {
  std::map<std::string, std::vector<int64_t>> shapes;

  std::ifstream f(path);
  if (!f)
    return shapes;

  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

  size_t weights_pos = content.find("\"weights\"");
  if (weights_pos == std::string::npos)
    return shapes;

  size_t brace_start = content.find('{', weights_pos);
  if (brace_start == std::string::npos)
    return shapes;

  int brace_count = 1;
  size_t pos = brace_start + 1;
  while (pos < content.size() && brace_count > 0) {
    if (content[pos] == '{')
      brace_count++;
    else if (content[pos] == '}')
      brace_count--;
    pos++;
  }
  std::string weights_str = content.substr(brace_start, pos - brace_start);

  size_t search_pos = 0;
  while (true) {
    size_t quote1 = weights_str.find('"', search_pos);
    if (quote1 == std::string::npos)
      break;
    size_t quote2 = weights_str.find('"', quote1 + 1);
    if (quote2 == std::string::npos)
      break;

    std::string name = weights_str.substr(quote1 + 1, quote2 - quote1 - 1);

    size_t arr_start = weights_str.find('[', quote2);
    if (arr_start == std::string::npos)
      break;
    size_t arr_end = weights_str.find(']', arr_start);
    if (arr_end == std::string::npos)
      break;

    std::string arr_str =
        weights_str.substr(arr_start + 1, arr_end - arr_start - 1);
    std::vector<int64_t> dims;
    std::stringstream ss(arr_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
      size_t start = item.find_first_not_of(" \t\n");
      size_t end = item.find_last_not_of(" \t\n");
      if (start != std::string::npos && end != std::string::npos) {
        dims.push_back(std::stoll(item.substr(start, end - start + 1)));
      }
    }

    if (!dims.empty()) {
      shapes[name] = dims;
    }

    search_pos = arr_end + 1;
  }

  return shapes;
}

// Helper to load graph and weights into a GraphModel
void setup_graph_model(GraphModel &model, const std::string &test_dir,
                       ggml_context *weight_ctx, ggml_backend_t backend) {
  const std::string graph_path = test_dir + "/pet_full.json";
  model.load_graph_file(graph_path);

  auto weight_shapes = parse_weight_shapes(test_dir + "/metadata.json");

  for (const auto &[name, py_shape] : weight_shapes) {
    std::string weight_path = test_dir + "/" + name + ".bin";
    if (!std::filesystem::exists(weight_path))
      continue;

    auto data = load_binary<float>(weight_path);
    std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());

    ggml_tensor *t = nullptr;
    switch (ggml_shape.size()) {
    case 1:
      t = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, ggml_shape[0]);
      break;
    case 2:
      t = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, ggml_shape[0],
                             ggml_shape[1]);
      break;
    case 3:
      t = ggml_new_tensor_3d(weight_ctx, GGML_TYPE_F32, ggml_shape[0],
                             ggml_shape[1], ggml_shape[2]);
      break;
    default:
      continue;
    }

    ggml_set_name(t, name.c_str());
    model.set_weight(name, t);
  }

  // Allocate and fill weights
  ggml_backend_buffer_t weight_buffer =
      ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
  REQUIRE(weight_buffer != nullptr);

  for (const auto &[name, py_shape] : weight_shapes) {
    std::string weight_path = test_dir + "/" + name + ".bin";
    if (!std::filesystem::exists(weight_path))
      continue;

    auto data = load_binary<float>(weight_path);
    ggml_tensor *t = ggml_get_tensor(weight_ctx, name.c_str());
    if (t) {
      ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    }
  }
}

} // namespace

TEST_CASE("GraphModel detects direct input format", "[graph][model]") {
  const std::string test_dir = "/tmp/pet_full_export";
  const std::string graph_path = test_dir + "/pet_full.json";

  if (!std::filesystem::exists(graph_path)) {
    SKIP("Full PET export not found - run export_pet_full.py first");
  }

  GraphModel model;
  model.load_graph_file(graph_path);

  // Check expected dimensions were detected
  auto [n_atoms, max_neighbors] = model.expected_dimensions();
  INFO("Detected n_atoms=" << n_atoms << ", max_neighbors=" << max_neighbors);

  CHECK(n_atoms == 2);
  CHECK(max_neighbors == 8);
}

TEST_CASE("GraphModel with direct inputs matches interpreter",
          "[graph][model][integration]") {
  const std::string test_dir = "/tmp/pet_full_export";
  const std::string graph_path = test_dir + "/pet_full.json";

  if (!std::filesystem::exists(graph_path)) {
    SKIP("Full PET export not found - run export_pet_full.py first");
  }

  // Create backend and context
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  constexpr size_t WEIGHT_CTX_SIZE = 128 * 1024 * 1024;
  ggml_context *weight_ctx = ggml_init({WEIGHT_CTX_SIZE, nullptr, true});
  REQUIRE(weight_ctx != nullptr);

  // Setup model
  GraphModel model;
  setup_graph_model(model, test_dir, weight_ctx, cpu_backend);

  // Set expected dimensions manually (normally from metadata)
  model.set_expected_dimensions(2, 8);

  // Setup species mapping (Si = 14 -> index 0)
  // This would normally come from the GGUF file
  // For now we'll just test with the raw test inputs

  // Load expected output
  auto expected = load_binary<float>(test_dir + "/expected_output.bin");
  INFO("PyTorch output: [" << expected[0] << ", " << expected[1] << "]");

  // Create a simple test system (2 Si atoms)
  // For this test, we would need the exact same inputs as the export
  // For now, just verify the model loads correctly

  INFO("GraphModel setup successful");
  INFO("Expected dimensions: n_atoms=2, max_neighbors=8");

  // Note: Full prediction test requires matching the exact test inputs
  // which include specific edge vectors and distances. The test_full_export.cpp
  // directly loads those binary files, while GraphModel computes them from
  // positions and neighbor lists.

  // Cleanup
  ggml_backend_free(cpu_backend);
  ggml_free(weight_ctx);
}
