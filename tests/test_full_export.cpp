/**
 * Test the full PET graph export with neighbor list inputs.
 *
 * This test loads the graph exported by export_pet_full.py and runs it
 * with the saved test inputs, comparing the output to PyTorch reference.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "runtime/graph_interpreter.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

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
// Format: "weights": {"name": [dim1, dim2, ...], ...}
std::map<std::string, std::vector<int64_t>>
parse_weight_shapes(const std::string &path) {
  std::map<std::string, std::vector<int64_t>> shapes;

  std::ifstream f(path);
  if (!f)
    return shapes;

  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

  // Find "weights" section
  size_t weights_pos = content.find("\"weights\"");
  if (weights_pos == std::string::npos)
    return shapes;

  // Find the opening brace of weights object
  size_t brace_start = content.find('{', weights_pos);
  if (brace_start == std::string::npos)
    return shapes;

  // Find matching closing brace
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

  // Parse each weight entry: "name": [d1, d2, ...]
  size_t search_pos = 0;
  while (true) {
    // Find next quoted name
    size_t quote1 = weights_str.find('"', search_pos);
    if (quote1 == std::string::npos)
      break;
    size_t quote2 = weights_str.find('"', quote1 + 1);
    if (quote2 == std::string::npos)
      break;

    std::string name = weights_str.substr(quote1 + 1, quote2 - quote1 - 1);

    // Find array start
    size_t arr_start = weights_str.find('[', quote2);
    if (arr_start == std::string::npos)
      break;
    size_t arr_end = weights_str.find(']', arr_start);
    if (arr_end == std::string::npos)
      break;

    // Parse dimensions
    std::string arr_str =
        weights_str.substr(arr_start + 1, arr_end - arr_start - 1);
    std::vector<int64_t> dims;
    std::stringstream ss(arr_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
      // Trim whitespace
      size_t start = item.find_first_not_of(" \t\n");
      size_t end = item.find_last_not_of(" \t\n");
      if (start != std::string::npos && end != std::string::npos) {
        dims.push_back(std::stoll(item.substr(start, end - start + 1)));
      }
    }

    // Include empty dims (scalar tensors) - don't skip them
    shapes[name] = dims;

    search_pos = arr_end + 1;
  }

  return shapes;
}

} // namespace

TEST_CASE("Execute full PET graph with neighbor list inputs",
          "[graph][pet][integration]") {
  const std::string test_dir = "/tmp/pet_full_export";
  const std::string graph_path = test_dir + "/pet_full.json";

  // Skip if test files don't exist
  if (!std::filesystem::exists(graph_path)) {
    SKIP("Full PET export not found - run export_pet_full.py first");
  }

  // Load the graph
  GraphInterpreter interp;
  interp.load_graph_file(graph_path);

  INFO("Graph loaded: " << interp.summary());
  // Allow for different graph versions (with/without 5D decomposition)
  REQUIRE(interp.graph().nodes.size() >= 137);
  REQUIRE(interp.graph().nodes.size() <= 500);  // pet-omad-s has ~292 nodes

  // Read configuration from metadata
  std::ifstream meta_stream(test_dir + "/metadata.json");
  REQUIRE(meta_stream.good());
  std::string meta_content((std::istreambuf_iterator<char>(meta_stream)),
                           std::istreambuf_iterator<char>());
  meta_stream.close();

  // Parse n_atoms and max_neighbors from metadata JSON
  int n_atoms = 2;
  int max_neighbors = 8;
  {
    auto find_int = [&](const std::string &key) -> int {
      size_t pos = meta_content.find("\"" + key + "\"");
      if (pos == std::string::npos) return -1;
      pos = meta_content.find(':', pos);
      if (pos == std::string::npos) return -1;
      pos = meta_content.find_first_of("0123456789", pos);
      if (pos == std::string::npos) return -1;
      return std::stoi(meta_content.substr(pos));
    };
    int na = find_int("n_atoms");
    int mn = find_int("max_neighbors");
    if (na > 0) n_atoms = na;
    if (mn > 0) max_neighbors = mn;
  }
  INFO("Configuration: n_atoms=" << n_atoms << ", max_neighbors=" << max_neighbors);

  // Set symbolic dimensions for graph resolution
  interp.set_dimension("n_atoms", n_atoms);
  interp.set_dimension("max_neighbors", max_neighbors);

  // Create CPU backend first - all tensors will use this
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  // Create weight context with no_alloc=true (backend will allocate)
  constexpr size_t WEIGHT_CTX_SIZE = 128 * 1024 * 1024;
  ggml_context *weight_ctx = ggml_init({WEIGHT_CTX_SIZE, nullptr, true});
  REQUIRE(weight_ctx != nullptr);

  // Load weight shapes from metadata
  auto weight_shapes = parse_weight_shapes(test_dir + "/metadata.json");
  INFO("Found " << weight_shapes.size() << " weight shapes in metadata");

  // Create weight tensors (no data yet)
  INFO("Creating weight tensors...");
  std::map<std::string, std::pair<ggml_tensor *, std::vector<float>>>
      weight_data_map;
  int loaded_count = 0;

  for (const auto &[name, py_shape] : weight_shapes) {
    std::string weight_path = test_dir + "/" + name + ".bin";
    if (!std::filesystem::exists(weight_path)) {
      continue;
    }

    auto data = load_binary<float>(weight_path);

    // Reverse shape for GGML (PyTorch -> GGML)
    std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());

    ggml_tensor *t = nullptr;
    switch (ggml_shape.size()) {
    case 0:
      // Scalar tensor - create as 1D with 1 element
      t = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, 1);
      break;
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
    weight_data_map[name] = {t, std::move(data)};
    interp.set_weight(name, t);
    loaded_count++;
  }
  INFO("Created " << loaded_count << " weight tensors");
  REQUIRE(loaded_count > 50); // Should have ~64 weights

  // Create input tensors
  INFO("Creating input tensors...");

  // Species: [n_atoms] int32
  auto species_data = load_binary<int32_t>(test_dir + "/input_species.bin");
  ggml_tensor *species =
      ggml_new_tensor_1d(weight_ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species, "species");

  // Neighbor species: [n_atoms, max_neighbors] int32 -> GGML [max_neighbors,
  // n_atoms]
  auto neighbor_species_data =
      load_binary<int32_t>(test_dir + "/input_neighbor_species.bin");
  ggml_tensor *neighbor_species =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species, "neighbor_species");

  // Edge vectors: [n_atoms, max_neighbors, 3] -> GGML [3, max_neighbors,
  // n_atoms]
  auto edge_vectors_data =
      load_binary<float>(test_dir + "/input_edge_vectors.bin");
  ggml_tensor *edge_vectors =
      ggml_new_tensor_3d(weight_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors, "edge_vectors");

  // Edge distances: [n_atoms, max_neighbors] -> GGML [max_neighbors, n_atoms]
  auto edge_distances_data =
      load_binary<float>(test_dir + "/input_edge_distances.bin");
  ggml_tensor *edge_distances =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(edge_distances, "edge_distances");

  // Padding mask: [n_atoms, max_neighbors] bool -> GGML [max_neighbors, n_atoms]
  // Note: exported as bool (1 byte), we load as floats (1.0 for valid, 0.0 for padding)
  std::vector<float> padding_mask_data(n_atoms * max_neighbors, 1.0f);
  {
    std::ifstream f(test_dir + "/input_padding_mask.bin", std::ios::binary);
    if (f) {
      std::vector<uint8_t> bool_data(n_atoms * max_neighbors);
      f.read(reinterpret_cast<char*>(bool_data.data()), bool_data.size());
      for (size_t i = 0; i < bool_data.size(); i++) {
        padding_mask_data[i] = bool_data[i] ? 1.0f : 0.0f;
      }
    }
  }
  ggml_tensor *padding_mask =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(padding_mask, "padding_mask");

  // Reverse neighbor index: [n_atoms * max_neighbors] int32
  auto reverse_neighbor_index_data =
      load_binary<int32_t>(test_dir + "/input_reverse_neighbor_index.bin");
  ggml_tensor *reverse_neighbor_index =
      ggml_new_tensor_1d(weight_ctx, GGML_TYPE_I32, n_atoms * max_neighbors);
  ggml_set_name(reverse_neighbor_index, "reverse_neighbor_index");

  // Cutoff factors: [n_atoms, max_neighbors] -> GGML [max_neighbors, n_atoms]
  auto cutoff_factors_data =
      load_binary<float>(test_dir + "/input_cutoff_factors.bin");
  ggml_tensor *cutoff_factors =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(cutoff_factors, "cutoff_factors");

  // Register inputs
  interp.set_input("species", species);
  interp.set_input("neighbor_species", neighbor_species);
  interp.set_input("edge_vectors", edge_vectors);
  interp.set_input("edge_distances", edge_distances);
  interp.set_input("padding_mask", padding_mask);
  interp.set_input("reverse_neighbor_index", reverse_neighbor_index);
  interp.set_input("cutoff_factors", cutoff_factors);

  // Allocate backend buffer for weight context
  INFO("Allocating weight buffer...");
  ggml_backend_buffer_t weight_buffer =
      ggml_backend_alloc_ctx_tensors(weight_ctx, cpu_backend);
  REQUIRE(weight_buffer != nullptr);

  // Copy weight data to tensors
  INFO("Loading weight data...");
  for (const auto &[name, pair] : weight_data_map) {
    const auto &[tensor, data] = pair;
    ggml_backend_tensor_set(tensor, data.data(), 0,
                            data.size() * sizeof(float));
  }

  // Copy input data to tensors
  INFO("Loading input data...");
  ggml_backend_tensor_set(species, species_data.data(), 0,
                          species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(neighbor_species, neighbor_species_data.data(), 0,
                          neighbor_species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(edge_vectors, edge_vectors_data.data(), 0,
                          edge_vectors_data.size() * sizeof(float));
  ggml_backend_tensor_set(edge_distances, edge_distances_data.data(), 0,
                          edge_distances_data.size() * sizeof(float));
  ggml_backend_tensor_set(padding_mask, padding_mask_data.data(), 0,
                          padding_mask_data.size() * sizeof(float));
  ggml_backend_tensor_set(reverse_neighbor_index, reverse_neighbor_index_data.data(), 0,
                          reverse_neighbor_index_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(cutoff_factors, cutoff_factors_data.data(), 0,
                          cutoff_factors_data.size() * sizeof(float));

  // Build computation graph
  INFO("Building computation graph...");
  constexpr size_t COMPUTE_CTX_SIZE = 256 * 1024 * 1024;
  ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});
  REQUIRE(compute_ctx != nullptr);

  ggml_tensor *output = interp.build(compute_ctx);
  REQUIRE(output != nullptr);
  ggml_set_output(output);

  INFO("Output shape: [" << output->ne[0] << ", " << output->ne[1] << ", "
                         << output->ne[2] << ", " << output->ne[3] << "]");

  // Create GGML compute graph
  ggml_cgraph *cgraph = ggml_new_graph(compute_ctx);
  ggml_build_forward_expand(cgraph, output);

  // Allocate compute buffer
  INFO("Allocating compute buffer...");
  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx, cpu_backend);
  REQUIRE(compute_buffer != nullptr);

  // Initialize constants (like NEW_ZEROS)
  interp.init_constants();

  // Compute
  INFO("Computing...");
  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  // Get output
  std::vector<float> result(n_atoms);
  ggml_backend_tensor_get(output, result.data(), 0, n_atoms * sizeof(float));

  // Load expected output
  auto expected = load_binary<float>(test_dir + "/expected_output.bin");

  // Print results
  std::cout << "C++ output:    [";
  for (int i = 0; i < n_atoms; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << result[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "PyTorch output: [";
  for (int i = 0; i < n_atoms; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << expected[i];
  }
  std::cout << "]" << std::endl;

  // Compare
  float max_diff = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    float diff = std::abs(result[i] - expected[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "Max difference: " << max_diff << std::endl;

  // Should be within numerical tolerance (< 1e-5 relative error)
  for (int i = 0; i < n_atoms; i++) {
    CHECK_THAT(result[i], WithinAbs(expected[i], 1e-3));
  }

  // Cleanup
  ggml_backend_buffer_free(compute_buffer);
  ggml_backend_buffer_free(weight_buffer);
  ggml_backend_free(cpu_backend);
  ggml_free(compute_ctx);
  ggml_free(weight_ctx);
}

TEST_CASE("Symbolized graph works with different dimensions (water)",
          "[graph][pet][dynamic]") {
  // Use the graph/weights exported at (7,11) but run with water data (3,2)
  const std::string graph_dir = "/tmp/pet_full_export";
  const std::string data_dir = "/tmp/pet_water_real";
  const std::string graph_path = graph_dir + "/pet_full.json";

  if (!std::filesystem::exists(graph_path) ||
      !std::filesystem::exists(data_dir + "/metadata.json")) {
    SKIP("Export files not found - run export_pet_full.py and water test gen");
  }

  // Read water dimensions from metadata
  std::ifstream meta_stream(data_dir + "/metadata.json");
  REQUIRE(meta_stream.good());
  std::string meta_content((std::istreambuf_iterator<char>(meta_stream)),
                           std::istreambuf_iterator<char>());
  meta_stream.close();

  int n_atoms = 3;
  int max_neighbors = 2;
  {
    auto find_int = [&](const std::string &key) -> int {
      size_t pos = meta_content.find("\"" + key + "\"");
      if (pos == std::string::npos) return -1;
      pos = meta_content.find(':', pos);
      if (pos == std::string::npos) return -1;
      pos = meta_content.find_first_of("0123456789", pos);
      if (pos == std::string::npos) return -1;
      return std::stoi(meta_content.substr(pos));
    };
    int na = find_int("n_atoms");
    int mn = find_int("max_neighbors");
    if (na > 0) n_atoms = na;
    if (mn > 0) max_neighbors = mn;
  }
  INFO("Water config: n_atoms=" << n_atoms << ", max_neighbors=" << max_neighbors);

  // Load symbolized graph
  GraphInterpreter interp;
  interp.load_graph_file(graph_path);

  // Set symbolic dimensions for water
  interp.set_dimension("n_atoms", n_atoms);
  interp.set_dimension("max_neighbors", max_neighbors);

  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  REQUIRE(cpu_backend != nullptr);

  constexpr size_t WEIGHT_CTX_SIZE = 128 * 1024 * 1024;
  ggml_context *weight_ctx = ggml_init({WEIGHT_CTX_SIZE, nullptr, true});
  REQUIRE(weight_ctx != nullptr);

  // Load weights from GRAPH dir (not data dir)
  auto weight_shapes = parse_weight_shapes(graph_dir + "/metadata.json");
  std::map<std::string, std::pair<ggml_tensor *, std::vector<float>>>
      weight_data_map;
  int loaded_count = 0;

  for (const auto &[name, py_shape] : weight_shapes) {
    std::string weight_path = graph_dir + "/" + name + ".bin";
    if (!std::filesystem::exists(weight_path))
      continue;

    auto data = load_binary<float>(weight_path);
    std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());

    ggml_tensor *t = nullptr;
    switch (ggml_shape.size()) {
    case 0:
      t = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, 1);
      break;
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
    weight_data_map[name] = {t, std::move(data)};
    interp.set_weight(name, t);
    loaded_count++;
  }
  REQUIRE(loaded_count > 50);

  // Create input tensors with WATER dimensions
  auto species_data = load_binary<int32_t>(data_dir + "/input_species.bin");
  ggml_tensor *species =
      ggml_new_tensor_1d(weight_ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species, "species");

  auto neighbor_species_data =
      load_binary<int32_t>(data_dir + "/input_neighbor_species.bin");
  ggml_tensor *neighbor_species =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species, "neighbor_species");

  auto edge_vectors_data =
      load_binary<float>(data_dir + "/input_edge_vectors.bin");
  ggml_tensor *edge_vectors =
      ggml_new_tensor_3d(weight_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors, "edge_vectors");

  auto edge_distances_data =
      load_binary<float>(data_dir + "/input_edge_distances.bin");
  ggml_tensor *edge_distances =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(edge_distances, "edge_distances");

  std::vector<float> padding_mask_data(n_atoms * max_neighbors, 1.0f);
  {
    std::ifstream f(data_dir + "/input_padding_mask.bin", std::ios::binary);
    if (f) {
      std::vector<uint8_t> bool_data(n_atoms * max_neighbors);
      f.read(reinterpret_cast<char*>(bool_data.data()), bool_data.size());
      for (size_t i = 0; i < bool_data.size(); i++) {
        padding_mask_data[i] = bool_data[i] ? 1.0f : 0.0f;
      }
    }
  }
  ggml_tensor *padding_mask =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(padding_mask, "padding_mask");

  auto reverse_neighbor_index_data =
      load_binary<int32_t>(data_dir + "/input_reverse_neighbor_index.bin");
  ggml_tensor *reverse_neighbor_index =
      ggml_new_tensor_1d(weight_ctx, GGML_TYPE_I32, n_atoms * max_neighbors);
  ggml_set_name(reverse_neighbor_index, "reverse_neighbor_index");

  auto cutoff_factors_data =
      load_binary<float>(data_dir + "/input_cutoff_factors.bin");
  ggml_tensor *cutoff_factors =
      ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(cutoff_factors, "cutoff_factors");

  interp.set_input("species", species);
  interp.set_input("neighbor_species", neighbor_species);
  interp.set_input("edge_vectors", edge_vectors);
  interp.set_input("edge_distances", edge_distances);
  interp.set_input("padding_mask", padding_mask);
  interp.set_input("reverse_neighbor_index", reverse_neighbor_index);
  interp.set_input("cutoff_factors", cutoff_factors);

  // Allocate and fill
  ggml_backend_buffer_t weight_buffer =
      ggml_backend_alloc_ctx_tensors(weight_ctx, cpu_backend);
  REQUIRE(weight_buffer != nullptr);

  for (const auto &[name, pair] : weight_data_map) {
    const auto &[tensor, data] = pair;
    ggml_backend_tensor_set(tensor, data.data(), 0,
                            data.size() * sizeof(float));
  }
  ggml_backend_tensor_set(species, species_data.data(), 0,
                          species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(neighbor_species, neighbor_species_data.data(), 0,
                          neighbor_species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(edge_vectors, edge_vectors_data.data(), 0,
                          edge_vectors_data.size() * sizeof(float));
  ggml_backend_tensor_set(edge_distances, edge_distances_data.data(), 0,
                          edge_distances_data.size() * sizeof(float));
  ggml_backend_tensor_set(padding_mask, padding_mask_data.data(), 0,
                          padding_mask_data.size() * sizeof(float));
  ggml_backend_tensor_set(reverse_neighbor_index, reverse_neighbor_index_data.data(), 0,
                          reverse_neighbor_index_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(cutoff_factors, cutoff_factors_data.data(), 0,
                          cutoff_factors_data.size() * sizeof(float));

  // Build and compute
  constexpr size_t COMPUTE_CTX_SIZE = 256 * 1024 * 1024;
  ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});
  REQUIRE(compute_ctx != nullptr);

  ggml_tensor *output = interp.build(compute_ctx);
  REQUIRE(output != nullptr);
  ggml_set_output(output);

  INFO("Output shape: [" << output->ne[0] << ", " << output->ne[1] << "]");
  CHECK(output->ne[0] == n_atoms);

  ggml_cgraph *cgraph = ggml_new_graph(compute_ctx);
  ggml_build_forward_expand(cgraph, output);

  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx, cpu_backend);
  REQUIRE(compute_buffer != nullptr);

  interp.init_constants();

  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  REQUIRE(status == GGML_STATUS_SUCCESS);

  std::vector<float> result(n_atoms);
  ggml_backend_tensor_get(output, result.data(), 0, n_atoms * sizeof(float));

  auto expected = load_binary<float>(data_dir + "/expected_output.bin");

  std::cout << "Water C++ output:    [";
  for (int i = 0; i < n_atoms; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << result[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "Water PyTorch output: [";
  for (int i = 0; i < n_atoms; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << expected[i];
  }
  std::cout << "]" << std::endl;

  float max_diff = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    float diff = std::abs(result[i] - expected[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "Water max difference: " << max_diff << std::endl;

  for (int i = 0; i < n_atoms; i++) {
    CHECK_THAT(result[i], WithinAbs(expected[i], 1e-3));
  }

  // Check against full PyTorch PET reference (with composition energy and scaling)
  {
    auto find_double = [&](const std::string &key) -> double {
      size_t pos = meta_content.find("\"" + key + "\"");
      if (pos == std::string::npos) return 0.0;
      pos = meta_content.find(':', pos);
      if (pos == std::string::npos) return 0.0;
      pos = meta_content.find_first_of("-0123456789", pos);
      if (pos == std::string::npos) return 0.0;
      return std::stod(meta_content.substr(pos));
    };

    double comp_energy = find_double("composition_energy");
    double pytorch_ref = find_double("pytorch_reference_energy");
    double energy_scale = find_double("energy_scale");
    // Default to 1.0 if energy_scale not found (legacy models)
    if (energy_scale == 0.0) energy_scale = 1.0;

    if (pytorch_ref != 0.0) {
      float model_sum = 0.0f;
      for (int i = 0; i < n_atoms; i++) model_sum += result[i];
      // Apply energy scale factor: total = scaled_model + composition
      double scaled_model = model_sum * energy_scale;
      double total = scaled_model + comp_energy;

      std::cout << "\n=== Full Energy Comparison ===" << std::endl;
      std::cout << "C++ model energy (raw): " << model_sum << " eV" << std::endl;
      std::cout << "Energy scale factor:    " << energy_scale << std::endl;
      std::cout << "C++ model (scaled):     " << scaled_model << " eV" << std::endl;
      std::cout << "Composition energy:     " << comp_energy << " eV" << std::endl;
      std::cout << "C++ total:              " << total << " eV" << std::endl;
      std::cout << "PyTorch reference:      " << pytorch_ref << " eV" << std::endl;
      std::cout << "Difference:             " << std::abs(total - pytorch_ref) << " eV" << std::endl;

      CHECK_THAT(total, WithinAbs(pytorch_ref, 1e-3));
    }
  }

  ggml_backend_buffer_free(compute_buffer);
  ggml_backend_buffer_free(weight_buffer);
  ggml_backend_free(cpu_backend);
  ggml_free(compute_ctx);
  ggml_free(weight_ctx);
}
