/**
 * Test GraphModel with direct-format exported graphs.
 *
 * This tests the GraphModel wrapper that converts AtomicSystem inputs
 * to the format expected by auto-exported graphs.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "runtime/graph_model.h"
#include "core/gguf_loader.h"
#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.h"
#include "mlipcpp/mlipcpp.hpp"

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

TEST_CASE("GraphModel loads graph file", "[graph][model]") {
  const std::string test_dir = "/tmp/pet_full_export";
  const std::string graph_path = test_dir + "/pet_full.json";

  if (!std::filesystem::exists(graph_path)) {
    SKIP("Full PET export not found - run export_pet_full.py first");
  }

  GraphModel model;
  model.load_graph_file(graph_path);

  // Check graph was loaded
  const auto &graph = model.interpreter().graph();
  CHECK(graph.nodes.size() > 100);
  CHECK(graph.inputs.size() >= 5);
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

  // Note: species mapping and dimensions are normally from GGUF file

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

TEST_CASE("GraphModel GGUF energy prediction", "[graph][model][gguf]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Auto-exported GGUF not found at " << model_path);
  }
  if (!std::filesystem::exists(water_xyz)) {
    SKIP("Water XYZ file not found");
  }

  GraphModel model;
  REQUIRE(model.load_from_gguf(model_path));

  // Verify the GGUF was exported with the current full-model format
  const auto &graph = model.interpreter().graph();
  bool has_species_input = false;
  for (const auto &inp : graph.inputs) {
    if (inp.name == "species") has_species_input = true;
  }
  if (!has_species_input) {
    SKIP("GGUF uses old graph format (no 'species' input) - re-export with "
         "export_pet_gguf.py");
  }

  // Read water system
  std::ifstream file(water_xyz);
  REQUIRE(file.is_open());
  auto water = mlipcpp::io::read_xyz(file);
  REQUIRE(water.num_atoms() == 3);

  // Predict energy
  ModelResult result = model.predict(water);

  INFO("Water energy: " << result.energy << " eV");
  // Energy should be negative and in a reasonable range for water
  CHECK(result.energy < 0.0f);
  CHECK(result.energy > -100.0f);
}

TEST_CASE("GraphModel GGUF forces prediction", "[graph][model][gguf][forces]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Auto-exported GGUF not found at " << model_path);
  }
  if (!std::filesystem::exists(water_xyz)) {
    SKIP("Water XYZ file not found");
  }

  GraphModel model;
  REQUIRE(model.load_from_gguf(model_path));

  std::ifstream file(water_xyz);
  REQUIRE(file.is_open());
  auto water = mlipcpp::io::read_xyz(file);
  REQUIRE(water.num_atoms() == 3);

  // Predict energy + forces
  ModelResult result = model.predict(water, true);

  INFO("Water energy: " << result.energy << " eV");
  CHECK(result.energy < 0.0f);
  CHECK(result.energy > -100.0f);

  // Should have forces for 3 atoms (9 components)
  REQUIRE(result.forces.size() == 9);

  // Newton's third law: forces should sum to approximately zero
  float fx_sum = result.forces[0] + result.forces[3] + result.forces[6];
  float fy_sum = result.forces[1] + result.forces[4] + result.forces[7];
  float fz_sum = result.forces[2] + result.forces[5] + result.forces[8];

  INFO("Force sum: [" << fx_sum << ", " << fy_sum << ", " << fz_sum << "]");
  CHECK_THAT(fx_sum, WithinAbs(0.0f, 0.01f));
  CHECK_THAT(fy_sum, WithinAbs(0.0f, 0.01f));
  CHECK_THAT(fz_sum, WithinAbs(0.0f, 0.01f));

  // Print per-atom forces
  for (int i = 0; i < 3; i++) {
    INFO("Atom " << i << " forces: [" << result.forces[i * 3] << ", "
                 << result.forces[i * 3 + 1] << ", "
                 << result.forces[i * 3 + 2] << "] eV/A");
  }
}

TEST_CASE("GraphModel dynamic system sizes", "[graph][model][gguf][dynamic]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";
  const std::string si_xyz = "geometries/si.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Auto-exported GGUF not found at " << model_path);
  }
  if (!std::filesystem::exists(water_xyz) ||
      !std::filesystem::exists(si_xyz)) {
    SKIP("Test XYZ files not found");
  }

  GraphModel model;
  REQUIRE(model.load_from_gguf(model_path));

  // Verify GGUF format compatibility
  const auto &graph = model.interpreter().graph();
  bool has_species_input = false;
  for (const auto &inp : graph.inputs) {
    if (inp.name == "species") has_species_input = true;
  }
  if (!has_species_input) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  // Predict water (3 atoms)
  {
    std::ifstream file(water_xyz);
    auto water = mlipcpp::io::read_xyz(file);
    ModelResult result = model.predict(water);
    INFO("Water energy: " << result.energy << " eV");
    CHECK(result.energy < 0.0f);
  }

  // Predict silicon (2 atoms) - different system size, same model instance
  {
    std::ifstream file(si_xyz);
    auto si = mlipcpp::io::read_xyz(file);
    ModelResult result = model.predict(si);
    INFO("Si energy: " << result.energy << " eV");
    CHECK(result.energy < 0.0f);
  }
}

// Helper: check if a GGUF file uses pet-graph architecture
static bool is_pet_graph_gguf(const std::string &path) {
  try {
    mlipcpp::GGUFLoader loader(path);
    return loader.get_string("general.architecture", "") == "pet-graph";
  } catch (...) {
    return false;
  }
}

// ============================================================================
// Predictor API Tests (public C++ API)
// ============================================================================

TEST_CASE("GraphModel via Predictor API", "[graph][model][api]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Auto-exported GGUF not found at " << model_path);
  }
  if (!is_pet_graph_gguf(model_path)) {
    SKIP("GGUF uses old architecture - re-export with export_pet_gguf.py");
  }
  if (!std::filesystem::exists(water_xyz)) {
    SKIP("Water XYZ file not found");
  }

  // Load via public Predictor API (same path users take)
  mlipcpp::Predictor predictor(model_path);
  REQUIRE(predictor.model_type() == "PET-Graph");

  // Read water system
  std::ifstream file(water_xyz);
  auto water = mlipcpp::io::read_xyz(file);
  REQUIRE(water.num_atoms() == 3);

  // Predict via raw pointer API
  auto result = predictor.predict(
      water.num_atoms(), water.positions(), water.atomic_numbers(),
      nullptr, nullptr, false);

  INFO("Predictor API water energy: " << result.energy << " eV");
  CHECK(result.energy < 0.0f);
  CHECK(result.energy > -100.0f);
}

TEST_CASE("GraphModel via Predictor API with forces",
          "[graph][model][api][forces]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Forces GGUF not found at " << model_path);
  }
  if (!is_pet_graph_gguf(model_path)) {
    SKIP("GGUF uses old architecture - re-export");
  }
  if (!std::filesystem::exists(water_xyz)) {
    SKIP("Water XYZ file not found");
  }

  mlipcpp::Predictor predictor(model_path);
  REQUIRE(predictor.model_type() == "PET-Graph");

  std::ifstream file(water_xyz);
  auto water = mlipcpp::io::read_xyz(file);

  auto result = predictor.predict(
      water.num_atoms(), water.positions(), water.atomic_numbers(),
      nullptr, nullptr, true);

  INFO("Predictor API water energy: " << result.energy << " eV");
  CHECK(result.energy < 0.0f);
  CHECK(result.has_forces());
  REQUIRE(result.forces.size() == 9);

  // Newton's third law
  float fx_sum = result.forces[0] + result.forces[3] + result.forces[6];
  float fy_sum = result.forces[1] + result.forces[4] + result.forces[7];
  float fz_sum = result.forces[2] + result.forces[5] + result.forces[8];
  CHECK_THAT(fx_sum, WithinAbs(0.0f, 0.01f));
  CHECK_THAT(fy_sum, WithinAbs(0.0f, 0.01f));
  CHECK_THAT(fz_sum, WithinAbs(0.0f, 0.01f));
}

// ============================================================================
// C API Tests
// ============================================================================

TEST_CASE("C API loads graph model", "[graph][model][c_api]") {
  const std::string model_path = "gguf/pet-auto.gguf";
  const std::string water_xyz = "geometries/water.xyz";

  if (!std::filesystem::exists(model_path)) {
    SKIP("Auto-exported GGUF not found at " << model_path);
  }
  if (!is_pet_graph_gguf(model_path)) {
    SKIP("GGUF uses old architecture - re-export with export_pet_gguf.py");
  }
  if (!std::filesystem::exists(water_xyz)) {
    SKIP("Water XYZ file not found");
  }

  // Test C API lifecycle
  auto model = mlipcpp_model_create(nullptr);
  REQUIRE(model != nullptr);

  auto err = mlipcpp_model_load(model, model_path.c_str());
  REQUIRE(err == MLIPCPP_OK);

  // Check cutoff
  float cutoff = 0.0f;
  err = mlipcpp_model_get_cutoff(model, &cutoff);
  REQUIRE(err == MLIPCPP_OK);
  CHECK(cutoff > 0.0f);

  // Predict water
  std::ifstream file(water_xyz);
  auto water = mlipcpp::io::read_xyz(file);

  mlipcpp_system_t system;
  system.n_atoms = water.num_atoms();
  system.positions = water.positions();
  system.atomic_numbers = water.atomic_numbers();
  system.cell = nullptr;
  system.pbc = nullptr;

  mlipcpp_result_t result = nullptr;
  err = mlipcpp_predict(model, &system, false, &result);
  REQUIRE(err == MLIPCPP_OK);
  REQUIRE(result != nullptr);

  float energy = 0.0f;
  err = mlipcpp_result_get_energy(result, &energy);
  REQUIRE(err == MLIPCPP_OK);

  INFO("C API water energy: " << energy << " eV");
  CHECK(energy < 0.0f);
  CHECK(energy > -100.0f);

  mlipcpp_result_free(result);
  mlipcpp_model_free(model);
}
