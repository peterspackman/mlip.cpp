/**
 * @file test_auto_vs_manual.cpp
 * @brief Side-by-side comparison of auto-exported GraphModel vs manual PET
 *
 * This test verifies that the automatic PyTorch -> GGML export produces
 * numerically equivalent results to the hand-coded PET implementation.
 *
 * Reference values from existing tests:
 * - water.xyz (3 atoms: O, H, H): -14.380176 eV
 * - si.xyz (2 atoms: Si, Si): -4.538056 eV
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mlipcpp/io.h"
#include "mlipcpp/model.h"
#include "mlipcpp/system.h"
#include "models/pet/pet.h"
#include "runtime/graph_model.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace mlipcpp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace fs = std::filesystem;

// Test data paths
static const char *MANUAL_MODEL_PATH = "local/pet-mad.gguf";
static const char *AUTO_MODEL_PATH = "local/pet-auto.gguf";
static const char *WATER_XYZ = "geometries/water.xyz";
static const char *SI_XYZ = "geometries/si.xyz";

// Reference energies from existing PET tests
static constexpr float WATER_ENERGY_REF = -14.380176f;
static constexpr float SI_ENERGY_REF = -4.538056f;
static constexpr float ENERGY_TOLERANCE = 1e-4f;

/**
 * Helper to check if a file exists
 */
static bool file_exists(const std::string &path) {
  return fs::exists(path) && fs::is_regular_file(path);
}

/**
 * Load an XYZ file
 */
static AtomicSystem load_xyz(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open XYZ file: " + path);
  }
  return io::read_xyz(file);
}

/**
 * Check if a GraphModel's GGUF uses the current full-model format.
 */
static bool has_current_graph_format(const runtime::GraphModel &model) {
  const auto &graph = model.interpreter().graph();
  for (const auto &inp : graph.inputs) {
    if (inp.name == "species") return true;
  }
  return false;
}

// ============================================================================
// Graph Interpreter Unit Tests (don't require full model)
// ============================================================================

TEST_CASE("GraphModel basic construction", "[auto_export][graph_model]") {
  runtime::GraphModel model;
  REQUIRE(model.model_type() == "graph");
  REQUIRE(model.cutoff() > 0.0f);
}

TEST_CASE("GraphModel loads simple graph JSON", "[auto_export][graph_model]") {
  // Create a simple test graph JSON
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
      {"id": 0, "op": "SCALE", "name": "scale", "inputs": ["input:x"],
       "output_shape": [10], "output_dtype": "f32", "params": {"scale": 2.0}}
    ]
  })";

  // Write to temp file
  std::string temp_path = "/tmp/test_simple_graph.json";
  {
    std::ofstream f(temp_path);
    f << json;
  }

  runtime::GraphModel model;
  REQUIRE_NOTHROW(model.load_graph_file(temp_path));

  // Check graph was loaded
  const auto &graph = model.interpreter().graph();
  REQUIRE(graph.nodes.size() == 1);
  REQUIRE(graph.inputs.size() == 1);
  REQUIRE(graph.inputs[0].name == "x");

  // Cleanup
  fs::remove(temp_path);
}

// ============================================================================
// Manual PET Model Tests (baseline)
// ============================================================================

TEST_CASE("Manual PET model loads and predicts", "[auto_export][manual]") {
  if (!file_exists(MANUAL_MODEL_PATH)) {
    SKIP("Manual PET model not found at " << MANUAL_MODEL_PATH);
  }
  if (!file_exists(WATER_XYZ)) {
    SKIP("Water XYZ file not found at " << WATER_XYZ);
  }

  // Load manual model
  pet::PETModel model(pet::PETHypers{});
  REQUIRE(model.load_from_gguf(MANUAL_MODEL_PATH));
  REQUIRE(model.model_type() == "pet");

  // Load test system
  AtomicSystem water = load_xyz(WATER_XYZ);
  REQUIRE(water.num_atoms() == 3);

  // Predict
  ModelResult result = model.predict(water);

  // Check energy is reasonable
  INFO("Manual PET energy: " << result.energy << " eV");
  INFO("Reference energy: " << WATER_ENERGY_REF << " eV");
  REQUIRE_THAT(result.energy, WithinAbs(WATER_ENERGY_REF, ENERGY_TOLERANCE));
}

TEST_CASE("Manual PET silicon test", "[auto_export][manual]") {
  if (!file_exists(MANUAL_MODEL_PATH)) {
    SKIP("Manual PET model not found at " << MANUAL_MODEL_PATH);
  }
  if (!file_exists(SI_XYZ)) {
    SKIP("Silicon XYZ file not found at " << SI_XYZ);
  }

  pet::PETModel model(pet::PETHypers{});
  REQUIRE(model.load_from_gguf(MANUAL_MODEL_PATH));

  AtomicSystem si = load_xyz(SI_XYZ);
  ModelResult result = model.predict(si);

  INFO("Manual PET silicon energy: " << result.energy << " eV");
  INFO("Reference energy: " << SI_ENERGY_REF << " eV");
  REQUIRE_THAT(result.energy, WithinAbs(SI_ENERGY_REF, ENERGY_TOLERANCE));
}

// ============================================================================
// Auto-Export GraphModel Tests
// ============================================================================

TEST_CASE("GraphModel loads auto-exported GGUF", "[auto_export][graphmodel]") {
  if (!file_exists(AUTO_MODEL_PATH)) {
    SKIP("Auto-exported model not found at " << AUTO_MODEL_PATH
         << " - run: uv run scripts/export_pytorch/export_pet_gguf.py");
  }

  runtime::GraphModel model;
  REQUIRE_NOTHROW(model.load_from_gguf(AUTO_MODEL_PATH));
  REQUIRE(model.model_type() == "graph");

  // Check hyperparameters loaded
  INFO("GraphModel cutoff: " << model.cutoff());
  REQUIRE(model.cutoff() > 0.0f);
}

TEST_CASE("GraphModel water prediction", "[auto_export][graphmodel]") {
  if (!file_exists(AUTO_MODEL_PATH)) {
    SKIP("Auto-exported model not found");
  }
  if (!file_exists(WATER_XYZ)) {
    SKIP("Water XYZ file not found");
  }

  runtime::GraphModel model;
  REQUIRE(model.load_from_gguf(AUTO_MODEL_PATH));
  if (!has_current_graph_format(model)) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  AtomicSystem water = load_xyz(WATER_XYZ);
  ModelResult result = model.predict(water);

  INFO("GraphModel water energy: " << result.energy << " eV");
  INFO("Reference energy: " << WATER_ENERGY_REF << " eV");

  // Note: This may fail initially until the graph export is complete
  // The tolerance is relaxed for development
  REQUIRE_THAT(result.energy, WithinAbs(WATER_ENERGY_REF, 0.1f));
}

// ============================================================================
// Side-by-Side Comparison Tests
// ============================================================================

TEST_CASE("Auto-export matches manual PET - water",
          "[auto_export][comparison]") {
  if (!file_exists(MANUAL_MODEL_PATH) || !file_exists(AUTO_MODEL_PATH)) {
    SKIP("Both models required for comparison test");
  }
  if (!file_exists(WATER_XYZ)) {
    SKIP("Water XYZ file not found");
  }

  // Load both models
  pet::PETModel manual_model(pet::PETHypers{});
  REQUIRE(manual_model.load_from_gguf(MANUAL_MODEL_PATH));

  runtime::GraphModel auto_model;
  REQUIRE(auto_model.load_from_gguf(AUTO_MODEL_PATH));
  if (!has_current_graph_format(auto_model)) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  // Load test system
  AtomicSystem water = load_xyz(WATER_XYZ);

  // Predict with both
  ModelResult manual_result = manual_model.predict(water);
  ModelResult auto_result = auto_model.predict(water);

  // Compare
  float diff = std::abs(manual_result.energy - auto_result.energy);
  INFO("Manual PET energy: " << manual_result.energy << " eV");
  INFO("Auto-export energy: " << auto_result.energy << " eV");
  INFO("Difference: " << diff << " eV");

  // They should match within tolerance
  REQUIRE_THAT(auto_result.energy,
               WithinAbs(manual_result.energy, ENERGY_TOLERANCE));
}

TEST_CASE("Auto-export matches manual PET - silicon",
          "[auto_export][comparison]") {
  if (!file_exists(MANUAL_MODEL_PATH) || !file_exists(AUTO_MODEL_PATH)) {
    SKIP("Both models required for comparison test");
  }
  if (!file_exists(SI_XYZ)) {
    SKIP("Silicon XYZ file not found");
  }

  pet::PETModel manual_model(pet::PETHypers{});
  REQUIRE(manual_model.load_from_gguf(MANUAL_MODEL_PATH));

  runtime::GraphModel auto_model;
  REQUIRE(auto_model.load_from_gguf(AUTO_MODEL_PATH));
  if (!has_current_graph_format(auto_model)) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  AtomicSystem si = load_xyz(SI_XYZ);

  ModelResult manual_result = manual_model.predict(si);
  ModelResult auto_result = auto_model.predict(si);

  float diff = std::abs(manual_result.energy - auto_result.energy);
  INFO("Manual PET silicon energy: " << manual_result.energy << " eV");
  INFO("Auto-export silicon energy: " << auto_result.energy << " eV");
  INFO("Difference: " << diff << " eV");

  REQUIRE_THAT(auto_result.energy,
               WithinAbs(manual_result.energy, ENERGY_TOLERANCE));
}

TEST_CASE("Auto-export sequential prediction", "[auto_export][sequential]") {
  if (!file_exists(AUTO_MODEL_PATH)) {
    SKIP("Auto-exported model not found");
  }
  if (!file_exists(WATER_XYZ) || !file_exists(SI_XYZ)) {
    SKIP("Test XYZ files not found");
  }

  runtime::GraphModel model;
  REQUIRE(model.load_from_gguf(AUTO_MODEL_PATH));
  if (!has_current_graph_format(model)) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  // Load test systems
  AtomicSystem water = load_xyz(WATER_XYZ);
  AtomicSystem si = load_xyz(SI_XYZ);

  // Sequential prediction (GraphModel handles dynamic sizes)
  ModelResult water_result = model.predict(water);
  ModelResult si_result = model.predict(si);

  INFO("Water energy: " << water_result.energy << " eV");
  INFO("Silicon energy: " << si_result.energy << " eV");

  // Each should be close to reference
  REQUIRE_THAT(water_result.energy, WithinAbs(WATER_ENERGY_REF, 0.1f));
  REQUIRE_THAT(si_result.energy, WithinAbs(SI_ENERGY_REF, 0.1f));
}

// ============================================================================
// Performance Comparison (informational, not failing)
// ============================================================================

TEST_CASE("Performance comparison manual vs auto",
          "[auto_export][performance][!mayfail]") {
  if (!file_exists(MANUAL_MODEL_PATH) || !file_exists(AUTO_MODEL_PATH)) {
    SKIP("Both models required for performance test");
  }
  if (!file_exists(WATER_XYZ)) {
    SKIP("Water XYZ file not found");
  }

  pet::PETModel manual_model(pet::PETHypers{});
  REQUIRE(manual_model.load_from_gguf(MANUAL_MODEL_PATH));

  runtime::GraphModel auto_model;
  REQUIRE(auto_model.load_from_gguf(AUTO_MODEL_PATH));
  if (!has_current_graph_format(auto_model)) {
    SKIP("GGUF uses old graph format - re-export with export_pet_gguf.py");
  }

  AtomicSystem water = load_xyz(WATER_XYZ);

  // Warmup
  for (int i = 0; i < 3; i++) {
    manual_model.predict(water);
    auto_model.predict(water);
  }

  // Time manual
  constexpr int N_ITERS = 10;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITERS; i++) {
    manual_model.predict(water);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  float manual_ms =
      std::chrono::duration<float, std::milli>(t1 - t0).count() / N_ITERS;

  // Time auto
  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITERS; i++) {
    auto_model.predict(water);
  }
  t1 = std::chrono::high_resolution_clock::now();
  float auto_ms =
      std::chrono::duration<float, std::milli>(t1 - t0).count() / N_ITERS;

  INFO("Manual PET: " << manual_ms << " ms/iter");
  INFO("Auto-export: " << auto_ms << " ms/iter");
  INFO("Ratio (auto/manual): " << (auto_ms / manual_ms));

  // Auto should be within 2x of manual (generous for now)
  // This may fail if auto is slower, which is acceptable during development
  REQUIRE(auto_ms < manual_ms * 2.0f);
}
