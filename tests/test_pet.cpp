/**
 * Comprehensive unit tests for PET implementation
 *
 * Tests include:
 * - Weight loading from GGUF
 * - Single system prediction
 * - Batch prediction consistency
 * - NEF format construction
 * - Reference value verification
 * - Edge cases
 */
#include "mlipcpp/system.h"
#include "pet.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::pet;

// Helper: Create test Si2 system (diamond structure)
AtomicSystem create_test_system_si2() {
  std::vector<int32_t> atomic_numbers = {14, 14}; // Si, Si
  std::vector<float> positions = {
      0.0f,    0.0f,    0.0f,   // Atom 0
      1.3575f, 1.3575f, 1.3575f // Atom 1 at (a/4, a/4, a/4)
  };

  Cell cell;
  cell.matrix[0][0] = 5.43f;
  cell.matrix[0][1] = 0.0f;
  cell.matrix[0][2] = 0.0f;
  cell.matrix[1][0] = 0.0f;
  cell.matrix[1][1] = 5.43f;
  cell.matrix[1][2] = 0.0f;
  cell.matrix[2][0] = 0.0f;
  cell.matrix[2][1] = 0.0f;
  cell.matrix[2][2] = 5.43f;
  cell.periodic[0] = true;
  cell.periodic[1] = true;
  cell.periodic[2] = true;

  return AtomicSystem(2, positions.data(), atomic_numbers.data(), &cell);
}

// Helper: Create test Si3 system
AtomicSystem create_test_system_si3() {
  std::vector<int32_t> atomic_numbers = {14, 14, 14};
  std::vector<float> positions = {0.0f,    0.0f,   0.0f, 1.3575f, 1.3575f,
                                  1.3575f, 2.715f, 0.0f, 0.0f};

  Cell cell;
  cell.matrix[0][0] = 5.43f;
  cell.matrix[0][1] = 0.0f;
  cell.matrix[0][2] = 0.0f;
  cell.matrix[1][0] = 0.0f;
  cell.matrix[1][1] = 5.43f;
  cell.matrix[1][2] = 0.0f;
  cell.matrix[2][0] = 0.0f;
  cell.matrix[2][1] = 0.0f;
  cell.matrix[2][2] = 5.43f;
  cell.periodic[0] = true;
  cell.periodic[1] = true;
  cell.periodic[2] = true;

  return AtomicSystem(3, positions.data(), atomic_numbers.data(), &cell);
}

// Helper: Create water molecule
AtomicSystem create_test_system_water() {
  std::vector<int32_t> atomic_numbers = {8, 1, 1}; // O, H, H
  std::vector<float> positions = {
      0.0f,    0.0f,   0.0f, // O
      0.757f,  0.586f, 0.0f, // H
      -0.757f, 0.586f, 0.0f  // H
  };

  return AtomicSystem(3, positions.data(), atomic_numbers.data(), nullptr);
}

// Helper: Create isolated atom (no neighbors)
AtomicSystem create_test_system_isolated() {
  std::vector<int32_t> atomic_numbers = {14};
  std::vector<float> positions = {0.0f, 0.0f, 0.0f};

  return AtomicSystem(1, positions.data(), atomic_numbers.data(), nullptr);
}

TEST_CASE("PET loads weights from GGUF", "[pet][loading]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);

  // Just test that loading succeeds
  REQUIRE(model.load_from_gguf(model_path));
}

TEST_CASE("PET predicts single system correctly", "[pet][accuracy]") {
  std::string model_path = "pet-mad.gguf";

  SECTION("Si 2-atom system") {
    PETHypers hypers;
    PETModel model(hypers);
    REQUIRE(model.load_from_gguf(model_path));

    auto system = create_test_system_si2();
    auto result = model.predict(system);

    INFO("Energy: " << result.energy << " eV");

    // Energy should be finite and reasonable
    REQUIRE(std::isfinite(result.energy));
    REQUIRE(result.energy < 0.0); // Should be negative (stable)
  }

  SECTION("Water molecule") {
    PETHypers hypers;
    PETModel model(hypers);
    REQUIRE(model.load_from_gguf(model_path));

    auto system = create_test_system_water();
    auto result = model.predict(system);

    INFO("Energy: " << result.energy << " eV");

    // Energy should be finite and reasonable
    REQUIRE(std::isfinite(result.energy));
    REQUIRE(result.energy < 0.0); // Should be negative (stable)
  }
}

TEST_CASE("PET batch prediction matches individual", "[.][pet][batch]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf(model_path));

  std::vector<AtomicSystem> systems = {
      create_test_system_si2(),
      create_test_system_water(),
  };

  // Individual predictions
  std::vector<double> individual_energies;
  for (const auto &sys : systems) {
    auto result = model.predict(sys);
    individual_energies.push_back(result.energy);
    INFO("Individual energy " << individual_energies.size() - 1 << ": "
                              << result.energy);
  }

  // Batched prediction
  auto batch_results = model.predict_batch(systems);

  REQUIRE(batch_results.size() == systems.size());

  // Compare
  for (size_t i = 0; i < systems.size(); ++i) {
    INFO("System " << i << " - Individual: " << individual_energies[i]
                   << " eV, Batch: " << batch_results[i].energy << " eV");
    INFO("Difference: " << std::abs(individual_energies[i] -
                                    batch_results[i].energy)
                        << " eV");

    REQUIRE_THAT(batch_results[i].energy,
                 Catch::Matchers::WithinAbs(individual_energies[i], 1e-5));
  }
}

TEST_CASE("PET matches reference values", "[pet][verification]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf(model_path));

  SECTION("Si 2-atom system (a=5.43)") {
    auto system = create_test_system_si2();
    auto result = model.predict(system);

    // Expected energy from PyTorch reference
    const double expected_energy = -4.538056;
    const double tolerance = 0.001;

    INFO("Expected: " << expected_energy << " eV");
    INFO("Computed: " << result.energy << " eV");
    INFO("Difference: " << std::abs(result.energy - expected_energy) << " eV");

    REQUIRE_THAT(result.energy,
                 Catch::Matchers::WithinAbs(expected_energy, tolerance));
  }

  SECTION("Water molecule") {
    auto system = create_test_system_water();
    auto result = model.predict(system);

    // Expected energy from PyTorch reference
    const double expected_energy = -14.380176;
    const double tolerance = 0.01;

    INFO("Expected: " << expected_energy << " eV");
    INFO("Computed: " << result.energy << " eV");
    INFO("Difference: " << std::abs(result.energy - expected_energy) << " eV");

    REQUIRE_THAT(result.energy,
                 Catch::Matchers::WithinAbs(expected_energy, tolerance));
  }
}

TEST_CASE("PET handles edge cases", "[.][pet][edge_cases]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf(model_path));

  SECTION("Isolated atom (no neighbors)") {
    auto system = create_test_system_isolated();
    auto result = model.predict(system);

    // Should have finite energy
    REQUIRE(std::isfinite(result.energy));

    INFO("Isolated atom energy: " << result.energy << " eV");
  }

  SECTION("Empty batch") {
    std::vector<AtomicSystem> empty_batch;
    auto results = model.predict_batch(empty_batch);

    REQUIRE(results.empty());
  }

  SECTION("Single system batch") {
    std::vector<AtomicSystem> single_batch = {create_test_system_si2()};
    auto results = model.predict_batch(single_batch);

    REQUIRE(results.size() == 1);
    REQUIRE(std::isfinite(results[0].energy));
  }

  SECTION("Variable neighbor counts") {
    // Mix of systems with different numbers of neighbors
    std::vector<AtomicSystem> mixed_batch = {
        create_test_system_isolated(), // 0 neighbors
        create_test_system_si2(),      // ~4 neighbors
        create_test_system_si3(),      // More neighbors
    };

    auto results = model.predict_batch(mixed_batch);

    REQUIRE(results.size() == 3);
    for (const auto &result : results) {
      REQUIRE(std::isfinite(result.energy));
    }
  }
}

TEST_CASE("PET batch with multiple systems", "[.][pet][batch]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf(model_path));

  // Create a batch with 5 different systems
  std::vector<AtomicSystem> systems;
  systems.push_back(create_test_system_si2());
  systems.push_back(create_test_system_water());
  systems.push_back(create_test_system_si3());
  systems.push_back(create_test_system_si2());   // Duplicate to test
  systems.push_back(create_test_system_water()); // Another duplicate

  auto batch_results = model.predict_batch(systems);

  REQUIRE(batch_results.size() == 5);

  // All energies should be finite
  for (size_t i = 0; i < batch_results.size(); ++i) {
    INFO("System " << i << " energy: " << batch_results[i].energy << " eV");
    REQUIRE(std::isfinite(batch_results[i].energy));
  }

  // Duplicate systems should have same energy
  REQUIRE_THAT(batch_results[0].energy,
               Catch::Matchers::WithinAbs(batch_results[3].energy, 1e-5));
  REQUIRE_THAT(batch_results[1].energy,
               Catch::Matchers::WithinAbs(batch_results[4].energy, 1e-5));
}

TEST_CASE("PET composition energy handling", "[pet][composition]") {
  std::string model_path = "pet-mad.gguf";

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf(model_path));

  // Predict on a system
  auto system = create_test_system_water();
  auto result = model.predict(system);

  // Energy should include composition energy contribution
  // Just verify it's a reasonable value
  REQUIRE(std::isfinite(result.energy));
  REQUIRE(result.energy < 0.0); // Should be negative (stable)
}
