#include "mlipcpp/system.h"
#include "pet.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <fmt/core.h>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::pet;

// Python PETMADCalculator reference values for water.xyz
// Structure: O at (0,0,0), H at (0.757, 0.586, 0), H at (-0.757, 0.586, 0)
static const float WATER_REF_FORCES[9] = {
    0.017518f,  -0.932378f, 0.005716f,  // O
    0.598539f,  0.463137f,  -0.000623f, // H1
    -0.616057f, 0.469241f,  -0.005093f  // H2
};

// Python PETMADCalculator reference values for Si 2-atom crystal
// Structure: Si at (0,0,0), Si at (1.3575, 1.3575, 1.3575), cell=5.43 A cubic
static const float SI_REF_FORCES[6] = {
    0.603346f,  0.602278f,  0.602655f, // Si1
    -0.603346f, -0.602278f, -0.602655f // Si2
};

static const float SI_REF_STRESS[6] = {
    0.005116f, 0.005107f, 0.005110f, // xx, yy, zz
    0.005108f, 0.005113f, 0.005111f  // yz, xz, xy
};

TEST_CASE("PET forces match Python reference (water molecule)",
          "[pet][gradient]") {
  // Water molecule from examples/water.xyz
  std::vector<int32_t> atomic_numbers = {8, 1, 1}; // O, H, H
  std::vector<float> positions = {
      0.0f,    0.0f,   0.0f, // O
      0.757f,  0.586f, 0.0f, // H1
      -0.757f, 0.586f, 0.0f  // H2
  };

  AtomicSystem system(3, positions.data(), atomic_numbers.data(), nullptr);

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf("pet-mad.gguf"));

  // Run prediction with forces
  auto result = model.predict(system, true);

  // Check that we got forces back
  REQUIRE(result.has_forces);
  REQUIRE(result.forces.size() == 9); // 3 atoms * 3 coords

  // Compare against Python reference
  fmt::print("Water molecule forces comparison vs Python PETMADCalculator:\n");
  float max_error = 0.0f;

  for (int atom = 0; atom < 3; ++atom) {
    for (int coord = 0; coord < 3; ++coord) {
      int idx = atom * 3 + coord;
      float cpp_force = result.forces[idx];
      float py_force = WATER_REF_FORCES[idx];
      float error = std::abs(cpp_force - py_force);

      fmt::print("  Atom {} coord {}: C++=%10.6f Python=%10.6f error=%.6f\n",
                 atom, coord, cpp_force, py_force, error);

      max_error = std::max(max_error, error);
    }
  }

  fmt::print("Max absolute error: %.6f eV/A\n", max_error);
  INFO("Max absolute error: " << max_error << " eV/A");

  // Require very tight tolerance - should match to ~6 decimal places
  REQUIRE(max_error < 1e-4f);
}

TEST_CASE("PET forces match Python reference (Si crystal)", "[pet][gradient]") {
  // Si 2-atom system (periodic)
  std::vector<int32_t> atomic_numbers = {14, 14};
  std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.3575f, 1.3575f, 1.3575f};

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

  AtomicSystem system(2, positions.data(), atomic_numbers.data(), &cell);

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf("pet-mad.gguf"));

  // Get analytical forces
  auto result = model.predict(system, true);
  REQUIRE(result.has_forces);
  REQUIRE(result.forces.size() == 6); // 2 atoms * 3 coords

  // Compare against Python reference
  fmt::print("Si crystal forces comparison vs Python PETMADCalculator:\n");
  float max_error = 0.0f;

  for (int atom = 0; atom < 2; ++atom) {
    for (int coord = 0; coord < 3; ++coord) {
      int idx = atom * 3 + coord;
      float cpp_force = result.forces[idx];
      float py_force = SI_REF_FORCES[idx];
      float error = std::abs(cpp_force - py_force);

      fmt::print("  Atom {} coord {}: C++=%10.6f Python=%10.6f error=%.6f\n",
                 atom, coord, cpp_force, py_force, error);

      max_error = std::max(max_error, error);
    }
  }

  fmt::print("Max absolute error: %.6f eV/A\n", max_error);
  INFO("Max absolute error: " << max_error << " eV/A");

  // Require very tight tolerance
  REQUIRE(max_error < 1e-4f);
}

TEST_CASE("PET stress matches Python reference (Si crystal)",
          "[pet][gradient][stress]") {
  // Si 2-atom system (periodic)
  std::vector<int32_t> atomic_numbers = {14, 14};
  std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.3575f, 1.3575f, 1.3575f};

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

  AtomicSystem system(2, positions.data(), atomic_numbers.data(), &cell);

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf("pet-mad.gguf"));

  // Get forces and stress
  auto result = model.predict(system, true);

  // Check that we got stress back
  REQUIRE(result.has_stress);
  REQUIRE(result.stress.size() == 6);

  // Compare against Python reference
  fmt::print("Si crystal stress comparison vs Python PETMADCalculator:\n");
  const char *labels[] = {"xx", "yy", "zz", "yz", "xz", "xy"};
  float max_error = 0.0f;

  for (int i = 0; i < 6; ++i) {
    float cpp_stress = result.stress[i];
    float py_stress = SI_REF_STRESS[i];
    float error = std::abs(cpp_stress - py_stress);

    fmt::print("  {}: C++=%10.6f Python=%10.6f error=%.6f\n", labels[i],
               cpp_stress, py_stress, error);

    max_error = std::max(max_error, error);
  }

  fmt::print("Max absolute error: %.6f eV/A^3\n", max_error);
  INFO("Max absolute error: " << max_error << " eV/A^3");

  // Require very tight tolerance
  REQUIRE(max_error < 1e-4f);
}

TEST_CASE("PET forces sum to zero (momentum conservation)", "[pet][gradient]") {
  // Water molecule from examples/water.xyz
  std::vector<int32_t> atomic_numbers = {8, 1, 1};
  std::vector<float> positions = {0.0f, 0.0f,    0.0f,   0.757f, 0.586f,
                                  0.0f, -0.757f, 0.586f, 0.0f};

  AtomicSystem system(3, positions.data(), atomic_numbers.data(), nullptr);

  PETHypers hypers;
  PETModel model(hypers);
  REQUIRE(model.load_from_gguf("pet-mad.gguf"));

  auto result = model.predict(system, true);
  REQUIRE(result.has_forces);

  // Sum of all forces should be zero (conservation of momentum)
  float total_fx = 0.0f, total_fy = 0.0f, total_fz = 0.0f;
  for (int i = 0; i < 3; ++i) {
    total_fx += result.forces[i * 3 + 0];
    total_fy += result.forces[i * 3 + 1];
    total_fz += result.forces[i * 3 + 2];
  }

  float total_mag = std::sqrt(total_fx * total_fx + total_fy * total_fy +
                              total_fz * total_fz);
  fmt::print("Force sum: (%.6f, %.6f, %.6f) magnitude=%.6f\n", total_fx,
             total_fy, total_fz, total_mag);
  INFO("Total force magnitude: " << total_mag);

  // Forces should sum to approximately zero
  REQUIRE(total_mag < 0.01f);
}
