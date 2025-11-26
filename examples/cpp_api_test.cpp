/**
 * @file cpp_api_test.cpp
 * @brief C++ API test for mlipcpp
 *
 * Demonstrates the modern C++ API with RAII and span-based interfaces.
 */

#include "mlipcpp/mlipcpp.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];

  // Suppress verbose logging
  mlipcpp::suppress_logging();

  printf("mlipcpp C++ API test\n");
  printf("Version: %s\n", mlipcpp::version());
  printf("Backend: %s\n\n", mlipcpp::get_backend_name());

  try {
    // Load model with default options (auto backend selection)
    printf("Loading model: %s\n", model_path);
    mlipcpp::Predictor model(model_path);

    printf("Model type: %.*s\n", static_cast<int>(model.model_type().size()),
           model.model_type().data());
    printf("Cutoff: %.2f Angstroms\n\n", model.cutoff());

    // Test 1: Non-periodic water molecule
    printf("=== Test 1: Non-periodic water molecule ===\n");
    {
      // Water molecule geometry (O-H bond ~0.96 A, H-O-H angle ~104.5deg)
      std::vector<float> positions = {
          0.000f,  0.000f, 0.117f,  // O
          -0.756f, 0.000f, -0.468f, // H
          0.756f,  0.000f, -0.468f  // H
      };
      std::vector<int32_t> atomic_numbers = {8, 1, 1}; // O, H, H

      // Energy only
      auto result = model.predict(positions, atomic_numbers, false);
      printf("Energy (no forces): %.6f eV\n", result.energy);

      // With forces
      result = model.predict(positions, atomic_numbers, true);
      printf("Energy: %.6f eV\n", result.energy);
      printf("Forces:\n");
      for (size_t i = 0; i < atomic_numbers.size(); ++i) {
        auto f = result.force(i);
        printf("  Atom %zu: [%10.6f, %10.6f, %10.6f] eV/A\n", i, f[0], f[1],
               f[2]);
      }

      // Check force sum (should be ~0 for isolated system)
      float fx_sum = 0, fy_sum = 0, fz_sum = 0;
      for (size_t i = 0; i < atomic_numbers.size(); ++i) {
        fx_sum += result.forces[i * 3 + 0];
        fy_sum += result.forces[i * 3 + 1];
        fz_sum += result.forces[i * 3 + 2];
      }
      printf("Force sum: [%.2e, %.2e, %.2e] (should be ~0)\n\n", fx_sum, fy_sum,
             fz_sum);
    }

    // Test 2: Periodic Si crystal (diamond structure, 2 atoms)
    printf("=== Test 2: Periodic Si crystal ===\n");
    {
      // Diamond Si primitive cell
      const float a = 5.43f; // Lattice constant in Angstroms

      std::vector<float> positions = {0.0f,      0.0f,      0.0f,
                                      a * 0.25f, a * 0.25f, a * 0.25f};
      std::vector<int32_t> atomic_numbers = {14, 14};

      // FCC lattice vectors
      std::array<float, 9> cell = {a * 0.5f, a * 0.5f, 0.0f, 0.0f,    a * 0.5f,
                                   a * 0.5f, a * 0.5f, 0.0f, a * 0.5f};
      std::array<bool, 3> pbc = {true, true, true};

      auto result = model.predict(positions, atomic_numbers, cell, pbc, true);
      printf("Energy: %.6f eV\n", result.energy);
      printf("Energy per atom: %.6f eV/atom\n", result.energy / 2.0f);

      if (result.has_forces()) {
        printf("Forces:\n");
        for (size_t i = 0; i < atomic_numbers.size(); ++i) {
          auto f = result.force(i);
          printf("  Atom %zu: [%10.6f, %10.6f, %10.6f] eV/A\n", i, f[0], f[1],
                 f[2]);
        }
      }

      if (result.has_stress()) {
        printf("Stress (Voigt): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] eV/A^3\n",
               result.stress[0], result.stress[1], result.stress[2],
               result.stress[3], result.stress[4], result.stress[5]);
      }
      printf("\n");
    }

    // Test 3: Raw pointer interface (same water molecule)
    printf("=== Test 3: Raw pointer interface ===\n");
    {
      float positions[] = {
          0.000f,  0.000f, 0.117f,  // O
          -0.756f, 0.000f, -0.468f, // H
          0.756f,  0.000f, -0.468f  // H
      };
      int32_t atomic_numbers[] = {8, 1, 1};

      auto result =
          model.predict(3, positions, atomic_numbers, nullptr, nullptr, true);
      printf("Energy: %.6f eV (same as Test 1)\n", result.energy);
      printf("Forces[0]: [%.6f, %.6f, %.6f]\n\n", result.forces[0],
             result.forces[1], result.forces[2]);
    }

    // Test 4: Move semantics (same Si crystal)
    printf("=== Test 4: Move semantics ===\n");
    {
      mlipcpp::Predictor model2 = std::move(model);

      const float a = 5.43f;
      std::vector<float> pos = {0.0f,      0.0f,      0.0f,
                                a * 0.25f, a * 0.25f, a * 0.25f};
      std::vector<int32_t> z = {14, 14};
      std::array<float, 9> cell = {a * 0.5f, a * 0.5f, 0.0f, 0.0f,    a * 0.5f,
                                   a * 0.5f, a * 0.5f, 0.0f, a * 0.5f};
      std::array<bool, 3> pbc = {true, true, true};

      auto result = model2.predict(pos, z, cell, pbc, false);
      printf("Energy after move: %.6f eV (same as Test 2)\n\n", result.energy);
    }

    printf("All tests passed!\n");
    return 0;

  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
