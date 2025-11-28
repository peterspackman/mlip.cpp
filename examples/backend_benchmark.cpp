/**
 * Backend benchmark for mlipcpp
 *
 * Benchmarks a single backend across Si supercell sizes.
 * Use scripts/benchmark_backends.sh to compare all backends.
 *
 * Usage: backend_benchmark <model.gguf> [--backend B] [--warmup N] [--iterations N]
 */

#include "../src/models/pet/pet.h"
#include "core/log.h"
#include "mlipcpp/system.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace mlipcpp;

// Si diamond structure builder
struct SiDiamond {
  static AtomicSystem build(int nx, int ny, int nz) {
    const float a = 5.43f; // lattice constant

    // Diamond basis (fractional coordinates)
    static constexpr std::array<std::array<float, 3>, 2> basis = {{
        {0.0f, 0.0f, 0.0f},
        {0.25f, 0.25f, 0.25f},
    }};

    int n_atoms = 2 * nx * ny * nz;
    std::vector<float> positions(n_atoms * 3);
    std::vector<int32_t> atomic_numbers(n_atoms, 14); // Si = 14

    int idx = 0;
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          for (const auto &b : basis) {
            positions[idx * 3 + 0] = (ix + b[0]) * a;
            positions[idx * 3 + 1] = (iy + b[1]) * a;
            positions[idx * 3 + 2] = (iz + b[2]) * a;
            ++idx;
          }
        }
      }
    }

    // Create cell
    float lattice[3][3] = {
        {nx * a, 0.0f, 0.0f},
        {0.0f, ny * a, 0.0f},
        {0.0f, 0.0f, nz * a},
    };
    Cell cell(lattice, true, true, true);

    return AtomicSystem(n_atoms, positions.data(), atomic_numbers.data(),
                        &cell);
  }
};

int main(int argc, char **argv) {
  log::suppress_ggml_logging();
  log::set_level(log::Level::Off);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --backend B     Backend: auto, cpu, metal, cuda, etc. (default: auto)\n";
    std::cerr << "  --warmup N      Warmup iterations (default: 2)\n";
    std::cerr << "  --iterations N  Timed iterations (default: 10)\n";
    std::cerr << "  --no-forces     Benchmark energy only (no forces)\n";
    std::cerr << "  --nc-forces     Use non-conservative forces (forward pass only)\n";
    std::cerr << "  --csv           Output CSV format for scripting\n";
    return 1;
  }

  std::string model_path = argv[1];
  int warmup = 2;
  int iterations = 10;
  bool compute_forces = true;
  bool compute_nc = false;
  bool csv_output = false;
  pet::BackendPreference backend_pref = pet::BackendPreference::Auto;
  std::string backend_name = "auto";

  // Backend name lookup
  static const std::unordered_map<std::string_view, pet::BackendPreference>
      backend_map = {
          {"auto", pet::BackendPreference::Auto},
          {"cpu", pet::BackendPreference::CPU},
          {"cuda", pet::BackendPreference::CUDA},
          {"hip", pet::BackendPreference::HIP},
          {"metal", pet::BackendPreference::Metal},
          {"vulkan", pet::BackendPreference::Vulkan},
          {"sycl", pet::BackendPreference::SYCL},
          {"cann", pet::BackendPreference::CANN},
      };

  // Parse options
  for (int i = 2; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--no-forces") {
      compute_forces = false;
    } else if (arg == "--nc-forces") {
      compute_nc = true;
      compute_forces = false;  // nc-forces replaces gradient forces
    } else if (arg == "--csv") {
      csv_output = true;
    } else if (arg == "--backend" && i + 1 < argc) {
      backend_name = argv[++i];
      auto it = backend_map.find(backend_name);
      if (it != backend_map.end()) {
        backend_pref = it->second;
      } else {
        std::cerr << "Unknown backend: " << backend_name << "\n";
        return 1;
      }
    }
  }

  // System sizes to test (nx, ny, nz) -> 2 * nx * ny * nz atoms
  std::vector<std::array<int, 3>> sizes = {
      {1, 1, 1},  // 2 atoms
      {2, 2, 2},  // 16 atoms
      {4, 4, 2},  // 64 atoms
      {4, 4, 4},  // 128 atoms
      {4, 4, 8},  // 256 atoms
      {4, 8, 8},  // 512 atoms
      {8, 8, 8},  // 1024 atoms
      {16, 8, 8},  // 2048 atoms
      {16, 16, 8},  // 4096 atoms
  };

  // Load model once
  pet::PETHypers hypers;
  pet::PETModel model(hypers);

  // Set backend preference BEFORE loading (backend is initialized during load)
  model.set_backend_preference(backend_pref);

  try {
    if (!model.load_from_gguf(model_path)) {
      std::cerr << "Failed to load model: " << model_path << "\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Backend error: " << e.what() << "\n";
    return 1;
  }

  // Determine mode string
  std::string mode_str = "Energy";
  if (compute_forces) {
    mode_str += " + Forces (gradient)";
  } else if (compute_nc) {
    mode_str += " + NC-Forces (forward)";
  } else {
    mode_str += " only";
  }

  if (csv_output) {
    // CSV header
    std::cout << "backend,atoms,time_ms,us_per_atom,energy\n";
  } else {
    std::cout << "Backend: " << backend_name << "\n";
    std::cout << "Mode: " << mode_str << "\n";
    std::cout << "Warmup: " << warmup << ", Iterations: " << iterations << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(8) << "Atoms" << std::setw(12) << "Time (ms)"
              << std::setw(12) << "us/atom" << std::setw(15) << "Energy (eV)" << "\n";
    std::cout << std::string(60, '-') << "\n";
  }

  for (const auto &size : sizes) {
    auto system = SiDiamond::build(size[0], size[1], size[2]);
    int n_atoms = system.num_atoms();

    // Warmup
    for (int i = 0; i < warmup; ++i) {
      model.predict_batch({system}, compute_forces, compute_nc);
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    ModelResult last_result;
    for (int i = 0; i < iterations; ++i) {
      auto results = model.predict_batch({system}, compute_forces, compute_nc);
      last_result = results[0];
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        iterations;
    double us_per_atom = (time_ms * 1000.0) / n_atoms;

    if (csv_output) {
      std::cout << backend_name << "," << n_atoms << "," << std::fixed
                << std::setprecision(2) << time_ms << "," << std::setprecision(2)
                << us_per_atom << "," << std::setprecision(6)
                << last_result.energy << "\n";
    } else {
      std::cout << std::setw(8) << n_atoms << std::setw(12) << std::fixed
                << std::setprecision(2) << time_ms << std::setw(12)
                << std::setprecision(2) << us_per_atom << std::setw(15)
                << std::setprecision(4) << last_result.energy << "\n";
    }
  }

  if (!csv_output) {
    std::cout << std::string(60, '-') << "\n";
  }

  return 0;
}
