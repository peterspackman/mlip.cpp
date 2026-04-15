/**
 * Backend benchmark for mlipcpp
 *
 * Benchmarks a single backend across Si supercell sizes.
 * Use scripts/benchmark_backends.sh to compare all backends.
 *
 * Usage: backend_benchmark <model.gguf> [--backend B] [--warmup N] [--iterations N]
 */

#include "../src/models/pet/pet.h"
#include "../src/runtime/graph_model.h"
#include "core/backend.h"
#include "core/gguf_loader.h"
#include "core/log.h"
#include "mlipcpp/model.h"
#include "mlipcpp/system.h"
#include <memory>
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
    std::cerr << "  --max-atoms N   Cap supercell size (default: 1024)\n";
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
  int max_atoms = 1024;
  BackendPreference backend_pref = BackendPreference::Auto;
  std::string backend_name = "auto";

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
    } else if (arg == "--max-atoms" && i + 1 < argc) {
      max_atoms = std::stoi(argv[++i]);
    } else if (arg == "--backend" && i + 1 < argc) {
      backend_name = argv[++i];
      try {
        backend_pref = parse_backend_preference(backend_name);
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
      }
    }
  }

  // System sizes to test (nx, ny, nz) -> 2 * nx * ny * nz atoms.
  // Filtered by --max-atoms.
  std::vector<std::array<int, 3>> all_sizes = {
      {1, 1, 1},   {2, 2, 2}, {4, 4, 2},  {4, 4, 4},   {4, 4, 8},
      {4, 8, 8},   {8, 8, 8}, {16, 8, 8}, {16, 16, 8},
  };
  std::vector<std::array<int, 3>> sizes;
  for (const auto &s : all_sizes) {
    if (2 * s[0] * s[1] * s[2] <= max_atoms) sizes.push_back(s);
  }

  // Load model via architecture dispatch
  std::unique_ptr<Model> model;
  try {
    GGUFLoader probe(model_path);
    std::string arch = probe.get_string("general.architecture", "");
    if (arch == "pet") {
      auto pm = std::make_unique<pet::PETModel>(pet::PETHypers{});
      pm->set_backend_preference(backend_pref);
      if (!pm->load_from_gguf(model_path)) {
        std::cerr << "Failed to load PET model\n";
        return 1;
      }
      model = std::move(pm);
    } else if (arch == "pet-graph") {
      auto gm = std::make_unique<runtime::GraphModel>();
      gm->set_backend_preference(backend_pref);
      if (!gm->load_from_gguf(model_path)) {
        std::cerr << "Failed to load graph model\n";
        return 1;
      }
      model = std::move(gm);
    } else {
      std::cerr << "Unsupported architecture: " << arch << "\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Backend error: " << e.what() << "\n";
    return 1;
  }

  auto *pet_model = dynamic_cast<pet::PETModel *>(model.get());
  if (compute_nc && !pet_model) {
    std::cerr << "--nc-forces requires a PET model\n";
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
      if (pet_model) pet_model->predict_batch({system}, compute_forces, compute_nc);
      else model->predict(system, compute_forces);
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    ModelResult last_result;
    for (int i = 0; i < iterations; ++i) {
      if (pet_model) {
        last_result = pet_model->predict_batch({system}, compute_forces, compute_nc)[0];
      } else {
        last_result = model->predict(system, compute_forces);
      }
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
