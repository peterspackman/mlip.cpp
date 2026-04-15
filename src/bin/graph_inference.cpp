/**
 * Graph-based inference on XYZ files using auto-exported PET models.
 *
 * Usage:
 *   graph_inference <model.gguf> <xyz_file> [--forces] [--backend <name>]
 */

#include "core/backend.h"
#include "core/gguf_loader.h"
#include "mlipcpp/io.h"
#include "mlipcpp/model.h"
#include "mlipcpp/system.h"
#include "models/pet/pet.h"
#include "runtime/graph_model.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

using namespace mlipcpp;

namespace {

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <model.gguf> <xyz_file> [--forces] [--backend <name>]\n\n"
            << "Options:\n"
            << "  --forces          Compute forces via backward pass\n"
            << "  --backend <name>  auto|cpu|metal|webgpu|cuda|hip|vulkan "
               "(default: auto)\n";
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::string xyz_path = argv[2];
  bool compute_forces = false;
  std::string backend_name = "auto";

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--forces") {
      compute_forces = true;
    } else if (arg == "--backend" && i + 1 < argc) {
      backend_name = argv[++i];
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  try {
    BackendPreference pref = parse_backend_preference(backend_name);

    // Route through load_model() for architecture dispatch, but for graph
    // models we want to set the backend preference before loading weights.
    GGUFLoader probe(model_path);
    std::string arch = probe.get_string("general.architecture", "");

    std::unique_ptr<Model> model;
    if (arch == "pet-graph") {
      auto gm = std::make_unique<runtime::GraphModel>();
      gm->set_backend_preference(pref);
      if (!gm->load_from_gguf(model_path)) {
        throw std::runtime_error("Failed to load graph model");
      }
      model = std::move(gm);
    } else {
      model = load_model(model_path);
      if (pref != BackendPreference::Auto &&
          pref != BackendPreference::CPU) {
        std::cerr << "Warning: --backend ignored for architecture '" << arch
                  << "'\n";
      }
    }

    AtomicSystem system = io::read_xyz(xyz_path);
    std::cout << "Input: " << xyz_path << " (" << system.num_atoms()
              << " atoms)\n";
    std::cout << "Model cutoff: " << model->cutoff() << " A\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    ModelResult result = model->predict(system, compute_forces);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Results ===\n";
    std::cout << "Total energy: " << result.energy << " eV\n";

    if (result.has_forces) {
      std::cout << "\nForces (eV/A):\n";
      float fsum[3] = {0, 0, 0};
      for (size_t i = 0; i < system.num_atoms(); i++) {
        std::cout << "  Atom " << i << ": [" << std::setw(12)
                  << result.forces[i * 3 + 0] << ", " << std::setw(12)
                  << result.forces[i * 3 + 1] << ", " << std::setw(12)
                  << result.forces[i * 3 + 2] << "]\n";
        for (int k = 0; k < 3; k++) fsum[k] += result.forces[i * 3 + k];
      }
      std::cout << "  Force sum: [" << fsum[0] << ", " << fsum[1] << ", "
                << fsum[2] << "]\n";
    }

    std::cout << "\nCompute time: " << std::fixed << std::setprecision(1)
              << ms << " ms\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
