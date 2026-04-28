/**
 * Graph-based inference on XYZ files using auto-exported PET models.
 *
 * Usage:
 *   graph_inference <model.gguf> <xyz_file> [--forces] [--stress] [--fd-stress]
 *                   [--backend <name>]
 */

#include "core/backend.h"
#include "core/gguf_loader.h"
#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.hpp"
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
            << " <model.gguf> <xyz_file> [--forces] [--stress] [--fd-stress] "
               "[--backend <name>]\n\n"
            << "Options:\n"
            << "  --forces          Compute forces via backward pass\n"
            << "  --stress          Compute stress (Voigt, eV/A^3) via the\n"
            << "                    analytical chain-rule path; implies --forces\n"
            << "  --fd-stress       Compute stress via finite differences on\n"
            << "                    energy under symmetric Voigt strain (slow,\n"
            << "                    intended as a validation oracle); implies\n"
            << "                    --stress and --forces\n"
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
  bool compute_stress = false;
  bool fd_stress = false;
  std::string backend_name = "auto";

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--forces") {
      compute_forces = true;
    } else if (arg == "--stress") {
      compute_stress = true;
      compute_forces = true;  // stress requires the autograd backward
    } else if (arg == "--fd-stress") {
      compute_stress = true;
      compute_forces = true;
      fd_stress = true;
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

    // Use the Model interface for the analytical path; route through the
    // higher-level Predictor only when --fd-stress is requested (it owns the
    // FD strain logic).
    float energy = 0.0f;
    std::vector<float> forces;
    std::vector<float> stress;
    bool has_forces = false;
    bool has_stress = false;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (fd_stress) {
      // Reload via Predictor so we get the FD path. (The Backend pref applies
      // through the global mlipcpp::set_backend path inside Predictor.)
      mlipcpp::ModelOptions opts;
      mlipcpp::Predictor pred(model_path, opts);
      mlipcpp::PredictOptions popts;
      popts.compute_forces = compute_forces;
      popts.compute_stress = compute_stress;
      popts.fd_stress = true;

      const Cell *cell_ptr = system.cell();
      float lattice_buf[9];
      bool pbc_buf[3] = {true, true, true};
      const float *cell_ptr_arg = nullptr;
      const bool *pbc_ptr_arg = nullptr;
      if (cell_ptr) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            lattice_buf[i * 3 + j] = cell_ptr->matrix[i][j];
          }
          pbc_buf[i] = cell_ptr->periodic[i];
        }
        cell_ptr_arg = lattice_buf;
        pbc_ptr_arg = pbc_buf;
      }
      mlipcpp::Result res = pred.predict(
          system.num_atoms(), system.positions(), system.atomic_numbers(),
          cell_ptr_arg, pbc_ptr_arg, popts);
      energy = res.energy;
      if (!res.forces.empty()) { forces = std::move(res.forces); has_forces = true; }
      if (!res.stress.empty()) { stress = std::move(res.stress); has_stress = true; }
    } else {
      ModelResult result = model->predict(system, compute_forces);
      energy = result.energy;
      if (result.has_forces) { forces = std::move(result.forces); has_forces = true; }
      if (compute_stress && result.has_stress) {
        stress = std::move(result.stress);
        has_stress = true;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Results ===\n";
    std::cout << "Total energy: " << energy << " eV\n";

    if (has_forces) {
      std::cout << "\nForces (eV/A):\n";
      float fsum[3] = {0, 0, 0};
      for (size_t i = 0; i < system.num_atoms(); i++) {
        std::cout << "  Atom " << i << ": [" << std::setw(12)
                  << forces[i * 3 + 0] << ", " << std::setw(12)
                  << forces[i * 3 + 1] << ", " << std::setw(12)
                  << forces[i * 3 + 2] << "]\n";
        for (int k = 0; k < 3; k++) fsum[k] += forces[i * 3 + k];
      }
      std::cout << "  Force sum: [" << fsum[0] << ", " << fsum[1] << ", "
                << fsum[2] << "]\n";
    }

    if (has_stress && stress.size() == 6) {
      std::cout << "\nStress (Voigt, eV/A^3"
                << (fd_stress ? ", finite-difference" : ", analytical")
                << "):\n";
      std::cout << "  xx=" << stress[0] << "  yy=" << stress[1]
                << "  zz=" << stress[2] << "\n";
      std::cout << "  yz=" << stress[3] << "  xz=" << stress[4]
                << "  xy=" << stress[5] << "\n";
    } else if (compute_stress && !has_stress) {
      std::cout << "\nStress requested but not produced (system is non-periodic"
                   " or model doesn't return stress).\n";
    }

    std::cout << "\nCompute time: " << std::fixed << std::setprecision(1)
              << ms << " ms\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
