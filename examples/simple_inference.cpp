#include "../src/models/pet/pet.h"
#include "core/log.h"
#include "mlipcpp/io.h"
#include "mlipcpp/model.h"
#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include "mlipcpp/timer.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace mlipcpp;

// Element symbols for printing
static const char *element_symbol(int z) {
  static const char *symbols[] = {"X",  "H", "He", "Li", "Be", "B",  "C",
                                  "N",  "O", "F",  "Ne", "Na", "Mg", "Al",
                                  "Si", "P", "S",  "Cl", "Ar", "K",  "Ca"};
  if (z >= 0 && z <= 20)
    return symbols[z];
  return "X";
}

int main(int argc, char **argv) {
  // Suppress verbose GGML logging before any ggml code runs
  log::suppress_ggml_logging();

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model.gguf> <structure.xyz> [options]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --forces          Compute forces via gradient (conservative "
                 "+ non-conservative)\n";
    std::cerr << "  --nc-forces       Compute only non-conservative forces "
                 "(faster, no gradient)\n";
    std::cerr << "  --stress          Compute stress tensor via gradient "
                 "(periodic only)\n";
    std::cerr << "  --nc-stress       Compute only non-conservative stress "
                 "(faster, no gradient)\n";
    std::cerr << "  --backend B       Set backend: auto, cpu, cuda, hip, metal, "
                 "vulkan, sycl, cann (default: auto)\n";
    std::cerr << "  --precision P     Set compute precision: f32, f16 "
                 "(default: f32)\n";
    std::cerr << "  --profile         Enable op-level profiling\n";
    std::cerr << "  --log-level N     Set log level (0=NONE, 1=DEBUG, 2=INFO, "
                 "3=WARN, 4=ERROR)\n";
    std::cerr
        << "  --cutoff R        Override model cutoff radius (Angstroms)\n";
    std::cerr << "  --threads t       Override num threads for cpu backend\n";
    std::cerr
        << "  --quiet           Minimal output (just energy/forces/stress)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << argv[0] << " pet-mad.gguf examples/water.xyz\n";
    std::cerr << "  " << argv[0]
              << " pet-mad.gguf examples/water.xyz --forces --stress\n";
    std::cerr << "  " << argv[0]
              << " pet-mad.gguf examples/water.xyz --backend cpu --forces\n";
    return 1;
  }

  std::string model_path = argv[1];
  std::string xyz_path = argv[2];

  // Parse options
  log::Level log_level = log::Level::Info; // Default: INFO
  float cutoff_override = -1.0f;           // -1 means use model's default
  bool compute_forces = false;
  bool compute_stress = false;
  bool show_nc_forces = false;  // Show non-conservative forces only
  bool show_nc_stress = false;  // Show non-conservative stress only
  bool quiet_mode = false;
  bool profile_mode = false;
  pet::BackendPreference backend_pref = pet::BackendPreference::Auto;
  pet::ComputePrecision precision = pet::ComputePrecision::F32;

  // Backend name lookup table
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

  static const std::unordered_map<std::string_view, pet::ComputePrecision>
      precision_map = {
          {"f32", pet::ComputePrecision::F32},
          {"f16", pet::ComputePrecision::F16},
      };

  for (int i = 3; i < argc; ++i) {
    std::string_view arg = argv[i];

    if (arg == "--log-level" && i + 1 < argc) {
      int level = std::stoi(argv[++i]);
      log_level = static_cast<log::Level>(level);
    } else if (arg == "--cutoff" && i + 1 < argc) {
      cutoff_override = std::stof(argv[++i]);
    } else if (arg == "--forces") {
      compute_forces = true;
    } else if (arg == "--nc-forces") {
      show_nc_forces = true;
    } else if (arg == "--stress") {
      compute_stress = true;
      compute_forces = true; // Stress requires gradient computation
    } else if (arg == "--nc-stress") {
      show_nc_stress = true;
    } else if (arg == "--quiet") {
      quiet_mode = true;
      log_level = log::Level::Warn; // Suppress info messages
    } else if (arg == "--profile") {
      profile_mode = true;
    } else if (arg == "--backend" && i + 1 < argc) {
      std::string_view backend_str = argv[++i];
      if (auto it = backend_map.find(backend_str); it != backend_map.end()) {
        backend_pref = it->second;
      } else {
        std::cerr << "Unknown backend: " << backend_str
                  << " (use: auto, cpu, cuda, hip, metal, vulkan, sycl, cann)\n";
        return 1;
      }
    } else if (arg == "--precision" && i + 1 < argc) {
      std::string_view prec_str = argv[++i];
      if (auto it = precision_map.find(prec_str); it != precision_map.end()) {
        precision = it->second;
      } else {
        std::cerr << "Unknown precision: " << prec_str << " (use: f32, f16)\n";
        return 1;
      }
    } else {
      // Legacy: single number argument is log level
      try {
        int level = std::stoi(std::string(arg));
        log_level = static_cast<log::Level>(level);
      } catch (...) {
        std::cerr << "Unknown option: " << arg << "\n";
        return 1;
      }
    }
  }

  // Set log level
  log::set_level(log_level);

  // Read atomic system from XYZ file
  AtomicSystem system;
  try {
    log::info("Reading structure from {}", xyz_path);
    system = io::read_xyz(xyz_path);

    log::info("Loaded {} atoms", system.num_atoms());

    // Count unique species
    const int32_t *atomic_nums = system.atomic_numbers();
    std::set<int32_t> unique_species(atomic_nums,
                                     atomic_nums + system.num_atoms());
    std::vector<int32_t> species_vec(unique_species.begin(),
                                     unique_species.end());
    log::info("Elements: {} ({})", fmt::join(species_vec, ", "),
              unique_species.size());

    if (system.is_periodic()) {
      const Cell *cell = system.cell();
      log::debug("Periodic boundary conditions: [{}, {}, {}]",
                 cell->periodic[0] ? "T" : "F", cell->periodic[1] ? "T" : "F",
                 cell->periodic[2] ? "T" : "F");
      log::debug("Cell:");
      for (int i = 0; i < 3; ++i) {
        log::debug("  [{:.6f}, {:.6f}, {:.6f}]", cell->matrix[i][0],
                   cell->matrix[i][1], cell->matrix[i][2]);
      }
    } else {
      log::debug("No periodic boundary conditions (isolated system)");
    }
  } catch (const std::exception &e) {
    log::error("Error reading XYZ file: {}", e.what());
    return 1;
  }

  // Build neighbor list
  NeighborListOptions nl_opts;
  nl_opts.cutoff =
      5.0f; // PET-MAD uses 5.0 A cutoff (match PyTorch for comparison)
  nl_opts.full_list = true;

  NeighborListBuilder builder(nl_opts);
  auto nlist = builder.build(system);

  log::debug("Built neighbor list with {} pairs", nlist.num_pairs());

  // Print first 5 edges
  for (int e = 0; e < std::min(5, nlist.num_pairs()); ++e) {
    log::trace(
        "  {}: ({}->{}) shift=[{},{},{}] D=[{:.3f},{:.3f},{:.3f}] d={:.3f}", e,
        nlist.centers[e], nlist.neighbors[e], nlist.cell_shifts[e][0],
        nlist.cell_shifts[e][1], nlist.cell_shifts[e][2],
        nlist.edge_vectors[e][0], nlist.edge_vectors[e][1],
        nlist.edge_vectors[e][2], nlist.distances[e]);
  }

  // Load model and run inference
  try {
    log::info("Loading model from {}", model_path);
    Timer::instance().reset(); // Reset timers before loading and inference

    // Use PETModel directly for forces/stress support
    pet::PETHypers hypers;
    pet::PETModel pet_model(hypers);

    // Set backend preference BEFORE loading (backend is initialized during load)
    pet_model.set_backend_preference(backend_pref);

    if (!pet_model.load_from_gguf(model_path)) {
      log::error("Failed to load model from {}", model_path);
      return 1;
    }

    log::info("Model cutoff from GGUF: {:.2f} A", pet_model.cutoff());

    // Override cutoff if requested
    if (cutoff_override > 0.0f) {
      pet_model.set_cutoff(cutoff_override);
      log::info("Overriding cutoff to: {:.2f} A", cutoff_override);
    }

    static constexpr std::array backend_names = {"auto", "cpu", "cuda", "hip",
                                                  "metal", "vulkan", "sycl", "cann"};
    log::info("Backend preference: {}", backend_names[static_cast<size_t>(backend_pref)]);

    // Set compute precision
    pet_model.set_precision(precision);
    static constexpr std::array precision_names = {"f32", "f16"};
    log::info("Precision: {}", precision_names[static_cast<size_t>(precision)]);

    // Set profiling mode
    pet_model.set_profiling(profile_mode);

    log::info("Running inference...");
    // Use predict_batch for full control over compute_nc parameter
    bool compute_nc = show_nc_forces || show_nc_stress;
    auto results = pet_model.predict_batch({system}, compute_forces, compute_nc);
    auto result = results[0];

    // Print results
    if (quiet_mode) {
      printf("Energy: %.6f eV\n", result.energy);
    } else {
      double energy_per_atom = result.energy / system.num_atoms();
      log::info("Energy: {:.6f} eV", result.energy);
      log::info("Energy per atom: {:.6f} eV/atom", energy_per_atom);
    }

    // Print forces if requested
    // Note: result.has_forces is true if gradient forces OR nc_forces are available
    bool should_show_forces =
        (compute_forces && result.has_forces) || (show_nc_forces && result.has_forces);
    if (should_show_forces) {
      const char *force_type =
          compute_forces ? "Forces" : "Non-conservative Forces";
      printf("\n%s (eV/A):\n", force_type);
      const int32_t *atomic_nums = system.atomic_numbers();
      for (int i = 0; i < system.num_atoms(); ++i) {
        printf("  Atom %d (%s): [%12.6f, %12.6f, %12.6f]\n", i,
               element_symbol(atomic_nums[i]), result.forces[i * 3 + 0],
               result.forces[i * 3 + 1], result.forces[i * 3 + 2]);
      }
    } else if (show_nc_forces && !result.has_forces) {
      printf("\nNon-conservative forces: not available (model lacks nc_forces heads)\n");
    }

    // Print stress if requested and available
    bool should_show_stress =
        (compute_stress && result.has_stress) || (show_nc_stress && result.has_stress);
    if (should_show_stress) {
      const char *stress_type =
          compute_stress ? "Stress" : "Non-conservative Stress";
      printf("\n%s (Voigt, eV/A^3):\n", stress_type);
      printf("  xx=%.6f, yy=%.6f, zz=%.6f\n", result.stress[0],
             result.stress[1], result.stress[2]);
      printf("  yz=%.6f, xz=%.6f, xy=%.6f\n", result.stress[3],
             result.stress[4], result.stress[5]);
    } else if (compute_stress && !result.has_stress) {
      printf("\nStress: not available (non-periodic system)\n");
    } else if (show_nc_stress && !result.has_stress) {
      printf("\nNon-conservative stress: not available (model lacks nc_stress heads)\n");
    }

    // Print timing summary (unless quiet)
    if (!quiet_mode) {
      Timer::instance().print_summary();
    }
  } catch (const std::exception &e) {
    log::error("Error: {}", e.what());
    return 1;
  }

  return 0;
}
