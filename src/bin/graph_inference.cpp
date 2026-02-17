/**
 * Graph-based inference on XYZ files using auto-exported PET models.
 *
 * Usage:
 *   graph_inference <model> <xyz_file> [--forces] [--debug]
 *
 * Where <model> is either:
 *   - A .gguf file (single file with graph + weights + metadata)
 *   - A directory containing pet_full.json, metadata.json, and *.bin weight files
 *
 * When --forces is specified, computes forces via backward pass (F = -dE/dr).
 * Requires the model to be exported with --forces mode.
 */

#include "core/gguf_loader.h"
#include "mlipcpp/io.h"
#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include "runtime/graph_interpreter.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::runtime;
using json = nlohmann::json;

namespace {

// Load binary file into vector
template <typename T> std::vector<T> load_binary(const std::string &path) {
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

struct ModelData {
  float cutoff = 4.5f;
  float cutoff_width = 0.2f;
  float energy_scale = 1.0f;          // scale factor applied to raw model output
  bool forces_mode = false;            // true if model was exported with --forces
  std::string cutoff_function = "cosine"; // "cosine" or "bump"
  float num_neighbors_adaptive = 0.0f;   // 0 = disabled, >0 = target neighbor count
  std::map<int, int> species_to_index;
  std::map<int, float> composition_energies;
};

// Bump cutoff function: smooth switching function
// f(x) = 1 for x <= 0, 0.5*(1+tanh(1/tan(pi*x))) for 0 < x < 1, 0 for x >= 1
// where x = (distance - (cutoff - width)) / width
float cutoff_func_bump(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  float tan_val = std::tan(M_PI * x);
  return 0.5f * (1.0f + std::tanh(1.0f / tan_val));
}

// Cosine cutoff function
float cutoff_func_cosine(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  return 0.5f * (1.0f + std::cos(M_PI * x));
}

// Bump cutoff in double precision (for adaptive cutoff computation)
double cutoff_func_bump_d(double distance, double cutoff, double width) {
  double x = (distance - (cutoff - width)) / width;
  if (x <= 0.0) return 1.0;
  if (x >= 1.0) return 0.0;
  double tan_val = std::tan(M_PI * x);
  return 0.5 * (1.0 + std::tanh(1.0 / tan_val));
}

// Compute adaptive per-atom cutoffs following metatrain's algorithm.
// Uses double precision throughout to match metatrain's float64 computation.
// Takes double-precision distances for accuracy.
// Returns per-atom cutoff distances.
std::vector<float> compute_adaptive_cutoffs(
    const std::vector<int32_t> &centers,
    const std::vector<double> &distances,
    float num_neighbors_adaptive,
    int num_nodes,
    float max_cutoff,
    float cutoff_width) {

  constexpr double MIN_PROBE_CUTOFF = 0.5;
  double probe_spacing = static_cast<double>(cutoff_width) / 4.0;
  double target = static_cast<double>(num_neighbors_adaptive);
  double max_cut = static_cast<double>(max_cutoff);

  // Generate probe cutoffs (match torch.arange: start + i*step to avoid accumulation error)
  int n_probes_est = static_cast<int>(std::ceil((max_cut - MIN_PROBE_CUTOFF) / probe_spacing));
  std::vector<double> probe_cutoffs;
  probe_cutoffs.reserve(n_probes_est);
  for (int i = 0; ; i++) {
    double c = MIN_PROBE_CUTOFF + i * probe_spacing;
    if (c >= max_cut) break;
    probe_cutoffs.push_back(c);
  }
  int n_probes = static_cast<int>(probe_cutoffs.size());
  if (n_probes == 0) {
    return std::vector<float>(num_nodes, max_cutoff);
  }

  int n_edges = static_cast<int>(distances.size());

  // Step 1: Compute effective neighbor counts per (atom, probe)
  // metatrain passes the model's cutoff_width (not the default 1.0) to
  // get_effective_num_neighbors
  double eff_width = static_cast<double>(cutoff_width);
  std::vector<std::vector<double>> eff_neighbors(num_nodes, std::vector<double>(n_probes, 0.0));

  for (int e = 0; e < n_edges; e++) {
    int center = centers[e];
    double dist = distances[e];
    for (int p = 0; p < n_probes; p++) {
      double w = cutoff_func_bump_d(dist, probe_cutoffs[p], eff_width);
      eff_neighbors[center][p] += w;
    }
  }

  // Step 2: Compute Gaussian cutoff selection weights
  // baseline = num_neighbors_adaptive * x^3 where x = linspace(0, 1, n_probes)
  std::vector<double> baseline(n_probes);
  for (int p = 0; p < n_probes; p++) {
    double x = (n_probes > 1) ? static_cast<double>(p) / (n_probes - 1) : 0.0;
    baseline[p] = target * x * x * x;
  }

  std::vector<float> adapted_cutoffs(num_nodes, max_cutoff);

  for (int a = 0; a < num_nodes; a++) {
    // diff[p] = eff_neighbors[a][p] - target + baseline[p]
    std::vector<double> diff(n_probes);
    for (int p = 0; p < n_probes; p++) {
      diff[p] = eff_neighbors[a][p] - target + baseline[p];
    }

    // Compute adaptive width via numerical gradient of diff
    std::vector<double> width_t(n_probes);
    constexpr double eps = 1e-12;
    if (n_probes == 1) {
      width_t[0] = std::abs(diff[0]) * 0.5 + eps;
    } else {
      for (int p = 1; p < n_probes - 1; p++) {
        width_t[p] = std::max(std::abs((diff[p + 1] - diff[p - 1]) / 2.0), eps);
      }
      width_t[0] = std::max(std::abs(diff[1] - diff[0]), eps);
      width_t[n_probes - 1] = std::max(std::abs(diff[n_probes - 1] - diff[n_probes - 2]), eps);
    }

    // Gaussian weights: logw = -0.5 * (diff / width_t)^2
    std::vector<double> logw(n_probes);
    double max_logw = -1e30;
    for (int p = 0; p < n_probes; p++) {
      double ratio = diff[p] / width_t[p];
      logw[p] = -0.5 * ratio * ratio;
      if (logw[p] > max_logw) max_logw = logw[p];
    }

    // weights = exp(logw - max_logw), then normalize
    std::vector<double> weights(n_probes);
    double weight_sum = 0.0;
    for (int p = 0; p < n_probes; p++) {
      weights[p] = std::exp(logw[p] - max_logw);
      weight_sum += weights[p];
    }
    for (int p = 0; p < n_probes; p++) {
      weights[p] /= weight_sum;
    }

    // Weighted average of probe cutoffs
    double cutoff_val = 0.0;
    for (int p = 0; p < n_probes; p++) {
      cutoff_val += probe_cutoffs[p] * weights[p];
    }
    adapted_cutoffs[a] = static_cast<float>(cutoff_val);
  }

  return adapted_cutoffs;
}

// Load model from a directory of loose files
void load_from_directory(const std::string &dir_path, GraphInterpreter &interp,
                         ModelData &model, ggml_context *weight_ctx,
                         ggml_backend_t backend) {
  namespace fs = std::filesystem;

  // Load metadata
  std::ifstream mf(fs::path(dir_path) / "metadata.json");
  if (!mf)
    throw std::runtime_error("Failed to open metadata.json");
  json metadata;
  mf >> metadata;

  model.cutoff = metadata.value("cutoff", 4.5f);
  model.cutoff_width = metadata.value("cutoff_width", 0.2f);
  model.energy_scale = metadata.value("energy_scale", 1.0f);
  model.forces_mode = metadata.value("forces", false);
  model.cutoff_function = metadata.value("cutoff_function", "cosine");
  if (metadata.contains("num_neighbors_adaptive") && !metadata["num_neighbors_adaptive"].is_null()) {
    model.num_neighbors_adaptive = metadata["num_neighbors_adaptive"].get<float>();
  }

  if (metadata.contains("species_to_index")) {
    for (auto &[key, val] : metadata["species_to_index"].items()) {
      model.species_to_index[std::stoi(key)] = val.get<int>();
    }
  }
  if (metadata.contains("composition_energies")) {
    for (auto &[key, val] : metadata["composition_energies"].items()) {
      model.composition_energies[std::stoi(key)] = val.get<float>();
    }
  }

  // Load graph
  interp.load_graph_file((fs::path(dir_path) / "pet_full.json").string());

  // Load weights
  if (!metadata.contains("weights"))
    throw std::runtime_error("No weights section in metadata.json");

  std::map<std::string, std::pair<ggml_tensor *, std::vector<float>>>
      weight_data;

  for (auto &[name, shape_arr] : metadata["weights"].items()) {
    std::string weight_path = (fs::path(dir_path) / (name + ".bin")).string();
    if (!fs::exists(weight_path))
      continue;

    auto data = load_binary<float>(weight_path);

    // Reverse shape for GGML
    std::vector<int64_t> py_shape;
    for (const auto &dim : shape_arr)
      py_shape.push_back(dim.get<int64_t>());
    std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());

    ggml_tensor *t = nullptr;
    switch (ggml_shape.size()) {
    case 0:
      t = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, 1);
      break;
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
    weight_data[name] = {t, std::move(data)};
    interp.set_weight(name, t);
  }

  // Allocate and fill weights
  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
  if (!buf)
    throw std::runtime_error("Failed to allocate weight buffer");

  for (const auto &[name, pair] : weight_data) {
    ggml_backend_tensor_set(pair.first, pair.second.data(), 0,
                            pair.second.size() * sizeof(float));
  }

  std::cout << "Loaded " << weight_data.size() << " weights from directory\n";
}

// Load model from a single GGUF file
void load_from_gguf(const std::string &gguf_path, GraphInterpreter &interp,
                    ModelData &model, ggml_context *weight_ctx,
                    ggml_backend_t backend) {
  // Load GGUF file with data into a temporary context
  constexpr size_t TEMP_CTX_SIZE = 512 * 1024 * 1024;
  ggml_context *temp_ctx = ggml_init({TEMP_CTX_SIZE, nullptr, false});
  if (!temp_ctx)
    throw std::runtime_error("Failed to create temp context");

  GGUFLoader loader(gguf_path, temp_ctx);

  // Read metadata
  model.cutoff = loader.get_float32("pet.cutoff", 4.5f);
  model.cutoff_width = loader.get_float32("pet.cutoff_width", 0.2f);
  model.energy_scale = loader.get_float32("pet.energy_scale", 1.0f);

  // Check for forces mode (stored as int32 since GGUF doesn't have bool)
  model.forces_mode = (loader.get_int32("pet.forces_mode", 0) != 0);

  // Species mapping: [Z1, idx1, Z2, idx2, ...]
  auto species_map = loader.get_array_int32("pet.species_map");
  for (size_t i = 0; i + 1 < species_map.size(); i += 2) {
    model.species_to_index[species_map[i]] = species_map[i + 1];
  }

  // Composition energies
  auto comp_keys = loader.get_array_int32("pet.composition_keys");
  auto comp_vals = loader.get_array_float32("pet.composition_values");
  if (comp_keys.size() != comp_vals.size()) {
    throw std::runtime_error(
        "GGUF: composition_keys (" + std::to_string(comp_keys.size()) +
        ") and composition_values (" + std::to_string(comp_vals.size()) +
        ") arrays have different lengths");
  }
  for (size_t i = 0; i < comp_keys.size(); i++) {
    model.composition_energies[comp_keys[i]] = comp_vals[i];
  }

  // Load graph JSON
  std::string graph_json = loader.get_string("graph.json");
  if (graph_json.empty()) {
    throw std::runtime_error("No graph.json in GGUF metadata");
  }
  interp.load_graph(graph_json);

  // Load weight shapes from metadata
  std::string shapes_json = loader.get_string("graph.weight_shapes");
  json weight_shapes;
  if (!shapes_json.empty()) {
    weight_shapes = json::parse(shapes_json);
  }

  // Load weight tensors
  auto tensor_names = loader.get_tensor_names();
  std::vector<std::pair<std::string, ggml_tensor *>> weight_pairs;

  for (const auto &name : tensor_names) {
    ggml_tensor *temp_tensor = loader.get_tensor(name);
    if (!temp_tensor)
      continue;

    // Use weight_shapes metadata to get correct PyTorch shape, then reverse for GGML.
    ggml_tensor *t = nullptr;
    if (weight_shapes.contains(name)) {
      auto py_shape = weight_shapes[name].get<std::vector<int64_t>>();
      std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());
      switch (ggml_shape.size()) {
      case 0:
        t = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, 1);
        break;
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
    } else {
      // Fallback: use GGUF stored shape directly
      t = ggml_new_tensor(weight_ctx, temp_tensor->type,
                          ggml_n_dims(temp_tensor), temp_tensor->ne);
    }

    ggml_set_name(t, name.c_str());
    weight_pairs.push_back({name, t});
    interp.set_weight(name, t);
  }

  // Allocate backend buffer and copy weight data
  ggml_backend_buffer_t buf =
      ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
  if (!buf) {
    throw std::runtime_error("Failed to allocate weight buffer");
  }

  for (const auto &[name, tensor] : weight_pairs) {
    ggml_tensor *temp = loader.get_tensor(name);
    if (temp && temp->data) {
      ggml_backend_tensor_set(tensor, temp->data, 0, ggml_nbytes(tensor));
    }
  }

  std::cout << "Loaded " << weight_pairs.size() << " weights from GGUF\n";
}

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <model> <xyz_file> [--forces] [--debug]\n\n";
  std::cerr << "Arguments:\n";
  std::cerr << "  model     .gguf file or export directory\n";
  std::cerr << "  xyz_file  Input structure in XYZ format\n";
  std::cerr << "  --forces  Compute forces via backward pass (F = -dE/dr)\n";
  std::cerr << "  --debug   Dump inputs and print intermediate tensor values\n";
  std::cerr << "\nExample:\n";
  std::cerr << "  " << prog << " pet-auto.gguf geometries/water.xyz\n";
  std::cerr << "  " << prog
            << " /tmp/pet_forces_export geometries/water.xyz --forces\n";
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string xyz_path = argv[2];
  bool debug = false;
  bool compute_forces = false;

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--debug")
      debug = true;
    else if (arg == "--forces")
      compute_forces = true;
    else {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  try {
    // Create backend
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
      std::cerr << "Error: Failed to create CPU backend\n";
      return 1;
    }

    // Create weight context
    constexpr size_t WEIGHT_CTX_SIZE = 128 * 1024 * 1024;
    ggml_context *weight_ctx = ggml_init({WEIGHT_CTX_SIZE, nullptr, true});
    if (!weight_ctx) {
      ggml_backend_free(cpu_backend);
      std::cerr << "Error: Failed to create weight context\n";
      return 1;
    }

    // Load model (auto-detect format)
    GraphInterpreter interp;
    ModelData model;

    bool is_gguf = model_path.size() >= 5 &&
                   model_path.substr(model_path.size() - 5) == ".gguf";

    if (is_gguf) {
      std::cout << "Loading GGUF: " << model_path << "\n";
      load_from_gguf(model_path, interp, model, weight_ctx, cpu_backend);
    } else {
      std::cout << "Loading directory: " << model_path << "\n";
      load_from_directory(model_path, interp, model, weight_ctx, cpu_backend);
    }

    // Validate force computation request
    if (compute_forces && !model.forces_mode) {
      std::cerr << "Error: --forces requested but model was not exported with "
                   "--forces mode.\n"
                << "  Re-export with: uv run scripts/export_pytorch/"
                   "export_pet_full.py --model <name> --forces\n";
      return 1;
    }

    std::cout << "  Cutoff: " << model.cutoff << " A\n";
    std::cout << "  Cutoff function: " << model.cutoff_function << "\n";
    if (model.num_neighbors_adaptive > 0.0f) {
      std::cout << "  Adaptive cutoff: " << model.num_neighbors_adaptive
                << " neighbors\n";
    }
    std::cout << "  Species mapped: " << model.species_to_index.size() << "\n";
    std::cout << "  Energy scale: " << model.energy_scale << "\n";
    std::cout << "  Forces mode: " << (model.forces_mode ? "yes" : "no")
              << "\n";
    std::cout << "  Graph: " << interp.graph().nodes.size() << " nodes\n";

    // Read XYZ file
    AtomicSystem system = io::read_xyz(xyz_path);
    const int n_atoms = static_cast<int>(system.num_atoms());
    const int32_t *atomic_numbers = system.atomic_numbers();

    std::cout << "\nInput: " << xyz_path << " (" << n_atoms << " atoms)\n";

    // Build neighbor list
    NeighborListBuilder nlist_builder(
        NeighborListOptions{model.cutoff, true, false});
    NeighborList nlist = nlist_builder.build(system);

    std::cout << "  Raw edges: " << nlist.num_pairs() << "\n";

    // Apply adaptive cutoff filtering if enabled
    // Per-pair cutoff distances (used for bump cutoff computation)
    std::vector<float> pair_cutoffs(nlist.num_pairs(), model.cutoff);

    if (model.num_neighbors_adaptive > 0.0f) {
      // Recompute distances in double precision for accurate adaptive cutoff.
      // metatrain uses float64 positions/distances throughout. Our neighbor list
      // stores float32 edge vectors, so we recompute distances from the original
      // double-precision positions and cell to match metatrain's precision.
      int n_pairs = nlist.num_pairs();
      std::vector<double> distances_d(n_pairs);

      // Read positions as double from the AtomicSystem
      // (positions were read as double from XYZ, converted to float for storage)
      const float *pos_f = system.positions();
      std::vector<double> pos_d(n_atoms * 3);
      for (int i = 0; i < n_atoms * 3; i++) {
        pos_d[i] = static_cast<double>(pos_f[i]);
      }

      // Read cell as double (if periodic)
      double cell_d[3][3] = {{0}};
      if (system.is_periodic()) {
        const Cell *cell = system.cell();
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            cell_d[i][j] = static_cast<double>(cell->matrix[i][j]);
          }
        }
      }

      bool has_shifts = !nlist.cell_shifts.empty();
      for (int e = 0; e < n_pairs; e++) {
        int ci = nlist.centers[e];
        int ni = nlist.neighbors[e];
        double dx = pos_d[ni * 3 + 0] - pos_d[ci * 3 + 0];
        double dy = pos_d[ni * 3 + 1] - pos_d[ci * 3 + 1];
        double dz = pos_d[ni * 3 + 2] - pos_d[ci * 3 + 2];
        if (has_shifts) {
          const auto &s = nlist.cell_shifts[e];
          dx += s[0] * cell_d[0][0] + s[1] * cell_d[1][0] + s[2] * cell_d[2][0];
          dy += s[0] * cell_d[0][1] + s[1] * cell_d[1][1] + s[2] * cell_d[2][1];
          dz += s[0] * cell_d[0][2] + s[1] * cell_d[1][2] + s[2] * cell_d[2][2];
        }
        distances_d[e] = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-15;
      }

      // Compute per-atom adaptive cutoffs
      std::vector<float> atomic_cutoffs = compute_adaptive_cutoffs(
          nlist.centers, distances_d,
          model.num_neighbors_adaptive, n_atoms,
          model.cutoff, model.cutoff_width);

      // Symmetrize: pair_cutoff = (cutoff[center] + cutoff[neighbor]) / 2
      // and filter: keep edges where distance <= pair_cutoff
      std::vector<bool> keep(n_pairs, false);
      int kept = 0;
      for (int e = 0; e < n_pairs; e++) {
        double pc = (static_cast<double>(atomic_cutoffs[nlist.centers[e]]) +
                     static_cast<double>(atomic_cutoffs[nlist.neighbors[e]])) / 2.0;
        if (distances_d[e] <= pc) {
          keep[e] = true;
          kept++;
        }
      }

      // Build filtered neighbor list
      NeighborList filtered;
      filtered.centers.reserve(kept);
      filtered.neighbors.reserve(kept);
      filtered.edge_vectors.reserve(kept);
      filtered.distances.reserve(kept);
      if (!nlist.cell_shifts.empty()) {
        filtered.cell_shifts.reserve(kept);
      }

      std::vector<float> filtered_pair_cutoffs;
      filtered_pair_cutoffs.reserve(kept);

      for (int e = 0; e < n_pairs; e++) {
        if (!keep[e]) continue;
        filtered.centers.push_back(nlist.centers[e]);
        filtered.neighbors.push_back(nlist.neighbors[e]);
        filtered.edge_vectors.push_back(nlist.edge_vectors[e]);
        filtered.distances.push_back(nlist.distances[e]);
        if (!nlist.cell_shifts.empty()) {
          filtered.cell_shifts.push_back(nlist.cell_shifts[e]);
        }
        double pc = (static_cast<double>(atomic_cutoffs[nlist.centers[e]]) +
                     static_cast<double>(atomic_cutoffs[nlist.neighbors[e]])) / 2.0;
        filtered_pair_cutoffs.push_back(static_cast<float>(pc));
      }

      nlist = std::move(filtered);
      pair_cutoffs = std::move(filtered_pair_cutoffs);

      std::cout << "  Adaptive cutoff filtered: " << nlist.num_pairs()
                << " edges kept\n";
    }

    // Count max neighbors (after filtering)
    std::vector<int> neighbor_counts(n_atoms, 0);
    for (int e = 0; e < nlist.num_pairs(); e++) {
      neighbor_counts[nlist.centers[e]]++;
    }
    int max_neighbors = 0;
    for (int i = 0; i < n_atoms; i++) {
      max_neighbors = std::max(max_neighbors, neighbor_counts[i]);
    }

    std::cout << "  Edges: " << nlist.num_pairs()
              << ", max_neighbors: " << max_neighbors << "\n";

    // Set symbolic dimensions
    interp.set_dimension("n_atoms", n_atoms);
    interp.set_dimension("max_neighbors", max_neighbors);
    interp.set_dimension("n_edges", n_atoms * max_neighbors);
    interp.set_dimension("max_neighbors_plus_one", max_neighbors + 1);

    // Create input context
    constexpr size_t INPUT_CTX_SIZE = 16 * 1024 * 1024;
    ggml_context *input_ctx = ggml_init({INPUT_CTX_SIZE, nullptr, true});

    // Create input tensors
    ggml_tensor *species =
        ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(species, "species");

    ggml_tensor *neighbor_species =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
    ggml_set_name(neighbor_species, "neighbor_species");

    ggml_tensor *edge_vectors =
        ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
    ggml_set_name(edge_vectors, "edge_vectors");

    ggml_tensor *padding_mask =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(padding_mask, "padding_mask");

    ggml_tensor *reverse_neighbor_index =
        ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms * max_neighbors);
    ggml_set_name(reverse_neighbor_index, "reverse_neighbor_index");

    // These inputs are only used in non-forces mode
    ggml_tensor *edge_distances = nullptr;
    ggml_tensor *cutoff_factors = nullptr;
    if (!model.forces_mode) {
      edge_distances =
          ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
      ggml_set_name(edge_distances, "edge_distances");

      cutoff_factors =
          ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
      ggml_set_name(cutoff_factors, "cutoff_factors");
    }

    // Per-pair cutoff values (forces mode only)
    ggml_tensor *cutoff_values = nullptr;
    if (model.forces_mode) {
      cutoff_values =
          ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
      ggml_set_name(cutoff_values, "cutoff_values");
    }

    // Mark edge_vectors as parameter for gradient computation
    if (compute_forces) {
      ggml_set_param(edge_vectors);
    }

    ggml_backend_buffer_t input_buffer =
        ggml_backend_alloc_ctx_tensors(input_ctx, cpu_backend);

    // Prepare input data
    std::vector<int32_t> species_data(n_atoms);
    for (int i = 0; i < n_atoms; i++) {
      int Z = atomic_numbers[i];
      auto it = model.species_to_index.find(Z);
      if (it == model.species_to_index.end()) {
        std::cerr << "Error: atomic number " << Z << " (atom " << i
                  << ") is not in the model's species map.\n"
                  << "The model does not support this element.\n";
        return 1;
      }
      species_data[i] = it->second;
    }
    ggml_backend_tensor_set(species, species_data.data(), 0,
                            species_data.size() * sizeof(int32_t));

    std::vector<int32_t> ns_data(n_atoms * max_neighbors, 0);
    std::vector<float> ev_data(n_atoms * max_neighbors * 3, 0.0f);
    std::vector<float> ed_data(n_atoms * max_neighbors, 0.0f);
    std::vector<float> pm_data(n_atoms * max_neighbors, 1.0f);  // 1.0 = padded (PyTorch True), 0.0 = valid
    std::vector<float> cf_data(n_atoms * max_neighbors, 0.0f);
    std::vector<float> cv_data(n_atoms * max_neighbors, model.cutoff);  // per-pair cutoff values (default: global)
    std::vector<int32_t> rni_data(n_atoms * max_neighbors, 0);  // 0 for padded edges (masked out later)

    // Track neighbor atom index for each slot (needed for force scatter)
    std::vector<int> neighbor_atoms(n_atoms * max_neighbors, -1);

    // Key: (center, neighbor, shift_a, shift_b, shift_c)
    using EdgeKey = std::tuple<int, int, int, int, int>;
    std::map<EdgeKey, int> edge_to_flat_idx;
    std::vector<int> slot_indices(n_atoms, 0);
    bool has_cell_shifts = !nlist.cell_shifts.empty();

    for (int e = 0; e < nlist.num_pairs(); e++) {
      int i = nlist.centers[e];
      int j = nlist.neighbors[e];
      int slot = slot_indices[i]++;
      if (slot >= max_neighbors)
        continue;

      int flat_idx = i * max_neighbors + slot;

      int sa = 0, sb = 0, sc = 0;
      if (has_cell_shifts) {
        sa = nlist.cell_shifts[e][0];
        sb = nlist.cell_shifts[e][1];
        sc = nlist.cell_shifts[e][2];
      }
      edge_to_flat_idx[{i, j, sa, sb, sc}] = flat_idx;

      int Z_j = atomic_numbers[j];
      auto it = model.species_to_index.find(Z_j);
      if (it == model.species_to_index.end()) {
        std::cerr << "Error: atomic number " << Z_j << " (neighbor atom " << j
                  << ") is not in the model's species map.\n"
                  << "The model does not support this element.\n";
        return 1;
      }
      ns_data[flat_idx] = it->second;

      const auto &ev = nlist.edge_vectors[e];
      int ev_idx = i * (max_neighbors * 3) + slot * 3;
      ev_data[ev_idx + 0] = ev[0];
      ev_data[ev_idx + 1] = ev[1];
      ev_data[ev_idx + 2] = ev[2];

      ed_data[flat_idx] = nlist.distances[e];
      pm_data[flat_idx] = 0.0f;  // 0.0 = valid edge (PyTorch False)

      // Store neighbor atom index for force scatter
      neighbor_atoms[flat_idx] = j;

      // Per-pair cutoff value and cutoff factor
      float r = nlist.distances[e];
      float pc = pair_cutoffs[e];
      cv_data[flat_idx] = pc;  // Store per-pair cutoff for forces-mode graph
      if (model.cutoff_function == "bump") {
        cf_data[flat_idx] = cutoff_func_bump(r, pc, model.cutoff_width);
      } else {
        cf_data[flat_idx] = cutoff_func_cosine(r, pc, model.cutoff_width);
      }
    }

    // Build reverse neighbor index
    for (int e = 0; e < nlist.num_pairs(); e++) {
      int i = nlist.centers[e];
      int j = nlist.neighbors[e];
      int sa = 0, sb = 0, sc = 0;
      if (has_cell_shifts) {
        sa = nlist.cell_shifts[e][0];
        sb = nlist.cell_shifts[e][1];
        sc = nlist.cell_shifts[e][2];
      }

      auto it_ij = edge_to_flat_idx.find({i, j, sa, sb, sc});
      if (it_ij == edge_to_flat_idx.end())
        continue;
      auto it_ji = edge_to_flat_idx.find({j, i, -sa, -sb, -sc});
      if (it_ji != edge_to_flat_idx.end()) {
        rni_data[it_ij->second] = it_ji->second;
      }
      // If reverse edge not found, leave as -1 (set during initialization)
    }

    ggml_backend_tensor_set(neighbor_species, ns_data.data(), 0,
                            ns_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(edge_vectors, ev_data.data(), 0,
                            ev_data.size() * sizeof(float));
    ggml_backend_tensor_set(padding_mask, pm_data.data(), 0,
                            pm_data.size() * sizeof(float));
    ggml_backend_tensor_set(reverse_neighbor_index, rni_data.data(), 0,
                            rni_data.size() * sizeof(int32_t));

    // Set inputs common to both modes
    interp.set_input("species", species);
    interp.set_input("neighbor_species", neighbor_species);
    interp.set_input("edge_vectors", edge_vectors);
    interp.set_input("padding_mask", padding_mask);
    interp.set_input("reverse_neighbor_index", reverse_neighbor_index);

    if (!model.forces_mode) {
      // Non-forces mode: provide edge_distances and cutoff_factors as inputs
      ggml_backend_tensor_set(edge_distances, ed_data.data(), 0,
                              ed_data.size() * sizeof(float));
      ggml_backend_tensor_set(cutoff_factors, cf_data.data(), 0,
                              cf_data.size() * sizeof(float));
      interp.set_input("edge_distances", edge_distances);
      interp.set_input("cutoff_factors", cutoff_factors);
    } else {
      // Forces mode: provide per-pair cutoff values for in-graph cutoff computation
      ggml_backend_tensor_set(cutoff_values, cv_data.data(), 0,
                              cv_data.size() * sizeof(float));
      interp.set_input("cutoff_values", cutoff_values);
    }

    if (debug) {
      namespace fs = std::filesystem;
      fs::path dump_dir = "/tmp/graph_inference_debug";
      fs::create_directories(dump_dir);

      auto dump = [&](const char *name, const void *data, size_t bytes) {
        std::ofstream f((dump_dir / name).string(), std::ios::binary);
        f.write(static_cast<const char *>(data), bytes);
      };
      dump("species.bin", species_data.data(),
           species_data.size() * sizeof(int32_t));
      dump("neighbor_species.bin", ns_data.data(),
           ns_data.size() * sizeof(int32_t));
      dump("edge_vectors.bin", ev_data.data(), ev_data.size() * sizeof(float));
      dump("edge_distances.bin", ed_data.data(),
           ed_data.size() * sizeof(float));
      dump("padding_mask.bin", pm_data.data(), pm_data.size() * sizeof(float));
      dump("reverse_neighbor_index.bin", rni_data.data(),
           rni_data.size() * sizeof(int32_t));
      dump("cutoff_factors.bin", cf_data.data(),
           cf_data.size() * sizeof(float));

      std::ofstream mf((dump_dir / "dims.txt").string());
      mf << n_atoms << " " << max_neighbors << "\n";
      for (int i = 0; i < n_atoms; i++)
        mf << atomic_numbers[i] << " ";
      mf << "\n";
      std::cout << "Dumped inputs to " << dump_dir.string() << "\n";
    }

    // Build and compute
    // Use larger context for backward pass (gradient computation creates many
    // additional tensors)
    constexpr size_t COMPUTE_CTX_SIZE =
        512 * 1024 * 1024; // 512MB for backward support
    ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});

    ggml_tensor *output = interp.build(compute_ctx);
    if (!output) {
      std::cerr << "Error: Failed to build computation graph\n";
      return 1;
    }
    ggml_set_output(output);

    ggml_cgraph *cgraph = nullptr;
    ggml_tensor *total_energy_tensor = nullptr;

    if (compute_forces) {
      // Forces mode: build forward + backward graph
      // Sum atomic energies to scalar loss for backward pass
      total_energy_tensor = ggml_sum(compute_ctx, output);
      ggml_set_loss(total_energy_tensor);
      ggml_set_output(total_energy_tensor);

      // Create graph with backward support (grads=true)
      cgraph = ggml_new_graph_custom(compute_ctx, 32768, true);
      ggml_build_forward_expand(cgraph, output);
      ggml_build_forward_expand(cgraph, total_energy_tensor);

      // Build backward graph (computes gradients for all param tensors)
      ggml_build_backward_expand(compute_ctx, cgraph, nullptr);

      // Mark gradient tensor as output so allocator computes it
      ggml_tensor *grad_tensor = ggml_graph_get_grad(cgraph, edge_vectors);
      if (grad_tensor) {
        ggml_set_output(grad_tensor);
      } else {
        std::cerr << "Warning: Could not get gradient tensor for edge_vectors. "
                     "Forces will not be computed.\n";
        compute_forces = false;
      }

      std::cout << "Graph nodes (forward+backward): "
                << ggml_graph_n_nodes(cgraph) << "\n";

      // Debug: print info about gradient tensor
      ggml_tensor *dbg_grad = ggml_graph_get_grad(cgraph, edge_vectors);
      std::cout << "  Gradient tensor: "
                << (dbg_grad ? "found" : "NOT FOUND") << "\n";
      if (dbg_grad) {
        std::cout << "  Gradient shape: [" << dbg_grad->ne[0] << ", "
                  << dbg_grad->ne[1] << ", " << dbg_grad->ne[2] << ", "
                  << dbg_grad->ne[3] << "]\n";
        std::cout << "  Gradient flags: " << dbg_grad->flags
                  << " (output=" << (dbg_grad->flags & 4) << ")\n";
      }
      std::cout << "  edge_vectors flags: " << edge_vectors->flags
                << " (param=" << (edge_vectors->flags & 2) << ")\n";
    } else {
      // Forward-only mode
      cgraph = ggml_new_graph(compute_ctx);
      ggml_build_forward_expand(cgraph, output);
    }

    ggml_backend_buffer_t compute_buffer =
        ggml_backend_alloc_ctx_tensors(compute_ctx, cpu_backend);
    interp.init_constants();

    // Initialize gradient accumulators: loss gradient = 1.0, all others = 0.0
    if (compute_forces) {
      ggml_graph_reset(cgraph);
    }

    std::cout << "\nComputing "
              << (compute_forces ? "energy + forces" : "energy") << "...\n";

    auto t_compute_start = std::chrono::high_resolution_clock::now();
    ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
    auto t_compute_end = std::chrono::high_resolution_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
      std::cerr << "Error: Graph computation failed\n";
      return 1;
    }
    double compute_ms = std::chrono::duration<double, std::milli>(
                            t_compute_end - t_compute_start)
                            .count();

    if (debug) {
      auto tensor_sum = [](ggml_tensor *t) -> float {
        if (!t || !t->data)
          return 0.0f;
        float sum = 0.0f;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
          for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
              for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                float *ptr =
                    (float *)((char *)t->data + i0 * t->nb[0] +
                              i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                sum += *ptr;
              }
            }
          }
        }
        return sum;
      };

      auto tensor_min_max = [](ggml_tensor *t, float &min_val,
                               float &max_val) {
        if (!t || !t->data) {
          min_val = max_val = 0.0f;
          return;
        }
        min_val = 1e30f;
        max_val = -1e30f;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
          for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
              for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                float *ptr = (float *)((char *)t->data + i0 * t->nb[0] +
                                       i1 * t->nb[1] + i2 * t->nb[2] +
                                       i3 * t->nb[3]);
                if (*ptr < min_val)
                  min_val = *ptr;
                if (*ptr > max_val)
                  max_val = *ptr;
              }
            }
          }
        }
      };

      std::cout << "\n=== Debug: Intermediate tensor sums ===\n";
      const auto &graph_ir = interp.graph();
      for (const auto &node : graph_ir.nodes) {
        ggml_tensor *t = ggml_graph_get_tensor(cgraph, node.name.c_str());
        if (!t) {
          for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
            ggml_tensor *gn = ggml_graph_node(cgraph, i);
            if (gn->name[0] != '\0' &&
                std::string(gn->name) == node.name) {
              t = gn;
              break;
            }
          }
        }
        if (t && t->data && t->type == GGML_TYPE_F32) {
          float sum = tensor_sum(t);
          float min_val, max_val;
          tensor_min_max(t, min_val, max_val);
          std::cout << std::fixed << std::setprecision(6);
          std::cout << "  [" << std::setw(3) << node.id << "] "
                    << std::setw(20) << std::left << node.op << std::setw(40)
                    << std::left << node.name << " sum=" << sum
                    << " min=" << min_val << " max=" << max_val << " shape=["
                    << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << ","
                    << t->ne[3] << "]" << std::endl;
        }
      }
      std::cout << "=== End debug ===\n\n";
    }

    // Get energy results
    std::vector<float> atomic_energies(n_atoms);
    ggml_backend_tensor_get(output, atomic_energies.data(), 0,
                            n_atoms * sizeof(float));

    float model_energy = 0.0f;
    for (int i = 0; i < n_atoms; i++)
      model_energy += atomic_energies[i];

    // Apply energy scale factor (raw model output → scaled output)
    float scaled_model_energy = model_energy * model.energy_scale;

    float composition_energy = 0.0f;
    for (int i = 0; i < n_atoms; i++) {
      auto it = model.composition_energies.find(atomic_numbers[i]);
      if (it != model.composition_energies.end())
        composition_energy += it->second;
    }

    float total_energy = scaled_model_energy + composition_energy;

    // Print energy results
    std::cout << "\n=== Results ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Atomic energies:\n";
    for (int i = 0; i < n_atoms; i++) {
      std::cout << "  Atom " << i << ": " << atomic_energies[i] << " eV\n";
    }
    std::cout << "\nModel energy (raw): " << model_energy << " eV\n";
    if (model.energy_scale != 1.0f) {
      std::cout << "Energy scale:       " << model.energy_scale << "\n";
      std::cout << "Model energy:       " << scaled_model_energy << " eV\n";
    }
    if (composition_energy != 0.0f) {
      std::cout << "Composition energy: " << composition_energy << " eV\n";
    }
    std::cout << "Total energy:       " << total_energy << " eV\n";

    // Extract and print forces
    if (compute_forces) {
      ggml_tensor *grad_tensor = ggml_graph_get_grad(cgraph, edge_vectors);

      if (grad_tensor && grad_tensor->data) {
        // Read gradient tensor: shape [3, max_neighbors, n_atoms] in GGML
        std::vector<float> grad_data(ggml_nelements(grad_tensor));
        ggml_backend_tensor_get(grad_tensor, grad_data.data(), 0,
                                ggml_nbytes(grad_tensor));

        // Print gradient statistics (skip NaN from padding positions)
        float grad_min = 1e30f, grad_max = -1e30f, grad_sum = 0.0f;
        int nonzero = 0;
        for (size_t i = 0; i < grad_data.size(); i++) {
          if (std::isnan(grad_data[i])) continue;
          if (grad_data[i] < grad_min) grad_min = grad_data[i];
          if (grad_data[i] > grad_max) grad_max = grad_data[i];
          grad_sum += grad_data[i];
          if (grad_data[i] != 0.0f) nonzero++;
        }
        std::cout << "\n  Gradient stats: min=" << grad_min
                  << " max=" << grad_max << " sum=" << grad_sum
                  << " nonzero=" << nonzero << "/" << grad_data.size() << "\n";

        // Initialize per-atom forces
        std::vector<float> forces(n_atoms * 3, 0.0f);

        // Scatter edge gradients to position gradients
        // Chain rule: edge_vec = pos[neighbor] - pos[center]
        // Therefore: F[center] += grad, F[neighbor] -= grad
        const int stride_slot = 3;
        const int stride_atom = 3 * max_neighbors;

        for (int center_atom = 0; center_atom < n_atoms; center_atom++) {
          for (int slot = 0; slot < max_neighbors; slot++) {
            int flat_idx = center_atom * max_neighbors + slot;

            // Skip padding entries (pm_data: 0.0 = valid, 1.0 = padded)
            if (pm_data[flat_idx] > 0.5f)
              continue;

            int neighbor_atom = neighbor_atoms[flat_idx];
            if (neighbor_atom < 0)
              continue;

            // Get gradient for this edge
            int base_idx = slot * stride_slot + center_atom * stride_atom;
            float gx = grad_data[0 + base_idx];
            float gy = grad_data[1 + base_idx];
            float gz = grad_data[2 + base_idx];

            // Force = -gradient of energy
            // edge_vec = pos[neighbor] - pos[center]
            // d(energy)/d(pos[center]) contributes +grad to force[center]
            // d(energy)/d(pos[neighbor]) contributes -grad to force[neighbor]
            forces[center_atom * 3 + 0] += gx;
            forces[center_atom * 3 + 1] += gy;
            forces[center_atom * 3 + 2] += gz;

            forces[neighbor_atom * 3 + 0] -= gx;
            forces[neighbor_atom * 3 + 1] -= gy;
            forces[neighbor_atom * 3 + 2] -= gz;
          }
        }

        // Apply energy scale to forces
        // Forces = -dE/dr = -energy_scale * d(sum(ae))/dr
        // The backward pass gives d(sum(ae))/dr, so multiply by energy_scale
        for (int i = 0; i < n_atoms * 3; i++) {
          forces[i] *= model.energy_scale;
        }

        // Print forces
        std::cout << "\nForces (eV/A):\n";
        float force_sum[3] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < n_atoms; i++) {
          std::cout << "  Atom " << i << ": [" << std::setw(12)
                    << forces[i * 3 + 0] << ", " << std::setw(12)
                    << forces[i * 3 + 1] << ", " << std::setw(12)
                    << forces[i * 3 + 2] << "]\n";
          force_sum[0] += forces[i * 3 + 0];
          force_sum[1] += forces[i * 3 + 1];
          force_sum[2] += forces[i * 3 + 2];
        }
        float sum_mag = std::sqrt(force_sum[0] * force_sum[0] +
                                  force_sum[1] * force_sum[1] +
                                  force_sum[2] * force_sum[2]);
        std::cout << "\n  Force sum:  [" << std::setw(12) << force_sum[0]
                  << ", " << std::setw(12) << force_sum[1] << ", "
                  << std::setw(12) << force_sum[2] << "]"
                  << "  |F_sum| = " << sum_mag << "\n";
        if (sum_mag > 0.1f) {
          std::cout << "  Warning: |F_sum| > 0.1, Newton's third law "
                       "violation may indicate an issue.\n";
        }
      } else {
        std::cerr << "Warning: Gradient tensor not available after compute.\n";
      }
    }

    std::cout << "\nCompute time: " << std::fixed << std::setprecision(1)
              << compute_ms << " ms\n";

    // Cleanup
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    ggml_free(weight_ctx);
    ggml_backend_free(cpu_backend);

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
