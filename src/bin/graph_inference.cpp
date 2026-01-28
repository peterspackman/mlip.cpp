/**
 * Graph-based inference on XYZ files using auto-exported PET models.
 *
 * Usage:
 *   graph_inference <model> <xyz_file>
 *
 * Where <model> is either:
 *   - A .gguf file (single file with graph + weights + metadata)
 *   - A directory containing pet_full.json, metadata.json, and *.bin weight files
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

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
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
  std::map<int, int> species_to_index;
  std::map<int, float> composition_energies;
};

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

  // Species mapping: [Z1, idx1, Z2, idx2, ...]
  auto species_map = loader.get_array_int32("pet.species_map");
  for (size_t i = 0; i + 1 < species_map.size(); i += 2) {
    model.species_to_index[species_map[i]] = species_map[i + 1];
  }

  // Composition energies
  auto comp_keys = loader.get_array_int32("pet.composition_keys");
  auto comp_vals = loader.get_array_float32("pet.composition_values");
  for (size_t i = 0; i < comp_keys.size() && i < comp_vals.size(); i++) {
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
    // Our Python writer stores shapes in PyTorch order, but the graph interpreter
    // expects GGML order (reversed).
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
  std::cerr << "Usage: " << prog << " <model> <xyz_file> [--debug]\n\n";
  std::cerr << "Arguments:\n";
  std::cerr << "  model     .gguf file or export directory\n";
  std::cerr << "  xyz_file  Input structure in XYZ format\n";
  std::cerr << "  --debug   Dump inputs and print intermediate tensor values\n\n";
  std::cerr << "Example:\n";
  std::cerr << "  " << prog << " pet-auto.gguf geometries/water.xyz\n";
  std::cerr << "  " << prog << " /tmp/pet_export geometries/water.xyz\n";
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string xyz_path = argv[2];
  bool debug = (argc == 4 && std::string(argv[3]) == "--debug");

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

    std::cout << "  Cutoff: " << model.cutoff << " A\n";
    std::cout << "  Species mapped: " << model.species_to_index.size() << "\n";
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

    // Count max neighbors
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

    ggml_tensor *edge_distances =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(edge_distances, "edge_distances");

    ggml_tensor *padding_mask =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(padding_mask, "padding_mask");

    ggml_tensor *reverse_neighbor_index =
        ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms * max_neighbors);
    ggml_set_name(reverse_neighbor_index, "reverse_neighbor_index");

    ggml_tensor *cutoff_factors =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(cutoff_factors, "cutoff_factors");

    ggml_backend_buffer_t input_buffer =
        ggml_backend_alloc_ctx_tensors(input_ctx, cpu_backend);

    // Prepare input data
    std::vector<int32_t> species_data(n_atoms);
    for (int i = 0; i < n_atoms; i++) {
      int Z = atomic_numbers[i];
      auto it = model.species_to_index.find(Z);
      species_data[i] = (it != model.species_to_index.end()) ? it->second : 0;
    }
    ggml_backend_tensor_set(species, species_data.data(), 0,
                            species_data.size() * sizeof(int32_t));

    std::vector<int32_t> ns_data(n_atoms * max_neighbors, 0);
    std::vector<float> ev_data(n_atoms * max_neighbors * 3, 0.0f);
    std::vector<float> ed_data(n_atoms * max_neighbors, 0.0f);
    std::vector<float> pm_data(n_atoms * max_neighbors, 0.0f);
    std::vector<float> cf_data(n_atoms * max_neighbors, 0.0f);
    std::vector<int32_t> rni_data(n_atoms * max_neighbors, 0);

    // Key: (center, neighbor, shift_a, shift_b, shift_c)
    // For periodic systems, the same (i,j) pair can have multiple edges
    // through different cell shifts, so we need the full key.
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
      ns_data[flat_idx] = (it != model.species_to_index.end()) ? it->second : 0;

      const auto &ev = nlist.edge_vectors[e];
      int ev_idx = i * (max_neighbors * 3) + slot * 3;
      ev_data[ev_idx + 0] = ev[0];
      ev_data[ev_idx + 1] = ev[1];
      ev_data[ev_idx + 2] = ev[2];

      ed_data[flat_idx] = nlist.distances[e];
      pm_data[flat_idx] = 1.0f;

      // PET cosine cutoff with width parameter
      float r = nlist.distances[e];
      float width = model.cutoff_width;
      if (r <= model.cutoff - width) {
        cf_data[flat_idx] = 1.0f;
      } else if (r < model.cutoff) {
        float scaled = (r - (model.cutoff - width)) / width;
        cf_data[flat_idx] = 0.5f * (1.0f + std::cos(scaled * 3.14159265f));
      } else {
        cf_data[flat_idx] = 0.0f;
      }
    }

    // Build reverse neighbor index
    // For edge i→j with cell shift (sa, sb, sc), the reverse is
    // j→i with cell shift (-sa, -sb, -sc).
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
      } else {
        rni_data[it_ij->second] = it_ij->second;
      }
    }

    ggml_backend_tensor_set(neighbor_species, ns_data.data(), 0,
                            ns_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(edge_vectors, ev_data.data(), 0,
                            ev_data.size() * sizeof(float));
    ggml_backend_tensor_set(edge_distances, ed_data.data(), 0,
                            ed_data.size() * sizeof(float));
    ggml_backend_tensor_set(padding_mask, pm_data.data(), 0,
                            pm_data.size() * sizeof(float));
    ggml_backend_tensor_set(reverse_neighbor_index, rni_data.data(), 0,
                            rni_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(cutoff_factors, cf_data.data(), 0,
                            cf_data.size() * sizeof(float));

    interp.set_input("species", species);
    interp.set_input("neighbor_species", neighbor_species);
    interp.set_input("edge_vectors", edge_vectors);
    interp.set_input("edge_distances", edge_distances);
    interp.set_input("padding_mask", padding_mask);
    interp.set_input("reverse_neighbor_index", reverse_neighbor_index);
    interp.set_input("cutoff_factors", cutoff_factors);

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
    constexpr size_t COMPUTE_CTX_SIZE = 256 * 1024 * 1024;
    ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});

    ggml_tensor *output = interp.build(compute_ctx);
    if (!output) {
      std::cerr << "Error: Failed to build computation graph\n";
      return 1;
    }
    ggml_set_output(output);

    ggml_cgraph *cgraph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(cgraph, output);

    ggml_backend_buffer_t compute_buffer =
        ggml_backend_alloc_ctx_tensors(compute_ctx, cpu_backend);
    interp.init_constants();

    std::cout << "\nComputing energy...\n";
    ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
    if (status != GGML_STATUS_SUCCESS) {
      std::cerr << "Error: Graph computation failed\n";
      return 1;
    }

    if (debug) {
      auto tensor_sum = [](ggml_tensor *t) -> float {
        if (!t || !t->data) return 0.0f;
        float sum = 0.0f;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
          for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
              for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                float *ptr = (float *)((char *)t->data +
                    i0 * t->nb[0] + i1 * t->nb[1] +
                    i2 * t->nb[2] + i3 * t->nb[3]);
                sum += *ptr;
              }
            }
          }
        }
        return sum;
      };

      auto tensor_min_max = [](ggml_tensor *t, float &min_val, float &max_val) {
        if (!t || !t->data) { min_val = max_val = 0.0f; return; }
        min_val = 1e30f; max_val = -1e30f;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
          for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
              for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                float *ptr = (float *)((char *)t->data +
                    i0 * t->nb[0] + i1 * t->nb[1] +
                    i2 * t->nb[2] + i3 * t->nb[3]);
                if (*ptr < min_val) min_val = *ptr;
                if (*ptr > max_val) max_val = *ptr;
              }
            }
          }
        }
      };

      std::cout << "\n=== Debug: Intermediate tensor sums ===\n";
      const auto &graph_ir = interp.graph();
      for (const auto &node : graph_ir.nodes) {
        // Find tensor by name using GGML graph API
        ggml_tensor *t = ggml_graph_get_tensor(cgraph, node.name.c_str());
        if (!t) {
          // Also search by iterating over graph nodes
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
                    << std::setw(20) << std::left << node.op
                    << std::setw(40) << std::left << node.name
                    << " sum=" << sum
                    << " min=" << min_val
                    << " max=" << max_val
                    << " shape=[" << t->ne[0] << "," << t->ne[1]
                    << "," << t->ne[2] << "," << t->ne[3] << "]"
                    << std::endl;
        }
      }
      std::cout << "=== End debug ===\n\n";
    }

    // Get results
    std::vector<float> atomic_energies(n_atoms);
    ggml_backend_tensor_get(output, atomic_energies.data(), 0,
                            n_atoms * sizeof(float));

    float model_energy = 0.0f;
    for (int i = 0; i < n_atoms; i++)
      model_energy += atomic_energies[i];

    float composition_energy = 0.0f;
    for (int i = 0; i < n_atoms; i++) {
      auto it = model.composition_energies.find(atomic_numbers[i]);
      if (it != model.composition_energies.end())
        composition_energy += it->second;
    }

    float total_energy = model_energy + composition_energy;

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Atomic energies:\n";
    for (int i = 0; i < n_atoms; i++) {
      std::cout << "  Atom " << i << ": " << atomic_energies[i] << " eV\n";
    }
    std::cout << "\nModel energy:       " << model_energy << " eV\n";
    if (composition_energy != 0.0f) {
      std::cout << "Composition energy: " << composition_energy << " eV\n";
    }
    std::cout << "Total energy:       " << total_energy << " eV\n";

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
