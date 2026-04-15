#include "graph_model.h"
#include "core/gguf_loader.h"

#include <nlohmann/json.hpp>

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

using json = nlohmann::json;

namespace mlipcpp::runtime {

namespace {

// Bump cutoff function
float cutoff_func_bump(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  float tan_val = std::tan(static_cast<float>(M_PI) * x);
  return 0.5f * (1.0f + std::tanh(1.0f / tan_val));
}

// Cosine cutoff function
float cutoff_func_cosine(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  return 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * x));
}

} // namespace

// Context sizes
static constexpr size_t INPUT_CTX_SIZE = 16 * 1024 * 1024;   // 16 MB
static constexpr size_t COMPUTE_CTX_SIZE = 512 * 1024 * 1024; // 512 MB

GraphModel::GraphModel()
    : neighbor_builder_(NeighborListOptions{cutoff_, true, false}) {}

GraphModel::~GraphModel() {
  if (weight_buffer_) {
    ggml_backend_buffer_free(weight_buffer_);
  }
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
  }
  // compute_backend_ is owned by backend_provider_; do not free here.
}

bool GraphModel::load_from_gguf(const std::string &path) {
  constexpr size_t TEMP_CTX_SIZE = 512 * 1024 * 1024;

  // Create temporary context with data allocation
  ggml_context *temp_ctx = ggml_init({TEMP_CTX_SIZE, nullptr, false});
  if (!temp_ctx) {
    throw std::runtime_error("Failed to create temporary context for loading");
  }

  GGUFLoader loader(path, temp_ctx);
  int n_tensors = static_cast<int>(loader.get_tensor_names().size());

  // Read model hyperparameters
  cutoff_ = loader.get_float32("pet.cutoff", 4.5f);
  cutoff_width_ = loader.get_float32("pet.cutoff_width", 0.2f);
  energy_scale_ = loader.get_float32("pet.energy_scale", 1.0f);
  cutoff_function_ = loader.get_string("pet.cutoff_function", "cosine");
  forces_mode_ = (loader.get_int32("pet.forces_mode", 0) != 0);
  num_neighbors_adaptive_ = loader.get_float32("pet.num_neighbors_adaptive", 0.0f);

  // Update neighbor list builder with loaded cutoff
  neighbor_builder_ = NeighborListBuilder(NeighborListOptions{cutoff_, true, false});

  // Load graph JSON
  std::string graph_json = loader.get_string("graph.json", "");
  if (graph_json.empty()) {
    ggml_free(temp_ctx);
    throw std::runtime_error("No graph.json found in GGUF file");
  }
  interp_.load_graph(graph_json);

  // Load species mapping
  auto species_map = loader.get_array_int32("pet.species_map");
  for (size_t i = 0; i + 1 < species_map.size(); i += 2) {
    species_to_index_[species_map[i]] = species_map[i + 1];
  }

  // Load composition energies
  auto comp_keys = loader.get_array_int32("pet.composition_keys");
  auto comp_vals = loader.get_array_float32("pet.composition_values");
  if (comp_keys.size() != comp_vals.size()) {
    ggml_free(temp_ctx);
    throw std::runtime_error(
        "GraphModel: composition_keys and composition_values mismatch");
  }
  for (size_t i = 0; i < comp_keys.size(); i++) {
    composition_energies_[comp_keys[i]] = comp_vals[i];
  }

  // Create backend
  backend_provider_ = BackendProvider::create(backend_preference_);

  // Load weight shapes from metadata (PyTorch shapes, need reversal for GGML)
  std::string shapes_json = loader.get_string("graph.weight_shapes", "");
  json weight_shapes;
  if (!shapes_json.empty()) {
    weight_shapes = json::parse(shapes_json);
  }

  // Create weight context (metadata only, no data allocation)
  size_t ctx_size = ggml_tensor_overhead() * static_cast<size_t>(n_tensors);
  ctx_weights_ = ggml_init({ctx_size, nullptr, true});
  if (!ctx_weights_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to create weight context");
  }

  // Create weight tensors with correct GGML shapes (reversed PyTorch dims)
  for (const auto &name : loader.get_tensor_names()) {
    ggml_tensor *temp = loader.get_tensor(name);
    if (!temp) continue;

    ggml_tensor *t = nullptr;
    if (weight_shapes.contains(name)) {
      // Use PyTorch shape from metadata, reversed for GGML convention
      auto py_shape = weight_shapes[name].get<std::vector<int64_t>>();
      std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());
      switch (ggml_shape.size()) {
      case 0:
        t = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, 1);
        break;
      case 1:
        t = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0]);
        break;
      case 2:
        t = ggml_new_tensor_2d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0],
                               ggml_shape[1]);
        break;
      case 3:
        t = ggml_new_tensor_3d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0],
                               ggml_shape[1], ggml_shape[2]);
        break;
      default:
        continue;
      }
    } else {
      // Fallback: use GGUF stored shape directly
      t = ggml_new_tensor(
          ctx_weights_, temp->type, ggml_n_dims(temp), temp->ne);
    }

    ggml_set_name(t, name.c_str());
  }

  // Allocate backend buffer for weights
  ggml_backend_buffer_type_t buft = backend_provider_->buffer_type();
  weight_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights_, buft);
  if (!weight_buffer_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to allocate weight buffer");
  }
  ggml_backend_buffer_set_usage(weight_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  // Copy weight data and register with interpreter
  for (const auto &name : loader.get_tensor_names()) {
    ggml_tensor *temp = loader.get_tensor(name);
    ggml_tensor *weight = ggml_get_tensor(ctx_weights_, name.c_str());
    if (temp && weight) {
      ggml_backend_tensor_set(weight, temp->data, 0, ggml_nbytes(weight));
      interp_.set_weight(name, weight);
    }
  }

  ggml_free(temp_ctx);

  // Use primary backend (may be GPU) for compute; owned by BackendProvider.
  compute_backend_ = backend_provider_->primary();
  if (!compute_backend_) {
    throw std::runtime_error("Failed to get compute backend");
  }

  return true;
}

void GraphModel::load_graph_file(const std::string &path) {
  interp_.load_graph_file(path);
}

void GraphModel::set_weight(const std::string &name, ggml_tensor *tensor) {
  interp_.set_weight(name, tensor);
}

ModelResult GraphModel::predict(const AtomicSystem &system) {
  return predict_single(system, false);
}

ModelResult GraphModel::predict(const AtomicSystem &system,
                                bool compute_forces) {
  return predict_single(system, compute_forces);
}

ModelResult GraphModel::predict_single(const AtomicSystem &system,
                                       bool compute_forces) {
  if (compute_forces && !forces_mode_) {
    throw std::runtime_error(
        "GraphModel: forces requested but model was exported with "
        "--no-forces. Re-export without --no-forces to enable forces.");
  }

  const int n_atoms = static_cast<int>(system.num_atoms());
  const int32_t *atomic_numbers = system.atomic_numbers();

  // Build neighbor list
  NeighborList nlist = neighbor_builder_.build(system);

  // Count max neighbors
  std::vector<int> neighbor_counts(n_atoms, 0);
  for (int e = 0; e < nlist.num_pairs(); e++) {
    neighbor_counts[nlist.centers[e]]++;
  }
  int max_neighbors = 0;
  for (int i = 0; i < n_atoms; i++) {
    max_neighbors = std::max(max_neighbors, neighbor_counts[i]);
  }
  if (max_neighbors == 0) {
    max_neighbors = 1;
  }

  // Per-pair cutoff distances (for bump cutoff computation)
  std::vector<float> pair_cutoffs(nlist.num_pairs(), cutoff_);

  // Set symbolic dimensions for this system
  interp_.set_dimension("n_atoms", n_atoms);
  interp_.set_dimension("max_neighbors", max_neighbors);
  interp_.set_dimension("n_edges", n_atoms * max_neighbors);
  interp_.set_dimension("max_neighbors_plus_one", max_neighbors + 1);

  const int total_slots = n_atoms * max_neighbors;

  // --- Create input context ---
  ggml_context *input_ctx = ggml_init({INPUT_CTX_SIZE, nullptr, true});
  if (!input_ctx) {
    throw std::runtime_error("Failed to create input context");
  }

  // Create input tensors
  ggml_tensor *species_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species_t, "species");

  ggml_tensor *neighbor_species_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species_t, "neighbor_species");

  ggml_tensor *edge_vectors_t =
      ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors_t, "edge_vectors");

  ggml_tensor *padding_mask_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(padding_mask_t, "padding_mask");

  ggml_tensor *reverse_neighbor_index_t =
      ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, total_slots);
  ggml_set_name(reverse_neighbor_index_t, "reverse_neighbor_index");

  // Mode-specific inputs
  ggml_tensor *edge_distances_t = nullptr;
  ggml_tensor *cutoff_factors_t = nullptr;
  ggml_tensor *cutoff_values_t = nullptr;

  if (!forces_mode_) {
    edge_distances_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(edge_distances_t, "edge_distances");

    cutoff_factors_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(cutoff_factors_t, "cutoff_factors");
  } else {
    cutoff_values_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(cutoff_values_t, "cutoff_values");
  }

  // Mark edge_vectors as parameter for gradient computation
  if (compute_forces) {
    ggml_set_param(edge_vectors_t);
  }

  // Allocate input buffer
  ggml_backend_buffer_t input_buffer =
      ggml_backend_alloc_ctx_tensors(input_ctx, compute_backend_);
  if (!input_buffer) {
    ggml_free(input_ctx);
    throw std::runtime_error("Failed to allocate input buffer");
  }

  // --- Pack neighbor list data ---
  std::vector<int32_t> species_data(n_atoms);
  for (int i = 0; i < n_atoms; i++) {
    auto it = species_to_index_.find(atomic_numbers[i]);
    if (it == species_to_index_.end()) {
      ggml_backend_buffer_free(input_buffer);
      ggml_free(input_ctx);
      throw std::runtime_error(
          "Atomic number " + std::to_string(atomic_numbers[i]) +
          " not in species map");
    }
    species_data[i] = it->second;
  }

  std::vector<int32_t> ns_data(total_slots, 0);
  std::vector<float> ev_data(total_slots * 3, 0.0f);
  std::vector<float> ed_data(total_slots, 0.0f);
  std::vector<float> pm_data(total_slots, 1.0f);  // 1.0 = padded
  std::vector<float> cf_data(total_slots, 0.0f);
  std::vector<float> cv_data(total_slots, cutoff_);
  std::vector<int32_t> rni_data(total_slots, 0);
  std::vector<int> neighbor_atoms_vec(total_slots, -1);

  // Build edge mapping
  using EdgeKey = std::tuple<int, int, int, int, int>;
  std::map<EdgeKey, int> edge_to_flat_idx;
  std::vector<int> slot_indices(n_atoms, 0);
  bool has_cell_shifts = !nlist.cell_shifts.empty();

  for (int e = 0; e < nlist.num_pairs(); e++) {
    int i = nlist.centers[e];
    int j = nlist.neighbors[e];
    int slot = slot_indices[i]++;
    if (slot >= max_neighbors) continue;

    int flat_idx = i * max_neighbors + slot;

    int sa = 0, sb = 0, sc = 0;
    if (has_cell_shifts) {
      sa = nlist.cell_shifts[e][0];
      sb = nlist.cell_shifts[e][1];
      sc = nlist.cell_shifts[e][2];
    }
    edge_to_flat_idx[{i, j, sa, sb, sc}] = flat_idx;

    auto it = species_to_index_.find(atomic_numbers[j]);
    if (it == species_to_index_.end()) {
      ggml_backend_buffer_free(input_buffer);
      ggml_free(input_ctx);
      throw std::runtime_error(
          "Atomic number " + std::to_string(atomic_numbers[j]) +
          " not in species map");
    }
    ns_data[flat_idx] = it->second;

    const auto &ev = nlist.edge_vectors[e];
    int ev_idx = i * (max_neighbors * 3) + slot * 3;
    ev_data[ev_idx + 0] = ev[0];
    ev_data[ev_idx + 1] = ev[1];
    ev_data[ev_idx + 2] = ev[2];

    ed_data[flat_idx] = nlist.distances[e];
    pm_data[flat_idx] = 0.0f;  // valid edge
    neighbor_atoms_vec[flat_idx] = j;

    float r = nlist.distances[e];
    float pc = pair_cutoffs[e];
    cv_data[flat_idx] = pc;
    if (cutoff_function_ == "bump") {
      cf_data[flat_idx] = cutoff_func_bump(r, pc, cutoff_width_);
    } else {
      cf_data[flat_idx] = cutoff_func_cosine(r, pc, cutoff_width_);
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
    if (it_ij == edge_to_flat_idx.end()) continue;
    auto it_ji = edge_to_flat_idx.find({j, i, -sa, -sb, -sc});
    if (it_ji != edge_to_flat_idx.end()) {
      rni_data[it_ij->second] = it_ji->second;
    }
  }

  // Copy data to tensors
  ggml_backend_tensor_set(species_t, species_data.data(), 0,
                          species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(neighbor_species_t, ns_data.data(), 0,
                          ns_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(edge_vectors_t, ev_data.data(), 0,
                          ev_data.size() * sizeof(float));
  ggml_backend_tensor_set(padding_mask_t, pm_data.data(), 0,
                          pm_data.size() * sizeof(float));
  ggml_backend_tensor_set(reverse_neighbor_index_t, rni_data.data(), 0,
                          rni_data.size() * sizeof(int32_t));

  // Register common inputs
  interp_.set_input("species", species_t);
  interp_.set_input("neighbor_species", neighbor_species_t);
  interp_.set_input("edge_vectors", edge_vectors_t);
  interp_.set_input("padding_mask", padding_mask_t);
  interp_.set_input("reverse_neighbor_index", reverse_neighbor_index_t);

  // Register mode-specific inputs
  if (!forces_mode_) {
    ggml_backend_tensor_set(edge_distances_t, ed_data.data(), 0,
                            ed_data.size() * sizeof(float));
    ggml_backend_tensor_set(cutoff_factors_t, cf_data.data(), 0,
                            cf_data.size() * sizeof(float));
    interp_.set_input("edge_distances", edge_distances_t);
    interp_.set_input("cutoff_factors", cutoff_factors_t);
  } else {
    ggml_backend_tensor_set(cutoff_values_t, cv_data.data(), 0,
                            cv_data.size() * sizeof(float));
    interp_.set_input("cutoff_values", cutoff_values_t);
  }

  // --- Build and compute ---
  ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});
  if (!compute_ctx) {
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    throw std::runtime_error("Failed to create compute context");
  }

  ggml_tensor *output = interp_.build(compute_ctx);
  if (!output) {
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    throw std::runtime_error("Failed to build computation graph");
  }
  ggml_set_output(output);

  ggml_cgraph *cgraph = nullptr;

  if (compute_forces) {
    // Build forward + backward graph
    ggml_tensor *total_energy = ggml_sum(compute_ctx, output);
    ggml_set_loss(total_energy);
    ggml_set_output(total_energy);

    cgraph = ggml_new_graph_custom(compute_ctx, 32768, true);
    ggml_build_forward_expand(cgraph, output);
    ggml_build_forward_expand(cgraph, total_energy);
    ggml_build_backward_expand(compute_ctx, cgraph, nullptr);

    ggml_tensor *grad = ggml_graph_get_grad(cgraph, edge_vectors_t);
    if (grad) {
      ggml_set_output(grad);
    } else {
      compute_forces = false;
    }
  } else {
    cgraph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(cgraph, output);
  }

  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx, compute_backend_);
  if (!compute_buffer) {
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    throw std::runtime_error("Failed to allocate compute buffer");
  }

  interp_.init_constants();

  if (compute_forces) {
    ggml_graph_reset(cgraph);
  }

  ggml_status status = ggml_backend_graph_compute(compute_backend_, cgraph);
  if (status != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(compute_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(input_ctx);
    throw std::runtime_error("Graph computation failed");
  }

  // --- Extract results ---
  ModelResult result;

  // Get atomic energies
  std::vector<float> atomic_energies(n_atoms);
  ggml_backend_tensor_get(output, atomic_energies.data(), 0,
                          n_atoms * sizeof(float));

  // Sum and scale
  float model_energy = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    model_energy += atomic_energies[i];
  }
  float scaled_energy = model_energy * energy_scale_;

  // Add composition energies
  float composition_energy = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    auto it = composition_energies_.find(atomic_numbers[i]);
    if (it != composition_energies_.end()) {
      composition_energy += it->second;
    }
  }

  result.energy = scaled_energy + composition_energy;

  // Extract forces
  if (compute_forces) {
    ggml_tensor *grad_tensor = ggml_graph_get_grad(cgraph, edge_vectors_t);
    if (grad_tensor && grad_tensor->data) {
      std::vector<float> grad_data(ggml_nelements(grad_tensor));
      ggml_backend_tensor_get(grad_tensor, grad_data.data(), 0,
                              ggml_nbytes(grad_tensor));

      // Scatter edge gradients to per-atom forces
      result.forces.resize(n_atoms * 3, 0.0f);
      const int stride_slot = 3;
      const int stride_atom = 3 * max_neighbors;

      for (int ca = 0; ca < n_atoms; ca++) {
        for (int slot = 0; slot < max_neighbors; slot++) {
          int flat_idx = ca * max_neighbors + slot;
          if (pm_data[flat_idx] > 0.5f) continue;

          int na = neighbor_atoms_vec[flat_idx];
          if (na < 0) continue;

          int base = slot * stride_slot + ca * stride_atom;
          float gx = grad_data[0 + base];
          float gy = grad_data[1 + base];
          float gz = grad_data[2 + base];

          result.forces[ca * 3 + 0] += gx;
          result.forces[ca * 3 + 1] += gy;
          result.forces[ca * 3 + 2] += gz;

          result.forces[na * 3 + 0] -= gx;
          result.forces[na * 3 + 1] -= gy;
          result.forces[na * 3 + 2] -= gz;
        }
      }

      // Apply energy scale
      for (int i = 0; i < n_atoms * 3; i++) {
        result.forces[i] *= energy_scale_;
      }

      result.has_forces = true;
    }
  }

  // Cleanup
  ggml_backend_buffer_free(compute_buffer);
  ggml_free(compute_ctx);
  ggml_backend_buffer_free(input_buffer);
  ggml_free(input_ctx);

  return result;
}

} // namespace mlipcpp::runtime
