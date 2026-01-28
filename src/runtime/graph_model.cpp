#include "graph_model.h"
#include "core/ggml_utils.h"
#include "core/gguf_loader.h"
#include "models/pet/pet_batch.h"
#include "models/pet/pet_types.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mlipcpp::runtime {

// Context sizes for batch preparation and graph computation
static constexpr size_t BATCH_CONTEXT_SIZE = 128 * 1024 * 1024;  // 128 MB
static constexpr size_t COMPUTE_CONTEXT_SIZE = 512 * 1024 * 1024; // 512 MB

GraphModel::GraphModel()
    : neighbor_builder_(NeighborListOptions{cutoff_, true, false}) {}

GraphModel::~GraphModel() {
  if (weight_buffer_) {
    ggml_backend_buffer_free(weight_buffer_);
  }
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
  }
}

bool GraphModel::load_from_gguf(const std::string &path) {
  constexpr size_t TEMP_CONTEXT_SIZE = 512 * 1024 * 1024;  // 512 MB for temp loading

  // Step 1: Create temporary context with no_alloc=false to load data
  ggml_context *temp_ctx = ggml_init({TEMP_CONTEXT_SIZE, nullptr, false});
  if (!temp_ctx) {
    throw std::runtime_error("Failed to create temporary context for loading");
  }

  // Load GGUF file into temp context
  GGUFLoader temp_loader(path, temp_ctx);
  int n_tensors = static_cast<int>(temp_loader.get_tensor_names().size());

  // Get model hyperparameters
  cutoff_ = temp_loader.get_float32("pet.cutoff", 4.5f);
  cutoff_width_ = temp_loader.get_float32("pet.cutoff_width", 0.5f);

  // Update neighbor list builder
  neighbor_builder_ = NeighborListBuilder(NeighborListOptions{cutoff_, true, false});

  // Load graph JSON from metadata
  std::string graph_json = temp_loader.get_string("graph.json", "");

  if (graph_json.empty()) {
    ggml_free(temp_ctx);
    throw std::runtime_error("No graph.json found in GGUF file");
  }

  // Parse the graph
  interp_.load_graph(graph_json);

  // Load species mapping
  auto species_map = temp_loader.get_array_int32("pet.species_map");
  for (size_t i = 0; i < species_map.size(); i += 2) {
    if (i + 1 < species_map.size()) {
      species_to_index_[species_map[i]] = species_map[i + 1];
    }
  }

  // Load composition energies
  auto comp_keys = temp_loader.get_array_int32("pet.composition_keys");
  auto comp_vals = temp_loader.get_array_float32("pet.composition_values");
  for (size_t i = 0; i < comp_keys.size() && i < comp_vals.size(); i++) {
    composition_energies_[comp_keys[i]] = comp_vals[i];
  }

  // Create backend
  backend_provider_ = BackendProvider::create(backend_preference_);

  // Step 2: Create weight context with no_alloc=true (metadata only)
  size_t ctx_size = ggml_tensor_overhead() * static_cast<size_t>(n_tensors);
  ctx_weights_ = ggml_init({ctx_size, nullptr, true});  // no_alloc=true
  if (!ctx_weights_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to create GGML weight context");
  }

  // Step 3: Create tensors (metadata only, tensor->data will be NULL)
  for (const auto &tensor_name : temp_loader.get_tensor_names()) {
    ggml_tensor *temp_tensor = temp_loader.get_tensor(tensor_name);
    if (!temp_tensor) continue;

    // Create metadata-only tensor in weight context
    ggml_tensor *tensor = ggml_new_tensor(
        ctx_weights_, temp_tensor->type,
        ggml_n_dims(temp_tensor), temp_tensor->ne);
    ggml_set_name(tensor, tensor_name.c_str());
  }

  // Step 4: Allocate backend buffer for all weight tensors
  ggml_backend_buffer_type_t buft = backend_provider_->buffer_type();
  weight_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights_, buft);
  if (!weight_buffer_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to allocate backend buffer for weights");
  }

  // Mark as weights buffer for scheduler
  ggml_backend_buffer_set_usage(weight_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  // Step 5: Copy weight data from temporary context to backend buffer
  for (const auto &tensor_name : temp_loader.get_tensor_names()) {
    ggml_tensor *temp_tensor = temp_loader.get_tensor(tensor_name);
    ggml_tensor *weight_tensor = ggml_get_tensor(ctx_weights_, tensor_name.c_str());

    if (temp_tensor && weight_tensor) {
      // Copy data from temp context to backend buffer
      ggml_backend_tensor_set(weight_tensor, temp_tensor->data, 0,
                              ggml_nbytes(weight_tensor));
      // Register weight with interpreter
      interp_.set_weight(tensor_name, weight_tensor);
    }
  }

  // Free temporary context
  ggml_free(temp_ctx);

  // Build input mappings
  build_input_mappings();

  return true;
}

void GraphModel::load_graph_file(const std::string &path) {
  interp_.load_graph_file(path);
  build_input_mappings();
}

void GraphModel::set_weight(const std::string &name, ggml_tensor *tensor) {
  interp_.set_weight(name, tensor);
}

void GraphModel::build_input_mappings() {
  // Map graph input names to BatchedInput tensor field names
  // This is based on the expected export format from export_pet_gguf.py
  input_mappings_.clear();

  const auto &graph = interp_.graph();

  // Check if this is a direct-format graph (has species, neighbor_species, edge_vectors, edge_distances)
  bool has_neighbor_species = false;
  bool has_edge_vectors = false;
  for (const auto &input : graph.inputs) {
    if (input.name == "neighbor_species") has_neighbor_species = true;
    if (input.name == "edge_vectors") has_edge_vectors = true;
  }
  uses_direct_inputs_ = has_neighbor_species && has_edge_vectors;

  for (const auto &input : graph.inputs) {
    InputMapping mapping;
    mapping.graph_name = input.name;

    if (uses_direct_inputs_) {
      // Direct format: inputs match graph input names exactly
      mapping.batch_field = input.name;
    } else {
      // NEF format: map to BatchedInput field names
      if (input.name == "tokens" || input.name == "input_messages") {
        mapping.batch_field = "tokens";
      } else if (input.name == "positions") {
        mapping.batch_field = "positions";
      } else if (input.name == "species") {
        mapping.batch_field = "species";
      } else if (input.name == "edge_vectors_nef") {
        mapping.batch_field = "edge_vectors_nef";
      } else if (input.name == "edge_distances_nef") {
        mapping.batch_field = "edge_distances_nef";
      } else if (input.name == "cutoff_factors" ||
                 input.name == "cutoff_factors_nef") {
        mapping.batch_field = "cutoff_factors_nef";
      } else if (input.name == "neighbor_species_nef") {
        mapping.batch_field = "neighbor_species_nef";
      } else if (input.name == "padding_mask_nef") {
        mapping.batch_field = "padding_mask_nef";
      } else if (input.name == "attn_mask" || input.name == "attention_mask") {
        mapping.batch_field = "attn_mask_layer0";
      } else {
        mapping.batch_field = input.name;
      }
    }

    input_mappings_.push_back(mapping);
  }

  // Detect dimensions from graph
  detect_dimensions_from_graph();
}

void GraphModel::detect_dimensions_from_graph() {
  // Extract expected dimensions from graph input shapes
  const auto &graph = interp_.graph();

  for (const auto &input : graph.inputs) {
    if (input.name == "species" && !input.shape.empty()) {
      // species shape is [n_atoms]
      expected_n_atoms_ = static_cast<int>(input.shape[0]);
    } else if (input.name == "neighbor_species" && input.shape.size() >= 2) {
      // neighbor_species shape is [n_atoms, max_neighbors]
      expected_n_atoms_ = static_cast<int>(input.shape[0]);
      expected_max_neighbors_ = static_cast<int>(input.shape[1]);
    } else if (input.name == "edge_vectors" && input.shape.size() >= 2) {
      // edge_vectors shape is [n_atoms, max_neighbors, 3]
      expected_n_atoms_ = static_cast<int>(input.shape[0]);
      expected_max_neighbors_ = static_cast<int>(input.shape[1]);
    }
  }
}

void GraphModel::register_batch_inputs(ggml_context * /*ctx*/,
                                       const pet::BatchedInput &batch) {
  // Register each graph input with the corresponding batch tensor
  for (const auto &mapping : input_mappings_) {
    ggml_tensor *tensor = nullptr;

    // Get the tensor from BatchedInput based on field name
    if (mapping.batch_field == "positions") {
      tensor = batch.positions;
    } else if (mapping.batch_field == "species") {
      tensor = batch.species;
    } else if (mapping.batch_field == "edge_vectors_nef") {
      tensor = batch.edge_vectors_nef;
    } else if (mapping.batch_field == "edge_distances_nef") {
      tensor = batch.edge_distances_nef;
    } else if (mapping.batch_field == "cutoff_factors_nef") {
      tensor = batch.cutoff_factors_nef;
    } else if (mapping.batch_field == "neighbor_species_nef") {
      tensor = batch.neighbor_species_nef;
    } else if (mapping.batch_field == "padding_mask_nef") {
      tensor = batch.padding_mask_nef;
    } else if (mapping.batch_field == "attn_mask_layer0") {
      tensor = batch.attn_mask_layer0;
    } else if (mapping.batch_field == "attn_mask_layer1") {
      tensor = batch.attn_mask_layer1;
    } else if (mapping.batch_field == "neighbor_indices_nef") {
      tensor = batch.neighbor_indices_nef;
    } else if (mapping.batch_field == "system_indices") {
      tensor = batch.system_indices;
    }

    if (tensor) {
      interp_.set_input(mapping.graph_name, tensor);
    }
  }
}

void GraphModel::prepare_direct_inputs(ggml_context *ctx,
                                       const AtomicSystem &system,
                                       const NeighborList &nlist) {
  // Prepare inputs in PyTorch format for direct-format exported graphs
  // Format: species[n_atoms], neighbor_species[n_atoms, max_neighbors],
  //         edge_vectors[n_atoms, max_neighbors, 3], edge_distances[n_atoms, max_neighbors]

  const int n_atoms = static_cast<int>(system.num_atoms());
  const int max_neighbors = expected_max_neighbors_;

  if (n_atoms != expected_n_atoms_) {
    std::ostringstream msg;
    msg << "GraphModel: system has " << n_atoms << " atoms but graph expects "
        << expected_n_atoms_ << " atoms. Re-export graph with matching dimensions.";
    throw std::runtime_error(msg.str());
  }

  // Count neighbors per atom from flat neighbor list
  std::vector<int> neighbor_counts(n_atoms, 0);
  for (int e = 0; e < nlist.num_pairs(); e++) {
    int i = nlist.centers[e];
    neighbor_counts[i]++;
  }

  // Check max neighbors
  int actual_max_neighbors = 0;
  for (int i = 0; i < n_atoms; i++) {
    actual_max_neighbors = std::max(actual_max_neighbors, neighbor_counts[i]);
  }
  if (actual_max_neighbors > max_neighbors) {
    std::ostringstream msg;
    msg << "GraphModel: system has " << actual_max_neighbors
        << " max neighbors but graph expects " << max_neighbors
        << ". Re-export graph with larger max_neighbors.";
    throw std::runtime_error(msg.str());
  }

  // Create tensors in PyTorch format (will be converted by interpreter)
  // Note: We create with no_alloc=false context, so data is allocated inline

  // Species: [n_atoms] int32
  ggml_tensor *species = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species, "species");
  auto *species_data = static_cast<int32_t *>(species->data);
  const int32_t *atomic_numbers = system.atomic_numbers();
  for (int i = 0; i < n_atoms; i++) {
    int Z = atomic_numbers[i];
    auto it = species_to_index_.find(Z);
    species_data[i] = (it != species_to_index_.end()) ? it->second : 0;
  }

  // Neighbor species: [n_atoms, max_neighbors] int32
  ggml_tensor *neighbor_species =
      ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species, "neighbor_species");
  auto *ns_data = static_cast<int32_t *>(neighbor_species->data);
  std::fill(ns_data, ns_data + n_atoms * max_neighbors, 0);

  // Edge vectors: [n_atoms, max_neighbors, 3] float32
  ggml_tensor *edge_vectors =
      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors, "edge_vectors");
  auto *ev_data = static_cast<float *>(edge_vectors->data);
  std::fill(ev_data, ev_data + n_atoms * max_neighbors * 3, 0.0f);

  // Edge distances: [n_atoms, max_neighbors] float32
  ggml_tensor *edge_distances =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(edge_distances, "edge_distances");
  auto *ed_data = static_cast<float *>(edge_distances->data);
  std::fill(ed_data, ed_data + n_atoms * max_neighbors, 0.0f);

  // Track slot indices for each atom
  std::vector<int> slot_indices(n_atoms, 0);

  // Fill neighbor data from flat neighbor list
  for (int e = 0; e < nlist.num_pairs(); e++) {
    int i = nlist.centers[e];      // center atom
    int j = nlist.neighbors[e];    // neighbor atom
    int slot = slot_indices[i]++;  // current slot for this center atom

    if (slot >= max_neighbors) continue;  // shouldn't happen if check above passed

    // Get neighbor species index
    int Z_j = atomic_numbers[j];
    auto it = species_to_index_.find(Z_j);
    int species_idx = (it != species_to_index_.end()) ? it->second : 0;

    // Store neighbor species
    // Memory layout: [n_atoms, max_neighbors] in row-major = data[i * max_neighbors + slot]
    ns_data[i * max_neighbors + slot] = species_idx;

    // Get edge vector (already computed in neighbor list)
    const auto &ev = nlist.edge_vectors[e];

    // Store edge vector
    // Memory layout: [n_atoms, max_neighbors, 3] in row-major
    int ev_idx = i * (max_neighbors * 3) + slot * 3;
    ev_data[ev_idx + 0] = ev[0];
    ev_data[ev_idx + 1] = ev[1];
    ev_data[ev_idx + 2] = ev[2];

    // Store edge distance
    ed_data[i * max_neighbors + slot] = nlist.distances[e];
  }

  // Register inputs with interpreter
  interp_.set_input("species", species);
  interp_.set_input("neighbor_species", neighbor_species);
  interp_.set_input("edge_vectors", edge_vectors);
  interp_.set_input("edge_distances", edge_distances);
}

ModelResult GraphModel::predict(const AtomicSystem &system) {
  return predict(system, false);
}

ModelResult GraphModel::predict(const AtomicSystem &system,
                                bool compute_forces) {
  auto results = predict_batch({system}, compute_forces);
  return results.empty() ? ModelResult{} : results[0];
}

std::vector<ModelResult>
GraphModel::predict_batch(const std::vector<AtomicSystem> &systems,
                          bool compute_forces) {
  if (systems.empty()) {
    return {};
  }

  // Currently force computation not supported via graph interpreter
  if (compute_forces) {
    throw std::runtime_error(
        "Force computation not yet supported in GraphModel");
  }

  // For direct-input graphs, only single systems are supported for now
  if (uses_direct_inputs_ && systems.size() > 1) {
    throw std::runtime_error(
        "GraphModel with direct inputs only supports single systems. "
        "Use NEF-format graphs for batched prediction.");
  }

  // Create input context (allocating)
  ggml::Context input_ctx(BATCH_CONTEXT_SIZE, false);

  int total_atoms = 0;
  std::vector<int> atoms_per_system;
  std::vector<int> system_atom_offsets;

  if (uses_direct_inputs_) {
    // Direct input format: prepare inputs from AtomicSystem directly
    const auto &system = systems[0];
    total_atoms = static_cast<int>(system.num_atoms());
    atoms_per_system.push_back(total_atoms);
    system_atom_offsets.push_back(0);

    // Build neighbor list
    NeighborList nlist = neighbor_builder_.build(system);

    // Prepare direct inputs
    prepare_direct_inputs(input_ctx.get(), system, nlist);
  } else {
    // NEF format: use PET's batch preparation
    pet::BatchedInput batch =
        pet::prepare_batch(input_ctx.get(), systems, neighbor_builder_, cutoff_,
                           cutoff_width_, species_to_index_);
    total_atoms = batch.total_atoms;
    atoms_per_system = batch.atoms_per_system;
    system_atom_offsets = batch.system_atom_offsets;

    // Register batch inputs
    register_batch_inputs(input_ctx.get(), batch);
  }

  // Create compute context (no_alloc for backend allocation)
  ggml::Context compute_ctx(COMPUTE_CONTEXT_SIZE, true);

  // Build the computation graph
  ggml_tensor *output = interp_.build(compute_ctx.get());
  if (!output) {
    throw std::runtime_error("Failed to build computation graph");
  }
  ggml_set_output(output);

  // Create GGML compute graph
  ggml_cgraph *cgraph = ggml_new_graph(compute_ctx.get());
  ggml_build_forward_expand(cgraph, output);

  // Allocate tensors on CPU backend
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  if (!cpu_backend) {
    throw std::runtime_error("Failed to create CPU backend");
  }

  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx.get(), cpu_backend);
  if (!compute_buffer) {
    ggml_backend_free(cpu_backend);
    throw std::runtime_error("Failed to allocate compute buffer");
  }

  // Initialize any pending constants
  interp_.init_constants();

  // Compute the graph
  ggml_status status = ggml_backend_graph_compute(cpu_backend, cgraph);
  if (status != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_free(cpu_backend);
    throw std::runtime_error("Graph computation failed");
  }

  // Extract results
  std::vector<ModelResult> results(systems.size());

  // Get output data (atomic energies)
  std::vector<float> atomic_energies(total_atoms);
  ggml_backend_tensor_get(output, atomic_energies.data(), 0,
                          total_atoms * sizeof(float));

  // Sum atomic energies per system and add composition energies
  for (size_t sys_idx = 0; sys_idx < systems.size(); sys_idx++) {
    float energy = 0.0f;
    int atom_start = system_atom_offsets[sys_idx];
    int n_atoms = atoms_per_system[sys_idx];

    for (int i = 0; i < n_atoms; i++) {
      energy += atomic_energies[atom_start + i];
    }

    // Add composition energies (atomic reference energies)
    for (int i = 0; i < n_atoms; i++) {
      int Z = systems[sys_idx].atomic_numbers()[i];
      auto it = composition_energies_.find(Z);
      if (it != composition_energies_.end()) {
        energy += it->second;
      }
    }

    results[sys_idx].energy = energy;
  }

  // Cleanup
  ggml_backend_buffer_free(compute_buffer);
  ggml_backend_free(cpu_backend);

  return results;
}

} // namespace mlipcpp::runtime
