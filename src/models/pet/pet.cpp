#include "pet.h"
#include "core/gguf_loader.h"
#include "core/log.h"
#include "mlipcpp/timer.h"
#include "pet_batch.h"
#include "pet_graph.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <stdexcept>
#include <string_view>

namespace mlipcpp::pet {

// Memory allocation constants
constexpr size_t BATCH_CONTEXT_SIZE =
    128 * 1024 * 1024; // 128 MB for batch data
constexpr size_t TEMP_CONTEXT_SIZE =
    512 * 1024 * 1024; // 512 MB temporary workspace
constexpr size_t MAX_GRAPH_NODES_FORWARD =
    4096; // Max nodes for forward pass only
constexpr size_t MAX_GRAPH_NODES_BACKWARD =
    16384; // Max nodes when computing gradients

PETModel::PETModel(const PETHypers &hypers)
    : hypers_(hypers),
      neighbor_builder_(NeighborListOptions{hypers.cutoff, true, false}) {
  // Backend is initialized lazily in load_from_gguf() based on preference
}

PETModel::~PETModel() {
  // Free scheduler first (uses backends)
  if (sched_) {
    ggml_backend_sched_free(sched_);
    sched_ = nullptr;
  }

  // Free backend buffer (must be freed before backend provider is released)
  if (weight_buffer_) {
    ggml_backend_buffer_free(weight_buffer_);
    weight_buffer_ = nullptr;
  }

  // BackendProvider is released automatically via shared_ptr

  // Free contexts last
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
    ctx_weights_ = nullptr;
  }
  if (ctx_compute_) {
    ggml_free(ctx_compute_);
    ctx_compute_ = nullptr;
  }
}

ModelResult PETModel::predict(const AtomicSystem &system) {
  auto results = predict_batch({system}, false);
  return results[0];
}

ModelResult PETModel::predict(const AtomicSystem &system, bool compute_forces) {
  auto results = predict_batch({system}, compute_forces);
  return results[0];
}

std::vector<ModelResult>
PETModel::predict_batch(const std::vector<AtomicSystem> &systems,
                        bool compute_forces) {
  ScopedTimer total_timer(TimerCategory::Total);

  if (systems.empty()) {
    return {};
  }

  // Free previous compute context if it exists
  if (ctx_compute_) {
    ggml_free(ctx_compute_);
    ctx_compute_ = nullptr;
  }

  // Phase 1: Prepare batched input in a temporary context with allocated memory
  size_t batch_ctx_size = BATCH_CONTEXT_SIZE;
  ggml_context *batch_ctx = ggml_init(
      {batch_ctx_size, nullptr, false}); // no_alloc=false to hold input data
  if (!batch_ctx) {
    throw std::runtime_error("Failed to create batch context");
  }

  BatchedInput batch_src;
  {
    ScopedTimer batch_timer(TimerCategory::BatchPrep);
    batch_src =
        prepare_batch(batch_ctx, systems, neighbor_builder_, hypers_.cutoff,
                      hypers_.cutoff_width, species_to_index_);
  }

  // Create metadata-only compute context (no_alloc=true) for graph building
  // Need more nodes if computing forces (backward graph is larger)
  const size_t max_nodes =
      compute_forces ? MAX_GRAPH_NODES_BACKWARD : MAX_GRAPH_NODES_FORWARD;
  size_t meta_size = ggml_tensor_overhead() * max_nodes +
                     ggml_graph_overhead_custom(max_nodes, compute_forces);

  ctx_compute_ = ggml_init({meta_size, nullptr, true}); //  no_alloc=true
  if (!ctx_compute_) {
    ggml_free(batch_ctx);
    throw std::runtime_error("Failed to create compute context");
  }

  // Create a BatchedInput with tensors in ctx_compute_ that will be allocated
  // by scheduler. We create new tensors (not dups) and mark them as inputs,
  // then copy data after scheduler allocation.
  BatchedInput batch;
  batch.total_atoms = batch_src.total_atoms;
  batch.n_systems = batch_src.n_systems;
  batch.max_neighbors = batch_src.max_neighbors;
  batch.total_edges = batch_src.total_edges;
  batch.atoms_per_system = batch_src.atoms_per_system;
  batch.system_atom_offsets = batch_src.system_atom_offsets;

  // Helper to create input tensor with same shape as source
  auto create_input_tensor = [&](ggml_tensor *src) -> ggml_tensor * {
    ggml_tensor *t =
        ggml_new_tensor(ctx_compute_, src->type, ggml_n_dims(src), src->ne);
    ggml_set_input(t);
    return t;
  };

  // Create input tensors (data will be copied after scheduler allocation)
  batch.positions = create_input_tensor(batch_src.positions);
  batch.species = create_input_tensor(batch_src.species);
  batch.system_indices = create_input_tensor(batch_src.system_indices);
  batch.edge_vectors = create_input_tensor(batch_src.edge_vectors);
  batch.edge_distances = create_input_tensor(batch_src.edge_distances);
  batch.cutoff_factors = create_input_tensor(batch_src.cutoff_factors);
  batch.edge_center_indices = create_input_tensor(batch_src.edge_center_indices);
  batch.edge_neighbor_indices =
      create_input_tensor(batch_src.edge_neighbor_indices);

  // For gradient computation, edge_vectors_nef must be a parameter (leaf
  // tensor) so gradients flow back to it. Otherwise, use dup like other
  // tensors.
  if (compute_forces) {
    batch.edge_vectors_nef = ggml_new_tensor_3d(
        ctx_compute_, GGML_TYPE_F32, 3, batch.max_neighbors, batch.total_atoms);
    ggml_set_param(batch.edge_vectors_nef);

    // Compute edge_distances from edge_vectors IN THE GRAPH so gradients flow
    // through distance = sqrt(x^2 + y^2 + z^2) = sqrt(sum_rows(edge_vectors^2))
    ggml_tensor *edge_vec_sq = ggml_sqr(ctx_compute_, batch.edge_vectors_nef);
    // edge_vec_sq: [3, max_neighbors, total_atoms]
    ggml_tensor *sum_sq = ggml_sum_rows(ctx_compute_, edge_vec_sq);
    // sum_sq: [1, max_neighbors, total_atoms]
    ggml_tensor *distances_3d = ggml_sqrt(ctx_compute_, sum_sq);
    // Reshape to [max_neighbors, total_atoms]
    batch.edge_distances_nef = ggml_reshape_2d(
        ctx_compute_, distances_3d, batch.max_neighbors, batch.total_atoms);

    // Compute cutoff_factors IN THE GRAPH so gradients flow through the cutoff
    // function smooth_cutoff formula:
    //   x = (r - (r_cut - delta)) / delta
    //   x_clamped = clamp(x, 0, 1)
    //   cutoff = 0.5 + 0.5 * cos(pi * x_clamped)
    // This gives cutoff=1 for r <= r_cut-delta, cutoff=0 for r >= r_cut
    float r_cut = hypers_.cutoff;
    float delta = hypers_.cutoff_width;
    float inv_delta = 1.0f / delta;
    float offset_val = (delta - r_cut) * inv_delta; // = 1 - r_cut/delta

    // Create scalar constant input tensors (data will be set after allocation)
    ggml_tensor *offset_tensor =
        ggml_new_tensor_1d(ctx_compute_, GGML_TYPE_F32, 1);
    ggml_set_input(offset_tensor);
    ggml_tensor *half_tensor =
        ggml_new_tensor_1d(ctx_compute_, GGML_TYPE_F32, 1);
    ggml_set_input(half_tensor);

    // Store scalar values to set after allocation
    batch.scalar_offset = offset_tensor;
    batch.scalar_half = half_tensor;
    batch.offset_val = offset_val;
    batch.half_val = 0.5f;

    // x = distances * inv_delta + offset
    ggml_tensor *x_scaled =
        ggml_scale(ctx_compute_, batch.edge_distances_nef, inv_delta);
    ggml_tensor *x = ggml_add1(ctx_compute_, x_scaled, offset_tensor);
    // x_clamped = clamp(x, 0, 1) - essential for correct cutoff values
    ggml_tensor *x_clamped = ggml_clamp(ctx_compute_, x, 0.0f, 1.0f);
    // cutoff = 0.5 + 0.5 * cos(pi * x_clamped)
    ggml_tensor *scaled_x =
        ggml_scale(ctx_compute_, x_clamped, static_cast<float>(M_PI));
    ggml_tensor *cos_val = ggml_cos(ctx_compute_, scaled_x);
    ggml_tensor *cutoff_scaled = ggml_scale(ctx_compute_, cos_val, 0.5f);
    batch.cutoff_factors_nef =
        ggml_add1(ctx_compute_, cutoff_scaled, half_tensor);
  } else {
    batch.edge_vectors_nef = create_input_tensor(batch_src.edge_vectors_nef);
    batch.edge_distances_nef = create_input_tensor(batch_src.edge_distances_nef);
    batch.cutoff_factors_nef = create_input_tensor(batch_src.cutoff_factors_nef);
  }

  batch.neighbor_species_nef =
      create_input_tensor(batch_src.neighbor_species_nef);
  batch.neighbor_species_transposed =
      create_input_tensor(batch_src.neighbor_species_transposed);
  batch.padding_mask_nef = create_input_tensor(batch_src.padding_mask_nef);
  batch.neighbor_indices_nef =
      create_input_tensor(batch_src.neighbor_indices_nef);
  batch.reversed_neighbor_list =
      create_input_tensor(batch_src.reversed_neighbor_list);
  batch.reverse_edge_mask_nef =
      create_input_tensor(batch_src.reverse_edge_mask_nef);
  batch.attn_mask_layer0 = create_input_tensor(batch_src.attn_mask_layer0);
  batch.attn_mask_layer1 = create_input_tensor(batch_src.attn_mask_layer1);

  // Build forward graph (returns atomic energies)
  ggml_tensor *atomic_energies;
  ggml_tensor *total_energy = nullptr;
  ggml_cgraph *gf;
  {
    ScopedTimer graph_timer(TimerCategory::GraphBuild);
    atomic_energies = build_forward_graph(batch);

    // For gradient computation, we need a scalar loss
    if (compute_forces) {
      total_energy = ggml_sum(ctx_compute_, atomic_energies);
      ggml_set_loss(total_energy);
    }

    gf = ggml_new_graph_custom(ctx_compute_, 32768, compute_forces);
    ggml_build_forward_expand(gf, atomic_energies);

    if (compute_forces) {
      ggml_build_forward_expand(gf, total_energy);

      ggml_build_backward_expand(ctx_compute_, gf, nullptr);

      // Mark gradient tensor as output so scheduler computes it
      ggml_tensor *grad_tensor =
          ggml_graph_get_grad(gf, batch.edge_vectors_nef);
      if (grad_tensor) {
        ggml_set_output(grad_tensor);
      }
    }
  }

  // Create or recreate scheduler if needed
  if (sched_) {
    ggml_backend_sched_free(sched_);
  }

  // Create scheduler with backends - CPU must be last (as required by ggml)
  // When GPU is primary: {gpu, cpu}  - ops run on GPU, fallback to CPU
  // When CPU is primary: {cpu}       - everything on CPU
  if (backend_provider_->is_gpu()) {
    // GPU mode: primary backend + CPU fallback
    ggml_backend_t backends[2] = {backend_provider_->primary(),
                                  backend_provider_->cpu()};
    ggml_backend_buffer_type_t bufts[2] = {backend_provider_->buffer_type(),
                                           backend_provider_->cpu_buffer_type()};
    sched_ = ggml_backend_sched_new(backends, bufts, 2, max_nodes, false, false);
  } else {
    // CPU-only mode
    ggml_backend_t cpu = backend_provider_->cpu();
    ggml_backend_buffer_type_t buft = backend_provider_->cpu_buffer_type();
    sched_ = ggml_backend_sched_new(&cpu, &buft, 1, max_nodes, false, false);
  }
  if (!sched_) {
    throw std::runtime_error("Failed to create backend scheduler");
  }

  // Debug: print out_prod tensor shapes when MLIP_DEBUG_OUT_PROD is set
  if (std::getenv("MLIP_DEBUG_OUT_PROD")) {
    int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
      ggml_tensor *node = ggml_graph_node(gf, i);
      if (node->op == GGML_OP_OUT_PROD) {
        fprintf(stderr, "OUT_PROD node %d: dst[%ld,%ld,%ld,%ld] = src0[%ld,%ld,%ld,%ld] x src1[%ld,%ld,%ld,%ld]\n",
                i,
                node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
                node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);
        fprintf(stderr, "  src0 strides: nb[%ld,%ld,%ld,%ld]\n",
                node->src[0]->nb[0], node->src[0]->nb[1], node->src[0]->nb[2], node->src[0]->nb[3]);
        fprintf(stderr, "  src1 strides: nb[%ld,%ld,%ld,%ld]\n",
                node->src[1]->nb[0], node->src[1]->nb[1], node->src[1]->nb[2], node->src[1]->nb[3]);
      }
    }
  }

  // Allocate graph tensors in backend buffers
  if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
    throw std::runtime_error("Failed to allocate graph in backend buffers");
  }

  // Log scheduler diagnostics when profiling is enabled
  if (profiling_enabled_) {
    int n_splits = ggml_backend_sched_get_n_splits(sched_);
    int n_copies = ggml_backend_sched_get_n_copies(sched_);
    log::info("Scheduler: {} splits, {} copies", n_splits, n_copies);

    // Count ops assigned to each backend
    int n_nodes = ggml_graph_n_nodes(gf);
    std::map<std::string, int> backend_counts;
    for (int i = 0; i < n_nodes; i++) {
      ggml_tensor *node = ggml_graph_node(gf, i);
      ggml_backend_t node_backend =
          ggml_backend_sched_get_tensor_backend(sched_, node);
      const char *backend_name =
          node_backend ? ggml_backend_name(node_backend) : "CPU";
      backend_counts[backend_name]++;
    }

    log::info("Node backend distribution:");
    for (const auto &[name, count] : backend_counts) {
      log::info("  {}: {} nodes", name, count);
    }
  }

  // Copy all batch input data from batch_src (CPU memory) to scheduler-allocated
  // buffers. Only copy if the tensor has a buffer (was actually used in the
  // graph)
  auto copy_tensor_data = [](ggml_tensor *dst, ggml_tensor *src) {
    if (dst && dst->buffer && src && src->data) {
      ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    }
  };

  copy_tensor_data(batch.positions, batch_src.positions);
  copy_tensor_data(batch.species, batch_src.species);
  copy_tensor_data(batch.system_indices, batch_src.system_indices);
  copy_tensor_data(batch.edge_vectors, batch_src.edge_vectors);
  copy_tensor_data(batch.edge_distances, batch_src.edge_distances);
  copy_tensor_data(batch.cutoff_factors, batch_src.cutoff_factors);
  copy_tensor_data(batch.edge_center_indices, batch_src.edge_center_indices);
  copy_tensor_data(batch.edge_neighbor_indices, batch_src.edge_neighbor_indices);

  copy_tensor_data(batch.neighbor_species_nef, batch_src.neighbor_species_nef);
  copy_tensor_data(batch.neighbor_species_transposed,
                   batch_src.neighbor_species_transposed);
  copy_tensor_data(batch.padding_mask_nef, batch_src.padding_mask_nef);
  copy_tensor_data(batch.neighbor_indices_nef, batch_src.neighbor_indices_nef);
  copy_tensor_data(batch.reversed_neighbor_list, batch_src.reversed_neighbor_list);
  copy_tensor_data(batch.reverse_edge_mask_nef, batch_src.reverse_edge_mask_nef);
  copy_tensor_data(batch.attn_mask_layer0, batch_src.attn_mask_layer0);
  copy_tensor_data(batch.attn_mask_layer1, batch_src.attn_mask_layer1);

  if (compute_forces) {
    // For gradient computation, edge_vectors_nef is a parameter tensor
    if (batch.edge_vectors_nef && batch.edge_vectors_nef->buffer) {
      ggml_backend_tensor_set(batch.edge_vectors_nef,
                              batch_src.edge_vectors_nef->data, 0,
                              ggml_nbytes(batch_src.edge_vectors_nef));
    }

    // Set scalar constant values
    if (batch.scalar_offset && batch.scalar_offset->buffer) {
      ggml_backend_tensor_set(batch.scalar_offset, &batch.offset_val, 0,
                              sizeof(float));
    }
    if (batch.scalar_half && batch.scalar_half->buffer) {
      ggml_backend_tensor_set(batch.scalar_half, &batch.half_val, 0,
                              sizeof(float));
    }

    // Reset graph to initialize gradients (loss grad = 1, others = 0)
    // Must be called after allocation
    ggml_graph_reset(gf);
  } else {
    // Non-gradient mode: copy NEF tensors directly
    copy_tensor_data(batch.edge_vectors_nef, batch_src.edge_vectors_nef);
    copy_tensor_data(batch.edge_distances_nef, batch_src.edge_distances_nef);
    copy_tensor_data(batch.cutoff_factors_nef, batch_src.cutoff_factors_nef);
  }

  // Set up profiling callback if enabled
  struct ProfData {
    std::map<std::string, double> op_times_us;
    std::map<std::string, int> op_counts;
    int64_t last_time_us = 0;
    std::string last_op;
  };
  ProfData prof_data;

  if (profiling_enabled_) {
    prof_data.last_time_us = ggml_time_us();
    ggml_backend_sched_set_eval_callback(
        sched_,
        [](ggml_tensor *t, bool ask, void *user_data) -> bool {
          auto *pd = static_cast<ProfData *>(user_data);
          int64_t now = ggml_time_us();

          if (!ask) {
            // After execution - record time for this op
            std::string op_name = ggml_op_name(t->op);
            double elapsed = static_cast<double>(now - pd->last_time_us);
            pd->op_times_us[op_name] += elapsed;
            pd->op_counts[op_name]++;
          }
          pd->last_time_us = now;
          return true; // continue execution
        },
        &prof_data);
  }

  // Compute graph using backend scheduler
  {
    ScopedTimer compute_timer(TimerCategory::Compute);

    ggml_status status = ggml_backend_sched_graph_compute(sched_, gf);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("Graph computation failed");
    }
  }

  // Clear callback
  if (profiling_enabled_) {
    ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr);
  }

  // Print graph profiling info if enabled
  if (profiling_enabled_) {
    log::info("\n=== Graph Profiling ===");
    int n_nodes = ggml_graph_n_nodes(gf);
    log::info("Total nodes: {}", n_nodes);

    // Calculate total time from profiling data
    double total_time_us = 0;
    for (const auto &[op, time] : prof_data.op_times_us) {
      total_time_us += time;
    }

    // Sort ops by time (descending)
    std::vector<std::pair<std::string, double>> sorted_by_time(
        prof_data.op_times_us.begin(), prof_data.op_times_us.end());
    std::sort(sorted_by_time.begin(), sorted_by_time.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    log::info("\nOp timing (sorted by time):");
    log::info("  {:<16} {:>6} {:>10} {:>8}", "OP", "COUNT", "TIME (ms)",
              "PERCENT");
    log::info("  {:<16} {:>6} {:>10} {:>8}", "----------------", "------",
              "----------", "--------");

    for (const auto &[op, time_us] : sorted_by_time) {
      int count = prof_data.op_counts[op];
      double time_ms = time_us / 1000.0;
      double pct = (total_time_us > 0) ? 100.0 * time_us / total_time_us : 0;
      log::info("  {:<16} {:>6} {:>10.2f} {:>7.1f}%", op, count, time_ms, pct);
    }

    log::info("  {:<16} {:>6} {:>10.2f} {:>7.1f}%", "TOTAL", "",
              total_time_us / 1000.0, 100.0);

    // Print MUL_MAT sizes for optimization analysis
    log::info("\nMUL_MAT sizes (M x N x K):");
    for (int i = 0; i < n_nodes; i++) {
      ggml_tensor *node = ggml_graph_node(gf, i);
      if (node->op == GGML_OP_MUL_MAT) {
        // MUL_MAT: result = src1 @ src0^T
        // src0: [K, N], src1: [K, M] -> result: [M, N]
        int64_t M = node->ne[1];
        int64_t N = node->ne[0];
        int64_t K = node->src[0]->ne[0];
        int64_t flops = 2 * M * N * K;
        log::info("  {:>4} x {:>4} x {:>4}  ({:.2f} MFLOPs)", M, N, K,
                  flops / 1e6);
      }
    }
  }

  // Extract atomic energies from backend buffer
  size_t n_atomic_energies = ggml_nelements(atomic_energies);
  std::vector<float> atomic_energy_data(n_atomic_energies);
  ggml_backend_tensor_get(atomic_energies, atomic_energy_data.data(), 0,
                          ggml_nbytes(atomic_energies));

  // Get system indices from batch_src (which has allocated memory)
  int32_t *system_indices_data = (int32_t *)batch_src.system_indices->data;

  // Sum energies per system
  std::vector<double> system_energy_sums(systems.size(), 0.0);
  for (int atom_idx = 0; atom_idx < batch.total_atoms; ++atom_idx) {
    int sys_idx = system_indices_data[atom_idx];
    // atomic_energies is [1, total_atoms], so access with stride
    double atomic_energy = atomic_energy_data[atom_idx];
    system_energy_sums[sys_idx] += atomic_energy;
  }

  // Add composition energies
  std::vector<ModelResult> results(systems.size());
  for (size_t i = 0; i < systems.size(); ++i) {
    double total_energy = system_energy_sums[i];

    // Add composition energy contribution
    for (int j = 0; j < systems[i].num_atoms(); ++j) {
      int atomic_num = systems[i].atomic_numbers()[j];
      auto it = composition_energies_.find(atomic_num);
      if (it != composition_energies_.end()) {
        total_energy += it->second;
      }
    }

    results[i].energy = total_energy;
    results[i].has_forces = false;
    results[i].has_stress = false;
  }

  // Extract forces if requested
  if (compute_forces) {
    extract_forces(batch, batch_src, gf, results);
    extract_stress(batch, batch_src, gf, systems, results);
  }

  // Cleanup batch context
  ggml_free(batch_ctx);

  return results;
}

ggml_tensor *PETModel::build_forward_graph(const BatchedInput &batch) {
  // Create graph context for helper functions
  pet_graph_context gctx = {ctx_compute_, hypers_, batch, weights_,
                            compute_precision_};

  // Phase 2: Initial embeddings
  ggml_tensor *input_messages = initial_embeddings(batch);

  // Phase 3: GNN layers with message passing
  std::vector<ggml_tensor *> node_features_list;
  std::vector<ggml_tensor *> edge_features_list;

  // Layer 0
  ggml_tensor *node_features_0 = nullptr;
  ggml_tensor *edge_features_0 = nullptr;
  apply_gnn_layer(0, batch, input_messages, node_features_0, edge_features_0);
  node_features_list.push_back(node_features_0);
  edge_features_list.push_back(edge_features_0);

  // Message passing: average forward and reversed edge features using extracted
  // helper
  ggml_tensor *updated_messages =
      build_reversed_message_avg(gctx, input_messages, edge_features_0);

  // Layer 1 (uses updated messages from message passing)
  ggml_tensor *node_features_1 = nullptr;
  ggml_tensor *edge_features_1 = nullptr;
  apply_gnn_layer(1, batch, updated_messages, node_features_1, edge_features_1);
  node_features_list.push_back(node_features_1);
  edge_features_list.push_back(edge_features_1);

  // Store for post-compute logging

  // Phase 4: Aggregation using extracted helper
  return build_aggregation_and_output(gctx, node_features_list,
                                      edge_features_list);
}

ggml_tensor *PETModel::initial_embeddings(const BatchedInput &batch) {
  // PyTorch: element_indices_neighbors [n_atoms, max_neighbors]
  // GGML:    neighbor_species_nef      [max_neighbors, n_atoms]
  //
  // PyTorch output: [n_atoms, max_neighbors, d_pet]
  // GGML output:    [d_pet, max_neighbors, n_atoms]

  // Note: batch.max_neighbors is always >= 1 even for isolated atoms
  // (batch preparation sets it to 1 when there are no edges)

  // Flatten neighbor_species for embedding lookup
  // neighbor_species_transposed is [n_atoms, max_neighbors] (pre-transposed on CPU)
  // This avoids ggml_cont(ggml_transpose(...)) which triggers unsupported
  // I32->I32 copy on CUDA/HIP for non-contiguous tensors.
  //
  // Row-major flatten gives atom-major order: atom0_all_slots, atom1_all_slots, ...
  // Then gather, reshape to [d_pet, n_atoms, max_neighbors], permute back to NEF

  ggml_tensor *neighbor_species_flat =
      ggml_reshape_1d(ctx_compute_, batch.neighbor_species_transposed,
                      batch.max_neighbors * batch.total_atoms);
  // Flattened order: atom0_all_slots, atom1_all_slots, ...

  // Gather embeddings
  ggml_tensor *embeddings_flat =
      ggml_get_rows(ctx_compute_, weights_.embedding, neighbor_species_flat);
  // Result: [d_pet, total_atoms * max_neighbors] with atom-major order

  // Reshape to [d_pet, n_atoms, max_neighbors] matching the flatten order
  ggml_tensor *embeddings_atom_major =
      ggml_reshape_3d(ctx_compute_, embeddings_flat, hypers_.d_pet,
                      batch.total_atoms, batch.max_neighbors);

  // Permute to NEF format [d_pet, max_neighbors, n_atoms]
  ggml_tensor *input_messages =
      ggml_cont(ctx_compute_,
                ggml_permute(ctx_compute_, embeddings_atom_major, 0, 2, 1, 3));

  // NOTE: Cannot access tensor->data here because ctx_compute_ has
  // no_alloc=true Tensors will be allocated by the backend scheduler before
  // computation

  return input_messages;
}

ggml_tensor *PETModel::apply_gnn_layer(int layer_idx, const BatchedInput &batch,
                                       ggml_tensor *input_messages,
                                       ggml_tensor *&node_features,
                                       ggml_tensor *&edge_features) {

  const auto &layer_weights = weights_.gnn[layer_idx];

  // Create graph context for helper functions
  pet_graph_context gctx = {ctx_compute_, hypers_, batch, weights_,
                            compute_precision_};

  // Step 3.1: Embed center atom species
  ggml_tensor *node_embeds = build_node_embedding(
      gctx, layer_weights.node_embedder_weight, batch.species);

  // Step 3.2: Embed edge features (vectors + distances)
  ggml_tensor *edge_embeds = build_edge_embedding(gctx, layer_idx);

  // Step 3.3: Form tokens using extracted helper
  ggml_tensor *tokens =
      build_tokens(gctx, layer_idx, edge_embeds, input_messages);

  // Step 3.4: Prepend node token
  ggml_tensor *node_tokens = ggml_reshape_3d(
      ctx_compute_, node_embeds, hypers_.d_pet, 1, batch.total_atoms);

  ggml_tensor *all_tokens = ggml_concat(ctx_compute_, node_tokens, tokens, 1);
  all_tokens = ggml_cont(ctx_compute_,
                         all_tokens); // Ensure contiguous before transformer
  // [d_pet, 1+max_neighbors, total_atoms]

  // Step 3.5: Get pre-computed attention mask from batch
  ggml_tensor *attn_mask = build_attention_mask(gctx, layer_idx);

  // Step 3.6: Apply transformer layers
  ggml_tensor *cur = all_tokens;

  // CHECKPOINT: Before transformer - mark for logging
  if (layer_idx == 0) {
    ggml_set_output(cur); // Mark as output so it gets computed
    ggml_set_name(cur, "all_tokens_before_transformer_0");
  }

  for (int tf_idx = 0; tf_idx < hypers_.num_attention_layers; ++tf_idx) {
    // Multi-head attention using extracted helper
    ggml_tensor *attn_out =
        build_multi_head_attention(gctx, layer_idx, tf_idx, cur, attn_mask);

    // Apply transformer block (residual + norm + MLP + residual + norm)
    cur = build_transformer_block(gctx, layer_idx, tf_idx, cur, attn_mask,
                                  attn_out);
  }

  // Step 3.7: Extract node and edge features

  // Node features: first token (position 0 in sequence)
  node_features =
      ggml_view_2d(ctx_compute_, cur,
                   hypers_.d_pet,     // Width: d_pet (innermost dimension)
                   batch.total_atoms, // Height: n_atoms
                   cur->nb[2],        // Row stride: jump between atoms
                   0);                // Offset: 0 (start at first token)

  // Edge features: remaining tokens (positions 1 through max_neighbors)
  edge_features = ggml_view_3d(
      ctx_compute_, cur,
      hypers_.d_pet,       // Width: d_pet (innermost dimension)
      batch.max_neighbors, // Depth: max_neighbors (skip first token)
      batch.total_atoms,   // Height: n_atoms
      cur->nb[1],          // Stride between sequence positions
      cur->nb[2],          // Stride between atoms
      hypers_.d_pet *
          sizeof(float)); // Offset: skip first token (1 * d_pet elements)

  // Return input_messages unchanged
  // Message passing is now handled in build_forward_graph() between layers
  return input_messages;
}

void PETModel::extract_forces(const BatchedInput &batch,
                              const BatchedInput &batch_src, ggml_cgraph *gf,
                              std::vector<ModelResult> &results) {

  ggml_tensor *grad_edge_vectors_nef =
      ggml_graph_get_grad(gf, batch.edge_vectors_nef);

  if (!grad_edge_vectors_nef || !grad_edge_vectors_nef->buffer) {
    return;
  }

  // Read gradient tensor from backend
  // Tensor shape is [3, max_neighbors, total_atoms] in GGML
  std::vector<float> grad_data(ggml_nelements(grad_edge_vectors_nef));
  ggml_backend_tensor_get(grad_edge_vectors_nef, grad_data.data(), 0,
                          ggml_nbytes(grad_edge_vectors_nef));

  // Read neighbor indices from batch_src (CPU memory)
  const int32_t *neighbor_indices =
      static_cast<const int32_t *>(batch_src.neighbor_indices_nef->data);
  const float *padding_mask =
      static_cast<const float *>(batch_src.padding_mask_nef->data);

  // Initialize forces for all systems
  for (size_t sys_idx = 0; sys_idx < results.size(); ++sys_idx) {
    int n_atoms = batch.atoms_per_system[sys_idx];
    results[sys_idx].forces.resize(n_atoms * 3, 0.0f);
    results[sys_idx].has_forces = true;
  }

  // Build atom-to-system mapping
  std::vector<int> atom_to_sys(batch.total_atoms);
  std::vector<int> atom_to_local(batch.total_atoms);
  int offset = 0;
  for (size_t sys = 0; sys < results.size(); ++sys) {
    int n_atoms = batch.atoms_per_system[sys];
    for (int i = 0; i < n_atoms; ++i) {
      atom_to_sys[offset + i] = static_cast<int>(sys);
      atom_to_local[offset + i] = i;
    }
    offset += n_atoms;
  }

  // Scatter edge gradients to position gradients (forces = -grad)
  // Chain rule: edge_vec = pos[neighbor] - pos[center]
  // Therefore: F[center] = +grad, F[neighbor] = -grad
  const int stride_slot = 3;
  const int stride_atom = 3 * batch.max_neighbors;

  for (int center_atom = 0; center_atom < batch.total_atoms; ++center_atom) {
    int center_sys = atom_to_sys[center_atom];
    int center_local = atom_to_local[center_atom];

    for (int slot = 0; slot < batch.max_neighbors; ++slot) {
      int nef_idx = slot + center_atom * batch.max_neighbors;

      // Skip padding entries
      if (padding_mask[nef_idx] < 0.5f)
        continue;

      int neighbor_atom = neighbor_indices[nef_idx];

      // Get gradient for this edge
      int base_idx = slot * stride_slot + center_atom * stride_atom;
      float gx = grad_data[0 + base_idx];
      float gy = grad_data[1 + base_idx];
      float gz = grad_data[2 + base_idx];

      // Force = -gradient of energy
      // edge_vec = pos[neighbor] - pos[center]
      // d(energy)/d(pos[center]) contributes -grad to force[center]
      // d(energy)/d(pos[neighbor]) contributes +grad to force[neighbor]
      results[center_sys].forces[center_local * 3 + 0] += gx;
      results[center_sys].forces[center_local * 3 + 1] += gy;
      results[center_sys].forces[center_local * 3 + 2] += gz;

      int neighbor_sys = atom_to_sys[neighbor_atom];
      int neighbor_local = atom_to_local[neighbor_atom];
      results[neighbor_sys].forces[neighbor_local * 3 + 0] -= gx;
      results[neighbor_sys].forces[neighbor_local * 3 + 1] -= gy;
      results[neighbor_sys].forces[neighbor_local * 3 + 2] -= gz;
    }
  }
}

void PETModel::extract_stress(const BatchedInput &batch,
                              const BatchedInput &batch_src, ggml_cgraph *gf,
                              const std::vector<AtomicSystem> &systems,
                              std::vector<ModelResult> &results) {

  ggml_tensor *grad_edge_vectors_nef =
      ggml_graph_get_grad(gf, batch.edge_vectors_nef);

  if (!grad_edge_vectors_nef || !grad_edge_vectors_nef->buffer) {
    return;
  }

  // Read gradient tensor from backend
  // Tensor shape is [3, max_neighbors, total_atoms] in GGML
  std::vector<float> grad_data(ggml_nelements(grad_edge_vectors_nef));
  ggml_backend_tensor_get(grad_edge_vectors_nef, grad_data.data(), 0,
                          ggml_nbytes(grad_edge_vectors_nef));

  // Read edge vectors from batch_src
  const float *edge_vectors =
      static_cast<const float *>(batch_src.edge_vectors_nef->data);
  const float *padding_mask =
      static_cast<const float *>(batch_src.padding_mask_nef->data);

  // Initialize stress tensors (6 components in Voigt: xx, yy, zz, yz, xz, xy)
  for (size_t sys_idx = 0; sys_idx < systems.size(); ++sys_idx) {
    results[sys_idx].stress.resize(6, 0.0f);
  }

  // Build atom-to-system mapping
  std::vector<int> atom_to_sys(batch.total_atoms);
  int offset = 0;
  for (size_t sys = 0; sys < systems.size(); ++sys) {
    int n_atoms = batch.atoms_per_system[sys];
    for (int i = 0; i < n_atoms; ++i) {
      atom_to_sys[offset + i] = static_cast<int>(sys);
    }
    offset += n_atoms;
  }

  // Accumulate virial contributions from each edge
  // Virial: sigma_ab = (1/V) Sum_edges edge_vector_a * grad_b
  const int stride_slot = 3;
  const int stride_atom = 3 * batch.max_neighbors;

  for (int center_atom = 0; center_atom < batch.total_atoms; ++center_atom) {
    int center_sys = atom_to_sys[center_atom];

    for (int slot = 0; slot < batch.max_neighbors; ++slot) {
      int nef_idx = slot + center_atom * batch.max_neighbors;

      // Skip padding entries
      if (padding_mask[nef_idx] < 0.5f)
        continue;

      // Get edge vector for this edge [3, max_neighbors, total_atoms]
      int base_idx = slot * stride_slot + center_atom * stride_atom;
      float ex = edge_vectors[0 + base_idx];
      float ey = edge_vectors[1 + base_idx];
      float ez = edge_vectors[2 + base_idx];

      // Get gradient for this edge
      float gx = grad_data[0 + base_idx];
      float gy = grad_data[1 + base_idx];
      float gz = grad_data[2 + base_idx];

      // Virial contribution: outer product of edge_vector and gradient
      // Voigt notation: [xx, yy, zz, yz, xz, xy]
      // Off-diagonal components symmetrized: (a_i*b_j + a_j*b_i)/2
      results[center_sys].stress[0] += ex * gx;                    // xx
      results[center_sys].stress[1] += ey * gy;                    // yy
      results[center_sys].stress[2] += ez * gz;                    // zz
      results[center_sys].stress[3] += 0.5f * (ey * gz + ez * gy); // yz
      results[center_sys].stress[4] += 0.5f * (ex * gz + ez * gx); // xz
      results[center_sys].stress[5] += 0.5f * (ex * gy + ey * gx); // xy
    }
  }

  // Divide by cell volume and convert to proper units
  for (size_t sys_idx = 0; sys_idx < systems.size(); ++sys_idx) {
    const Cell *cell = systems[sys_idx].cell();
    if (cell && (cell->periodic[0] || cell->periodic[1] || cell->periodic[2])) {
      // Compute cell volume: det(cell_matrix)
      const auto &m = cell->matrix;
      float vol = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                  m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                  m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      vol = std::abs(vol);

      if (vol > 1e-10f) {
        for (int i = 0; i < 6; ++i) {
          results[sys_idx].stress[i] /= vol;
        }
        results[sys_idx].has_stress = true;
      }
    }
  }
}

bool PETModel::load_from_gguf(const std::string &path) {
  ScopedTimer load_timer(TimerCategory::WeightLoad);

  // Create backend provider if not already set
  if (!backend_provider_) {
    backend_provider_ = BackendProvider::create(backend_preference_);
  }

  try {
    // Free old resources if they exist
    if (weight_buffer_) {
      ggml_backend_buffer_free(weight_buffer_);
      weight_buffer_ = nullptr;
    }
    if (ctx_weights_) {
      ggml_free(ctx_weights_);
      ctx_weights_ = nullptr;
    }

    // Step 1: Open GGUF file with no_alloc=false to load data into a temporary
    // context
    size_t temp_ctx_size = TEMP_CONTEXT_SIZE;
    ggml_context *temp_ctx = ggml_init({temp_ctx_size, nullptr, false});
    if (!temp_ctx) {
      throw std::runtime_error(
          "Failed to create temporary context for loading");
    }

    GGUFLoader temp_loader(path, temp_ctx);
    int n_tensors = temp_loader.get_tensor_names().size();

    // Step 2: Create weight context with no_alloc=true (metadata only)
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    ctx_weights_ = ggml_init({ctx_size, nullptr, true}); //  no_alloc=true
    if (!ctx_weights_) {
      ggml_free(temp_ctx);
      throw std::runtime_error("Failed to create GGML weight context");
    }

    // Step 3: Create tensors (metadata only, tensor->data will be NULL)
    for (const auto &tensor_name : temp_loader.get_tensor_names()) {
      ggml_tensor *temp_tensor = temp_loader.get_tensor(tensor_name);
      if (!temp_tensor)
        continue;

      // Create metadata-only tensor in weight context
      ggml_tensor *tensor =
          ggml_new_tensor(ctx_weights_, temp_tensor->type,
                          ggml_n_dims(temp_tensor), temp_tensor->ne);
      ggml_set_name(tensor, tensor_name.c_str());
    }

    // Step 4: Allocate backend buffer for all weight tensors on the primary backend
    ggml_backend_buffer_type_t buft = backend_provider_->buffer_type();
    weight_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights_, buft);
    if (!weight_buffer_) {
      ggml_free(temp_ctx);
      throw std::runtime_error("Failed to allocate backend buffer for weights");
    }

    // Mark as weights buffer for scheduler
    ggml_backend_buffer_set_usage(weight_buffer_,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Step 5: Copy weight data from temporary context to backend buffer
    for (const auto &tensor_name : temp_loader.get_tensor_names()) {
      ggml_tensor *temp_tensor = temp_loader.get_tensor(tensor_name);
      ggml_tensor *weight_tensor =
          ggml_get_tensor(ctx_weights_, tensor_name.c_str());

      if (temp_tensor && weight_tensor) {
        // Copy data from temp context to backend buffer
        ggml_backend_tensor_set(weight_tensor, temp_tensor->data, 0,
                                ggml_nbytes(weight_tensor));
      }
    }

    // Load hyperparameters from metadata
    hypers_.cutoff = temp_loader.get_float32("pet.cutoff", hypers_.cutoff);
    hypers_.num_heads =
        temp_loader.get_int32("pet.num_heads", hypers_.num_heads);
    hypers_.num_gnn_layers =
        temp_loader.get_int32("pet.num_gnn_layers", hypers_.num_gnn_layers);
    hypers_.d_feedforward =
        temp_loader.get_int32("pet.d_feedforward", hypers_.d_feedforward);
    hypers_.d_head = temp_loader.get_int32("pet.d_head", hypers_.d_head);
    hypers_.cutoff_width =
        temp_loader.get_float32("pet.cutoff_width", hypers_.cutoff_width);

    // Update neighbor list builder with loaded cutoff
    neighbor_builder_ =
        NeighborListBuilder(NeighborListOptions{hypers_.cutoff, true, false});

    // Get weight tensors by name (they now have data in backend buffer)
    auto get_weight_tensor = [&](const std::string &name) -> ggml_tensor * {
      ggml_tensor *t = ggml_get_tensor(ctx_weights_, name.c_str());
      if (!t) {
        log::warn("Tensor '{}' not found in loaded weights", name);
      }
      return t;
    };

    // Load main embedding table
    weights_.embedding = get_weight_tensor("model.embedding.weight");
    if (!weights_.embedding) {
      throw std::runtime_error(
          "Required tensor 'model.embedding.weight' not found");
    }

    // Determine d_pet from embedding shape
    hypers_.d_pet = weights_.embedding->ne[0];

    // Load GNN layers (0 and 1)
    for (int gnn_idx = 0; gnn_idx < hypers_.num_gnn_layers; ++gnn_idx) {
      std::string prefix = "model.gnn." + std::to_string(gnn_idx) + ".";
      auto &gnn = weights_.gnn[gnn_idx];

      // Edge embedder
      gnn.edge_embedder_weight =
          get_weight_tensor(prefix + "edge_embedder.weight");
      gnn.edge_embedder_bias = get_weight_tensor(prefix + "edge_embedder.bias");

      // Node embedder
      gnn.node_embedder_weight =
          get_weight_tensor(prefix + "node_embedder.weight");

      // Neighbor embedder (only in layer 1+)
      if (gnn_idx > 0) {
        gnn.neighbor_embedder_weight =
            get_weight_tensor(prefix + "neighbor_embedder.weight");
      }

      // Compress layers
      gnn.compress_0_weight = get_weight_tensor(prefix + "compress.0.weight");
      gnn.compress_0_bias = get_weight_tensor(prefix + "compress.0.bias");
      gnn.compress_2_weight = get_weight_tensor(prefix + "compress.2.weight");
      gnn.compress_2_bias = get_weight_tensor(prefix + "compress.2.bias");

      // Transformer layers
      for (int tl_idx = 0; tl_idx < hypers_.num_attention_layers; ++tl_idx) {
        std::string tl_prefix = prefix + "tl." + std::to_string(tl_idx) + ".";
        auto &tl = gnn.tl[tl_idx];

        tl.attn_in_weight = get_weight_tensor(tl_prefix + "attn.in.weight");
        tl.attn_in_bias = get_weight_tensor(tl_prefix + "attn.in.bias");
        tl.attn_out_weight = get_weight_tensor(tl_prefix + "attn.out.weight");
        tl.attn_out_bias = get_weight_tensor(tl_prefix + "attn.out.bias");
        tl.norm_a_weight = get_weight_tensor(tl_prefix + "norm_a.weight");
        tl.norm_a_bias = get_weight_tensor(tl_prefix + "norm_a.bias");
        tl.norm_m_weight = get_weight_tensor(tl_prefix + "norm_m.weight");
        tl.norm_m_bias = get_weight_tensor(tl_prefix + "norm_m.bias");
        tl.mlp_0_weight = get_weight_tensor(tl_prefix + "mlp.0.weight");
        tl.mlp_0_bias = get_weight_tensor(tl_prefix + "mlp.0.bias");
        tl.mlp_3_weight = get_weight_tensor(tl_prefix + "mlp.3.weight");
        tl.mlp_3_bias = get_weight_tensor(tl_prefix + "mlp.3.bias");
      }
    }

    // Load output head weights for each GNN layer
    for (int layer_idx = 0; layer_idx < hypers_.num_gnn_layers; ++layer_idx) {
      std::string layer_str = std::to_string(layer_idx);
      auto &head = weights_.heads[layer_idx];

      // Node head
      head.node_head_0_weight = get_weight_tensor("model.node_heads.energy." +
                                                  layer_str + ".0.weight");
      head.node_head_0_bias =
          get_weight_tensor("model.node_heads.energy." + layer_str + ".0.bias");
      head.node_head_2_weight = get_weight_tensor("model.node_heads.energy." +
                                                  layer_str + ".2.weight");
      head.node_head_2_bias =
          get_weight_tensor("model.node_heads.energy." + layer_str + ".2.bias");

      // Edge head
      head.edge_head_0_weight = get_weight_tensor("model.edge_heads.energy." +
                                                  layer_str + ".0.weight");
      head.edge_head_0_bias =
          get_weight_tensor("model.edge_heads.energy." + layer_str + ".0.bias");
      head.edge_head_2_weight = get_weight_tensor("model.edge_heads.energy." +
                                                  layer_str + ".2.weight");
      head.edge_head_2_bias =
          get_weight_tensor("model.edge_heads.energy." + layer_str + ".2.bias");

      // Last layers
      head.node_last_weight = get_weight_tensor(
          "model.node_last_layers.energy." + layer_str + ".energy___0.weight");
      head.node_last_bias = get_weight_tensor("model.node_last_layers.energy." +
                                              layer_str + ".energy___0.bias");
      head.edge_last_weight = get_weight_tensor(
          "model.edge_last_layers.energy." + layer_str + ".energy___0.weight");
      head.edge_last_bias = get_weight_tensor("model.edge_last_layers.energy." +
                                              layer_str + ".energy___0.bias");
    }

    // Load composition energies from GGUF (stored as tensors in backend buffer)
    ggml_tensor *atomic_numbers_tensor =
        get_weight_tensor("composition.atomic_numbers");
    ggml_tensor *energies_tensor = get_weight_tensor("composition.energies");

    if (atomic_numbers_tensor && energies_tensor) {
      int n_species = atomic_numbers_tensor->ne[0];

      // Read data from backend buffer
      std::vector<int32_t> atomic_numbers(n_species);
      std::vector<float> energies(n_species);

      ggml_backend_tensor_get(atomic_numbers_tensor, atomic_numbers.data(), 0,
                              ggml_nbytes(atomic_numbers_tensor));
      ggml_backend_tensor_get(energies_tensor, energies.data(), 0,
                              ggml_nbytes(energies_tensor));

      for (int i = 0; i < n_species; ++i) {
        composition_energies_[atomic_numbers[i]] = energies[i];
        // Build species_to_index mapping: atomic_number -> embedding row index
        species_to_index_[atomic_numbers[i]] = i;
      }

      // Debug output
      if (getenv("MLIP_DEBUG_COMP")) {
        log::debug("Loaded {} composition energies:", n_species);
        for (int i = 0; i < n_species; ++i) {
          log::debug("  Z={}: {:.6f} eV", atomic_numbers[i], energies[i]);
        }
      }
    }

    // Free the temporary context (we're done with it)
    ggml_free(temp_ctx);

    return true;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load GGUF: " + std::string(e.what()));
  }
}

} // namespace mlipcpp::pet
