#include "pet_batch.h"
#include "core/log.h"
#include "mlipcpp/timer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <unordered_map>

namespace mlipcpp::pet {

namespace {

// Internal data structure for passing edge-related data between helper
// functions
struct EdgeData {
  std::vector<float> edge_vectors_cpu;
  std::vector<float> edge_distances_cpu;
  std::vector<float> cutoff_factors_cpu;
  std::vector<int32_t> edge_center_indices_cpu;
  std::vector<int32_t> edge_neighbor_indices_cpu;
  std::vector<std::array<int32_t, 3>> edge_cell_shifts_cpu;
};

// Internal data structure for passing atom-related data between helper
// functions
struct AtomData {
  std::vector<float> positions_cpu;
  std::vector<int32_t> species_cpu;
  std::vector<int32_t> species_indices_cpu;
  std::vector<int32_t> system_indices_cpu;
};

} // anonymous namespace

float smooth_cutoff(float r, float r_cut, float delta) {
  if (r >= r_cut) {
    return 0.0f;
  }
  if (r <= r_cut - delta) {
    return 1.0f;
  }
  // Cosine cutoff in transition region
  float x = (r - r_cut + delta) / delta;
  return 0.5f + 0.5f * std::cos(static_cast<float>(M_PI) * x);
}

namespace {

/**
 * Build neighbor lists for all systems and count totals
 */
std::vector<NeighborList>
build_neighbor_lists(const std::vector<AtomicSystem> &systems,
                     const NeighborListBuilder &neighbor_builder,
                     BatchedInput &batch) {

  ScopedTimer nl_timer(TimerCategory::NeighborList);

  std::vector<NeighborList> nlists;
  nlists.reserve(systems.size());

  batch.total_atoms = 0;
  batch.total_edges = 0;

  for (const auto &system : systems) {
    nlists.push_back(neighbor_builder.build(system));
    batch.total_atoms += system.num_atoms();
    batch.total_edges += nlists.back().num_pairs();
    batch.atoms_per_system.push_back(system.num_atoms());
  }

  // Debug output
  if (getenv("MLIP_DEBUG_NL")) {
    log::debug("total_atoms={} total_edges={}", batch.total_atoms,
               batch.total_edges);
    for (size_t i = 0; i < nlists.size(); ++i) {
      const auto &nlist = nlists[i];
      log::debug("System {}: n_pairs={}", i, nlist.num_pairs());

      // Print first 20 edges sorted
      struct Edge {
        int i, j;
        float d;
        float vec[3];
      };
      std::vector<Edge> edges;
      for (int e = 0; e < nlist.num_pairs(); ++e) {
        edges.push_back({nlist.centers[e],
                         nlist.neighbors[e],
                         nlist.distances[e],
                         {nlist.edge_vectors[e][0], nlist.edge_vectors[e][1],
                          nlist.edge_vectors[e][2]}});
      }
      std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
        if (a.i != b.i)
          return a.i < b.i;
        if (a.j != b.j)
          return a.j < b.j;
        return a.d < b.d;
      });
      log::debug("First 20 edges (sorted):");
      for (int e = 0; e < std::min(20, (int)edges.size()); ++e) {
        log::debug(
            "  {:>2} -> {:>2}: d={:.6f} vec=({:>8.4f}, {:>8.4f}, {:>8.4f})",
            edges[e].i, edges[e].j, edges[e].d, edges[e].vec[0],
            edges[e].vec[1], edges[e].vec[2]);
      }
    }
  }

  // Compute cumulative offsets
  batch.system_atom_offsets.resize(systems.size());
  int offset = 0;
  for (size_t i = 0; i < systems.size(); ++i) {
    batch.system_atom_offsets[i] = offset;
    offset += batch.atoms_per_system[i];
  }

  return nlists;
}

/**
 * Build atom tensors (positions, species, system indices)
 */
AtomData build_atom_tensors(const std::vector<AtomicSystem> &systems,
                            const std::map<int, int> &species_to_index,
                            int total_atoms) {

  AtomData atom_data;
  atom_data.positions_cpu.resize(3 * total_atoms);
  atom_data.species_cpu.resize(total_atoms);
  atom_data.species_indices_cpu.resize(total_atoms);
  atom_data.system_indices_cpu.resize(total_atoms);

  int atom_offset = 0;
  for (size_t sys_idx = 0; sys_idx < systems.size(); ++sys_idx) {
    const auto &system = systems[sys_idx];
    const int n_atoms = system.num_atoms();

    // Copy atom data
    const float *pos_data = system.positions();
    const int32_t *species_data = system.atomic_numbers();

    for (int i = 0; i < n_atoms; ++i) {
      atom_data.positions_cpu[3 * (atom_offset + i) + 0] = pos_data[3 * i + 0];
      atom_data.positions_cpu[3 * (atom_offset + i) + 1] = pos_data[3 * i + 1];
      atom_data.positions_cpu[3 * (atom_offset + i) + 2] = pos_data[3 * i + 2];

      // Store raw atomic number
      int atomic_num = species_data[i];
      atom_data.species_cpu[atom_offset + i] = atomic_num;

      // Convert to species index for embedding lookup
      auto it = species_to_index.find(atomic_num);
      if (it != species_to_index.end()) {
        atom_data.species_indices_cpu[atom_offset + i] = it->second;
      } else {
        throw std::runtime_error("Unknown atomic number " +
                                 std::to_string(atomic_num) +
                                 " not found in species_to_index mapping");
      }

      atom_data.system_indices_cpu[atom_offset + i] =
          static_cast<int32_t>(sys_idx);
    }

    atom_offset += n_atoms;
  }

  return atom_data;
}

/**
 * Build edge tensors (vectors, distances, cutoff factors, indices)
 */
EdgeData build_edge_tensors(const std::vector<AtomicSystem> &systems,
                            const std::vector<NeighborList> &nlists,
                            const BatchedInput &batch, float r_cut,
                            float cutoff_width) {

  EdgeData edge_data;
  edge_data.edge_vectors_cpu.resize(3 * batch.total_edges);
  edge_data.edge_distances_cpu.resize(batch.total_edges);
  edge_data.cutoff_factors_cpu.resize(batch.total_edges);
  edge_data.edge_center_indices_cpu.resize(batch.total_edges);
  edge_data.edge_neighbor_indices_cpu.resize(batch.total_edges);
  edge_data.edge_cell_shifts_cpu.resize(batch.total_edges);

  int atom_offset = 0;
  int edge_offset = 0;

  for (size_t sys_idx = 0; sys_idx < systems.size(); ++sys_idx) {
    const auto &system = systems[sys_idx];
    const auto &nlist = nlists[sys_idx];
    const int n_atoms = system.num_atoms();
    const int n_edges = nlist.num_pairs();

    // Count neighbors per atom for info logging
    std::vector<int> neighbor_count(n_atoms, 0);
    for (int e = 0; e < n_edges; ++e) {
      neighbor_count[nlist.centers[e]]++;
    }

    // Copy edge data with global indexing
    for (int e = 0; e < n_edges; ++e) {
      int global_edge_idx = edge_offset + e;

      edge_data.edge_center_indices_cpu[global_edge_idx] =
          nlist.centers[e] + atom_offset;
      edge_data.edge_neighbor_indices_cpu[global_edge_idx] =
          nlist.neighbors[e] + atom_offset;

      // Store in row-major format for GGML tensor [3, total_edges]
      // Element [dim, edge] at position: dim + edge * 3
      edge_data.edge_vectors_cpu[global_edge_idx * 3 + 0] =
          nlist.edge_vectors[e][0];
      edge_data.edge_vectors_cpu[global_edge_idx * 3 + 1] =
          nlist.edge_vectors[e][1];
      edge_data.edge_vectors_cpu[global_edge_idx * 3 + 2] =
          nlist.edge_vectors[e][2];

      edge_data.edge_distances_cpu[global_edge_idx] = nlist.distances[e];

      // Copy cell shift for periodic systems
      if (!nlist.cell_shifts.empty()) {
        edge_data.edge_cell_shifts_cpu[global_edge_idx] = nlist.cell_shifts[e];
      } else {
        edge_data.edge_cell_shifts_cpu[global_edge_idx] = {0, 0, 0};
      }

      // Compute cutoff factor
      float r = nlist.distances[e];
      edge_data.cutoff_factors_cpu[global_edge_idx] =
          smooth_cutoff(r, r_cut, cutoff_width);
    }

    // Log neighbor statistics
    int min_neighbors =
        *std::min_element(neighbor_count.begin(), neighbor_count.end());
    int max_neighbors =
        *std::max_element(neighbor_count.begin(), neighbor_count.end());
    float avg_neighbors = static_cast<float>(n_edges) / n_atoms;
    log::debug("System {}: {} atoms, {} edges, neighbors/atom: min={}, max={}, "
               "avg={:.1f}",
               sys_idx, n_atoms, n_edges, min_neighbors, max_neighbors,
               avg_neighbors);

    atom_offset += n_atoms;
    edge_offset += n_edges;
  }

  return edge_data;
}

/**
 * Convert edge data to NEF (Node-Edge Format) with padding
 */
void build_nef_tensors(ggml_context *ctx, BatchedInput &batch,
                       const EdgeData &edge_data, const AtomData &atom_data) {

  // Find max_neighbors
  std::vector<int> neighbor_count(batch.total_atoms, 0);
  for (int e = 0; e < batch.total_edges; ++e) {
    int center = edge_data.edge_center_indices_cpu[e];
    neighbor_count[center]++;
  }
  batch.max_neighbors =
      *std::max_element(neighbor_count.begin(), neighbor_count.end());

  if (batch.max_neighbors == 0) {
    batch.max_neighbors = 1;
  }

  // Build NEF indices
  std::vector<int32_t> nef_indices_cpu(batch.max_neighbors * batch.total_atoms,
                                       -1);
  std::vector<float> padding_mask_cpu(batch.max_neighbors * batch.total_atoms,
                                      0.0f);

  // Sort edges by center atom
  std::vector<int> edge_order(batch.total_edges);
  std::iota(edge_order.begin(), edge_order.end(), 0);
  std::stable_sort(edge_order.begin(), edge_order.end(), [&](int a, int b) {
    int center_a = edge_data.edge_center_indices_cpu[a];
    int center_b = edge_data.edge_center_indices_cpu[b];
    return center_a < center_b;
  });

  // Fill NEF indices
  std::vector<int> neighbor_slot(batch.total_atoms, 0);
  for (int e : edge_order) {
    int center = edge_data.edge_center_indices_cpu[e];
    int slot = neighbor_slot[center]++;

    int nef_pos = slot + center * batch.max_neighbors;
    nef_indices_cpu[nef_pos] = e;
    padding_mask_cpu[nef_pos] = 1.0f;
  }

  // Gather edge vectors into NEF format
  std::vector<float> edge_vectors_nef_cpu(
      3 * batch.max_neighbors * batch.total_atoms, 0.0f);
  if (batch.total_edges > 0) {
    for (int atom = 0; atom < batch.total_atoms; ++atom) {
      for (int slot = 0; slot < batch.max_neighbors; ++slot) {
        int nef_idx = slot + atom * batch.max_neighbors;
        int edge_idx = nef_indices_cpu[nef_idx];

        if (edge_idx != -1) {
          for (int dim = 0; dim < 3; ++dim) {
            int target_idx = dim + slot * 3 + atom * (3 * batch.max_neighbors);
            int source_idx = dim + edge_idx * 3;
            edge_vectors_nef_cpu[target_idx] =
                edge_data.edge_vectors_cpu[source_idx];
          }
        }
      }
    }
  }

  batch.edge_vectors_nef = ggml_new_tensor_3d(
      ctx, GGML_TYPE_F32, 3, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.edge_vectors_nef->data, edge_vectors_nef_cpu.data(),
         sizeof(float) * edge_vectors_nef_cpu.size());

  // Gather neighbor species
  std::vector<int32_t> neighbor_species_nef_cpu(
      batch.max_neighbors * batch.total_atoms, 0);
  if (batch.total_edges > 0) {
    for (int atom = 0; atom < batch.total_atoms; ++atom) {
      for (int slot = 0; slot < batch.max_neighbors; ++slot) {
        int nef_pos = slot + atom * batch.max_neighbors;
        int edge_idx_raw = nef_indices_cpu[nef_pos];

        if (edge_idx_raw != -1) {
          int neighbor_atom = edge_data.edge_neighbor_indices_cpu[edge_idx_raw];
          neighbor_species_nef_cpu[nef_pos] =
              atom_data.species_indices_cpu[neighbor_atom];
        }
      }
    }
  }

  batch.neighbor_species_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_I32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.neighbor_species_nef->data, neighbor_species_nef_cpu.data(),
         sizeof(int32_t) * neighbor_species_nef_cpu.size());

  // Create transposed version for GPU compatibility
  // CUDA/HIP don't support I32->I32 copy for non-contiguous tensors, so we
  // pre-compute the transposed layout to avoid ggml_cont(ggml_transpose(...))
  // Original: [max_neighbors, total_atoms] - slot varies fastest
  // Transposed: [total_atoms, max_neighbors] - atom varies fastest
  std::vector<int32_t> neighbor_species_transposed_cpu(
      batch.max_neighbors * batch.total_atoms, 0);
  for (int atom = 0; atom < batch.total_atoms; ++atom) {
    for (int slot = 0; slot < batch.max_neighbors; ++slot) {
      int src_idx = slot + atom * batch.max_neighbors;  // NEF layout
      int dst_idx = atom + slot * batch.total_atoms;    // transposed layout
      neighbor_species_transposed_cpu[dst_idx] = neighbor_species_nef_cpu[src_idx];
    }
  }
  batch.neighbor_species_transposed = ggml_new_tensor_2d(
      ctx, GGML_TYPE_I32, batch.total_atoms, batch.max_neighbors);
  memcpy(batch.neighbor_species_transposed->data,
         neighbor_species_transposed_cpu.data(),
         sizeof(int32_t) * neighbor_species_transposed_cpu.size());

  // Gather edge distances
  std::vector<float> edge_distances_nef_cpu(
      batch.max_neighbors * batch.total_atoms, 0.0f);
  if (batch.total_edges > 0) {
    for (int i = 0; i < batch.max_neighbors * batch.total_atoms; ++i) {
      int edge_idx_raw = nef_indices_cpu[i];
      if (edge_idx_raw != -1) {
        edge_distances_nef_cpu[i] = edge_data.edge_distances_cpu[edge_idx_raw];
      }
    }
  }

  batch.edge_distances_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.edge_distances_nef->data, edge_distances_nef_cpu.data(),
         sizeof(float) * edge_distances_nef_cpu.size());

  log::trace(
      "NEF format atom 0 slot 0: vec=[{:.6f}, {:.6f}, {:.6f}], dist={:.6f}",
      edge_vectors_nef_cpu[0],
      edge_vectors_nef_cpu[batch.max_neighbors * batch.total_atoms],
      edge_vectors_nef_cpu[2 * batch.max_neighbors * batch.total_atoms],
      edge_distances_nef_cpu[0]);

  // Gather cutoff factors
  std::vector<float> cutoff_factors_nef_cpu(
      batch.max_neighbors * batch.total_atoms, 0.0f);
  if (batch.total_edges > 0) {
    for (int i = 0; i < batch.max_neighbors * batch.total_atoms; ++i) {
      int edge_idx_raw = nef_indices_cpu[i];
      if (edge_idx_raw != -1) {
        cutoff_factors_nef_cpu[i] = edge_data.cutoff_factors_cpu[edge_idx_raw];
      }
    }
  }

  batch.cutoff_factors_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.cutoff_factors_nef->data, cutoff_factors_nef_cpu.data(),
         sizeof(float) * cutoff_factors_nef_cpu.size());

  // Padding mask
  batch.padding_mask_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.padding_mask_nef->data, padding_mask_cpu.data(),
         sizeof(float) * padding_mask_cpu.size());

  // Neighbor indices
  std::vector<int32_t> neighbor_indices_nef_cpu(
      batch.max_neighbors * batch.total_atoms, 0);
  if (batch.total_edges > 0) {
    for (int i = 0; i < batch.max_neighbors * batch.total_atoms; ++i) {
      int edge_idx_raw = nef_indices_cpu[i];
      if (edge_idx_raw != -1) {
        neighbor_indices_nef_cpu[i] =
            edge_data.edge_neighbor_indices_cpu[edge_idx_raw];
      }
    }
  }

  batch.neighbor_indices_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_I32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.neighbor_indices_nef->data, neighbor_indices_nef_cpu.data(),
         sizeof(int32_t) * neighbor_indices_nef_cpu.size());
}

/**
 * Build reverse neighbor mapping for message passing
 */
void build_reverse_mapping(ggml_context *ctx, BatchedInput &batch,
                           const EdgeData &edge_data) {

  // Build edge lookup
  using EdgeKey = std::tuple<int, int, int, int, int>;
  struct EdgeKeyHash {
    size_t operator()(const EdgeKey &k) const {
      size_t h = 0;
      h ^= std::hash<int>{}(std::get<0>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(std::get<1>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(std::get<2>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(std::get<3>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>{}(std::get<4>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_lookup;
  edge_lookup.reserve(batch.total_edges);
  for (int e = 0; e < batch.total_edges; ++e) {
    int i = edge_data.edge_center_indices_cpu[e];
    int j = edge_data.edge_neighbor_indices_cpu[e];
    const auto &shift = edge_data.edge_cell_shifts_cpu[e];
    edge_lookup[{i, j, shift[0], shift[1], shift[2]}] = e;
  }

  // Find corresponding edges (reverse direction)
  std::vector<int32_t> corresponding_edges(batch.total_edges, -1);
  for (int e = 0; e < batch.total_edges; ++e) {
    int i = edge_data.edge_center_indices_cpu[e];
    int j = edge_data.edge_neighbor_indices_cpu[e];
    const auto &shift = edge_data.edge_cell_shifts_cpu[e];

    auto it = edge_lookup.find({j, i, -shift[0], -shift[1], -shift[2]});
    if (it != edge_lookup.end()) {
      corresponding_edges[e] = it->second;
    }
  }

  // Get NEF indices from batch tensor
  std::vector<int32_t> nef_indices_cpu(batch.max_neighbors * batch.total_atoms);
  memcpy(nef_indices_cpu.data(), batch.neighbor_indices_nef->data,
         sizeof(int32_t) * nef_indices_cpu.size());

  // Actually we need to get the original NEF indices, not neighbor_indices_nef
  // We need to reconstruct nef_indices_cpu from the batch data
  // Let me read it from edge_vectors_nef by checking which are non-zero
  std::fill(nef_indices_cpu.begin(), nef_indices_cpu.end(), -1);

  // Reconstruct NEF indices from edge data
  std::vector<int> neighbor_slot(batch.total_atoms, 0);
  std::vector<int> edge_order(batch.total_edges);
  std::iota(edge_order.begin(), edge_order.end(), 0);
  std::stable_sort(edge_order.begin(), edge_order.end(), [&](int a, int b) {
    return edge_data.edge_center_indices_cpu[a] <
           edge_data.edge_center_indices_cpu[b];
  });

  for (int e : edge_order) {
    int center = edge_data.edge_center_indices_cpu[e];
    int slot = neighbor_slot[center]++;
    int nef_pos = slot + center * batch.max_neighbors;
    nef_indices_cpu[nef_pos] = e;
  }

  // Convert to NEF positions
  std::vector<int32_t> reversed_neighbor_list_cpu(
      batch.max_neighbors * batch.total_atoms, -1);
  std::vector<float> reverse_edge_mask_cpu(
      batch.max_neighbors * batch.total_atoms, 0.0f);

  for (int i = 0; i < batch.total_atoms; ++i) {
    for (int k = 0; k < batch.max_neighbors; ++k) {
      int nef_pos = k + i * batch.max_neighbors;
      int edge_idx = nef_indices_cpu[nef_pos];

      if (edge_idx == -1)
        continue;

      int j = edge_data.edge_neighbor_indices_cpu[edge_idx];
      int reverse_edge = corresponding_edges[edge_idx];

      if (reverse_edge == -1)
        continue;

      // Find position of reverse edge in atom j's NEF list
      for (int kj = 0; kj < batch.max_neighbors; ++kj) {
        int j_nef_pos = kj + j * batch.max_neighbors;
        if (nef_indices_cpu[j_nef_pos] == reverse_edge) {
          int flat_idx = j * batch.max_neighbors + kj;
          reversed_neighbor_list_cpu[nef_pos] = flat_idx;
          reverse_edge_mask_cpu[nef_pos] = 1.0f;
          break;
        }
      }
    }
  }

  // Validation
  int validation_errors = 0;
  for (int i = 0; i < std::min(2, batch.total_atoms); ++i) {
    for (int k = 0; k < batch.max_neighbors; ++k) {
      int nef_pos = i * batch.max_neighbors + k;
      int edge_idx = nef_indices_cpu[nef_pos];

      if (edge_idx == -1)
        continue;

      int j = edge_data.edge_neighbor_indices_cpu[edge_idx];
      int reverse_flat = reversed_neighbor_list_cpu[nef_pos];

      if (reverse_flat >= 0) {
        int neighbor_atom = reverse_flat / batch.max_neighbors;
        int reverse_slot = reverse_flat % batch.max_neighbors;

        int reverse_nef_pos =
            neighbor_atom * batch.max_neighbors + reverse_slot;
        int reverse_edge_idx = nef_indices_cpu[reverse_nef_pos];

        if (reverse_edge_idx >= 0) {
          int rev_center = edge_data.edge_center_indices_cpu[reverse_edge_idx];
          int rev_neighbor =
              edge_data.edge_neighbor_indices_cpu[reverse_edge_idx];

          bool valid =
              (rev_center == j && rev_neighbor == i && neighbor_atom == j);

          if (!valid) {
            log::error("Reverse mapping error: [{}:{}] edge {} ({}->{}) "
                       "flat={} (atom={},slot={}) -> edge {} ({}->{})",
                       i, k, edge_idx, i, j, reverse_flat, neighbor_atom,
                       reverse_slot, reverse_edge_idx, rev_center,
                       rev_neighbor);
            validation_errors++;
          }
        } else {
          log::error("Reverse mapping error: [{}:{}] edge {} ({}->{}) flat={} "
                     "-> PADDING",
                     i, k, edge_idx, i, j, reverse_flat);
          validation_errors++;
        }
      }
    }
  }

  if (validation_errors > 0) {
    log::error("Reverse mapping validation found {} errors", validation_errors);
  } else {
    log::debug("Reverse mapping validation passed");
  }

  // Replace -1 with 0
  for (auto &idx : reversed_neighbor_list_cpu) {
    if (idx == -1)
      idx = 0;
  }

  batch.reversed_neighbor_list = ggml_new_tensor_2d(
      ctx, GGML_TYPE_I32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.reversed_neighbor_list->data, reversed_neighbor_list_cpu.data(),
         sizeof(int32_t) * reversed_neighbor_list_cpu.size());

  batch.reverse_edge_mask_nef = ggml_new_tensor_2d(
      ctx, GGML_TYPE_F32, batch.max_neighbors, batch.total_atoms);
  memcpy(batch.reverse_edge_mask_nef->data, reverse_edge_mask_cpu.data(),
         sizeof(float) * reverse_edge_mask_cpu.size());
}

/**
 * Build attention masks for transformer layers
 */
void build_attention_masks(ggml_context *ctx, BatchedInput &batch,
                           const AtomData &atom_data) {

  ScopedTimer mask_timer(TimerCategory::AttentionMask);

  int seq_len = 1 + batch.max_neighbors;
  std::vector<float> mask_data(seq_len * seq_len * batch.total_atoms, 0.0f);

  // Get padding mask and cutoff factors from batch
  std::vector<float> padding_mask_cpu(batch.max_neighbors * batch.total_atoms);
  memcpy(padding_mask_cpu.data(), batch.padding_mask_nef->data,
         sizeof(float) * padding_mask_cpu.size());

  std::vector<int32_t> neighbor_indices_cpu(batch.max_neighbors *
                                            batch.total_atoms);
  memcpy(neighbor_indices_cpu.data(), batch.neighbor_indices_nef->data,
         sizeof(int32_t) * neighbor_indices_cpu.size());

  std::vector<float> cutoff_factors_cpu(batch.max_neighbors *
                                        batch.total_atoms);
  memcpy(cutoff_factors_cpu.data(), batch.cutoff_factors_nef->data,
         sizeof(float) * cutoff_factors_cpu.size());

  for (int atom_idx = 0; atom_idx < batch.total_atoms; ++atom_idx) {
    int atom_system_idx = atom_data.system_indices_cpu[atom_idx];

    // Build cutoff pattern
    std::vector<float> cutoff_values(seq_len);
    cutoff_values[0] = 1.0f;

    for (int key_pos = 1; key_pos < seq_len; ++key_pos) {
      int edge_slot = key_pos - 1;
      int nef_idx = edge_slot + atom_idx * batch.max_neighbors;

      if (padding_mask_cpu[nef_idx] > 0.5f) {
        int neighbor_atom = neighbor_indices_cpu[nef_idx];
        int neighbor_system = atom_data.system_indices_cpu[neighbor_atom];

        if (neighbor_system == atom_system_idx) {
          cutoff_values[key_pos] = cutoff_factors_cpu[nef_idx];
        } else {
          cutoff_values[key_pos] = 0.0f;
        }
      } else {
        cutoff_values[key_pos] = 0.0f;
      }
    }

    // Take log of clamped values
    std::vector<float> cutoff_pattern(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      cutoff_pattern[i] = std::log(std::max(cutoff_values[i], 1e-15f));
    }

    // Broadcast to all query positions
    for (int query_pos = 0; query_pos < seq_len; ++query_pos) {
      for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
        int mask_idx =
            atom_idx * (seq_len * seq_len) + query_pos * seq_len + key_pos;
        mask_data[mask_idx] = cutoff_pattern[key_pos];
      }
    }
  }

  // Create identical masks for both GNN layers
  batch.attn_mask_layer0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len,
                                              seq_len, batch.total_atoms);
  memcpy(batch.attn_mask_layer0->data, mask_data.data(),
         sizeof(float) * mask_data.size());

  batch.attn_mask_layer1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len,
                                              seq_len, batch.total_atoms);
  memcpy(batch.attn_mask_layer1->data, mask_data.data(),
         sizeof(float) * mask_data.size());
}

} // anonymous namespace

BatchedInput prepare_batch(ggml_context *ctx,
                           const std::vector<AtomicSystem> &systems,
                           const NeighborListBuilder &neighbor_builder,
                           float cutoff, float cutoff_width,
                           const std::map<int, int> &species_to_index) {

  BatchedInput batch;
  batch.n_systems = static_cast<int>(systems.size());

  if (systems.empty()) {
    return batch; // Empty batch
  }

  float r_cut = cutoff;

  // Phase 1: Build neighbor lists
  std::vector<NeighborList> nlists =
      build_neighbor_lists(systems, neighbor_builder, batch);

  // Phase 2: Build atom tensors
  AtomData atom_data =
      build_atom_tensors(systems, species_to_index, batch.total_atoms);

  // Phase 3: Build edge tensors
  EdgeData edge_data =
      build_edge_tensors(systems, nlists, batch, r_cut, cutoff_width);

  // Phase 4: Create GGML tensors for atom data
  batch.positions =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, batch.total_atoms);
  memcpy(batch.positions->data, atom_data.positions_cpu.data(),
         sizeof(float) * atom_data.positions_cpu.size());

  batch.species = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.total_atoms);
  memcpy(batch.species->data, atom_data.species_indices_cpu.data(),
         sizeof(int32_t) * atom_data.species_indices_cpu.size());

  batch.system_indices =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.total_atoms);
  memcpy(batch.system_indices->data, atom_data.system_indices_cpu.data(),
         sizeof(int32_t) * atom_data.system_indices_cpu.size());

  batch.edge_vectors =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, batch.total_edges);
  memcpy(batch.edge_vectors->data, edge_data.edge_vectors_cpu.data(),
         sizeof(float) * edge_data.edge_vectors_cpu.size());

  log::trace("edge_vectors_cpu for edge 0: total_edges={}, x={:.6f}, y={:.6f}, "
             "z={:.6f}",
             batch.total_edges, edge_data.edge_vectors_cpu[0],
             edge_data.edge_vectors_cpu.size() >
                     static_cast<size_t>(batch.total_edges)
                 ? edge_data.edge_vectors_cpu[batch.total_edges]
                 : 0.0f,
             edge_data.edge_vectors_cpu.size() >
                     static_cast<size_t>(2 * batch.total_edges)
                 ? edge_data.edge_vectors_cpu[2 * batch.total_edges]
                 : 0.0f);

  batch.edge_distances =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch.total_edges);
  memcpy(batch.edge_distances->data, edge_data.edge_distances_cpu.data(),
         sizeof(float) * edge_data.edge_distances_cpu.size());

  batch.cutoff_factors =
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch.total_edges);
  memcpy(batch.cutoff_factors->data, edge_data.cutoff_factors_cpu.data(),
         sizeof(float) * edge_data.cutoff_factors_cpu.size());

  batch.edge_center_indices =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.total_edges);
  memcpy(batch.edge_center_indices->data,
         edge_data.edge_center_indices_cpu.data(),
         sizeof(int32_t) * edge_data.edge_center_indices_cpu.size());

  batch.edge_neighbor_indices =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.total_edges);
  memcpy(batch.edge_neighbor_indices->data,
         edge_data.edge_neighbor_indices_cpu.data(),
         sizeof(int32_t) * edge_data.edge_neighbor_indices_cpu.size());

  // Phase 5: Convert to NEF format
  build_nef_tensors(ctx, batch, edge_data, atom_data);

  // Phase 6: Build reverse neighbor mapping
  build_reverse_mapping(ctx, batch, edge_data);

  // Phase 7: Build attention masks
  build_attention_masks(ctx, batch, atom_data);

  return batch;
}

} // namespace mlipcpp::pet
