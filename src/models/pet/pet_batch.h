#pragma once

#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include "pet_types.h"

namespace mlipcpp::pet {

// BatchedInput is now defined in pet_types.h
// This header provides the legacy prepare_batch() function for compatibility

/**
 * Computes smooth cutoff function for edge weighting
 *
 * Uses cosine cutoff:
 *   if r >= r_cut: return 0.0
 *   if r <= r_cut - delta: return 1.0
 *   else: x = (r - r_cut + delta) / delta
 *         return 0.5 + 0.5 * cos(pi * x)
 *
 * @param r Distance
 * @param r_cut Cutoff radius
 * @param delta Smoothing width
 * @return Cutoff factor in [0, 1]
 */
float smooth_cutoff(float r, float r_cut, float delta);

/**
 * Prepare batched input from multiple atomic systems
 *
 * This function:
 * 1. Builds neighbor lists for each system
 * 2. Concatenates all systems into global tensors
 * 3. Converts edge data to NEF format with padding
 * 4. Builds reverse neighbor mapping for message passing
 * 5. Computes cutoff factors and masks
 *
 * All tensors are allocated in the provided GGML context.
 *
 * @param ctx GGML context for tensor allocation
 * @param systems Vector of atomic systems to batch
 * @param neighbor_builder Neighbor list builder (configured with cutoff)
 * @param cutoff Cutoff radius for computing smooth cutoff factors
 * @param cutoff_width Smoothing width for cutoff function
 * @param species_to_index Mapping from atomic numbers to species indices
 * @return BatchedInput structure with all tensors populated
 */
BatchedInput prepare_batch(ggml_context *ctx,
                           const std::vector<AtomicSystem> &systems,
                           const NeighborListBuilder &neighbor_builder,
                           float cutoff, float cutoff_width,
                           const std::map<int, int> &species_to_index);

/**
 * Prepare batched input from a single system (convenience function)
 *
 * @param ctx GGML context
 * @param system Single atomic system
 * @param neighbor_builder Neighbor list builder
 * @param cutoff Cutoff radius
 * @param cutoff_width Smoothing width for cutoff function
 * @param species_to_index Mapping from atomic numbers to species indices
 * @return BatchedInput structure (batch size = 1)
 */
inline BatchedInput prepare_single(ggml_context *ctx,
                                   const AtomicSystem &system,
                                   const NeighborListBuilder &neighbor_builder,
                                   float cutoff, float cutoff_width,
                                   const std::map<int, int> &species_to_index) {
  return prepare_batch(ctx, {system}, neighbor_builder, cutoff, cutoff_width,
                       species_to_index);
}

} // namespace mlipcpp::pet
