#pragma once

#include "system.h"
#include <array>
#include <vector>

namespace mlipcpp {

struct NeighborList {
  int n_atoms = 0;

  std::vector<int32_t> centers;
  std::vector<int32_t> neighbors;
  std::vector<std::array<int32_t, 3>> cell_shifts;
  std::vector<std::array<float, 3>> edge_vectors;
  std::vector<float> distances;

  int num_pairs() const { return static_cast<int>(centers.size()); }
};

struct NeighborListOptions {
  float cutoff = 6.0f;
  bool full_list = true;
  bool self_interaction = false;
};

class NeighborListBuilder {
public:
  explicit NeighborListBuilder(const NeighborListOptions &options = {});

  NeighborList build(const AtomicSystem &system) const;

private:
  NeighborListOptions options_;

  NeighborList build_non_periodic(const AtomicSystem &system) const;
  NeighborList build_periodic(const AtomicSystem &system) const;
};

} // namespace mlipcpp
