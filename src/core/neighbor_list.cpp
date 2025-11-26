#include "mlipcpp/neighbor_list.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace mlipcpp {

namespace {

struct Vec3 {
  double x, y, z;

  Vec3() : x(0), y(0), z(0) {}
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  double dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
  double norm_sq() const { return dot(*this); }
  double norm() const { return std::sqrt(norm_sq()); }

  Vec3 cross(const Vec3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
};

struct CellGrid {
  std::array<Vec3, 3> cell_vecs;
  std::array<Vec3, 3> bin_vecs;
  std::array<double, 9> inv_cell;
  std::array<bool, 3> pbc;
  std::array<int, 3> n_bins;
  std::array<int, 3> n_search;

  std::vector<int> head;
  std::vector<int> next;

  CellGrid(const Cell &cell, double cutoff, int n_atoms) {
    for (int i = 0; i < 3; ++i) {
      cell_vecs[i] = {cell.matrix[i][0], cell.matrix[i][1], cell.matrix[i][2]};
      pbc[i] = cell.periodic[i];
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        inv_cell[i * 3 + j] = cell.inverse[i][j];
      }
    }

    Vec3 normals[3] = {cell_vecs[1].cross(cell_vecs[2]),
                       cell_vecs[2].cross(cell_vecs[0]),
                       cell_vecs[0].cross(cell_vecs[1])};

    double volume = std::abs(cell_vecs[2].dot(normals[2]));

    for (int i = 0; i < 3; ++i) {
      double face_dist = volume / normals[i].norm();
      n_bins[i] = std::max(1, static_cast<int>(std::floor(face_dist / cutoff)));
      n_search[i] = static_cast<int>(std::ceil(cutoff * n_bins[i] / face_dist));
      bin_vecs[i] = cell_vecs[i] * (1.0 / n_bins[i]);
    }

    int total_bins = n_bins[0] * n_bins[1] * n_bins[2];
    head.assign(total_bins, -1);
    next.resize(n_atoms);
  }

  std::array<int, 3> position_to_bin(const Vec3 &pos) const {
    double frac[3];
    for (int i = 0; i < 3; ++i) {
      frac[i] = inv_cell[i * 3 + 0] * pos.x + inv_cell[i * 3 + 1] * pos.y +
                inv_cell[i * 3 + 2] * pos.z;
    }
    return {static_cast<int>(std::floor(frac[0] * n_bins[0])),
            static_cast<int>(std::floor(frac[1] * n_bins[1])),
            static_cast<int>(std::floor(frac[2] * n_bins[2]))};
  }

  int wrap_bin(int idx, int n, bool periodic) const {
    if (periodic) {
      idx = idx % n;
      if (idx < 0)
        idx += n;
    } else {
      idx = std::clamp(idx, 0, n - 1);
    }
    return idx;
  }

  int bin_index(int i, int j, int k) const {
    return i + n_bins[0] * (j + n_bins[1] * k);
  }

  void insert_atom(int atom_idx, const Vec3 &pos) {
    auto bin = position_to_bin(pos);
    for (int d = 0; d < 3; ++d) {
      bin[d] = wrap_bin(bin[d], n_bins[d], pbc[d]);
    }
    int idx = bin_index(bin[0], bin[1], bin[2]);
    next[atom_idx] = head[idx];
    head[idx] = atom_idx;
  }

  Vec3 relative_position(const Vec3 &pos, const std::array<int, 3> &bin) const {
    return pos - bin_vecs[0] * bin[0] - bin_vecs[1] * bin[1] -
           bin_vecs[2] * bin[2];
  }
};

} // namespace

NeighborListBuilder::NeighborListBuilder(const NeighborListOptions &options)
    : options_(options) {}

NeighborList NeighborListBuilder::build(const AtomicSystem &system) const {
  if (system.is_periodic()) {
    return build_periodic(system);
  } else {
    return build_non_periodic(system);
  }
}

NeighborList
NeighborListBuilder::build_non_periodic(const AtomicSystem &system) const {
  NeighborList nlist;
  const int n_atoms = system.num_atoms();
  const float *pos = system.positions();
  const float cutoff_sq = options_.cutoff * options_.cutoff;

  nlist.n_atoms = n_atoms;

  for (int i = 0; i < n_atoms; ++i) {
    Vec3 pos_i(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);

    for (int j = 0; j < n_atoms; ++j) {
      if (!options_.self_interaction && i == j)
        continue;
      if (!options_.full_list && j <= i)
        continue;

      Vec3 pos_j(pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]);
      Vec3 dr = pos_j - pos_i;
      double dist_sq = dr.norm_sq();

      if (dist_sq < cutoff_sq && dist_sq > 1e-16) {
        nlist.centers.push_back(i);
        nlist.neighbors.push_back(j);
        nlist.cell_shifts.push_back({0, 0, 0});
        nlist.edge_vectors.push_back({static_cast<float>(dr.x),
                                      static_cast<float>(dr.y),
                                      static_cast<float>(dr.z)});
        nlist.distances.push_back(static_cast<float>(std::sqrt(dist_sq)));
      }
    }
  }

  return nlist;
}

NeighborList
NeighborListBuilder::build_periodic(const AtomicSystem &system) const {
  NeighborList nlist;
  const int n_atoms = system.num_atoms();
  const float *pos_f = system.positions();
  const Cell *cell = system.cell();
  const double cutoff = options_.cutoff;
  const double cutoff_sq = cutoff * cutoff;

  nlist.n_atoms = n_atoms;

  if (n_atoms == 0)
    return nlist;

  std::vector<Vec3> positions(n_atoms);
  for (int i = 0; i < n_atoms; ++i) {
    positions[i] = {pos_f[i * 3], pos_f[i * 3 + 1], pos_f[i * 3 + 2]};
  }

  CellGrid grid(*cell, cutoff, n_atoms);

  for (int i = 0; i < n_atoms; ++i) {
    grid.insert_atom(i, positions[i]);
  }

  for (int i = 0; i < n_atoms; ++i) {
    auto bin_i_raw = grid.position_to_bin(positions[i]);
    std::array<int, 3> bin_i;
    for (int d = 0; d < 3; ++d) {
      bin_i[d] = grid.pbc[d]
                     ? bin_i_raw[d]
                     : grid.wrap_bin(bin_i_raw[d], grid.n_bins[d], false);
    }

    Vec3 rel_i = grid.relative_position(positions[i], bin_i);

    std::array<int, 3> bin_i_wrapped;
    for (int d = 0; d < 3; ++d) {
      bin_i_wrapped[d] =
          grid.wrap_bin(bin_i_raw[d], grid.n_bins[d], grid.pbc[d]);
    }

    for (int dz = -grid.n_search[2]; dz <= grid.n_search[2]; ++dz) {
      int bz = bin_i_wrapped[2] + dz;
      if (grid.pbc[2]) {
        bz = grid.wrap_bin(bz, grid.n_bins[2], true);
      } else if (bz < 0 || bz >= grid.n_bins[2]) {
        continue;
      }

      Vec3 off_z = grid.bin_vecs[2] * dz;

      for (int dy = -grid.n_search[1]; dy <= grid.n_search[1]; ++dy) {
        int by = bin_i_wrapped[1] + dy;
        if (grid.pbc[1]) {
          by = grid.wrap_bin(by, grid.n_bins[1], true);
        } else if (by < 0 || by >= grid.n_bins[1]) {
          continue;
        }

        Vec3 off_yz = off_z + grid.bin_vecs[1] * dy;

        for (int dx = -grid.n_search[0]; dx <= grid.n_search[0]; ++dx) {
          int bx = bin_i_wrapped[0] + dx;
          if (grid.pbc[0]) {
            bx = grid.wrap_bin(bx, grid.n_bins[0], true);
          } else if (bx < 0 || bx >= grid.n_bins[0]) {
            continue;
          }

          Vec3 offset = off_yz + grid.bin_vecs[0] * dx;

          int j = grid.head[grid.bin_index(bx, by, bz)];
          while (j >= 0) {
            bool same_image = (dx == 0 && dy == 0 && dz == 0);
            if (i != j || !same_image) {
              auto bin_j_raw = grid.position_to_bin(positions[j]);
              std::array<int, 3> bin_j;
              for (int d = 0; d < 3; ++d) {
                bin_j[d] = grid.pbc[d] ? bin_j_raw[d]
                                       : grid.wrap_bin(bin_j_raw[d],
                                                       grid.n_bins[d], false);
              }

              Vec3 rel_j = grid.relative_position(positions[j], bin_j);
              Vec3 dr = rel_j - rel_i + offset;
              double dist_sq = dr.norm_sq();

              if (dist_sq < cutoff_sq) {
                bool should_add = true;

                if (i == j && same_image) {
                  should_add = options_.self_interaction;
                }

                if (!options_.full_list && should_add) {
                  if (j < i) {
                    should_add = false;
                  } else if (j == i) {
                    if (dz < 0 || (dz == 0 && dy < 0) ||
                        (dz == 0 && dy == 0 && dx < 0)) {
                      should_add = false;
                    }
                  }
                }

                if (should_add) {
                  nlist.centers.push_back(i);
                  nlist.neighbors.push_back(j);

                  std::array<int, 3> shift;
                  for (int d = 0; d < 3; ++d) {
                    int delta = (d == 0 ? dx : (d == 1 ? dy : dz));
                    shift[d] =
                        (bin_i_raw[d] - bin_j_raw[d] + delta) / grid.n_bins[d];
                  }

                  nlist.cell_shifts.push_back(shift);
                  nlist.edge_vectors.push_back({static_cast<float>(dr.x),
                                                static_cast<float>(dr.y),
                                                static_cast<float>(dr.z)});
                  nlist.distances.push_back(
                      static_cast<float>(std::sqrt(dist_sq)));
                }
              }
            }
            j = grid.next[j];
          }
        }
      }
    }
  }

  return nlist;
}

} // namespace mlipcpp
