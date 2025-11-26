#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace mlipcpp {

/// Represents a periodic cell with lattice vectors
struct Cell {
  float matrix[3][3];  // Row-major: matrix[i] is the i-th lattice vector
  float inverse[3][3]; // Cached inverse matrix
  bool periodic[3];    // Periodicity in each direction

  Cell();
  explicit Cell(const float lattice[3][3], bool pbc_x = true, bool pbc_y = true,
                bool pbc_z = true);

  /// Convert fractional coordinates to Cartesian
  void to_cartesian(const float frac[3], float cart[3]) const;

  /// Convert Cartesian coordinates to fractional
  void to_fractional(const float cart[3], float frac[3]) const;

  /// Apply minimum image convention for a displacement vector
  void minimum_image(float dr[3]) const;

private:
  void compute_inverse();
};

/// Atomic system representation
class AtomicSystem {
public:
  AtomicSystem() = default;
  AtomicSystem(int n_atoms, const float *positions,
               const int32_t *atomic_numbers, const Cell *cell = nullptr);

  int num_atoms() const { return n_atoms_; }
  const float *positions() const { return positions_.data(); }
  const int32_t *atomic_numbers() const { return atomic_numbers_.data(); }
  const Cell *cell() const { return cell_ ? &cell_.value() : nullptr; }

  bool is_periodic() const { return cell_.has_value(); }

  // Mutable access for construction
  std::vector<float> &positions_mut() { return positions_; }
  std::vector<int32_t> &atomic_numbers_mut() { return atomic_numbers_; }
  void set_cell(const Cell &cell) { cell_ = cell; }

private:
  int n_atoms_ = 0;
  std::vector<float> positions_; // [n_atoms * 3], flattened (x,y,z) per atom
  std::vector<int32_t> atomic_numbers_;
  std::optional<Cell> cell_;
};

} // namespace mlipcpp
