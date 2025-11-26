#include "mlipcpp/system.h"
#include <cmath>
#include <cstring>

namespace mlipcpp {

Cell::Cell() : periodic{false, false, false} {
  std::memset(matrix, 0, sizeof(matrix));
  matrix[0][0] = matrix[1][1] = matrix[2][2] = 1.0f;
  compute_inverse();
}

Cell::Cell(const float lattice[3][3], bool pbc_x, bool pbc_y, bool pbc_z)
    : periodic{pbc_x, pbc_y, pbc_z} {
  std::memcpy(matrix, lattice, sizeof(matrix));
  compute_inverse();
}

void Cell::compute_inverse() {
  // Compute inverse matrix using cofactor method
  float det = matrix[0][0] *
                  (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
              matrix[0][1] *
                  (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
              matrix[0][2] *
                  (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

  float invdet = 1.0f / det;

  inverse[0][0] =
      (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) * invdet;
  inverse[0][1] =
      (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * invdet;
  inverse[0][2] =
      (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * invdet;
  inverse[1][0] =
      (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * invdet;
  inverse[1][1] =
      (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * invdet;
  inverse[1][2] =
      (matrix[1][0] * matrix[0][2] - matrix[0][0] * matrix[1][2]) * invdet;
  inverse[2][0] =
      (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]) * invdet;
  inverse[2][1] =
      (matrix[2][0] * matrix[0][1] - matrix[0][0] * matrix[2][1]) * invdet;
  inverse[2][2] =
      (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]) * invdet;
}

void Cell::to_cartesian(const float frac[3], float cart[3]) const {
  cart[0] =
      matrix[0][0] * frac[0] + matrix[1][0] * frac[1] + matrix[2][0] * frac[2];
  cart[1] =
      matrix[0][1] * frac[0] + matrix[1][1] * frac[1] + matrix[2][1] * frac[2];
  cart[2] =
      matrix[0][2] * frac[0] + matrix[1][2] * frac[1] + matrix[2][2] * frac[2];
}

void Cell::to_fractional(const float cart[3], float frac[3]) const {
  frac[0] = inverse[0][0] * cart[0] + inverse[1][0] * cart[1] +
            inverse[2][0] * cart[2];
  frac[1] = inverse[0][1] * cart[0] + inverse[1][1] * cart[1] +
            inverse[2][1] * cart[2];
  frac[2] = inverse[0][2] * cart[0] + inverse[1][2] * cart[1] +
            inverse[2][2] * cart[2];
}

void Cell::minimum_image(float dr[3]) const {
  float frac[3];
  to_fractional(dr, frac);

  // Apply periodic boundary conditions
  for (int i = 0; i < 3; ++i) {
    if (periodic[i]) {
      frac[i] -= std::round(frac[i]);
    }
  }

  to_cartesian(frac, dr);
}

AtomicSystem::AtomicSystem(int n_atoms, const float *positions,
                           const int32_t *atomic_numbers, const Cell *cell)
    : n_atoms_(n_atoms), positions_(positions, positions + n_atoms * 3),
      atomic_numbers_(atomic_numbers, atomic_numbers + n_atoms) {
  if (cell) {
    cell_ = *cell;
  }
}

} // namespace mlipcpp
