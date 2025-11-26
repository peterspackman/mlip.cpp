/**
 * Unit tests for AtomicSystem and Cell classes
 */
#include "mlipcpp/system.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace mlipcpp;

TEST_CASE("Cell default constructor creates identity", "[cell]") {
  Cell cell;

  // Should be identity matrix
  REQUIRE_THAT(cell.matrix[0][0], Catch::Matchers::WithinAbs(1.0f, 1e-6f));
  REQUIRE_THAT(cell.matrix[1][1], Catch::Matchers::WithinAbs(1.0f, 1e-6f));
  REQUIRE_THAT(cell.matrix[2][2], Catch::Matchers::WithinAbs(1.0f, 1e-6f));

  REQUIRE_THAT(cell.matrix[0][1], Catch::Matchers::WithinAbs(0.0f, 1e-6f));
  REQUIRE_THAT(cell.matrix[0][2], Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Cell constructor sets lattice vectors", "[cell]") {
  float lattice[3][3] = {
      {5.0f, 0.0f, 0.0f}, {0.0f, 6.0f, 0.0f}, {0.0f, 0.0f, 7.0f}};

  Cell cell(lattice, true, true, false);

  REQUIRE_THAT(cell.matrix[0][0], Catch::Matchers::WithinAbs(5.0f, 1e-6f));
  REQUIRE_THAT(cell.matrix[1][1], Catch::Matchers::WithinAbs(6.0f, 1e-6f));
  REQUIRE_THAT(cell.matrix[2][2], Catch::Matchers::WithinAbs(7.0f, 1e-6f));

  REQUIRE(cell.periodic[0] == true);
  REQUIRE(cell.periodic[1] == true);
  REQUIRE(cell.periodic[2] == false);
}

TEST_CASE("Cell coordinate conversions", "[cell]") {
  // Orthorhombic cell: 4 x 5 x 6
  float lattice[3][3] = {
      {4.0f, 0.0f, 0.0f}, {0.0f, 5.0f, 0.0f}, {0.0f, 0.0f, 6.0f}};
  Cell cell(lattice);

  SECTION("fractional to Cartesian") {
    float frac[3] = {0.5f, 0.5f, 0.5f};
    float cart[3];
    cell.to_cartesian(frac, cart);

    REQUIRE_THAT(cart[0], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    REQUIRE_THAT(cart[1], Catch::Matchers::WithinAbs(2.5f, 1e-5f));
    REQUIRE_THAT(cart[2], Catch::Matchers::WithinAbs(3.0f, 1e-5f));
  }

  SECTION("Cartesian to fractional") {
    float cart[3] = {2.0f, 2.5f, 3.0f};
    float frac[3];
    cell.to_fractional(cart, frac);

    REQUIRE_THAT(frac[0], Catch::Matchers::WithinAbs(0.5f, 1e-5f));
    REQUIRE_THAT(frac[1], Catch::Matchers::WithinAbs(0.5f, 1e-5f));
    REQUIRE_THAT(frac[2], Catch::Matchers::WithinAbs(0.5f, 1e-5f));
  }

  SECTION("round-trip conversion") {
    float orig[3] = {1.23f, 4.56f, 2.89f};
    float frac[3], cart[3];

    cell.to_fractional(orig, frac);
    cell.to_cartesian(frac, cart);

    REQUIRE_THAT(cart[0], Catch::Matchers::WithinAbs(orig[0], 1e-5f));
    REQUIRE_THAT(cart[1], Catch::Matchers::WithinAbs(orig[1], 1e-5f));
    REQUIRE_THAT(cart[2], Catch::Matchers::WithinAbs(orig[2], 1e-5f));
  }
}

TEST_CASE("Cell minimum image convention", "[cell]") {
  // Cubic cell: 10 x 10 x 10
  float lattice[3][3] = {
      {10.0f, 0.0f, 0.0f}, {0.0f, 10.0f, 0.0f}, {0.0f, 0.0f, 10.0f}};
  Cell cell(lattice, true, true, true);

  SECTION("vector within cell unchanged") {
    float dr[3] = {3.0f, 4.0f, 2.0f};
    cell.minimum_image(dr);

    REQUIRE_THAT(dr[0], Catch::Matchers::WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(dr[1], Catch::Matchers::WithinAbs(4.0f, 1e-5f));
    REQUIRE_THAT(dr[2], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
  }

  SECTION("vector wrapped to nearest image") {
    float dr[3] = {8.0f, 0.0f, 0.0f}; // > L/2, should wrap to -2
    cell.minimum_image(dr);

    REQUIRE_THAT(dr[0], Catch::Matchers::WithinAbs(-2.0f, 1e-5f));
  }

  SECTION("negative vector wrapped") {
    float dr[3] = {-8.0f, 0.0f, 0.0f}; // < -L/2, should wrap to +2
    cell.minimum_image(dr);

    REQUIRE_THAT(dr[0], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
  }

  SECTION("all components wrapped") {
    float dr[3] = {7.0f, -8.0f, 9.0f};
    cell.minimum_image(dr);

    REQUIRE_THAT(dr[0], Catch::Matchers::WithinAbs(-3.0f, 1e-5f));
    REQUIRE_THAT(dr[1], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    REQUIRE_THAT(dr[2], Catch::Matchers::WithinAbs(-1.0f, 1e-5f));
  }
}

TEST_CASE("AtomicSystem construction", "[system]") {
  std::vector<int32_t> Z = {6, 7, 8}; // C, N, O
  std::vector<float> pos = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f};

  SECTION("non-periodic system") {
    AtomicSystem system(3, pos.data(), Z.data(), nullptr);

    REQUIRE(system.num_atoms() == 3);
    REQUIRE_FALSE(system.is_periodic());
    REQUIRE(system.cell() == nullptr);

    REQUIRE(system.atomic_numbers()[0] == 6);
    REQUIRE(system.atomic_numbers()[1] == 7);
    REQUIRE(system.atomic_numbers()[2] == 8);
  }

  SECTION("periodic system") {
    float lattice[3][3] = {
        {10.0f, 0.0f, 0.0f}, {0.0f, 10.0f, 0.0f}, {0.0f, 0.0f, 10.0f}};
    Cell cell(lattice);

    AtomicSystem system(3, pos.data(), Z.data(), &cell);

    REQUIRE(system.num_atoms() == 3);
    REQUIRE(system.is_periodic());
    REQUIRE(system.cell() != nullptr);
  }
}

TEST_CASE("AtomicSystem default constructor", "[system]") {
  AtomicSystem system;

  REQUIRE(system.num_atoms() == 0);
  REQUIRE_FALSE(system.is_periodic());
}

TEST_CASE("AtomicSystem mutable access", "[system]") {
  AtomicSystem system;

  // Build system using mutable access
  system.positions_mut() = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  system.atomic_numbers_mut() = {14, 14};

  float lattice[3][3] = {
      {5.0f, 0.0f, 0.0f}, {0.0f, 5.0f, 0.0f}, {0.0f, 0.0f, 5.0f}};
  Cell cell(lattice);
  system.set_cell(cell);

  // Note: n_atoms_ won't be updated by just modifying vectors
  // This tests the mutable interface, not recommended usage
  REQUIRE(system.is_periodic());
  REQUIRE(system.positions_mut().size() == 6);
}
