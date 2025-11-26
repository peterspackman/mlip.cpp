/**
 * Unit tests for XYZ file I/O
 */
#include "mlipcpp/io.h"
#include "mlipcpp/system.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <sstream>

using namespace mlipcpp;
using namespace mlipcpp::io;

// Water molecule - basic XYZ format
static const char *WATER_XYZ = R"(3
Water molecule
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
)";

// Si crystal - extended XYZ with lattice
static const char *SI_EXYZ = R"(2
Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" Properties=species:S:1:pos:R:3 pbc="T T T"
Si 0.000000 0.000000 0.000000
Si 1.357500 1.357500 1.357500
)";

// Single atom
static const char *SINGLE_ATOM_XYZ = R"(1
Single carbon atom
C 1.5 2.5 3.5
)";

// Non-cubic cell
static const char *TRICLINIC_EXYZ = R"(1
Lattice="4.0 0.0 0.0 1.0 5.0 0.0 0.5 0.5 6.0" pbc="T T T"
Fe 0.5 0.5 0.5
)";

TEST_CASE("read_xyz parses basic XYZ format", "[io]") {
  std::istringstream ss(WATER_XYZ);
  auto system = read_xyz(ss);

  REQUIRE(system.num_atoms() == 3);
  REQUIRE_FALSE(system.is_periodic());

  // Check atomic numbers (O=8, H=1)
  const auto *Z = system.atomic_numbers();
  REQUIRE(Z[0] == 8);
  REQUIRE(Z[1] == 1);
  REQUIRE(Z[2] == 1);

  // Check positions
  const auto *pos = system.positions();
  // Oxygen at origin
  REQUIRE_THAT(pos[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(pos[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(pos[2], Catch::Matchers::WithinAbs(0.0, 1e-6));
  // First hydrogen
  REQUIRE_THAT(pos[3], Catch::Matchers::WithinAbs(0.757, 1e-6));
  REQUIRE_THAT(pos[4], Catch::Matchers::WithinAbs(0.586, 1e-6));
  REQUIRE_THAT(pos[5], Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("read_xyz parses extended XYZ with lattice", "[io]") {
  std::istringstream ss(SI_EXYZ);
  auto system = read_xyz(ss);

  REQUIRE(system.num_atoms() == 2);
  REQUIRE(system.is_periodic());

  // Check atomic numbers (Si=14)
  const auto *Z = system.atomic_numbers();
  REQUIRE(Z[0] == 14);
  REQUIRE(Z[1] == 14);

  // Check cell
  const Cell *cell = system.cell();
  REQUIRE(cell != nullptr);

  // Cubic cell with a=5.43
  REQUIRE_THAT(cell->matrix[0][0], Catch::Matchers::WithinAbs(5.43, 1e-6));
  REQUIRE_THAT(cell->matrix[1][1], Catch::Matchers::WithinAbs(5.43, 1e-6));
  REQUIRE_THAT(cell->matrix[2][2], Catch::Matchers::WithinAbs(5.43, 1e-6));

  // Off-diagonal should be zero
  REQUIRE_THAT(cell->matrix[0][1], Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(cell->matrix[0][2], Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("read_xyz handles single atom", "[io]") {
  std::istringstream ss(SINGLE_ATOM_XYZ);
  auto system = read_xyz(ss);

  REQUIRE(system.num_atoms() == 1);
  REQUIRE(system.atomic_numbers()[0] == 6); // Carbon

  const auto *pos = system.positions();
  REQUIRE_THAT(pos[0], Catch::Matchers::WithinAbs(1.5, 1e-6));
  REQUIRE_THAT(pos[1], Catch::Matchers::WithinAbs(2.5, 1e-6));
  REQUIRE_THAT(pos[2], Catch::Matchers::WithinAbs(3.5, 1e-6));
}

TEST_CASE("read_xyz parses triclinic cell", "[io]") {
  std::istringstream ss(TRICLINIC_EXYZ);
  auto system = read_xyz(ss);

  REQUIRE(system.is_periodic());
  const Cell *cell = system.cell();

  // First lattice vector: (4, 0, 0)
  REQUIRE_THAT(cell->matrix[0][0], Catch::Matchers::WithinAbs(4.0, 1e-6));
  REQUIRE_THAT(cell->matrix[0][1], Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(cell->matrix[0][2], Catch::Matchers::WithinAbs(0.0, 1e-6));

  // Second lattice vector: (1, 5, 0)
  REQUIRE_THAT(cell->matrix[1][0], Catch::Matchers::WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(cell->matrix[1][1], Catch::Matchers::WithinAbs(5.0, 1e-6));
  REQUIRE_THAT(cell->matrix[1][2], Catch::Matchers::WithinAbs(0.0, 1e-6));

  // Third lattice vector: (0.5, 0.5, 6)
  REQUIRE_THAT(cell->matrix[2][0], Catch::Matchers::WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(cell->matrix[2][1], Catch::Matchers::WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(cell->matrix[2][2], Catch::Matchers::WithinAbs(6.0, 1e-6));
}

TEST_CASE("write_xyz produces valid output", "[io]") {
  // Create a simple system
  std::vector<int32_t> atomic_numbers = {8, 1, 1};
  std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f};

  AtomicSystem system(3, positions.data(), atomic_numbers.data(), nullptr);

  // Write to string
  std::ostringstream oss;
  write_xyz(oss, system, "Test comment");

  std::string output = oss.str();

  // Check output contains expected content
  REQUIRE(output.find("3") != std::string::npos);
  REQUIRE(output.find("Test comment") != std::string::npos);
  REQUIRE(output.find("O") != std::string::npos);
  REQUIRE(output.find("H") != std::string::npos);
}

TEST_CASE("XYZ round-trip preserves data", "[io]") {
  // Create original system
  std::vector<int32_t> atomic_numbers = {6, 7, 8}; // C, N, O
  std::vector<float> positions = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f};

  AtomicSystem original(3, positions.data(), atomic_numbers.data(), nullptr);

  // Write to string
  std::ostringstream oss;
  write_xyz(oss, original);

  // Read back
  std::istringstream iss(oss.str());
  auto roundtrip = read_xyz(iss);

  // Verify
  REQUIRE(roundtrip.num_atoms() == original.num_atoms());

  for (int i = 0; i < 3; ++i) {
    REQUIRE(roundtrip.atomic_numbers()[i] == original.atomic_numbers()[i]);
  }

  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(roundtrip.positions()[i],
                 Catch::Matchers::WithinAbs(original.positions()[i], 1e-4));
  }
}

TEST_CASE("parse_exyz_lattice extracts cell vectors", "[io]") {
  double cell[3][3];

  SECTION("standard format") {
    bool ok = parse_exyz_lattice(
        R"(Lattice="5.43 0.0 0.0 0.0 5.43 0.0 0.0 0.0 5.43" other="stuff")",
        cell);
    REQUIRE(ok);
    REQUIRE_THAT(cell[0][0], Catch::Matchers::WithinAbs(5.43, 1e-6));
    REQUIRE_THAT(cell[1][1], Catch::Matchers::WithinAbs(5.43, 1e-6));
    REQUIRE_THAT(cell[2][2], Catch::Matchers::WithinAbs(5.43, 1e-6));
  }

  SECTION("no lattice returns false") {
    bool ok = parse_exyz_lattice("Just a comment line", cell);
    REQUIRE_FALSE(ok);
  }

  SECTION("triclinic cell") {
    bool ok = parse_exyz_lattice(
        R"(Lattice="4.0 0.0 0.0 1.0 5.0 0.0 0.5 0.5 6.0")", cell);
    REQUIRE(ok);
    REQUIRE_THAT(cell[1][0], Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(cell[2][0], Catch::Matchers::WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(cell[2][1], Catch::Matchers::WithinAbs(0.5, 1e-6));
  }
}
