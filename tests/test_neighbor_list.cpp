/**
 * Unit tests for neighbor list construction
 */
#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <set>

using namespace mlipcpp;

// Helper to create a simple dimer (two atoms)
AtomicSystem create_dimer(float distance) {
  std::vector<int32_t> Z = {14, 14};
  std::vector<float> pos = {0.0f, 0.0f, 0.0f, distance, 0.0f, 0.0f};
  return AtomicSystem(2, pos.data(), Z.data(), nullptr);
}

// Helper to create water molecule
AtomicSystem create_water() {
  std::vector<int32_t> Z = {8, 1, 1};
  std::vector<float> pos = {0.0f, 0.0f,    0.0f,   0.757f, 0.586f,
                            0.0f, -0.757f, 0.586f, 0.0f};
  return AtomicSystem(3, pos.data(), Z.data(), nullptr);
}

// Helper to create Si crystal (2 atoms, periodic)
AtomicSystem create_si_crystal() {
  std::vector<int32_t> Z = {14, 14};
  std::vector<float> pos = {0.0f, 0.0f, 0.0f, 1.3575f, 1.3575f, 1.3575f};

  Cell cell;
  cell.matrix[0][0] = 5.43f;
  cell.matrix[0][1] = 0.0f;
  cell.matrix[0][2] = 0.0f;
  cell.matrix[1][0] = 0.0f;
  cell.matrix[1][1] = 5.43f;
  cell.matrix[1][2] = 0.0f;
  cell.matrix[2][0] = 0.0f;
  cell.matrix[2][1] = 0.0f;
  cell.matrix[2][2] = 5.43f;
  cell.periodic[0] = true;
  cell.periodic[1] = true;
  cell.periodic[2] = true;

  return AtomicSystem(2, pos.data(), Z.data(), &cell);
}

// Helper to create single isolated atom
AtomicSystem create_isolated_atom() {
  std::vector<int32_t> Z = {6};
  std::vector<float> pos = {0.0f, 0.0f, 0.0f};
  return AtomicSystem(1, pos.data(), Z.data(), nullptr);
}

TEST_CASE("NeighborList finds pairs within cutoff", "[neighbor_list]") {
  NeighborListOptions opts;
  opts.cutoff = 3.0f;
  opts.full_list = true;
  NeighborListBuilder builder(opts);

  SECTION("dimer within cutoff") {
    auto system = create_dimer(2.0f); // distance = 2.0 < cutoff
    auto nlist = builder.build(system);

    // Full list: should have (0,1) and (1,0)
    REQUIRE(nlist.num_pairs() == 2);

    // Check distances are correct
    for (int i = 0; i < nlist.num_pairs(); ++i) {
      REQUIRE_THAT(nlist.distances[i], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    }
  }

  SECTION("dimer outside cutoff") {
    auto system = create_dimer(5.0f); // distance = 5.0 > cutoff
    auto nlist = builder.build(system);

    REQUIRE(nlist.num_pairs() == 0);
  }

  SECTION("dimer at exact cutoff") {
    auto system = create_dimer(3.0f); // distance = cutoff (exclusive)
    auto nlist = builder.build(system);

    // At exact cutoff should not be included (< not <=)
    REQUIRE(nlist.num_pairs() == 0);
  }
}

TEST_CASE("NeighborList computes correct distances and vectors",
          "[neighbor_list]") {
  NeighborListOptions opts;
  opts.cutoff = 5.0f;
  opts.full_list = true;
  NeighborListBuilder builder(opts);

  auto system = create_water();
  auto nlist = builder.build(system);

  // Water: O-H1, O-H2, H1-H2 pairs (full list = 6 pairs)
  REQUIRE(nlist.num_pairs() == 6);

  // Check that distances match vector norms
  for (int i = 0; i < nlist.num_pairs(); ++i) {
    const auto &vec = nlist.edge_vectors[i];
    float computed_dist =
        std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);

    REQUIRE_THAT(nlist.distances[i],
                 Catch::Matchers::WithinAbs(computed_dist, 1e-5f));
  }

  // O-H distance should be ~0.96 A
  // Find an O-H pair
  for (int i = 0; i < nlist.num_pairs(); ++i) {
    int c = nlist.centers[i];
    int n = nlist.neighbors[i];
    // O is atom 0, H is atom 1 or 2
    if ((c == 0 && n > 0) || (n == 0 && c > 0)) {
      REQUIRE(nlist.distances[i] < 1.5f); // O-H should be short
    }
  }
}

TEST_CASE("NeighborList handles isolated atom", "[neighbor_list]") {
  NeighborListOptions opts;
  opts.cutoff = 5.0f;
  NeighborListBuilder builder(opts);

  auto system = create_isolated_atom();
  auto nlist = builder.build(system);

  REQUIRE(nlist.num_pairs() == 0);
  REQUIRE(nlist.n_atoms == 1);
}

TEST_CASE("NeighborList half vs full list", "[neighbor_list]") {
  auto system = create_water();

  SECTION("full list has both directions") {
    NeighborListOptions opts;
    opts.cutoff = 5.0f;
    opts.full_list = true;
    NeighborListBuilder builder(opts);

    auto nlist = builder.build(system);

    // 3 atoms, all within cutoff = 3 unique pairs * 2 = 6
    REQUIRE(nlist.num_pairs() == 6);

    // Check symmetry: for each (i,j) there should be (j,i)
    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < nlist.num_pairs(); ++i) {
      pairs.insert({nlist.centers[i], nlist.neighbors[i]});
    }

    for (int i = 0; i < nlist.num_pairs(); ++i) {
      int c = nlist.centers[i];
      int n = nlist.neighbors[i];
      REQUIRE(pairs.count({n, c}) == 1);
    }
  }

  SECTION("half list has one direction only") {
    NeighborListOptions opts;
    opts.cutoff = 5.0f;
    opts.full_list = false;
    NeighborListBuilder builder(opts);

    auto nlist = builder.build(system);

    // 3 atoms = 3 unique pairs
    REQUIRE(nlist.num_pairs() == 3);

    // No duplicates: (i,j) but not (j,i)
    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < nlist.num_pairs(); ++i) {
      auto p = std::minmax(nlist.centers[i], nlist.neighbors[i]);
      REQUIRE(pairs.count(p) == 0);
      pairs.insert(p);
    }
  }
}

TEST_CASE("NeighborList handles periodic boundaries", "[neighbor_list]") {
  NeighborListOptions opts;
  opts.cutoff = 5.0f;
  opts.full_list = true;
  NeighborListBuilder builder(opts);

  auto system = create_si_crystal();
  auto nlist = builder.build(system);

  // With periodic boundaries, each Si sees multiple images
  // The nearest neighbor distance in diamond Si is ~2.35 A
  // Within 5 A cutoff, each atom should see several neighbors
  REQUIRE(nlist.num_pairs() > 2);

  // All distances should be within cutoff
  for (int i = 0; i < nlist.num_pairs(); ++i) {
    REQUIRE(nlist.distances[i] < opts.cutoff);
    REQUIRE(nlist.distances[i] > 0.0f);
  }

  // Check that cell shifts are used for periodic images
  bool has_nonzero_shift = false;
  for (int i = 0; i < nlist.num_pairs(); ++i) {
    const auto &shift = nlist.cell_shifts[i];
    if (shift[0] != 0 || shift[1] != 0 || shift[2] != 0) {
      has_nonzero_shift = true;
      break;
    }
  }
  REQUIRE(has_nonzero_shift);
}

TEST_CASE("NeighborList edge vectors point from center to neighbor",
          "[neighbor_list]") {
  NeighborListOptions opts;
  opts.cutoff = 5.0f;
  opts.full_list = true;
  NeighborListBuilder builder(opts);

  // Simple dimer along x-axis
  auto system = create_dimer(2.0f);
  auto nlist = builder.build(system);

  REQUIRE(nlist.num_pairs() == 2);

  for (int i = 0; i < nlist.num_pairs(); ++i) {
    int c = nlist.centers[i];
    int n = nlist.neighbors[i];
    const auto &vec = nlist.edge_vectors[i];

    // For 0->1: vector should be (+2, 0, 0)
    // For 1->0: vector should be (-2, 0, 0)
    if (c == 0 && n == 1) {
      REQUIRE_THAT(vec[0], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
    } else if (c == 1 && n == 0) {
      REQUIRE_THAT(vec[0], Catch::Matchers::WithinAbs(-2.0f, 1e-5f));
    }

    REQUIRE_THAT(vec[1], Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(vec[2], Catch::Matchers::WithinAbs(0.0f, 1e-5f));
  }
}

TEST_CASE("NeighborList respects self_interaction option", "[neighbor_list]") {
  auto system = create_isolated_atom();

  SECTION("no self interaction by default") {
    NeighborListOptions opts;
    opts.cutoff = 5.0f;
    opts.self_interaction = false;
    NeighborListBuilder builder(opts);

    auto nlist = builder.build(system);
    REQUIRE(nlist.num_pairs() == 0);
  }

  // Note: self_interaction=true would add (i,i) pairs with distance 0
  // This is rarely used but could be tested if the implementation supports it
}
