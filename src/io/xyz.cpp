#include "mlipcpp/io.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mlipcpp {
namespace io {

// Element symbols indexed by atomic number (index 0 unused). Covers the full
// periodic table (Z 1-118) so any element can be read/written.
static const char *const ELEMENT_SYMBOLS[] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

static constexpr int MAX_ATOMIC_NUMBER =
    static_cast<int>(sizeof(ELEMENT_SYMBOLS) / sizeof(ELEMENT_SYMBOLS[0])) - 1;

static int element_to_atomic_number(const std::string &symbol) {
  for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; ++Z) {
    if (symbol == ELEMENT_SYMBOLS[Z]) {
      return Z;
    }
  }

  throw std::runtime_error("Unknown element symbol: " + symbol);
}

bool parse_exyz_lattice(const std::string &comment_line, double cell[3][3]) {
  // Look for: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"
  size_t lattice_pos = comment_line.find("Lattice=\"");
  if (lattice_pos == std::string::npos) {
    return false;
  }

  size_t start = lattice_pos + 9; // Skip 'Lattice="'
  size_t end = comment_line.find("\"", start);
  if (end == std::string::npos) {
    return false;
  }

  std::string lattice_str = comment_line.substr(start, end - start);
  std::istringstream iss(lattice_str);

  // Parse 9 values for 3x3 cell matrix
  double values[9];
  for (int i = 0; i < 9; ++i) {
    if (!(iss >> values[i])) {
      return false;
    }
  }

  // Store in cell array (row-major)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cell[i][j] = values[i * 3 + j];
    }
  }

  return true;
}

AtomicSystem read_xyz(std::istream &stream) {
  std::string line;

  // Line 1: number of atoms
  if (!std::getline(stream, line)) {
    throw std::runtime_error("XYZ: Failed to read number of atoms");
  }
  int n_atoms = std::stoi(line);

  // Line 2: comment (may contain lattice for EXYZ)
  if (!std::getline(stream, line)) {
    throw std::runtime_error("XYZ: Failed to read comment line");
  }
  std::string comment = line;

  // Try to parse lattice from comment (EXYZ format)
  double cell[3][3] = {{0}};
  bool has_cell = parse_exyz_lattice(comment, cell);

  // Read atoms
  std::vector<int> atomic_numbers;
  std::vector<double> positions;

  atomic_numbers.reserve(n_atoms);
  positions.reserve(n_atoms * 3);

  for (int i = 0; i < n_atoms; ++i) {
    if (!std::getline(stream, line)) {
      throw std::runtime_error("XYZ: Unexpected end of file at atom " +
                               std::to_string(i));
    }

    std::istringstream iss(line);
    std::string element;
    double x, y, z;

    if (!(iss >> element >> x >> y >> z)) {
      throw std::runtime_error("XYZ: Failed to parse atom line: " + line);
    }

    atomic_numbers.push_back(element_to_atomic_number(element));
    positions.push_back(x);
    positions.push_back(y);
    positions.push_back(z);
  }

  // Create atomic system
  if (has_cell) {
    // Convert double cell to float
    float cell_f[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cell_f[i][j] = static_cast<float>(cell[i][j]);
      }
    }
    Cell cell_obj(cell_f);

    // Convert to float positions
    std::vector<float> positions_f(positions.begin(), positions.end());

    return AtomicSystem(n_atoms, positions_f.data(), atomic_numbers.data(),
                        &cell_obj);
  } else {
    // No cell - non-periodic
    std::vector<float> positions_f(positions.begin(), positions.end());
    return AtomicSystem(n_atoms, positions_f.data(), atomic_numbers.data(),
                        nullptr);
  }
}

AtomicSystem read_xyz(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("XYZ: Failed to open file: " + filename);
  }
  return read_xyz(file);
}

void write_xyz(std::ostream &stream, const AtomicSystem &system,
               const std::string &comment) {
  int n_atoms = system.num_atoms();
  const int32_t *atomic_nums = system.atomic_numbers();
  const float *pos = system.positions();

  // Write number of atoms
  stream << n_atoms << "\n";

  // Write comment
  stream << comment << "\n";

  // Write atoms
  for (int i = 0; i < n_atoms; ++i) {
    int Z = atomic_nums[i];
    if (Z <= 0 || Z > MAX_ATOMIC_NUMBER) {
      throw std::runtime_error("XYZ: Unsupported atomic number: " +
                               std::to_string(Z));
    }

    stream << ELEMENT_SYMBOLS[Z] << " " << pos[i * 3 + 0] << " "
           << pos[i * 3 + 1] << " " << pos[i * 3 + 2] << "\n";
  }
}

void write_xyz(const std::string &filename, const AtomicSystem &system,
               const std::string &comment) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("XYZ: Failed to open file for writing: " +
                             filename);
  }
  write_xyz(file, system, comment);
}

} // namespace io
} // namespace mlipcpp
