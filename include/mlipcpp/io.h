#pragma once

#include "system.h"
#include <istream>
#include <string>
#include <vector>

namespace mlipcpp {
namespace io {

/**
 * Read atomic system from XYZ file format
 *
 * Basic XYZ format:
 *   <number_of_atoms>
 *   <comment line>
 *   <element> <x> <y> <z>
 *   <element> <x> <y> <z>
 *   ...
 *
 * Extended XYZ (EXYZ) format adds properties in comment line:
 *   Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33" Properties=...
 *
 * @param filename Path to XYZ file
 * @return AtomicSystem with positions and cell (if EXYZ)
 */
AtomicSystem read_xyz(const std::string &filename);

/**
 * Read atomic system from XYZ stream
 *
 * @param stream Input stream containing XYZ data
 * @return AtomicSystem with positions and cell (if EXYZ)
 */
AtomicSystem read_xyz(std::istream &stream);

/**
 * Write atomic system to XYZ file format
 *
 * @param filename Path to output XYZ file
 * @param system Atomic system to write
 * @param comment Optional comment line (default: empty)
 */
void write_xyz(const std::string &filename, const AtomicSystem &system,
               const std::string &comment = "");

/**
 * Write atomic system to XYZ stream
 *
 * @param stream Output stream
 * @param system Atomic system to write
 * @param comment Optional comment line (default: empty)
 */
void write_xyz(std::ostream &stream, const AtomicSystem &system,
               const std::string &comment = "");

/**
 * Parse extended XYZ comment line to extract lattice vectors
 *
 * Looks for: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"
 *
 * @param comment_line The comment line from EXYZ file
 * @param cell Output 3x3 cell matrix (row-major)
 * @return true if lattice was found and parsed, false otherwise
 */
bool parse_exyz_lattice(const std::string &comment_line, double cell[3][3]);

} // namespace io
} // namespace mlipcpp
