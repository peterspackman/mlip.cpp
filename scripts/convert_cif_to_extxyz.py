#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["ase>=3.22.0"]
# ///
"""
Convert CIF files to extended XYZ format.

Usage:
    uv run scripts/convert_cif_to_extxyz.py input.cif -o output.xyz
    uv run scripts/convert_cif_to_extxyz.py input.cif --supercell 2 2 2
"""

import argparse
from pathlib import Path

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIF to extended XYZ format"
    )
    parser.add_argument("input", type=str, help="Input CIF file")
    parser.add_argument("-o", "--output", type=str, help="Output XYZ file")
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Create supercell with NX x NY x NZ repetitions",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = args.output or input_path.with_suffix(".xyz")

    atoms = read(input_path)

    if args.supercell:
        atoms = atoms * tuple(args.supercell)
        print(f"Created {args.supercell[0]}x{args.supercell[1]}x{args.supercell[2]} supercell")

    print(f"Structure: {len(atoms)} atoms")
    print(f"Cell: {atoms.cell.lengths()} A")
    print(f"PBC: {atoms.pbc.tolist()}")

    write(output_path, atoms)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
