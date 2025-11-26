#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ase>=3.22.0",
#     "torch>=2.0.0",
#     "pet-mad",
# ]
# ///
"""
Calculate energy, forces, and stress using PET-MAD PyTorch reference.

Useful for validating mlipcpp results against the official implementation.

Usage:
    uv run scripts/calc_energy_pytorch.py structure.xyz
    uv run scripts/calc_energy_pytorch.py structure.xyz --device cuda
"""

import argparse
import time
from pathlib import Path

import ase.io


def main():
    parser = argparse.ArgumentParser(
        description="Calculate energy using PET-MAD PyTorch"
    )
    parser.add_argument("structure", type=str, help="Input structure file (XYZ, CIF, etc.)")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--version", type=str, default="latest", help="PET-MAD version (default: latest)"
    )
    parser.add_argument(
        "--no-forces", action="store_true", help="Skip force calculation"
    )
    parser.add_argument(
        "--no-stress", action="store_true", help="Skip stress calculation"
    )

    args = parser.parse_args()

    structure_path = Path(args.structure)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure not found: {structure_path}")

    t_start = time.perf_counter()

    atoms = ase.io.read(structure_path)

    print(f"Structure: {structure_path.name}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    if atoms.cell.volume > 0:
        print(f"  Cell volume: {atoms.cell.volume:.2f} A^3")
    print(f"  PBC: {atoms.pbc.tolist()}")
    print()

    from pet_mad.calculator import PETMADCalculator

    calculator = PETMADCalculator(version=args.version, device=args.device)
    atoms.calc = calculator

    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.6f} eV")
    print(f"Energy/atom: {energy / len(atoms):.6f} eV")
    print()

    if not args.no_forces:
        forces = atoms.get_forces()
        print("Forces (eV/A):")
        for i, (symbol, force) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
            print(f"  {i:3d} {symbol:2s}: [{force[0]:12.6f}, {force[1]:12.6f}, {force[2]:12.6f}]")
        print()

    if not args.no_stress and all(atoms.pbc):
        stress = atoms.get_stress()  # Voigt: xx, yy, zz, yz, xz, xy
        print("Stress (Voigt, eV/A^3):")
        print(f"  xx={stress[0]:.6f}, yy={stress[1]:.6f}, zz={stress[2]:.6f}")
        print(f"  yz={stress[3]:.6f}, xz={stress[4]:.6f}, xy={stress[5]:.6f}")
        print()

    t_end = time.perf_counter()
    print(f"Time: {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()
