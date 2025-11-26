#!/usr/bin/env python3
"""
Example of using mlipcpp with ASE (Atomic Simulation Environment).

This demonstrates:
- Using the MLIPCalculator with ASE Atoms objects
- Running geometry optimization
- Computing vibrational frequencies

Usage:
    python python_ase.py path/to/model.gguf

Requirements:
    pip install ase
"""

import sys

try:
    from ase import Atoms
    from ase.build import molecule, bulk
    from ase.optimize import BFGS
except ImportError:
    print("ASE is required for this example. Install with: pip install ase")
    sys.exit(1)

from mlipcpp.ase import MLIPCalculator


def example_molecule(model_path: str):
    """Example with a water molecule."""
    print("\n=== Water Molecule ===")

    # Create calculator
    calc = MLIPCalculator(model_path)
    print(f"Model: {calc.model_type}, cutoff: {calc.cutoff:.2f} A")

    # Create water molecule
    atoms = molecule("H2O")
    atoms.calc = calc

    # Get energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Initial energy: {energy:.6f} eV")
    print(f"Max force: {abs(forces).max():.6f} eV/A")

    # Optimize geometry
    print("\nOptimizing geometry...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01)

    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"Max force: {abs(atoms.get_forces()).max():.6f} eV/A")
    print(f"Steps: {opt.nsteps}")


def example_crystal(model_path: str):
    """Example with a silicon crystal."""
    print("\n=== Silicon Crystal ===")

    # Create calculator
    calc = MLIPCalculator(model_path)

    # Create silicon bulk
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = calc

    # Get energy
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.6f} eV")
    print(f"Energy per atom: {energy / len(atoms):.6f} eV/atom")

    # Get stress tensor
    try:
        stress = atoms.get_stress()
        print(f"Stress (Voigt): {stress}")
    except Exception as e:
        print(f"Stress not available: {e}")


def example_supercell(model_path: str):
    """Example with a larger supercell."""
    print("\n=== Silicon Supercell (2x2x2) ===")

    calc = MLIPCalculator(model_path)

    # Create 2x2x2 supercell
    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    atoms.calc = calc

    print(f"Number of atoms: {len(atoms)}")

    import time

    start = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    elapsed = time.perf_counter() - start

    print(f"Energy: {energy:.6f} eV")
    print(f"Energy per atom: {energy / len(atoms):.6f} eV/atom")
    print(f"Time: {elapsed * 1000:.2f} ms")


def main():
    if len(sys.argv) < 2:
        print("Usage: python python_ase.py <model.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]

    example_molecule(model_path)
    example_crystal(model_path)
    example_supercell(model_path)


if __name__ == "__main__":
    main()
