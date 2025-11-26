#!/usr/bin/env python3
"""
Basic example of using mlipcpp for energy and force predictions.

Usage:
    python python_basic.py path/to/model.gguf
"""

import sys
import numpy as np
import mlipcpp


def main():
    if len(sys.argv) < 2:
        print("Usage: python python_basic.py <model.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load the model
    print(f"Loading model: {model_path}")
    model = mlipcpp.Predictor(model_path)
    print(f"Model type: {model.model_type}")
    print(f"Cutoff: {model.cutoff:.2f} Angstroms")

    # Water molecule (non-periodic) - same geometry as C++ tests
    positions = np.array(
        [
            [0.000, 0.000, 0.000],  # O
            [0.757, 0.586, 0.000],  # H
            [-0.757, 0.586, 0.000],  # H
        ],
        dtype=np.float32,
    )
    atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

    # Predict energy and forces
    print("\n--- Water molecule (non-periodic) ---")
    result = model.predict(positions, atomic_numbers, compute_forces=True)

    print(f"Energy: {result.energy:.6f} eV")
    print(f"Forces (eV/Angstrom):")
    for i, (Z, force) in enumerate(zip(atomic_numbers, result.forces)):
        symbol = {1: "H", 8: "O"}[Z]
        print(f"  {symbol}{i}: [{force[0]:>8.4f}, {force[1]:>8.4f}, {force[2]:>8.4f}]")

    # Silicon crystal (periodic)
    a = 5.43  # lattice constant in Angstroms
    positions_si = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    ) * a
    atomic_numbers_si = np.array([14, 14], dtype=np.int32)
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
        ],
        dtype=np.float32,
    )

    print("\n--- Silicon crystal (periodic) ---")
    result = model.predict(
        positions_si,
        atomic_numbers_si,
        cell=cell,
        pbc=(True, True, True),
        compute_forces=True,
    )

    print(f"Energy: {result.energy:.6f} eV")
    print(f"Energy per atom: {result.energy / len(atomic_numbers_si):.6f} eV/atom")


if __name__ == "__main__":
    main()
