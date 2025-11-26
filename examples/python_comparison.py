#!/usr/bin/env python3
"""
Compare mlipcpp (C++) vs PET-MAD (PyTorch) predictions.

This script runs both calculators on the same structures and compares
energy, forces, and timing.

Usage:
    python examples/python_comparison.py
    python examples/python_comparison.py structure.xyz
    python examples/python_comparison.py --cpp-backend metal --torch-device mps
    python examples/python_comparison.py structure.xyz --cpp-backend cpu --torch-device cuda

Requirements:
    pip install ase pet-mad torch
"""

import argparse
import sys
import time
import numpy as np

try:
    from ase import Atoms
    from ase.build import bulk
    import ase.io
except ImportError:
    print("ASE is required. Install with: pip install ase")
    sys.exit(1)

import mlipcpp
from mlipcpp.ase import MLIPCalculator

try:
    from pet_mad.calculator import PETMADCalculator
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


def compare_results(
    atoms: Atoms,
    name: str,
    cpp_backend: str,
    torch_device: str,
    model_path: str,
):
    """Compare results from both calculators."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Atoms: {len(atoms)}, Formula: {atoms.get_chemical_formula()}")
    if atoms.cell.volume > 0:
        print(f"Cell: {atoms.cell.lengths()} A, PBC: {atoms.pbc.tolist()}")
    print()

    # mlipcpp (C++)
    print(f"--- mlipcpp (C++) [{cpp_backend}] ---")
    calc_cpp = MLIPCalculator(model_path)
    atoms_cpp = atoms.copy()
    atoms_cpp.calc = calc_cpp

    t0 = time.perf_counter()
    energy_cpp = atoms_cpp.get_potential_energy()
    forces_cpp = atoms_cpp.get_forces()
    t_cpp = time.perf_counter() - t0

    print(f"Energy: {energy_cpp:.6f} eV")
    print(f"Time:   {t_cpp*1000:.2f} ms")

    if not PYTORCH_AVAILABLE:
        print("\n(Skipping PyTorch comparison - pet-mad not installed)")
        return

    # PET-MAD (PyTorch)
    print(f"\n--- PET-MAD (PyTorch) [{torch_device}] ---")
    calc_torch = PETMADCalculator(version="latest", device=torch_device)
    atoms_torch = atoms.copy()
    atoms_torch.calc = calc_torch

    t0 = time.perf_counter()
    energy_torch = atoms_torch.get_potential_energy()
    forces_torch = atoms_torch.get_forces()
    t_torch = time.perf_counter() - t0

    print(f"Energy: {energy_torch:.6f} eV")
    print(f"Time:   {t_torch*1000:.2f} ms")

    # Comparison
    print("\n--- Comparison ---")
    energy_diff = abs(energy_cpp - energy_torch)
    force_max_diff = np.abs(forces_cpp - forces_torch).max()
    force_rmse = np.sqrt(np.mean((forces_cpp - forces_torch) ** 2))

    print(f"Energy difference: {energy_diff:.6e} eV")
    print(f"Force max diff:    {force_max_diff:.6e} eV/A")
    print(f"Force RMSE:        {force_rmse:.6e} eV/A")
    print(f"Speedup:           {t_torch/t_cpp:.1f}x")

    # Check if within tolerance
    if energy_diff < 1e-4 and force_max_diff < 1e-3:
        print("Status: PASS (within tolerance)")
    else:
        print("Status: MISMATCH (check implementation)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare mlipcpp (C++) vs PET-MAD (PyTorch) predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run on test structures
  %(prog)s structure.xyz                      # Run on user structure
  %(prog)s --cpp-backend metal                # Use Metal for mlipcpp
  %(prog)s --torch-device cuda                # Use CUDA for PyTorch
  %(prog)s --cpp-backend cpu --torch-device cpu  # Both on CPU
""",
    )
    parser.add_argument(
        "structure",
        nargs="?",
        help="Path to structure file (XYZ format). If not provided, uses test structures.",
    )
    parser.add_argument(
        "--model",
        default="pet-mad.gguf",
        help="Path to GGUF model file (default: pet-mad.gguf)",
    )
    parser.add_argument(
        "--cpp-backend",
        choices=["auto", "cpu", "metal", "cuda", "hip", "vulkan"],
        default="auto",
        help="Backend for mlipcpp (default: auto)",
    )
    parser.add_argument(
        "--torch-device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for PyTorch (default: cpu)",
    )

    args = parser.parse_args()

    # Set mlipcpp backend
    backend_map = {
        "auto": mlipcpp.Backend.Auto,
        "cpu": mlipcpp.Backend.CPU,
        "metal": mlipcpp.Backend.Metal,
        "cuda": mlipcpp.Backend.CUDA,
        "hip": mlipcpp.Backend.HIP,
        "vulkan": mlipcpp.Backend.Vulkan,
    }
    mlipcpp.set_backend(backend_map[args.cpp_backend])
    actual_backend = mlipcpp.get_backend_name()
    print(f"mlipcpp backend: {actual_backend}")

    if not PYTORCH_AVAILABLE:
        print("Warning: pet-mad not available. Install with: pip install pet-mad torch")
    else:
        print(f"PyTorch device: {args.torch_device}")

    if args.structure:
        # Load user-provided structure
        atoms = ase.io.read(args.structure)
        compare_results(
            atoms,
            f"User structure: {args.structure}",
            actual_backend,
            args.torch_device,
            args.model,
        )
    else:
        # Run on test structures
        test_structures = [
            (
                Atoms(
                    symbols=["O", "H", "H"],
                    positions=[
                        [0.000, 0.000, 0.000],
                        [0.757, 0.586, 0.000],
                        [-0.757, 0.586, 0.000],
                    ],
                ),
                "Water molecule (H2O)",
            ),
            (bulk("Si", "diamond", a=5.43), "Silicon 2-atom (diamond)"),
            (bulk("Si", "diamond", a=5.43) * (2, 2, 2), "Silicon 16-atom supercell"),
            (bulk("Si", "diamond", a=5.43) * (2, 2, 4), "Silicon 32-atom supercell"),
        ]

        for atoms, name in test_structures:
            compare_results(
                atoms,
                name,
                actual_backend,
                args.torch_device,
                args.model,
            )

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
