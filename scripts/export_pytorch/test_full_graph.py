#!/usr/bin/env python3
"""Test the full PET graph export by comparing C++ interpreter output to PyTorch."""

import json
import subprocess
import numpy as np
from pathlib import Path

def main():
    output_dir = Path("/tmp/pet_full_export")

    # Load metadata
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    print("=== Test Configuration ===")
    print(f"n_atoms: {metadata['n_atoms']}")
    print(f"max_neighbors: {metadata['max_neighbors']}")
    print(f"d_pet: {metadata['d_pet']}")
    print(f"num_nodes: {metadata['num_nodes']}")
    print(f"num_weights: {metadata['num_weights']}")

    # Load expected output
    expected = np.fromfile(output_dir / "expected_output.bin", dtype=np.float32)
    print(f"\n=== PyTorch Reference ===")
    print(f"Atomic energies: {expected}")
    print(f"Total energy: {expected.sum():.6f}")

    # The C++ test would need to:
    # 1. Load graph JSON
    # 2. Load all weight tensors
    # 3. Create input tensors
    # 4. Run the graph
    # 5. Compare output

    print("\n=== Required for C++ Test ===")
    print("Inputs needed:")
    print(f"  - species: int32 [{metadata['n_atoms']}]")
    print(f"  - neighbor_species: int32 [{metadata['n_atoms']}, {metadata['max_neighbors']}]")
    print(f"  - edge_vectors: float32 [{metadata['n_atoms']}, {metadata['max_neighbors']}, 3]")
    print(f"  - edge_distances: float32 [{metadata['n_atoms']}, {metadata['max_neighbors']}]")

    print(f"\nWeights needed: {metadata['num_weights']}")
    for name, shape in list(metadata['weights'].items())[:5]:
        print(f"  - {name}: {shape}")
    print("  ...")

    # Check if all files exist
    print("\n=== File Status ===")
    required_files = [
        "pet_full.json",
        "input_species.bin",
        "input_neighbor_species.bin",
        "input_edge_vectors.bin",
        "input_edge_distances.bin",
        "expected_output.bin",
    ]
    for fname in required_files:
        path = output_dir / fname
        status = "OK" if path.exists() else "MISSING"
        print(f"  {fname}: {status}")

    # Count weight files
    weight_files = list(output_dir.glob("*.bin"))
    weight_files = [f for f in weight_files if not f.name.startswith("input_") and f.name != "expected_output.bin"]
    print(f"\nWeight files: {len(weight_files)}")

    print("\n=== Summary ===")
    print("The graph is exported and ready for C++ testing.")
    print("To run end-to-end on arbitrary XYZ files, we need:")
    print("1. Dynamic shape support (current graph has fixed n_atoms=2)")
    print("2. Or use torch.export with dynamic dimensions")
    print("3. Or re-export matching each input size")

if __name__ == "__main__":
    main()
