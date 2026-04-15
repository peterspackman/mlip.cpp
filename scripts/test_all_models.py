#!/usr/bin/env python3
"""Test all PET models against PyTorch reference on all geometries.

Usage:
    uv run scripts/test_all_models.py [--models MODEL1,MODEL2] [--geometries water.xyz,urea.xyz]

Examples:
    uv run scripts/test_all_models.py                    # Test all models on all geometries
    uv run scripts/test_all_models.py --models pet-mad-s # Test only pet-mad-s
    uv run scripts/test_all_models.py --forces           # Test with forces
"""

import argparse
import subprocess
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Available PET models (from HuggingFace lab-cosmo/upet)
AVAILABLE_MODELS = [
    "pet-mad-s",
    "pet-omad-xs",
    "pet-omad-s",
    "pet-omat-xs",
    "pet-omat-s",
    "pet-spice-s",
]

def get_geometries(geometries_dir: Path) -> list[Path]:
    """Get all XYZ files in the geometries directory."""
    return sorted(geometries_dir.glob("*.xyz"))

EXPORT_TIMEOUT = 300  # 5 minutes for model download + tracing
INFERENCE_TIMEOUT = 120  # 2 minutes per inference

def export_model(model_name: str, output_dir: Path, forces: bool = False) -> bool:
    """Export a PET model using export_pet_full.py."""
    cmd = [
        "uv", "run", "scripts/export_pytorch/export_pet_full.py",
        "--model", model_name,
        "-o", str(output_dir),
    ]
    if forces:
        cmd.append("--forces")

    print(f"  Exporting {model_name}{'(forces)' if forces else ''}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=EXPORT_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Export timed out after {EXPORT_TIMEOUT}s")
        return False
    if result.returncode != 0:
        print(f"    ERROR: Export failed")
        print(f"    {result.stderr[:500]}")
        return False
    return True

def run_cpp_inference(model_dir: Path, xyz_path: Path, forces: bool = False) -> dict | None:
    """Run C++ graph_inference and parse results."""
    cmd = ["./build/bin/graph_inference", str(model_dir), str(xyz_path)]
    if forces:
        cmd.append("--forces")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=INFERENCE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"    C++ ERROR: timed out after {INFERENCE_TIMEOUT}s")
        return None
    if result.returncode != 0:
        print(f"    C++ ERROR: {result.stderr[:200]}")
        return None

    # Parse output
    output = result.stdout
    data = {"atomic_energies": [], "forces": None}

    # Parse atomic energies
    in_energies = False
    in_forces = False
    forces_list = []

    for line in output.split("\n"):
        if "Atomic energies:" in line:
            in_energies = True
            continue
        if "Forces:" in line:
            in_energies = False
            in_forces = True
            continue
        if in_energies and line.strip().startswith("Atom"):
            # "  Atom 0: 1.234567 eV"
            parts = line.split(":")
            if len(parts) >= 2:
                energy = float(parts[1].strip().split()[0])
                data["atomic_energies"].append(energy)
        if in_forces and line.strip().startswith("Atom"):
            # "  Atom 0: [1.23, 4.56, 7.89] eV/A"
            parts = line.split(":")
            if len(parts) >= 2:
                force_str = parts[1].strip()
                # Extract [x, y, z]
                import re
                match = re.search(r'\[([-\d.e+]+),\s*([-\d.e+]+),\s*([-\d.e+]+)\]', force_str)
                if match:
                    fx, fy, fz = map(float, match.groups())
                    forces_list.append([fx, fy, fz])
        if "Model energy (raw):" in line:
            data["raw_energy"] = float(line.split(":")[1].strip().split()[0])

    if forces_list:
        data["forces"] = forces_list

    return data

def run_python_reference(model_name: str, xyz_path: Path, forces: bool = False) -> dict | None:
    """Run PyTorch reference computation."""
    # Import here to avoid slow startup
    sys.path.insert(0, str(Path(__file__).parent))
    from export_pytorch.export_pet_full import PETFullModel, load_pet_model, get_model_params, get_species_mapping

    import torch
    from ase.io import read

    # Load model
    try:
        model = load_pet_model(model_name)
        params = get_model_params(model)
        species_map = get_species_mapping(model)
    except Exception as e:
        print(f"    Python ERROR loading model: {e}")
        return None

    # Read structure
    atoms = read(str(xyz_path))
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)

    # Build neighbor list
    from ase.neighborlist import neighbor_list
    cutoff = params['cutoff']

    i_list, j_list, d_list, D_list = neighbor_list('ijdD', atoms, cutoff, self_interaction=False)

    # Build padded arrays
    neighbor_counts = np.bincount(i_list, minlength=n_atoms)
    max_neighbors = int(neighbor_counts.max()) if len(neighbor_counts) > 0 else 1

    # Prepare tensors
    species = torch.tensor([species_map.get(Z, 0) for Z in atomic_numbers], dtype=torch.long)
    neighbor_species = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    edge_vectors = torch.zeros(n_atoms, max_neighbors, 3, dtype=torch.float32)
    edge_distances = torch.zeros(n_atoms, max_neighbors, dtype=torch.float32)
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)  # True = padded
    cutoff_factors = torch.zeros(n_atoms, max_neighbors, dtype=torch.float32)
    reverse_neighbor_index = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)

    # Fill arrays
    slot_indices = np.zeros(n_atoms, dtype=np.int32)
    edge_to_flat = {}

    for e, (i, j, d, D) in enumerate(zip(i_list, j_list, d_list, D_list)):
        slot = slot_indices[i]
        if slot >= max_neighbors:
            continue
        slot_indices[i] += 1

        flat_idx = i * max_neighbors + slot
        edge_to_flat[(i, j)] = flat_idx

        neighbor_species[i, slot] = species_map.get(atomic_numbers[j], 0)
        edge_vectors[i, slot] = torch.tensor(D, dtype=torch.float32)
        edge_distances[i, slot] = d
        padding_mask[i, slot] = False  # Valid edge

        # Cutoff factor
        width = params.get('cutoff_width', 0.2)
        if d <= cutoff - width:
            cutoff_factors[i, slot] = 1.0
        elif d < cutoff:
            scaled = (d - (cutoff - width)) / width
            cutoff_factors[i, slot] = 0.5 * (1.0 + np.cos(np.pi * scaled))

    # Build reverse neighbor index
    for e, (i, j) in enumerate(zip(i_list, j_list)):
        if (i, j) in edge_to_flat and (j, i) in edge_to_flat:
            flat_ij = edge_to_flat[(i, j)]
            flat_ji = edge_to_flat[(j, i)]
            slot_ij = flat_ij % max_neighbors
            reverse_neighbor_index[i, slot_ij] = flat_ji

    # Create wrapper and run
    wrapper = PETFullModel(
        model, n_atoms=n_atoms, max_neighbors=max_neighbors,
        d_pet=params['d_pet'], forces=forces,
        cutoff=params['cutoff'], cutoff_width=params.get('cutoff_width', 0.2)
    )
    wrapper.eval()

    if forces:
        edge_vectors.requires_grad_(True)

    with torch.set_grad_enabled(forces):
        if forces:
            result = wrapper(species, neighbor_species, edge_vectors, padding_mask, reverse_neighbor_index)
        else:
            result = wrapper(species, neighbor_species, edge_vectors, edge_distances,
                           padding_mask, reverse_neighbor_index, cutoff_factors)

    data = {
        "atomic_energies": result.detach().numpy().flatten().tolist(),
        "raw_energy": float(result.sum().item()),
    }

    if forces:
        # Compute forces via backward pass
        total_energy = result.sum()
        total_energy.backward()

        # Scatter edge gradients to atom forces
        grad = edge_vectors.grad  # [n_atoms, max_neighbors, 3]
        forces_np = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            for slot in range(max_neighbors):
                if not padding_mask[i, slot]:
                    forces_np[i] -= grad[i, slot].numpy()

        data["forces"] = forces_np.tolist()

    return data

def compare_results(cpp_data: dict, py_data: dict, forces: bool = False) -> dict:
    """Compare C++ and Python results."""
    cpp_energies = np.array(cpp_data["atomic_energies"])
    py_energies = np.array(py_data["atomic_energies"])

    energy_diff = np.abs(cpp_energies - py_energies)

    result = {
        "energy_max_diff": float(energy_diff.max()),
        "energy_mean_diff": float(energy_diff.mean()),
        "total_energy_diff": abs(cpp_energies.sum() - py_energies.sum()),
        "pass": energy_diff.max() < 1e-2,  # 10 meV tolerance
    }

    if forces and cpp_data.get("forces") and py_data.get("forces"):
        cpp_forces = np.array(cpp_data["forces"])
        py_forces = np.array(py_data["forces"])
        force_diff = np.abs(cpp_forces - py_forces)

        result["force_max_diff"] = float(force_diff.max())
        result["force_mean_diff"] = float(force_diff.mean())
        result["pass"] = result["pass"] and force_diff.max() < 1e-2  # 10 meV/A tolerance

    return result

def main():
    parser = argparse.ArgumentParser(description="Test PET models against PyTorch reference")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated list of models to test (default: all)")
    parser.add_argument("--geometries", type=str, default=None,
                       help="Comma-separated list of geometry files (default: all in geometries/)")
    parser.add_argument("--forces", action="store_true",
                       help="Test with forces computation")
    parser.add_argument("--keep-exports", action="store_true",
                       help="Keep exported model directories (in /tmp/)")
    args = parser.parse_args()

    # Get models to test
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = AVAILABLE_MODELS

    # Get geometries to test
    geometries_dir = Path("geometries")
    if args.geometries:
        geometries = [geometries_dir / g.strip() for g in args.geometries.split(",")]
    else:
        geometries = get_geometries(geometries_dir)

    if not geometries:
        print("No geometry files found!")
        return 1

    print(f"Testing {len(models)} model(s) on {len(geometries)} geometry file(s)")
    print(f"Forces: {'Yes' if args.forces else 'No'}")
    print("=" * 70)

    results_summary = []

    for model_name in models:
        print(f"\n[{model_name}]")

        # Export model
        export_dir = Path(f"/tmp/test_model_{model_name.replace('-', '_')}")
        if not export_model(model_name, export_dir, forces=args.forces):
            results_summary.append({"model": model_name, "status": "EXPORT_FAILED"})
            continue

        for xyz_path in geometries:
            if not xyz_path.exists():
                print(f"  {xyz_path.name}: SKIP (file not found)")
                continue

            print(f"  {xyz_path.name}:")

            # Run C++ inference
            cpp_data = run_cpp_inference(export_dir, xyz_path, forces=args.forces)
            if cpp_data is None:
                results_summary.append({
                    "model": model_name, "geometry": xyz_path.name,
                    "status": "CPP_FAILED"
                })
                continue

            # Run Python reference
            py_data = run_python_reference(model_name, xyz_path, forces=args.forces)
            if py_data is None:
                results_summary.append({
                    "model": model_name, "geometry": xyz_path.name,
                    "status": "PYTHON_FAILED"
                })
                continue

            # Compare
            comparison = compare_results(cpp_data, py_data, forces=args.forces)

            status = "PASS" if comparison["pass"] else "FAIL"
            energy_info = f"E_diff: {comparison['energy_max_diff']:.6f} eV"

            if args.forces and "force_max_diff" in comparison:
                force_info = f", F_diff: {comparison['force_max_diff']:.6f} eV/A"
            else:
                force_info = ""

            print(f"    {status} - {energy_info}{force_info}")
            print(f"    C++: {cpp_data['raw_energy']:.6f} eV, Python: {py_data['raw_energy']:.6f} eV")

            results_summary.append({
                "model": model_name,
                "geometry": xyz_path.name,
                "status": status,
                **comparison
            })

        # Cleanup export directory
        if not args.keep_exports:
            import shutil
            shutil.rmtree(export_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results_summary if r.get("status") == "PASS")
    failed = sum(1 for r in results_summary if r.get("status") == "FAIL")
    errors = sum(1 for r in results_summary if r.get("status") not in ("PASS", "FAIL"))

    print(f"Passed: {passed}, Failed: {failed}, Errors: {errors}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results_summary:
            if r.get("status") == "FAIL":
                print(f"  {r['model']} / {r['geometry']}: E_diff={r.get('energy_max_diff', '?'):.6f}")

    if errors > 0:
        print("\nErrors:")
        for r in results_summary:
            if r.get("status") not in ("PASS", "FAIL"):
                print(f"  {r['model']}: {r.get('status', 'UNKNOWN')}")

    return 0 if (failed == 0 and errors == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
