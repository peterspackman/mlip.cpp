#!/usr/bin/env python3
"""Compare Python and C++ intermediate tensors for debugging.

This script:
1. Loads tensor traces from both Python and C++ output directories
2. Attempts to match tensors by name/node_id
3. Computes differences and reports the first significant divergence

Usage:
    uv run scripts/export_pytorch/compare_traces.py [--py-dir /tmp/pet_debug/py] [--cpp-dir /tmp/pet_debug/cpp]
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


@dataclass
class TensorComparison:
    """Result of comparing two tensors."""
    name: str
    py_node_id: int
    cpp_node_id: int
    py_shape: List[int]
    cpp_shape: List[int]
    shape_match: bool
    max_diff: float
    mean_diff: float
    rel_max_diff: float  # Relative to tensor magnitude
    first_diff_idx: int  # Index of first significant difference
    py_values_at_diff: List[float]
    cpp_values_at_diff: List[float]


def load_tensor_from_bin(bin_path: Path, json_path: Path) -> Tuple[np.ndarray, dict]:
    """Load a tensor from binary file with metadata."""
    # Load metadata
    with open(json_path) as f:
        meta = json.load(f)

    # Load binary data
    data = np.fromfile(bin_path, dtype=np.float32)

    # Reshape according to metadata
    shape = meta.get("shape", [len(data)])
    # Filter out trailing 1s for reshape
    shape = [s for s in shape if s > 0]
    if len(shape) == 0:
        shape = [1]

    # Handle potential size mismatches
    expected_size = 1
    for s in shape:
        expected_size *= s

    if len(data) != expected_size:
        # Try using n_elements from metadata
        n_elements = meta.get("n_elements", len(data))
        if n_elements == len(data):
            # Can't reshape, return flat
            return data, meta
        else:
            # Use n_dims to determine actual shape
            n_dims = meta.get("n_dims", len(shape))
            actual_shape = shape[:n_dims]
            actual_size = 1
            for s in actual_shape:
                actual_size *= s
            if actual_size == len(data):
                data = data.reshape(actual_shape)
            # Otherwise keep flat
    else:
        data = data.reshape(shape)

    return data, meta


def load_py_tensors(py_dir: Path) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Load all Python trace tensors."""
    tensors = {}
    for json_path in sorted(py_dir.glob("node_*.json")):
        bin_path = json_path.with_suffix(".bin")
        if bin_path.exists():
            try:
                data, meta = load_tensor_from_bin(bin_path, json_path)
                name = meta.get("name", json_path.stem)
                tensors[name] = (data, meta)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
    return tensors


def load_cpp_tensors(cpp_dir: Path) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Load all C++ trace tensors."""
    tensors = {}
    for json_path in sorted(cpp_dir.glob("node_*.json")):
        bin_path = json_path.with_suffix(".bin")
        if bin_path.exists():
            try:
                data, meta = load_tensor_from_bin(bin_path, json_path)
                name = meta.get("name", json_path.stem)
                tensors[name] = (data, meta)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
    return tensors


def find_matching_tensor(py_name: str, cpp_tensors: Dict[str, Tuple[np.ndarray, dict]]) -> Optional[str]:
    """Find the best matching C++ tensor for a Python tensor name."""
    # Exact match
    if py_name in cpp_tensors:
        return py_name

    # Try partial matches
    py_lower = py_name.lower()
    for cpp_name in cpp_tensors:
        cpp_lower = cpp_name.lower()
        # Check if one contains the other
        if py_lower in cpp_lower or cpp_lower in py_lower:
            return cpp_name

    # Match by pattern (gnn0_layer0 -> gnn_layers_0_layers_0)
    parts = py_name.split("_")
    for cpp_name in cpp_tensors:
        cpp_parts = cpp_name.split("_")
        # Count matching parts
        matches = sum(1 for p in parts if p in cpp_parts)
        if matches >= len(parts) // 2:
            return cpp_name

    return None


def compare_tensors(
    py_data: np.ndarray,
    py_meta: dict,
    cpp_data: np.ndarray,
    cpp_meta: dict,
    name: str
) -> TensorComparison:
    """Compare two tensors and compute difference metrics."""
    py_shape = list(py_data.shape)
    cpp_shape = list(cpp_data.shape)

    # Check shape compatibility
    py_flat = py_data.flatten()
    cpp_flat = cpp_data.flatten()

    shape_match = (py_shape == cpp_shape) or (len(py_flat) == len(cpp_flat))

    if len(py_flat) != len(cpp_flat):
        # Cannot compare - different sizes
        return TensorComparison(
            name=name,
            py_node_id=py_meta.get("node_id", -1),
            cpp_node_id=cpp_meta.get("node_id", -1),
            py_shape=py_shape,
            cpp_shape=cpp_shape,
            shape_match=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            rel_max_diff=float("inf"),
            first_diff_idx=-1,
            py_values_at_diff=[],
            cpp_values_at_diff=[],
        )

    # Compute differences
    diff = np.abs(py_flat - cpp_flat)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Relative difference (normalized by tensor magnitude)
    py_mag = float(np.max(np.abs(py_flat)))
    cpp_mag = float(np.max(np.abs(cpp_flat)))
    tensor_mag = max(py_mag, cpp_mag, 1e-10)
    rel_max_diff = max_diff / tensor_mag

    # Find first significant difference
    threshold = max(1e-5, tensor_mag * 1e-5)
    sig_diff_indices = np.where(diff > threshold)[0]

    if len(sig_diff_indices) > 0:
        first_diff_idx = int(sig_diff_indices[0])
        # Get values around the difference
        start = max(0, first_diff_idx - 2)
        end = min(len(py_flat), first_diff_idx + 5)
        py_vals = py_flat[start:end].tolist()
        cpp_vals = cpp_flat[start:end].tolist()
    else:
        first_diff_idx = -1
        py_vals = []
        cpp_vals = []

    return TensorComparison(
        name=name,
        py_node_id=py_meta.get("node_id", -1),
        cpp_node_id=cpp_meta.get("node_id", -1),
        py_shape=py_shape,
        cpp_shape=cpp_shape,
        shape_match=shape_match,
        max_diff=max_diff,
        mean_diff=mean_diff,
        rel_max_diff=rel_max_diff,
        first_diff_idx=first_diff_idx,
        py_values_at_diff=py_vals,
        cpp_values_at_diff=cpp_vals,
    )


def print_comparison_report(comparisons: List[TensorComparison], verbose: bool = False):
    """Print a summary of tensor comparisons."""
    print("\n" + "=" * 80)
    print("TENSOR COMPARISON REPORT")
    print("=" * 80)

    # Summary stats
    total = len(comparisons)
    shape_mismatches = sum(1 for c in comparisons if not c.shape_match)
    large_diffs = sum(1 for c in comparisons if c.max_diff > 1e-4 and c.shape_match)
    perfect_matches = sum(1 for c in comparisons if c.max_diff < 1e-6 and c.shape_match)

    print(f"\nTotal tensors compared: {total}")
    print(f"Shape mismatches: {shape_mismatches}")
    print(f"Large differences (>1e-4): {large_diffs}")
    print(f"Perfect matches (<1e-6): {perfect_matches}")

    # Sort by max_diff descending
    sorted_comps = sorted(comparisons, key=lambda c: c.max_diff, reverse=True)

    # Print worst offenders
    print("\n" + "-" * 80)
    print("TOP DIFFERENCES:")
    print("-" * 80)

    for comp in sorted_comps[:10]:
        status = ""
        if not comp.shape_match:
            status = "SHAPE MISMATCH"
        elif comp.max_diff > 1e-3:
            status = "LARGE DIFF"
        elif comp.max_diff > 1e-5:
            status = "DIFF"
        else:
            status = "OK"

        print(f"\n[{status}] {comp.name}")
        print(f"  Py node: {comp.py_node_id}, C++ node: {comp.cpp_node_id}")
        print(f"  Py shape: {comp.py_shape}, C++ shape: {comp.cpp_shape}")
        print(f"  Max diff: {comp.max_diff:.2e}, Mean diff: {comp.mean_diff:.2e}")
        print(f"  Relative max diff: {comp.rel_max_diff:.2e}")

        if comp.first_diff_idx >= 0 and verbose:
            print(f"  First difference at index {comp.first_diff_idx}:")
            print(f"    Py:  {comp.py_values_at_diff}")
            print(f"    C++: {comp.cpp_values_at_diff}")

    # Find first major divergence
    print("\n" + "-" * 80)
    print("FIRST MAJOR DIVERGENCE:")
    print("-" * 80)

    # Sort by node_id to find temporal order
    by_node_id = sorted(comparisons, key=lambda c: c.py_node_id)
    for comp in by_node_id:
        if comp.max_diff > 1e-3:
            print(f"\nNode {comp.py_node_id}: {comp.name}")
            print(f"  Max diff: {comp.max_diff:.2e}")
            print(f"  This is likely where the divergence starts.")
            if comp.first_diff_idx >= 0:
                print(f"  First difference at index {comp.first_diff_idx}:")
                print(f"    Py:  {comp.py_values_at_diff}")
                print(f"    C++: {comp.cpp_values_at_diff}")
            break
    else:
        print("No major divergence found (all differences < 1e-3)")


def main():
    parser = argparse.ArgumentParser(description="Compare Python and C++ tensor traces")
    parser.add_argument("--py-dir", type=Path, default=Path("/tmp/pet_debug/py"),
                        help="Python trace directory")
    parser.add_argument("--cpp-dir", type=Path, default=Path("/tmp/pet_debug/cpp"),
                        help="C++ trace directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed diff values")
    args = parser.parse_args()

    print(f"Loading Python tensors from {args.py_dir}...")
    py_tensors = load_py_tensors(args.py_dir)
    print(f"Loaded {len(py_tensors)} Python tensors")

    print(f"Loading C++ tensors from {args.cpp_dir}...")
    cpp_tensors = load_cpp_tensors(args.cpp_dir)
    print(f"Loaded {len(cpp_tensors)} C++ tensors")

    if not py_tensors:
        print("No Python tensors found. Run debug_pet_trace.py first.")
        return

    if not cpp_tensors:
        print("No C++ tensors found. Run C++ test with debug mode first.")
        return

    # Compare all Python tensors that have C++ matches
    comparisons = []
    matched = 0
    unmatched = []

    for py_name, (py_data, py_meta) in py_tensors.items():
        cpp_name = find_matching_tensor(py_name, cpp_tensors)
        if cpp_name:
            cpp_data, cpp_meta = cpp_tensors[cpp_name]
            comp = compare_tensors(py_data, py_meta, cpp_data, cpp_meta, py_name)
            comparisons.append(comp)
            matched += 1
        else:
            unmatched.append(py_name)

    print(f"\nMatched {matched}/{len(py_tensors)} Python tensors")
    if unmatched:
        print(f"Unmatched Python tensors: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    # Print report
    print_comparison_report(comparisons, verbose=args.verbose)


if __name__ == "__main__":
    main()
