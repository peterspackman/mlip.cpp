#!/usr/bin/env python3
"""
Export PET-MAD model to GGUF format with embedded computation graph.

Produces a single .gguf file containing:
1. Model weights as GGUF tensors
2. Computation graph as JSON string in metadata ("graph.json")
3. Model hyperparameters, species mappings, composition energies

Usage:
    uv run python3 scripts/export_pytorch/export_pet_gguf.py -o pet-auto.gguf
"""

import json
import argparse
import struct
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_pytorch.fx_converter import export_torch_model, symbolize_dimensions
from export_pytorch.export_pet_full import (
    PETFullModel, load_pet_model, get_model_params,
    get_species_mapping, get_composition_energies, get_energy_scale,
)

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 4

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


@dataclass
class GGUFTensor:
    """GGUF tensor descriptor."""
    name: str
    shape: List[int]
    dtype: int
    data: bytes
    offset: int = 0


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self):
        self.metadata: Dict[str, Tuple[int, Any]] = {}
        self.tensors: List[GGUFTensor] = []

    def add_string(self, key: str, value: str):
        self.metadata[key] = (GGUF_TYPE_STRING, value)

    def add_int32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_INT32, value)

    def add_uint32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_UINT32, value)

    def add_float32(self, key: str, value: float):
        self.metadata[key] = (GGUF_TYPE_FLOAT32, value)

    def add_array_int32(self, key: str, values: List[int]):
        self.metadata[key] = (GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, values))

    def add_array_float32(self, key: str, values: List[float]):
        self.metadata[key] = (GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, values))

    def add_tensor(self, name: str, tensor: torch.Tensor, transpose_2d: bool = True):
        """Add a tensor to the GGUF file.

        Args:
            name: Tensor name
            tensor: PyTorch tensor
            transpose_2d: If True, transpose 2D weight matrices for GGML MUL_MAT
        """
        # Convert to float32 if needed
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()

        # Transpose 2D weight matrices for GGML MUL_MAT compatibility
        # GGML MUL_MAT: C = A @ B where A is [out, in] -> need [in, out] in GGML
        if transpose_2d and tensor.dim() == 2:
            tensor = tensor.T.contiguous()

        # Get shape in GGML format (reversed PyTorch shape)
        shape = list(tensor.shape)

        # Determine dtype
        if tensor.dtype == torch.float32:
            dtype = GGML_TYPE_F32
        elif tensor.dtype == torch.int32:
            dtype = GGML_TYPE_I32
        else:
            raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

        # Convert to bytes
        data = tensor.detach().contiguous().numpy().tobytes()

        self.tensors.append(GGUFTensor(
            name=name,
            shape=shape,
            dtype=dtype,
            data=data,
        ))

    def write(self, path: str):
        """Write GGUF file."""
        with open(path, "wb") as f:
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Write metadata
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))

                if vtype == GGUF_TYPE_STRING:
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr = value
                    f.write(struct.pack("<I", elem_type))
                    f.write(struct.pack("<Q", len(arr)))
                    if elem_type == GGUF_TYPE_INT32:
                        for v in arr:
                            f.write(struct.pack("<i", v))
                    elif elem_type == GGUF_TYPE_FLOAT32:
                        for v in arr:
                            f.write(struct.pack("<f", v))

            # Write tensor info
            for tensor in self.tensors:
                self._write_string(f, tensor.name)
                f.write(struct.pack("<I", len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", tensor.dtype))
                f.write(struct.pack("<Q", tensor.offset))

            # Align to 32 bytes
            current_pos = f.tell()
            alignment = 32
            padding = (alignment - (current_pos % alignment)) % alignment
            f.write(b"\x00" * padding)
            data_offset = f.tell()

            # Update tensor offsets and write data
            current_offset = 0
            for tensor in self.tensors:
                tensor.offset = current_offset
                current_offset += len(tensor.data)
                # Align each tensor to 32 bytes
                current_offset = (current_offset + 31) // 32 * 32

            # Rewrite tensor info with correct offsets
            f.seek(0)
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Re-write metadata (same as before)
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                if vtype == GGUF_TYPE_STRING:
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr = value
                    f.write(struct.pack("<I", elem_type))
                    f.write(struct.pack("<Q", len(arr)))
                    if elem_type == GGUF_TYPE_INT32:
                        for v in arr:
                            f.write(struct.pack("<i", v))
                    elif elem_type == GGUF_TYPE_FLOAT32:
                        for v in arr:
                            f.write(struct.pack("<f", v))

            # Re-write tensor info with updated offsets
            for tensor in self.tensors:
                self._write_string(f, tensor.name)
                f.write(struct.pack("<I", len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", tensor.dtype))
                f.write(struct.pack("<Q", tensor.offset))

            # Seek to data section and write tensor data
            f.seek(data_offset)
            for tensor in self.tensors:
                f.write(tensor.data)
                # Pad to 32-byte alignment
                padding = (32 - (len(tensor.data) % 32)) % 32
                f.write(b"\x00" * padding)

    def _write_string(self, f, s: str):
        """Write a GGUF string (length-prefixed)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)


def main():
    parser = argparse.ArgumentParser(
        description="Export PET model to GGUF with computation graph"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="pet-auto.gguf",
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--model", type=str, default="pet-mad-1.0.2",
        help="Model name: 'pet-mad-1.0.2' (legacy) or upet name like 'pet-mad-s'",
    )
    parser.add_argument(
        "--forces", action="store_true",
        help="Export with forces support (manual attention, in-graph distance/cutoff)",
    )
    parser.add_argument(
        "--n-atoms", type=int, default=7,
        help="Export atoms (use primes to avoid collisions with model constants)",
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=11,
        help="Export neighbors (use primes to avoid collisions with model constants)",
    )
    args = parser.parse_args()

    n_atoms = args.n_atoms
    max_neighbors = args.max_neighbors

    print(f"Loading PET model: {args.model}...")
    pet = load_pet_model(args.model)
    pet.eval()

    params = get_model_params(pet)
    d_pet = params['d_pet']
    cutoff = params['cutoff']
    cutoff_width = params['cutoff_width']
    cutoff_function = params['cutoff_function']
    num_neighbors_adaptive = params['num_neighbors_adaptive']

    print(f"  d_pet={d_pet}, cutoff={cutoff}, cutoff_width={cutoff_width}")
    print(f"  cutoff_function={cutoff_function}, num_neighbors_adaptive={num_neighbors_adaptive}")
    print(f"  Export dimensions: n_atoms={n_atoms}, max_neighbors={max_neighbors}")
    print(f"  Forces mode: {args.forces}")

    # Create wrapper with full computation path
    wrapper = PETFullModel(
        pet, n_atoms=n_atoms, max_neighbors=max_neighbors, d_pet=d_pet,
        forces=args.forces, cutoff=cutoff, cutoff_width=cutoff_width,
        cutoff_function=cutoff_function,
    )
    wrapper.eval()

    # Create test inputs for tracing
    torch.manual_seed(42)
    species = torch.zeros(n_atoms, dtype=torch.long)
    neighbor_species = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    edge_vectors = torch.randn(n_atoms, max_neighbors, 3)
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)
    reverse_neighbor_index = torch.arange(n_atoms * max_neighbors, dtype=torch.long)

    if args.forces:
        cutoff_values_input = torch.full((n_atoms, max_neighbors), cutoff)
        example_inputs = (species, neighbor_species, edge_vectors,
                         padding_mask, reverse_neighbor_index, cutoff_values_input)
        input_names = ["species", "neighbor_species", "edge_vectors",
                       "padding_mask", "reverse_neighbor_index", "cutoff_values"]
    else:
        edge_distances = torch.rand(n_atoms, max_neighbors) * 3.0
        cutoff_factors = torch.ones(n_atoms, max_neighbors)
        example_inputs = (species, neighbor_species, edge_vectors, edge_distances,
                         padding_mask, reverse_neighbor_index, cutoff_factors)
        input_names = ["species", "neighbor_species", "edge_vectors", "edge_distances",
                       "padding_mask", "reverse_neighbor_index", "cutoff_factors"]

    # Export via torch.export
    print("\nExporting graph via torch.export...")
    graph, weights = export_torch_model(
        wrapper, example_inputs,
        output_path=None,
        input_names=input_names,
        input_dtypes={
            "species": "i32",
            "neighbor_species": "i32",
            "reverse_neighbor_index": "i32",
        },
        strict=False,
    )
    print(f"  Graph: {len(graph.nodes)} nodes, {len(weights)} weights")

    # Symbolize dimensions for dynamic shapes
    print("Symbolizing dimensions...")
    model_constants = {1, 3, 4, 8, 32, 128, 256, 512, 768, d_pet}
    protected = model_constants - {n_atoms, max_neighbors,
                                   n_atoms * max_neighbors,
                                   max_neighbors + 1,
                                   n_atoms * (max_neighbors + 1)}
    graph = symbolize_dimensions(graph, {
        "n_atoms": n_atoms,
        "max_neighbors": max_neighbors,
    }, protected_values=protected)

    graph_json = json.dumps(graph.to_dict())
    print(f"  Symbolized graph: {len(graph_json)} bytes")

    # Get species mapping, composition energies, and energy scale
    species_to_index = get_species_mapping(pet)
    composition_energies = get_composition_energies(pet)
    energy_scale = get_energy_scale(pet)

    print(f"  Species mapped: {len(species_to_index)}")
    print(f"  Composition energies: {len(composition_energies)}")
    print(f"  Energy scale: {energy_scale}")
    if energy_scale == 1.0:
        print("  Warning: energy_scale is 1.0 - verify this is correct for your model")

    # Validate composition energies
    composition_keys = list(composition_energies.keys())
    composition_values = list(composition_energies.values())
    assert len(composition_keys) == len(composition_values), \
        f"Composition keys ({len(composition_keys)}) and values ({len(composition_values)}) mismatch"

    # Write GGUF
    print(f"\nWriting GGUF to {args.output}...")
    writer = GGUFWriter()

    # General metadata
    writer.add_string("general.architecture", "pet-graph")
    writer.add_string("general.name", args.model)
    writer.add_string("general.version", "1.0.2")

    # Model hyperparameters
    writer.add_float32("pet.cutoff", cutoff)
    writer.add_float32("pet.cutoff_width", cutoff_width)
    writer.add_int32("pet.d_pet", d_pet)
    writer.add_float32("pet.energy_scale", energy_scale)
    writer.add_string("pet.cutoff_function", cutoff_function)
    writer.add_int32("pet.forces_mode", 1 if args.forces else 0)
    writer.add_float32("pet.num_neighbors_adaptive",
                       float(num_neighbors_adaptive) if num_neighbors_adaptive is not None else 0.0)

    # Species mapping: pairs of [Z, index, Z, index, ...]
    species_map = []
    for Z, idx in sorted(species_to_index.items()):
        species_map.extend([Z, idx])
    writer.add_array_int32("pet.species_map", species_map)

    # Composition energies
    if composition_keys:
        writer.add_array_int32("pet.composition_keys", composition_keys)
        writer.add_array_float32("pet.composition_values", composition_values)

    # Computation graph as JSON string
    writer.add_string("graph.json", graph_json)

    # Weight shapes as JSON (for loader to reconstruct tensors)
    weight_shapes = {name: list(t.shape) for name, t in weights.items()}
    writer.add_string("graph.weight_shapes", json.dumps(weight_shapes))

    # Add weight tensors (no transpose - graph handles layout)
    print(f"Adding {len(weights)} weight tensors...")
    for name, tensor in weights.items():
        writer.add_tensor(name, tensor, transpose_2d=False)

    writer.write(args.output)

    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done! {file_size:.1f} MB, {len(graph.nodes)} graph nodes, {len(weights)} weights")


if __name__ == "__main__":
    main()
