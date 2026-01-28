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
from export_pytorch.export_pet_full import PETFullModel, get_pet_model

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
        description="Export PET-MAD model to GGUF with computation graph"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="pet-auto.gguf",
        help="Output GGUF file path",
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

    print("Loading PET-MAD model...")
    pet = get_pet_model()
    pet.eval()

    hypers = pet.hypers
    d_pet = hypers['d_pet'] if isinstance(hypers, dict) else hypers.D_PET
    cutoff = hypers.get('cutoff', 4.5) if isinstance(hypers, dict) else 4.5
    cutoff_width = hypers.get('cutoff_width', 0.2) if isinstance(hypers, dict) else 0.2

    print(f"  d_pet={d_pet}, cutoff={cutoff}, cutoff_width={cutoff_width}")
    print(f"  Export dimensions: n_atoms={n_atoms}, max_neighbors={max_neighbors}")

    # Create wrapper with full computation path
    wrapper = PETFullModel(pet, n_atoms=n_atoms, max_neighbors=max_neighbors, d_pet=d_pet)
    wrapper.eval()

    # Create test inputs for tracing
    torch.manual_seed(42)
    species = torch.zeros(n_atoms, dtype=torch.long)
    neighbor_species = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    edge_vectors = torch.randn(n_atoms, max_neighbors, 3)
    edge_distances = torch.rand(n_atoms, max_neighbors) * 3.0
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)
    cutoff_factors = torch.ones(n_atoms, max_neighbors)
    reverse_neighbor_index = torch.arange(n_atoms * max_neighbors, dtype=torch.long)

    example_inputs = (species, neighbor_species, edge_vectors, edge_distances,
                      padding_mask, reverse_neighbor_index, cutoff_factors)

    # Export via torch.export
    print("\nExporting graph via torch.export...")
    graph, weights = export_torch_model(
        wrapper, example_inputs,
        output_path=None,  # Don't save JSON yet
        input_names=["species", "neighbor_species", "edge_vectors", "edge_distances",
                      "padding_mask", "reverse_neighbor_index", "cutoff_factors"],
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

    # Get species mapping and composition energies
    species_keys = []
    species_indices = []
    for Z in range(1, 86):
        species_keys.append(Z)
        species_indices.append(Z - 1)

    composition_keys = []
    composition_values = []
    if hasattr(pet, 'additive_models') and len(pet.additive_models) > 0:
        comp_model = pet.additive_models[0]
        if hasattr(comp_model, 'model'):
            inner = comp_model.model
            if hasattr(inner, 'weights') and 'energy' in inner.weights:
                energy_weights = inner.weights['energy']
                block = energy_weights.block(0)
                t2i = inner.type_to_index
                for Z in range(1, 86):
                    idx = t2i[Z].item()
                    if idx >= 0 and idx < block.values.shape[0]:
                        composition_keys.append(Z)
                        composition_values.append(float(block.values[idx, 0].item()))

    # Write GGUF
    print(f"\nWriting GGUF to {args.output}...")
    writer = GGUFWriter()

    # Metadata
    writer.add_string("general.architecture", "pet-graph")
    writer.add_string("general.name", "PET-MAD")
    writer.add_string("general.version", "1.0.2")
    writer.add_float32("pet.cutoff", cutoff)
    writer.add_float32("pet.cutoff_width", cutoff_width)
    writer.add_int32("pet.d_pet", d_pet)

    # Species mapping: pairs of [Z, index, Z, index, ...]
    species_map = []
    for k, v in zip(species_keys, species_indices):
        species_map.extend([k, v])
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
