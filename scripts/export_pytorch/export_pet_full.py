#!/usr/bin/env python3
"""Export complete PET-MAD model with neighbor list inputs to GIR format.

This creates a traceable wrapper that uses the actual GNN layers:
1. Input: species, neighbor_species, edge_features (neighbor list format)
2. Embedding lookups for nodes and neighbors
3. GNN layers with proper message passing
4. Energy head MLP
5. Output: atomic energies [n_atoms]
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from export_pytorch.fx_converter import export_torch_model, symbolize_dimensions


def get_pet_model():
    """Get the PET model."""
    from pet_mad._models import get_pet_mad
    model = get_pet_mad(version="1.0.2")
    return model.module.model


class PETFullModel(torch.nn.Module):
    """Full PET energy computation using actual GNN layers.

    Inputs:
        species: [n_atoms] - atomic species indices
        neighbor_species: [n_atoms, max_neighbors] - neighbor species indices
        edge_vectors: [n_atoms, max_neighbors, 3] - edge vectors (dx, dy, dz)
        edge_distances: [n_atoms, max_neighbors] - edge distances
        padding_mask: [n_atoms, max_neighbors] - True for valid neighbors
        reverse_neighbor_index: [n_atoms * max_neighbors] - index for reverse edges

    Output:
        atomic_energies: [n_atoms] - per-atom energy predictions
    """

    def __init__(self, pet_model, n_atoms: int, max_neighbors: int, d_pet: int):
        super().__init__()

        # Store dimensions for tracing
        self.n_atoms = n_atoms
        self.max_neighbors = max_neighbors
        self.d_pet = d_pet

        # Node embeddings - one per GNN layer
        self.node_embedders = pet_model.node_embedders

        # Neighbor species embedding (top-level)
        self.neighbor_embedder = pet_model.edge_embedder

        # GNN layers (CartesianTransformer)
        self.gnn_layers = pet_model.gnn_layers

        # Node energy heads and final layers (one per GNN layer)
        self.node_energy_heads = pet_model.node_heads['energy']
        self.node_final_layers = torch.nn.ModuleList([
            pet_model.node_last_layers['energy'][i]['energy___0']
            for i in range(len(pet_model.gnn_layers))
        ])

        # Edge energy heads and final layers (one per GNN layer)
        self.edge_energy_heads = pet_model.edge_heads['energy']
        self.edge_final_layers = torch.nn.ModuleList([
            pet_model.edge_last_layers['energy'][i]['energy___0']
            for i in range(len(pet_model.gnn_layers))
        ])

    def forward(self, species, neighbor_species, edge_vectors, edge_distances,
                padding_mask, reverse_neighbor_index, cutoff_factors):
        """
        Args:
            species: [n_atoms] - species indices (int64)
            neighbor_species: [n_atoms, max_neighbors] - neighbor species (int64)
            edge_vectors: [n_atoms, max_neighbors, 3] - edge vectors
            edge_distances: [n_atoms, max_neighbors] - edge distances
            padding_mask: [n_atoms, max_neighbors] - True for valid neighbors
            reverse_neighbor_index: [n_atoms * max_neighbors] - reverse edge indices
            cutoff_factors: [n_atoms, max_neighbors] - cutoff weights

        Returns:
            atomic_energies: [n_atoms]
        """
        # Initial neighbor species embeddings
        neighbor_embeds_flat = self.neighbor_embedder(neighbor_species.flatten())
        input_messages = neighbor_embeds_flat.view(self.n_atoms, self.max_neighbors, self.d_pet)

        # Initialize atomic energies accumulator
        atomic_energies = species.new_zeros(self.n_atoms, dtype=torch.float32)

        # Process through GNN layers
        for gnn_idx, (node_embedder, gnn_layer) in enumerate(
            zip(self.node_embedders, self.gnn_layers)
        ):
            # Get node embeddings for this layer
            input_node_embeddings = node_embedder(species)

            # Run GNN layer
            output_node, output_edge = gnn_layer(
                input_node_embeddings,
                input_messages,
                neighbor_species,
                edge_vectors,
                padding_mask,
                edge_distances,
                cutoff_factors,
                use_manual_attention=False
            )

            # Node energy readout
            node_feat = self.node_energy_heads[gnn_idx](output_node)  # [n_atoms, 128]
            node_e = self.node_final_layers[gnn_idx](node_feat)  # [n_atoms, 1]

            # Edge energy readout
            edge_feat = self.edge_energy_heads[gnn_idx](output_edge)  # [n_atoms, max_neighbors, 128]
            edge_e = self.edge_final_layers[gnn_idx](edge_feat)  # [n_atoms, max_neighbors, 1]
            # Mask out padded edges and apply cutoff
            # padding_mask is True for valid neighbors
            edge_e_masked = torch.where(
                padding_mask.unsqueeze(-1),
                edge_e,
                torch.zeros_like(edge_e)
            )
            # Apply cutoff factors and sum over neighbors
            edge_e_sum = (edge_e_masked.squeeze(-1) * cutoff_factors).sum(dim=1)  # [n_atoms]

            # Accumulate both node and edge contributions
            atomic_energies = atomic_energies + node_e.squeeze(-1) + edge_e_sum

            # Message passing: prepare input for next layer
            # Reverse the messages using reverse_neighbor_index
            flat_output = output_edge.reshape(
                self.n_atoms * self.max_neighbors, self.d_pet
            )
            reversed_messages = flat_output[reverse_neighbor_index].reshape(
                self.n_atoms, self.max_neighbors, self.d_pet
            )
            # Average forward and reverse messages
            input_messages = 0.5 * (input_messages + reversed_messages)

        return atomic_energies


def compute_reverse_neighbor_index(n_atoms: int, max_neighbors: int,
                                   centers: list, neighbors: list) -> torch.Tensor:
    """Compute the reverse neighbor index for message passing.

    For each edge (i -> j), find the index of the reverse edge (j -> i).
    """
    # Build a lookup: (center, neighbor) -> flat_index
    edge_to_idx = {}
    for flat_idx, (c, n) in enumerate(zip(centers, neighbors)):
        edge_to_idx[(c, n)] = flat_idx

    # For each edge, find its reverse
    reverse_idx = torch.zeros(n_atoms * max_neighbors, dtype=torch.long)

    # Build per-atom neighbor slots
    slot_counts = [0] * n_atoms
    atom_neighbors = [[] for _ in range(n_atoms)]
    for c, n in zip(centers, neighbors):
        atom_neighbors[c].append(n)

    for atom_i in range(n_atoms):
        for slot_j, neighbor_j in enumerate(atom_neighbors[atom_i]):
            flat_idx = atom_i * max_neighbors + slot_j
            # Find the reverse edge (neighbor_j -> atom_i)
            reverse_key = (neighbor_j, atom_i)
            if reverse_key in edge_to_idx:
                # Find which slot atom_i is in for neighbor_j
                for slot_k, n in enumerate(atom_neighbors[neighbor_j]):
                    if n == atom_i:
                        reverse_flat_idx = neighbor_j * max_neighbors + slot_k
                        reverse_idx[flat_idx] = reverse_flat_idx
                        break

    return reverse_idx


def export_pet_full(
    output_dir: Path = Path("/tmp/pet_full_export"),
    n_atoms: int = 7,
    max_neighbors: int = 11
):
    """Export full PET computation path with neighbor list inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PET model...")
    pet = get_pet_model()
    pet.eval()

    hypers = pet.hypers
    d_pet = hypers['d_pet'] if isinstance(hypers, dict) else hypers.D_PET

    print(f"d_pet: {d_pet}")
    print(f"n_atoms: {n_atoms}, max_neighbors: {max_neighbors}")

    # Create wrapper using actual GNN layers
    wrapper = PETFullModel(pet, n_atoms=n_atoms, max_neighbors=max_neighbors, d_pet=d_pet)
    wrapper.eval()

    # Create test inputs
    torch.manual_seed(42)
    species = torch.zeros(n_atoms, dtype=torch.long)  # All species 0
    neighbor_species = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    edge_vectors = torch.randn(n_atoms, max_neighbors, 3)
    edge_distances = torch.rand(n_atoms, max_neighbors) * 3.0
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)
    cutoff_factors = torch.ones(n_atoms, max_neighbors)

    # Simple reverse index for test (identity for now)
    reverse_neighbor_index = torch.arange(n_atoms * max_neighbors, dtype=torch.long)

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        expected_output = wrapper(
            species, neighbor_species, edge_vectors, edge_distances,
            padding_mask, reverse_neighbor_index, cutoff_factors
        )

    print(f"Output shape: {expected_output.shape}")
    print(f"Atomic energies: {expected_output}")
    print(f"Total energy: {expected_output.sum().item():.6f}")

    # Export via torch.export (handles dynamic operations like torch.empty)
    print("\nExporting via torch.export...")
    try:
        graph, weights = export_torch_model(
            wrapper,
            (species, neighbor_species, edge_vectors, edge_distances,
             padding_mask, reverse_neighbor_index, cutoff_factors),
            output_dir / "pet_full.json",
            input_names=["species", "neighbor_species", "edge_vectors", "edge_distances",
                        "padding_mask", "reverse_neighbor_index", "cutoff_factors"],
            input_dtypes={
                "species": "i32",
                "neighbor_species": "i32",
                "reverse_neighbor_index": "i32",
            },
            strict=False,  # Allow dynamic operations
        )

        # Symbolize dynamic dimensions so the graph can be used with any system size
        print("\nSymbolizing dimensions...")
        # Protect known model constants from being symbolized even if they
        # happen to match n_atoms or max_neighbors.
        # NOTE: Export dimensions (n_atoms, max_neighbors) should be chosen to
        # avoid collisions with model constants. Use --n-atoms=7 --max-neighbors=11
        # (primes that don't appear as model dimensions).
        model_constants = {1, 3, 4, 8, 32, 128, 256, 512, 768, d_pet}
        # Don't protect values that are our actual export dimensions
        protected = model_constants - {n_atoms, max_neighbors,
                                       n_atoms * max_neighbors,
                                       max_neighbors + 1,
                                       n_atoms * (max_neighbors + 1)}
        graph = symbolize_dimensions(graph, {
            "n_atoms": n_atoms,
            "max_neighbors": max_neighbors,
        }, protected_values=protected)

        # Re-save with symbolized dimensions
        with open(output_dir / "pet_full.json", "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"Saved symbolized graph with dynamic dimensions")

        # Save weights
        print(f"\nSaving {len(weights)} weights...")
        for name, tensor in weights.items():
            data = tensor.detach().cpu().numpy()
            filepath = output_dir / f"{name}.bin"
            data.astype(np.float32).tofile(filepath)

        # Save inputs
        species.numpy().astype(np.int32).tofile(output_dir / "input_species.bin")
        neighbor_species.numpy().astype(np.int32).tofile(output_dir / "input_neighbor_species.bin")
        edge_vectors.numpy().astype(np.float32).tofile(output_dir / "input_edge_vectors.bin")
        edge_distances.numpy().astype(np.float32).tofile(output_dir / "input_edge_distances.bin")
        padding_mask.numpy().astype(np.bool_).tofile(output_dir / "input_padding_mask.bin")
        reverse_neighbor_index.numpy().astype(np.int32).tofile(output_dir / "input_reverse_neighbor_index.bin")
        cutoff_factors.numpy().astype(np.float32).tofile(output_dir / "input_cutoff_factors.bin")

        # Save expected output
        expected_output.numpy().astype(np.float32).tofile(output_dir / "expected_output.bin")

        # Get species mapping and composition energies
        species_to_index = {}
        composition_energies = {}

        # Default: atomic numbers 1-85 map to indices 0-84
        for Z in range(1, 86):
            species_to_index[Z] = Z - 1

        # Get composition energies from additive models
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
                            composition_energies[Z] = float(block.values[idx, 0].item())

        # Get cutoff from hyperparameters
        cutoff = hypers.get('cutoff', 4.5) if isinstance(hypers, dict) else 4.5
        cutoff_width = hypers.get('cutoff_width', 0.2) if isinstance(hypers, dict) else 0.2

        # Save metadata
        metadata = {
            "n_atoms": n_atoms,
            "max_neighbors": max_neighbors,
            "d_pet": d_pet,
            "num_nodes": len(graph.nodes),
            "num_weights": len(weights),
            "expected_total_energy": expected_output.sum().item(),
            "cutoff": float(cutoff),
            "cutoff_width": float(cutoff_width),
            "species_to_index": species_to_index,
            "composition_energies": composition_energies,
            "weights": {name: list(t.shape) for name, t in weights.items()}
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nAll files saved to {output_dir}")
        print(f"Graph: {len(graph.nodes)} nodes")

        return graph, weights

    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export PET model to GIR format")
    parser.add_argument("--output", "-o", type=str, default="/tmp/pet_full_export",
                        help="Output directory")
    parser.add_argument("--n-atoms", type=int, default=7,
                        help="Number of atoms (use primes like 7 to avoid collision with model constants)")
    parser.add_argument("--max-neighbors", type=int, default=11,
                        help="Maximum neighbors per atom (use primes like 11 to avoid collision with model constants)")
    args = parser.parse_args()

    export_pet_full(
        output_dir=Path(args.output),
        n_atoms=args.n_atoms,
        max_neighbors=args.max_neighbors
    )
