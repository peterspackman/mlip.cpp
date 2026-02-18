#!/usr/bin/env python3
"""Export PET models (pet-mad, upet) with neighbor list inputs to GIR format.

Supports:
- Legacy pet-mad-1.0.2 via pet_mad package
- Any upet model (pet-mad-s, pet-omat-l, pet-spice-s, etc.) via metatrain

Usage:
  # Legacy pet-mad
  uv run scripts/export_pytorch/export_pet_full.py --model pet-mad-1.0.2 -o /tmp/pet_export

  # upet models
  uv run scripts/export_pytorch/export_pet_full.py --model pet-mad-s -o /tmp/pet_mad_s_export

  # With forces (manual attention, in-graph distance/cutoff)
  uv run scripts/export_pytorch/export_pet_full.py --model pet-mad-s --forces -o /tmp/pet_forces
"""

import json
import math
import argparse
import re
import torch
import numpy as np
import warnings
from pathlib import Path
import sys
from packaging.version import Version
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from export_pytorch.fx_converter import export_torch_model, symbolize_dimensions


# --- Model Loading ---

def resolve_upet_checkpoint_name(model_base: str, size: str) -> str:
    """Resolve checkpoint filename for a upet model across API versions."""
    from huggingface_hub import list_repo_files

    version = None
    try:
        from upet._models import upet_get_version_to_load as _resolve_version  # type: ignore
        version = _resolve_version(model_base, size)
    except Exception:
        try:
            # Compatibility with older naming on some installations.
            from upet._models import get_version_to_load as _resolve_version  # type: ignore
            version = _resolve_version(model_base, size)
        except Exception:
            version = None

    if version is not None:
        return f"{model_base}-{size}-v{version}.ckpt"

    pattern = re.compile(rf"^models/{re.escape(model_base)}-{re.escape(size)}-v(.+)\.ckpt$")
    candidates = []
    for path in list_repo_files(repo_id="lab-cosmo/upet"):
        match = pattern.match(path)
        if match:
            try:
                candidates.append((Version(match.group(1)), path.split("/", 1)[1]))
            except Exception:
                continue

    if not candidates:
        raise RuntimeError(
            f"Could not resolve checkpoint for {model_base}-{size} "
            "from upet API or Hugging Face file listing."
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def load_pet_model(model_name: str):
    """Load a raw PET model by name.

    Args:
        model_name: One of:
            - "pet-mad-1.0.2": Legacy pet-mad package
            - "pet-xxx-{size}": upet model (e.g., "pet-mad-s", "pet-omat-l")

    Returns:
        The raw PET model (with gnn_layers, node_embedders, etc.)
    """
    if model_name == "pet-mad-1.0.2":
        from pet_mad._models import get_pet_mad
        atomistic = get_pet_mad(version="1.0.2")
        return atomistic.module.model

    # Parse upet model name: "pet-xxx-{size}" -> model="pet-xxx", size="{size}"
    valid_sizes = {"xs", "s", "m", "l", "xl"}
    parts = model_name.rsplit("-", 1)
    if len(parts) != 2 or parts[1] not in valid_sizes:
        raise ValueError(
            f"Invalid model name '{model_name}'. Expected format: "
            f"'pet-xxx-size' where size is one of {valid_sizes}, "
            f"or 'pet-mad-1.0.2' for legacy."
        )
    model_base, size = parts[0], parts[1]

    from huggingface_hub import hf_hub_download
    from metatrain.utils.io import load_model as load_metatrain_model

    path = None
    model_string = None
    try:
        model_string = resolve_upet_checkpoint_name(model_base, size)
        print(f"Downloading {model_string} from HuggingFace...")
        path = hf_hub_download(
            repo_id="lab-cosmo/upet",
            filename=model_string,
            subfolder="models",
        )
    except Exception as e:
        # Offline/cached fallback: resolve latest matching checkpoint from local HF cache.
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--lab-cosmo--upet" / "snapshots"
        pattern = f"{model_base}-{size}-v*.ckpt"
        candidates = sorted(cache_root.glob(f"*/models/{pattern}"))
        if not candidates:
            raise RuntimeError(
                f"Failed to resolve {model_base}-{size} from HuggingFace and no cached "
                f"checkpoint found matching {pattern} under {cache_root}"
            ) from e

        def _ver_key(p: Path):
            stem = p.stem  # pet-oam-l-v0.1.0
            v = stem.rsplit("-v", 1)[-1]
            try:
                return Version(v)
            except Exception:
                return Version("0")

        path_obj = max(candidates, key=_ver_key)
        model_string = path_obj.name
        path = str(path_obj)
        print(f"Using cached checkpoint {model_string} at {path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pet_model = load_metatrain_model(path)

    return pet_model


def get_model_params(pet_model):
    """Extract model parameters from a PET model (handles both old and new formats).

    Returns:
        dict with keys: d_pet, cutoff, cutoff_width, cutoff_function, num_neighbors_adaptive
    """
    # Metatrain PET caches these as direct attributes
    if hasattr(pet_model, 'd_pet'):
        return {
            'd_pet': pet_model.d_pet,
            'cutoff': getattr(pet_model, 'cutoff', 4.5),
            'cutoff_width': getattr(pet_model, 'cutoff_width', 0.2),
            'cutoff_function': getattr(pet_model, 'cutoff_function', 'Cosine').lower(),
            'num_neighbors_adaptive': getattr(pet_model, 'num_neighbors_adaptive', None),
        }

    # Legacy pet-mad format
    hypers = pet_model.hypers
    if isinstance(hypers, dict):
        return {
            'd_pet': hypers.get('d_pet', 256),
            'cutoff': hypers.get('cutoff', 4.5),
            'cutoff_width': hypers.get('cutoff_width', 0.2),
            'cutoff_function': 'cosine',
            'num_neighbors_adaptive': None,
        }

    return {
        'd_pet': getattr(hypers, 'D_PET', 256),
        'cutoff': getattr(hypers, 'cutoff', 4.5),
        'cutoff_width': getattr(hypers, 'cutoff_width', 0.2),
        'cutoff_function': 'cosine',
        'num_neighbors_adaptive': None,
    }


def get_species_mapping(pet_model):
    """Get species-to-index mapping from a PET model."""
    # Metatrain PET stores atomic_types
    if hasattr(pet_model, 'atomic_types'):
        species_to_index = {}
        for idx, Z in enumerate(pet_model.atomic_types):
            species_to_index[int(Z)] = idx
        return species_to_index

    # Default: atomic numbers 1-85 map to indices 0-84
    return {Z: Z - 1 for Z in range(1, 86)}


def get_composition_energies(pet_model):
    """Extract composition energies from a PET model (if available)."""
    composition_energies = {}

    # Legacy pet-mad format with additive_models
    if hasattr(pet_model, 'additive_models') and len(pet_model.additive_models) > 0:
        try:
            comp_model = pet_model.additive_models[0]
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
        except Exception as e:
            print(f"Warning: could not extract composition energies: {e}")

    # Metatrain PET: composition energies are part of the training wrapper
    # and may not be accessible from the raw model. The exported AtomisticModel
    # includes them, but for our graph export we just skip them.
    # Forces are unaffected by composition energies (they're constant per type).

    return composition_energies


def get_energy_scale(pet_model) -> float:
    """Extract energy scale factor from a PET model's scaler (if available).

    The scaler multiplies raw model output to produce the final energy.
    For models without a scaler, returns 1.0.
    """
    if hasattr(pet_model, 'scaler'):
        scaler = pet_model.scaler
        if hasattr(scaler, 'model') and hasattr(scaler.model, 'scales'):
            if 'energy' in scaler.model.scales:
                scale_block = scaler.model.scales['energy'].block(0)
                return float(scale_block.values.item())
    return 1.0


# --- Model Wrapper ---

class PETFullModel(torch.nn.Module):
    """Full PET energy computation using actual GNN layers.

    Supports two featurization types:
    - "residual" (pet-mad-s): Per-layer energy accumulation, multiple node embedders
    - "feedforward" (pet-omad-s): combination_mlps between layers, final-only energy

    When forces=False (default):
        Inputs: species, neighbor_species, edge_vectors, edge_distances,
                padding_mask, reverse_neighbor_index, cutoff_factors
        Uses flash attention (fast, no backward).

    When forces=True:
        Inputs: species, neighbor_species, edge_vectors,
                padding_mask, reverse_neighbor_index
        Computes edge_distances and cutoff_factors in-graph (from edge_vectors).
        Uses manual attention (supports backward pass for force computation).

    Output:
        atomic_energies: [n_atoms] - per-atom energy predictions
    """

    def __init__(self, pet_model, n_atoms: int, max_neighbors: int, d_pet: int,
                 forces: bool = False, cutoff: float = 4.5, cutoff_width: float = 0.2,
                 cutoff_function: str = "cosine"):
        super().__init__()

        # Store dimensions for tracing
        self.n_atoms = n_atoms
        self.max_neighbors = max_neighbors
        self.d_pet = d_pet
        self.forces = forces
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.cutoff_function = cutoff_function

        # Detect featurization type
        self.featurizer_type = getattr(pet_model, 'featurizer_type', 'residual')
        self.num_readout_layers = getattr(pet_model, 'num_readout_layers', len(pet_model.gnn_layers))

        # Node embeddings
        self.node_embedders = pet_model.node_embedders

        # Neighbor species embedding (top-level)
        self.neighbor_embedder = pet_model.edge_embedder

        # GNN layers (CartesianTransformer)
        self.gnn_layers = pet_model.gnn_layers

        # Feedforward-specific: combination MLPs and norms
        if self.featurizer_type == 'feedforward':
            self.combination_mlps = pet_model.combination_mlps
            self.combination_norms = pet_model.combination_norms

        # Energy heads and final layers
        # For residual: one per GNN layer
        # For feedforward: one for final layer only (num_readout_layers=1)
        self.node_energy_heads = pet_model.node_heads['energy']
        self.node_final_layers = torch.nn.ModuleList([
            pet_model.node_last_layers['energy'][i]['energy___0']
            for i in range(self.num_readout_layers)
        ])

        self.edge_energy_heads = pet_model.edge_heads['energy']
        self.edge_final_layers = torch.nn.ModuleList([
            pet_model.edge_last_layers['energy'][i]['energy___0']
            for i in range(self.num_readout_layers)
        ])

    def _compute_cutoff_factors(self, edge_distances, cutoff_values=None):
        """Cutoff function computed in-graph for gradient flow.

        When cutoff_values is provided, uses per-pair cutoffs (for adaptive cutoff models).
        Otherwise uses self.cutoff (global cutoff).

        Supports both cosine and bump cutoff functions.
        """
        if cutoff_values is not None:
            cutoff = cutoff_values
        else:
            cutoff = self.cutoff

        scaled = torch.clamp(
            (edge_distances - (cutoff - self.cutoff_width)) / self.cutoff_width,
            0.0, 1.0
        )

        if self.cutoff_function == "bump":
            # Bump cutoff: 0.5 * (1 + tanh(1 / tan(pi * x)))
            # Rewrite as: 0.5 * (1 + tanh(cos(pi*x) / sin(pi*x)))
            # This avoids torch.tan which has no GGML equivalent.
            # Clamp away from 0 and 1 to avoid singularities
            scaled_safe = torch.clamp(scaled, min=1e-6, max=1.0 - 1e-6)
            angle = torch.tensor(math.pi) * scaled_safe
            return 0.5 * (1.0 + torch.tanh(torch.cos(angle) / torch.sin(angle)))
        else:
            # Cosine cutoff: 0.5 * (1 + cos(pi * x))
            return 0.5 * (1.0 + torch.cos(torch.tensor(math.pi) * scaled))

    def forward(self, species, neighbor_species, edge_vectors,
                *args):
        """Forward pass with variable signature based on forces mode.

        When forces=False: args = (edge_distances, padding_mask, reverse_neighbor_index, cutoff_factors)
        When forces=True:  args = (padding_mask, reverse_neighbor_index, cutoff_values)
        """
        if self.forces:
            padding_mask = args[0]
            reverse_neighbor_index = args[1]
            cutoff_values = args[2]  # per-pair cutoff radii [n_atoms, max_neighbors]

            # Compute distances from edge vectors (in-graph for gradient flow)
            # Use explicit multiply instead of ** 2 to avoid POW op
            edge_distances = torch.sqrt((edge_vectors * edge_vectors).sum(dim=-1))
            # Compute cutoff factors from distances and per-pair cutoffs
            cutoff_factors = self._compute_cutoff_factors(edge_distances, cutoff_values)
        else:
            edge_distances = args[0]
            padding_mask = args[1]
            reverse_neighbor_index = args[2]
            cutoff_factors = args[3]

        n_atoms = species.shape[0]
        max_neighbors = neighbor_species.shape[1]

        # Initial neighbor species embeddings
        neighbor_embeds_flat = self.neighbor_embedder(neighbor_species.flatten())
        input_messages = neighbor_embeds_flat.view(n_atoms, max_neighbors, self.d_pet)

        if self.featurizer_type == 'feedforward':
            return self._forward_feedforward(
                species, neighbor_species, edge_vectors, edge_distances,
                padding_mask, reverse_neighbor_index, cutoff_factors,
                input_messages, n_atoms, max_neighbors
            )
        else:
            return self._forward_residual(
                species, neighbor_species, edge_vectors, edge_distances,
                padding_mask, reverse_neighbor_index, cutoff_factors,
                input_messages, n_atoms, max_neighbors
            )

    def _forward_residual(self, species, neighbor_species, edge_vectors, edge_distances,
                          padding_mask, reverse_neighbor_index, cutoff_factors,
                          input_messages, n_atoms, max_neighbors):
        """Residual featurization: per-layer energy accumulation (pet-mad-s style)."""
        # Initialize atomic energies accumulator
        atomic_energies = species.new_zeros(n_atoms, dtype=torch.float32)

        # Process through GNN layers with per-layer energy readout
        for gnn_idx, (node_embedder, gnn_layer) in enumerate(
            zip(self.node_embedders, self.gnn_layers)
        ):
            # Get node embeddings for this layer
            input_node_embeddings = node_embedder(species)

            # Run GNN layer
            # Note: metatrain uses True=valid, False=padded convention
            # Our wrapper uses True=padded, False=valid, so we invert here
            output_node, output_edge = gnn_layer(
                input_node_embeddings,
                input_messages,
                neighbor_species,
                edge_vectors,
                ~padding_mask,  # Invert for metatrain convention
                edge_distances,
                cutoff_factors,
                use_manual_attention=self.forces
            )
            # Zero out padded edge positions (GNN may produce non-zero values)
            output_edge = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(output_edge),
                output_edge,
            )

            # Node energy readout
            node_feat = self.node_energy_heads[gnn_idx](output_node)
            node_e = self.node_final_layers[gnn_idx](node_feat)

            # Edge energy readout
            edge_feat = self.edge_energy_heads[gnn_idx](output_edge)
            edge_e = self.edge_final_layers[gnn_idx](edge_feat)
            # Mask out padded edges and apply cutoff
            # padding_mask: True=padded (invalid), False=valid
            edge_e_masked = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(edge_e),  # Zero out padded edges
                edge_e,                     # Keep valid edges
            )
            # Apply cutoff factors and sum over neighbors
            edge_e_sum = (edge_e_masked.squeeze(-1) * cutoff_factors).sum(dim=1)

            # Accumulate both node and edge contributions
            atomic_energies = atomic_energies + node_e.squeeze(-1) + edge_e_sum

            # Message passing: prepare input for next layer (simple average)
            flat_output = output_edge.reshape(
                n_atoms * max_neighbors, self.d_pet
            )
            reversed_messages = flat_output[reverse_neighbor_index].reshape(
                n_atoms, max_neighbors, self.d_pet
            )
            # Zero out padded positions (reverse_idx for padded slots may point to valid edges)
            reversed_messages = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(reversed_messages),
                reversed_messages,
            )
            input_messages = 0.5 * (input_messages + reversed_messages)

        return atomic_energies

    def _forward_feedforward(self, species, neighbor_species, edge_vectors, edge_distances,
                             padding_mask, reverse_neighbor_index, cutoff_factors,
                             input_messages, n_atoms, max_neighbors):
        """Feedforward featurization: combination_mlps between layers, final-only energy (pet-omad-s style)."""
        # Single node embedder used for all layers
        input_node_embeddings = self.node_embedders[0](species)

        # Zero out padded positions in initial edge embeddings
        input_messages = torch.where(
            padding_mask.unsqueeze(-1),
            torch.zeros_like(input_messages),
            input_messages,
        )

        # Process through GNN layers with combination MLPs
        for combination_norm, combination_mlp, gnn_layer in zip(
            self.combination_norms, self.combination_mlps, self.gnn_layers
        ):
            # Note: metatrain uses True=valid, False=padded convention
            output_node, output_edge = gnn_layer(
                input_node_embeddings,
                input_messages,
                neighbor_species,
                edge_vectors,
                ~padding_mask,  # Invert for metatrain convention
                edge_distances,
                cutoff_factors,
                use_manual_attention=self.forces
            )
            # Zero out padded edge positions (GNN may produce non-zero values)
            output_edge = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(output_edge),
                output_edge,
            )

            # Update node embeddings for next layer
            input_node_embeddings = output_node

            # Message passing with combination MLPs
            # Reverse the edge messages
            flat_output = output_edge.reshape(
                n_atoms * max_neighbors, self.d_pet
            )
            new_input_messages = flat_output[reverse_neighbor_index].reshape(
                n_atoms, max_neighbors, self.d_pet
            )
            # Zero out padded positions (reverse_idx for padded slots may point to valid edges)
            new_input_messages = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(new_input_messages),
                new_input_messages,
            )

            # Concatenate forward and reversed, apply norm + MLP
            concatenated = torch.cat([output_edge, new_input_messages], dim=-1)
            # Residual connection: input + output + combination_mlp(norm(concat))
            # Zero out the update for padded positions (mlp(norm(zeros)) is non-zero due to bias)
            update = output_edge + combination_mlp(combination_norm(concatenated))
            update = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros_like(update),
                update,
            )
            input_messages = input_messages + update

        # Energy readout from final features only (num_readout_layers=1)
        # Node energy
        node_feat = self.node_energy_heads[0](input_node_embeddings)
        node_e = self.node_final_layers[0](node_feat)

        # Edge energy
        edge_feat = self.edge_energy_heads[0](input_messages)
        edge_e = self.edge_final_layers[0](edge_feat)

        # Mask out padded edges and apply cutoff
        # padding_mask: True=padded (invalid), False=valid
        edge_e_masked = torch.where(
            padding_mask.unsqueeze(-1),
            torch.zeros_like(edge_e),  # Zero out padded edges
            edge_e,                     # Keep valid edges
        )
        edge_e_sum = (edge_e_masked.squeeze(-1) * cutoff_factors).sum(dim=1)

        # Total atomic energies
        atomic_energies = node_e.squeeze(-1) + edge_e_sum
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


def build_example_inputs(
    example_n_atoms: int,
    example_max_neighbors: int,
    cutoff: float,
    forces: bool,
) -> Tuple[Tuple[torch.Tensor, ...], List[str]]:
    """Build deterministic example inputs and input names for tracing/export."""
    torch.manual_seed(42)
    species = torch.zeros(example_n_atoms, dtype=torch.long)
    neighbor_species = torch.zeros(example_n_atoms, example_max_neighbors, dtype=torch.long)
    edge_vectors = torch.randn(example_n_atoms, example_max_neighbors, 3)
    padding_mask = torch.ones(example_n_atoms, example_max_neighbors, dtype=torch.bool)
    reverse_neighbor_index = torch.arange(example_n_atoms * example_max_neighbors, dtype=torch.long)

    if forces:
        cutoff_values = torch.full((example_n_atoms, example_max_neighbors), cutoff)
        example_inputs = (
            species,
            neighbor_species,
            edge_vectors,
            padding_mask,
            reverse_neighbor_index,
            cutoff_values,
        )
        input_names = [
            "species",
            "neighbor_species",
            "edge_vectors",
            "padding_mask",
            "reverse_neighbor_index",
            "cutoff_values",
        ]
    else:
        edge_distances = torch.rand(example_n_atoms, example_max_neighbors) * 3.0
        cutoff_factors = torch.ones(example_n_atoms, example_max_neighbors)
        example_inputs = (
            species,
            neighbor_species,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
        )
        input_names = [
            "species",
            "neighbor_species",
            "edge_vectors",
            "edge_distances",
            "padding_mask",
            "reverse_neighbor_index",
            "cutoff_factors",
        ]

    return example_inputs, input_names


def save_weights(weights: Dict[str, torch.Tensor], output_dir: Path) -> None:
    """Save all exported weights as float32 binary blobs."""
    print(f"\nSaving {len(weights)} weights...")
    for name, tensor in weights.items():
        filepath = output_dir / f"{name}.bin"
        tensor.detach().cpu().numpy().astype(np.float32).tofile(filepath)


def save_example_inputs(
    input_names: List[str],
    example_inputs: Tuple[torch.Tensor, ...],
    output_dir: Path,
) -> None:
    """Save tracing inputs in binary format used by local tooling."""
    for name, tensor in zip(input_names, example_inputs):
        path = output_dir / f"input_{name}.bin"
        if tensor.dtype in (torch.long, torch.int32, torch.int64):
            tensor.cpu().numpy().astype(np.int32).tofile(path)
        elif tensor.dtype == torch.bool:
            tensor.cpu().numpy().astype(np.bool_).tofile(path)
        else:
            tensor.cpu().numpy().astype(np.float32).tofile(path)


def save_exported_program(
    wrapper: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
) -> None:
    """Export and save a PyTorch ExportedProgram (.pt2)."""
    print(f"\nSaving compiled exported program to {output_path} ...")
    exported_program = torch.export.export(wrapper, example_inputs, strict=False)
    torch.export.save(exported_program, str(output_path))
    print("Saved compiled exported program.")


def build_metadata(
    example_n_atoms: int,
    example_max_neighbors: int,
    d_pet: int,
    graph,
    weights: Dict[str, torch.Tensor],
    expected_output: torch.Tensor,
    cutoff: float,
    cutoff_width: float,
    cutoff_function: str,
    num_neighbors_adaptive,
    forces: bool,
    model_name: str,
    featurizer_type: str,
    num_gnn_layers: int,
    num_readout_layers: int,
    species_to_index: Dict[int, int],
    composition_energies: Dict[int, float],
    energy_scale: float,
) -> Dict:
    """Assemble metadata payload persisted to metadata.json."""
    return {
        "example_n_atoms": example_n_atoms,
        "example_max_neighbors": example_max_neighbors,
        # Backward-compatible aliases for existing tooling.
        "n_atoms": example_n_atoms,
        "max_neighbors": example_max_neighbors,
        "d_pet": d_pet,
        "num_nodes": len(graph.nodes),
        "num_weights": len(weights),
        "expected_total_energy": expected_output.sum().item(),
        "cutoff": float(cutoff),
        "cutoff_width": float(cutoff_width),
        "cutoff_function": cutoff_function,
        "num_neighbors_adaptive": float(num_neighbors_adaptive) if num_neighbors_adaptive is not None else None,
        "forces": forces,
        "model_name": model_name,
        "featurizer_type": featurizer_type,
        "num_gnn_layers": num_gnn_layers,
        "num_readout_layers": num_readout_layers,
        "species_to_index": species_to_index,
        "composition_energies": composition_energies,
        "energy_scale": energy_scale,
        "weights": {name: list(t.shape) for name, t in weights.items()},
    }


# --- Export ---

def export_pet_full(
    output_dir: Path = Path("/tmp/pet_full_export"),
    example_n_atoms: int = 7,
    example_max_neighbors: int = 11,
    model_name: str = "pet-mad-1.0.2",
    forces: bool = False,
    save_compiled: bool = False,
    compiled_filename: str = "pet_full_exported.pt2",
):
    """Export full PET computation path with neighbor list inputs.

    Args:
        output_dir: Directory for output files
        example_n_atoms: Example atom count used only for tracing/export
        example_max_neighbors: Example max neighbors used only for tracing/export
        model_name: Model identifier (see load_pet_model docstring)
        forces: If True, export with manual attention and in-graph distance/cutoff
        save_compiled: If True, save a compiled torch.export program (.pt2)
        compiled_filename: File name for the compiled export artifact
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PET model: {model_name}...")
    pet = load_pet_model(model_name)
    pet.eval()

    params = get_model_params(pet)
    d_pet = params['d_pet']
    cutoff = params['cutoff']
    cutoff_width = params['cutoff_width']
    cutoff_function = params['cutoff_function']
    num_neighbors_adaptive = params['num_neighbors_adaptive']

    featurizer_type = getattr(pet, 'featurizer_type', 'residual')
    num_gnn_layers = len(pet.gnn_layers)
    num_readout_layers = getattr(pet, 'num_readout_layers', num_gnn_layers)

    print(f"d_pet: {d_pet}, cutoff: {cutoff}, cutoff_width: {cutoff_width}")
    print(f"cutoff_function: {cutoff_function}, num_neighbors_adaptive: {num_neighbors_adaptive}")
    print(f"featurizer_type: {featurizer_type}, gnn_layers: {num_gnn_layers}, readout_layers: {num_readout_layers}")
    print(f"example_n_atoms: {example_n_atoms}, example_max_neighbors: {example_max_neighbors}")
    print(f"forces: {forces}")

    # Create wrapper using actual GNN layers
    wrapper = PETFullModel(
        pet, n_atoms=example_n_atoms, max_neighbors=example_max_neighbors, d_pet=d_pet,
        forces=forces, cutoff=cutoff, cutoff_width=cutoff_width,
        cutoff_function=cutoff_function
    )
    wrapper.eval()

    # Create deterministic tracing inputs.
    example_inputs, input_names = build_example_inputs(
        example_n_atoms=example_n_atoms,
        example_max_neighbors=example_max_neighbors,
        cutoff=cutoff,
        forces=forces,
    )

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        expected_output = wrapper(*example_inputs)

    print(f"Output shape: {expected_output.shape}")
    print(f"Atomic energies: {expected_output}")
    print(f"Total energy: {expected_output.sum().item():.6f}")

    # Export via torch.export
    print("\nExporting via torch.export...")
    input_dtypes = {
        "species": "i32",
        "neighbor_species": "i32",
        "reverse_neighbor_index": "i32",
    }

    graph, weights = export_torch_model(
        wrapper,
        example_inputs,
        output_dir / "pet_full.json",
        input_names=input_names,
        input_dtypes=input_dtypes,
        strict=False,
    )

    # Symbolize dynamic dimensions
    print("\nSymbolizing dimensions...")
    model_constants = {1, 3, 4, 8, 32, 128, 256, 512, 768, d_pet}
    protected = model_constants - {
        example_n_atoms,
        example_max_neighbors,
        example_n_atoms * example_max_neighbors,
        example_max_neighbors + 1,
        example_n_atoms * (example_max_neighbors + 1),
    }
    graph = symbolize_dimensions(
        graph,
        {"n_atoms": example_n_atoms, "max_neighbors": example_max_neighbors},
        protected_values=protected,
    )

    # Re-save with symbolized dimensions
    with open(output_dir / "pet_full.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
    print("Saved symbolized graph with dynamic dimensions")

    save_weights(weights=weights, output_dir=output_dir)
    save_example_inputs(
        input_names=input_names,
        example_inputs=example_inputs,
        output_dir=output_dir,
    )
    expected_output.numpy().astype(np.float32).tofile(output_dir / "expected_output.bin")

    if save_compiled:
        save_exported_program(
            wrapper=wrapper,
            example_inputs=example_inputs,
            output_path=output_dir / compiled_filename,
        )

    # Get species mapping, composition energies, and scale factor
    species_to_index = get_species_mapping(pet)
    composition_energies = get_composition_energies(pet)
    energy_scale = get_energy_scale(pet)
    print(f"Energy scale factor: {energy_scale}")

    metadata = build_metadata(
        example_n_atoms=example_n_atoms,
        example_max_neighbors=example_max_neighbors,
        d_pet=d_pet,
        graph=graph,
        weights=weights,
        expected_output=expected_output,
        cutoff=cutoff,
        cutoff_width=cutoff_width,
        cutoff_function=cutoff_function,
        num_neighbors_adaptive=num_neighbors_adaptive,
        forces=forces,
        model_name=model_name,
        featurizer_type=featurizer_type,
        num_gnn_layers=num_gnn_layers,
        num_readout_layers=num_readout_layers,
        species_to_index=species_to_index,
        composition_energies=composition_energies,
        energy_scale=energy_scale,
    )
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll files saved to {output_dir}")
    print(f"Graph: {len(graph.nodes)} nodes")
    return graph, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PET model to GIR format")
    parser.add_argument("--output", "-o", type=str, default="/tmp/pet_full_export",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="pet-mad-1.0.2",
                        help="Model name: 'pet-mad-1.0.2' (legacy) or upet name like 'pet-mad-s'")
    parser.add_argument("--forces", action="store_true",
                        help="Export with forces support (manual attention, in-graph distance/cutoff)")
    parser.add_argument("--example-n-atoms", type=int, default=7,
                        help="Example atom count used only for tracing/export")
    parser.add_argument("--example-max-neighbors", type=int, default=11,
                        help="Example max neighbors used only for tracing/export")
    parser.add_argument("--n-atoms", dest="deprecated_n_atoms", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-neighbors", dest="deprecated_max_neighbors", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--save-compiled", action="store_true",
                        help="Also save compiled torch.export artifact (.pt2)")
    parser.add_argument("--compiled-filename", type=str, default="pet_full_exported.pt2",
                        help="Filename for compiled export artifact")
    args = parser.parse_args()

    example_n_atoms = args.example_n_atoms
    example_max_neighbors = args.example_max_neighbors
    if args.deprecated_n_atoms is not None:
        print("Warning: --n-atoms is deprecated; use --example-n-atoms.")
        example_n_atoms = args.deprecated_n_atoms
    if args.deprecated_max_neighbors is not None:
        print("Warning: --max-neighbors is deprecated; use --example-max-neighbors.")
        example_max_neighbors = args.deprecated_max_neighbors

    export_pet_full(
        output_dir=Path(args.output),
        example_n_atoms=example_n_atoms,
        example_max_neighbors=example_max_neighbors,
        model_name=args.model,
        forces=args.forces,
        save_compiled=args.save_compiled,
        compiled_filename=args.compiled_filename,
    )
