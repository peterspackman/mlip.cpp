#!/usr/bin/env python3
"""Export complete PET energy computation path to GIR format.

This creates a traceable wrapper for the PET energy computation:
1. Input: pre-computed token features [n_atoms, seq_len, d_pet]
2. Transformer layers (x2)
3. Energy head MLP
4. Output: atomic energies [n_atoms]
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from export_pytorch.fx_converter import export_fx_model


def get_pet_model():
    """Get the PET model."""
    from pet_mad._models import get_pet_mad
    model = get_pet_mad(version="1.0.2")
    return model.module.model


class PETEnergyPath(torch.nn.Module):
    """Full PET energy computation path.

    This captures:
    - Transformer layers (applied to token features)
    - Node feature extraction (first position)
    - Energy head MLP
    - Final linear projection

    NOT captured (handled separately):
    - Node/edge embeddings (lookup tables)
    - Neighbor list construction
    - Attention mask computation

    The model expects pre-computed token features that combine:
    - Node embedding [n_atoms, 1, d_pet]
    - Edge embeddings [n_atoms, n_neighbors, d_pet]
    -> tokens [n_atoms, seq_len, d_pet] where seq_len = 1 + n_neighbors
    """

    def __init__(self, pet_model, n_atoms: int, seq_len: int, d_pet: int):
        super().__init__()

        # Store dimensions for tracing
        self.n_atoms = n_atoms
        self.seq_len = seq_len
        self.d_pet = d_pet

        # Transformer layers from GNN
        self.trans_layers = torch.nn.ModuleList()
        for gnn_layer in pet_model.gnn_layers:
            self.trans_layers.append(gnn_layer.trans.layers)

        # Energy head (one per GNN layer)
        self.energy_heads = pet_model.node_heads['energy']

        # Final projection layers (one per GNN layer, using element 0 for Si)
        self.final_layers = torch.nn.ModuleList([
            pet_model.node_last_layers['energy'][i]['energy___0']
            for i in range(len(pet_model.gnn_layers))
        ])

    def forward(self, tokens):
        """
        Args:
            tokens: [n_atoms, seq_len, d_pet] - Combined node+edge features

        Returns:
            atomic_energies: [n_atoms] - Per-atom energy predictions
        """
        cur = tokens
        atomic_energies = torch.zeros(self.n_atoms)

        # Apply transformer layers from each GNN layer, with readout after each
        for gnn_idx, layers in enumerate(self.trans_layers):
            for layer in layers:
                # Pre-norm attention
                normed = layer.norm_attention(cur)

                # QKV projection
                qkv = layer.attention.input_linear(normed)

                # Split Q, K, V
                q, k, v = qkv.chunk(3, dim=-1)

                # Reshape for multi-head attention
                n_heads = layer.attention.num_heads
                head_dim = layer.attention.head_dim

                q = q.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)
                k = k.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)
                v = v.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)

                # Attention (no mask for simplicity)
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

                # Reshape back
                attn_out = attn_out.transpose(1, 2).contiguous().view(
                    self.n_atoms, self.seq_len, self.d_pet
                )

                # Output projection + residual
                attn_out = layer.attention.output_linear(attn_out)
                cur = cur + attn_out

                # Pre-norm MLP
                normed = layer.norm_mlp(cur)
                mlp_out = layer.mlp(normed)
                cur = cur + mlp_out

            # Readout: extract node features and apply energy head for this GNN layer
            node_features = cur[:, 0, :]  # [n_atoms, d_pet]

            # Apply this layer's energy head
            x = self.energy_heads[gnn_idx](node_features)  # [n_atoms, 128]

            # Apply final projection
            e = self.final_layers[gnn_idx](x)  # [n_atoms, 1]

            atomic_energies = atomic_energies + e.squeeze(-1)

        return atomic_energies  # [n_atoms]


def export_pet_energy(output_dir: Path = Path("/tmp/pet_energy_validation")):
    """Export PET energy computation path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PET model...")
    pet = get_pet_model()
    pet.eval()

    hypers = pet.hypers
    d_pet = hypers['d_pet'] if isinstance(hypers, dict) else hypers.D_PET

    print(f"d_pet: {d_pet}")

    # Test dimensions matching Si 2-atom structure
    n_atoms = 2
    n_neighbors = 8  # max neighbors
    seq_len = 1 + n_neighbors  # node + neighbors

    print(f"n_atoms: {n_atoms}, seq_len: {seq_len}")

    # Create wrapper
    wrapper = PETEnergyPath(pet, n_atoms=n_atoms, seq_len=seq_len, d_pet=d_pet)
    wrapper.eval()

    # Create reproducible test input
    torch.manual_seed(42)
    tokens = torch.randn(n_atoms, seq_len, d_pet)

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        expected_output = wrapper(tokens)

    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {expected_output.shape}")
    print(f"Atomic energies: {expected_output}")
    print(f"Total energy: {expected_output.sum().item():.6f}")

    # Export via FX
    print("\nExporting via torch.fx...")
    graph, weights = export_fx_model(
        wrapper,
        (tokens,),
        output_dir / "pet_energy.json",
        input_names=["tokens"]
    )

    # Save weights as binary files (no transpose - stored in PyTorch order)
    print(f"\nSaving {len(weights)} weights...")
    for name, tensor in weights.items():
        data = tensor.numpy()
        filepath = output_dir / f"{name}.bin"
        data.astype(np.float32).tofile(filepath)

    # Save input - no transpose needed, GGML and PyTorch have same memory layout
    # PyTorch [2, 9, 256] = GGML [256, 9, 2] (same bytes, reversed dim labels)
    tokens.numpy().astype(np.float32).tofile(output_dir / "input_tokens.bin")
    print(f"Input: {tokens.shape} -> input_tokens.bin (GGML: {tuple(reversed(tokens.shape))})")

    # Save expected output
    expected_output.numpy().astype(np.float32).tofile(output_dir / "expected_output.bin")
    print(f"Output: {expected_output.shape} -> expected_output.bin")

    # Save metadata
    metadata = {
        "n_atoms": n_atoms,
        "seq_len": seq_len,
        "d_pet": d_pet,
        "num_nodes": len(graph.nodes),
        "num_weights": len(weights),
        "expected_total_energy": expected_output.sum().item(),
        "weights": {name: list(t.shape) for name, t in weights.items()}
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll files saved to {output_dir}")
    print(f"Graph: {len(graph.nodes)} nodes")

    return graph, weights


if __name__ == "__main__":
    export_pet_energy()
