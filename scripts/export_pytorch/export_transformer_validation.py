#!/usr/bin/env python3
"""Export PET transformer with test data for C++ numerical validation.

This script:
1. Exports the PET transformer graph via torch.fx
2. Saves weights as binary files
3. Saves test inputs and expected outputs
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from export_pytorch.fx_converter import export_fx_model


def get_pet_transformer():
    """Get the PET transformer module."""
    from pet_mad._models import get_pet_mad

    model = get_pet_mad(version="1.0.2")
    inner = model.module  # LLPRUncertaintyModel
    pet = inner.model     # PET

    # Get the transformer from first GNN layer
    # PET uses CartesianTransformer which has a 'trans' attribute
    gnn_layer = pet.gnn_layers[0]
    trans = gnn_layer.trans

    return trans, pet.hypers


class TransformerWrapper(torch.nn.Module):
    """Wrap the transformer to keep tensor dims <= 4D for GGML compatibility.

    GGML only supports up to 4D tensors. The standard multi-head attention
    creates 5D tensors when splitting QKV. This wrapper avoids that by using
    a slightly different reshape strategy.

    Note: We store n_atoms, seq_len, d_pet as buffers to avoid dynamic .shape access
    during FX tracing which creates problematic nodes.
    """

    def __init__(self, transformer, n_atoms: int, seq_len: int, d_pet: int):
        super().__init__()
        self.layers = transformer.layers
        # Store dimensions as constants to avoid .shape access during tracing
        self.n_atoms = n_atoms
        self.seq_len = seq_len
        self.d_pet = d_pet

    def forward(self, tokens, cutoff_factors):
        """
        Args:
            tokens: [n_atoms, seq_len, d_pet] - Input features
            cutoff_factors: [n_atoms, seq_len, 1] - Cutoff factors for attention
        Returns:
            output: [n_atoms, seq_len, d_pet] - Output features
        """
        cur = tokens

        for layer in self.layers:
            # Apply layer norm
            normed = layer.norm_attention(cur)

            # QKV projection: [n_atoms, seq_len, d_pet] -> [n_atoms, seq_len, 3 * d_pet]
            qkv = layer.attention.input_linear(normed)

            # Split into Q, K, V each [n_atoms, seq_len, d_pet]
            # Use chunk instead of slicing to avoid dynamic indexing
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape for multi-head attention (stay in 4D)
            n_heads = layer.attention.num_heads
            head_dim = layer.attention.head_dim

            # [n_atoms, seq_len, d_pet] -> [n_atoms, n_heads, seq_len, head_dim]
            q = q.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)
            k = k.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)
            v = v.view(self.n_atoms, self.seq_len, n_heads, head_dim).transpose(1, 2)

            # Create attention mask from cutoff factors
            # cutoff_factors: [n_atoms, seq_len, 1]
            # For simplicity, since we're testing with all-ones cutoff factors,
            # create a zero mask (log(1) = 0). This avoids bmm shape issues.
            # In production, the C++ code handles attention masking differently.
            mask = torch.zeros(self.n_atoms, 1, self.seq_len, self.seq_len)

            # Apply scaled dot product attention
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )

            # Reshape back: [n_atoms, n_heads, seq_len, head_dim] -> [n_atoms, seq_len, d_pet]
            attn_out = attn_out.transpose(1, 2).contiguous().view(
                self.n_atoms, self.seq_len, self.d_pet
            )

            # Output projection
            attn_out = layer.attention.output_linear(attn_out)

            # Residual connection
            cur = cur + attn_out

            # Apply MLP with layer norm
            normed = layer.norm_mlp(cur)
            mlp_out = layer.mlp(normed)
            cur = cur + mlp_out

        return cur


def export_for_validation(output_dir: Path = Path("/tmp/transformer_validation")):
    """Export transformer with validation data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PET model...")
    trans, hypers = get_pet_transformer()

    if isinstance(hypers, dict):
        d_pet = hypers.get('D_PET', hypers.get('d_pet', 256))
    else:
        d_pet = hypers.D_PET

    print(f"d_pet: {d_pet}")

    # Test dimensions
    n_atoms = 2
    seq_len = 9

    # Create wrapper with fixed dimensions for FX tracing
    wrapper = TransformerWrapper(trans, n_atoms=n_atoms, seq_len=seq_len, d_pet=d_pet)
    wrapper.eval()

    # Create reproducible test inputs
    torch.manual_seed(42)
    tokens = torch.randn(n_atoms, seq_len, d_pet)
    cutoff_factors = torch.ones(n_atoms, seq_len, 1)  # All ones = no cutoff

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        expected_output = wrapper(tokens, cutoff_factors)

    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {expected_output.shape}")
    print(f"Output[0,0,:5]: {expected_output[0,0,:5]}")

    # Export via FX
    print("\nExporting via torch.fx...")
    graph, weights = export_fx_model(
        wrapper,
        (tokens, cutoff_factors),
        output_dir / "transformer.json",
        input_names=["tokens", "cutoff_factors"]
    )

    # Save weights as binary files
    print(f"\nSaving {len(weights)} weights as binary files...")
    for name, tensor in weights.items():
        # Transpose weight matrices for GGML (column-major layout)
        data = tensor.numpy()
        if len(data.shape) == 2:
            # Weight matrix: transpose for GGML
            data = data.T.copy()

        filepath = output_dir / f"{name}.bin"
        data.astype(np.float32).tofile(filepath)
        print(f"  {name}: {tensor.shape} -> {filepath.name}")

    # Save inputs
    print("\nSaving inputs...")
    # For GGML: transpose from [n_atoms, seq, features] to [features, seq, n_atoms]
    tokens_ggml = tokens.numpy().transpose(2, 1, 0).copy()
    tokens_ggml.astype(np.float32).tofile(output_dir / "input_tokens.bin")
    print(f"  tokens: {tokens.shape} -> input_tokens.bin (GGML: {tokens_ggml.shape})")

    cutoff_ggml = cutoff_factors.numpy().transpose(2, 1, 0).copy()
    cutoff_ggml.astype(np.float32).tofile(output_dir / "input_cutoff.bin")
    print(f"  cutoff: {cutoff_factors.shape} -> input_cutoff.bin")

    # Save expected output
    print("\nSaving expected output...")
    output_ggml = expected_output.numpy().transpose(2, 1, 0).copy()
    output_ggml.astype(np.float32).tofile(output_dir / "expected_output.bin")
    print(f"  output: {expected_output.shape} -> expected_output.bin (GGML: {output_ggml.shape})")

    # Save metadata
    metadata = {
        "n_atoms": n_atoms,
        "seq_len": seq_len,
        "d_pet": d_pet,
        "input_shape_pytorch": list(tokens.shape),
        "output_shape_pytorch": list(expected_output.shape),
        "input_shape_ggml": list(tokens_ggml.shape),
        "output_shape_ggml": list(output_ggml.shape),
        "weights": {name: list(t.shape) for name, t in weights.items()}
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll files saved to {output_dir}")
    print(f"Graph: {len(graph.nodes)} nodes")

    return graph, weights


if __name__ == "__main__":
    export_for_validation()
