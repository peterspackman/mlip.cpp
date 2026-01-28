#!/usr/bin/env python3
"""Debug tracer for PET model - saves intermediate tensors for comparison with C++.

This script:
1. Loads the PET model
2. Runs a forward pass with hooks to capture every intermediate tensor
3. Saves tensors in a format that can be compared with C++ output

Usage:
    uv run scripts/export_pytorch/debug_pet_trace.py

The output is saved to /tmp/pet_debug/py/ with:
- node_{id}_{name}.bin - Binary tensor data
- node_{id}_{name}.json - Shape and dtype metadata
- trace_summary.json - Complete trace information
"""

import json
import numpy as np
import torch
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the same PETEnergyPath used in export_pet_energy.py
from export_pytorch.export_pet_energy import PETEnergyPath, get_pet_model


@dataclass
class TensorInfo:
    """Metadata about a traced tensor."""
    node_id: int
    name: str
    shape: List[int]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    first_values: List[float]  # First 10 values for quick comparison


class PETDebugTracer:
    """Traces intermediate tensors in PET model execution."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tensors: Dict[int, np.ndarray] = {}
        self.tensor_infos: Dict[int, TensorInfo] = {}
        self.node_id = 0

    def trace_tensor(self, tensor: torch.Tensor, name: str) -> int:
        """Save a tensor and return its node ID."""
        node_id = self.node_id
        self.node_id += 1

        # Convert to numpy
        data = tensor.detach().cpu().numpy().copy()

        # Save binary data
        bin_path = self.output_dir / f"node_{node_id:04d}_{name}.bin"
        data.astype(np.float32).tofile(bin_path)

        # Create metadata
        flat = data.flatten()
        first_vals = flat[:10].tolist() if len(flat) >= 10 else flat.tolist()

        info = TensorInfo(
            node_id=node_id,
            name=name,
            shape=list(data.shape),
            dtype=str(data.dtype),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            mean_val=float(np.mean(data)),
            std_val=float(np.std(data)),
            first_values=first_vals,
        )

        # Save metadata
        json_path = self.output_dir / f"node_{node_id:04d}_{name}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(info), f, indent=2)

        self.tensors[node_id] = data
        self.tensor_infos[node_id] = info

        return node_id

    def save_summary(self):
        """Save a summary of all traced tensors."""
        summary = {
            "num_tensors": len(self.tensor_infos),
            "tensors": [asdict(info) for info in self.tensor_infos.values()]
        }
        with open(self.output_dir / "trace_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def trace_pet_energy_manual(tracer: PETDebugTracer, wrapper: PETEnergyPath,
                            tokens: torch.Tensor) -> torch.Tensor:
    """Manually trace through PET energy path, saving intermediates.

    This replicates the forward pass of PETEnergyPath but saves every
    intermediate tensor for debugging.
    """
    # Save input
    tracer.trace_tensor(tokens, "input_tokens")

    cur = tokens
    atomic_energies = torch.zeros(wrapper.n_atoms)

    for gnn_idx, layers in enumerate(wrapper.trans_layers):
        for layer_idx, layer in enumerate(layers):
            prefix = f"gnn{gnn_idx}_layer{layer_idx}"

            # Pre-norm attention
            normed = layer.norm_attention(cur)
            tracer.trace_tensor(normed, f"{prefix}_norm_attn")

            # QKV projection
            qkv = layer.attention.input_linear(normed)
            tracer.trace_tensor(qkv, f"{prefix}_qkv")

            # Split Q, K, V
            q, k, v = qkv.chunk(3, dim=-1)
            tracer.trace_tensor(q, f"{prefix}_q_chunk")
            tracer.trace_tensor(k, f"{prefix}_k_chunk")
            tracer.trace_tensor(v, f"{prefix}_v_chunk")

            # Reshape for multi-head attention
            n_heads = layer.attention.num_heads
            head_dim = layer.attention.head_dim

            q_view = q.view(wrapper.n_atoms, wrapper.seq_len, n_heads, head_dim)
            tracer.trace_tensor(q_view, f"{prefix}_q_view")

            q_trans = q_view.transpose(1, 2)
            k_trans = k.view(wrapper.n_atoms, wrapper.seq_len, n_heads, head_dim).transpose(1, 2)
            v_trans = v.view(wrapper.n_atoms, wrapper.seq_len, n_heads, head_dim).transpose(1, 2)

            tracer.trace_tensor(q_trans, f"{prefix}_q_trans")
            tracer.trace_tensor(k_trans, f"{prefix}_k_trans")
            tracer.trace_tensor(v_trans, f"{prefix}_v_trans")

            # Make contiguous for attention
            q_cont = q_trans.contiguous()
            k_cont = k_trans.contiguous()
            v_cont = v_trans.contiguous()

            tracer.trace_tensor(q_cont, f"{prefix}_q_cont")
            tracer.trace_tensor(k_cont, f"{prefix}_k_cont")
            tracer.trace_tensor(v_cont, f"{prefix}_v_cont")

            # Attention
            attn_out = torch.nn.functional.scaled_dot_product_attention(q_cont, k_cont, v_cont)
            tracer.trace_tensor(attn_out, f"{prefix}_attn_out")

            # Reshape back
            attn_trans = attn_out.transpose(1, 2)
            tracer.trace_tensor(attn_trans, f"{prefix}_attn_trans")

            attn_cont = attn_trans.contiguous()
            tracer.trace_tensor(attn_cont, f"{prefix}_attn_cont")

            attn_view = attn_cont.view(wrapper.n_atoms, wrapper.seq_len, wrapper.d_pet)
            tracer.trace_tensor(attn_view, f"{prefix}_attn_view")

            # Output projection
            attn_proj = layer.attention.output_linear(attn_view)
            tracer.trace_tensor(attn_proj, f"{prefix}_attn_proj")

            # Residual
            cur = cur + attn_proj
            tracer.trace_tensor(cur, f"{prefix}_residual1")

            # Pre-norm MLP
            normed_mlp = layer.norm_mlp(cur)
            tracer.trace_tensor(normed_mlp, f"{prefix}_norm_mlp")

            mlp_out = layer.mlp(normed_mlp)
            tracer.trace_tensor(mlp_out, f"{prefix}_mlp_out")

            # Residual
            cur = cur + mlp_out
            tracer.trace_tensor(cur, f"{prefix}_residual2")

        # Readout: extract node features
        node_features = cur[:, 0, :]  # [n_atoms, d_pet]
        tracer.trace_tensor(node_features, f"gnn{gnn_idx}_node_features")

        # Apply energy head
        x = wrapper.energy_heads[gnn_idx](node_features)
        tracer.trace_tensor(x, f"gnn{gnn_idx}_energy_head")

        # Apply final projection
        e = wrapper.final_layers[gnn_idx](x)
        tracer.trace_tensor(e, f"gnn{gnn_idx}_final_proj")

        atomic_energies = atomic_energies + e.squeeze(-1)
        tracer.trace_tensor(atomic_energies, f"gnn{gnn_idx}_atomic_energies")

    return atomic_energies


def trace_pet_with_hooks(output_dir: Path = Path("/tmp/pet_debug/py")):
    """Trace PET model using forward hooks on each module.

    This is an alternative approach that uses PyTorch hooks instead of
    manual tracing. It's more general but may miss some operations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PET model...")
    pet = get_pet_model()
    pet.eval()

    hypers = pet.hypers
    d_pet = hypers['d_pet'] if isinstance(hypers, dict) else hypers.D_PET

    # Test dimensions
    n_atoms = 2
    seq_len = 9

    # Create wrapper
    wrapper = PETEnergyPath(pet, n_atoms=n_atoms, seq_len=seq_len, d_pet=d_pet)
    wrapper.eval()

    # Same input as export_pet_energy.py
    torch.manual_seed(42)
    tokens = torch.randn(n_atoms, seq_len, d_pet)

    print(f"Input shape: {tokens.shape}")
    print(f"Input[0,0,:5]: {tokens[0,0,:5]}")

    # Create tracer
    tracer = PETDebugTracer(output_dir)

    # Run manual trace
    print("\nRunning traced forward pass...")
    with torch.no_grad():
        output = trace_pet_energy_manual(tracer, wrapper, tokens)

    tracer.trace_tensor(output, "final_output")
    tracer.save_summary()

    print(f"\nOutput: {output}")
    print(f"Total energy: {output.sum().item():.6f}")
    print(f"\nSaved {len(tracer.tensor_infos)} intermediate tensors to {output_dir}")

    # Print summary of key tensors
    print("\n=== Key Tensor Summary ===")
    for info in tracer.tensor_infos.values():
        if any(key in info.name for key in ["input", "output", "q_chunk", "attn_out", "node_features"]):
            print(f"{info.node_id:4d} {info.name:30s} shape={info.shape} "
                  f"mean={info.mean_val:.4f} std={info.std_val:.4f}")

    return tracer


def save_input_for_cpp_test(output_dir: Path = Path("/tmp/pet_debug")):
    """Save the exact input used for tracing, for C++ testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Same input as export_pet_energy.py
    torch.manual_seed(42)
    n_atoms = 2
    seq_len = 9
    d_pet = 256

    tokens = torch.randn(n_atoms, seq_len, d_pet)

    # Save in same format as export_pet_energy.py
    tokens.numpy().astype(np.float32).tofile(output_dir / "input_tokens.bin")

    # Also save metadata
    metadata = {
        "n_atoms": n_atoms,
        "seq_len": seq_len,
        "d_pet": d_pet,
        "input_shape_pytorch": list(tokens.shape),
        "input_shape_ggml": [d_pet, seq_len, n_atoms],  # Reversed
    }
    with open(output_dir / "input_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved input to {output_dir / 'input_tokens.bin'}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Trace PET model intermediate tensors")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pet_debug/py"),
                        help="Output directory for trace files")
    args = parser.parse_args()

    # Save input for C++ testing
    save_input_for_cpp_test(args.output_dir.parent)

    # Run trace
    trace_pet_with_hooks(args.output_dir)


if __name__ == "__main__":
    main()
