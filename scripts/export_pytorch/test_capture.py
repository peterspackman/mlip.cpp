#!/usr/bin/env python3
"""
Simple test script for graph capture.

Tests the basic functionality with a simple MLP model before
trying more complex models like PET-MAD.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)

from export_pytorch.graph_capture import capture_model, CaptureConfig
from export_pytorch.graph_ir import GGMLGraph


class SimpleMLP(nn.Module):
    """Simple MLP for testing graph capture."""

    def __init__(self, d_in: int = 64, d_hidden: int = 128, d_out: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing attention capture."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


def test_simple_mlp():
    """Test graph capture with a simple MLP."""
    print("=" * 60)
    print("Testing SimpleMLP")
    print("=" * 60)

    model = SimpleMLP()
    model.eval()

    example_inputs = {"x": torch.randn(8, 64)}

    config = CaptureConfig(
        dynamic_shapes={"x": {0: "batch_size"}},
        verbose=True,
    )

    try:
        gir = capture_model(model, example_inputs, config)
        print()
        print(gir.summary())
        print()
        print("JSON output:")
        print(gir.to_json(indent=2))
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_block():
    """Test graph capture with a transformer block."""
    print("=" * 60)
    print("Testing SimpleTransformerBlock")
    print("=" * 60)

    model = SimpleTransformerBlock()
    model.eval()

    # [batch, seq, features]
    example_inputs = {"x": torch.randn(4, 10, 64)}

    config = CaptureConfig(
        dynamic_shapes={
            "x": {0: "batch_size", 1: "seq_len"},
        },
        verbose=True,
    )

    try:
        gir = capture_model(model, example_inputs, config)
        print()
        print(gir.summary())
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    results = []

    results.append(("SimpleMLP", test_simple_mlp()))
    print()
    results.append(("TransformerBlock", test_transformer_block()))

    print()
    print("=" * 60)
    print("Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
