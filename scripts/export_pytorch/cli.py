#!/usr/bin/env python3
"""
Command-line interface for exporting PyTorch models to GGML format.

Usage:
    python -m export_pytorch.cli pet-mad --output model.gguf
    python -m export_pytorch.cli model.pt --output model.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from .graph_capture import capture_model, CaptureConfig
from .graph_ir import GGMLGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_pet_mad(version: str = "latest") -> tuple[torch.nn.Module, dict]:
    """Load PET-MAD model."""
    try:
        from pet_mad._models import get_pet_mad
    except ImportError:
        logger.error("pet-mad not installed. Run: pip install pet-mad")
        sys.exit(1)

    logger.info(f"Loading PET-MAD version: {version}")
    model = get_pet_mad(version=version)

    # Get the inner model for export
    if hasattr(model, "module") and hasattr(model.module, "model"):
        inner_model = model.module.model
    else:
        inner_model = model

    return inner_model, {}


def create_example_inputs(model, n_atoms: int = 10, max_neighbors: int = 20) -> dict[str, torch.Tensor]:
    """Create example inputs for model tracing."""
    # Standard inputs for atomistic models with neighbor list
    n_edges = n_atoms * max_neighbors

    return {
        "positions": torch.randn(n_atoms, 3, dtype=torch.float32),
        "species": torch.randint(0, 85, (n_atoms,), dtype=torch.long),
        # Neighbor list format: [center_atom, neighbor_atom] pairs
        "neighbor_i": torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),
        "neighbor_j": torch.randint(0, n_atoms, (n_edges,), dtype=torch.long),
        # Edge vectors and distances (computed from positions in practice)
        "edge_vectors": torch.randn(n_edges, 3, dtype=torch.float32),
        "edge_distances": torch.abs(torch.randn(n_edges, dtype=torch.float32)) + 0.5,
    }


def create_pet_gnn_inputs(n_atoms: int = 10, max_neighbors: int = 20, d_pet: int = 256) -> dict[str, torch.Tensor]:
    """Create inputs for PET GNN layers (bypassing metatensor wrapper)."""
    n_edges = n_atoms * max_neighbors
    seq_len = max_neighbors + 1  # neighbors + self

    return {
        # Initial node embeddings [d_pet, n_atoms] in GGML order
        "node_features": torch.randn(n_atoms, d_pet, dtype=torch.float32),
        # Edge features after embedding [d_pet, n_edges]
        "edge_features": torch.randn(n_edges, d_pet, dtype=torch.float32),
        # Species indices for each atom
        "species": torch.randint(0, 85, (n_atoms,), dtype=torch.long),
        # Attention mask [seq_len, seq_len, n_atoms]
        "attention_mask": torch.zeros(n_atoms, seq_len, seq_len, dtype=torch.float32),
    }


def export_graph_json(gir: GGMLGraph, output_path: Path):
    """Export graph to JSON file."""
    with open(output_path, "w") as f:
        f.write(gir.to_json(indent=2))
    logger.info(f"Wrote graph to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to GGML format",
    )
    parser.add_argument(
        "model",
        help="Model to export: 'pet-mad' or path to .pt file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("model_graph.json"),
        help="Output file (JSON for now, GGUF later)",
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Model version (for pet-mad)",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=10,
        help="Number of atoms for example inputs",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of nodes to capture (for debugging)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model
    if args.model.lower() == "pet-mad":
        model, metadata = load_pet_mad(args.version)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location="cpu")
        metadata = {}

    # Create example inputs
    logger.info(f"Creating example inputs with {args.n_atoms} atoms")
    example_inputs = create_example_inputs(model, args.n_atoms)

    # Configure capture
    config = CaptureConfig(
        dynamic_shapes={
            "positions": {0: "n_atoms"},
            "species": {0: "n_atoms"},
        },
        verbose=args.verbose,
        max_nodes=args.max_nodes,
    )

    # Capture the model
    try:
        gir = capture_model(model, example_inputs, config)
    except Exception as e:
        logger.error(f"Failed to capture model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print summary
    print()
    print(gir.summary())
    print()

    # Export
    export_graph_json(gir, args.output)

    print(f"\nExported {len(gir.nodes)} nodes to {args.output}")


if __name__ == "__main__":
    main()
