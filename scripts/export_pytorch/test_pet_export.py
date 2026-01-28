#!/usr/bin/env python3
"""
Test script for exporting PET-MAD GNN layers.

This script attempts to trace and export the inner GNN layers from PET-MAD,
bypassing the metatensor wrapper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pet_gnn_layer():
    """Get a single GNN layer from PET-MAD."""
    try:
        from pet_mad._models import get_pet_mad
    except ImportError:
        logger.error("pet-mad not installed. Run: pip install pet-mad")
        return None, None

    logger.info("Loading PET-MAD model...")
    model = get_pet_mad(version="latest")

    # Navigate to the inner model
    if hasattr(model, "module") and hasattr(model.module, "model"):
        inner = model.module.model
    else:
        inner = model

    logger.info(f"Inner model type: {type(inner).__name__}")

    # Get the GNN layers
    if hasattr(inner, "gnn"):
        gnn_layers = inner.gnn
        logger.info(f"Found {len(gnn_layers)} GNN layers")
        if len(gnn_layers) > 0:
            layer = gnn_layers[0]
            logger.info(f"GNN layer type: {type(layer).__name__}")

            # Get hyperparameters
            hypers = {}
            if hasattr(inner, "hypers"):
                hypers = inner.hypers
                logger.info(f"Hyperparameters: d_pet={hypers.get('d_pet')}")

            return layer, hypers

    return None, None


def analyze_gnn_layer(layer):
    """Analyze the structure of a GNN layer."""
    logger.info("\n=== GNN Layer Structure ===")

    for name, module in layer.named_modules():
        if name:  # Skip the root module
            logger.info(f"  {name}: {type(module).__name__}")

    logger.info("\n=== Parameters ===")
    for name, param in layer.named_parameters():
        logger.info(f"  {name}: {list(param.shape)}")


def create_gnn_inputs(hypers, n_atoms=4, max_neighbors=8):
    """Create inputs for a GNN layer."""
    d_pet = hypers.get("d_pet", 256)
    n_edges = n_atoms * max_neighbors
    seq_len = max_neighbors + 1  # neighbors + self

    return {
        # Node embeddings [n_atoms, d_pet] (PyTorch order)
        "x": torch.randn(n_atoms, d_pet, dtype=torch.float32),
        # Edge embeddings [n_edges, d_pet]
        "edge_attr": torch.randn(n_edges, d_pet, dtype=torch.float32),
        # Edge indices [2, n_edges]
        "edge_index": torch.stack([
            torch.repeat_interleave(torch.arange(n_atoms), max_neighbors),
            torch.randint(0, n_atoms, (n_edges,))
        ]),
        # Attention mask [n_atoms, seq_len, seq_len]
        "attn_mask": torch.zeros(n_atoms, seq_len, seq_len, dtype=torch.float32),
    }


def try_export_layer(layer, inputs, layer_name="gnn_layer"):
    """Try to export a layer using torch.export."""
    from export_pytorch.graph_capture import capture_model, CaptureConfig

    config = CaptureConfig(
        verbose=False,
        max_nodes=500,  # Limit for debugging
    )

    try:
        layer.eval()
        logger.info(f"\nExporting {layer_name}...")
        gir = capture_model(layer, inputs, config)
        logger.info(f"Success! {len(gir.nodes)} nodes")
        return gir
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_submodules(layer, hypers):
    """Try exporting individual submodules."""
    d_pet = hypers.get("d_pet", 256)
    n_atoms = 4
    n_edges = n_atoms * 8
    seq_len = 9

    results = {}

    # Try to export transformer layers if present
    if hasattr(layer, "transformer") or hasattr(layer, "tl"):
        tl = getattr(layer, "transformer", None) or getattr(layer, "tl", None)
        if tl is not None:
            for i, block in enumerate(tl if hasattr(tl, "__iter__") else [tl]):
                # Transformer block typically takes [batch, seq, features]
                inputs = {
                    "x": torch.randn(n_atoms, seq_len, d_pet),
                }
                gir = try_export_layer(block, inputs, f"transformer_{i}")
                if gir:
                    results[f"transformer_{i}"] = gir

    # Try MLP heads
    for name in ["node_head", "edge_head", "output_head"]:
        if hasattr(layer, name):
            head = getattr(layer, name)
            inputs = {"x": torch.randn(n_atoms, d_pet)}
            gir = try_export_layer(head, inputs, name)
            if gir:
                results[name] = gir

    return results


def main():
    layer, hypers = get_pet_gnn_layer()

    if layer is None:
        logger.error("Could not load PET-MAD model")
        return

    # Analyze the layer structure
    analyze_gnn_layer(layer)

    # Try to find the forward signature
    logger.info("\n=== Forward Method ===")
    import inspect
    try:
        sig = inspect.signature(layer.forward)
        logger.info(f"forward{sig}")
    except Exception as e:
        logger.info(f"Could not get signature: {e}")

    # Try exporting submodules first (more likely to work)
    logger.info("\n=== Exporting Submodules ===")
    results = export_submodules(layer, hypers)

    for name, gir in results.items():
        logger.info(f"\n{name}:")
        logger.info(gir.summary())

    if not results:
        logger.info("No submodules could be exported.")
        logger.info("The full GNN layer may require custom handling for metatensor.")


if __name__ == "__main__":
    main()
