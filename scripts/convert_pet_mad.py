#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "pet-mad",
#     "gguf>=0.10.0",
# ]
# ///
"""
Convert PET-MAD PyTorch checkpoint to GGUF format for mlipcpp.

Usage:
    uv run scripts/convert_pet_mad.py --output pet-mad.gguf
    uv run scripts/convert_pet_mad.py --version 1.0.2 --output pet-mad.gguf
"""

import argparse
import io
import zipfile
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch


def load_pet_mad_model(version: str = "latest", checkpoint_path: str | None = None):
    """Load PET-MAD model from checkpoint or download."""
    from pet_mad._models import get_pet_mad

    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = checkpoint
        hypers = {}
    else:
        print(f"Downloading PET-MAD version: {version}")
        model = get_pet_mad(version=version)
        hypers = {}

    return model, hypers


def extract_pet_hypers(model) -> dict[str, Any]:
    """Extract hyperparameters from PET model."""
    hypers = {}

    # For AtomisticModel wrapper, access the inner PET model
    if hasattr(model, "module") and hasattr(model.module, "model"):
        pet_model = model.module.model
        if hasattr(pet_model, "hypers"):
            hypers = dict(pet_model.hypers)
            print(f"Extracted hypers from model: {hypers}")
            return hypers

    if hasattr(model, "hypers"):
        hypers = dict(model.hypers)
    elif hasattr(model, "hparams"):
        hypers = dict(model.hparams)
    else:
        print("Warning: Could not find hypers, using fallback values")
        hypers = {
            "cutoff": 4.5,
            "d_pet": 256,
            "d_head": 128,
            "num_heads": 8,
            "num_gnn_layers": 2,
            "d_feedforward": 512,
            "num_attention_layers": 2,
            "cutoff_width": 0.2,
        }

    return hypers


def extract_composition_energies(model) -> dict[int, float] | None:
    """Extract composition energies from PET model's additive models."""
    try:
        if not hasattr(model, "module"):
            return None
        if not hasattr(model.module, "model"):
            return None
        if not hasattr(model.module.model, "additive_models"):
            return None

        additive_models = model.module.model.additive_models
        if len(additive_models) == 0:
            return None

        comp_model = additive_models[0]

        if not hasattr(comp_model, "energy_composition_buffer"):
            return None

        buffer = comp_model.energy_composition_buffer
        buffer_bytes = buffer.cpu().numpy().tobytes()

        bio = io.BytesIO(buffer_bytes)
        with zipfile.ZipFile(bio, "r") as zf:
            samples = np.load(
                io.BytesIO(zf.read("blocks/0/samples.npy")), allow_pickle=True
            )
            values = np.load(
                io.BytesIO(zf.read("blocks/0/values.npy")), allow_pickle=True
            )

            composition_energies = {}
            for i in range(len(samples)):
                z = int(samples[i][0])
                energy = float(values[i][0])
                composition_energies[z] = energy

            return composition_energies

    except Exception as e:
        print(f"Warning: Failed to extract composition energies: {e}")
        return None


def convert_pet_to_gguf(model, hypers: dict[str, Any], output_path: str):
    """Convert PET model to GGUF format."""
    writer = gguf.GGUFWriter(output_path, arch="pet")

    writer.add_name("PET-MAD")
    writer.add_description(
        "Point Edge Transformer trained on Massive Atomic Diversity dataset"
    )

    # Hyperparameters
    writer.add_float32("pet.cutoff", float(hypers.get("cutoff", 6.0)))
    writer.add_float32("pet.cutoff_width", float(hypers.get("cutoff_width", 0.5)))
    writer.add_int32("pet.d_pet", int(hypers.get("d_pet", 128)))
    writer.add_int32("pet.d_head", int(hypers.get("d_head", 128)))
    writer.add_int32("pet.num_heads", int(hypers.get("num_heads", 8)))
    writer.add_int32("pet.num_gnn_layers", int(hypers.get("num_gnn_layers", 3)))
    writer.add_int32("pet.d_feedforward", int(hypers.get("d_feedforward", 512)))
    writer.add_int32(
        "pet.num_attention_layers", int(hypers.get("num_attention_layers", 3))
    )

    # Atomic types
    atomic_types = None
    if hasattr(model, "atomic_types"):
        atomic_types = [int(x) for x in model.atomic_types]
    elif hasattr(model, "module") and hasattr(model.module, "model"):
        if hasattr(model.module.model, "atomic_types"):
            atomic_types = [int(x) for x in model.module.model.atomic_types]

    if atomic_types is None:
        atomic_types = list(range(1, 86))
        print("  Using default atomic_types: [1..85]")
    else:
        print(f"  Found atomic_types: {len(atomic_types)} elements")

    atomic_types_tensor = np.array(atomic_types, dtype=np.int32)
    writer.add_tensor("pet.atomic_types", atomic_types_tensor)
    writer.add_int32("pet.n_atom_types", len(atomic_types))

    # Composition energies
    print("Extracting composition energies...")
    composition_energies = extract_composition_energies(model)

    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}

    if composition_energies is not None:
        atomic_numbers = sorted(composition_energies.keys())
        energies_list = [composition_energies[z] for z in atomic_numbers]

        writer.add_tensor(
            "composition.atomic_numbers", np.array(atomic_numbers, dtype=np.int32)
        )
        writer.add_tensor(
            "composition.energies", np.array(energies_list, dtype=np.float32)
        )
        writer.add_int32("pet.composition.num_species", len(atomic_numbers))

        print(f"  Added composition energies for {len(atomic_numbers)} elements")
    else:
        print("  Warning: No composition energies found")

    # Model weights
    print(f"Extracting {len(state_dict)} tensors...")
    tensor_count = 0

    # Name abbreviations to fit GGML's 64 char limit
    abbreviations = [
        (".trans.layers.", ".tl."),
        (".attention.", ".attn."),
        (".input_linear.", ".in."),
        (".output_linear.", ".out."),
        (".norm_attention.", ".norm_a."),
        (".norm_mlp.", ".norm_m."),
        ("gnn_layers.", "gnn."),
    ]

    for name, param in state_dict.items():
        clean_name = name.replace("module.", "")
        for old, new in abbreviations:
            clean_name = clean_name.replace(old, new)

        if len(clean_name) >= 64:
            print(f"  Warning: name too long ({len(clean_name)}): {clean_name}")

        np_tensor = param.detach().cpu().numpy().astype(np.float32)
        writer.add_tensor(clean_name, np_tensor)
        tensor_count += 1

        if tensor_count <= 3:
            print(f"  {clean_name}: {list(param.shape)}")

    if tensor_count > 3:
        print(f"  ... and {tensor_count - 3} more")

    print(f"Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = Path(output_path).stat().st_size
    print(f"Done: {tensor_count} tensors, {file_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PET-MAD PyTorch model to GGUF format"
    )
    parser.add_argument(
        "--version", type=str, default="latest", help="PET-MAD version (default: latest)"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Local checkpoint path (overrides --version)"
    )
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")

    args = parser.parse_args()

    print("PET-MAD to GGUF Converter")
    print("=" * 40)

    model, hypers = load_pet_mad_model(
        version=args.version, checkpoint_path=args.checkpoint
    )

    if not hypers:
        hypers = extract_pet_hypers(model)

    print(f"\nHyperparameters:")
    for key, value in hypers.items():
        print(f"  {key}: {value}")
    print()

    convert_pet_to_gguf(model, hypers, args.output)

    print(f"\nUsage: ./build/bin/simple_inference {args.output} structure.xyz")


if __name__ == "__main__":
    main()
