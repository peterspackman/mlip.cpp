#!/usr/bin/env python3
"""
Batch convert uPET models to GGUF format.

Wraps export_pet_gguf.py for each model, producing GGUF files
suitable for use with mlipcpp's GraphModel / Predictor API.

Usage:
    uv run scripts/convert_models.py                         # Convert default (small) models
    uv run scripts/convert_models.py --all                   # Convert all models incl. large/xl
    uv run scripts/convert_models.py --models pet-omat-xl    # Convert specific model(s)
    uv run scripts/convert_models.py --list                   # List available models
    uv run scripts/convert_models.py --force                  # Re-convert existing files
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Default models converted by `convert_models.py` (no flags)
DEFAULT_MODELS = [
    "pet-mad-s",
    "pet-oam-l",
    "pet-omad-xs",
    "pet-omad-s",
    "pet-omat-xs",
    "pet-omat-s",
    "pet-spice-s",
]

# All available uPET models (including large/xl variants)
ALL_MODELS = DEFAULT_MODELS + [
    "pet-oam-xl",
    "pet-omad-l",
    "pet-omat-m",
    "pet-omat-l",
    "pet-omat-xl",
    "pet-omatpes-l",
    "pet-spice-l",
]

EXPORT_SCRIPT = Path(__file__).parent / "export_pytorch" / "export_pet_gguf.py"


def convert_model(
    model_name: str,
    output_dir: Path,
    n_atoms: int = 7,
    max_neighbors: int = 11,
) -> bool:
    """Convert a single model to GGUF format.

    Returns True on success, False on failure.
    """
    output_path = output_dir / f"{model_name}.gguf"

    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--model", model_name,
        "--output", str(output_path),
        "--n-atoms", str(n_atoms),
        "--max-neighbors", str(max_neighbors),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {model_name}")
        # Show last few lines of stderr for diagnosis
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-5:]:
            print(f"    {line}")
        return False

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  OK: {output_path.name} ({size_mb:.1f} MB)")
        return True

    print(f"  FAILED: output file not created")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert uPET models to GGUF format"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to convert (default: small/xs/s variants)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all models including large/xl variants",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="gguf",
        help="Output directory for GGUF files (default: gguf/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if GGUF file already exists",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--n-atoms", type=int, default=7,
        help="Export atoms dimension (default: 7)",
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=11,
        help="Export neighbors dimension (default: 11)",
    )
    args = parser.parse_args()

    if args.list:
        print("Default models:")
        for m in DEFAULT_MODELS:
            print(f"  {m}")
        print("\nAdditional models (use --all or --models):")
        for m in ALL_MODELS:
            if m not in DEFAULT_MODELS:
                print(f"  {m}")
        return

    if args.models:
        models = args.models
    elif args.all:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate model names
    for m in models:
        if m not in ALL_MODELS:
            print(f"Warning: '{m}' not in known models list, attempting anyway")

    # Build list of conversions, skip already-converted unless --force
    conversions = []
    for model_name in models:
        output_path = output_dir / f"{model_name}.gguf"
        if not args.force and output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  SKIP: {output_path.name} already exists ({size_mb:.1f} MB)")
        else:
            conversions.append(model_name)

    if not conversions:
        print("Nothing to convert.")
        return

    print(f"\nConverting {len(conversions)} model(s) to {output_dir}/\n")

    t0 = time.time()
    success = 0
    failed = 0

    for i, model_name in enumerate(conversions):
        print(f"[{i+1}/{len(conversions)}] {model_name}...")
        if convert_model(model_name, output_dir,
                         args.n_atoms, args.max_neighbors):
            success += 1
        else:
            failed += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {success} succeeded, {failed} failed")

    # Summary of output files
    if success > 0:
        print(f"\nOutput files in {output_dir}/:")
        total_size = 0
        for f in sorted(output_dir.glob("*.gguf")):
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name:30s} {size / (1024*1024):6.1f} MB")
        print(f"  {'Total:':30s} {total_size / (1024*1024):6.1f} MB")


if __name__ == "__main__":
    main()
