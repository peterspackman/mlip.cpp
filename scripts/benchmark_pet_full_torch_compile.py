#!/usr/bin/env python3
"""Benchmark PETFullModel eager vs torch.compile on CPU/CUDA/MPS.

Examples:
  ./.venv/bin/python scripts/benchmark_pet_full_torch_compile.py --model pet-mad-s --device cpu
  ./.venv/bin/python scripts/benchmark_pet_full_torch_compile.py --model pet-mad-s --device mps --compile
  ./.venv/bin/python scripts/benchmark_pet_full_torch_compile.py --model pet-mad-s --forces --with-backward --compile
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import ase.io
from ase.neighborlist import neighbor_list

from export_pytorch.export_pet_full import (
    PETFullModel,
    build_example_inputs,
    get_model_params,
    load_pet_model,
)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def move_inputs_to_device(
    inputs: tuple[torch.Tensor, ...], device: str
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in inputs)


def run_once(
    model: torch.nn.Module,
    base_inputs: tuple[torch.Tensor, ...],
    with_backward: bool,
    device: str,
) -> tuple[float, float]:
    synchronize(device)
    t0 = time.perf_counter()

    if with_backward:
        inputs = list(base_inputs)
        edge_vectors = inputs[2].detach().clone().requires_grad_(True)
        inputs[2] = edge_vectors

        output = model(*inputs)
        total_energy = output.sum()
        grad = torch.autograd.grad(total_energy, edge_vectors, create_graph=False)[0]
        checksum = float(total_energy.detach().item() + grad.abs().sum().detach().item())
    else:
        with torch.no_grad():
            output = model(*base_inputs)
            checksum = float(output.sum().detach().item())

    synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, checksum


def benchmark(
    model: torch.nn.Module,
    base_inputs: tuple[torch.Tensor, ...],
    with_backward: bool,
    device: str,
    warmup: int,
    runs: int,
) -> tuple[float, float, float, float]:
    samples: list[float] = []
    checksum = 0.0

    for i in range(warmup + runs):
        elapsed_ms, checksum = run_once(
            model=model,
            base_inputs=base_inputs,
            with_backward=with_backward,
            device=device,
        )
        if i >= warmup:
            samples.append(elapsed_ms)

    return statistics.mean(samples), min(samples), max(samples), checksum


def resolve_compile_backend(requested_backend: str, device: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if device == "mps":
        # MPS + inductor often falls back or fails; aot_eager is safer.
        return "aot_eager"
    return "inductor"


def validate_device(device: str) -> None:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    elif device == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("Requested --device mps, but this PyTorch build has no MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available on this machine/runtime.")


def infer_shape_from_structure(structure_path: str, cutoff: float) -> tuple[int, int]:
    atoms = ase.io.read(structure_path)
    n_atoms = len(atoms)
    centers = neighbor_list("i", atoms, cutoff=cutoff, self_interaction=False)

    counts = [0] * n_atoms
    for center in centers:
        counts[int(center)] += 1
    max_neighbors = max(counts) if counts else 0
    return n_atoms, max_neighbors


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PETFullModel eager vs torch.compile")
    parser.add_argument("--model", default="pet-mad-s", help="Model name, e.g. pet-mad-s")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Optional structure file to infer example_n_atoms/example_max_neighbors from cutoff",
    )
    parser.add_argument("--forces", action="store_true", help="Use forces-compatible wrapper mode")
    parser.add_argument(
        "--with-backward",
        action="store_true",
        help="Also measure backward pass (grad wrt edge_vectors); requires --forces",
    )
    parser.add_argument("--example-n-atoms", type=int, default=None)
    parser.add_argument("--example-max-neighbors", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile benchmark")
    parser.add_argument(
        "--compile-backend",
        choices=["auto", "inductor", "aot_eager", "eager"],
        default="auto",
        help="torch.compile backend (default: auto)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode",
    )
    parser.add_argument("--fullgraph", action="store_true", help="Pass fullgraph=True to torch.compile")
    args = parser.parse_args()

    if args.with_backward and not args.forces:
        raise ValueError("--with-backward requires --forces (manual-attention backward path).")

    validate_device(args.device)

    print(f"Loading model: {args.model}")
    pet = load_pet_model(args.model)
    pet.eval()
    params = get_model_params(pet)
    cutoff = float(params["cutoff"])

    inferred_n_atoms = None
    inferred_max_neighbors = None
    if args.structure is not None:
        inferred_n_atoms, inferred_max_neighbors = infer_shape_from_structure(
            structure_path=args.structure,
            cutoff=cutoff,
        )
        print(
            f"Inferred from structure {args.structure}: "
            f"n_atoms={inferred_n_atoms}, max_neighbors={inferred_max_neighbors} (cutoff={cutoff})"
        )

    example_n_atoms = (
        args.example_n_atoms if args.example_n_atoms is not None
        else inferred_n_atoms if inferred_n_atoms is not None
        else 32
    )
    example_max_neighbors = (
        args.example_max_neighbors if args.example_max_neighbors is not None
        else inferred_max_neighbors if inferred_max_neighbors is not None
        else 16
    )

    wrapper = PETFullModel(
        pet_model=pet,
        n_atoms=example_n_atoms,
        max_neighbors=example_max_neighbors,
        d_pet=params["d_pet"],
        forces=args.forces,
        cutoff=cutoff,
        cutoff_width=params["cutoff_width"],
        cutoff_function=params["cutoff_function"],
    ).to(args.device)
    wrapper.eval()

    example_inputs, _ = build_example_inputs(
        example_n_atoms=example_n_atoms,
        example_max_neighbors=example_max_neighbors,
        cutoff=cutoff,
        forces=args.forces,
    )
    example_inputs = move_inputs_to_device(example_inputs, args.device)

    mode = "energy+forces(backward)" if args.with_backward else "energy-only"
    print(
        f"Config: device={args.device}, mode={mode}, forces_wrapper={args.forces}, "
        f"shape=({example_n_atoms}, {example_max_neighbors})"
    )

    eager_mean, eager_min, eager_max, eager_ck = benchmark(
        model=wrapper,
        base_inputs=example_inputs,
        with_backward=args.with_backward,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(f"Eager:    mean={eager_mean:.2f} ms, min={eager_min:.2f}, max={eager_max:.2f}, checksum={eager_ck:.6f}")

    if args.compile:
        backend = resolve_compile_backend(args.compile_backend, args.device)
        print(f"Compiling with backend={backend}, mode={args.compile_mode}, fullgraph={args.fullgraph}")
        compiled = torch.compile(
            wrapper,
            backend=backend,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        comp_mean, comp_min, comp_max, comp_ck = benchmark(
            model=compiled,
            base_inputs=example_inputs,
            with_backward=args.with_backward,
            device=args.device,
            warmup=args.warmup,
            runs=args.runs,
        )
        speedup = eager_mean / comp_mean if comp_mean > 0 else float("inf")
        print(f"Compiled: mean={comp_mean:.2f} ms, min={comp_min:.2f}, max={comp_max:.2f}, checksum={comp_ck:.6f}")
        print(f"Speedup (compiled/eager): {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
