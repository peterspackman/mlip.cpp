#!/usr/bin/env python3
"""Validate analytical stress (virial) against finite-difference stress.

For each model GGUF, evaluate stress on a small periodic structure (Si by
default, plus optional CIFs/XYZs) and compare:

  σ_voigt[k] (analytical, from compute_stress=True)
  vs
  (E(+δ) - E(-δ)) / (2 V₀ δ)        where ε(δ) corresponds to η_k = δ

Voigt order: [xx, yy, zz, yz, xz, xy]. Strain matrix from Voigt η is:
  ε = [[η₁, η₆/2, η₅/2],
       [η₆/2, η₂, η₄/2],
       [η₅/2, η₄/2, η₃]]

Cell rows are lattice vectors and positions/cell both transform as
new = old @ (I + ε).T

Optionally compares against the upet PyTorch calculator (--pytorch).

Usage:
    .venv/bin/python scripts/test_stress_fd.py
    .venv/bin/python scripts/test_stress_fd.py --models pet-mad-xs,pet-mad-s
    .venv/bin/python scripts/test_stress_fd.py --delta 5e-3 --backend cpu
    .venv/bin/python scripts/test_stress_fd.py --pytorch --model pet-mad-s
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mlipcpp


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")


def voigt_to_matrix(eta: np.ndarray) -> np.ndarray:
    """ASE Voigt convention: shear stored as engineering strain (factor 2)."""
    return np.array(
        [
            [eta[0],     eta[5] / 2, eta[4] / 2],
            [eta[5] / 2, eta[1],     eta[3] / 2],
            [eta[4] / 2, eta[3] / 2, eta[2]],
        ],
        dtype=np.float64,
    )


@dataclass
class Structure:
    name: str
    positions: np.ndarray      # (N, 3) float32
    atomic_numbers: np.ndarray # (N,)   int32
    cell: np.ndarray           # (3, 3) float32, rows are lattice vectors


def make_silicon(a: float = 5.43) -> Structure:
    positions = np.array(
        [[0.0, 0.0, 0.0],
         [a * 0.25, a * 0.25, a * 0.25]],
        dtype=np.float32,
    )
    atomic_numbers = np.array([14, 14], dtype=np.int32)
    cell = np.array(
        [[a * 0.5, a * 0.5, 0.0],
         [0.0, a * 0.5, a * 0.5],
         [a * 0.5, 0.0, a * 0.5]],
        dtype=np.float32,
    )
    return Structure("Si (FCC primitive)", positions, atomic_numbers, cell)


def make_diamond_carbon(a: float = 3.567) -> Structure:
    positions = np.array(
        [[0.0, 0.0, 0.0],
         [a * 0.25, a * 0.25, a * 0.25]],
        dtype=np.float32,
    )
    atomic_numbers = np.array([6, 6], dtype=np.int32)
    cell = np.array(
        [[a * 0.5, a * 0.5, 0.0],
         [0.0, a * 0.5, a * 0.5],
         [a * 0.5, 0.0, a * 0.5]],
        dtype=np.float32,
    )
    return Structure("Diamond C", positions, atomic_numbers, cell)


def make_fcc_aluminum(a: float = 4.05) -> Structure:
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    atomic_numbers = np.array([13], dtype=np.int32)
    cell = np.array(
        [[a * 0.5, a * 0.5, 0.0],
         [0.0, a * 0.5, a * 0.5],
         [a * 0.5, 0.0, a * 0.5]],
        dtype=np.float32,
    )
    return Structure("FCC Al", positions, atomic_numbers, cell)


def apply_strain(struct: Structure, eta: np.ndarray) -> Structure:
    eps = voigt_to_matrix(eta)
    F = np.eye(3) + eps  # deformation gradient
    new_cell = (struct.cell.astype(np.float64) @ F.T).astype(np.float32)
    new_positions = (struct.positions.astype(np.float64) @ F.T).astype(np.float32)
    return Structure(struct.name, new_positions, struct.atomic_numbers, new_cell)


def cell_volume(cell: np.ndarray) -> float:
    return abs(float(np.linalg.det(cell.astype(np.float64))))


def predict_energy(predictor, struct: Structure) -> float:
    res = predictor.predict(
        struct.positions, struct.atomic_numbers,
        cell=struct.cell, pbc=(True, True, True),
        compute_forces=False, compute_stress=False,
    )
    return float(res.energy)


def predict_stress(predictor, struct: Structure) -> np.ndarray:
    res = predictor.predict(
        struct.positions, struct.atomic_numbers,
        cell=struct.cell, pbc=(True, True, True),
        compute_forces=True, compute_stress=True,
    )
    if not res.has_stress():
        raise RuntimeError("Predictor did not return stress (has_stress=False)")
    return np.array(res.stress, dtype=np.float64)


def fd_stress(predictor, struct: Structure, delta: float,
              order: int = 4) -> np.ndarray:
    """σ_voigt[k] = (1/V₀) ∂E/∂η_k via central-difference FD.

    order=2: σ = (E(+δ) - E(-δ)) / (2Vδ)
    order=4: σ = [8(E(+δ) - E(-δ)) - (E(+2δ) - E(-2δ))] / (12Vδ)

    The 4th-order stencil cancels the δ² truncation term, leaving a smaller
    residual when float32 noise lets us shrink δ.
    """
    V0 = cell_volume(struct.cell)
    sigma = np.zeros(6, dtype=np.float64)
    for k in range(6):
        eta = np.zeros(6, dtype=np.float64)
        def E_at(scale: float) -> float:
            eta[k] = delta * scale
            return predict_energy(predictor, apply_strain(struct, eta))
        if order == 2:
            sigma[k] = (E_at(+1.0) - E_at(-1.0)) / (2.0 * V0 * delta)
        elif order == 4:
            ep1 = E_at(+1.0); em1 = E_at(-1.0)
            ep2 = E_at(+2.0); em2 = E_at(-2.0)
            sigma[k] = (8.0 * (ep1 - em1) - (ep2 - em2)) / (12.0 * V0 * delta)
        else:
            raise ValueError(f"unsupported FD order {order}")
    return sigma


def report(model_name: str, struct_name: str,
           sigma_an: np.ndarray, sigma_fd: np.ndarray) -> bool:
    diff = sigma_an - sigma_fd
    max_abs = float(np.max(np.abs(diff)))
    scale = max(1.0, float(np.max(np.abs(sigma_fd))))
    rel = max_abs / scale
    ok = max_abs < 5e-3 or rel < 5e-2  # loose tolerance for f32 + finite-diff error

    print(f"\n  Model: {model_name}    Structure: {struct_name}")
    print(f"  {'comp':>4} {'analytical':>14} {'finite-diff':>14} {'diff':>12}")
    for k, label in enumerate(VOIGT_LABELS):
        marker = "" if abs(diff[k]) < 1e-3 else "  *"
        print(f"  {label:>4} {sigma_an[k]:>14.6f} {sigma_fd[k]:>14.6f} {diff[k]:>12.3e}{marker}")
    status = "PASS" if ok else "FAIL"
    print(f"  -> {status}  (max |Δ| = {max_abs:.3e}, rel = {rel:.3e})")
    return ok


SYMBOL_BY_Z = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    19: "K", 20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn", 35: "Br",
}


def write_extxyz(path: Path, struct: Structure) -> None:
    cell = struct.cell.flatten()
    lattice = " ".join(f"{x:.10f}" for x in cell)
    with path.open("w") as f:
        f.write(f"{len(struct.atomic_numbers)}\n")
        f.write(
            f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for z, (x, y, zc) in zip(struct.atomic_numbers, struct.positions):
            sym = SYMBOL_BY_Z.get(int(z), str(int(z)))
            f.write(f"{sym} {x:.10f} {y:.10f} {zc:.10f}\n")


_STRESS_RE = re.compile(
    r"xx=([-\d.eE+]+),\s*yy=([-\d.eE+]+),\s*zz=([-\d.eE+]+)"
    r"[\s\S]*?yz=([-\d.eE+]+),\s*xz=([-\d.eE+]+),\s*xy=([-\d.eE+]+)"
)


def run_pytorch_reference(struct: Structure, model_name: str,
                          script_path: Path) -> np.ndarray | None:
    """Drive scripts/calc_energy_pytorch.py via uv-run and parse stress.

    The reference script declares its deps inline (PEP-723), so `uv run`
    materialises an isolated env with torch+upet+ase.
    """
    with tempfile.TemporaryDirectory() as tmp:
        xyz = Path(tmp) / "struct.xyz"
        write_extxyz(xyz, struct)
        cmd = [
            "uv", "run", str(script_path),
            str(xyz),
            "--model", model_name,
            "--no-forces",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600)
        except FileNotFoundError:
            print("  (uv not found, skipping PyTorch reference)")
            return None
        except subprocess.TimeoutExpired:
            print("  (PyTorch reference timed out)")
            return None
        if result.returncode != 0:
            print(f"  PyTorch reference failed (rc={result.returncode})")
            tail = "\n".join(result.stderr.strip().splitlines()[-6:])
            if tail:
                print(f"    stderr: {tail}")
            return None
        m = _STRESS_RE.search(result.stdout)
        if not m:
            print("  Could not parse stress from PyTorch reference output")
            return None
        return np.array([float(x) for x in m.groups()], dtype=np.float64)


def find_models(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        out = []
        for name in requested:
            p = root / (name if name.endswith(".gguf") else f"{name}.gguf")
            if not p.exists():
                print(f"  warn: {p} not found, skipping", file=sys.stderr)
                continue
            out.append(p)
        return out
    return sorted(root.glob("*.gguf"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finite-difference stress validation for mlipcpp models",
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all *.gguf in gguf/)")
    parser.add_argument("--gguf-dir", type=str, default="gguf",
                        help="Directory containing model GGUFs (default: gguf)")
    parser.add_argument("--structures", type=str, default="si",
                        help="Comma-separated structures: si, diamond, al (default: si)")
    parser.add_argument("--delta", type=float, default=2e-3,
                        help="Strain magnitude for finite difference (default: 2e-3)")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["cpu", "metal", "cuda"],
                        help="Force backend (default: auto)")
    parser.add_argument("--pytorch", action="store_true",
                        help="Also compare to upet PyTorch calculator (uv run)")
    parser.add_argument("--pytorch-script", type=str,
                        default="scripts/calc_energy_pytorch.py",
                        help="Path to calc_energy_pytorch.py reference script")
    parser.add_argument("--no-fd", action="store_true",
                        help="Skip finite-difference comparison (only --pytorch)")
    args = parser.parse_args()

    if args.backend:
        backend = {
            "cpu": mlipcpp.Backend.CPU,
            "metal": mlipcpp.Backend.Metal,
            "cuda": mlipcpp.Backend.CUDA,
        }[args.backend]
        if not mlipcpp.is_backend_available(backend):
            print(f"Backend {args.backend} not available", file=sys.stderr)
            return 2
        mlipcpp.set_backend(backend)

    mlipcpp.set_log_level(mlipcpp.LogLevel.Error)
    print(f"Backend: {mlipcpp.get_backend_name()}")
    print(f"Strain magnitude δ = {args.delta}")

    structure_factories = {
        "si": make_silicon,
        "diamond": make_diamond_carbon,
        "al": make_fcc_aluminum,
    }
    selected_structs = []
    for name in args.structures.split(","):
        name = name.strip().lower()
        if name not in structure_factories:
            print(f"Unknown structure '{name}'", file=sys.stderr)
            return 2
        selected_structs.append(structure_factories[name]())

    requested = [m.strip() for m in args.models.split(",")] if args.models else None
    model_paths = find_models(Path(args.gguf_dir), requested)
    if not model_paths:
        print("No models found", file=sys.stderr)
        return 1

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for mpath in model_paths:
        model_name = mpath.stem
        print(f"\n{'='*70}\nModel: {model_name}\n{'='*70}")
        try:
            predictor = mlipcpp.Predictor(str(mpath))
        except Exception as e:
            print(f"  load failed: {e}")
            n_skip += 1
            continue

        for struct in selected_structs:
            try:
                sigma_an = predict_stress(predictor, struct)
            except Exception as e:
                print(f"  Skip {struct.name}: {e}")
                n_skip += 1
                continue

            ok_fd = True
            if not args.no_fd:
                sigma_fd = fd_stress(predictor, struct, args.delta)
                ok_fd = report(model_name, struct.name, sigma_an, sigma_fd)

            ok_torch = True
            if args.pytorch:
                sigma_torch = run_pytorch_reference(
                    struct, model_name, Path(args.pytorch_script))
                if sigma_torch is None:
                    n_skip += 1
                else:
                    diff = sigma_an - sigma_torch
                    max_abs = float(np.max(np.abs(diff)))
                    scale = max(1.0, float(np.max(np.abs(sigma_torch))))
                    rel = max_abs / scale
                    ok_torch = max_abs < 5e-3 or rel < 5e-2
                    print(f"\n  Model: {model_name}    Structure: {struct.name}    PyTorch reference")
                    print(f"  {'comp':>4} {'mlipcpp':>14} {'pytorch':>14} {'diff':>12}")
                    for k, label in enumerate(VOIGT_LABELS):
                        marker = "" if abs(diff[k]) < 1e-3 else "  *"
                        print(f"  {label:>4} {sigma_an[k]:>14.6f} {sigma_torch[k]:>14.6f} {diff[k]:>12.3e}{marker}")
                    status = "PASS" if ok_torch else "FAIL"
                    print(f"  -> {status} vs PyTorch  (max |Δ| = {max_abs:.3e}, rel = {rel:.3e})")

            ok = ok_fd and ok_torch
            n_pass += int(ok)
            n_fail += int(not ok)

    print(f"\n{'='*70}\nSummary: {n_pass} pass, {n_fail} fail, {n_skip} skip")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
