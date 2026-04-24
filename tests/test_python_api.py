#!/usr/bin/env python3
"""
Integration test: verify graph-exported models work via Python bindings.

Usage:
    uv run pytest tests/test_python_api.py -v
"""

import os
import pytest
import numpy as np

# Skip all tests if mlipcpp is not importable
mlipcpp = pytest.importorskip("mlipcpp")


def model_path(name: str) -> str:
    """Resolve model path relative to project root."""
    return os.path.join(os.path.dirname(__file__), "..", "gguf", name)


def geometry_path(name: str) -> str:
    """Resolve geometry path relative to project root."""
    return os.path.join(os.path.dirname(__file__), "..", "geometries", name)


def read_xyz(path: str):
    """Read an XYZ file and return (positions, atomic_numbers) as numpy arrays."""
    SYMBOL_TO_Z = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Cu": 29,
        "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    }
    with open(path) as f:
        n_atoms = int(f.readline().strip())
        f.readline()  # comment
        positions = []
        atomic_numbers = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            symbol = parts[0]
            z = SYMBOL_TO_Z.get(symbol)
            atomic_numbers.append(z if z is not None else int(symbol))
            positions.extend(float(x) for x in parts[1:4])
    return (
        np.array(positions, dtype=np.float32).reshape(-1, 3),
        np.array(atomic_numbers, dtype=np.int32),
    )


# --- Predictor API tests ---

class TestPredictorAPI:
    """Test the mlipcpp.Predictor API with graph-exported models."""

    @pytest.fixture
    def auto_model(self):
        path = model_path("pet-auto.gguf")
        if not os.path.exists(path):
            pytest.skip(f"Model not found: {path}")
        return mlipcpp.Predictor(path)

    def test_model_type(self, auto_model):
        assert auto_model.model_type in ("PET", "PET-Graph")

    def test_cutoff_positive(self, auto_model):
        assert auto_model.cutoff > 0.0

    def test_water_energy(self, auto_model):
        water_path = geometry_path("water.xyz")
        if not os.path.exists(water_path):
            pytest.skip("water.xyz not found")

        positions, atomic_numbers = read_xyz(water_path)
        result = auto_model.predict(positions, atomic_numbers, compute_forces=False)

        # Reference from upet PyTorch calculator with model=pet-mad-xs
        # (the variant CI fetches from HuggingFace). Tolerance accommodates
        # small CPU↔GPU floating-point drift.
        WATER_ENERGY_REF = -15.293853
        np.testing.assert_allclose(
            result.energy, WATER_ENERGY_REF, atol=0.05,
            err_msg=f"Water energy {result.energy} eV doesn't match reference {WATER_ENERGY_REF} eV")

    def test_water_forces(self, auto_model):
        water_path = geometry_path("water.xyz")
        if not os.path.exists(water_path):
            pytest.skip("water.xyz not found")

        positions, atomic_numbers = read_xyz(water_path)
        result = auto_model.predict(positions, atomic_numbers, compute_forces=True)
        assert result.energy < 0.0
        assert result.has_forces()

        # Newton's third law: forces should sum to ~0
        forces = np.array(result.forces)
        force_sum = forces.sum(axis=0)
        np.testing.assert_allclose(force_sum, 0.0, atol=0.01)

    def test_sequential_predictions(self, auto_model):
        """Test that the same model can predict multiple systems."""
        water_path = geometry_path("water.xyz")
        si_path = geometry_path("si.xyz")
        if not os.path.exists(water_path) or not os.path.exists(si_path):
            pytest.skip("geometry files not found")

        pos_w, z_w = read_xyz(water_path)
        pos_s, z_s = read_xyz(si_path)

        r1 = auto_model.predict(pos_w, z_w, compute_forces=False)
        r2 = auto_model.predict(pos_s, z_s, compute_forces=False)

        assert r1.energy < 0.0
        assert r2.energy < 0.0
        # Energies should differ (different systems)
        assert abs(r1.energy - r2.energy) > 0.1


# --- Named model tests ---

KNOWN_MODELS = [
    "pet-mad-s",
    "pet-oam-l",
    "pet-oam-xl",
    "pet-omad-xs",
    "pet-omad-s",
    "pet-omad-l",
    "pet-omat-xs",
    "pet-omat-s",
    "pet-omat-m",
    "pet-omat-l",
    "pet-omat-xl",
    "pet-omatpes-l",
    "pet-spice-s",
    "pet-spice-l",
]


@pytest.mark.parametrize("model_name", KNOWN_MODELS)
def test_named_model_loads(model_name):
    """Test that each named model GGUF loads and produces reasonable energy."""
    path = model_path(f"{model_name}.gguf")
    if not os.path.exists(path):
        pytest.skip(f"{model_name}.gguf not found in gguf/")

    pred = mlipcpp.Predictor(path)
    assert pred.cutoff > 0.0

    water_path = geometry_path("water.xyz")
    if not os.path.exists(water_path):
        pytest.skip("water.xyz not found")

    positions, atomic_numbers = read_xyz(water_path)
    result = pred.predict(positions, atomic_numbers, compute_forces=False)
    assert np.isfinite(result.energy), f"Energy is not finite: {result.energy}"
    assert result.energy < 0.0, f"Expected negative energy, got {result.energy}"


def test_spice_force_matches_finite_difference():
    """Sanity check: reported forces should match -dE/dx for PET-Graph SPICE model."""
    path = model_path("pet-spice-s.gguf")
    if not os.path.exists(path):
        pytest.skip("pet-spice-s.gguf not found in gguf/")

    urea_path = geometry_path("urea_molecule.xyz")
    if not os.path.exists(urea_path):
        pytest.skip("urea_molecule.xyz not found")

    pred = mlipcpp.Predictor(path)
    positions, atomic_numbers = read_xyz(urea_path)

    result = pred.predict(positions, atomic_numbers, compute_forces=True)
    assert result.has_forces()
    forces = np.array(result.forces, dtype=np.float32)

    eps = 1e-2
    atom_idx = 0
    coord_idx = 0

    pos_plus = positions.copy()
    pos_minus = positions.copy()
    pos_plus[atom_idx, coord_idx] += eps
    pos_minus[atom_idx, coord_idx] -= eps

    e_plus = pred.predict(pos_plus, atomic_numbers, compute_forces=False).energy
    e_minus = pred.predict(pos_minus, atomic_numbers, compute_forces=False).energy
    fd_force = -(e_plus - e_minus) / (2.0 * eps)

    np.testing.assert_allclose(
        forces[atom_idx, coord_idx],
        fd_force,
        atol=0.1,
        err_msg="Force/energy gradient mismatch on pet-spice-s (urea)",
    )


# --- ASE calculator tests ---

class TestASECalculator:
    """Test ASE integration if available."""

    @pytest.fixture
    def ase_calc(self):
        pytest.importorskip("ase")
        path = model_path("pet-auto.gguf")
        if not os.path.exists(path):
            pytest.skip(f"Model not found: {path}")

        try:
            from mlipcpp.ase import MLIPCalculator
        except ImportError:
            pytest.skip("mlipcpp.ase not available")

        return MLIPCalculator(path)

    def test_ase_energy(self, ase_calc):
        from ase.io import read

        water_path = geometry_path("water.xyz")
        if not os.path.exists(water_path):
            pytest.skip("water.xyz not found")

        atoms = read(water_path)
        atoms.calc = ase_calc
        energy = atoms.get_potential_energy()

        assert energy < 0.0
        assert energy > -100.0
