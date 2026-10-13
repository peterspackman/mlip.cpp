"""The mlip.cpp metatomic plugin, driven through metatomic's Python API.

    MLIPCPP_METATOMIC_PLUGIN=build/lib/mlipcpp-metatomic.so pytest plugins/metatomic/test_plugin.py

Needs metatomic (core, with its Python package), metatensor, numpy, ase and
pytest; mlip.cpp's own Python package (mlipcpp), which is the reference; and a
graph-format pet-mad-xs, at gguf/pet-mad-xs.gguf or MLIPCPP_MODEL (CI fetches it
from huggingface.co/peterspackman/mlip-gguf). A missing model fails, it is not
skipped.

The plugin and mlipcpp run the same model on the same neighbors, so they agree
to float32 round-off: the pairs come in a different order.
"""

import os
import pathlib

import ase
import ase.build
import ase.neighborlist
import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock

import metatomic as mta


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL = os.environ.get("MLIPCPP_MODEL", str(ROOT / "gguf" / "pet-mad-xs.gguf"))
OPTIONS = {"backend": "cpu"}

ENERGY = mta.Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)
ENERGY_POSITIONS = mta.Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions"]
)
ENERGY_NO_GRADIENTS = mta.Quantity(name="energy", unit="eV", sample_kind="system")
ATOM_ENERGY = mta.Quantity(name="energy", unit="eV", sample_kind="atom")

# float32 round-off, from summing the same terms in another order
ENERGY_TOLERANCE = 2e-4  # eV
FORCE_TOLERANCE = 2e-4  # eV/A
STRESS_TOLERANCE = 2e-6  # eV/A^3


@pytest.fixture(scope="module", autouse=True)
def plugin():
    path = os.environ.get("MLIPCPP_METATOMIC_PLUGIN")
    if not path:
        raise RuntimeError("set MLIPCPP_METATOMIC_PLUGIN to the built mlipcpp-metatomic.so")
    mta.load_plugin(path)
    metatensor.register_external_data_wrapper(
        "metatensor::SimpleDataArray", metatensor.ExternalCpuArray
    )


_models = {}


def model(**options):
    key = tuple(sorted(options.items()))
    if key not in _models:
        _models[key] = mta.load_model(MODEL, {**OPTIONS, **options}, plugin_name="mlip.cpp")
    return _models[key]


@pytest.fixture(scope="module")
def reference():
    from mlipcpp.ase import MLIPCalculator

    return MLIPCalculator(MODEL, backend="cpu")


def structures():
    rng = np.random.default_rng(3)
    si = ase.build.bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    si.rattle(0.05, seed=1)
    water = ase.Atoms(
        "OH2", positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]
    )
    # A small triclinic box of something disordered: pairs across every face.
    mix = ase.Atoms(
        numbers=rng.choice([1, 6, 7, 8], size=24),
        scaled_positions=rng.random((24, 3)),
        cell=[[6.0, 0.0, 0.0], [1.2, 6.5, 0.0], [-0.8, 0.9, 7.0]],
        pbc=True,
    )
    keep = [0]
    for k in range(1, len(mix)):
        d = mix.get_distances(k, keep, mic=True)
        if d.min() > 1.0:
            keep.append(k)
    mix = mix[keep]
    return {"si": si, "water": water, "mix": mix}


STRUCTURES = structures()


def make_system(atoms, cutoff=None, order=None, dtype=np.float32):
    """A metatomic System for `atoms`, with the full pair list at `cutoff`."""
    if cutoff is None:
        cutoff = model().requested_pair_lists()[0].cutoff
    system = mta.System(
        "Angstrom",
        np.asarray(atoms.numbers, dtype=np.int32),
        np.asarray(atoms.positions, dtype=dtype),
        np.asarray(atoms.cell[:], dtype=dtype),
        np.asarray(atoms.pbc, dtype=bool),
    )
    i, j, S, D = ase.neighborlist.neighbor_list("ijSD", atoms, cutoff)
    if order is not None:
        p = order(len(i))
        i, j, S, D = i[p], j[p], S[p], D[p]
    samples = np.column_stack([i, j, S]).astype(np.int32)
    pairs = TensorBlock(
        values=D.reshape(-1, 3, 1).astype(dtype),
        samples=Labels(
            ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    (requested,) = model().requested_pair_lists()
    system.add_pairs(requested, pairs)
    return system


def evaluate(systems, outputs, selected_atoms=None, m=None):
    return mta.execute_model(
        m or model(), systems, outputs, selected_atoms=selected_atoms, check_consistency=True
    )


def test_capabilities():
    capabilities = model().capabilities()
    assert capabilities.length_unit.lower() == "angstrom"
    assert capabilities.dtype == "float32"
    assert capabilities.supported_devices == ["cpu"]
    assert 1 in capabilities.atomic_types and 14 in capabilities.atomic_types
    # Two message-passing layers and the adaptive cutoff, 7.5 A each.
    assert capabilities.interaction_range == pytest.approx(3 * 7.5)

    (pairs,) = model().requested_pair_lists()
    assert pairs.full_list
    assert pairs.cutoff == pytest.approx(7.5)
    assert model().requested_inputs() == []


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_matches_mlipcpp(name, reference):
    atoms = STRUCTURES[name].copy()
    periodic = atoms.pbc.all()
    energy, atom_energy = evaluate(
        [make_system(atoms)], [ENERGY if periodic else ENERGY_POSITIONS, ATOM_ENERGY]
    )
    block = energy.block()

    atoms.calc = reference
    e = block.values[0, 0]
    assert e == pytest.approx(atoms.get_potential_energy(), abs=ENERGY_TOLERANCE)
    assert atom_energy.block().values[:, 0].sum() == pytest.approx(e, abs=ENERGY_TOLERANCE)

    forces = -block.gradient("positions").values[:, :, 0]
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=FORCE_TOLERANCE)

    if periodic:
        virial = block.gradient("strain").values[0, :, :, 0]
        np.testing.assert_allclose(virial, virial.T, atol=1e-6)
        np.testing.assert_allclose(
            virial / atoms.get_volume(), atoms.get_stress(voigt=False), atol=STRESS_TOLERANCE
        )


def test_water_matches_upet():
    # Reference from upet's PyTorch calculator with pet-mad-xs, as in
    # tests/test_python_api.py.
    (energy,) = evaluate([make_system(STRUCTURES["water"])], [ENERGY_NO_GRADIENTS])
    assert energy.block().values[0, 0] == pytest.approx(-15.293853, abs=0.05)


def test_pairs_in_any_order():
    atoms = STRUCTURES["mix"]
    ordered = evaluate([make_system(atoms)], [ENERGY])[0].block()
    shuffled = evaluate(
        [make_system(atoms, order=lambda n: np.random.default_rng(5).permutation(n))], [ENERGY]
    )[0].block()
    assert shuffled.values[0, 0] == pytest.approx(ordered.values[0, 0], abs=ENERGY_TOLERANCE)
    for g in ("positions", "strain"):
        np.testing.assert_allclose(
            shuffled.gradient(g).values, ordered.gradient(g).values, atol=FORCE_TOLERANCE
        )


def test_pairs_beyond_the_cutoff_are_ignored():
    # The request is not strict: an engine may hand over a longer list.
    atoms = STRUCTURES["si"]
    exact = evaluate([make_system(atoms)], [ENERGY])[0].block()
    longer = evaluate([make_system(atoms, cutoff=9.0)], [ENERGY])[0].block()
    assert longer.values[0, 0] == pytest.approx(exact.values[0, 0], abs=ENERGY_TOLERANCE)
    np.testing.assert_allclose(
        longer.gradient("positions").values, exact.gradient("positions").values, atol=FORCE_TOLERANCE
    )


def test_batch_matches_single():
    systems = [STRUCTURES["si"], STRUCTURES["mix"]]
    batch = evaluate([make_system(a) for a in systems], [ENERGY])[0].block()
    gradients = batch.gradient("positions")
    for s, atoms in enumerate(systems):
        single = evaluate([make_system(atoms)], [ENERGY])[0].block()
        assert batch.values[s, 0] == single.values[0, 0]
        rows = gradients.samples.values[:, 1] == s
        np.testing.assert_array_equal(gradients.values[rows], single.gradient("positions").values)
        np.testing.assert_array_equal(
            batch.gradient("strain").values[s], single.gradient("strain").values[0]
        )


def test_selected_atoms():
    """A selection's energy is its atoms' share; selections that split the
    atoms add up to the whole, gradients included."""
    atoms = STRUCTURES["mix"]
    n = len(atoms)
    whole = evaluate([make_system(atoms)], [ENERGY, ATOM_ENERGY])
    per_atom = whole[1].block().values[:, 0]

    rng = np.random.default_rng(7)
    first = np.sort(rng.choice(n, size=n // 3, replace=False))
    rest = np.setdiff1d(np.arange(n), first)

    parts = []
    for chosen in (first, rest):
        selection = Labels(
            ["system", "atom"],
            np.column_stack([np.zeros_like(chosen), chosen]).astype(np.int32),
        )
        energy, atom_energy = evaluate(
            [make_system(atoms)], [ENERGY, ATOM_ENERGY], selected_atoms=selection
        )
        block = energy.block()
        assert block.values[0, 0] == pytest.approx(per_atom[chosen].sum(), abs=ENERGY_TOLERANCE)
        np.testing.assert_array_equal(atom_energy.block().samples.values[:, 1], chosen)
        np.testing.assert_allclose(atom_energy.block().values[:, 0], per_atom[chosen], atol=1e-5)
        # The gradient is still with respect to every atom.
        assert block.gradient("positions").values.shape[0] == n
        parts.append(block)

    total = whole[0].block()
    assert parts[0].values[0, 0] + parts[1].values[0, 0] == pytest.approx(
        total.values[0, 0], abs=ENERGY_TOLERANCE
    )
    for g in ("positions", "strain"):
        np.testing.assert_allclose(
            parts[0].gradient(g).values + parts[1].gradient(g).values,
            total.gradient(g).values,
            atol=FORCE_TOLERANCE,
        )


def test_system_without_selected_atoms():
    """A system the selection leaves out gets no energy row, as metatomic
    expects; the others keep theirs, with gradients that point at them."""
    water, si = STRUCTURES["water"], STRUCTURES["si"]
    chosen = np.array([0, 3, 5])
    selection = Labels(
        ["system", "atom"], np.column_stack([np.ones_like(chosen), chosen]).astype(np.int32)
    )
    energy, atom_energy = evaluate(
        [make_system(water), make_system(si)], [ENERGY_POSITIONS, ATOM_ENERGY], selected_atoms=selection
    )

    block = energy.block()
    np.testing.assert_array_equal(block.samples.values, [[1]])
    gradient = block.gradient("positions")
    assert gradient.values.shape[0] == len(si)
    np.testing.assert_array_equal(gradient.samples.values[:, :2], [[0, 1]] * len(si))

    alone = evaluate([make_system(si)], [ATOM_ENERGY])[0].block().values[:, 0]
    assert block.values[0, 0] == pytest.approx(alone[chosen].sum(), abs=ENERGY_TOLERANCE)
    np.testing.assert_array_equal(atom_energy.block().samples.values, selection.values)


def test_energy_only():
    (energy,) = evaluate([make_system(STRUCTURES["si"])], [ENERGY_NO_GRADIENTS])
    with_gradients = evaluate([make_system(STRUCTURES["si"])], [ENERGY])[0].block()
    assert energy.block().values[0, 0] == pytest.approx(with_gradients.values[0, 0], abs=1e-5)
    assert energy.block().gradients_list() == []


def test_strain_needs_a_periodic_system():
    with pytest.raises(mta.MetatomicError, match="strain gradients need a periodic system"):
        evaluate([make_system(STRUCTURES["water"])], [ENERGY])


def test_unit_conversion():
    """metatomic converts the outputs to the units the engine asks for."""
    system = make_system(STRUCTURES["si"])
    ev = evaluate([system], [ENERGY_NO_GRADIENTS])[0].block().values[0, 0]
    kj = mta.Quantity(name="energy", unit="kJ/mol", sample_kind="system")
    (energy,) = evaluate([make_system(STRUCTURES["si"])], [kj])
    assert energy.block().values[0, 0] == pytest.approx(ev * 96.48533212, rel=1e-6)


def test_threads_change_nothing():
    atoms = STRUCTURES["mix"]
    one = evaluate([make_system(atoms)], [ENERGY], m=model(threads="1"))[0].block()
    four = evaluate([make_system(atoms)], [ENERGY], m=model(threads="4"))[0].block()
    assert four.values[0, 0] == pytest.approx(one.values[0, 0], abs=ENERGY_TOLERANCE)
    np.testing.assert_allclose(
        four.gradient("positions").values, one.gradient("positions").values, atol=FORCE_TOLERANCE
    )


def test_options():
    with pytest.raises(mta.MetatomicError, match="unknown mlip.cpp option 'colour'"):
        mta.load_model(MODEL, {"colour": "blue"}, plugin_name="mlip.cpp")
    with pytest.raises(mta.MetatomicError, match="threads"):
        mta.load_model(MODEL, {"threads": "many"}, plugin_name="mlip.cpp")
    with pytest.raises(mta.MetatomicError, match="backend"):
        mta.load_model(MODEL, {"backend": "abacus"}, plugin_name="mlip.cpp")
    with pytest.raises(mta.MetatomicError):
        mta.load_model("not-a-gguf-model", plugin_name="mlip.cpp")
