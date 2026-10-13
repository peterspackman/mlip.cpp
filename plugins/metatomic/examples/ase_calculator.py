"""An ASE calculator over metatomic's C API, for any metatomic plugin model.

metatomic's own ASE calculator (metatomic_ase) takes TorchScript models only,
for now. This one loads a plugin model with metatomic.load_model and calls
metatomic.execute_model, with neighbor lists from vesin:

    import metatomic
    metatomic.load_plugin("build/lib/mlipcpp-metatomic.so")
    atoms.calc = PluginCalculator("pet-mad-xs.gguf", plugin="mlip.cpp", options={"backend": "cpu"})

The systems are built in the dtype the model asks for (float32 for mlip.cpp).
Needs metatensor, metatomic (core), vesin and numpy. Run as a script, it does a
short MD run of rattled silicon and prints the energy drift:

    python ase_calculator.py build/lib/mlipcpp-metatomic.so pet-mad-xs.gguf
"""

import metatensor
import numpy as np
import vesin
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from metatensor import Labels, TensorBlock

import metatomic as mta


metatensor.register_external_data_wrapper("metatensor::SimpleDataArray", metatensor.ExternalCpuArray)


class PluginCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(self, model, plugin=None, options=None, **kwargs):
        super().__init__(**kwargs)
        self.model = mta.load_model(model, options or {}, plugin_name=plugin)
        capabilities = self.model.capabilities()
        if "cpu" not in capabilities.supported_devices:
            raise ValueError(f"the model runs on {capabilities.supported_devices}, not the CPU")
        self.dtype = {"float32": np.float32, "float64": np.float64}[capabilities.dtype]
        (self.pairs,) = self.model.requested_pair_lists()
        if self.model.requested_inputs():
            raise ValueError("this example does not provide the model's extra inputs")
        self.neighbors = vesin.NeighborList(cutoff=self.pairs.cutoff, full_list=self.pairs.full_list)

    def _system(self, atoms):
        cell = atoms.cell[:] if atoms.pbc.any() else np.zeros((3, 3))
        system = mta.System(
            "Angstrom",
            np.ascontiguousarray(atoms.numbers, dtype=np.int32),
            np.ascontiguousarray(atoms.positions, dtype=self.dtype),
            np.ascontiguousarray(cell, dtype=self.dtype),
            np.ascontiguousarray(atoms.pbc, dtype=bool),
        )
        i, j, S, D = self.neighbors.compute(atoms.positions, cell, atoms.pbc, "ijSD", copy=False)
        samples = np.empty((len(i), 5), dtype=np.int32)
        samples[:, 0], samples[:, 1], samples[:, 2:] = i, j, S
        pairs = TensorBlock(
            values=np.ascontiguousarray(D.reshape(-1, 3, 1), dtype=self.dtype),
            # A neighbor list's pairs are unique: no need for metatensor to check.
            samples=Labels(
                ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                samples,
                assume_unique=True,
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )
        system.add_pairs(self.pairs, pairs)
        return system

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        periodic = atoms.pbc.all()
        gradients = ["positions"] + (["strain"] if periodic else [])
        outputs = [mta.Quantity(name="energy", unit="eV", sample_kind="system", gradients=gradients)]
        if "energies" in properties:
            outputs.append(mta.Quantity(name="energy", unit="eV", sample_kind="atom"))
        results = mta.execute_model(self.model, [self._system(atoms)], outputs)

        block = results[0].block()
        self.results["energy"] = float(block.values[0, 0])
        self.results["forces"] = -np.asarray(block.gradient("positions").values, dtype=np.float64)[:, :, 0]
        if periodic:
            virial = np.asarray(block.gradient("strain").values, dtype=np.float64)[0, :, :, 0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / atoms.get_volume())
        if "energies" in properties:
            self.results["energies"] = np.asarray(results[1].block().values, dtype=np.float64)[:, 0]


if __name__ == "__main__":
    import sys

    import ase.build
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    plugin, gguf = sys.argv[1], sys.argv[2]
    mta.load_plugin(plugin)
    atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    atoms.rattle(0.02, seed=0)
    atoms.calc = PluginCalculator(gguf, plugin="mlip.cpp", options={"backend": "auto"})
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))

    md = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    energies = []
    for step in range(200):
        md.run(1)
        energies.append(atoms.get_total_energy() / len(atoms))
        if step % 40 == 0:
            print(f"step {step:4d}  T = {atoms.get_temperature():6.1f} K  E = {energies[-1]:.6f} eV/atom")
    print(f"total energy spread over 200 steps: {1000 * (max(energies) - min(energies)):.3f} meV/atom")
