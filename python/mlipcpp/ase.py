"""
ASE Calculator interface for mlipcpp.

Example usage:
    from ase.build import molecule
    from mlipcpp.ase import MLIPCalculator

    calc = MLIPCalculator("pet-mad.gguf")
    atoms = molecule("H2O")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

from typing import Optional
import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase import Atoms

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Calculator = object
    all_changes = []

from ._mlipcpp import Predictor, ModelOptions, Backend


class MLIPCalculator(Calculator):
    """
    ASE Calculator interface for mlipcpp MLIP models.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file
    backend : str, optional
        Compute backend (default: "auto"):
        - "auto": Automatically select best available GPU, fallback to CPU
        - "cpu": CPU only
        - "cuda": NVIDIA CUDA GPU
        - "hip": AMD HIP/ROCm GPU
        - "metal": Apple Metal GPU (macOS/iOS)
        - "vulkan": Vulkan GPU (cross-platform)
        - "sycl": Intel SYCL (oneAPI)
        - "cann": Huawei Ascend NPU
    cutoff_override : float, optional
        Override the model's cutoff radius (default: use model's cutoff)

    Examples
    --------
    >>> from ase.build import molecule
    >>> from mlipcpp.ase import MLIPCalculator
    >>> calc = MLIPCalculator("pet-mad.gguf")
    >>> atoms = molecule("H2O")
    >>> atoms.calc = calc
    >>> atoms.get_potential_energy()
    -14.123...

    # Use specific backend
    >>> calc = MLIPCalculator("pet-mad.gguf", backend="cuda")
    """

    implemented_properties = ["energy", "forces", "stress"]
    default_parameters = {
        "backend": "auto",
        "cutoff_override": 0.0,
    }

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        cutoff_override: float = 0.0,
        **kwargs,
    ):
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is not installed. Install it with: pip install ase"
            )

        super().__init__(**kwargs)

        self.model_path = model_path

        # Set up model options
        options = ModelOptions()

        backend_map = {
            "auto": Backend.Auto,
            "cpu": Backend.CPU,
            "cuda": Backend.CUDA,
            "hip": Backend.HIP,
            "metal": Backend.Metal,
            "vulkan": Backend.Vulkan,
            "sycl": Backend.SYCL,
            "cann": Backend.CANN,
        }
        if backend.lower() not in backend_map:
            valid_backends = ", ".join(sorted(backend_map.keys()))
            raise ValueError(
                f"Unknown backend '{backend}'. Valid options: {valid_backends}"
            )
        options.backend = backend_map[backend.lower()]
        options.cutoff_override = cutoff_override

        # Load the model
        self._predictor = Predictor(model_path, options)

    @property
    def cutoff(self) -> float:
        """Get the model's cutoff radius in Angstroms."""
        return self._predictor.cutoff

    @property
    def model_type(self) -> str:
        """Get the model type string (e.g., 'PET')."""
        return self._predictor.model_type

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: list = ["energy"],
        system_changes: list = all_changes,
    ):
        """
        Calculate the requested properties for the given atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic system to calculate
        properties : list of str
            Properties to calculate. Can include "energy", "forces", "stress"
        system_changes : list
            List of changes since last calculation (used for caching)
        """
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # Prepare input arrays
        positions = atoms.positions.astype(np.float32)
        atomic_numbers = atoms.numbers.astype(np.int32)

        # Check for periodic boundary conditions
        pbc = atoms.pbc
        cell = None
        pbc_tuple = None

        if any(pbc):
            cell = atoms.cell.array.astype(np.float32)
            pbc_tuple = tuple(bool(p) for p in pbc)

        # Determine what to compute
        compute_forces = "forces" in properties
        compute_stress = "stress" in properties

        # Run prediction
        result = self._predictor.predict(
            positions,
            atomic_numbers,
            cell=cell,
            pbc=pbc_tuple,
            compute_forces=compute_forces or compute_stress,
            compute_stress=compute_stress,
        )

        # Store results
        self.results["energy"] = float(result.energy)

        if compute_forces and result.has_forces():
            self.results["forces"] = np.array(result.forces)

        if compute_stress and result.has_stress():
            # ASE expects stress in Voigt notation: xx, yy, zz, yz, xz, xy
            # with units of eV/Angstrom^3
            stress = np.array(result.stress)
            self.results["stress"] = stress


def load_model(model_path: str, **kwargs) -> MLIPCalculator:
    """
    Convenience function to load an MLIP model as an ASE calculator.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file
    **kwargs
        Additional arguments passed to MLIPCalculator

    Returns
    -------
    MLIPCalculator
        ASE calculator instance
    """
    return MLIPCalculator(model_path, **kwargs)
