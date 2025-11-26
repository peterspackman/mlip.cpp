"""
mlipcpp - Fast C++ Machine Learning Interatomic Potentials

A high-performance implementation of MLIPs using GGML for inference.

Example usage:
    import mlipcpp

    # Direct C++ API
    predictor = mlipcpp.Predictor("pet-mad.gguf")
    result = predictor.predict(positions, atomic_numbers)
    print(f"Energy: {result.energy} eV")

    # ASE Calculator (if ASE is installed)
    from mlipcpp.ase import MLIPCalculator
    calc = MLIPCalculator("pet-mad.gguf")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
"""

from ._mlipcpp import (
    Backend,
    LogLevel,
    ModelOptions,
    Predictor,
    Result,
    enable_logging,
    get_backend_name,
    set_backend,
    set_log_level,
    version,
    __version__,
)

__all__ = [
    "Backend",
    "LogLevel",
    "ModelOptions",
    "Predictor",
    "Result",
    "enable_logging",
    "get_backend_name",
    "set_backend",
    "set_log_level",
    "version",
    "__version__",
]


def load_model(path: str, **kwargs) -> Predictor:
    """
    Load an MLIP model from a GGUF file.

    Parameters
    ----------
    path : str
        Path to the GGUF model file
    **kwargs
        Additional options:
        - backend: "auto", "cpu", "cuda", "hip", "metal", "vulkan", "sycl", "cann"
                   (default: "auto")
        - cutoff_override: float (default: use model's cutoff)

    Returns
    -------
    Predictor
        Model predictor instance
    """
    options = ModelOptions()

    if "backend" in kwargs:
        backend_str = kwargs["backend"].lower()
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
        if backend_str in backend_map:
            options.backend = backend_map[backend_str]

    if "cutoff_override" in kwargs:
        options.cutoff_override = float(kwargs["cutoff_override"])

    return Predictor(path, options)
