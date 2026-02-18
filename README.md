# mlipcpp

> **WARNING: Pre-alpha software**
> This project is in early development. APIs may change without notice, and there may be bugs or incomplete features. Not recommended for production use.

Standalone C++ implementation of Machine Learning Interatomic Potentials (MLIPs) using [ggml](https://github.com/ggml-org/ggml).

Currently supports [PET/uPET](https://github.com/lab-cosmo/pet-mad) models (energy, forces, stresses).

## Quick start (Python)

```bash
# Install the package
pip install .

# Download and convert a model to GGUF
uv run scripts/convert_models.py --models pet-mad-s
```

```python
import numpy as np
import mlipcpp

# Load a model
model = mlipcpp.Predictor("gguf/pet-mad-s.gguf")
print(f"Model type: {model.model_type}, cutoff: {model.cutoff} A")

# Water molecule
positions = np.array([
    [0.000,  0.000, 0.000],  # O
    [0.757,  0.586, 0.000],  # H
    [-0.757, 0.586, 0.000],  # H
], dtype=np.float32)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# Predict energy
result = model.predict(positions, atomic_numbers, compute_forces=False)
print(f"Energy: {result.energy:.4f} eV")
# => Energy: -14.3693 eV

# Predict energy + forces
result = model.predict(positions, atomic_numbers, compute_forces=True)
print(f"Energy: {result.energy:.4f} eV")
forces = np.array(result.forces).reshape(-1, 3)
print(f"Forces (eV/A):\n{forces}")
```

### ASE integration

```python
from ase.io import read
from mlipcpp.ase import MLIPCalculator

atoms = read("structure.xyz")
atoms.calc = MLIPCalculator("gguf/pet-mad-s.gguf")
print(f"Energy: {atoms.get_potential_energy():.4f} eV")
```

## Converting models

Download and convert uPET models from HuggingFace to GGUF format:

```bash
# Convert all available models
uv run scripts/convert_models.py

# Convert a specific model
uv run scripts/convert_models.py --models pet-mad-s

# List available models
uv run scripts/convert_models.py --list
```

Default models: `pet-mad-s`, `pet-oam-l`, `pet-omad-xs`, `pet-omad-s`, `pet-omat-xs`, `pet-omat-s`, `pet-spice-s`

Use `--all` to also convert larger variants: `pet-oam-xl`, `pet-omad-l`, `pet-omat-m`, `pet-omat-l`, `pet-omat-xl`, `pet-omatpes-l`, `pet-spice-l`

## Building from source

### Dependencies

- [ggml](https://github.com/ggml-org/ggml) - Tensor library (fetched automatically via CMake)
- [fmt](https://github.com/fmtlib/fmt) - Formatting library (fetched automatically)

**Note:** This project uses a [modified fork of ggml](https://github.com/peterspackman/ggml) with additional backpropagation support for `CONCAT` and `CLAMP` operations, required for force/stress computation.

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### C++ CLI

```bash
# Energy only
./build/bin/simple_inference gguf/pet-mad-s.gguf structure.xyz

# With forces
./build/bin/simple_inference gguf/pet-mad-s.gguf structure.xyz --forces

# With forces and stress (periodic systems)
./build/bin/simple_inference gguf/pet-mad-s.gguf structure.xyz --forces --stress
```

## API

C, C++, Fortran, and Python APIs are provided. See `examples/` for usage.

## License

BSD 3-Clause
