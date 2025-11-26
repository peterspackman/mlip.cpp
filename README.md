# mlipcpp

> **WARNING: Pre-alpha software**
> This project is in early development. APIs may change without notice, and there may be bugs or incomplete features. Not recommended for production use.

Standalone C++ implementation of Machine Learning Interatomic Potentials (MLIPs) using [ggml](https://github.com/ggml-org/ggml).

Currently supports [PET-MAD](https://github.com/lab-cosmo/pet-mad) for energies, forces, stresses

## Dependencies

- [ggml](https://github.com/ggml-org/ggml) - Tensor library (fetched automatically via CMake)
- [fmt](https://github.com/fmtlib/fmt) - Formatting library (fetched automatically)

**Note:** This project uses a [modified fork of ggml](https://github.com/peterspackman/ggml) with additional backpropagation support for `CONCAT` and `CLAMP` operations, required for force/stress computation.

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Converting PET-MAD weights

Download and convert the official PET-MAD model to GGUF format:

```bash
uv run scripts/convert_pet_mad.py --output pet-mad.gguf
```

## Usage

```bash
# Energy only
./build/bin/simple_inference pet-mad.gguf structure.xyz

# With forces
./build/bin/simple_inference pet-mad.gguf structure.xyz --forces

# With forces and stress (periodic systems)
./build/bin/simple_inference pet-mad.gguf structure.xyz --forces --stress
```

## API

C, C++, and Fortran APIs are provided. See `examples/` for usage.

## License

BSD 3-Clause
