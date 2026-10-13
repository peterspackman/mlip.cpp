# metatomic plugin

`mlipcpp-metatomic.so` is a [metatomic](https://github.com/metatensor/metatomic)
plugin, written against metatomic's C++ plugin API (`metatomic::BaseModel`,
`MTA_REGISTER_CXX_PLUGIN`). An engine that loads metatomic plugins can run a
graph-format GGUF model through it:

```python
import metatomic

metatomic.load_plugin("build/lib/mlipcpp-metatomic.so")
model = metatomic.load_model("pet-mad-xs.gguf", {"backend": "cuda"}, plugin_name="mlip.cpp")
```

The models at [huggingface.co/peterspackman/mlip-gguf](https://huggingface.co/peterspackman/mlip-gguf)
are graph-format (`general.architecture = pet-graph`). A GGUF in the older
hand-written PET format is refused with a message saying so, and anything else
is left to other plugins.

## What it provides

- `energy` per system, with `positions` and `strain` gradients (forces and the
  virial), and per atom.
- `selected_atoms`: the energy is the selected atoms', and the gradients are
  with respect to every atom, as a domain decomposition with ghost atoms needs.
  `interaction_range` is the cutoff times the number of message-passing layers,
  plus one cutoff for the adaptive cutoff.
- The engine's pair list (full, at the model cutoff) is the model's neighbor
  list. Pairs beyond the cutoff are dropped, so a longer list is fine.
- `dtype` is float32, as mlip.cpp computes in float32; the engine builds its
  systems in float32. Systems are read on the host (`supported_devices` is
  `cpu`), and ggml moves the data to whichever backend it runs on.

Options (strings, passed to `load_model`):

| option    | values                                      | default |
| --------- | ------------------------------------------- | ------- |
| `backend` | `auto`, `cpu`, `cuda`, `hip`, `metal`, `vulkan`, ... | `auto`  |
| `threads` | threads for ggml's CPU backend              | ggml's  |

## Building

```bash
pip install metatensor-core   # and metatomic-core, with its C++ library
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_BUILD_METATOMIC=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import metatomic; print(metatomic.utils.cmake_prefix_path)');$(python -c 'import metatensor; print(metatensor.utils.cmake_prefix_path)')"
cmake --build build --target mlip_metatomic
```

Add `-DMLIPCPP_USE_CUDA=ON` (or another backend) as for the library. The module
exports only `mta_plugin_init`; mlipcpp and ggml are linked in with their
symbols hidden.

## Testing

```bash
MLIPCPP_METATOMIC_PLUGIN=build/lib/mlipcpp-metatomic.so \
MLIPCPP_MODEL=gguf/pet-mad-xs.gguf pytest plugins/metatomic/test_plugin.py
```

The reference is mlip.cpp's own Python package (`mlipcpp`), which runs the
same model on its own neighbor list. The tests also cover batches, selections,
pairs in any order or beyond the cutoff, and metatomic's unit conversion.

## Engines

No released engine loads metatomic's C++ plugins yet: `metatomic_ase` and
torch-sim take TorchScript models. `examples/ase_calculator.py` is an ASE
calculator over metatomic's C API, enough for MD and relaxations:

```bash
python plugins/metatomic/examples/ase_calculator.py build/lib/mlipcpp-metatomic.so gguf/pet-mad-xs.gguf
```
