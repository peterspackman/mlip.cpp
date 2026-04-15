---
license: bsd-3-clause
tags:
- mlip
- machine-learning-potentials
- ggml
- gguf
---

# MLIP GGUFs

GGUF-format conversions of the uPET family of machine-learning interatomic
potentials, for use with [mlip.cpp](https://github.com/peterspackman/mlip.cpp)
and [mlip.js](https://github.com/peterspackman/mlip.cpp/tree/main/packages/mlip.js).

## Source

Checkpoints converted from [`lab-cosmo/upet`](https://huggingface.co/lab-cosmo/upet)
(BSD-3-Clause). See the LICENSE file in this repo.

## Usage

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="peterspackman/mlip-gguf", filename="pet-mad-s.gguf")
```

Or in the browser via `mlip.js`:

```js
const buf = await fetch(
  "https://huggingface.co/peterspackman/mlip-gguf/resolve/main/pet-mad-s.gguf"
).then(r => r.arrayBuffer())
const model = await Model.loadFromBuffer(buf)
```

## Files

- `pet-mad-s.gguf` (95.4 MB)
- `pet-mad-xs.gguf` (16.3 MB)
- `pet-oam-l.gguf` (721.9 MB)
- `pet-omad-s.gguf` (95.4 MB)
- `pet-omad-xs.gguf` (16.3 MB)
- `pet-omat-s.gguf` (95.4 MB)
- `pet-omat-xs.gguf` (16.3 MB)
- `pet-spice-s.gguf` (53.4 MB)

## Conversion

These files are produced by `scripts/convert_models.py` in the mlip.cpp repo,
which wraps `scripts/export_pytorch/export_pet_gguf.py` (an exact torch.export
of the PyTorch forward + backward graph into GGUF tensors + a graph interpreter
preamble).
