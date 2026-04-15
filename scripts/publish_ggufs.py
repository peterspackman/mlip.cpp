#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.24",
# ]
# ///
"""Push converted GGUF models to a HuggingFace dataset/model repo.

Typical flow:
    # First time only
    huggingface-cli login

    # Convert (produces gguf/*.gguf)
    uv run scripts/convert_models.py

    # Publish
    uv run scripts/publish_ggufs.py --repo peterspackman/mlip-gguf

Re-run `publish_ggufs.py` any time you re-convert; only changed files are
uploaded (HF Hub content-addresses by hash).

Attribution: the source checkpoints come from lab-cosmo/upet (BSD-3-Clause).
A README and LICENSE are written into the repo automatically on first push
unless they already exist in `--dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

REPO_DEFAULT = "peterspackman/mlip-gguf"
DIR_DEFAULT = Path("gguf")

BSD_3_CLAUSE = """BSD 3-Clause License

Copyright (c) 2024, COSMO lab, EPFL.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

README_TEMPLATE = """---
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
path = hf_hub_download(repo_id="{repo}", filename="pet-mad-s.gguf")
```

Or in the browser via `mlip.js`:

```js
const buf = await fetch(
  "https://huggingface.co/{repo}/resolve/main/pet-mad-s.gguf"
).then(r => r.arrayBuffer())
const model = await Model.loadFromBuffer(buf)
```

## Files

{file_list}

## Conversion

These files are produced by `scripts/convert_models.py` in the mlip.cpp repo,
which wraps `scripts/export_pytorch/export_pet_gguf.py` (an exact torch.export
of the PyTorch forward + backward graph into GGUF tensors + a graph interpreter
preamble).
"""


def ensure_license(directory: Path) -> Path:
    path = directory / "LICENSE"
    if not path.exists():
        path.write_text(BSD_3_CLAUSE)
        print(f"wrote {path}")
    return path


def ensure_readme(directory: Path, repo: str, gguf_files: list[Path]) -> Path:
    path = directory / "README.md"
    if path.exists():
        return path
    entries = []
    for f in sorted(gguf_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        entries.append(f"- `{f.name}` ({size_mb:.1f} MB)")
    content = README_TEMPLATE.format(
        repo=repo,
        file_list="\n".join(entries) if entries else "_(no files yet)_",
    )
    path.write_text(content)
    print(f"wrote {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO_DEFAULT,
                        help=f"HuggingFace repo id (default: {REPO_DEFAULT})")
    parser.add_argument("--dir", type=Path, default=DIR_DEFAULT,
                        help=f"Directory containing .gguf files (default: {DIR_DEFAULT})")
    parser.add_argument("--commit-message", default=None,
                        help="Custom commit message")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private if it doesn't exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without pushing")
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Directory not found: {args.dir}")
        print("Run `uv run scripts/convert_models.py` first.")
        return 1

    gguf_files = sorted(args.dir.glob("*.gguf"))
    if not gguf_files:
        print(f"No .gguf files found in {args.dir}")
        return 1

    ensure_license(args.dir)
    ensure_readme(args.dir, args.repo, gguf_files)

    total_mb = sum(f.stat().st_size for f in gguf_files) / (1024 * 1024)
    print(f"\nRepo:   {args.repo}")
    print(f"Source: {args.dir}")
    print(f"Files:  {len(gguf_files)} gguf + LICENSE + README  ({total_mb:.1f} MB total)")
    for f in gguf_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\n(dry-run — not uploading)")
        return 0

    api = HfApi()
    try:
        api.repo_info(args.repo)
        print(f"\nRepo {args.repo} exists, uploading…")
    except RepositoryNotFoundError:
        print(f"\nRepo {args.repo} not found, creating…")
        create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    commit_message = args.commit_message or f"Update GGUFs ({len(gguf_files)} models, {total_mb:.0f} MB)"
    api.upload_folder(
        folder_path=str(args.dir),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=["*.gguf", "README.md", "LICENSE"],
        commit_message=commit_message,
    )
    print(f"\nPushed to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
