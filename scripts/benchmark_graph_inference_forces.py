#!/usr/bin/env python3
"""Benchmark graph_inference energy-only vs energy+forces compute times.

Example:
  ./.venv/bin/python scripts/benchmark_graph_inference_forces.py \
    --model /tmp/pet-oam-l-dyn.gguf \
    --structures geometries/si.xyz geometries/urea_molecule.xyz
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path


TIME_RE = re.compile(r"Compute time:\s*([0-9.]+)\s*ms")
NODES_RE = re.compile(r"Graph nodes \(forward\+backward\):\s*([0-9]+)")


def run_once(model: str, structure: str, forces: bool) -> tuple[float, int | None]:
    cmd = ["./build/bin/graph_inference", model, structure]
    if forces:
        cmd.append("--forces")
    out = subprocess.check_output(cmd, text=True)

    m = TIME_RE.search(out)
    if m is None:
        raise RuntimeError("Could not parse compute time from graph_inference output")
    time_ms = float(m.group(1))

    n = NODES_RE.search(out)
    node_count = int(n.group(1)) if n else None
    return time_ms, node_count


def benchmark_mode(
    model: str,
    structure: str,
    forces: bool,
    warmup: int,
    runs: int,
) -> tuple[float, float, float, int | None]:
    node_count = None
    total = warmup + runs
    samples: list[float] = []
    for i in range(total):
        t_ms, nodes = run_once(model, structure, forces)
        if nodes is not None:
            node_count = nodes
        if i >= warmup:
            samples.append(t_ms)
    return statistics.mean(samples), min(samples), max(samples), node_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .gguf model")
    parser.add_argument(
        "--structures",
        nargs="+",
        required=True,
        help="One or more XYZ files",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    model = str(Path(args.model))
    print("structure,mode,mean_ms,min_ms,max_ms,runs,forward_backward_nodes")
    for structure in args.structures:
        for forces in (False, True):
            mode = "energy+forces" if forces else "energy"
            mean_ms, min_ms, max_ms, nodes = benchmark_mode(
                model=model,
                structure=structure,
                forces=forces,
                warmup=args.warmup,
                runs=args.runs,
            )
            node_str = str(nodes) if nodes is not None else ""
            print(
                f"{structure},{mode},{mean_ms:.2f},{min_ms:.2f},{max_ms:.2f},"
                f"{args.runs},{node_str}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
