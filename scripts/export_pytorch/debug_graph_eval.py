#!/usr/bin/env python3
"""
Evaluate a GIR graph node-by-node in Python (using NumPy).
Compare with C++ graph_inference outputs to find divergence.

Usage:
    python3 scripts/export_pytorch/debug_graph_eval.py /tmp/pet_urea_nosymbol
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_graph(graph_path):
    with open(graph_path) as f:
        return json.load(f)


def load_weights(export_dir):
    """Load all weight tensors from binary files."""
    meta_path = export_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    weights = {}
    for name, shape in metadata.get("weights", {}).items():
        bin_path = export_dir / f"{name}.bin"
        if bin_path.exists():
            data = np.fromfile(str(bin_path), dtype=np.float32)
            if len(shape) == 0:
                weights[name] = data  # scalar
            else:
                weights[name] = data.reshape(shape)
    return weights


def load_inputs(debug_dir):
    """Load inputs dumped by C++ graph_inference."""
    dims_path = debug_dir / "dims.txt"
    with open(dims_path) as f:
        lines = f.readlines()
    n_atoms, max_neighbors = map(int, lines[0].split())

    inputs = {}
    inputs["species"] = np.fromfile(str(debug_dir / "species.bin"), dtype=np.int32)
    inputs["neighbor_species"] = np.fromfile(
        str(debug_dir / "neighbor_species.bin"), dtype=np.int32
    ).reshape(n_atoms, max_neighbors)
    inputs["edge_vectors"] = np.fromfile(
        str(debug_dir / "edge_vectors.bin"), dtype=np.float32
    ).reshape(n_atoms, max_neighbors, 3)
    inputs["edge_distances"] = np.fromfile(
        str(debug_dir / "edge_distances.bin"), dtype=np.float32
    ).reshape(n_atoms, max_neighbors)
    inputs["padding_mask"] = np.fromfile(
        str(debug_dir / "padding_mask.bin"), dtype=np.float32
    ).reshape(n_atoms, max_neighbors)
    inputs["reverse_neighbor_index"] = np.fromfile(
        str(debug_dir / "reverse_neighbor_index.bin"), dtype=np.int32
    )
    inputs["cutoff_factors"] = np.fromfile(
        str(debug_dir / "cutoff_factors.bin"), dtype=np.float32
    ).reshape(n_atoms, max_neighbors)

    return inputs, n_atoms, max_neighbors


def tensor_summary(t, name=""):
    """Print a compact summary of a tensor."""
    if isinstance(t, (int, float)):
        return f"scalar={t}"
    shape_str = str(list(t.shape))
    if t.dtype in (np.float32, np.float64):
        return f"{shape_str} sum={t.sum():.6f} min={t.min():.6f} max={t.max():.6f} mean={t.mean():.6f}"
    else:
        return f"{shape_str} dtype={t.dtype}"


graph_nodes_cache = []

def eval_node(node, node_outputs, inputs, weights, all_nodes=None):
    """Evaluate a single GIR node using NumPy."""
    global graph_nodes_cache
    graph_nodes_cache = all_nodes or []
    op = node["op"]
    node_inputs = node.get("inputs", [])
    params = node.get("params", {})
    output_shape = node.get("output_shape", [])

    def resolve(ref):
        """Resolve an input reference."""
        kind, name = ref.split(":", 1)
        if kind == "node":
            return node_outputs[int(name)]
        elif kind == "input":
            return inputs[name]
        elif kind == "weight":
            return weights[name]
        elif kind == "const":
            return np.float32(float(name))
        else:
            raise ValueError(f"Unknown ref type: {kind}")

    # ---- Evaluate operations ----
    if op == "RESHAPE":
        a = resolve(node_inputs[0])
        shape = params.get("shape", output_shape)
        return a.reshape(shape)

    elif op == "VIEW":
        a = resolve(node_inputs[0])
        shape = output_shape
        idx = params.get("index", -1)
        if idx >= 0:
            # Chunk extraction from SPLIT
            # Find which dimension was split by comparing input and output shapes
            if shape:
                # Determine split dimension: find dim where input and output differ
                split_dim = None
                for d in range(len(a.shape)):
                    if d < len(shape) and a.shape[d] != shape[d]:
                        split_dim = d
                        break
                if split_dim is not None:
                    # Calculate offset: need to find previous chunks' sizes
                    # For index 0: start at 0
                    # For index 1: start at (input_dim_size - output_dim_size) if only 2 chunks
                    # More general: look at the SPLIT node params
                    # The source node should be a SPLIT with params.shape = [size1, size2, ...]
                    src_ref = node_inputs[0]
                    src_kind, src_id = src_ref.split(":", 1)
                    split_sizes = None
                    if src_kind == "node":
                        # Find the SPLIT node
                        for n in graph_nodes_cache:
                            if n["id"] == int(src_id) and n["op"] == "SPLIT":
                                split_sizes = n.get("params", {}).get("shape", [])
                                break
                    if split_sizes:
                        start = sum(split_sizes[:idx])
                        end = start + split_sizes[idx]
                    else:
                        # Fallback: compute from output shape
                        start = 0
                        for prev_idx in range(idx):
                            start += shape[split_dim]  # approximate
                        end = start + shape[split_dim]
                    slices = [slice(None)] * len(a.shape)
                    slices[split_dim] = slice(start, end)
                    return a[tuple(slices)].reshape(shape)
                else:
                    # No dimension differs - just reshape
                    return a.reshape(shape)
            return a
        if shape:
            return a.reshape(shape)
        return a

    elif op == "GET_ROWS":
        table = resolve(node_inputs[0])
        indices = resolve(node_inputs[1])
        flat_idx = indices.flatten()
        result = table[flat_idx]
        if len(output_shape) > 2:
            return result.reshape(output_shape)
        return result

    elif op == "NEW_ZEROS":
        if not output_shape or output_shape == [0]:
            return np.array(0.0, dtype=np.float32)
        return np.zeros(output_shape, dtype=np.float32)

    elif op == "NEW_ONES":
        return np.ones(output_shape, dtype=np.float32)

    elif op == "SLICE":
        a = resolve(node_inputs[0])
        # SLICE is typically a pass-through when shapes match
        if output_shape and list(a.shape) != output_shape:
            # Need actual slicing - for now just return view
            return a[tuple(slice(0, s) for s in output_shape)]
        return a

    elif op == "CONCAT":
        tensors = [resolve(r) for r in node_inputs]
        dim = params.get("dim", 0)
        return np.concatenate(tensors, axis=dim)

    elif op == "BITWISE_NOT":
        a = resolve(node_inputs[0])
        return 1.0 - a

    elif op == "CONT":
        a = resolve(node_inputs[0])
        return np.ascontiguousarray(a)

    elif op == "INDEX_PUT":
        source = resolve(node_inputs[0])
        mask = resolve(node_inputs[1])
        values = resolve(node_inputs[2])
        # result = source * (1 - mask) + values * mask
        return source * (1.0 - mask) + values * mask

    elif op == "REPEAT":
        a = resolve(node_inputs[0])
        if output_shape:
            # Compute repeat factors
            reps = []
            for i, (s_out, s_in) in enumerate(zip(output_shape, a.shape)):
                reps.append(s_out // s_in)
            return np.tile(a, reps)
        return a

    elif op == "CLAMP":
        a = resolve(node_inputs[0])
        min_val = params.get("min", -np.inf)
        max_val = params.get("max", np.inf)
        return np.clip(a, min_val, max_val)

    elif op == "LOG":
        a = resolve(node_inputs[0])
        return np.log(a)

    elif op == "LINEAR":
        x = resolve(node_inputs[0])
        w = resolve(node_inputs[1])
        b = resolve(node_inputs[2]) if len(node_inputs) > 2 else None
        result = x @ w.T
        if b is not None:
            result = result + b
        return result

    elif op == "ADD":
        a = resolve(node_inputs[0])
        if len(node_inputs) == 1:
            return a
        b = resolve(node_inputs[1])
        return a + b

    elif op == "SUB":
        a = resolve(node_inputs[0])
        b = resolve(node_inputs[1])
        return a - b

    elif op == "MUL":
        a = resolve(node_inputs[0])
        if len(node_inputs) == 1:
            scalar = params.get("scalar", 1.0)
            return a * scalar
        b = resolve(node_inputs[1])
        return a * b

    elif op == "DIV":
        a = resolve(node_inputs[0])
        b = resolve(node_inputs[1])
        return a / b

    elif op == "UNARY_SILU":
        a = resolve(node_inputs[0])
        return a / (1.0 + np.exp(-a))  # SiLU = x * sigmoid(x)

    elif op == "LAYER_NORM":
        x = resolve(node_inputs[0])
        if len(node_inputs) == 3:
            w = resolve(node_inputs[1])
            b = resolve(node_inputs[2])
        else:
            w = resolve(node_inputs[2])
            b = resolve(node_inputs[3])
        eps = params.get("eps", 1e-5)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized * w + b

    elif op == "PERMUTE":
        a = resolve(node_inputs[0])
        axes = params.get("axes", [0, 1, 2, 3])
        axes = axes[: len(a.shape)]
        return np.transpose(a, axes)

    elif op == "TRANSPOSE":
        a = resolve(node_inputs[0])
        dims = params.get("dims", [0, 1])
        axes = list(range(len(a.shape)))
        axes[dims[0]], axes[dims[1]] = axes[dims[1]], axes[dims[0]]
        return np.transpose(a, axes)

    elif op == "SUM_ROWS":
        a = resolve(node_inputs[0])
        # SUM_ROWS reduces the last dimension
        result = a.sum(axis=-1, keepdims=True)
        if output_shape:
            return result.reshape(output_shape)
        return result

    elif op == "FLASH_ATTN_EXT":
        q = resolve(node_inputs[0])
        k = resolve(node_inputs[1])
        v = resolve(node_inputs[2])
        mask = resolve(node_inputs[3]) if len(node_inputs) > 3 else None
        scale = params.get("scale", None)
        if scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / np.sqrt(head_dim)

        # q,k,v: [batch, heads, seq, head_dim]
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            # mask is additive bias [batch, heads, seq_q, seq_k] or broadcastable
            scores = scores + mask
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return np.einsum("bhqk,bhkd->bhqd", attn, v)

    elif op == "SCALE":
        a = resolve(node_inputs[0])
        s = params.get("scale", 1.0)
        return a * s

    elif op == "SQR":
        a = resolve(node_inputs[0])
        return a * a

    elif op == "SQRT":
        a = resolve(node_inputs[0])
        return np.sqrt(a)

    elif op == "SPLIT":
        return resolve(node_inputs[0])

    elif op == "WHERE":
        cond = resolve(node_inputs[0])
        x = resolve(node_inputs[1])
        y = resolve(node_inputs[2])
        return np.where(cond > 0.5, x, y)

    elif op == "SELECT":
        a = resolve(node_inputs[0])
        dim = params.get("dim", 1)
        idx = params.get("index", 0)
        return np.take(a, idx, axis=dim)

    elif op == "INDEX":
        a = resolve(node_inputs[0])
        indices = resolve(node_inputs[1])
        flat_idx = indices.flatten()
        result = a[flat_idx]
        if output_shape:
            return result.reshape(output_shape)
        return result

    elif op == "MUL_MAT":
        a = resolve(node_inputs[0])
        b = resolve(node_inputs[1])
        return b @ a.T

    elif op == "SOFT_MAX":
        a = resolve(node_inputs[0])
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    else:
        raise ValueError(f"Unsupported op: {op}")


def main():
    if len(sys.argv) < 2:
        print("Usage: debug_graph_eval.py <export_dir> [debug_input_dir]")
        sys.exit(1)

    export_dir = Path(sys.argv[1])
    debug_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/graph_inference_debug")

    # Load everything
    print("Loading graph...")
    graph = load_graph(export_dir / "pet_full.json")

    print("Loading weights...")
    weights = load_weights(export_dir)
    print(f"  {len(weights)} weights loaded")

    print("Loading inputs...")
    inputs, n_atoms, max_neighbors = load_inputs(debug_dir)
    print(f"  n_atoms={n_atoms}, max_neighbors={max_neighbors}")

    for name, arr in inputs.items():
        print(f"  {name}: {tensor_summary(arr)}")

    # Evaluate nodes
    print(f"\nEvaluating {len(graph['nodes'])} nodes...")
    node_outputs = {}

    for node in graph["nodes"]:
        nid = node["id"]
        op = node["op"]
        name = node.get("name", "")

        try:
            result = eval_node(node, node_outputs, inputs, weights, graph["nodes"])
            node_outputs[nid] = result

            # Print summary for key nodes
            summary = tensor_summary(result) if isinstance(result, np.ndarray) else str(result)

            # Always print mask-related nodes and first/last nodes
            is_mask_related = any(
                kw in name.lower()
                for kw in ["mask", "pad", "bitwise", "index_put", "clamp", "log", "attn", "cutoff"]
            )
            is_energy = "energy" in name.lower() or "final" in name.lower()
            is_first_50 = nid < 50
            is_last_10 = nid >= len(graph["nodes"]) - 10

            if is_mask_related or is_energy or is_first_50 or is_last_10:
                print(f"  [{nid:3d}] {op:20s} {name:40s} → {summary}")

        except Exception as e:
            print(f"  [{nid:3d}] {op:20s} {name:40s} → ERROR: {e}")
            node_outputs[nid] = np.zeros(
                node.get("output_shape", [1]), dtype=np.float32
            )

    # Print final output
    output_ref = graph["outputs"][0]["node_ref"]
    _, out_id = output_ref.split(":")
    final = node_outputs[int(out_id)]
    print(f"\n=== FINAL OUTPUT ===")
    print(f"Shape: {final.shape}")
    print(f"Values: {final}")
    print(f"Sum (model energy): {final.sum():.6f}")


if __name__ == "__main__":
    main()
