"""
GGML Intermediate Representation (GIR) data structures.

This module defines the graph representation that can be serialized
to JSON and stored in GGUF files for runtime interpretation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GGMLDtype(Enum):
    """GGML data types."""
    F32 = "f32"
    F16 = "f16"
    I32 = "i32"
    I16 = "i16"
    I8 = "i8"
    BOOL = "bool"  # Represented as I8 in GGML

    @classmethod
    def from_torch_dtype(cls, dtype) -> "GGMLDtype":
        """Convert torch dtype to GGML dtype."""
        import torch
        mapping = {
            torch.float32: cls.F32,
            torch.float16: cls.F16,
            torch.bfloat16: cls.F16,  # Approximate as F16
            torch.int32: cls.I32,
            torch.int16: cls.I16,
            torch.int8: cls.I8,
            torch.int64: cls.I32,  # Downcast
            torch.long: cls.I32,   # Downcast
            torch.bool: cls.BOOL,
            torch.uint8: cls.I8,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @classmethod
    def from_string(cls, s: str) -> "GGMLDtype":
        """Convert string to GGML dtype."""
        mapping = {
            "f32": cls.F32,
            "f16": cls.F16,
            "i32": cls.I32,
            "i16": cls.I16,
            "i8": cls.I8,
            "bool": cls.BOOL,
        }
        if s not in mapping:
            raise ValueError(f"Unknown dtype string: {s}")
        return mapping[s]


def _sanitize_shape(shape: list) -> list[int | str]:
    """Convert shape to plain integers or symbolic dimension names.

    Symbolic dimensions are preserved as strings (e.g., "n_atoms", "max_neighbors").
    """
    result = []
    for dim in shape:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, str):
            # Symbolic dimension name - preserve it
            result.append(dim)
        else:
            # SymInt or other symbolic type - try to convert to int
            try:
                result.append(int(dim))
            except (TypeError, ValueError):
                result.append(-1)
    return result


def _sanitize_params(params: dict) -> dict:
    """Sanitize parameters for JSON serialization."""
    result = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [_sanitize_value(v) for v in value]
        elif isinstance(value, dict):
            result[key] = _sanitize_params(value)
        else:
            result[key] = _sanitize_value(value)
    return result


def _sanitize_value(value):
    """Sanitize a single value for JSON serialization."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


@dataclass
class GGMLInput:
    """Model input specification."""
    name: str
    dtype: GGMLDtype
    shape: list[int]  # -1 for dynamic dimensions
    dynamic_dims: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "shape": _sanitize_shape(self.shape),
            "dynamic_dims": self.dynamic_dims,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GGMLInput":
        return cls(
            name=d["name"],
            dtype=GGMLDtype(d["dtype"]),
            shape=d["shape"],
            dynamic_dims=d.get("dynamic_dims", []),
        )


@dataclass
class GGMLOutput:
    """Model output specification."""
    name: str
    node_ref: str  # Reference to node that produces this output
    dtype: GGMLDtype
    shape: list[int]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_ref": self.node_ref,
            "dtype": self.dtype.value,
            "shape": _sanitize_shape(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GGMLOutput":
        return cls(
            name=d["name"],
            node_ref=d["node_ref"],
            dtype=GGMLDtype(d["dtype"]),
            shape=d["shape"],
        )


@dataclass
class GGMLNode:
    """A node in the GGML computation graph."""
    id: int
    op: str  # GGML operation name (e.g., "ADD", "MUL_MAT")
    name: str  # Human-readable name for debugging
    inputs: list[str]  # References: "node:N", "input:name", "weight:name"
    output_shape: list[int]
    output_dtype: GGMLDtype
    params: dict[str, Any] = field(default_factory=dict)  # Op-specific parameters

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "op": self.op,
            "name": self.name,
            "inputs": self.inputs,
            "output_shape": _sanitize_shape(self.output_shape),
            "output_dtype": self.output_dtype.value,
        }
        if self.params:
            d["params"] = _sanitize_params(self.params)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GGMLNode":
        return cls(
            id=d["id"],
            op=d["op"],
            name=d["name"],
            inputs=d["inputs"],
            output_shape=d["output_shape"],
            output_dtype=GGMLDtype(d["output_dtype"]),
            params=d.get("params", {}),
        )


@dataclass
class GGMLGraph:
    """Complete GGML computation graph."""
    version: str = "1.0.0"
    model_type: str = "generic"
    inputs: list[GGMLInput] = field(default_factory=list)
    outputs: list[GGMLOutput] = field(default_factory=list)
    nodes: list[GGMLNode] = field(default_factory=list)
    constants: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # For tracking during graph construction
    _next_node_id: int = field(default=0, repr=False)
    _node_name_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def add_input(self, name: str, dtype: GGMLDtype, shape: list[int],
                  dynamic_dims: list[int] | None = None) -> GGMLInput:
        """Add an input specification."""
        inp = GGMLInput(
            name=name,
            dtype=dtype,
            shape=shape,
            dynamic_dims=dynamic_dims or [],
        )
        self.inputs.append(inp)
        return inp

    def add_output(self, name: str, node_ref: str, dtype: GGMLDtype,
                   shape: list[int]) -> GGMLOutput:
        """Add an output specification."""
        out = GGMLOutput(
            name=name,
            node_ref=node_ref,
            dtype=dtype,
            shape=shape,
        )
        self.outputs.append(out)
        return out

    def add_node(self, op: str, name: str, inputs: list[str],
                 output_shape: list[int], output_dtype: GGMLDtype,
                 params: dict[str, Any] | None = None) -> GGMLNode:
        """Add a computation node."""
        # Generate unique name if needed
        if name in self._node_name_counts:
            self._node_name_counts[name] += 1
            unique_name = f"{name}_{self._node_name_counts[name]}"
        else:
            self._node_name_counts[name] = 0
            unique_name = name

        node = GGMLNode(
            id=self._next_node_id,
            op=op,
            name=unique_name,
            inputs=inputs,
            output_shape=output_shape,
            output_dtype=output_dtype,
            params=params or {},
        )
        self.nodes.append(node)
        self._next_node_id += 1
        return node

    def node_ref(self, node: GGMLNode) -> str:
        """Get the reference string for a node."""
        return f"node:{node.id}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "$schema": "ggml-graph-v1",
            "version": self.version,
            "model_type": self.model_type,
            "metadata": self.metadata,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "constants": self.constants,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "GGMLGraph":
        """Create graph from dictionary."""
        graph = cls(
            version=d.get("version", "1.0.0"),
            model_type=d.get("model_type", "generic"),
            metadata=d.get("metadata", {}),
            constants=d.get("constants", {}),
        )
        graph.inputs = [GGMLInput.from_dict(i) for i in d.get("inputs", [])]
        graph.outputs = [GGMLOutput.from_dict(o) for o in d.get("outputs", [])]
        graph.nodes = [GGMLNode.from_dict(n) for n in d.get("nodes", [])]
        if graph.nodes:
            graph._next_node_id = max(n.id for n in graph.nodes) + 1
        return graph

    @classmethod
    def from_json(cls, json_str: str) -> "GGMLGraph":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"GGMLGraph(model_type={self.model_type!r}, "
            f"inputs={len(self.inputs)}, outputs={len(self.outputs)}, "
            f"nodes={len(self.nodes)})"
        )

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [
            f"GGML Graph v{self.version}",
            f"Model type: {self.model_type}",
            "",
            "Inputs:",
        ]
        for inp in self.inputs:
            shape_str = str(_sanitize_shape(inp.shape))
            lines.append(f"  {inp.name}: {inp.dtype.value} {shape_str}")

        lines.append("")
        lines.append("Outputs:")
        for out in self.outputs:
            shape_str = str(_sanitize_shape(out.shape))
            lines.append(f"  {out.name}: {out.dtype.value} {shape_str} <- {out.node_ref}")

        lines.append("")
        lines.append(f"Nodes: {len(self.nodes)}")

        # Count ops
        op_counts: dict[str, int] = {}
        for node in self.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1

        lines.append("Operation counts:")
        for op, count in sorted(op_counts.items()):
            lines.append(f"  {op}: {count}")

        return "\n".join(lines)
