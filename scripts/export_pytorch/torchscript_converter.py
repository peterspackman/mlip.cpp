"""Convert TorchScript graphs to GGML IR (GIR) format."""

import json
import torch
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from .graph_ir import GGMLDtype, GGMLGraph, GGMLNode, GGMLInput, GGMLOutput


# TorchScript op to GGML op mapping (values are GGML op string names)
TS_TO_GGML = {
    # Linear algebra
    "aten::linear": "MUL_MAT",  # linear(x, W, b) = x @ W.T + b
    "aten::mm": "MUL_MAT",
    "aten::bmm": "MUL_MAT",
    "aten::matmul": "MUL_MAT",

    # Element-wise
    "aten::add": "ADD",
    "aten::sub": "SUB",
    "aten::mul": "MUL",
    "aten::div": "DIV",

    # Unary
    "aten::silu": "UNARY_SILU",
    "aten::relu": "UNARY_RELU",
    "aten::gelu": "UNARY_GELU",
    "aten::tanh": "UNARY_TANH",
    "aten::exp": "UNARY_EXP",
    "aten::neg": "UNARY_NEG",
    "aten::sqrt": "SQRT",
    "aten::log": "LOG",
    "aten::rsqrt": "RSQRT",  # 1/sqrt(x)

    # Shape ops
    "aten::reshape": "RESHAPE",
    "aten::view": "VIEW",
    "aten::permute": "PERMUTE",
    "aten::transpose": "TRANSPOSE",
    "aten::contiguous": "CONT",
    "aten::select": "VIEW",  # Select single index
    "aten::slice": "VIEW",   # Slice range
    "aten::unsqueeze": "RESHAPE",  # Add dimension
    "aten::squeeze": "RESHAPE",    # Remove dimension
    "aten::flatten": "RESHAPE",
    "aten::expand": "REPEAT",
    "aten::repeat": "REPEAT",

    # Reduction
    "aten::sum": "SUM_ROWS",
    "aten::mean": "MEAN",

    # Attention
    "aten::scaled_dot_product_attention": "FLASH_ATTN_EXT",
    "aten::softmax": "SOFT_MAX",
    "aten::_softmax": "SOFT_MAX",

    # Other
    "aten::clamp": "CLAMP",
    "aten::layer_norm": "DECOMPOSE",  # Needs decomposition
    "aten::native_layer_norm": "DECOMPOSE",

    # Skip ops (no tensor output, just metadata)
    "aten::size": None,
    "aten::Int": None,
    "aten::__getitem__": None,
    "prim::NumToTensor": None,
}


@dataclass
class TSNode:
    """Parsed TorchScript node."""
    kind: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    scope: str = ""


def parse_ts_graph(graph: torch.Graph) -> tuple[list[TSNode], dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    """Parse a TorchScript graph into nodes, constants, and weight names.

    Returns:
        Tuple of (nodes, tensor_constants dict, weight_names dict, scalar_constants dict)
    """
    nodes = []
    tensor_constants = {}
    scalar_constants = {}  # For shapes, indices, scalars
    weight_names = {}  # Map from debug name to meaningful weight name

    for node in graph.nodes():
        kind = node.kind()

        # Get inputs
        inputs = []
        for inp in node.inputs():
            inputs.append(inp.debugName())

        # Get outputs
        outputs = []
        for out in node.outputs():
            outputs.append(out.debugName())

        # Get attributes
        attrs = {}
        for attr_name in node.attributeNames():
            attr_kind = node.kindOf(attr_name)
            if attr_kind == 'i':
                attrs[attr_name] = node.i(attr_name)
            elif attr_kind == 'f':
                attrs[attr_name] = node.f(attr_name)
            elif attr_kind == 's':
                attrs[attr_name] = node.s(attr_name)
            elif attr_kind == 'is':
                attrs[attr_name] = list(node.is_(attr_name))
            elif attr_kind == 't':
                # Tensor constant
                tensor = node.t(attr_name)
                attrs[attr_name] = tensor

        # Handle prim::Constant specially
        if kind == "prim::Constant":
            if 'value' in attrs:
                val = attrs['value']
                debug_name = outputs[0]
                if isinstance(val, torch.Tensor):
                    tensor_constants[debug_name] = val
                    # Try to extract meaningful weight name from the variable name
                    # TorchScript names look like "self.transformer.layers.0.mlp.0.weight"
                    if debug_name.startswith("self."):
                        # Clean up the name
                        weight_name = debug_name[5:].replace(".", "_")  # Remove "self."
                        weight_names[debug_name] = weight_name
                    else:
                        weight_names[debug_name] = debug_name
                else:
                    # Scalar or list constant (shapes, indices, etc.)
                    scalar_constants[debug_name] = val

        # Get scope for debugging
        scope = ""
        if node.scopeName():
            scope = node.scopeName()

        nodes.append(TSNode(
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            scope=scope
        ))

    return nodes, tensor_constants, weight_names, scalar_constants


def convert_ts_to_gir(
    traced_model: torch.jit.ScriptModule,
    input_names: list[str] = None,
) -> tuple[GGMLGraph, dict[str, torch.Tensor]]:
    """Convert a traced/frozen TorchScript module to GIR.

    Args:
        traced_model: Frozen TorchScript module
        input_names: Names for input tensors

    Returns:
        Tuple of (GGMLGraph, weights dict)
    """
    graph = traced_model.graph

    # Parse the graph
    ts_nodes, constants, weight_name_map, scalar_constants = parse_ts_graph(graph)

    # Get graph inputs
    graph_inputs = list(graph.inputs())

    # Build GIR
    gir_inputs = []
    gir_nodes = []
    weights = {}

    # Map from TS names to GIR references
    name_map = {}
    node_id = 0

    # Process inputs (skip self)
    for i, inp in enumerate(graph_inputs):
        if i == 0:  # Skip self
            continue
        name = input_names[i-1] if input_names and i-1 < len(input_names) else f"input_{i-1}"
        inp_type = inp.type()

        # Get shape and dtype from type info
        shape = []
        dtype = GGMLDtype.F32
        if hasattr(inp_type, 'sizes') and inp_type.sizes():
            shape = list(inp_type.sizes())
        if hasattr(inp_type, 'dtype'):
            try:
                dt = inp_type.dtype()
                if dt is not None:
                    dtype = GGMLDtype.from_torch_dtype(dt)
            except Exception:
                pass  # Keep default F32

        gir_inputs.append(GGMLInput(
            name=name,
            dtype=dtype,
            shape=shape,
        ))
        name_map[inp.debugName()] = f"input:{name}"

    # Process constants as weights with meaningful names
    for const_name, tensor in constants.items():
        # Use meaningful name from TorchScript if available
        weight_name = weight_name_map.get(const_name, const_name)
        weights[weight_name] = tensor
        name_map[const_name] = f"weight:{weight_name}"

    # Track list constructs that build shapes
    list_values = {}  # Maps list output name to resolved list values

    # Process nodes
    for ts_node in ts_nodes:
        kind = ts_node.kind

        # Skip certain primitives
        if kind in ("prim::Constant", "prim::GetAttr", "prim::TupleConstruct"):
            continue

        # Handle ListConstruct specially - resolve to actual list values
        if kind == "prim::ListConstruct":
            # Build the list from individual scalar constants
            values = []
            for inp in ts_node.inputs:
                if inp in scalar_constants:
                    values.append(scalar_constants[inp])
                else:
                    values.append(None)  # Unknown value
            if values and all(v is not None for v in values):
                list_values[ts_node.outputs[0]] = values
                scalar_constants[ts_node.outputs[0]] = values  # Also add to scalar_constants
            continue

        # Map the operation
        ggml_op = TS_TO_GGML.get(kind)

        # Check if op is explicitly skipped (None in mapping)
        if kind in TS_TO_GGML and ggml_op is None:
            # Skip ops that produce no tensor output (e.g., aten::size)
            continue

        if ggml_op is None:
            print(f"Warning: Unmapped op {kind}")
            continue

        # Get input references
        input_refs = []
        scalar_values = []  # Store resolved scalar values for shape params
        for inp in ts_node.inputs:
            ref = name_map.get(inp)
            if ref is None:
                # Try to get from constants or use placeholder
                if inp in constants:
                    ref = f"weight:{inp}"
                elif inp in scalar_constants:
                    # This is a scalar constant (shape, index, etc.)
                    scalar_values.append((inp, scalar_constants[inp]))
                    ref = f"const:{scalar_constants[inp]}"  # Include the value
                else:
                    ref = f"const:0"  # Placeholder
            input_refs.append(ref)

        # Handle specific ops
        params = {}

        if kind == "aten::linear":
            # linear(input, weight, bias) -> out = input @ weight.T + bias
            # In GGML: mul_mat(weight, input) does input @ weight.T
            # We need to handle bias separately
            bias_ref = None
            if len(input_refs) >= 2:
                # Swap order for GGML (weight first)
                if len(input_refs) > 2:
                    bias_ref = input_refs[2]  # Save bias for later
                input_refs = [input_refs[1], input_refs[0]]  # Just weight and input

            # Create the MUL_MAT node first
            gir_node = GGMLNode(
                id=node_id,
                op=ggml_op,
                name=ts_node.scope.split("/")[-1] if ts_node.scope else f"node_{node_id}",
                inputs=input_refs,
                output_shape=[],
                output_dtype=GGMLDtype.F32,
                params={},
            )
            gir_nodes.append(gir_node)
            matmul_node_id = node_id
            node_id += 1

            # If there's a bias, add an ADD node
            if bias_ref and bias_ref != "const:0":
                gir_node = GGMLNode(
                    id=node_id,
                    op="ADD",
                    name=f"linear_bias_{node_id}",
                    inputs=[f"node:{matmul_node_id}", bias_ref],
                    output_shape=[],
                    output_dtype=GGMLDtype.F32,
                    params={},
                )
                gir_nodes.append(gir_node)

                # Map output to the ADD node, not the MUL_MAT
                for out in ts_node.outputs:
                    name_map[out] = f"node:{node_id}"
                node_id += 1
            else:
                # No bias, map output to MUL_MAT
                for out in ts_node.outputs:
                    name_map[out] = f"node:{matmul_node_id}"
            continue  # Skip the default node creation below

        elif kind == "aten::scaled_dot_product_attention":
            # SDPA(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
            params["scale"] = 1.0  # Will be handled by interpreter

        elif kind == "aten::layer_norm" or kind == "aten::native_layer_norm":
            # layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)
            params["eps"] = ts_node.attrs.get("eps", 1e-5)
            ggml_op = "DECOMPOSE"  # Mark for decomposition

        elif kind == "aten::clamp":
            # Get min/max from scalar_values
            for name, val in scalar_values:
                if isinstance(val, (int, float)):
                    if "min" not in params:
                        params["min"] = float(val)
                    elif "max" not in params:
                        params["max"] = float(val)

        elif kind == "aten::permute":
            if "dims" in ts_node.attrs:
                params["axes"] = ts_node.attrs["dims"]
            else:
                # Extract from scalar_values
                for name, val in scalar_values:
                    if isinstance(val, list):
                        params["axes"] = val
                        break

        elif kind in ("aten::reshape", "aten::view"):
            # Extract shape from scalar_values
            for name, val in scalar_values:
                if isinstance(val, list):
                    params["shape"] = val
                    break

        elif kind == "aten::transpose":
            # Extract dimensions from scalar_values
            dims = []
            for name, val in scalar_values:
                if isinstance(val, int):
                    dims.append(val)
            if dims:
                params["dims"] = dims

        elif kind in ("aten::select", "aten::slice"):
            # Extract dim, start, end from scalar_values
            int_vals = [v for _, v in scalar_values if isinstance(v, int)]
            if int_vals:
                params["dim"] = int_vals[0] if len(int_vals) > 0 else 0
                params["start"] = int_vals[1] if len(int_vals) > 1 else 0
                params["end"] = int_vals[2] if len(int_vals) > 2 else -1

        # Create GIR node
        gir_node = GGMLNode(
            id=node_id,
            op=ggml_op,
            name=ts_node.scope.split("/")[-1] if ts_node.scope else f"node_{node_id}",
            inputs=input_refs,
            output_shape=[],  # Would need type inference
            output_dtype=GGMLDtype.F32,
            params=params,
        )
        gir_nodes.append(gir_node)

        # Map outputs
        for out in ts_node.outputs:
            name_map[out] = f"node:{node_id}"

        node_id += 1

    # Get graph output
    graph_outputs = list(graph.outputs())
    gir_outputs = []
    for out in graph_outputs:
        ref = name_map.get(out.debugName(), f"node:{node_id-1}")
        gir_outputs.append(GGMLOutput(
            name="output",
            node_ref=ref,
            dtype=GGMLDtype.F32,
            shape=[],
        ))

    return GGMLGraph(
        version="1.0.0",
        model_type="torchscript",
        inputs=gir_inputs,
        outputs=gir_outputs,
        nodes=gir_nodes,
    ), weights


def export_torchscript_model(
    module: torch.nn.Module,
    example_inputs: tuple,
    output_path: Path,
    input_names: list[str] = None,
):
    """Export a PyTorch module via TorchScript to GIR.

    Args:
        module: PyTorch module to export
        example_inputs: Example inputs for tracing
        output_path: Path for output JSON
        input_names: Names for inputs
    """
    module.eval()

    # Trace
    traced = torch.jit.trace(module, example_inputs)

    # Freeze to inline everything
    frozen = torch.jit.freeze(traced)

    # Convert
    gir_graph, weights = convert_ts_to_gir(frozen, input_names)

    # Save graph
    with open(output_path, 'w') as f:
        json.dump(gir_graph.to_dict(), f, indent=2)

    # Save weights
    weights_path = output_path.with_suffix('.weights.pt')
    torch.save(weights, weights_path)

    print(f"Saved graph to {output_path}")
    print(f"Saved {len(weights)} weights to {weights_path}")
    print(f"Graph has {len(gir_graph.nodes)} nodes")

    return gir_graph, weights
