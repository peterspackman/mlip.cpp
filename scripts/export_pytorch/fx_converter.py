"""Convert torch.fx graphs to GGML IR (GIR) format with shape inference.

Supports two export modes:
1. fx.symbolic_trace - Fast but limited to static operations
2. torch.export - Handles dynamic operations like torch.empty() with attributes
"""

import json
import operator
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

# Handle both package and direct script execution
try:
    from .graph_ir import GGMLDtype, GGMLGraph, GGMLNode, GGMLInput, GGMLOutput
except ImportError:
    from graph_ir import GGMLDtype, GGMLGraph, GGMLNode, GGMLInput, GGMLOutput


# FX op to GGML op mapping
FX_TO_GGML = {
    # Tensor creation (these create constants/zeros)
    "new_zeros": "NEW_ZEROS",

    # Arithmetic (operator module)
    "add": "ADD",
    "sub": "SUB",
    "mul": "MUL",
    "truediv": "DIV",
    "neg": "UNARY_NEG",

    # torch functions
    "torch.add": "ADD",
    "torch.sub": "SUB",
    "torch.mul": "MUL",
    "torch.div": "DIV",
    "torch.matmul": "MATMUL",
    "torch.mm": "MATMUL",
    "torch.bmm": "MATMUL",
    "torch.clamp": "CLAMP",
    "torch.log": "LOG",
    "torch.exp": "UNARY_EXP",
    "torch.sqrt": "SQRT",
    "torch.rsqrt": "RSQRT",
    "torch.tanh": "UNARY_TANH",
    "torch.softmax": "SOFT_MAX",
    "torch.silu": "UNARY_SILU",
    "silu": "UNARY_SILU",  # torch.nn.functional.silu
    "torch.relu": "UNARY_RELU",
    "torch.gelu": "UNARY_GELU",
    "torch.neg": "UNARY_NEG",
    "torch.cos": "COS",
    "torch.sin": "SIN",
    "torch.sum": "SUM_ROWS",
    "torch.mean": "MEAN",

    # torch._C._nn functions
    "scaled_dot_product_attention": "FLASH_ATTN_EXT",

    # Methods
    "reshape": "RESHAPE",
    "view": "VIEW",
    "permute": "PERMUTE",
    "transpose": "TRANSPOSE",
    "contiguous": "CONT",
    "squeeze": "RESHAPE",
    "unsqueeze": "RESHAPE",
    "flatten": "RESHAPE",
    "expand": "REPEAT",
    "repeat": "REPEAT",
    "clamp": "CLAMP",
    "chunk": "CHUNK",  # Will be decomposed into multiple VIEW operations
}

# ATen op mapping (used by torch.export)
# Note: torch.export uses different name formats depending on PyTorch version
# We handle both "aten.op.variant" and "aten.op" formats
ATEN_TO_GGML = {
    # Arithmetic
    "aten.add.Tensor": "ADD",
    "aten.add.Scalar": "ADD",
    "aten.add": "ADD",
    "aten.sub.Tensor": "SUB",
    "aten.sub.Scalar": "SUB",
    "aten.sub": "SUB",
    "aten.mul.Tensor": "MUL",
    "aten.mul.Scalar": "MUL",
    "aten.mul": "MUL",
    "aten.div.Tensor": "DIV",
    "aten.div.Scalar": "DIV",
    "aten.div": "DIV",
    "aten.neg.default": "UNARY_NEG",
    "aten.neg": "UNARY_NEG",

    # Matrix ops
    "aten.mm.default": "MATMUL",
    "aten.mm": "MATMUL",
    "aten.bmm.default": "MATMUL",
    "aten.bmm": "MATMUL",
    "aten.matmul.default": "MATMUL",
    "aten.matmul": "MATMUL",
    "aten.linear.default": "LINEAR",
    "aten.linear": "LINEAR",
    "aten.t.default": "TRANSPOSE",
    "aten.t": "TRANSPOSE",
    "aten.addmm.default": "ADDMM",
    "aten.addmm": "ADDMM",

    # Activations
    "aten.silu.default": "UNARY_SILU",
    "aten.silu": "UNARY_SILU",
    "aten.relu.default": "UNARY_RELU",
    "aten.relu": "UNARY_RELU",
    "aten.gelu.default": "UNARY_GELU",
    "aten.gelu": "UNARY_GELU",
    "aten.tanh.default": "UNARY_TANH",
    "aten.tanh": "UNARY_TANH",
    "aten.sigmoid.default": "UNARY_SIGMOID",
    "aten.sigmoid": "UNARY_SIGMOID",

    # Math ops
    "aten.exp.default": "UNARY_EXP",
    "aten.exp": "UNARY_EXP",
    "aten.log.default": "LOG",
    "aten.log": "LOG",
    "aten.sqrt.default": "SQRT",
    "aten.sqrt": "SQRT",
    "aten.rsqrt.default": "RSQRT",
    "aten.rsqrt": "RSQRT",
    "aten.pow.Tensor_Scalar": "POW",
    "aten.pow": "POW",
    "aten.mean.dim": "MEAN",
    "aten.mean": "MEAN",
    "aten.sum.dim_IntList": "SUM_ROWS",
    "aten.sum.default": "SUM_ROWS",
    "aten.sum": "SUM_ROWS",
    "aten.cos.default": "COS",
    "aten.cos": "COS",
    "aten.sin.default": "SIN",
    "aten.sin": "SIN",
    "aten.clamp.default": "CLAMP",
    "aten.clamp": "CLAMP",

    # Shape ops
    "aten.view.default": "VIEW",
    "aten.view": "VIEW",
    "aten.reshape.default": "RESHAPE",
    "aten.reshape": "RESHAPE",
    "aten._unsafe_view.default": "VIEW",
    "aten._unsafe_view": "VIEW",
    "aten.permute.default": "PERMUTE",
    "aten.permute": "PERMUTE",
    "aten.transpose.int": "TRANSPOSE",
    "aten.transpose": "TRANSPOSE",
    "aten.contiguous.default": "CONT",
    "aten.contiguous": "CONT",
    "aten.squeeze.dim": "RESHAPE",
    "aten.squeeze.default": "RESHAPE",
    "aten.squeeze": "RESHAPE",
    "aten.unsqueeze.default": "RESHAPE",
    "aten.unsqueeze": "RESHAPE",
    "aten.flatten.using_ints": "RESHAPE",
    "aten.flatten": "RESHAPE",
    "aten.expand.default": "REPEAT",
    "aten.expand": "REPEAT",
    "aten.repeat.default": "REPEAT",
    "aten.repeat": "REPEAT",
    "aten.clone.default": "CONT",
    "aten.clone": "CONT",

    # Reduction/pooling
    "aten.softmax.int": "SOFT_MAX",
    "aten.softmax": "SOFT_MAX",
    "aten._softmax.default": "SOFT_MAX",
    "aten._softmax": "SOFT_MAX",

    # Embedding/indexing
    "aten.embedding.default": "GET_ROWS",
    "aten.embedding": "GET_ROWS",
    "aten.index_select.default": "GET_ROWS",
    "aten.index_select": "GET_ROWS",
    "aten.select.int": "SELECT",
    "aten.select": "SELECT",
    "aten.slice.Tensor": "SLICE",
    "aten.slice": "SLICE",
    "aten.index.Tensor": "INDEX",
    "aten.index": "INDEX",

    # Concat/split
    "aten.cat.default": "CONCAT",
    "aten.cat": "CONCAT",
    "aten.split.Tensor": "SPLIT",
    "aten.split": "SPLIT",
    "aten.split_with_sizes": "SPLIT",
    "aten.chunk.default": "CHUNK",
    "aten.chunk": "CHUNK",

    # Layer norm
    "aten.layer_norm.default": "LAYER_NORM",
    "aten.layer_norm": "LAYER_NORM",
    "aten.native_layer_norm.default": "LAYER_NORM",
    "aten.native_layer_norm": "LAYER_NORM",
    "aten.rms_norm.default": "RMS_NORM",
    "aten.rms_norm": "RMS_NORM",

    # Attention
    "aten.scaled_dot_product_attention.default": "FLASH_ATTN_EXT",
    "aten.scaled_dot_product_attention": "FLASH_ATTN_EXT",
    "aten._scaled_dot_product_flash_attention.default": "FLASH_ATTN_EXT",
    "aten._scaled_dot_product_flash_attention": "FLASH_ATTN_EXT",

    # Tensor creation
    "aten.zeros.default": "NEW_ZEROS",
    "aten.zeros": "NEW_ZEROS",
    "aten.zeros_like.default": "NEW_ZEROS",
    "aten.zeros_like": "NEW_ZEROS",
    "aten.ones.default": "NEW_ONES",
    "aten.ones_like.default": "NEW_ONES",
    "aten.ones": "NEW_ONES",
    "aten.empty.memory_format": "NEW_ZEROS",
    "aten.empty": "NEW_ZEROS",
    "aten.new_zeros": "NEW_ZEROS",
    "aten.new_ones": "NEW_ONES",

    # Comparison/mask
    "aten.where.self": "WHERE",
    "aten.where": "WHERE",
    "aten.masked_fill.Scalar": "MASKED_FILL",
    "aten.masked_fill": "MASKED_FILL",
    "aten.bitwise_not": "BITWISE_NOT",

    # Copy
    "aten.copy_.default": "COPY",
    "aten.copy_": "COPY",
    "aten.copy": "COPY",

    # Internal ops (pass through)
    "aten.lift_fresh_copy": "CONT",
    "aten.index_put_": "INDEX_PUT",
}


@dataclass
class FXNodeInfo:
    """Information about an FX node for GIR conversion."""
    name: str
    op_type: str  # placeholder, call_module, call_function, call_method, output
    target: Any
    args: Tuple
    kwargs: Dict
    shape: Optional[List[int]] = None
    dtype: GGMLDtype = GGMLDtype.F32


def get_target_name(target) -> str:
    """Get string name from FX target."""
    if isinstance(target, str):
        return target
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def convert_fx_to_gir(
    traced_module: fx.GraphModule,
    input_shapes: Dict[str, List[int]],
    input_names: List[str] = None,
    strict_mode: bool = False,
) -> Tuple[GGMLGraph, Dict[str, torch.Tensor]]:
    """Convert a traced FX graph module to GIR.

    Args:
        traced_module: FX traced and shape-propagated module
        input_shapes: Dict mapping input names to shapes
        input_names: Optional list of input names
        strict_mode: If True, raise errors on unhandled ops instead of passing through

    Returns:
        Tuple of (GGMLGraph, weights dict)
    """
    gir_inputs = []
    gir_nodes = []
    gir_outputs = []
    weights = {}

    # Map from FX node name to GIR reference
    name_map: Dict[str, str] = {}
    node_id = 0

    # Process all nodes
    for node in traced_module.graph.nodes:
        shape = None
        dtype = GGMLDtype.F32  # Default dtype
        if "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            if hasattr(meta, "shape"):
                shape = list(meta.shape)
            if hasattr(meta, "dtype"):
                try:
                    dtype = GGMLDtype.from_torch_dtype(meta.dtype)
                except ValueError:
                    dtype = GGMLDtype.F32

        if node.op == "placeholder":
            # Input tensor
            inp_name = node.name if input_names is None else input_names[len(gir_inputs)]
            gir_inputs.append(GGMLInput(
                name=inp_name,
                dtype=dtype,
                shape=shape or [],
            ))
            name_map[node.name] = f"input:{inp_name}"

        elif node.op == "call_module":
            # Module call (Linear, LayerNorm, etc.)
            module = traced_module.get_submodule(node.target)
            module_type = type(module).__name__

            if isinstance(module, torch.nn.Linear):
                # Linear: y = x @ W.T + b
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")

                # Extract weight and bias
                weight_name = f"{node.target.replace('.', '_')}_weight"
                weights[weight_name] = module.weight.data.clone()

                # MUL_MAT node
                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="MUL_MAT",
                    name=f"{node.name}_matmul",
                    inputs=[f"weight:{weight_name}", input_ref],
                    output_shape=shape[:-1] + [module.out_features] if shape else [],
                    output_dtype=GGMLDtype.F32,
                ))
                matmul_id = node_id
                node_id += 1

                # ADD bias if present
                if module.bias is not None:
                    bias_name = f"{node.target.replace('.', '_')}_bias"
                    weights[bias_name] = module.bias.data.clone()
                    gir_nodes.append(GGMLNode(
                        id=node_id,
                        op="ADD",
                        name=f"{node.name}_bias",
                        inputs=[f"node:{matmul_id}", f"weight:{bias_name}"],
                        output_shape=shape or [],
                        output_dtype=GGMLDtype.F32,
                    ))
                    name_map[node.name] = f"node:{node_id}"
                    node_id += 1
                else:
                    name_map[node.name] = f"node:{matmul_id}"

            elif isinstance(module, torch.nn.LayerNorm):
                # LayerNorm decomposition
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")

                weight_name = f"{node.target.replace('.', '_')}_weight"
                bias_name = f"{node.target.replace('.', '_')}_bias"
                weights[weight_name] = module.weight.data.clone()
                if module.bias is not None:
                    weights[bias_name] = module.bias.data.clone()

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="LAYER_NORM",
                    name=node.name,
                    inputs=[input_ref, f"weight:{weight_name}",
                            f"weight:{bias_name}" if module.bias is not None else "const:0"],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                    params={"eps": module.eps},
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1

            elif isinstance(module, torch.nn.Embedding):
                # Embedding: lookup table using indices
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")

                weight_name = f"{node.target.replace('.', '_')}_weight"
                weights[weight_name] = module.weight.data.clone()

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="GET_ROWS",
                    name=node.name,
                    inputs=[f"weight:{weight_name}", input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1

            elif isinstance(module, torch.nn.SiLU):
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="UNARY_SILU",
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1

            elif isinstance(module, torch.nn.ReLU):
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="UNARY_RELU",
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1

            elif isinstance(module, torch.nn.GELU):
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="UNARY_GELU",
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1

            elif isinstance(module, torch.nn.Dropout):
                # Skip dropout (identity in eval mode)
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                name_map[node.name] = input_ref

            elif hasattr(module, "weight") or hasattr(module, "bias"):
                # Generic module with parameters - try to handle
                if strict_mode:
                    raise ValueError(f"Unhandled module type {module_type} at {node.target}")
                print(f"Warning: Unhandled module type {module_type} at {node.target}")
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                name_map[node.name] = input_ref

            else:
                # Pass-through for unknown modules
                if strict_mode:
                    raise ValueError(f"Unhandled module type {module_type} at {node.target}")
                if node.args:
                    input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                    name_map[node.name] = input_ref

        elif node.op == "call_function":
            target_name = get_target_name(node.target)
            ggml_op = None

            # Check for known functions
            if node.target == operator.add:
                ggml_op = "ADD"
            elif node.target == operator.sub:
                ggml_op = "SUB"
            elif node.target == operator.mul:
                ggml_op = "MUL"
            elif node.target == operator.truediv:
                ggml_op = "DIV"
            elif node.target == operator.getitem:
                # Handle tensor indexing
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                idx = node.args[1]
                if isinstance(idx, int):
                    # Simple integer index -> VIEW
                    gir_nodes.append(GGMLNode(
                        id=node_id,
                        op="VIEW",
                        name=node.name,
                        inputs=[input_ref],
                        output_shape=shape or [],
                        output_dtype=GGMLDtype.F32,
                        params={"index": idx},
                    ))
                    name_map[node.name] = f"node:{node_id}"
                    node_id += 1
                elif isinstance(idx, tuple):
                    # Check for pattern like [:, 0, :] = (slice(None), int, slice(None))
                    # This selects a specific index from a dimension
                    select_dim = None
                    select_idx = None
                    for i, item in enumerate(idx):
                        if isinstance(item, int):
                            select_dim = i
                            select_idx = item
                        elif isinstance(item, slice) and item == slice(None):
                            continue  # Full slice, no-op
                        else:
                            # Complex slice pattern, not yet supported
                            select_dim = None
                            break

                    if select_dim is not None:
                        # Emit SELECT operation
                        gir_nodes.append(GGMLNode(
                            id=node_id,
                            op="SELECT",
                            name=node.name,
                            inputs=[input_ref],
                            output_shape=shape or [],
                            output_dtype=GGMLDtype.F32,
                            params={"dim": select_dim, "index": select_idx},
                        ))
                        name_map[node.name] = f"node:{node_id}"
                        node_id += 1
                    else:
                        # Complex slice - pass through for now
                        name_map[node.name] = input_ref
                else:
                    # Unknown index type - pass through
                    name_map[node.name] = input_ref
                continue
            elif "scaled_dot_product_attention" in target_name:
                ggml_op = "FLASH_ATTN_EXT"
            elif node.target == torch.cat:
                # torch.cat([a, b, ...], dim=0)
                # First arg is a list/tuple of tensors to concatenate
                tensors_arg = node.args[0]
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

                input_refs = []
                for tensor in tensors_arg:
                    if isinstance(tensor, fx.Node):
                        ref = name_map.get(tensor.name)
                        if ref:
                            input_refs.append(ref)

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="CONCAT",
                    name=node.name,
                    inputs=input_refs,
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                    params={"dim": dim},
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
                continue
            elif node.target == torch.clamp:
                ggml_op = "CLAMP"
            elif node.target == torch.chunk or "chunk" in target_name:
                # torch.chunk(input, chunks, dim=0) -> split into chunks pieces along dim
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                num_chunks = node.args[1] if len(node.args) > 1 else 2
                dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="CHUNK",
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                    params={"num_chunks": num_chunks, "dim": dim},
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
                continue
            elif node.target == torch.log:
                ggml_op = "LOG"
            elif node.target == torch.exp:
                ggml_op = "UNARY_EXP"
            elif node.target == torch.sqrt:
                ggml_op = "SQRT"
            elif node.target == torch.tanh:
                ggml_op = "UNARY_TANH"
            elif node.target == torch.softmax:
                ggml_op = "SOFT_MAX"
            elif target_name in FX_TO_GGML:
                ggml_op = FX_TO_GGML[target_name]
            elif hasattr(node.target, "__name__") and node.target.__name__ in FX_TO_GGML:
                ggml_op = FX_TO_GGML[node.target.__name__]

            if ggml_op:
                # Build input refs
                input_refs = []
                params = {}

                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        ref = name_map.get(arg.name)
                        if ref:
                            input_refs.append(ref)
                    elif isinstance(arg, (int, float)):
                        # Scalar parameter
                        if ggml_op == "CLAMP":
                            if "min" not in params:
                                params["min"] = float(arg)
                            else:
                                params["max"] = float(arg)

                # Handle kwargs
                if "attn_mask" in node.kwargs:
                    mask_node = node.kwargs["attn_mask"]
                    if isinstance(mask_node, fx.Node):
                        mask_ref = name_map.get(mask_node.name)
                        if mask_ref:
                            input_refs.append(mask_ref)

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op=ggml_op,
                    name=node.name,
                    inputs=input_refs,
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                    params=params if params else None,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
            elif "getattr" in target_name.lower():
                # Attribute access (like .shape) - skip
                pass
            else:
                if strict_mode:
                    raise ValueError(f"Unhandled function {target_name}")
                print(f"Warning: Unhandled function {target_name}")

        elif node.op == "call_method":
            method_name = node.target
            ggml_op = FX_TO_GGML.get(method_name)

            if method_name == "new_zeros":
                # tensor.new_zeros(size) - create a zero tensor
                # We'll handle this as a constant zero creation
                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op="NEW_ZEROS",
                    name=node.name,
                    inputs=[],  # No inputs - creates zeros
                    output_shape=shape or [],
                    output_dtype=dtype,
                    params={"shape": shape or []},
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
                continue

            if ggml_op:
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                params = {}

                if method_name == "reshape":
                    # Extract shape from args
                    shape_args = []
                    for arg in node.args[1:]:
                        if isinstance(arg, int):
                            shape_args.append(arg)
                        elif isinstance(arg, fx.Node):
                            # Dynamic shape - use the propagated output shape
                            pass
                    if shape_args:
                        params["shape"] = shape_args
                    elif shape:
                        params["shape"] = shape

                elif method_name == "permute":
                    # Extract permutation from args
                    perm = [arg for arg in node.args[1:] if isinstance(arg, int)]
                    if perm:
                        params["axes"] = perm

                elif method_name == "transpose":
                    # Extract dimensions
                    dims = [arg for arg in node.args[1:] if isinstance(arg, int)]
                    if dims:
                        params["dims"] = dims

                elif method_name == "view":
                    # Similar to reshape
                    shape_args = [arg for arg in node.args[1:] if isinstance(arg, int)]
                    if shape_args:
                        params["shape"] = shape_args
                    elif shape:
                        params["shape"] = shape

                elif method_name == "clamp":
                    # Extract min/max from args or kwargs
                    if len(node.args) > 1:
                        params["min"] = float(node.args[1]) if node.args[1] is not None else None
                    if len(node.args) > 2:
                        params["max"] = float(node.args[2]) if node.args[2] is not None else None
                    if "min" in node.kwargs:
                        params["min"] = float(node.kwargs["min"])
                    if "max" in node.kwargs:
                        params["max"] = float(node.kwargs["max"])

                elif method_name == "chunk":
                    # chunk returns a tuple - subsequent getitem ops extract pieces
                    # Store the chunk info for downstream getitem handling
                    num_chunks = node.args[1] if len(node.args) > 1 else 1
                    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
                    params["num_chunks"] = num_chunks
                    params["dim"] = dim
                    # Pass through - getitem will extract pieces
                    name_map[node.name] = input_ref
                    continue

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op=ggml_op,
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=GGMLDtype.F32,
                    params=params if params else None,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
            else:
                if strict_mode:
                    raise ValueError(f"Unhandled method {method_name}")
                print(f"Warning: Unhandled method {method_name}")
                if node.args:
                    input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                    name_map[node.name] = input_ref

        elif node.op == "output":
            # Graph output
            for arg in node.args[0] if isinstance(node.args[0], tuple) else [node.args[0]]:
                if isinstance(arg, fx.Node):
                    ref = name_map.get(arg.name, f"node:{node_id-1}")
                    gir_outputs.append(GGMLOutput(
                        name="output",
                        node_ref=ref,
                        dtype=GGMLDtype.F32,
                        shape=shape or [],
                    ))

    return GGMLGraph(
        version="1.0.0",
        model_type="fx",
        inputs=gir_inputs,
        outputs=gir_outputs,
        nodes=gir_nodes,
    ), weights


def export_fx_model(
    module: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str] = None,
) -> Tuple[GGMLGraph, Dict[str, torch.Tensor]]:
    """Export a PyTorch module via torch.fx to GIR with shape inference.

    Args:
        module: PyTorch module to export
        example_inputs: Example inputs for tracing and shape propagation
        output_path: Path for output JSON
        input_names: Names for inputs

    Returns:
        Tuple of (GGMLGraph, weights dict)
    """
    module.eval()

    # FX symbolic trace
    traced = fx.symbolic_trace(module)

    # Propagate shapes
    ShapeProp(traced).propagate(*example_inputs)

    # Build input shapes dict
    input_shapes = {}
    for i, (name, inp) in enumerate(zip(input_names or [], example_inputs)):
        input_shapes[name] = list(inp.shape)

    # Convert to GIR
    gir_graph, weights = convert_fx_to_gir(traced, input_shapes, input_names)

    # Save graph
    with open(output_path, "w") as f:
        json.dump(gir_graph.to_dict(), f, indent=2)

    # Save weights
    weights_path = output_path.with_suffix(".weights.pt")
    torch.save(weights, weights_path)

    print(f"Saved graph to {output_path}")
    print(f"Saved {len(weights)} weights to {weights_path}")
    print(f"Graph has {len(gir_graph.nodes)} nodes")

    return gir_graph, weights


def get_aten_op_name(target) -> str:
    """Get the ATen op name string from an FX target."""
    # Try common attribute patterns
    if hasattr(target, "_name"):
        name = target._name
        # Remove "aten::" prefix if present
        if name.startswith("aten::"):
            name = name[6:]
        return f"aten.{name}"

    if hasattr(target, "name"):
        # OpOverload: e.g., torch.ops.aten.add.Tensor
        try:
            name = target.name()
            # Some ops return "aten::add.Tensor" format
            if "::" in name:
                parts = name.split("::")
                name = parts[-1]  # e.g., "add.Tensor"
            return f"aten.{name}"
        except:
            pass

    # Handle string representation like "aten.aten::embedding"
    target_str = str(target)
    if "aten::" in target_str:
        # Extract the op name: "aten.aten::embedding" -> "embedding"
        parts = target_str.split("aten::")
        if len(parts) >= 2:
            name = parts[-1].split()[0]  # Get first word
            return f"aten.{name}"

    return target_str


def convert_exported_to_gir(
    exported_module: fx.GraphModule,
    input_shapes: Dict[str, List[int]],
    input_names: List[str] = None,
    input_dtypes: Dict[str, GGMLDtype] = None,
    pre_extracted_weights: Dict[str, torch.Tensor] = None,
    strict_mode: bool = False,
) -> Tuple[GGMLGraph, Dict[str, torch.Tensor]]:
    """Convert a torch.export exported graph to GIR.

    This handles the ATen ops produced by torch.export instead of the
    higher-level ops from fx.symbolic_trace.

    Args:
        exported_module: FX GraphModule from torch.export
        input_shapes: Dict mapping input names to shapes
        input_names: Optional list of input names
        input_dtypes: Optional dict mapping input names to dtypes
        pre_extracted_weights: Weights already extracted from ExportedProgram.state_dict
        strict_mode: If True, raise errors on unhandled ops instead of passing through

    Returns:
        Tuple of (GGMLGraph, weights dict)
    """
    gir_inputs = []
    gir_nodes = []
    gir_outputs = []
    weights = pre_extracted_weights.copy() if pre_extracted_weights else {}

    # Map from FX node name to GIR reference
    name_map: Dict[str, str] = {}
    node_id = 0

    # Track placeholder count for input names (excluding parameter placeholders)
    placeholder_idx = 0

    # Get any additional parameters and buffers
    for name, param in exported_module.named_parameters():
        weight_name = name.replace(".", "_")
        if weight_name not in weights:
            weights[weight_name] = param.data.clone()

    for name, buf in exported_module.named_buffers():
        weight_name = name.replace(".", "_")
        if weight_name not in weights:
            weights[weight_name] = buf.data.clone()

    # Process all nodes
    for node in exported_module.graph.nodes:
        shape = None
        dtype = GGMLDtype.F32  # Default dtype

        # Try to get shape from various meta formats
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "shape"):
                shape = list(val.shape)
            if hasattr(val, "dtype"):
                try:
                    dtype = GGMLDtype.from_torch_dtype(val.dtype)
                except ValueError:
                    dtype = GGMLDtype.F32
        elif "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            if hasattr(meta, "shape"):
                shape = list(meta.shape)
            if hasattr(meta, "dtype"):
                try:
                    dtype = GGMLDtype.from_torch_dtype(meta.dtype)
                except ValueError:
                    dtype = GGMLDtype.F32

        if node.op == "placeholder":
            # torch.export lifts parameters as placeholders with p_ prefix
            # and constants with c_ prefix
            node_target = str(node.target)
            if node_target.startswith("p_") or node_target.startswith("c_"):
                # This is a lifted parameter or constant - treat as weight
                # The state_dict key matches the original module path
                # p_node_embedders_0_weight -> node_embedders.0.weight in state_dict
                # But we already converted state_dict keys to use underscores
                weight_name = node_target[2:]  # Remove p_ or c_ prefix
                name_map[node.name] = f"weight:{weight_name}"
            else:
                # This is an actual input
                inp_name = input_names[placeholder_idx] if input_names and placeholder_idx < len(input_names) else node.name
                inp_dtype = input_dtypes.get(inp_name, dtype) if input_dtypes else dtype
                gir_inputs.append(GGMLInput(
                    name=inp_name,
                    dtype=inp_dtype,
                    shape=shape or [],
                ))
                name_map[node.name] = f"input:{inp_name}"
                placeholder_idx += 1

        elif node.op == "get_attr":
            # Attribute access (parameters/buffers)
            attr_name = node.target.replace(".", "_")
            # Already in weights dict, just record mapping
            name_map[node.name] = f"weight:{attr_name}"

        elif node.op == "call_function":
            target_name = get_aten_op_name(node.target)
            ggml_op = ATEN_TO_GGML.get(target_name)

            # Handle special cases
            if node.target == operator.getitem:
                # getitem is used for tuple unpacking (e.g., after split/chunk)
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                idx = node.args[1]
                if isinstance(idx, int):
                    # Check if input is from a CHUNK node - need to compute proper shape
                    chunk_output_shape = shape or []
                    input_node = node.args[0]
                    if isinstance(input_node, fx.Node) and hasattr(input_node, 'target'):
                        # Use str() to get target name - works for both OpOverload and regular targets
                        input_target_name = str(input_node.target)
                        if "chunk" in input_target_name.lower():
                            # This is getitem after chunk - compute output shape
                            # Get chunk params from the input node
                            chunk_num = input_node.args[1] if len(input_node.args) > 1 else 2
                            chunk_dim = input_node.args[2] if len(input_node.args) > 2 else -1
                            # Get input tensor shape from chunk's input
                            if len(input_node.args) > 0 and isinstance(input_node.args[0], fx.Node):
                                chunk_input = input_node.args[0]
                                if "val" in chunk_input.meta and hasattr(chunk_input.meta["val"], "shape"):
                                    input_shape = list(chunk_input.meta["val"].shape)
                                    # Compute chunk output shape
                                    if chunk_dim < 0:
                                        chunk_dim = len(input_shape) + chunk_dim
                                    if 0 <= chunk_dim < len(input_shape):
                                        chunk_size = input_shape[chunk_dim] // chunk_num
                                        chunk_output_shape = input_shape.copy()
                                        chunk_output_shape[chunk_dim] = chunk_size

                    gir_nodes.append(GGMLNode(
                        id=node_id,
                        op="VIEW",
                        name=node.name,
                        inputs=[input_ref],
                        output_shape=chunk_output_shape,
                        output_dtype=dtype,
                        params={"index": idx},
                    ))
                    name_map[node.name] = f"node:{node_id}"
                    node_id += 1
                else:
                    name_map[node.name] = input_ref
                continue

            if not ggml_op:
                # Try FX mapping as fallback
                short_name = target_name.split(".")[-1].split("_")[0] if "." in target_name else target_name
                ggml_op = FX_TO_GGML.get(short_name)

            if not ggml_op:
                if strict_mode:
                    raise ValueError(f"Unhandled ATen op: {target_name}")
                print(f"Warning: Unhandled ATen op {target_name}")
                # Try to pass through
                if node.args and isinstance(node.args[0], fx.Node):
                    name_map[node.name] = name_map.get(node.args[0].name, f"node:{node_id-1}")
                continue

            # Build input refs and params
            input_refs = []
            params = {}

            for arg in node.args:
                if isinstance(arg, fx.Node):
                    ref = name_map.get(arg.name)
                    if ref:
                        input_refs.append(ref)
                elif isinstance(arg, (list, tuple)):
                    # Could be a shape list or tensor list
                    for item in arg:
                        if isinstance(item, fx.Node):
                            ref = name_map.get(item.name)
                            if ref:
                                input_refs.append(ref)
                        elif isinstance(item, (int, float)):
                            if "shape" not in params:
                                params["shape"] = []
                            params["shape"].append(item)

            # Handle specific ops
            if ggml_op == "VIEW" or ggml_op == "RESHAPE":
                # Shape is usually in args[1] or the rest of args
                if len(node.args) > 1:
                    shape_arg = node.args[1]
                    if isinstance(shape_arg, (list, tuple)):
                        params["shape"] = list(shape_arg)
                    elif isinstance(shape_arg, fx.Node) and shape:
                        params["shape"] = shape

            elif ggml_op == "PERMUTE":
                # Permutation indices
                if len(node.args) > 1:
                    perm = node.args[1]
                    if isinstance(perm, (list, tuple)):
                        params["axes"] = list(perm)

            elif ggml_op == "TRANSPOSE":
                # Transpose dimensions
                if len(node.args) > 1:
                    dims = [node.args[i] for i in range(1, min(3, len(node.args))) if isinstance(node.args[i], int)]
                    if dims:
                        params["dims"] = dims
                elif target_name == "aten.t.default":
                    # 2D transpose: swap dims 0 and 1
                    params["dims"] = [1, 0]

            elif ggml_op == "CONCAT":
                # Cat: inputs are a list in args[0], dim in args[1]
                input_refs = []
                if isinstance(node.args[0], (list, tuple)):
                    for t in node.args[0]:
                        if isinstance(t, fx.Node):
                            ref = name_map.get(t.name)
                            if ref:
                                input_refs.append(ref)
                dim = node.args[1] if len(node.args) > 1 else 0
                params["dim"] = dim

            elif ggml_op == "SOFT_MAX":
                # Softmax dim
                if len(node.args) > 1 and isinstance(node.args[1], int):
                    params["dim"] = node.args[1]

            elif ggml_op == "LAYER_NORM":
                # native_layer_norm: input, normalized_shape, weight, bias, eps
                # Reorder to: input, weight, bias
                if len(node.args) >= 4:
                    inp_ref = name_map.get(node.args[0].name) if isinstance(node.args[0], fx.Node) else None
                    weight_ref = name_map.get(node.args[2].name) if isinstance(node.args[2], fx.Node) else None
                    bias_ref = name_map.get(node.args[3].name) if isinstance(node.args[3], fx.Node) else None
                    eps = node.args[4] if len(node.args) > 4 else 1e-5
                    input_refs = [r for r in [inp_ref, weight_ref, bias_ref] if r]
                    params["eps"] = eps

            elif ggml_op == "RMS_NORM":
                # rms_norm: input, normalized_shape, weight, eps
                # Args: (input, normalized_shape, weight, eps) or similar
                # Reorder to: input, weight
                if len(node.args) >= 3:
                    inp_ref = name_map.get(node.args[0].name) if isinstance(node.args[0], fx.Node) else None
                    # normalized_shape is args[1], weight is args[2]
                    weight_ref = name_map.get(node.args[2].name) if isinstance(node.args[2], fx.Node) else None
                    # When PyTorch RMSNorm has eps=None, torch.export only produces 3 args
                    # (input, normalized_shape, weight) - no eps arg at all
                    # eps=None in PyTorch means effectively 0, but we use a tiny value
                    # for numerical stability in GGML's rsqrt computation
                    if len(node.args) > 3 and node.args[3] is not None:
                        eps = float(node.args[3])
                    else:
                        eps = 1e-8  # eps=None in PyTorch, use tiny value for GGML stability
                    input_refs = [r for r in [inp_ref, weight_ref] if r]
                    params["eps"] = eps

            elif ggml_op == "GET_ROWS":
                # embedding: weight, indices
                if len(node.args) >= 2:
                    weight_ref = name_map.get(node.args[0].name) if isinstance(node.args[0], fx.Node) else None
                    idx_ref = name_map.get(node.args[1].name) if isinstance(node.args[1], fx.Node) else None
                    if weight_ref and idx_ref:
                        input_refs = [weight_ref, idx_ref]

            elif ggml_op == "SELECT":
                # select.int: input, dim, index
                if len(node.args) >= 3:
                    params["dim"] = node.args[1]
                    params["index"] = node.args[2]

            elif ggml_op == "FLASH_ATTN_EXT":
                # scaled_dot_product_attention: q, k, v, attn_mask, dropout_p, is_causal, scale
                if len(node.args) >= 3:
                    q_ref = name_map.get(node.args[0].name) if isinstance(node.args[0], fx.Node) else None
                    k_ref = name_map.get(node.args[1].name) if isinstance(node.args[1], fx.Node) else None
                    v_ref = name_map.get(node.args[2].name) if isinstance(node.args[2], fx.Node) else None
                    input_refs = [r for r in [q_ref, k_ref, v_ref] if r]
                    # Handle mask if present
                    if len(node.args) > 3 and isinstance(node.args[3], fx.Node):
                        mask_ref = name_map.get(node.args[3].name)
                        if mask_ref:
                            input_refs.append(mask_ref)
                    # Handle scale
                    if "scale" in node.kwargs:
                        params["scale"] = float(node.kwargs["scale"])

            elif ggml_op == "CLAMP":
                # clamp: input, min, max
                # Args can be: (input, min, max) or kwargs min/max
                if len(node.args) >= 2 and node.args[1] is not None:
                    params["min"] = float(node.args[1])
                if len(node.args) >= 3 and node.args[2] is not None:
                    params["max"] = float(node.args[2])
                # Also check kwargs
                if "min" in node.kwargs and node.kwargs["min"] is not None:
                    params["min"] = float(node.kwargs["min"])
                if "max" in node.kwargs and node.kwargs["max"] is not None:
                    params["max"] = float(node.kwargs["max"])

            elif ggml_op == "POW":
                # POW: input ** exponent (scalar)
                if len(node.args) >= 2 and isinstance(node.args[1], (int, float)):
                    params["exponent"] = float(node.args[1])

            elif ggml_op in ("MUL", "ADD", "SUB", "DIV"):
                # Binary ops: handle scalar second arg
                if len(node.args) >= 2:
                    if isinstance(node.args[1], (int, float)):
                        params["scalar"] = float(node.args[1])

            elif ggml_op == "CHUNK":
                # chunk: input, num_chunks, dim
                # aten.chunk.default(tensor, num_chunks, dim)
                if len(node.args) >= 2:
                    params["num_chunks"] = node.args[1]
                if len(node.args) >= 3:
                    params["dim"] = node.args[2]
                elif "dim" in node.kwargs:
                    params["dim"] = node.kwargs["dim"]
                else:
                    params["dim"] = 0  # default dim

            gir_nodes.append(GGMLNode(
                id=node_id,
                op=ggml_op,
                name=node.name,
                inputs=input_refs,
                output_shape=shape or [],
                output_dtype=dtype,
                params=params if params else None,
            ))
            name_map[node.name] = f"node:{node_id}"
            node_id += 1

        elif node.op == "call_method":
            method_name = node.target
            ggml_op = FX_TO_GGML.get(method_name)

            if ggml_op:
                input_ref = name_map.get(node.args[0].name, f"node:{node_id-1}")
                params = {}

                if method_name in ("view", "reshape"):
                    shape_args = [a for a in node.args[1:] if isinstance(a, int)]
                    params["shape"] = shape_args if shape_args else (shape or [])

                elif method_name == "permute":
                    perm = [a for a in node.args[1:] if isinstance(a, int)]
                    params["axes"] = perm

                elif method_name == "transpose":
                    dims = [a for a in node.args[1:] if isinstance(a, int)]
                    params["dims"] = dims

                gir_nodes.append(GGMLNode(
                    id=node_id,
                    op=ggml_op,
                    name=node.name,
                    inputs=[input_ref],
                    output_shape=shape or [],
                    output_dtype=dtype,
                    params=params if params else None,
                ))
                name_map[node.name] = f"node:{node_id}"
                node_id += 1
            else:
                print(f"Warning: Unhandled method {method_name}")
                if node.args:
                    name_map[node.name] = name_map.get(node.args[0].name, f"node:{node_id-1}")

        elif node.op == "output":
            # Graph output - handle tuple outputs
            output_args = node.args[0]
            if isinstance(output_args, (tuple, list)):
                for i, arg in enumerate(output_args):
                    if isinstance(arg, fx.Node):
                        ref = name_map.get(arg.name, f"node:{node_id-1}")
                        out_shape = []
                        if "val" in arg.meta and hasattr(arg.meta["val"], "shape"):
                            out_shape = list(arg.meta["val"].shape)
                        gir_outputs.append(GGMLOutput(
                            name=f"output_{i}" if len(output_args) > 1 else "output",
                            node_ref=ref,
                            dtype=dtype,
                            shape=out_shape,
                        ))
            elif isinstance(output_args, fx.Node):
                ref = name_map.get(output_args.name, f"node:{node_id-1}")
                gir_outputs.append(GGMLOutput(
                    name="output",
                    node_ref=ref,
                    dtype=dtype,
                    shape=shape or [],
                ))

    return GGMLGraph(
        version="1.0.0",
        model_type="torch_export",
        inputs=gir_inputs,
        outputs=gir_outputs,
        nodes=gir_nodes,
    ), weights


def decompose_5d_attention_pattern(graph: GGMLGraph) -> GGMLGraph:
    """Decompose 5D attention reshape patterns into 4D-compatible operations.

    Detects patterns like:
      RESHAPE [B, S, 3, H, D]  (5D: batch, seq, QKV, heads, head_dim)
      PERMUTE [2, 0, 3, 1, 4]  (reorder to [3, B, H, S, D])
      SELECT dim=0, idx=0/1/2 (extract Q, K, V)

    And converts to:
      VIEW [B, S, 3, H*D]  (4D: combine heads*dim)
      SELECT dim=2, idx=i  (extract Q/K/V as [B, S, H*D])
      RESHAPE [B, S, H, D] (4D: split heads and dim)
      PERMUTE [0, 2, 1, 3] (4D: reorder to [B, H, S, D])
    """
    nodes = graph.nodes[:]
    new_nodes = []
    node_id_remap = {}  # old_id -> new_ref

    # Track which nodes consume 5D RESHAPE output for pattern detection
    reshape_5d_info = {}  # node_id -> {"shape": [...], "input": "..."}

    # First pass: identify 5D RESHAPE nodes
    for node in nodes:
        if node.op == "RESHAPE":
            shape = node.params.get("shape", []) if node.params else []
            if len(shape) == 5:
                reshape_5d_info[node.id] = {
                    "shape": shape,
                    "input": node.inputs[0] if node.inputs else None,
                    "node": node,
                }

    if not reshape_5d_info:
        return graph  # No 5D reshapes to decompose

    print(f"Decomposing {len(reshape_5d_info)} 5D reshape patterns...")

    # Build consumer map: which nodes use each node's output
    consumers = {}  # node_id -> list of consumer nodes
    for node in nodes:
        for inp in node.inputs:
            if inp.startswith("node:"):
                src_id = int(inp.split(":")[1])
                if src_id not in consumers:
                    consumers[src_id] = []
                consumers[src_id].append(node)

    # Identify patterns: 5D RESHAPE -> PERMUTE -> SELECT
    patterns_to_decompose = []
    for reshape_id, info in reshape_5d_info.items():
        shape = info["shape"]
        # Look for: [B, S, 3, H, D] pattern (QKV split)
        if len(shape) == 5 and shape[2] == 3:
            # This is likely QKV attention reshape
            B, S, _, H, D = shape
            # Find the PERMUTE that follows
            if reshape_id in consumers:
                for consumer in consumers[reshape_id]:
                    if consumer.op == "PERMUTE":
                        axes = consumer.params.get("axes", []) if consumer.params else []
                        if axes == [2, 0, 3, 1, 4]:
                            # Found the pattern! Now find SELECTs
                            patterns_to_decompose.append({
                                "reshape_id": reshape_id,
                                "reshape_node": info["node"],
                                "permute_node": consumer,
                                "B": B, "S": S, "H": H, "D": D,
                                "input_ref": info["input"],
                            })

    # Track nodes to skip (will be replaced)
    nodes_to_skip = set()
    # Track SELECT replacements
    select_replacements = {}  # old SELECT node_id -> new node ref

    next_node_id = max(n.id for n in nodes) + 1

    for pattern in patterns_to_decompose:
        B, S, H, D = pattern["B"], pattern["S"], pattern["H"], pattern["D"]
        input_ref = pattern["input_ref"]
        reshape_node = pattern["reshape_node"]
        permute_node = pattern["permute_node"]

        nodes_to_skip.add(reshape_node.id)
        nodes_to_skip.add(permute_node.id)

        # Find SELECT nodes that use this PERMUTE output
        selects = []
        if permute_node.id in consumers:
            for consumer in consumers[permute_node.id]:
                if consumer.op == "SELECT":
                    dim = consumer.params.get("dim", 0) if consumer.params else 0
                    idx = consumer.params.get("index", 0) if consumer.params else 0
                    if dim == 0:
                        selects.append((consumer, idx))
                        nodes_to_skip.add(consumer.id)

        # Generate decomposed nodes:
        # 1. VIEW input [B, S, 3*H*D] -> [B, S, 3, H*D]
        view_shape = [B, S, 3, H * D]
        view_node = GGMLNode(
            id=next_node_id,
            op="VIEW",
            name=f"decomposed_view_{reshape_node.id}",
            inputs=[input_ref],
            output_shape=view_shape,
            output_dtype=reshape_node.output_dtype,
            params={"shape": view_shape},
        )
        new_nodes.append(view_node)
        view_ref = f"node:{next_node_id}"
        next_node_id += 1

        # For each SELECT (Q, K, V), generate:
        # SELECT -> RESHAPE -> PERMUTE
        for select_node, qkv_idx in selects:
            # SELECT from dim=2 (the QKV dimension)
            select_new = GGMLNode(
                id=next_node_id,
                op="SELECT",
                name=f"decomposed_select_{select_node.id}",
                inputs=[view_ref],
                output_shape=[B, S, H * D],
                output_dtype=select_node.output_dtype,
                params={"dim": 2, "index": qkv_idx},
            )
            new_nodes.append(select_new)
            select_ref = f"node:{next_node_id}"
            next_node_id += 1

            # RESHAPE to [B, S, H, D]
            reshape_new = GGMLNode(
                id=next_node_id,
                op="RESHAPE",
                name=f"decomposed_reshape_{select_node.id}",
                inputs=[select_ref],
                output_shape=[B, S, H, D],
                output_dtype=select_node.output_dtype,
                params={"shape": [B, S, H, D]},
            )
            new_nodes.append(reshape_new)
            reshape_ref = f"node:{next_node_id}"
            next_node_id += 1

            # PERMUTE to [B, H, S, D]
            permute_new = GGMLNode(
                id=next_node_id,
                op="PERMUTE",
                name=f"decomposed_permute_{select_node.id}",
                inputs=[reshape_ref],
                output_shape=[B, H, S, D],
                output_dtype=select_node.output_dtype,
                params={"axes": [0, 2, 1, 3]},
            )
            new_nodes.append(permute_new)
            select_replacements[select_node.id] = f"node:{next_node_id}"
            next_node_id += 1

    # Build final node list with updated references
    all_nodes = {}
    for node in nodes:
        if node.id not in nodes_to_skip:
            # Update input references
            updated_inputs = []
            for inp in node.inputs:
                if inp.startswith("node:"):
                    src_id = int(inp.split(":")[1])
                    if src_id in select_replacements:
                        updated_inputs.append(select_replacements[src_id])
                    else:
                        updated_inputs.append(inp)
                else:
                    updated_inputs.append(inp)

            all_nodes[node.id] = GGMLNode(
                id=node.id,
                op=node.op,
                name=node.name,
                inputs=updated_inputs,
                output_shape=node.output_shape,
                output_dtype=node.output_dtype,
                params=node.params,
            )

    # Add the new decomposed nodes
    for node in new_nodes:
        all_nodes[node.id] = node

    # Topological sort based on dependencies
    def topological_sort(nodes_dict):
        # Build adjacency list
        in_degree = {nid: 0 for nid in nodes_dict}
        deps = {nid: [] for nid in nodes_dict}

        for nid, node in nodes_dict.items():
            for inp in node.inputs:
                if inp.startswith("node:"):
                    src_id = int(inp.split(":")[1])
                    if src_id in nodes_dict:
                        in_degree[nid] += 1
                        deps[src_id].append(nid)

        # Start with nodes that have no node dependencies
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort queue by original ID to maintain some stability
            queue.sort()
            nid = queue.pop(0)
            result.append(nid)

            for consumer in deps[nid]:
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    queue.append(consumer)

        if len(result) != len(nodes_dict):
            # Cycle or missing deps - fall back to ID sort
            print(f"Warning: Topological sort incomplete ({len(result)}/{len(nodes_dict)}), using ID sort")
            return sorted(nodes_dict.keys())

        return result

    sorted_ids = topological_sort(all_nodes)

    # Renumber nodes sequentially
    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

    renumbered_nodes = []
    for old_id in sorted_ids:
        node = all_nodes[old_id]
        new_inputs = []
        for inp in node.inputs:
            if inp.startswith("node:"):
                src_id = int(inp.split(":")[1])
                if src_id in old_to_new_id:
                    new_inputs.append(f"node:{old_to_new_id[src_id]}")
                else:
                    new_inputs.append(inp)
            else:
                new_inputs.append(inp)

        renumbered_nodes.append(GGMLNode(
            id=old_to_new_id[old_id],
            op=node.op,
            name=node.name,
            inputs=new_inputs,
            output_shape=node.output_shape,
            output_dtype=node.output_dtype,
            params=node.params,
        ))

    # Update output references
    new_outputs = []
    for out in graph.outputs:
        ref = out.node_ref
        if ref.startswith("node:"):
            old_id = int(ref.split(":")[1])
            if old_id in old_to_new_id:
                ref = f"node:{old_to_new_id[old_id]}"
        new_outputs.append(GGMLOutput(
            name=out.name,
            node_ref=ref,
            dtype=out.dtype,
            shape=out.shape,
        ))

    print(f"Decomposition complete: {len(nodes)} -> {len(renumbered_nodes)} nodes")

    return GGMLGraph(
        version=graph.version,
        model_type=graph.model_type,
        inputs=graph.inputs,
        outputs=new_outputs,
        nodes=renumbered_nodes,
    )


def export_torch_model(
    module: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str] = None,
    input_dtypes: Dict[str, str] = None,
    strict: bool = False,
) -> Tuple[GGMLGraph, Dict[str, torch.Tensor]]:
    """Export a PyTorch module via torch.export to GIR.

    This uses torch.export which can handle more dynamic operations than
    fx.symbolic_trace (like torch.empty with dynamic attributes).

    Args:
        module: PyTorch module to export
        example_inputs: Example inputs for tracing
        output_path: Path for output JSON
        input_names: Names for inputs
        input_dtypes: Dict mapping input names to dtype strings ("f32", "i32", etc.)
        strict: Whether to use strict mode (default False for more flexibility)

    Returns:
        Tuple of (GGMLGraph, weights dict)
    """
    module.eval()

    # Use torch.export
    print("Running torch.export...")
    exported = torch.export.export(module, example_inputs, strict=strict)

    print(f"Export succeeded! Graph has {len(list(exported.graph_module.graph.nodes))} nodes")

    # Build input shapes and dtypes dict
    input_shapes = {}
    input_dtype_map = {}
    for i, inp in enumerate(example_inputs):
        name = input_names[i] if input_names and i < len(input_names) else f"input_{i}"
        input_shapes[name] = list(inp.shape)
        if input_dtypes and name in input_dtypes:
            input_dtype_map[name] = GGMLDtype.from_string(input_dtypes[name])
        else:
            try:
                input_dtype_map[name] = GGMLDtype.from_torch_dtype(inp.dtype)
            except ValueError:
                input_dtype_map[name] = GGMLDtype.F32

    # Extract weights from state_dict
    # torch.export lifts parameters as placeholders prefixed with "p_"
    weights = {}
    for name, tensor in exported.state_dict.items():
        weight_name = name.replace(".", "_")
        weights[weight_name] = tensor.clone()

    # Also get constants (prefixed with "c_")
    if hasattr(exported, 'constants'):
        for name, tensor in exported.constants.items():
            if isinstance(tensor, torch.Tensor):
                weight_name = name.replace(".", "_")
                weights[weight_name] = tensor.clone()

    print(f"Extracted {len(weights)} weights from state_dict")

    # Convert to GIR using the graph_module
    gir_graph, extra_weights = convert_exported_to_gir(
        exported.graph_module,
        input_shapes,
        input_names,
        input_dtype_map,
        weights,  # Pass pre-extracted weights
    )

    # Merge any additional weights found during conversion
    weights.update(extra_weights)

    # Decompose 5D attention patterns into 4D-compatible operations
    gir_graph = decompose_5d_attention_pattern(gir_graph)

    # Save graph (if output path provided)
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(gir_graph.to_dict(), f, indent=2)

        weights_path = output_path.with_suffix(".weights.pt")
        torch.save(weights, weights_path)

        print(f"Saved graph to {output_path}")
        print(f"Saved {len(weights)} weights to {weights_path}")

    print(f"Graph has {len(gir_graph.nodes)} nodes")

    return gir_graph, weights


def symbolize_dimensions(
    graph: GGMLGraph,
    dim_mapping: Dict[str, int],
    protected_values: Optional[Set[int]] = None,
) -> GGMLGraph:
    """Replace concrete dimension values with symbolic names.

    This enables the graph to be instantiated with different sizes at runtime.

    Args:
        graph: The input GGMLGraph
        dim_mapping: Dict mapping symbolic names to concrete values that were
                     used during export. E.g., {"n_atoms": 2, "max_neighbors": 8}
        protected_values: Set of values that should NOT be symbolized even if
                         they match a dynamic dimension. Use this to protect
                         known model constants (e.g., 3 for xyz, 32 for head_dim).

    Returns:
        Modified graph with symbolic dimension names in shapes
    """
    if protected_values is None:
        protected_values = set()

    # Create reverse mapping: concrete value -> symbolic name
    # Note: if multiple symbols have the same value, we need to be careful
    # We'll prioritize based on typical usage patterns
    value_to_symbol = {}
    for sym, val in dim_mapping.items():
        if val not in protected_values:
            value_to_symbol[val] = sym

    # Add computed dimensions
    if "n_atoms" in dim_mapping and "max_neighbors" in dim_mapping:
        n_atoms = dim_mapping["n_atoms"]
        max_neighbors = dim_mapping["max_neighbors"]
        # n_edges = n_atoms * max_neighbors
        n_edges = n_atoms * max_neighbors
        if n_edges not in value_to_symbol and n_edges not in protected_values:
            value_to_symbol[n_edges] = "n_edges"
        # seq_len = n_atoms * (max_neighbors + 1)
        seq_len = n_atoms * (max_neighbors + 1)
        if seq_len not in value_to_symbol and seq_len not in protected_values:
            value_to_symbol[seq_len] = "seq_len"
        # max_neighbors_plus_one = max_neighbors + 1 (for concatenated node+neighbors)
        mn_plus_one = max_neighbors + 1
        if mn_plus_one not in value_to_symbol and mn_plus_one not in protected_values:
            value_to_symbol[mn_plus_one] = "max_neighbors_plus_one"

    def symbolize_shape(shape: List) -> List:
        """Replace known concrete values with symbolic names."""
        result = []
        for dim in shape:
            if isinstance(dim, int) and dim in value_to_symbol:
                result.append(value_to_symbol[dim])
            else:
                result.append(dim)
        return result

    def symbolize_params(params: Dict) -> Dict:
        """Symbolize dimension values in parameters.

        Only symbolizes shape-like parameters. Axis indices and other
        positional parameters should not be symbolized.
        """
        # Parameters that represent axis indices, not dimension sizes
        # These should never be symbolized
        axis_params = {"axes", "axis", "dim", "dim0", "dim1", "start_dim", "end_dim"}

        result = {}
        for key, value in params.items():
            if key in axis_params:
                # Don't symbolize axis indices
                result[key] = value
            elif key == "shape" and isinstance(value, list):
                result[key] = symbolize_shape(value)
            elif key == "new_shape" and isinstance(value, list):
                result[key] = symbolize_shape(value)
            elif key == "size" and isinstance(value, list):
                result[key] = symbolize_shape(value)
            elif key == "repeat_counts" and isinstance(value, list):
                result[key] = symbolize_shape(value)
            else:
                # Don't symbolize other parameters by default
                result[key] = value
        return result

    # Symbolize input shapes
    new_inputs = []
    for inp in graph.inputs:
        new_inputs.append(GGMLInput(
            name=inp.name,
            dtype=inp.dtype,
            shape=symbolize_shape(inp.shape),
            dynamic_dims=inp.dynamic_dims,
        ))

    # Symbolize output shapes
    new_outputs = []
    for out in graph.outputs:
        new_outputs.append(GGMLOutput(
            name=out.name,
            node_ref=out.node_ref,
            dtype=out.dtype,
            shape=symbolize_shape(out.shape),
        ))

    # Symbolize node shapes and params
    new_nodes = []
    for node in graph.nodes:
        new_nodes.append(GGMLNode(
            id=node.id,
            op=node.op,
            name=node.name,
            inputs=node.inputs,
            output_shape=symbolize_shape(node.output_shape),
            output_dtype=node.output_dtype,
            params=symbolize_params(node.params) if node.params else {},
        ))

    # Store dimension mapping in metadata
    new_metadata = dict(graph.metadata)
    new_metadata["dynamic_dims"] = dim_mapping

    return GGMLGraph(
        version=graph.version,
        model_type=graph.model_type,
        inputs=new_inputs,
        outputs=new_outputs,
        nodes=new_nodes,
        constants=graph.constants,
        metadata=new_metadata,
    )
