"""
Graph capture using torch.export/torch.fx.

This module provides the core functionality for capturing PyTorch model
computation graphs and converting them to GGML Intermediate Representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
from torch.export import export, ExportedProgram

from .dimension_mapper import pytorch_to_ggml_shape, pytorch_to_ggml_dim
from .graph_ir import GGMLGraph, GGMLNode, GGMLDtype
from .op_registry import get_registry, GGMLOp

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for graph capture."""
    # Dynamic shape specifications: {input_name: {dim_index: dim_name}}
    dynamic_shapes: dict[str, dict[int, str]] | None = None
    # Whether to decompose operations without backward support
    decompose_for_backward: bool = True
    # Maximum number of nodes (for debugging)
    max_nodes: int | None = None
    # Verbose logging
    verbose: bool = False


class GraphConverter:
    """Converts PyTorch FX graphs to GGML IR."""

    def __init__(self, config: CaptureConfig | None = None):
        self.config = config or CaptureConfig()
        self.registry = get_registry()
        self._node_outputs: dict[str, str] = {}  # FX node name -> GIR reference
        self._weight_names: dict[str, str] = {}  # Parameter name -> weight reference

    def convert(
        self,
        exported: ExportedProgram,
        model_type: str = "generic",
    ) -> GGMLGraph:
        """
        Convert an exported PyTorch program to GGML IR.

        Args:
            exported: The exported PyTorch program
            model_type: Type identifier for the model

        Returns:
            GGML graph representation
        """
        gir = GGMLGraph(model_type=model_type)
        graph = exported.graph

        # Reset tracking state
        self._node_outputs = {}
        self._weight_names = {}

        # Extract weight names from state dict
        for name in exported.state_dict.keys():
            clean_name = name.replace(".", "_")
            self._weight_names[name] = f"weight:{clean_name}"

        # Process graph nodes in order
        for node in graph.nodes:
            if self.config.max_nodes and len(gir.nodes) >= self.config.max_nodes:
                logger.warning(f"Reached max_nodes limit ({self.config.max_nodes})")
                break

            self._process_node(node, gir, exported)

        return gir

    def _process_node(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        exported: ExportedProgram,
    ):
        """Process a single FX graph node."""
        if self.config.verbose:
            logger.info(f"Processing node: {node.op} {node.target} {node.name}")

        if node.op == "placeholder":
            self._handle_placeholder(node, gir, exported)
        elif node.op == "get_attr":
            self._handle_get_attr(node, gir)
        elif node.op == "call_function":
            self._handle_call_function(node, gir)
        elif node.op == "call_method":
            self._handle_call_method(node, gir)
        elif node.op == "call_module":
            # Modules should be inlined by torch.export
            logger.warning(f"Unexpected call_module node: {node.name}")
        elif node.op == "output":
            self._handle_output(node, gir)
        else:
            logger.warning(f"Unknown node op: {node.op}")

    def _handle_placeholder(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        exported: ExportedProgram,
    ):
        """Handle input placeholder nodes."""
        # Get shape and dtype from node metadata
        meta = node.meta.get("val")
        if meta is None:
            logger.warning(f"No metadata for placeholder {node.name}")
            return

        if isinstance(meta, torch.Tensor):
            shape = list(meta.shape)
            dtype = GGMLDtype.from_torch_dtype(meta.dtype)
        else:
            # Could be a non-tensor input
            logger.info(f"Non-tensor placeholder: {node.name} = {type(meta)}")
            return

        # Check for dynamic dimensions
        dynamic_dims = []
        if self.config.dynamic_shapes and node.name in self.config.dynamic_shapes:
            for dim_idx in self.config.dynamic_shapes[node.name].keys():
                dynamic_dims.append(dim_idx)
                shape[dim_idx] = -1  # Mark as dynamic

        # Convert to GGML shape (reversed)
        ggml_shape = pytorch_to_ggml_shape(shape)
        ggml_dynamic = [len(shape) - 1 - d for d in dynamic_dims]

        inp = gir.add_input(
            name=node.name,
            dtype=dtype,
            shape=ggml_shape,
            dynamic_dims=ggml_dynamic,
        )
        self._node_outputs[node.name] = f"input:{node.name}"

    def _handle_get_attr(self, node: torch.fx.Node, gir: GGMLGraph):
        """Handle attribute access (weights/parameters)."""
        # The target is the attribute path
        attr_path = str(node.target)
        weight_ref = f"weight:{attr_path.replace('.', '_')}"
        self._node_outputs[node.name] = weight_ref

    def _handle_call_function(self, node: torch.fx.Node, gir: GGMLGraph):
        """Handle function call nodes (the main computation)."""
        # Get the operation name
        target = node.target
        if hasattr(target, "__module__") and hasattr(target, "__name__"):
            # ATen operation
            op_name = f"{target.__module__}.{target.__name__}".replace("torch.ops.", "")
        else:
            op_name = str(target)

        # Look up the mapping
        mapping = self.registry.get(op_name)
        if mapping is None:
            logger.warning(f"Unsupported operation: {op_name}")
            return

        # Get output shape and dtype from metadata
        meta = node.meta.get("val")
        if meta is None:
            logger.warning(f"No metadata for node {node.name}")
            return

        if isinstance(meta, torch.Tensor):
            pt_shape = list(meta.shape)
            dtype = GGMLDtype.from_torch_dtype(meta.dtype)
        elif isinstance(meta, (tuple, list)):
            # Multiple outputs - take the first
            if len(meta) > 0 and isinstance(meta[0], torch.Tensor):
                pt_shape = list(meta[0].shape)
                dtype = GGMLDtype.from_torch_dtype(meta[0].dtype)
            else:
                logger.warning(f"Cannot determine shape for {node.name}")
                return
        else:
            logger.warning(f"Unexpected meta type for {node.name}: {type(meta)}")
            return

        # Convert shape to GGML order
        ggml_shape = pytorch_to_ggml_shape(pt_shape)

        # Resolve input references
        inputs = self._resolve_inputs(node.args, node.kwargs)

        # Handle decomposition
        if mapping.ggml_op == GGMLOp.DECOMPOSE:
            # Check if we have a decomposition function
            decompose_fn = self.registry.get_decomposition(op_name)
            if decompose_fn:
                # Decomposition would add multiple nodes
                logger.info(f"Decomposing {op_name}")
                # For now, just add a placeholder
                gir_node = gir.add_node(
                    op=f"DECOMPOSED_{op_name.split('.')[-1].upper()}",
                    name=node.name,
                    inputs=inputs,
                    output_shape=ggml_shape,
                    output_dtype=dtype,
                    params={"original_op": op_name},
                )
            else:
                logger.warning(f"No decomposition for {op_name}, using placeholder")
                gir_node = gir.add_node(
                    op=f"UNSUPPORTED_{op_name.split('.')[-1].upper()}",
                    name=node.name,
                    inputs=inputs,
                    output_shape=ggml_shape,
                    output_dtype=dtype,
                    params={"original_op": op_name},
                )
        else:
            # Build operation parameters
            params = self._build_op_params(node, mapping, pt_shape)

            gir_node = gir.add_node(
                op=mapping.ggml_op.value,
                name=node.name,
                inputs=inputs,
                output_shape=ggml_shape,
                output_dtype=dtype,
                params=params,
            )

        self._node_outputs[node.name] = gir.node_ref(gir_node)

    def _handle_call_method(self, node: torch.fx.Node, gir: GGMLGraph):
        """Handle method call nodes."""
        method_name = node.target
        # Common methods that map to ops
        method_mappings = {
            "view": "aten.view.default",
            "reshape": "aten.reshape.default",
            "permute": "aten.permute.default",
            "transpose": "aten.transpose.int",
            "contiguous": "aten.contiguous.default",
            "to": "aten.to.dtype",
            "float": "aten.to.dtype",
            "half": "aten.to.dtype",
        }

        if method_name in method_mappings:
            # Treat as the corresponding ATen op
            op_name = method_mappings[method_name]
            # Create a synthetic node for processing
            node.target = op_name
            self._handle_call_function(node, gir)
        else:
            logger.warning(f"Unsupported method: {method_name}")

    def _handle_output(self, node: torch.fx.Node, gir: GGMLGraph):
        """Handle output nodes."""
        # node.args contains the output values
        for i, arg in enumerate(node.args):
            if isinstance(arg, (tuple, list)):
                for j, sub_arg in enumerate(arg):
                    self._add_output(gir, sub_arg, f"output_{i}_{j}")
            else:
                self._add_output(gir, arg, f"output_{i}")

    def _add_output(self, gir: GGMLGraph, arg, name: str):
        """Add an output to the graph."""
        if isinstance(arg, torch.fx.Node):
            ref = self._node_outputs.get(arg.name)
            if ref:
                # Get output info from the referenced node
                meta = arg.meta.get("val")
                if isinstance(meta, torch.Tensor):
                    shape = pytorch_to_ggml_shape(list(meta.shape))
                    dtype = GGMLDtype.from_torch_dtype(meta.dtype)
                    gir.add_output(name, ref, dtype, shape)

    def _resolve_inputs(
        self,
        args: tuple,
        kwargs: dict,
    ) -> list[str]:
        """Resolve input references from node arguments."""
        inputs = []

        for arg in args:
            ref = self._resolve_single_input(arg)
            if ref:
                inputs.append(ref)

        # Also include relevant kwargs
        for key, value in kwargs.items():
            if key in ("input", "x", "other", "weight", "bias"):
                ref = self._resolve_single_input(value)
                if ref:
                    inputs.append(ref)

        return inputs

    def _resolve_single_input(self, arg) -> str | None:
        """Resolve a single argument to a reference string."""
        if isinstance(arg, torch.fx.Node):
            return self._node_outputs.get(arg.name)
        elif isinstance(arg, (int, float)):
            # Scalar constant - could be stored in params instead
            return f"const:{arg}"
        elif isinstance(arg, (list, tuple)):
            # Could be shape or other metadata
            return None
        else:
            return None

    def _build_op_params(
        self,
        node: torch.fx.Node,
        mapping,
        pt_shape: list[int],
    ) -> dict[str, Any]:
        """Build operation-specific parameters."""
        params = {}
        op = mapping.ggml_op

        if op == GGMLOp.RESHAPE:
            # Extract target shape from args
            if len(node.args) > 1:
                target_shape = node.args[1]
                if isinstance(target_shape, (list, tuple)):
                    params["target_shape"] = pytorch_to_ggml_shape(list(target_shape))

        elif op == GGMLOp.PERMUTE:
            # Extract permutation from args
            if len(node.args) > 1:
                perm = node.args[1]
                if isinstance(perm, (list, tuple)):
                    params["permutation"] = list(perm)

        elif op == GGMLOp.TRANSPOSE:
            # Extract dimensions
            if len(node.args) >= 3:
                dim0, dim1 = node.args[1], node.args[2]
                params["dim0"] = dim0
                params["dim1"] = dim1

        elif op == GGMLOp.SUM_ROWS:
            # Extract reduction dimensions
            if len(node.args) > 1:
                dims = node.args[1]
                if isinstance(dims, (list, tuple)):
                    params["dims"] = list(dims)
                elif isinstance(dims, int):
                    params["dims"] = [dims]

        elif op == GGMLOp.SOFT_MAX:
            # Extract dimension
            if len(node.args) > 1 and isinstance(node.args[1], int):
                params["dim"] = node.args[1]

        elif op == GGMLOp.SCALE:
            # Extract scale factor
            if len(node.args) > 1 and isinstance(node.args[1], (int, float)):
                params["scale"] = float(node.args[1])

        elif op == GGMLOp.CLAMP:
            # Extract min/max from kwargs or args
            if "min" in node.kwargs:
                params["min"] = node.kwargs["min"]
            if "max" in node.kwargs:
                params["max"] = node.kwargs["max"]

        return params


def capture_model(
    model: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    config: CaptureConfig | None = None,
) -> GGMLGraph:
    """
    Capture a PyTorch model and convert to GGML IR.

    Args:
        model: PyTorch model to capture
        example_inputs: Example inputs for tracing
        config: Capture configuration

    Returns:
        GGML graph representation
    """
    config = config or CaptureConfig()

    # Build dynamic shapes spec for torch.export
    dynamic_shapes = None
    if config.dynamic_shapes:
        from torch.export import Dim
        dynamic_shapes = {}
        for name, dims in config.dynamic_shapes.items():
            dynamic_shapes[name] = {}
            for dim_idx, dim_name in dims.items():
                dynamic_shapes[name][dim_idx] = Dim(dim_name)

    # Export the model
    logger.info("Exporting model with torch.export...")
    exported = export(
        model,
        args=(),
        kwargs=example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    # Convert to GGML IR
    logger.info("Converting to GGML IR...")
    converter = GraphConverter(config)
    gir = converter.convert(exported, model_type=type(model).__name__)

    return gir


def capture_model_fx(
    model: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    config: CaptureConfig | None = None,
) -> GGMLGraph:
    """
    Capture a PyTorch model using torch.fx.symbolic_trace.

    This is a fallback for models that don't work with torch.export.

    Args:
        model: PyTorch model to capture
        example_inputs: Example inputs for tracing
        config: Capture configuration

    Returns:
        GGML graph representation
    """
    config = config or CaptureConfig()

    # Symbolic trace
    logger.info("Tracing model with torch.fx...")
    traced = torch.fx.symbolic_trace(model)

    # Run shape propagation
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(**example_inputs)

    # The traced model has a graph but not the same structure as ExportedProgram
    # We need to adapt the converter or create a wrapper
    # For now, this is a placeholder
    raise NotImplementedError(
        "torch.fx fallback not yet implemented. Use capture_model() with torch.export."
    )
