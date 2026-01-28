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
from .decompositions import get_decomposition, decompose_layer_norm, decompose_dropout

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
        self._chunk_info: dict[str, Any] | None = None  # For tracking chunk ops

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
            output_ref = self._handle_decomposition(
                node, gir, op_name, inputs, ggml_shape, dtype
            )
            if output_ref:
                self._node_outputs[node.name] = output_ref
                return
            # If decomposition failed, fall through to placeholder
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

    def _handle_decomposition(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        op_name: str,
        inputs: list[str],
        output_shape: list[int],
        output_dtype: GGMLDtype,
    ) -> str | None:
        """
        Handle decomposition of complex operations into primitives.

        Returns the reference to the output node, or None if decomposition failed.
        """
        # Layer normalization
        if "layer_norm" in op_name:
            return self._decompose_layer_norm(node, gir, inputs, output_shape)

        # Dropout (identity in inference)
        if "dropout" in op_name:
            return self._decompose_dropout(node, gir, inputs, output_shape)

        # rsqrt
        if "rsqrt" in op_name:
            return self._decompose_rsqrt(node, gir, inputs, output_shape)

        # addmm (bias + matmul)
        if "addmm" in op_name:
            return self._decompose_addmm(node, gir, output_shape)

        # mean.dim - sum + scale
        if "mean.dim" in op_name:
            return self._decompose_mean_dim(node, gir, inputs, output_shape)

        # cat/stack - needs special handling based on downstream ops
        if "cat" in op_name or "stack" in op_name:
            # For now, emit as CONCAT and handle at runtime
            gir_node = gir.add_node(
                op="CONCAT",
                name=node.name,
                inputs=inputs,
                output_shape=output_shape,
                output_dtype=output_dtype,
                params=self._get_concat_params(node),
            )
            return gir.node_ref(gir_node)

        # chunk/split - decompose to views
        if "chunk" in op_name or "split" in op_name:
            return self._decompose_chunk(node, gir, inputs, output_shape, output_dtype)

        # getitem - access tuple/list elements
        if "getitem" in op_name:
            return self._decompose_getitem(node, gir, inputs)

        return None

    def _decompose_layer_norm(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
        output_shape: list[int],
    ) -> str | None:
        """Decompose LayerNorm into primitives."""
        # Args: input, normalized_shape, weight, bias, eps
        if len(node.args) < 1:
            return None

        input_ref = inputs[0] if inputs else None
        if not input_ref:
            return None

        # Get weight and bias references
        weight_ref = None
        bias_ref = None

        if len(node.args) >= 3:
            # weight is arg[2]
            if isinstance(node.args[2], torch.fx.Node):
                weight_ref = self._node_outputs.get(node.args[2].name)
        if len(node.args) >= 4:
            # bias is arg[3]
            if isinstance(node.args[3], torch.fx.Node):
                bias_ref = self._node_outputs.get(node.args[3].name)

        # Get eps (usually arg[4] or in kwargs)
        eps = 1e-5
        if len(node.args) >= 5:
            eps = node.args[4]
        elif "eps" in node.kwargs:
            eps = node.kwargs["eps"]

        # If no weight/bias, we can't use the full affine decomposition
        # Fall back to a simplified version
        if weight_ref is None or bias_ref is None:
            logger.info(f"LayerNorm without affine params: {node.name}")
            # Just emit normalized output without affine transform
            return self._decompose_layer_norm_no_affine(
                gir, input_ref, output_shape, eps
            )

        return decompose_layer_norm(
            gir, input_ref, weight_ref, bias_ref, output_shape, eps
        )

    def _decompose_layer_norm_no_affine(
        self,
        gir: GGMLGraph,
        input_ref: str,
        input_shape: list[int],
        eps: float,
    ) -> str:
        """Decompose LayerNorm without affine parameters."""
        d_feat = input_shape[0]
        inv_d = 1.0 / float(d_feat)
        reduced_shape = [1] + input_shape[1:]

        # mean
        sum_node = gir.add_node(
            op="SUM_ROWS", name="ln_sum", inputs=[input_ref],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
        )
        mean_node = gir.add_node(
            op="SCALE", name="ln_mean", inputs=[gir.node_ref(sum_node)],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
            params={"scale": inv_d},
        )

        # centered
        mean_broadcast = gir.add_node(
            op="REPEAT", name="ln_mean_bc", inputs=[gir.node_ref(mean_node)],
            output_shape=input_shape, output_dtype=GGMLDtype.F32,
        )
        centered = gir.add_node(
            op="SUB", name="ln_centered",
            inputs=[input_ref, gir.node_ref(mean_broadcast)],
            output_shape=input_shape, output_dtype=GGMLDtype.F32,
        )

        # variance
        centered_sq = gir.add_node(
            op="SQR", name="ln_sq", inputs=[gir.node_ref(centered)],
            output_shape=input_shape, output_dtype=GGMLDtype.F32,
        )
        sum_sq = gir.add_node(
            op="SUM_ROWS", name="ln_sum_sq", inputs=[gir.node_ref(centered_sq)],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
        )
        var_node = gir.add_node(
            op="SCALE", name="ln_var", inputs=[gir.node_ref(sum_sq)],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
            params={"scale": inv_d},
        )

        # std
        var_stab = gir.add_node(
            op="SCALE", name="ln_var_stab", inputs=[gir.node_ref(var_node)],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
            params={"scale": 1.0 + eps},
        )
        std_node = gir.add_node(
            op="SQRT", name="ln_std", inputs=[gir.node_ref(var_stab)],
            output_shape=reduced_shape, output_dtype=GGMLDtype.F32,
        )

        # normalize
        std_broadcast = gir.add_node(
            op="REPEAT", name="ln_std_bc", inputs=[gir.node_ref(std_node)],
            output_shape=input_shape, output_dtype=GGMLDtype.F32,
        )
        normalized = gir.add_node(
            op="DIV", name="ln_out",
            inputs=[gir.node_ref(centered), gir.node_ref(std_broadcast)],
            output_shape=input_shape, output_dtype=GGMLDtype.F32,
        )

        return gir.node_ref(normalized)

    def _decompose_dropout(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
        output_shape: list[int],
    ) -> str:
        """Decompose dropout (identity in inference)."""
        input_ref = inputs[0] if inputs else None
        if not input_ref:
            return None

        # In inference, dropout is identity
        # Emit a CONT node as identity
        output = gir.add_node(
            op="CONT",
            name=node.name,
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )
        return gir.node_ref(output)

    def _decompose_rsqrt(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
        output_shape: list[int],
    ) -> str:
        """Decompose rsqrt (1/sqrt(x))."""
        input_ref = inputs[0] if inputs else None
        if not input_ref:
            return None

        # sqrt(x)
        sqrt_node = gir.add_node(
            op="SQRT",
            name=f"{node.name}_sqrt",
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )

        # 1/sqrt(x) - emit as custom RSQRT op for runtime to handle
        # This is because GGML doesn't have a direct reciprocal op
        output = gir.add_node(
            op="RSQRT",
            name=node.name,
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )
        return gir.node_ref(output)

    def _decompose_addmm(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        output_shape: list[int],
    ) -> str | None:
        """Decompose addmm (bias + input @ weight)."""
        # Args: bias, input, weight, [alpha], [beta]
        if len(node.args) < 3:
            return None

        bias_arg, input_arg, weight_arg = node.args[:3]

        bias_ref = self._node_outputs.get(bias_arg.name) if isinstance(bias_arg, torch.fx.Node) else None
        input_ref = self._node_outputs.get(input_arg.name) if isinstance(input_arg, torch.fx.Node) else None
        weight_ref = self._node_outputs.get(weight_arg.name) if isinstance(weight_arg, torch.fx.Node) else None

        if not all([bias_ref, input_ref, weight_ref]):
            return None

        # matmul: input @ weight.T
        mm_node = gir.add_node(
            op="MUL_MAT",
            name=f"{node.name}_mm",
            inputs=[weight_ref, input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )

        # add bias
        output = gir.add_node(
            op="ADD",
            name=node.name,
            inputs=[gir.node_ref(mm_node), bias_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )
        return gir.node_ref(output)

    def _decompose_mean_dim(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
        output_shape: list[int],
    ) -> str | None:
        """Decompose mean along dimension to sum + scale."""
        input_ref = inputs[0] if inputs else None
        if not input_ref:
            return None

        # Get dimension(s) from args
        dims = []
        if len(node.args) > 1:
            dim_arg = node.args[1]
            if isinstance(dim_arg, int):
                dims = [dim_arg]
            elif isinstance(dim_arg, (list, tuple)):
                dims = list(dim_arg)

        # Get input shape from metadata
        input_meta = None
        if isinstance(node.args[0], torch.fx.Node):
            input_meta = node.args[0].meta.get("val")

        if input_meta is None or not isinstance(input_meta, torch.Tensor):
            return None

        input_shape = list(input_meta.shape)

        # Compute the size of dimensions being reduced
        dim_size = 1
        for d in dims:
            dim_size *= input_shape[d]

        # For GGML, we need to emit sum followed by scale
        # This is a simplification - full implementation would handle keepdim properly
        sum_node = gir.add_node(
            op="SUM",
            name=f"{node.name}_sum",
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
            params={"dims": dims},
        )

        output = gir.add_node(
            op="SCALE",
            name=node.name,
            inputs=[gir.node_ref(sum_node)],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
            params={"scale": 1.0 / float(dim_size)},
        )

        return gir.node_ref(output)

    def _decompose_chunk(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
        output_shape: list[int],
        output_dtype: GGMLDtype,
    ) -> str:
        """
        Decompose chunk into multiple view operations.

        chunk(input, chunks, dim) returns a tuple of tensors.
        We emit a special CHUNK node and track outputs for getitem access.
        """
        input_ref = inputs[0] if inputs else None
        if not input_ref:
            return None

        # Get chunk parameters
        chunks = 2  # Default
        dim = -1

        if len(node.args) > 1:
            chunks = node.args[1]
        if len(node.args) > 2:
            dim = node.args[2]
        if "chunks" in node.kwargs:
            chunks = node.kwargs["chunks"]
        if "dim" in node.kwargs:
            dim = node.kwargs["dim"]

        # Get input shape from metadata
        input_meta = None
        if isinstance(node.args[0], torch.fx.Node):
            input_meta = node.args[0].meta.get("val")

        if input_meta is None or not isinstance(input_meta, torch.Tensor):
            return None

        input_shape = list(input_meta.shape)

        # Convert negative dim
        if dim < 0:
            dim = len(input_shape) + dim

        # Calculate chunk size
        dim_size = input_shape[dim]
        chunk_size = dim_size // chunks

        # Store chunk info for getitem access
        # The meta for this node is a tuple of tensors
        self._chunk_info = {
            "input_ref": input_ref,
            "input_shape": input_shape,
            "chunks": chunks,
            "dim": dim,
            "chunk_size": chunk_size,
        }

        # Emit a CHUNK placeholder that runtime will handle
        gir_node = gir.add_node(
            op="CHUNK",
            name=node.name,
            inputs=[input_ref],
            output_shape=output_shape,  # Shape of first chunk
            output_dtype=output_dtype,
            params={
                "chunks": chunks,
                "dim": dim,
                "chunk_size": chunk_size,
            },
        )

        return gir.node_ref(gir_node)

    def _decompose_getitem(
        self,
        node: torch.fx.Node,
        gir: GGMLGraph,
        inputs: list[str],
    ) -> str | None:
        """
        Decompose getitem (tuple access) into view operations.

        getitem(tuple, index) gets the element at index from a tuple.
        For chunk outputs, this creates a VIEW into the appropriate slice.
        """
        if len(node.args) < 2:
            return None

        source_node = node.args[0]
        index = node.args[1]

        if not isinstance(source_node, torch.fx.Node):
            return None

        # Check if source is a chunk operation
        source_ref = self._node_outputs.get(source_node.name)
        if not source_ref:
            return None

        # Get the output shape/dtype from metadata
        meta = node.meta.get("val")
        if meta is None or not isinstance(meta, torch.Tensor):
            return None

        pt_shape = list(meta.shape)
        ggml_shape = pytorch_to_ggml_shape(pt_shape)
        dtype = GGMLDtype.from_torch_dtype(meta.dtype)

        # Check if this is accessing a chunk result
        if hasattr(self, "_chunk_info") and self._chunk_info:
            info = self._chunk_info
            dim = info["dim"]
            chunk_size = info["chunk_size"]
            input_ref = info["input_ref"]

            # Create VIEW for this chunk
            # Offset calculation depends on dimension ordering
            gir_node = gir.add_node(
                op="VIEW",
                name=node.name,
                inputs=[input_ref],
                output_shape=ggml_shape,
                output_dtype=dtype,
                params={
                    "chunk_index": index,
                    "dim": dim,
                    "chunk_size": chunk_size,
                },
            )
            return gir.node_ref(gir_node)

        # Generic getitem - just reference the source
        return source_ref

    def _get_concat_params(self, node: torch.fx.Node) -> dict[str, Any]:
        """Extract concat parameters."""
        params = {}
        if len(node.args) > 1:
            if isinstance(node.args[1], int):
                params["dim"] = node.args[1]
        if "dim" in node.kwargs:
            params["dim"] = node.kwargs["dim"]
        return params

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
            if dims is None:
                # Static or non-tensor input
                dynamic_shapes[name] = None
            elif isinstance(dims, dict):
                dynamic_shapes[name] = {}
                for dim_idx, dim_name in dims.items():
                    dynamic_shapes[name][dim_idx] = Dim(dim_name)
            else:
                dynamic_shapes[name] = dims

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
