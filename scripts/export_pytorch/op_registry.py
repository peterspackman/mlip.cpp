"""
Operation registry mapping PyTorch/ATen operations to GGML operations.

This module provides the mapping between PyTorch's ATen operators
(as captured by torch.export) and GGML's operation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from .dimension_mapper import pytorch_to_ggml_shape, pytorch_to_ggml_dim


class GGMLOp(Enum):
    """GGML operations."""
    # Arithmetic
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    SQR = "SQR"
    SQRT = "SQRT"
    LOG = "LOG"
    SIN = "SIN"
    COS = "COS"
    SCALE = "SCALE"
    CLAMP = "CLAMP"

    # Unary activations
    UNARY_ABS = "UNARY_ABS"
    UNARY_NEG = "UNARY_NEG"
    UNARY_EXP = "UNARY_EXP"
    UNARY_TANH = "UNARY_TANH"
    UNARY_SIGMOID = "UNARY_SIGMOID"
    UNARY_RELU = "UNARY_RELU"
    UNARY_GELU = "UNARY_GELU"
    UNARY_SILU = "UNARY_SILU"
    UNARY_ELU = "UNARY_ELU"
    UNARY_HARDSWISH = "UNARY_HARDSWISH"

    # Matrix operations
    MUL_MAT = "MUL_MAT"
    OUT_PROD = "OUT_PROD"

    # Shape operations
    RESHAPE = "RESHAPE"
    VIEW = "VIEW"
    PERMUTE = "PERMUTE"
    TRANSPOSE = "TRANSPOSE"
    CONT = "CONT"
    REPEAT = "REPEAT"
    CONCAT = "CONCAT"
    PAD = "PAD"

    # Reduction
    SUM = "SUM"
    SUM_ROWS = "SUM_ROWS"
    MEAN = "MEAN"
    ARGMAX = "ARGMAX"

    # Indexing
    GET_ROWS = "GET_ROWS"
    SET_ROWS = "SET_ROWS"

    # Normalization
    NORM = "NORM"
    RMS_NORM = "RMS_NORM"
    GROUP_NORM = "GROUP_NORM"

    # Attention
    SOFT_MAX = "SOFT_MAX"
    FLASH_ATTN_EXT = "FLASH_ATTN_EXT"
    ROPE = "ROPE"

    # Special
    DECOMPOSE = "DECOMPOSE"  # Marker for ops that need decomposition


@dataclass
class OpMapping:
    """Mapping from an ATen operation to GGML operation(s)."""
    ggml_op: GGMLOp
    # Function to transform arguments
    arg_transform: Callable[[list, dict], tuple[list, dict]] | None = None
    # Function to compute output shape
    shape_fn: Callable[[list[list[int]], dict], list[int]] | None = None
    # Additional notes
    notes: str = ""


class OpRegistry:
    """Registry of PyTorch to GGML operation mappings."""

    def __init__(self):
        self._registry: dict[str, OpMapping] = {}
        self._decompositions: dict[str, Callable] = {}
        self._register_default_ops()

    def register(self, aten_op: str, mapping: OpMapping):
        """Register an operation mapping."""
        self._registry[aten_op] = mapping

    def register_decomposition(self, aten_op: str, decompose_fn: Callable):
        """Register a decomposition function for an operation."""
        self._decompositions[aten_op] = decompose_fn

    def get(self, aten_op: str) -> OpMapping | None:
        """Get the mapping for an ATen operation."""
        # Normalize the op name (remove torch._ops. prefix if present)
        normalized = self._normalize_op_name(aten_op)
        return self._registry.get(normalized)

    def _normalize_op_name(self, op_name: str) -> str:
        """Normalize operation name to canonical form."""
        # Remove torch._ops. prefix
        if op_name.startswith("torch._ops."):
            op_name = op_name[len("torch._ops."):]
        # Remove torch.ops. prefix
        if op_name.startswith("torch.ops."):
            op_name = op_name[len("torch.ops."):]
        return op_name

    def get_decomposition(self, aten_op: str) -> Callable | None:
        """Get the decomposition function for an operation."""
        normalized = self._normalize_op_name(aten_op)
        return self._decompositions.get(normalized)

    def is_supported(self, aten_op: str) -> bool:
        """Check if an operation is supported."""
        normalized = self._normalize_op_name(aten_op)
        return normalized in self._registry or normalized in self._decompositions

    def needs_decomposition(self, aten_op: str) -> bool:
        """Check if an operation needs decomposition."""
        normalized = self._normalize_op_name(aten_op)
        mapping = self._registry.get(normalized)
        if mapping and mapping.ggml_op == GGMLOp.DECOMPOSE:
            return True
        return normalized in self._decompositions

    def list_supported(self) -> list[str]:
        """List all supported ATen operations."""
        return sorted(set(self._registry.keys()) | set(self._decompositions.keys()))

    def _register_default_ops(self):
        """Register the default operation mappings."""

        # ===== Arithmetic Operations =====
        self.register("aten.add.Tensor", OpMapping(GGMLOp.ADD))
        self.register("aten.add.Scalar", OpMapping(GGMLOp.ADD))
        self.register("aten.sub.Tensor", OpMapping(GGMLOp.SUB))
        self.register("aten.sub.Scalar", OpMapping(GGMLOp.SUB))
        self.register("aten.mul.Tensor", OpMapping(GGMLOp.MUL))
        self.register("aten.mul.Scalar", OpMapping(GGMLOp.SCALE))
        self.register("aten.div.Tensor", OpMapping(GGMLOp.DIV))
        self.register("aten.div.Scalar", OpMapping(
            GGMLOp.SCALE,
            arg_transform=lambda args, kw: ([args[0], 1.0 / args[1]], kw),
        ))
        self.register("aten.pow.Tensor_Scalar", OpMapping(
            GGMLOp.SQR,
            notes="Only power=2 supported directly",
        ))
        self.register("aten.sqrt.default", OpMapping(GGMLOp.SQRT))
        self.register("aten.rsqrt.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose to 1/sqrt(x)",
        ))
        self.register("aten.log.default", OpMapping(GGMLOp.LOG))
        self.register("aten.sin.default", OpMapping(GGMLOp.SIN))
        self.register("aten.cos.default", OpMapping(GGMLOp.COS))
        self.register("aten.neg.default", OpMapping(GGMLOp.UNARY_NEG))
        self.register("aten.abs.default", OpMapping(GGMLOp.UNARY_ABS))
        self.register("aten.exp.default", OpMapping(GGMLOp.UNARY_EXP))
        self.register("aten.clamp.default", OpMapping(GGMLOp.CLAMP))
        self.register("aten.clamp_min.default", OpMapping(GGMLOp.CLAMP))
        self.register("aten.clamp_max.default", OpMapping(GGMLOp.CLAMP))

        # ===== Activation Functions =====
        self.register("aten.relu.default", OpMapping(GGMLOp.UNARY_RELU))
        self.register("aten.silu.default", OpMapping(GGMLOp.UNARY_SILU))
        self.register("aten.gelu.default", OpMapping(GGMLOp.UNARY_GELU))
        self.register("aten.tanh.default", OpMapping(GGMLOp.UNARY_TANH))
        self.register("aten.sigmoid.default", OpMapping(GGMLOp.UNARY_SIGMOID))
        self.register("aten.elu.default", OpMapping(GGMLOp.UNARY_ELU))
        self.register("aten.hardswish.default", OpMapping(GGMLOp.UNARY_HARDSWISH))

        # ===== Matrix Operations =====
        self.register("aten.mm.default", OpMapping(
            GGMLOp.MUL_MAT,
            notes="Matrix multiply: output = a @ b.T in GGML convention",
        ))
        self.register("aten.bmm.default", OpMapping(
            GGMLOp.MUL_MAT,
            notes="Batched matrix multiply",
        ))
        self.register("aten.matmul.default", OpMapping(
            GGMLOp.MUL_MAT,
            notes="General matrix multiply, may need reshape",
        ))
        self.register("aten.linear.default", OpMapping(
            GGMLOp.MUL_MAT,
            notes="Linear layer: y = x @ W.T + b",
        ))
        self.register("aten.addmm.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose to mm + add",
        ))

        # ===== Shape Operations =====
        self.register("aten.view.default", OpMapping(GGMLOp.RESHAPE))
        self.register("aten.reshape.default", OpMapping(GGMLOp.RESHAPE))
        self.register("aten._unsafe_view.default", OpMapping(GGMLOp.VIEW))
        self.register("aten.permute.default", OpMapping(GGMLOp.PERMUTE))
        self.register("aten.transpose.int", OpMapping(GGMLOp.TRANSPOSE))
        self.register("aten.t.default", OpMapping(
            GGMLOp.TRANSPOSE,
            notes="2D transpose",
        ))
        self.register("aten.contiguous.default", OpMapping(GGMLOp.CONT))
        self.register("aten.expand.default", OpMapping(GGMLOp.REPEAT))
        self.register("aten.repeat.default", OpMapping(GGMLOp.REPEAT))
        self.register("aten.cat.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose for backward pass support",
        ))
        self.register("aten.stack.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose to unsqueeze + cat",
        ))
        self.register("aten.squeeze.dim", OpMapping(GGMLOp.RESHAPE))
        self.register("aten.unsqueeze.default", OpMapping(GGMLOp.RESHAPE))
        self.register("aten.flatten.using_ints", OpMapping(GGMLOp.RESHAPE))
        self.register("aten.unflatten.int", OpMapping(GGMLOp.RESHAPE))

        # ===== Size/Shape Query Operations =====
        # These don't produce tensors, just metadata
        self.register("aten.sym_size.int", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Shape query - no tensor output, handled in graph construction",
        ))
        self.register("aten.sym_numel.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Element count query",
        ))

        # ===== Reduction Operations =====
        self.register("aten.sum.default", OpMapping(GGMLOp.SUM))
        self.register("aten.sum.dim_IntList", OpMapping(
            GGMLOp.SUM_ROWS,
            notes="Reduce along specified dimensions",
        ))
        self.register("aten.mean.default", OpMapping(GGMLOp.MEAN))
        self.register("aten.mean.dim", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose to sum/count",
        ))
        self.register("aten.argmax.default", OpMapping(GGMLOp.ARGMAX))
        self.register("aten.max.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="GGML has no direct max reduction",
        ))
        self.register("aten.min.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="GGML has no direct min reduction",
        ))

        # ===== Indexing Operations =====
        self.register("aten.embedding.default", OpMapping(
            GGMLOp.GET_ROWS,
            notes="Embedding lookup = row selection",
        ))
        self.register("aten.index_select.default", OpMapping(GGMLOp.GET_ROWS))
        self.register("aten.gather.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Complex gather needs decomposition",
        ))
        self.register("aten.slice.Tensor", OpMapping(
            GGMLOp.VIEW,
            notes="Slicing via view with offset",
        ))
        self.register("aten.select.int", OpMapping(
            GGMLOp.VIEW,
            notes="Select single index via view",
        ))

        # ===== Normalization =====
        # LayerNorm needs decomposition for backward pass support
        self.register("aten.layer_norm.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose for gradient support (GGML norm has no backward)",
        ))
        self.register("aten.native_layer_norm.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose for gradient support",
        ))
        self.register("aten.group_norm.default", OpMapping(GGMLOp.GROUP_NORM))
        self.register("aten.batch_norm.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Decompose to primitive ops",
        ))
        self.register("aten.rms_norm.default", OpMapping(GGMLOp.RMS_NORM))

        # ===== Attention =====
        self.register("aten.softmax.int", OpMapping(GGMLOp.SOFT_MAX))
        self.register("aten._softmax.default", OpMapping(GGMLOp.SOFT_MAX))
        self.register("aten.scaled_dot_product_attention.default", OpMapping(
            GGMLOp.FLASH_ATTN_EXT,
            notes="Fused attention kernel",
        ))

        # ===== Type Conversion =====
        self.register("aten.to.dtype", OpMapping(
            GGMLOp.DECOMPOSE,  # Use CAST op
            notes="Type casting",
        ))
        self.register("aten._to_copy.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Type casting with copy",
        ))

        # ===== Comparison Operations =====
        # These often need special handling
        self.register("aten.eq.Tensor", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="No direct GGML support, use masking",
        ))
        self.register("aten.ne.Tensor", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="No direct GGML support",
        ))
        self.register("aten.gt.Tensor", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="No direct GGML support",
        ))
        self.register("aten.lt.Tensor", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="No direct GGML support",
        ))

        # ===== Creation Operations =====
        self.register("aten.zeros_like.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Create zero tensor of same shape",
        ))
        self.register("aten.ones_like.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Create ones tensor of same shape",
        ))
        self.register("aten.full_like.default", OpMapping(
            GGMLOp.DECOMPOSE,
            notes="Create filled tensor of same shape",
        ))


# Global registry instance
_default_registry: OpRegistry | None = None


def get_registry() -> OpRegistry:
    """Get the default operation registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = OpRegistry()
    return _default_registry


def is_supported(aten_op: str) -> bool:
    """Check if an ATen operation is supported."""
    return get_registry().is_supported(aten_op)


def get_ggml_op(aten_op: str) -> GGMLOp | None:
    """Get the GGML operation for an ATen operation."""
    mapping = get_registry().get(aten_op)
    return mapping.ggml_op if mapping else None
